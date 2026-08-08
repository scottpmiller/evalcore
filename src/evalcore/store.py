"""Scorecard/comparison persistence and the results-store export seam.

Scorecards and comparisons serialize to JSON (CI artifacts, local files). For
trend tracking you can store them in a column store such as ClickHouse, keyed
by ``project``/``suite``; rather than couple the engine to any particular
driver, this module flattens a scorecard into a stable row shape and writes it
to a JSONL **outbox** a separate shipper drains. Swap ``JsonlOutboxExporter``
for a real database client without touching the runner or any consumer.

The emitted rows are one flat feed at (run, case, sample, grader, metric)
grain. Nothing derived is written: a run's scorecard and its trend are
read-time aggregations over these rows, so they cannot disagree with the scores
behind them.

A measurement that does not exist is ``null``, not 0 - a real ``0.0`` is a
meaningful score, and ``avg``/``sum``/``count`` skip nulls in every store worth
targeting, so no query has to remember a filter. ``metric_kind`` carries
``Score.kind`` verbatim and says nothing about presence: a metric some cases
could not score keeps its kind on every row and reads as one metric, with the
null doing the work. ``'none'`` is reserved for the two shapes that hold no
score at all - a failed invocation and a named-but-unscored metric. ``passed``
is the tri-state string ``'true'|'false'|'null'``: not a missing measurement,
since a judge has no pass line by design.

A store with no nullable columns is free to fill those nulls in at ingest; that
mapping belongs at its boundary, not here. Row keys are column names rather
than model attribute names, so ``project`` is emitted as ``application``,
``mode`` as ``adapter_mode``, ``created_at`` as ``started_at``, ``revision`` as
``application_revision``, ``n_cases`` as ``case_count`` and ``n_samples`` as
``sample_count``.
"""

import json
import pathlib
import typing

from evalcore import graders, models


def write_scorecard(
    path: str | pathlib.Path, scorecard: models.Scorecard
) -> None:
    """Write a scorecard as pretty JSON."""
    pathlib.Path(path).write_text(
        scorecard.model_dump_json(indent=2), encoding='utf-8'
    )


def read_scorecard(path: str | pathlib.Path) -> models.Scorecard:
    """Read a scorecard previously written by :func:`write_scorecard`."""
    return models.Scorecard.model_validate_json(
        pathlib.Path(path).read_text(encoding='utf-8')
    )


def load_scorecard(path: str | pathlib.Path) -> models.Scorecard:
    """Read a scorecard from either a scorecard file or a full-run file.

    Accepts both ``run --out`` (a bare Scorecard) and ``run --run-out`` (a
    RunResult wrapping one), so a comparison can consume whichever artifact
    a prior run left behind.
    """
    data = json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    if isinstance(data, dict) and 'scorecard' in data and 'results' in data:
        return models.Scorecard.model_validate(data['scorecard'])
    return models.Scorecard.model_validate(data)


def write_run(path: str | pathlib.Path, run: models.RunResult) -> None:
    """Write a full run (scorecard + every per-sample result) as JSON.

    This is the persisted ground truth behind a scorecard: transcript
    review, human rating, and judge-agreement analysis all read it back
    rather than re-running the suite.
    """
    pathlib.Path(path).write_text(
        run.model_dump_json(indent=2), encoding='utf-8'
    )


def read_run(path: str | pathlib.Path) -> models.RunResult:
    """Read a run previously written by :func:`write_run`."""
    return models.RunResult.model_validate_json(
        pathlib.Path(path).read_text(encoding='utf-8')
    )


# -- run checkpointing (idempotent resume) ----------------------------------
#
# A checkpoint is a JSONL file: line 1 is a meta header (run_id + the content
# hashes that identify *which* eval it belongs to), and each subsequent line is
# one completed ``CaseResult`` under ``result``, tagged with the ``sample``
# ordinal that produced it. The runner appends a line as each (case, sample)
# finishes, so an interrupted run leaves a valid partial file that a
# ``--resume`` re-run reads back to skip work already done.
#
# The ordinal is here and nowhere else. A ``CaseResult`` identifies its sample
# by output digest, which a pending sample cannot have yet, so resume needs the
# position to know which samples are still owed. This file is an internal
# resume artifact rather than the published row shape, so it can carry one
# field the interchange format deliberately does not.


def init_checkpoint(path: str | pathlib.Path, meta: dict) -> None:
    """Start a fresh checkpoint file with its meta header line (truncating)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        handle.write(json.dumps(meta) + '\n')


def checkpoint_meta(path: str | pathlib.Path) -> dict | None:
    """The checkpoint meta header, or ``None`` if the file is absent/empty."""
    path = pathlib.Path(path)
    if not path.is_file():
        return None
    with path.open(encoding='utf-8') as handle:
        first = handle.readline()
    return json.loads(first) if first.strip() else None


def append_checkpoint_result(
    path: str | pathlib.Path, result: models.CaseResult, ordinal: int
) -> None:
    """Append one completed sample, tagged with the ordinal that ran it."""
    line = json.dumps(
        {'sample': ordinal, 'result': result.model_dump(mode='json')}
    )
    with pathlib.Path(path).open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')


def read_checkpoint_samples(
    path: str | pathlib.Path,
) -> list[tuple[int, models.CaseResult]]:
    """``(ordinal, result)`` for each sample recorded after the meta line."""
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    lines = path.read_text(encoding='utf-8').splitlines()
    entries = []
    for line in lines[1:]:
        if not line.strip():
            continue
        entry = json.loads(line)
        entries.append(
            (
                entry['sample'],
                models.CaseResult.model_validate(entry['result']),
            )
        )
    return entries


def write_comparison(
    path: str | pathlib.Path, comparison: models.Comparison
) -> None:
    """Write a comparison as pretty JSON."""
    pathlib.Path(path).write_text(
        comparison.model_dump_json(indent=2), encoding='utf-8'
    )


def append_rating(path: str | pathlib.Path, rating: models.Rating) -> None:
    """Append one human rating to a JSONL ratings file (creating it)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(rating.model_dump_json() + '\n')


def read_ratings(path: str | pathlib.Path) -> list[models.Rating]:
    """Read a JSONL ratings file (the open human-rating interchange format).

    Rows may come from the ``rate`` web app or any external tool that emits
    the same shape - one JSON object per line.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    return [
        models.Rating.model_validate_json(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def append_preference(
    path: str | pathlib.Path, preference: models.Preference
) -> None:
    """Append one side-by-side preference to a JSONL file (creating it)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(preference.model_dump_json() + '\n')


def read_preferences(path: str | pathlib.Path) -> list[models.Preference]:
    """Read a JSONL preferences file (the open A-vs-B interchange format).

    Rows may come from the ``rank`` web app or any external tool that emits
    the same shape - one JSON object per line.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    return [
        models.Preference.model_validate_json(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def grader_lookups(specs: list[dict]) -> tuple[dict[str, str], dict[str, int]]:
    """Grader name to category, and grader name to judge scale.

    A ``Score`` names the grader that emitted it but carries neither the
    grader's category nor the scale its raw judge points sit on, so the caller
    builds both from ``suite.graders`` and hands them to :func:`score_rows`.

    The category comes from the registry, where each grader declared it at
    registration, so a consumer plug-in is categorised the same way a built-in
    is. A type nothing has registered lands on ``'unknown'`` rather than being
    guessed at, which in practice means the caller built rows without
    importing the plug-ins: a suite naming an unregistered type cannot run.
    """
    types: dict[str, str] = {}
    scales: dict[str, int] = {}
    for spec in specs:
        name = spec.get('name') or spec.get('type')
        if not name:
            continue
        grader_type = spec.get('type', '')
        types[name] = str(graders.category_of(grader_type))
        if grader_type == 'llm_judge':
            scales[name] = int(spec.get('scale', 5))
    return types, scales


_NO_UUID = '00000000-0000-0000-0000-000000000000'

#: An aggregate row has no invocation behind it, so every invocation field is
#: absent rather than zeroed - a 0 duration would read as a call that returned
#: instantly.
_NO_INVOCATION = {
    'duration': None,
    'is_error': False,
    'error_text': '',
    'retryable': False,
    'artifacts': {},
}

_NO_JUDGES = {
    'judges.name': [],
    'judges.version': [],
    'judges.points': [],
    'judges.score': [],
    'judges.rationale': [],
    'judge_scale': 0,
}


def _run_key(scorecard: models.Scorecard) -> dict:
    """The reproducibility key carried on every store row.

    Keys are the results-store column names rather than the model field names,
    so a JSONEachRow feed lands without a mapping layer. ``model_id`` and
    ``prompt_version`` are not columns: both are ``variant.knobs`` projections
    and travel inside ``variant_knobs``.
    """
    return {
        'started_at': scorecard.created_at,
        'application': scorecard.project,
        'run_id': scorecard.run_id,
        'suite': scorecard.suite,
        'variant': scorecard.variant.name,
        'dataset_version': scorecard.dataset_version,
        'judge_version': scorecard.judge_version or '',
        'application_revision': scorecard.revision or '',
        'suite_hash': scorecard.suite_hash or '',
        'dataset_hash': scorecard.dataset_hash or '',
        'adapter_mode': scorecard.mode,
        'case_count': scorecard.n_cases,
        'sample_count': scorecard.n_samples,
        'variant_knobs': scorecard.variant.knobs,
    }


def _gate(
    comparison: models.Comparison | None, baseline_run_id: str | None
) -> dict:
    """The run-grain gate columns, repeated on every row of a gated run.

    ``'none'`` on both enums means the run was never gated - a plain run, or
    the baseline half of a pair, neither of which has a ``Comparison``. The
    three ``win_*`` fields are null there rather than 0, since on a gated run a
    0 delta genuinely means no change.
    """
    if comparison is None:
        return {
            'gate_verdict': 'none',
            'gate_win': 'none',
            'baseline_run_id': _NO_UUID,
            'baseline_variant': '',
            'win_baseline': None,
            'win_candidate': None,
            'win_delta': None,
            'gate_summary': '',
        }
    delta = next(
        (d for d in comparison.deltas if d.metric == comparison.win_metric),
        None,
    )
    return {
        'gate_verdict': comparison.verdict,
        'gate_win': comparison.win,
        'baseline_run_id': baseline_run_id or _NO_UUID,
        'baseline_variant': comparison.baseline_variant,
        'win_baseline': delta.baseline if delta else None,
        'win_candidate': delta.candidate if delta else None,
        'win_delta': delta.delta if delta else None,
        'gate_summary': comparison.summary,
    }


def _metric_gate(
    metric: str,
    comparison: models.Comparison | None,
    rails: dict[str, models.GuardrailResult],
) -> dict:
    """The gate columns that belong to one metric rather than to the run."""
    rail = rails.get(metric)
    return {
        'win': bool(comparison and comparison.win_metric == metric),
        'guardrail': (
            'none' if rail is None else 'pass' if rail.passed else 'fail'
        ),
        'guardrail_gap': rail.detail if rail else '',
    }


def _invocation(output: models.Output) -> dict:
    """Facts about the call, at (case, sample) grain.

    ``latency_ms`` becomes ``duration`` in seconds, or null when the adapter
    reported none. ``Output.tokens`` and ``Output.cost`` are not exported:
    usage accounting is captured outside the results store.
    """
    return {
        'duration': (
            output.latency_ms / 1000 if output.latency_ms is not None else None
        ),
        'is_error': output.error is not None,
        'error_text': output.error or '',
        'retryable': output.retryable,
        'artifacts': output.artifacts,
    }


def _judges(score: models.Score, scale: int) -> dict:
    """The per-judge breakdown.

    Present only on a ``<grader>.overall`` score; every other row carries
    empty arrays. A dimension the judge did not score, and a judge that scored
    nothing at all, are carried as null rather than 0.
    """
    return {
        'judges.name': [judge.key for judge in score.judges],
        'judges.version': [judge.version or '' for judge in score.judges],
        'judges.points': [judge.points for judge in score.judges],
        'judges.score': [judge.overall for judge in score.judges],
        'judges.rationale': [judge.rationale or '' for judge in score.judges],
        'judge_scale': scale if score.judges else 0,
    }


def _passed(value: bool | None) -> str:
    """``passed`` as a tri-state string: deterministic true/false, judge null.

    A string rather than a bool-or-null because it is not a missing
    measurement: a judge has no pass line by design, so 'null' is one of three
    real states.
    """
    return 'true' if value is True else 'false' if value is False else 'null'


def score_rows(
    run: models.RunResult,
    comparison: models.Comparison | None = None,
    *,
    baseline_run_id: str | None = None,
    grader_types: dict[str, str] | None = None,
    judge_scales: dict[str, int] | None = None,
) -> list[dict]:
    """Flatten a run into ``evaluation_scores`` rows.

    One row per (case, sample, grader, metric), plus three shapes that carry
    an empty key column to say they do not sit at that grain: an aggregate
    metric has no ``case_id``, an invocation that failed before any grader ran
    has no ``grader`` or ``metric``, and a metric named by a guardrail or the
    win metric but never scored has neither and a null ``value``.

    ``comparison`` stamps the run-grain gate columns and the per-metric ``win``
    and ``guardrail`` fields; without it the run reads as ungated.
    The grader category and judge scale come from ``run.graders``, which the
    runner fills in - a ``Score`` carries neither, and a run that could not
    answer for itself was not publishable from its own artifact.
    ``grader_types`` and ``judge_scales`` override it, and are how a run
    written before 2.3.0 (whose map is empty) still publishes correctly;
    :func:`grader_lookups` builds them from the suite.
    """
    key = _run_key(run.scorecard)
    gate = _gate(comparison, baseline_run_id)
    rails = (
        {rail.metric: rail for rail in comparison.guardrails}
        if comparison
        else {}
    )
    # The run describes its own graders as of 2.3.0. The lookups remain for
    # runs written before that, which carry an empty map - a `run.json` on
    # disk outlives the release that wrote it. An explicit lookup still wins,
    # so a caller can correct a run it did not produce.
    types = {name: info.category for name, info in run.graders.items()}
    scales = {
        name: info.scale for name, info in run.graders.items() if info.scale
    }
    types.update(grader_types or {})
    scales.update(judge_scales or {})
    rows: list[dict] = []
    scored: set[str] = set()

    for result in run.results:
        invocation = _invocation(result.output)
        if not result.scores:
            rows.append(
                {
                    **key,
                    'case_id': result.case.id,
                    'sample_hash': result.sample_hash,
                    'grader': '',
                    'grader_type': 'unknown',
                    'metric': '',
                    'metric_kind': 'none',
                    'value': None,
                    'passed': 'null',
                    'detail': '',
                    'case_labels': result.case.labels,
                    'win': False,
                    'guardrail': 'none',
                    'guardrail_gap': '',
                    **invocation,
                    **_NO_JUDGES,
                    **gate,
                }
            )
            continue
        for score in result.scores:
            scored.add(score.metric)
            rows.append(
                {
                    **key,
                    'case_id': result.case.id,
                    'sample_hash': result.sample_hash,
                    'grader': score.grader,
                    'grader_type': types.get(score.grader, 'unknown'),
                    'metric': score.metric,
                    'metric_kind': score.kind,
                    'value': score.value,
                    'passed': _passed(score.passed),
                    'detail': score.detail or '',
                    'case_labels': result.case.labels,
                    **_metric_gate(score.metric, comparison, rails),
                    **invocation,
                    **_judges(score, scales.get(score.grader, 0)),
                    **gate,
                }
            )

    for score in run.aggregate_scores:
        scored.add(score.metric)
        rows.append(
            {
                **key,
                'case_id': '',
                'sample_hash': '',
                'grader': score.grader,
                'grader_type': types.get(score.grader, 'unknown'),
                'metric': score.metric,
                'metric_kind': 'aggregate',
                'value': score.value,
                'passed': _passed(score.passed),
                'detail': score.detail or '',
                'case_labels': {},
                **_metric_gate(score.metric, comparison, rails),
                **_NO_INVOCATION,
                **_NO_JUDGES,
                **gate,
            }
        )

    named = set(rails)
    if comparison and comparison.win_metric:
        named.add(comparison.win_metric)
    for metric in sorted(named - scored):
        rows.append(
            {
                **key,
                'case_id': '',
                'sample_hash': '',
                'grader': '',
                'grader_type': 'unknown',
                'metric': metric,
                'metric_kind': 'none',
                'value': None,
                'passed': 'null',
                'detail': '',
                'case_labels': {},
                **_metric_gate(metric, comparison, rails),
                **_NO_INVOCATION,
                **_NO_JUDGES,
                **gate,
            }
        )
    return rows


@typing.runtime_checkable
class ScoreExporter(typing.Protocol):
    """Where a run's score rows go.

    The seam between the engine and a results store. :func:`score_rows`
    produces the rows; an exporter decides what happens to them. Implement
    this to publish to a real store - a Kafka producer, a database client -
    without the runner or any consumer changing.

    An implementation lives with the store it targets, not here. It is the
    store that knows its own column types, its null policy, and its
    transport; the engine only knows the row.

    Swapping one for another is a constructor line at the call site, so an
    offline run and a live one share the same code path::

        exporter = JsonlOutboxExporter(path)      # offline
        exporter = KafkaOutboxExporter(...)       # live, from another package
        exporter.export_scores(run, comparison, grader_types=..., ...)

    """

    def export_scores(
        self,
        run: models.RunResult,
        comparison: models.Comparison | None = None,
        **kwargs,
    ) -> int:
        """Export a run's rows and return how many were exported.

        Args:
            run: The completed run.
            comparison: The gate result, on the candidate half of a gate;
                left off the baseline half, which was not itself gated.
            kwargs: Passed through to :func:`score_rows` - ``baseline_run_id``,
                ``grader_types`` and ``judge_scales``.

        Returns:
            The number of rows exported.

        """
        ...


class JsonlOutboxExporter:
    """Append score rows to a JSONL outbox for a shipper to drain.

    A no-network stand-in for direct ingestion: real deployments point a
    shipper at this file, or swap in another :class:`ScoreExporter` that
    writes to the store directly.
    """

    def __init__(self, outbox_path: str | pathlib.Path):
        self.outbox_path = pathlib.Path(outbox_path)

    def _append(self, rows: list[dict]) -> int:
        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        with self.outbox_path.open('a', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row) + '\n')
        return len(rows)

    def export_scores(
        self,
        run: models.RunResult,
        comparison: models.Comparison | None = None,
        **kwargs,
    ) -> int:
        """Append one row per store row; return how many were written."""
        return self._append(score_rows(run, comparison, **kwargs))
