"""Reporter protocol and registry (the report-rendering seam).

A reporter turns engine results into a rendered report. Two report kinds are
first-class, mirroring the two analyses the engine produces:

* ``scorecard`` - a SINGLE analysis (one suite x one variant run);
* ``comparison`` - a COMPARATIVE analysis (candidate vs baseline).

Built-ins ``markdown`` and ``html`` cover both. Register more the same way as
adapters and graders - ``@reporters.base.register('pdf')`` + ``--plugins`` -
so a consumer can add a format without touching the core.

Three further kinds are optional; a reporter is discovered to support each
via ``getattr`` and only implements the ones it cares about:

* ``run`` - a scorecard plus its per-case drill-down (see :func:`render_run`);
* ``agreement`` - a judge<->human calibration table (see
  :func:`render_agreement`);
* ``document`` - wrap a rendered body in a standalone document (e.g. an HTML
  page). Reporters that omit it are treated as identity.

Every ``*`` method returns a *fragment*; wrapping into a standalone document
happens once, at the edge, so fragments compose (a ``run`` body and an
``agreement`` table can share one page).
"""

import typing

from evalcore import models, report
from evalcore.errors import ConfigError


@typing.runtime_checkable
class Reporter(typing.Protocol):
    """Render a single scorecard and a candidate-vs-baseline comparison."""

    def scorecard(self, scorecard: models.Scorecard) -> str: ...

    def comparison(self, comparison: models.Comparison) -> str: ...


_REGISTRY: dict[str, type] = {}


def register(type_name: str) -> typing.Callable[[type], type]:
    """Class decorator registering a reporter under a report ``type``."""

    def _decorate(cls: type) -> type:
        if type_name in _REGISTRY:
            raise ConfigError(f'report type {type_name!r} already registered')
        _REGISTRY[type_name] = cls
        return cls

    return _decorate


def build_reporter(spec: str | dict) -> Reporter:
    """Instantiate a reporter from a name or a ``{type, ...}`` config spec."""
    spec = {'type': spec} if isinstance(spec, str) else dict(spec)
    type_name = spec.pop('type')
    if type_name not in _REGISTRY:
        raise ConfigError(
            f'unknown report type {type_name!r}; known: {sorted(_REGISTRY)}'
        )
    return _REGISTRY[type_name](**spec)


def wrap_document(reporter: Reporter, body: str) -> str:
    """Wrap ``body`` with the reporter's optional ``document`` hook."""
    document = getattr(reporter, 'document', None)
    return document(body) if callable(document) else body


def per_case_matrix(run: models.RunResult) -> tuple[list[str], list[dict]]:
    """Per-case drill-down behind a scorecard's aggregate means.

    Returns ``(metric_names, rows)`` where each row is
    ``{label, error, output, cells}`` - ``cells`` maps each per-case metric
    to its :class:`~evalcore.models.Score` (or is absent). Reporters format
    this into a case x metric table so a low aggregate points at the case
    that caused it.
    """
    metrics = sorted(
        {
            score.metric
            for result in run.results
            for score in result.scores
            if score.kind == 'per_case'
        }
    )
    multi = run.scorecard.n_samples > 1
    rows: list[dict] = []
    for result in run.results:
        by_metric = {
            s.metric: s for s in result.scores if s.kind == 'per_case'
        }
        # Multi-sample runs put several rows on one case, so the label carries
        # the sample's hash to tell them apart. Truncated: the row only needs
        # to be distinguishable by eye, and the full digest is in the run file.
        suffix = f' #{result.sample_hash[:8]}' if multi else ''
        label = result.case.id + suffix
        rows.append(
            {
                'label': label,
                'error': result.output.error,
                'output': result.output,
                'cells': by_metric,
            }
        )
    return metrics, rows


def render_run(reporter: Reporter, run: models.RunResult) -> str:
    """Render a full run as a standalone document: the detailed per-case
    report if the reporter supports ``run``, else the aggregate scorecard."""
    return wrap_document(reporter, run_body(reporter, run))


def run_body(reporter: Reporter, run: models.RunResult) -> str:
    """The run's report *fragment* (unwrapped), so it can be composed with
    other fragments before a single :func:`wrap_document` at the edge."""
    detailed = getattr(reporter, 'run', None)
    if callable(detailed):
        return detailed(run)
    return reporter.scorecard(run.scorecard)


def render_agreement(
    reporter: Reporter, result: models.AgreementResult
) -> str:
    """A judge<->human agreement *fragment*, using the reporter's ``agreement``
    hook when present and falling back to the Markdown table otherwise."""
    render = getattr(reporter, 'agreement', None)
    if callable(render):
        return render(result)
    return report.render_agreement(result)


def render_preferences(
    reporter: Reporter, result: models.PreferenceResult
) -> str:
    """A human A-vs-B win-rate *fragment*, using the reporter's ``preferences``
    hook when present and falling back to the Markdown table otherwise."""
    render = getattr(reporter, 'preferences', None)
    if callable(render):
        return render(result)
    return report.render_preferences(result)


def render_pairwise_agreement(
    reporter: Reporter, result: models.PairwiseAgreement
) -> str:
    """A human-panel vs LLM-judge agreement *fragment*, using the reporter's
    ``pairwise_agreement`` hook when present, else the Markdown table."""
    render = getattr(reporter, 'pairwise_agreement', None)
    if callable(render):
        return render(result)
    return report.render_pairwise_agreement(result)


def render_sweep(reporter: Reporter, result: models.SweepResult) -> str:
    """An N-way sweep *fragment* (ranking + metric x variant leaderboard),
    using the reporter's ``sweep`` hook if present, else the Markdown table."""
    render = getattr(reporter, 'sweep', None)
    if callable(render):
        return render(result)
    return report.render_sweep(result)


def render_pairwise(reporter: Reporter, result: models.PairwiseResult) -> str:
    """An A-vs-B pairwise win-rate *fragment*, using the reporter's
    ``pairwise`` hook when present, else the Markdown summary."""
    render = getattr(reporter, 'pairwise', None)
    if callable(render):
        return render(result)
    return report.render_pairwise(result)
