"""The runner: execute one suite x one variant over a dataset -> RunResult.

Generic and consumer-agnostic. For each case it invokes the adapter
``n_samples`` times, runs the per-case graders, then runs the aggregate graders
over the whole result set, and returns the aggregated ``Scorecard`` together
with every per-sample ``CaseResult`` (outputs, artifacts, scores) so nothing
downstream has to re-run the suite to inspect individual generations.

``suite.concurrency > 1`` runs the (case, sample) invocations concurrently
under a semaphore; the adapter and per-case graders must then tolerate
concurrent calls.
"""

import asyncio
import inspect
import secrets
import statistics
import time
import uuid

from evalcore import loader, models, store
from evalcore import retry as retry_mod
from evalcore.adapters import base as adapters_base
from evalcore.graders import base as graders_base


async def _invoke_with_retry(
    adapter,
    case: models.Case,
    variant: models.Variant,
    retry: retry_mod.RetryConfig,
) -> models.Output:
    """Invoke the adapter, retrying transient failures with backoff.

    Retries only an ``Output`` the adapter flagged ``retryable`` (a 429, 5xx,
    or network error); a terminal error or a success returns immediately. The
    adapter contract still holds - a failed call sets ``Output.error`` rather
    than raising - so this loop reads the flag, it does not catch exceptions.
    Returns the last ``Output`` (errored) once the attempt budget is spent.
    """
    attempt = 1
    while True:
        output = await adapter.invoke(case, variant)
        if not (output.error and output.retryable):
            return output
        if attempt >= retry.max_attempts:
            return output
        await asyncio.sleep(retry_mod.backoff_delay(attempt, retry))
        attempt += 1


def _sample_hash(output: models.Output, ordinal: int) -> str:
    """Digest identifying one sample by the output it produced.

    The ordinal rides along because the output alone is not unique: a
    deterministic system under test, and every ``replay`` run, returns
    byte-identical output for all ``n_samples`` of a case. Hashing content
    alone would give those samples one id, and the results store's key is
    (run, case, sample) on a replacing engine - identical hashes would collapse
    N rows into one on merge. Two samples that genuinely differ still get
    different hashes from their content; the ordinal only breaks the tie.
    """
    return loader.content_hash(
        {'sample': ordinal, 'output': output.model_dump(mode='json')}
    )


def _uuid7() -> uuid.UUID:
    """A UUIDv7 (RFC 9562): 48-bit millisecond timestamp, then random.

    Time-ordered, so a run id carries its own creation time and ids sort in
    the order the runs happened. ``uuid.uuid7`` is 3.14; this is the fallback
    for 3.11 through 3.13.
    """
    if hasattr(uuid, 'uuid7'):
        return uuid.uuid7()
    ms = int(time.time() * 1000) & 0xFFFFFFFFFFFF
    return uuid.UUID(
        int=(ms << 80)
        | (0x7 << 76)
        | (secrets.randbits(12) << 64)
        | (0b10 << 62)
        | secrets.randbits(62)
    )


def _aggregate_metrics(
    results: list[models.CaseResult], agg_scores: list[models.Score]
) -> dict[str, models.MetricValue]:
    """Fold per-case scores (mean) and aggregate scores (as-is) to metrics."""
    metrics: dict[str, models.MetricValue] = {}

    by_metric: dict[str, list[float]] = {}
    for result in results:
        for score in result.scores:
            if score.value is not None:
                by_metric.setdefault(score.metric, []).append(score.value)
    for metric, values in by_metric.items():
        metrics[metric] = models.MetricValue(
            metric=metric,
            value=statistics.fmean(values) if values else None,
            stdev=statistics.stdev(values) if len(values) > 1 else None,
            kind='mean',
            n=len(values),
        )

    for score in agg_scores:
        metrics[score.metric] = models.MetricValue(
            metric=score.metric,
            value=score.value,
            kind='aggregate',
            n=len(results),
        )
    return metrics


async def run_suite(
    suite: loader.SuiteConfig,
    variant_name: str,
    *,
    mode: str | None = None,
    grader_mode: str | None = None,
    revision: str | None = None,
    created_at: str | None = None,
    checkpoint: str | None = None,
    resume: bool = False,
) -> models.RunResult:
    """Run ``variant_name`` of ``suite``; return scorecard + all results.

    ``mode`` overrides ``suite.mode_default`` and drives the ADAPTER
    (``'replay'`` swaps the configured adapter for the replay adapter backed
    by ``replay_fixtures``). ``grader_mode`` drives the GRADERS independently
    (defaults to ``mode``); set it to run live judges over recorded outputs
    (``mode='replay', grader_mode='http'``) or replayed judges over a live
    target, without coupling the two.

    ``checkpoint`` is a JSONL file the runner appends each completed
    ``(case, sample)`` to; if ``resume`` and it already holds results for the
    *same* suite/dataset/variant (verified by content hash), those are loaded
    and their invocations skipped, so an interrupted run continues instead of
    restarting. Without ``resume`` an existing checkpoint is overwritten.
    """
    if variant_name not in suite.variants:
        raise KeyError(
            f'variant {variant_name!r} not in suite '
            f'(have {sorted(suite.variants)})'
        )
    mode = mode or suite.mode_default
    grader_mode = grader_mode or mode
    variant = models.Variant(
        name=variant_name, knobs=suite.variants[variant_name]
    )

    if mode == 'replay':
        if not suite.replay_fixtures:
            raise ValueError('replay mode requires suite.replay_fixtures')
        adapter = adapters_base.build_adapter(
            {'type': 'replay', 'fixtures': suite.replay_fixtures}
        )
    else:
        adapter = adapters_base.build_adapter(suite.adapter)

    per_case_graders, aggregate_graders = graders_base.build_graders(
        suite.graders
    )
    # Let graders that care (e.g. the LLM judge) bind to grader_mode so they
    # pick a live vs. replay client - independently of the adapter's mode -
    # and to the suite's retry policy so a transient judge-client error backs
    # off like an adapter call. Graders without the hook are left untouched.
    for grader in (*per_case_graders, *aggregate_graders):
        setter = getattr(grader, 'set_mode', None)
        if callable(setter):
            setter(grader_mode)
        retry_setter = getattr(grader, 'set_retry', None)
        if callable(retry_setter):
            retry_setter(suite.retry)

    cases = loader.load_cases(suite.dataset)
    dataset_hash = loader.dataset_hash(cases)
    run_id = str(_uuid7())

    # Resume: reuse already-completed (case, sample) results from a checkpoint
    # of the *same* eval; otherwise start (or restart) the checkpoint fresh.
    # Keyed by (case, ordinal) - the sample's position, which the checkpoint
    # carries for exactly this reason. Keying on the output hash instead would
    # not say *which* samples are missing, only how many, and a concurrent run
    # does not finish them in order.
    done: dict[tuple[str, int], models.CaseResult] = {}
    if checkpoint:
        meta = store.checkpoint_meta(checkpoint) if resume else None
        if meta:
            mismatch = (
                meta.get('suite_hash') != suite.suite_hash
                or meta.get('dataset_hash') != dataset_hash
                or meta.get('variant') != variant_name
            )
            if mismatch:
                raise ValueError(
                    'checkpoint is for a different suite/dataset/variant; '
                    'cannot resume (delete it to start over)'
                )
            run_id = meta['run_id']
            for ordinal, result in store.read_checkpoint_samples(checkpoint):
                done[result.case.id, ordinal] = result
        else:
            store.init_checkpoint(
                checkpoint,
                {
                    'run_id': run_id,
                    'suite_hash': suite.suite_hash,
                    'dataset_hash': dataset_hash,
                    'project': suite.project,
                    'suite': suite.suite,
                    'variant': variant_name,
                },
            )

    async def _run_one(
        case: models.Case, ordinal: int
    ) -> tuple[int, models.CaseResult]:
        output = await _invoke_with_retry(adapter, case, variant, suite.retry)
        scores: list[models.Score] = []
        for grader in per_case_graders:
            graded = grader.grade(case, output)
            if inspect.isawaitable(graded):
                graded = await graded
            scores.extend(graded)
        result = models.CaseResult(
            case=case,
            variant_name=variant_name,
            sample_hash=_sample_hash(output, ordinal),
            output=output,
            scores=scores,
        )
        # Record on completion so an interrupt leaves a resumable trail. The
        # append is synchronous within this coroutine, so concurrent runs
        # never interleave a half-written line.
        if checkpoint:
            store.append_checkpoint_result(checkpoint, result, ordinal)
        return ordinal, result

    jobs = [
        (case, ordinal)
        for case in cases
        for ordinal in range(suite.n_samples)
        if (case.id, ordinal) not in done
    ]
    try:
        if suite.concurrency > 1:
            semaphore = asyncio.Semaphore(suite.concurrency)

            async def _bounded(
                case: models.Case, ordinal: int
            ) -> tuple[int, models.CaseResult]:
                async with semaphore:
                    return await _run_one(case, ordinal)

            fresh = list(
                await asyncio.gather(
                    *(_bounded(case, ordinal) for case, ordinal in jobs)
                )
            )
        else:
            fresh = [await _run_one(case, ordinal) for case, ordinal in jobs]
    finally:
        # Adapters holding resources (browser, injected session, pooled
        # connections) may expose an optional async ``aclose`` hook.
        closer = getattr(adapter, 'aclose', None)
        if callable(closer):
            closed = closer()
            if inspect.isawaitable(closed):
                await closed

    # Merge resumed + fresh into dataset order (case position, then sample
    # ordinal), so the run is identical regardless of completion order or how
    # many resumes it took. Walking the job grid rather than sorting the
    # results is what keeps a gap a gap: a sample that never produced a result
    # is absent here instead of shifting the ones after it.
    merged = {**done, **{(r.case.id, o): r for o, r in fresh}}
    results = [
        merged[case.id, ordinal]
        for case in cases
        for ordinal in range(suite.n_samples)
        if (case.id, ordinal) in merged
    ]

    agg_scores: list[models.Score] = []
    for grader in aggregate_graders:
        agg_scores.extend(grader.aggregate(results))

    # Provenance: a grader may expose a ``judge_version`` pin (the LLM judge
    # does). Collect it onto the scorecard so a judge change re-baselines;
    # multiple judge graders join with ';'. Generic - non-judge graders have
    # no such attribute and contribute nothing.
    judge_pins = [
        pin
        for grader in (*per_case_graders, *aggregate_graders)
        if (pin := getattr(grader, 'judge_version', None))
    ]

    scorecard = models.Scorecard(
        run_id=run_id,
        project=suite.project,
        suite=suite.suite,
        variant=variant,
        dataset_version=suite.dataset_version,
        model_id=variant.knobs.get('model'),
        prompt_version=variant.knobs.get('prompt_version'),
        judge_version=';'.join(judge_pins) or None,
        revision=revision,
        suite_hash=suite.suite_hash,
        dataset_hash=dataset_hash,
        mode=mode,
        n_samples=suite.n_samples,
        n_cases=len(cases),
        created_at=created_at,
        metrics=_aggregate_metrics(results, agg_scores),
    )
    return models.RunResult(
        run_id=run_id,
        scorecard=scorecard,
        results=results,
        aggregate_scores=agg_scores,
    )


def run_suite_sync(
    suite: loader.SuiteConfig, variant_name: str, **kwargs
) -> models.RunResult:
    """Blocking convenience wrapper around :func:`run_suite`."""
    return asyncio.run(run_suite(suite, variant_name, **kwargs))
