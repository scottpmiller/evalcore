"""Comparison / regression engine - candidate vs baseline -> gate verdict.

Generic across all consumers. Two ideas:

* **Guardrails** - metrics that must hold regardless of the headline result
  (e.g. ``false_negative_rate`` must stay under a ceiling and must not increase
  vs. baseline). A guardrail breach is a hard ``fail``.
* **Win metric** - the headline signal (``f1`` for ``/validation``; a pairwise
  win-rate for judged suites). A regression beyond the band is a ``warn``.

Verdict: any guardrail breach -> ``fail``; else win regressed -> ``warn``; else
``pass``. The on-regression policy is configurable per suite.
"""

from evalcore import models

_EPS = 1e-9


def _metric(card: models.Scorecard, name: str) -> float | None:
    found = card.metrics.get(name)
    return found.value if found else None


def _check_guardrail(
    rule: dict, baseline: models.Scorecard | None, candidate: models.Scorecard
) -> models.GuardrailResult:
    """Evaluate one guardrail rule.

    ``baseline`` is ``None`` when only one run exists. The absolute parts of
    a rule (``min``/``max``) still evaluate; the relative parts
    (``must_not_increase``/``must_not_decrease``) cannot, and are reported
    rather than silently passed - a rule with nothing left to check comes
    back ``passed=None``.
    """
    metric = rule['metric']
    cand = _metric(candidate, metric)
    base = _metric(baseline, metric) if baseline is not None else None
    relative = [
        name
        for name in ('must_not_increase', 'must_not_decrease')
        if rule.get(name)
    ]
    # A relative rule with no baseline is unevaluable, not satisfied.
    skipped = relative if baseline is None else []
    absolute = 'max' in rule or 'min' in rule
    if cand is None:
        return models.GuardrailResult(
            metric=metric, passed=False, detail='metric absent on candidate'
        )

    problems: list[str] = []
    if 'max' in rule and cand > rule['max'] + _EPS:
        problems.append(f'{cand:.4f} > max {rule["max"]}')
    if 'min' in rule and cand < rule['min'] - _EPS:
        problems.append(f'{cand:.4f} < min {rule["min"]}')
    if (
        rule.get('must_not_increase')
        and base is not None
        and (cand > base + _EPS)
    ):
        problems.append(f'increased {base:.4f} -> {cand:.4f}')
    if (
        rule.get('must_not_decrease')
        and base is not None
        and (cand < base - _EPS)
    ):
        problems.append(f'decreased {base:.4f} -> {cand:.4f}')

    note = (
        f'{", ".join(skipped)} not evaluated: needs a baseline'
        if skipped
        else ''
    )
    if problems:
        detail = '; '.join([*problems, note] if note else problems)
        return models.GuardrailResult(
            metric=metric, passed=False, detail=detail
        )
    if skipped and not absolute:
        # Nothing was checked, so this is neither a pass nor a breach.
        return models.GuardrailResult(metric=metric, passed=None, detail=note)
    detail = f'{cand:.4f} ok' + (f'; {note}' if note else '')
    return models.GuardrailResult(metric=metric, passed=True, detail=detail)


def check_thresholds(
    scorecard: models.Scorecard, thresholds: dict | None = None
) -> models.ThresholdCheck:
    """Measure one run against its suite's absolute thresholds.

    A gate needs two runs, but a suite's guardrails are mostly absolute - a
    ceiling on the error rate, a floor on a format check - and those are
    answerable from a single run. The runner calls this so a run carries its
    own verdict; a caller with one run should not have to reach for
    ``compare``, which is for comparing.

    The win metric is deliberately not evaluated: it is a comparison by
    definition, and reporting it here would claim a baseline that does not
    exist. Relative guardrails come back ``passed=None`` for the same reason.

    Args:
        scorecard: The run to measure.
        thresholds: The suite's ``thresholds`` block.

    Returns:
        The verdict and one result per configured guardrail. ``verdict`` is
        ``'none'`` when the suite declares no guardrails.

    """
    rules = (thresholds or {}).get('guardrails', [])
    if not rules:
        return models.ThresholdCheck()
    guardrails = [_check_guardrail(rule, None, scorecard) for rule in rules]
    breached = [g for g in guardrails if g.passed is False]
    return models.ThresholdCheck(
        verdict='fail' if breached else 'pass', guardrails=guardrails
    )


def _evaluate_win(
    thresholds: dict, baseline: models.Scorecard, candidate: models.Scorecard
) -> tuple[str | None, str]:
    metric = thresholds.get('win_metric')
    if not metric:
        return None, 'neutral'
    base = _metric(baseline, metric)
    cand = _metric(candidate, metric)
    if base is None or cand is None:
        return metric, 'neutral'
    higher_better = thresholds.get('win_higher_is_better', True)
    min_delta = thresholds.get('win_min_delta', 0.0)
    delta = cand - base if higher_better else base - cand
    if delta > min_delta:
        return metric, 'improved'
    if delta < -min_delta:
        return metric, 'regressed'
    return metric, 'neutral'


def compare(
    baseline: models.Scorecard,
    candidate: models.Scorecard,
    thresholds: dict | None = None,
) -> models.Comparison:
    """Compare two scorecards and produce a gate verdict."""
    thresholds = thresholds or {}

    deltas: list[models.MetricDelta] = []
    for metric in sorted(set(baseline.metrics) | set(candidate.metrics)):
        base = _metric(baseline, metric)
        cand = _metric(candidate, metric)
        delta = cand - base if base is not None and cand is not None else None
        deltas.append(
            models.MetricDelta(
                metric=metric, baseline=base, candidate=cand, delta=delta
            )
        )

    guardrails = [
        _check_guardrail(rule, baseline, candidate)
        for rule in thresholds.get('guardrails', [])
    ]
    win_metric, win = _evaluate_win(thresholds, baseline, candidate)

    breached = [g for g in guardrails if g.passed is False]
    on_regression = thresholds.get('on_regression', 'warn')
    if breached:
        verdict = 'fail'
    elif win == 'regressed':
        verdict = 'fail' if on_regression == 'fail' else 'warn'
    else:
        verdict = 'pass'

    if breached:
        summary = 'guardrail breach: ' + '; '.join(
            f'{g.metric} ({g.detail})' for g in breached
        )
    elif win_metric:
        summary = f'{win_metric} {win}'
    else:
        summary = 'no win metric configured'

    return models.Comparison(
        project=candidate.project,
        suite=candidate.suite,
        baseline_variant=baseline.variant.name,
        candidate_variant=candidate.variant.name,
        win_metric=win_metric,
        win=win,
        verdict=verdict,
        deltas=deltas,
        guardrails=guardrails,
        summary=summary,
    )
