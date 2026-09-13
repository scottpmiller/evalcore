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


_POLARITY_WORDS = {
    'higher_is_better': True,
    'higher': True,
    'up': True,
    'lower_is_better': False,
    'lower': False,
    'down': False,
    'neutral': None,
    'none': None,
}


def _declared_polarity(thresholds: dict) -> dict[str, bool | None]:
    """Polarity as the suite states it, under ``thresholds.metrics``.

    A metric mapped to ``neutral`` is present here with a ``None`` value,
    which is not the same as being absent: it says the suite considered the
    metric and decided it has no good direction, and that beats inference.
    An unrecognised word is ignored rather than guessed at.
    """
    declared: dict[str, bool | None] = {}
    for metric, value in (thresholds.get('metrics') or {}).items():
        if isinstance(value, bool):
            declared[metric] = value
        elif isinstance(value, str):
            word = value.strip().lower()
            if word in _POLARITY_WORDS:
                declared[metric] = _POLARITY_WORDS[word]
    return declared


def _inferred_polarity(rules: list[dict]) -> dict[str, bool | None]:
    """Polarity the guardrail rules already imply.

    A ceiling (``max``, ``must_not_increase``) is only worth writing about a
    metric you want low; a floor (``min``, ``must_not_decrease``) about one
    you want high. So a suite that gates a metric has usually stated its
    direction already without meaning to, and this reads it back - which is
    what lets existing suites get correct directions with no edits.

    A metric whose rules imply both - a band, ``min`` and ``max`` together -
    resolves to ``None``, not to the default. Being fenced in on both sides
    is a statement that neither direction is the good one, and letting it
    fall through to higher-is-better would turn the one case we know is
    ambiguous into a confident wrong answer.
    """
    votes: dict[str, set[bool]] = {}
    for rule in rules:
        metric = rule.get('metric')
        if not metric:
            continue
        seen = votes.setdefault(metric, set())
        if 'max' in rule or rule.get('must_not_increase'):
            seen.add(False)
        if 'min' in rule or rule.get('must_not_decrease'):
            seen.add(True)
    return {
        metric: next(iter(seen)) if len(seen) == 1 else None
        for metric, seen in votes.items()
        if seen
    }


def _polarity(thresholds: dict) -> dict[str, bool | None]:
    """Every metric the suite says something about, weakest source first.

    Inference from the guardrails is the fallback; ``win_higher_is_better``
    beats it for the win metric because it is a statement rather than a
    reading; an explicit ``metrics:`` entry beats both. Metrics named by none
    of the three are absent, and callers treat absent as higher-is-better.
    """
    resolved: dict[str, bool | None] = dict(
        _inferred_polarity(thresholds.get('guardrails', []))
    )
    win_metric = thresholds.get('win_metric')
    if win_metric:
        resolved[win_metric] = bool(
            thresholds.get('win_higher_is_better', True)
        )
    resolved.update(_declared_polarity(thresholds))
    return resolved


def _direction(
    delta: float | None, higher_is_better: bool | None, min_delta: float = 0.0
) -> str:
    """Turn a signed delta into what it means for this metric.

    ``min_delta`` is the dead band: a move smaller than it is noise, not a
    result.
    """
    if delta is None or higher_is_better is None:
        return 'neutral'
    effective = delta if higher_is_better else -delta
    if effective > min_delta:
        return 'improved'
    if effective < -min_delta:
        return 'regressed'
    return 'neutral'


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
    # Same call the metric's own MetricDelta makes, so the headline verdict
    # and that row cannot disagree about the same number.
    return metric, _direction(
        cand - base,
        _polarity(thresholds).get(metric, True),
        thresholds.get('win_min_delta', 0.0),
    )


def compare(
    baseline: models.Scorecard,
    candidate: models.Scorecard,
    thresholds: dict | None = None,
) -> models.Comparison:
    """Compare two scorecards and produce a gate verdict."""
    thresholds = thresholds or {}

    polarity = _polarity(thresholds)
    win_metric = thresholds.get('win_metric')

    deltas: list[models.MetricDelta] = []
    for metric in sorted(set(baseline.metrics) | set(candidate.metrics)):
        base = _metric(baseline, metric)
        cand = _metric(candidate, metric)
        delta = cand - base if base is not None and cand is not None else None
        # Absent from the map means nothing in the suite spoke to this
        # metric, and higher-is-better is the convention almost every score
        # follows. A declared 'neutral' is present with a None value and is
        # a different answer: say nothing.
        higher_is_better = polarity.get(metric, True)
        deltas.append(
            models.MetricDelta(
                metric=metric,
                baseline=base,
                candidate=cand,
                delta=delta,
                higher_is_better=higher_is_better,
                direction=_direction(
                    delta,
                    higher_is_better,
                    thresholds.get('win_min_delta', 0.0)
                    if metric == win_metric
                    else 0.0,
                ),
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
