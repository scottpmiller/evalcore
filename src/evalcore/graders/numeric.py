"""Built-in numeric (per-case) grader.

Promotes numeric ``Output`` fields onto the scorecard as metrics, so values an
adapter merely *extracted* (a cost, a tool-call error rate, a token count)
become first-class metrics the runner averages and ``compare``/guardrails can
gate on. Without this, an adapter's ``fields`` never reach a scorecard - only
``Score``s do.

Each configured field emits one per-case ``Score`` whose ``value`` is the field
coerced to float. Give a field a ``min``/``max`` bound and the score also
carries ``passed`` (in-range), turning it into a pass-rate; leave bounds off
and it's a pure measurement (``passed`` stays ``None``). Non-numeric or absent
fields degrade to ``value=None`` rather than raising.
"""

from evalcore import models, refs
from evalcore.graders import base


def _context(case: models.Case, output: models.Output) -> dict:
    return {
        'input': case.input,
        'expected': case.expected or {},
        'output': output.fields,
        'case': case.model_dump(),
    }


class _Field:
    """One resolved field spec: a ``$ref`` selector, a metric name, bounds."""

    def __init__(self, spec: str | dict):
        if isinstance(spec, str):
            spec = {'ref': spec}
        self.ref = spec['ref']
        self.minimum = spec.get('min')
        self.maximum = spec.get('max')
        # Metric name defaults to the ref's leaf (``output.cost`` -> ``cost``).
        self.metric = spec.get('name') or self.ref.rsplit('.', 1)[-1]


def _as_float(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@base.register('numeric', base.GraderType.HEURISTIC)
class Numeric:
    """Surface numeric output fields as metrics, optionally range-checked."""

    def __init__(self, fields: list[str | dict], name: str = 'numeric'):
        self.name = name
        self.fields = [_Field(spec) for spec in fields]

    def grade(
        self, case: models.Case, output: models.Output
    ) -> list[models.Score]:
        context = _context(case, output)
        scores: list[models.Score] = []
        for field in self.fields:
            value = _as_float(refs.resolve_ref(context, field.ref))
            passed, detail = _check(value, field)
            scores.append(
                models.Score(
                    grader=self.name,
                    metric=field.metric,
                    value=value,
                    passed=passed,
                    detail=detail,
                    case_id=case.id,
                    kind='per_case',
                )
            )
        return scores


def _check(value: float | None, field: _Field) -> tuple[bool | None, str]:
    """Range-check ``value`` against a field's bounds (``None`` if unbound)."""
    bounded = field.minimum is not None or field.maximum is not None
    if value is None:
        return (False if bounded else None), 'absent/non-numeric'
    if not bounded:
        return None, f'{value:g}'
    ok = (field.minimum is None or value >= field.minimum) and (
        field.maximum is None or value <= field.maximum
    )
    return ok, f'{value:g}'
