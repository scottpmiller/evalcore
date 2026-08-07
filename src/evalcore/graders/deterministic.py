"""Built-in deterministic (per-case) graders.

Cheap, free, fully reproducible - the first tier. They read a field off the
output via a ``$ref`` selector and emit a 1.0/0.0 score plus ``passed`` so the
runner aggregates them into a pass-rate.
"""

import re

from evalcore import models, refs
from evalcore.graders import base


def _context(case: models.Case, output: models.Output) -> dict:
    return {
        'input': case.input,
        'expected': case.expected or {},
        'output': output.fields,
        'case': case.model_dump(),
    }


def _score(name: str, metric: str, case_id: str, ok: bool, detail: str):
    return models.Score(
        grader=name,
        metric=metric,
        value=1.0 if ok else 0.0,
        passed=ok,
        detail=detail,
        case_id=case_id,
        kind='per_case',
    )


@base.register('max_chars', base.GraderType.HEURISTIC)
class MaxChars:
    """Assert a text field is at most ``maximum`` characters long."""

    def __init__(self, field: str, maximum: int, name: str = 'max_chars'):
        self.name = name
        self.field = field
        self.maximum = maximum

    def grade(
        self, case: models.Case, output: models.Output
    ) -> list[models.Score]:
        value = refs.resolve_ref(_context(case, output), self.field)
        length = len(value) if isinstance(value, str) else 0
        ok = length <= self.maximum
        return [
            _score(
                self.name,
                self.name,
                case.id,
                ok,
                f'len={length} max={self.maximum}',
            )
        ]


@base.register('regex_absent', base.GraderType.HEURISTIC)
class RegexAbsent:
    """Assert a text field does NOT match ``pattern`` (e.g. no tokens)."""

    def __init__(self, field: str, pattern: str, name: str = 'regex_absent'):
        self.name = name
        self.field = field
        self.regex = re.compile(pattern)

    def grade(
        self, case: models.Case, output: models.Output
    ) -> list[models.Score]:
        value = refs.resolve_ref(_context(case, output), self.field)
        text = value if isinstance(value, str) else ''
        hit = self.regex.search(text)
        ok = hit is None
        detail = 'clean' if ok else f'matched {hit.group(0)!r}'
        return [_score(self.name, self.name, case.id, ok, detail)]


@base.register('regex_present', base.GraderType.HEURISTIC)
class RegexPresent:
    """Assert a text field matches EVERY pattern in ``patterns`` (all-of).

    The positive twin of :class:`RegexAbsent`: the cheap "did it include all
    the required constructs" gate (e.g. a form's action URL, required hidden
    fields). Patterns are compiled case-insensitively; a single ``pattern``
    string is accepted as shorthand for a one-element list.
    """

    def __init__(
        self,
        patterns: list[str] | None = None,
        field: str = 'html',
        pattern: str | None = None,
        name: str = 'regex_present',
    ):
        self.name = name
        self.field = field
        raw = list(patterns or ([] if pattern is None else [pattern]))
        self.regexes = [(p, re.compile(p, re.IGNORECASE)) for p in raw]

    def grade(
        self, case: models.Case, output: models.Output
    ) -> list[models.Score]:
        value = refs.resolve_ref(_context(case, output), self.field)
        text = value if isinstance(value, str) else ''
        missing = [p for p, regex in self.regexes if not regex.search(text)]
        ok = not missing
        detail = 'all present' if ok else f'missing {missing}'
        return [_score(self.name, self.name, case.id, ok, detail)]


@base.register('non_empty', base.GraderType.HEURISTIC)
class NonEmpty:
    """Assert a field resolves to a non-empty value."""

    def __init__(self, field: str, name: str = 'non_empty'):
        self.name = name
        self.field = field

    def grade(
        self, case: models.Case, output: models.Output
    ) -> list[models.Score]:
        value = refs.resolve_ref(_context(case, output), self.field)
        ok = bool(value)
        return [_score(self.name, self.name, case.id, ok, f'value={value!r}')]
