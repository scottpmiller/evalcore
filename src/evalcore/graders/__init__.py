"""Grader registry and built-in graders.

Two extension shapes (see ``base``):

* ``base.Grader`` - per-case: ``grade(case, output) -> [Score]`` (may be
  async). Averaged by the runner. Used for deterministic checks (length,
  format, regex) and LLM-judge rubric scoring.
* ``base.AggregateGrader`` - whole-run: ``aggregate(results) -> [Score]``.
  Used for metrics that only exist over a set (precision/recall/F1, win-rate).

Importing this package registers the built-in grader types. Consumers register
custom graders with ``base.register`` and load them via the CLI ``--plugins``
flag, proving the generic/custom seam.
"""

from evalcore.graders import (
    base,
    classification,
    deterministic,
    judge,
    numeric,
)
from evalcore.graders.base import GraderType, category_of

__all__ = [
    'GraderType',
    'base',
    'category_of',
    'classification',
    'deterministic',
    'judge',
    'numeric',
]
