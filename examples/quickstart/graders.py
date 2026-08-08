"""Quickstart custom graders (the consumer plug-in seam).

Registers one grader of each kind against the engine's registry:

* ``acknowledges_customer`` - per-case: the reply must open with an
  acknowledgement/empathy cue (an errored invocation counts as a miss).
* ``distinct_reply_rate`` - aggregate: fraction of cases whose reply is unique
  across the run, catching a generator that collapses onto one generic reply.

Importing this module also registers the ``canned_support`` adapter, so a
single ``--plugins examples.quickstart.graders`` wires up the whole consumer.
Nothing here leaks into the engine.
"""

import re

from evalcore import models
from evalcore.graders import base

# Registering the adapter alongside the graders means one plug-in module wires
# up the whole consumer (adapter + custom graders).
from examples.quickstart import adapter  # noqa: F401

_ACK = re.compile(
    r"\b(sorry|understand|happy to help|you're right|apolog)", re.IGNORECASE
)


@base.register('acknowledges_customer', base.GraderType.HEURISTIC)
class AcknowledgesCustomer:
    """The reply must acknowledge the customer, not just brush them off.

    A generic "we'll get back to you" acknowledges nothing. Per-case check
    -> pass-rate.
    """

    def __init__(self, name: str = 'acknowledges_customer'):
        self.name = name

    def grade(
        self, case: models.Case, output: models.Output
    ) -> list[models.Score]:
        reply = str(output.fields.get('reply') or '')
        if output.error:
            ok, detail = False, f'output errored: {output.error}'
        else:
            ok = bool(_ACK.search(reply))
            detail = 'acknowledged' if ok else 'no acknowledgement cue'
        return [
            models.Score(
                grader=self.name,
                metric=self.name,
                value=1.0 if ok else 0.0,
                passed=ok,
                detail=detail,
                case_id=case.id,
                kind='per_case',
            )
        ]


@base.register('distinct_reply_rate', base.GraderType.HEURISTIC)
class DistinctReplyRate:
    """Fraction of cases whose reply is unique (mode-collapse check).

    Repeated samples of the same case do not count against distinctness;
    two *different* cases with the same reply do. Errored outputs excluded.
    """

    def __init__(self, name: str = 'distinct_reply_rate'):
        self.name = name

    def aggregate(
        self, results: list[models.CaseResult]
    ) -> list[models.Score]:
        by_case: dict[str, str] = {}
        for result in results:
            if result.output.error:
                continue
            reply = result.output.fields.get('reply')
            if isinstance(reply, str) and reply:
                by_case.setdefault(result.case.id, reply)
        distinct = len(set(by_case.values()))
        rate = distinct / len(by_case) if by_case else 0.0
        return [
            models.Score(
                grader=self.name,
                metric=self.name,
                value=rate,
                detail=f'{distinct} distinct across {len(by_case)} cases',
                kind='aggregate',
            )
        ]
