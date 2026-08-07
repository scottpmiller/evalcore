"""Grader protocols and the type registry.

A grader spec is a plain dict from the suite config: ``{type, name, ...}``.
``build_graders`` turns a list of specs into grader instances, split into the
per-case and aggregate buckets the runner needs.
"""

import enum
import typing

from evalcore import models
from evalcore.errors import ConfigError


class GraderType(enum.StrEnum):
    """What kind of check a grader performs, not which one.

    A closed set, unlike the registry's ``type`` names, which any consumer
    may extend. Declared once per grader at registration and read back by
    ``store.grader_lookups``; a ``StrEnum`` so it needs no serializer of its
    own on the way to a row.

    The same set is spelled out in three other places, all of which have to
    change together: ``GraderType`` in ``internal-eval-results``, the
    ``grader_type`` ``Enum8`` in that repo's ``schema.sql``, and the deployed
    DDL in ``schemata/clickhouse`` on GHE, which is the source of truth.

    ``UNKNOWN`` exists for a producer with no registry behind it. Nothing in
    evalcore emits it: ``register`` requires a category, so a grader that
    reaches a suite has always declared one.

    """

    UNKNOWN = 'unknown'
    HEURISTIC = 'heuristic'
    STATISTICAL = 'statistical'
    LLM_AS_JUDGE = 'llm_as_judge'
    TRAJECTORY = 'trajectory'
    HUMAN = 'human'


@typing.runtime_checkable
class Grader(typing.Protocol):
    """Per-case grader. Scores are averaged across cases by the runner."""

    name: str

    def grade(
        self, case: models.Case, output: models.Output
    ) -> list[models.Score]: ...


@typing.runtime_checkable
class AggregateGrader(typing.Protocol):
    """Whole-run grader for set-level metrics (P/R/F1, win-rate, ...)."""

    name: str

    def aggregate(
        self, results: list[models.CaseResult]
    ) -> list[models.Score]: ...


_REGISTRY: dict[str, type] = {}

_CATEGORIES: dict[str, GraderType] = {}


def register(
    type_name: str, category: GraderType
) -> typing.Callable[[type], type]:
    """Class decorator registering a grader under a suite-config ``type``.

    ``category`` is required rather than defaulting, because it is the only
    source of the row's ``grader_type`` and a default would be the value
    every grader forgets to override. A grader's category belongs to its
    implementation, not to a suite's use of it, so it is declared here and
    not in the suite config.

    Args:
        type_name: The ``type`` a suite spec names to select this grader.
        category: What kind of check it performs.

    Returns:
        The decorator.

    Raises:
        ConfigError: If ``type_name`` is already registered.

    """

    def _decorate(cls: type) -> type:
        if type_name in _REGISTRY:
            raise ConfigError(f'grader type {type_name!r} already registered')
        _REGISTRY[type_name] = cls
        _CATEGORIES[type_name] = GraderType(category)
        return cls

    return _decorate


def category_of(type_name: str) -> GraderType:
    """Return the category a grader type registered under.

    Args:
        type_name: The suite spec's ``type``.

    Returns:
        The declared category, or ``UNKNOWN`` for a type no plug-in has
        registered. A suite naming one cannot run - ``build_graders``
        raises - so ``UNKNOWN`` only reaches a row when a caller builds
        rows without loading the plug-ins that produced them.

    """
    return _CATEGORIES.get(type_name, GraderType.UNKNOWN)


def build_graders(
    specs: list[dict],
) -> tuple[list[Grader], list[AggregateGrader]]:
    """Instantiate grader specs, partitioned into per-case and aggregate.

    Each spec's ``type`` selects a registered class; remaining keys (minus
    ``type``) are passed as keyword arguments to its constructor.
    """
    per_case: list[Grader] = []
    aggregate: list[AggregateGrader] = []
    for spec in specs:
        spec = dict(spec)
        type_name = spec.pop('type')
        if type_name not in _REGISTRY:
            raise ConfigError(
                f'unknown grader type {type_name!r}; '
                f'known: {sorted(_REGISTRY)}'
            )
        grader = _REGISTRY[type_name](**spec)
        if isinstance(grader, AggregateGrader):
            aggregate.append(grader)
        elif isinstance(grader, Grader):
            per_case.append(grader)
        else:  # pragma: no cover - defensive
            raise TypeError(
                f'{type_name!r} is neither Grader nor AggregateGrader'
            )
    return per_case, aggregate
