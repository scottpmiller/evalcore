"""Core data model for the generic eval engine.

None of these types reference any particular consumer. ``Case.input``,
``Case.expected``, and ``Variant.knobs`` are opaque blobs interpreted only by
adapters and graders - that opacity is what keeps the engine generic.
"""

import typing

import pydantic

Json = typing.Any


class Case(pydantic.BaseModel):
    """One eval input. ``input``/``expected`` are consumer-defined blobs."""

    id: str
    input: dict[str, Json] = pydantic.Field(default_factory=dict)
    expected: dict[str, Json] | None = None
    labels: dict[str, Json] = pydantic.Field(default_factory=dict)


class Variant(pydantic.BaseModel):
    """A configuration under test (e.g. baseline vs candidate).

    ``knobs`` is opaque to the engine; the adapter understands it (commonly
    ``{model, prompt_version, env}``).
    """

    name: str
    knobs: dict[str, Json] = pydantic.Field(default_factory=dict)


class Output(pydantic.BaseModel):
    """The system-under-test's response for one case, normalized.

    ``fields`` holds the values the adapter extracted (graders read these via
    ``output.<field>`` references). ``error`` is set when the invocation
    failed; graders/aggregators treat errored outputs explicitly rather than
    silently scoring them.
    """

    fields: dict[str, Json] = pydantic.Field(default_factory=dict)
    raw: Json = None
    error: str | None = None
    #: Set by the adapter alongside ``error`` when the failure is *transient*
    #: (a 429, a 5xx, a network timeout) and worth retrying; the runner's
    #: retry loop acts on this. A terminal error (bad request, non-JSON body)
    #: leaves it False so the run doesn't waste attempts on it.
    retryable: bool = False
    latency_ms: float | None = None
    tokens: dict[str, int] | None = None
    cost: float | None = None
    #: Named files the adapter saved for this invocation (screenshots,
    #: rendered HTML, transcripts): name -> filesystem path. Persisted
    #: with the run so downstream review/rating tools can load them.
    artifacts: dict[str, str] = pydantic.Field(default_factory=dict)


class JudgeDetail(pydantic.BaseModel):
    """One judge's full verdict on a case, retained for review.

    The runner folds ``points``/``overall`` into the aggregate means; this
    keeps the per-judge breakdown the aggregate hides - the raw 1..scale
    point each dimension got and the free-text ``rationale`` - so a low
    score can be read back to *why* without re-running the judge.
    """

    key: str
    version: str | None = None
    rationale: str | None = None
    #: raw per-dimension points as the judge returned them (1..scale), not
    #: normalized; ``None`` for a dimension the judge did not score.
    points: dict[str, float | None] = pydantic.Field(default_factory=dict)
    #: this judge's own normalized (0..1) overall, so a generous judge shows.
    overall: float | None = None


class Score(pydantic.BaseModel):
    """A single metric emitted by a grader for one case (or aggregate).

    ``kind='per_case'`` scores are averaged across cases by the runner;
    ``kind='aggregate'`` scores are computed once over the whole run (e.g.
    precision/recall) and stored as-is.
    """

    grader: str
    metric: str
    value: float | None = None
    passed: bool | None = None
    detail: str | None = None
    case_id: str | None = None
    kind: typing.Literal['per_case', 'aggregate'] = 'per_case'
    #: Populated only on an LLM-judge ``<name>.overall`` score: the per-judge
    #: breakdown (rationale + raw points) behind the aggregated value, so the
    #: store and reporters can surface it. Empty for every other grader.
    judges: list[JudgeDetail] = pydantic.Field(default_factory=list)


class CaseResult(pydantic.BaseModel):
    """The output + per-case scores for one (case, sample)."""

    case: Case
    variant_name: str
    sample_idx: int
    output: Output
    scores: list[Score] = pydantic.Field(default_factory=list)


class MetricValue(pydantic.BaseModel):
    """An aggregated metric on a scorecard.

    ``stdev`` is populated for ``mean`` metrics with 2+ observations
    (e.g. ``n_samples > 1``) so repeat runs carry their spread, not just
    the average.
    """

    metric: str
    value: float | None
    kind: typing.Literal['mean', 'aggregate']
    n: int
    stdev: float | None = None


class Scorecard(pydantic.BaseModel):
    """Aggregated result of running one suite x one variant over a dataset.

    The key tuple (project, suite, variant, dataset_version, model_id,
    prompt_version, judge_version, revision) makes a scorecard reproducible
    and is exactly the multi-tenant key used by the results store.

    ``revision`` is an opaque consumer-supplied provenance id (a git SHA,
    image digest, package version, release label, ...) - the engine never
    interprets it. ``suite_hash``/``dataset_hash`` are engine-computed
    content digests of the loaded suite config and cases: declared versions
    state intent, the hashes prove the content actually matched.
    """

    run_id: str | None = None
    project: str
    suite: str
    variant: Variant
    dataset_version: str
    model_id: str | None = None
    prompt_version: str | None = None
    judge_version: str | None = None
    revision: str | None = None
    suite_hash: str | None = None
    dataset_hash: str | None = None
    mode: str = 'http'
    n_samples: int = 1
    n_cases: int = 0
    created_at: str | None = None
    metrics: dict[str, MetricValue] = pydantic.Field(default_factory=dict)


class RunResult(pydantic.BaseModel):
    """Everything one run produced: the scorecard plus per-sample results.

    ``results`` is the ground truth the scorecard was aggregated from -
    one entry per (case, sample) with the full output, artifacts, and
    per-case scores. Persisting it is what makes transcript review,
    human rating, and judge-agreement analysis possible after the fact.

    ``aggregate_scores`` holds the ``kind='aggregate'`` scores computed once
    over the whole run. The scorecard keeps their values but not which grader
    emitted them or what it reported, so they are retained here too - a
    results-store row needs the grader attribution.
    """

    run_id: str
    scorecard: Scorecard
    results: list[CaseResult] = pydantic.Field(default_factory=list)
    aggregate_scores: list[Score] = pydantic.Field(default_factory=list)


class Rating(pydantic.BaseModel):
    """One human's blind rating of a single (case, sample) output.

    The open interchange format: emitted by the ``rate`` web app and
    ingestible from any external tool. ``scores`` maps rubric dimension ->
    an integer on the same 1..scale the judge used, so human and judge are
    directly comparable.
    """

    run_id: str
    case_id: str
    sample_idx: int = 0
    rater: str
    scores: dict[str, int] = pydantic.Field(default_factory=dict)
    rated_at: str | None = None


class Preference(pydantic.BaseModel):
    """One human's blind side-by-side preference between two variants' outputs
    for a single (case, sample) - the A-vs-B analog of :class:`Rating`.

    The open interchange format for the ``rank`` web app (one JSON object per
    line). ``winner`` is the overall pick and ``dims`` the per-dimension picks,
    each ``'a'`` / ``'b'`` / ``'tie'`` in *variant* terms: the app
    counterbalances left/right per rater and un-blinds server-side, so a
    stored ``'a'`` always means ``variant_a`` won regardless of which side it
    was shown on.
    """

    case_id: str
    sample_idx: int = 0
    variant_a: str
    variant_b: str
    rater: str
    winner: typing.Literal['a', 'b', 'tie'] = 'tie'
    dims: dict[str, typing.Literal['a', 'b', 'tie']] = pydantic.Field(
        default_factory=dict
    )
    rated_at: str | None = None


class DimensionAgreement(pydantic.BaseModel):
    """Judge-vs-human agreement on one rubric dimension."""

    dimension: str
    n: int
    human_mean: float | None = None
    judge_mean: float | None = None
    mae: float | None = None
    correlation: float | None = None


class AgreementResult(pydantic.BaseModel):
    """How well the judge tracks human raters over a run.

    The calibration gate: a judge whose scores don't agree with human
    ratings isn't trustworthy as a win metric yet. All values compare the
    per-(case,sample) human mean against the judge's normalized 0..1 score.
    """

    judge_name: str
    scale: int
    n_ratings: int
    n_raters: int
    dimensions: list[DimensionAgreement] = pydantic.Field(default_factory=list)
    overall_mae: float | None = None
    overall_correlation: float | None = None


class SweepEntry(pydantic.BaseModel):
    """One variant's standing in a sweep, ranked by the win metric."""

    variant: str
    win_value: float | None
    rank: int


class SweepResult(pydantic.BaseModel):
    """N-way comparison of a suite's variants (a model x prompt matrix).

    ``entries`` ranks the variants by the configured win metric;
    ``matrix`` is metric -> {variant: value} for the full leaderboard.
    """

    project: str
    suite: str
    win_metric: str | None = None
    win_higher_is_better: bool = True
    entries: list[SweepEntry] = pydantic.Field(default_factory=list)
    matrix: dict[str, dict[str, float | None]] = pydantic.Field(
        default_factory=dict
    )


class PairwiseOutcome(pydantic.BaseModel):
    """The head-to-head result for one case (counterbalanced for order)."""

    case_id: str
    sample_idx: int = 0
    winner: typing.Literal['a', 'b', 'tie'] = 'tie'
    detail: str | None = None


class PairwiseResult(pydantic.BaseModel):
    """A-vs-B win-rate from a judge comparing two variants head-to-head.

    The headline subjective-regression signal: for each case the judge is
    shown both variants' outputs and picks a winner (order counterbalanced,
    so a position-biased flip becomes a tie). ``win_rate_a`` counts ties as
    half, the standard convention.
    """

    project: str
    suite: str
    variant_a: str
    variant_b: str
    judge_name: str = 'pairwise'
    judge_version: str = 'v1'
    n: int = 0
    a_wins: int = 0
    b_wins: int = 0
    ties: int = 0
    win_rate_a: float | None = None
    outcomes: list[PairwiseOutcome] = pydantic.Field(default_factory=list)


class DimensionPreference(pydantic.BaseModel):
    """Human A-vs-B win-rate on one rubric dimension (ties count as half)."""

    dimension: str
    n: int = 0
    a_wins: int = 0
    b_wins: int = 0
    ties: int = 0
    win_rate_a: float | None = None


class PreferenceResult(pydantic.BaseModel):
    """Human side-by-side win-rate of A vs B - the human analog of
    :class:`PairwiseResult`.

    Reports an overall win-rate plus a per-dimension breakdown; ``win_rate_a``
    counts ties as half, the standard convention the pairwise judge uses.
    """

    project: str
    suite: str
    variant_a: str
    variant_b: str
    n_raters: int = 0
    n: int = 0
    a_wins: int = 0
    b_wins: int = 0
    ties: int = 0
    win_rate_a: float | None = None
    dimensions: list[DimensionPreference] = pydantic.Field(
        default_factory=list
    )


class PairwiseAgreementCase(pydantic.BaseModel):
    """Human-panel vs LLM-judge winner for one case."""

    case_id: str
    sample_idx: int = 0
    human: typing.Literal['a', 'b', 'tie'] = 'tie'
    judge: typing.Literal['a', 'b', 'tie'] = 'tie'
    agree: bool = False


class PairwiseAgreement(pydantic.BaseModel):
    """How often the human panel and the LLM pairwise judge pick the same
    per-case winner - the head-to-head calibration gate (the A-vs-B analog of
    :class:`AgreementResult`).
    """

    variant_a: str
    variant_b: str
    judge_name: str = 'pairwise'
    n: int = 0
    agree: int = 0
    human_win_rate_a: float | None = None
    judge_win_rate_a: float | None = None
    agreement_rate: float | None = None
    outcomes: list[PairwiseAgreementCase] = pydantic.Field(
        default_factory=list
    )


class MetricDelta(pydantic.BaseModel):
    """Per-metric baseline->candidate comparison."""

    metric: str
    baseline: float | None
    candidate: float | None
    delta: float | None


class GuardrailResult(pydantic.BaseModel):
    """Outcome of one guardrail check against the candidate."""

    metric: str
    passed: bool
    detail: str


class Comparison(pydantic.BaseModel):
    """Candidate-vs-baseline comparison and gate verdict."""

    project: str
    suite: str
    baseline_variant: str
    candidate_variant: str
    win_metric: str | None = None
    win: typing.Literal['improved', 'regressed', 'neutral'] = 'neutral'
    verdict: typing.Literal['pass', 'warn', 'fail'] = 'pass'
    deltas: list[MetricDelta] = pydantic.Field(default_factory=list)
    guardrails: list[GuardrailResult] = pydantic.Field(default_factory=list)
    summary: str = ''
