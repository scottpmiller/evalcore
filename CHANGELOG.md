# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `MetricRange` and a `value_range` field on `Score` and `MetricValue`.
  Nothing about a float says what it is on: `0.86` is 86% if the metric runs
  0..1 and 4.3 out of 5 if it runs 1..5. A consumer holding only the number
  guesses from magnitude, which is how a run whose costs happened to stay
  under a dollar gets rendered as percentages, and how the same metric gets
  classified differently depending on the window you look at.
- Built-in graders declare their own ranges. Deterministic checks are 0..1;
  `classification` separates its confusion-matrix metrics (0..1) from
  `support_*` and `errors` (0..unbounded); judge metrics are 0..1, being
  points over the scale, except `disagreement`, a spread in raw points,
  which is 0..scale-1.
- `range:` on a `numeric` field spec, as `{min, max}` or a two-element list.
  That grader surfaces whatever the adapter put in the field, so it is the
  one place the engine cannot know. It is a separate key from the `min`/`max`
  bounds beside it, which are a pass/fail threshold rather than a domain: a
  cost that must stay under a dollar can still cost five.
- `maximum=None` is a positive statement - unbounded above, so not a
  fraction of anything - and is different from carrying no range at all,
  which says only that nobody declared one. Nothing is inferred from
  observed values.
- `direction` and `higher_is_better` on `MetricDelta`. A delta's sign and its
  meaning are different questions - `f1` rising is an improvement, and
  `false_negative_rate` rising is a regression - and until now the engine only
  knew the difference for the single win metric, via `win_higher_is_better`.
  Every other metric came out of `compare()` as a bare number, so anything
  downstream that wanted to rank, colour or summarize deltas had to keep its
  own list of which metrics are inverted, or get it wrong.
- `thresholds.metrics`, an optional map declaring polarity per metric:
  `lower_is_better`, `higher_is_better`, or `neutral` for one that moves
  without either direction being a result.
- **Polarity is inferred from the guardrails when it is not declared**, which
  is what makes this useful without editing a single existing suite. A `max`
  or `must_not_increase` rule is only ever written about a metric you want
  low, and `min`/`must_not_decrease` about one you want high, so a gated
  metric has already stated its direction. Precedence, strongest first: an
  explicit `metrics:` entry, then `win_higher_is_better` for the win metric,
  then the guardrails. A metric none of them mention defaults to
  higher-is-better; a metric fenced in on both sides by a band resolves to
  `neutral` rather than falling through to that default, since a band says
  neither direction is the good one.

### Changed
- `_evaluate_win` derives the win verdict through the same call that fills in
  each `MetricDelta.direction`, so `Comparison.win` and the win metric's own
  delta row cannot disagree about the same number.

## [2.5.0] - 2026-08-11

A suite can declare the modules it needs imported.

### Added
- `plugins:` on a suite: a list of module paths the runner imports before it
  looks up any `type`, so a suite that names a custom adapter or grader
  resolves it without a flag at the call site. Registration is an import side
  effect and nothing in the engine imports a consumer's module on its own, so
  until now every entry point had to remember `--plugins my.graders` (CLI) or a
  bare `import my.graders` (Python API) - and the two could disagree. A suite
  is now self-contained: the same file runs from the CLI, from a consumer's own
  `run_eval.py`, and from a test with nothing to remember.

  `--plugins` is unchanged and still the way to add a module without editing
  the suite, which would change `suite_hash`.

  The import happens when a **run** starts, never in `load_suite`. Parsing,
  hashing, diffing or reporting on a suite executes no consumer code, so a
  suite you have not decided to run is still only data. `compare` and `report`
  therefore do not import a suite's plugins - they do not need the registries.

  A module that cannot be imported raises `ConfigError` naming it, rather than
  the unknown-`type` error one lookup later.

### Fixed
- `examples/quickstart/run_eval.py` called `JsonlOutboxExporter.export()`,
  removed when 2.2.0 named the exporter seam, so `just example-api` had been
  failing with `AttributeError` since. It now exports score rows only, which is
  the one grain the store has: a scorecard is a read-time aggregation over
  those rows, so exporting it too would persist something derived that could
  disagree with them.

### Changed
- `examples/quickstart` declares its own `plugins:` and no longer needs
  `--plugins` on the command line, nor the `import ... # noqa: F401` that three
  of its entry points carried to force registration. `graders.py` no longer
  imports `adapter.py` for the side effect either. The example is the same
  eval; it just stops demonstrating the workaround.

## [2.4.3] - 2026-08-11

Live Anthropic judges work on current Claude models again.

### Fixed
- The Anthropic judge and pairwise clients no longer send `temperature=0`.
  `temperature` (with `top_p`/`top_k`) was removed from the Claude request
  surface at Opus 4.7, and sending it at all is a 400 there and on every model
  after it - so a judge or a pairwise comparison pinned to `claude-opus-4-7`,
  `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5` or `claude-fable-5`
  failed every call. The OpenAI clients still send it; that API still takes it.

### Changed
- Live judge `max_tokens` defaults are now 8192 (from 1024 on the rubric
  judge, 512 on pairwise). Thinking is on by default from Opus 5 and Sonnet 5
  onward and `max_tokens` bounds thinking plus reply together, so a
  1024-token budget could be spent on reasoning before the forced tool call
  landed - which surfaced as error scores rather than as an error. It is a
  ceiling, not a spend: a model that does not think generates the same handful
  of tokens it did before.

**Upgrading:** a judge on a model that still accepts `temperature` (Sonnet
4.6, Opus 4.6, the 4.5 line and older) now samples at the API default instead
of 0, so its scores are no longer pinned run to run - expect more variance in
a rubric dimension or a win-rate than before, and re-baseline if a gate sits
close to its threshold. A judge on a thinking model also now bills thinking
tokens on every call. Pass `max_tokens=` to a client to keep the old budget.

## [2.4.2] - 2026-08-09

Every `llm_as_judge` row now describes what it measures.

Shipped as a patch. It changes what published rows contain, so the 1.0.0
policy would call it a minor; it goes out as 2.4.2 as a deliberate exception,
alongside 2.4.1's judge-pin fix that it completes. No schema change - the
columns already exist and were empty.

### Fixed
- A judge's dimension row (`<grader>.<dimension>`) carried no judge
  information at all: no `judges.*`, and `judge_scale` of 0. So `0.400` on a
  row could not say who scored it, on what scale, or that the raw number was
  2 out of 5 - the scale lived only on `<grader>.overall`, and recovering the
  raw point meant joining back to it.

  Each dimension row now carries the verdict for the dimension it measures:
  `judges.name`, `judges.version`, `judges.rationale`, `judge_scale`, a
  `judges.points` **scoped to that dimension**, and a `judges.score` that is
  that judge's number for it.

  `<grader>.overall` is unchanged and still carries the full points map.

  This establishes one invariant across every judge row, dimension rows
  included: `value == mean(judges.score)`. With one judge they are equal;
  with a panel, `value` is the mean and `judges.score` shows the spread - so
  per-dimension disagreement is readable from the dimension row instead of
  by unpacking the map on `.overall`.

**Upgrading:** `notEmpty(judges.name)` now matches every judge row rather
than only `<grader>.overall`, so a six-case run goes from 12 judged rows to
48. Anything aggregating over judges must filter to `.overall` or it
multiplies by the dimension count. That is a change in results, not an error.

## [2.4.1] - 2026-08-09

The judge's identity is now in its provenance pin.

Shipped as a patch. Both changes below alter a published value, so the 1.0.0
policy would call this a minor at least; it goes out as 2.4.1 as a deliberate
exception, because the old pin was answering a provenance question wrongly
and the sooner it stops the fewer runs are affected.

### Fixed
- `Scorecard.judge_version` includes the judge's model:
  `anthropic:claude-sonnet-4-6@v1` rather than `judge@v1`. It was
  `key@judge_version`, so swapping the judge's model while leaving the
  declared `judge_version` alone produced a byte-identical pin - and any
  provenance check reading that field passed a comparison against a baseline
  scored by a different model. The model is the thing most likely to change
  and the thing a declared version is most likely to miss.

### Changed
- A single judge's key defaults to its provider (`anthropic`) instead of the
  literal `judge`. A panel already defaulted to `key or provider`, so the
  one-judge case was the odd one out, and `judge` named nothing the `grader`
  column did not already say. It appears in the `judges.name` column.

**Upgrading:** every suite with a judge re-baselines once, because
`judge_version` is part of what identifies a comparable run - which is the
intended behaviour, just paid all at once. Queries filtering
`judges.name = 'judge'` need the provider instead. Set `key:` on a `judges:`
entry to pin a name of your own.

## [2.4.0] - 2026-08-09

A run without a baseline can now say whether it passed.

### Added
- `RunResult.checks`, a `models.ThresholdCheck` the runner fills in: the
  suite's **absolute** guardrails (`min`/`max`) measured against this run.
  Most of a suite's rules are absolute and answerable from one run; only the
  win metric and the relative rules need a baseline. Nothing new to call -
  `compare` is for comparing, and a single run was never a comparison.
- `compare.check_thresholds(scorecard, thresholds)`, which the runner uses.

### Changed
- **Breaking:** `GuardrailResult.passed` is now `bool | None`. `None` means
  the rule could not be evaluated - a `must_not_increase` /
  `must_not_decrease` with no baseline. Test `passed is False` for a breach;
  `not passed` now also catches the skipped case. The Markdown and HTML
  reporters render it as `skipped` rather than a breach.
- `store.score_rows` reports `run.checks` on the gate columns when no
  `Comparison` is passed: `gate_verdict` and the per-metric `guardrail` become
  real, while `gate_win` stays `'none'`. That is deliberate - `gate_win` is
  what marks the three `win_*` fields as never computed rather than measured
  at zero, and an ungated run must not claim a comparison happened. A rule
  that could not run reports `guardrail = 'none'` with the reason in
  `guardrail_gap`, not a false `pass`.
- A `Comparison` supersedes the run's own checks: it evaluates the same rules
  with a baseline available, so it gets the relative ones too.

## [2.3.0] - 2026-08-08

A run now describes its own scores.

### Added
- `RunResult.graders`, a map of grader name to `models.GraderInfo`
  (`category`, `scale`), filled in by the runner from the graders it built.
  A `Score` names the grader that emitted it and nothing else, so a run could
  not previously answer for itself: a `run.json` written by one CI step and
  published by a later one needed the suite file alongside it, and without one
  every grader reported `unknown` and a suite with a judge could not be
  published at all. Run-grain rather than score-grain - one entry per grader,
  not two fields repeated across a couple of hundred rows.
- `models.GraderInfo`.
- `register` also records the category on the decorated class as
  `grader_category`, so a grader instance can answer for itself. The runner
  reads it the same way it already collects `judge_version`.

### Changed
- `store.score_rows` takes the grader category and judge scale from
  `run.graders`. `grader_types` and `judge_scales` still override it, and are
  how a run written before this release - whose map is empty - publishes
  correctly. `grader_lookups` is unchanged and still builds them from a suite.

## [2.2.0] - 2026-08-08

### Added
- `store.ScoreExporter`, a `runtime_checkable` Protocol naming the seam
  `JsonlOutboxExporter` already occupied. `store.py` has always said to
  "replace this class with a database client implementing the same
  `export_scores` method", but the contract was a docstring sentence and
  `**kwargs`, so an implementation had to duck-type a private shape. An
  exporter belongs in the package that owns the store it targets - it is the
  store that knows its own column types, null policy, and transport - and
  swapping one for another is now a constructor line at the call site, so an
  offline run and a live one share a code path.

### Fixed
- `examples/quickstart/graders.py` still used the one-argument `register` and
  raised `TypeError` on import, so 2.1.0 shipped with its own bundled example
  broken. `just test` does not run the example; `just test-all` does.
- The README's Python API example called a `.export()` that does not exist and
  passed `RunResult`s to `compare.compare` and `render_scorecard`, which take
  `Scorecard`s. It now runs verbatim, and shows `grader_lookups` feeding the
  exporter.

## [2.1.0] - 2026-08-07

A grader declares what kind of check it is at registration, so a consumer
plug-in is categorised the same way a built-in is.

Shipped as a minor despite the signature change below. `register` is public,
so the 1.0.0 policy would call this a major; it goes out as 2.1.0 as a
deliberate exception, because the break is a one-line edit per grader that
fails loudly at import.

### Changed
- **Breaking:** `graders.base.register` takes a required second argument,
  `category`, a `graders.GraderType`. Every `@base.register('foo')` becomes
  `@base.register('foo', base.GraderType.HEURISTIC)` or whichever member
  applies; omitting it is a `TypeError` at import. Required rather than
  defaulted on purpose - it is the only source of a row's `grader_type`, and
  a default would be the value every grader forgets to override.
- `store.grader_lookups` reads the category from the registry instead of a
  closed table of built-in type names, so a plug-in that declares
  `HEURISTIC` reports `heuristic` where it used to report `unknown`. Rows for
  consumer graders change value in the `grader_type` column; nothing about
  the row shape changes.

### Added
- `graders.GraderType`, a `StrEnum` over the closed set the results store's
  `grader_type` column accepts: `unknown`, `heuristic`, `statistical`,
  `llm_as_judge`, `trajectory`, `human`. A `StrEnum` so it needs no
  serializer of its own on the way to a row.
- `graders.category_of`, the registry lookup behind `grader_lookups`.

### Removed
- `store._GRADER_TYPES`, the private table the categories used to live in.
  Keeping it alongside the registration argument would mean two sources for
  one fact and a precedence rule between them.

## [2.0.0] - 2026-07-31

Breaks both the public API and the outbox row shape, so it's a major per the
1.0.0 policy.

### Changed
- **Breaking:** a sample is identified by `sample_hash` - a content digest of
  the output it produced - rather than by the ordinal `sample_idx`. A row now
  names the exact response behind it. The field is renamed on `CaseResult`,
  `Rating`, `Preference`, `PairwiseOutcome` and `PairwiseAgreementCase`, and in
  the outbox rows. Two runs of the same case never share a hash, so anything
  comparing runs aligns on `case_id` and sample order; pairwise and the ranking
  app anchor a pair on the `variant_a` side's digest so human and judge picks
  still join. **Ratings and preferences files written by 1.x do not carry a
  hash and will not join to a run** - re-collect them, or backfill the field.
- **Breaking:** outbox keys track the store's column names: `created_at` is
  emitted as `started_at`, `n_cases` as `case_count`, `n_samples` as
  `sample_count`.
- **Breaking:** `store.read_checkpoint_results` is now
  `store.read_checkpoint_samples` and returns `(ordinal, result)` pairs.
  Checkpoint lines nest the result under `result` and tag it with `sample`, so
  resume knows which samples are still owed. 1.x checkpoints cannot be resumed.

### Fixed
- Resuming a run with `concurrency > 1` could skip a sample and re-run another.
  A concurrent run checkpoints in completion order, so an interrupt leaves a
  hole rather than a clean prefix; resume now reruns the samples that are
  actually missing. On a deterministic target the re-run collided with a digest
  already recorded, so the store collapsed two samples into one row.
- The pairwise judge and the side-by-side ranking app filtered samples
  differently - non-empty content vs. no error - so with `n_samples > 1` and an
  error on either side they paired A's n-th sample against different B samples.
  `agreement` joins the two on A's digest, so it scored two different
  comparisons as one. Both now go through
  `pairwise.comparable_samples`, which requires a successful invocation *and*
  resolvable content. Two behaviour changes fall out: an errored output is no
  longer judged even when its content ref still resolves, and a sample with
  empty content is no longer shown to a rater as a blank panel.

### Changed (internal)
- `pairwise._content_map` is now `pairwise.comparable_samples` and is the one
  place that decides whether a sample can take part in a comparison.

### Removed
- `docs/clickhouse-schema.{sql,html}`. The outbox targets a flat row shape, not
  one vendor's DDL, and the file documented a specific deployment - database
  name, ingestion topology, sample data - none of which the engine needs. A
  store with column types the feed does not match maps the rows at its own
  boundary.

## [1.0.0] - 2026-07-28

First stable release. The public API and the outbox row shape are now covered
by semantic versioning: a breaking change to either means a 2.0.

### Changed
- **Breaking:** outbox rows no longer carry `input_tokens`, `output_tokens` or
  `cost`. `Output.tokens` is an open dict an adapter may put anything in, and
  usage accounting is captured outside the results store.
- **Breaking:** the run-grain `revision` key is emitted as
  `application_revision`, which is its column name in the store.

### Added
- `docs/clickhouse-schema.sql`: the ClickHouse schema the outbox rows target -
  the written `evaluation_scores` table, the row grammar as `CONSTRAINT`s, and
  the canonical scorecard query. Also rendered as
  `docs/clickhouse-schema.html`. (Both removed again in 2.0.0.)

## [0.3.0] - 2026-07-28

### Changed
- **Breaking:** the results-store outbox is now one feed at
  `(run, case, sample, grader, metric)` grain, targeting a single flat
  `evaluation_scores` table with the run trend and the invocation grain as
  plain views over it. `store.scorecard_rows` and
  `JsonlOutboxExporter.export` are removed; use `store.score_rows` and
  `export_scores`. `--export-scores` is now an alias for `--export`.
- **Breaking:** a missing measurement is emitted as `null` rather than the
  `(value=0, has_value=false)` sentinel pair, and the `has_value` / `has_stdev`
  companion keys are gone. The store columns are `Nullable`, and a real `0.0` is
  a meaningful score.
- **Breaking:** outbox row keys are the store column names, so `project` is
  emitted as `application`, `mode` as `adapter_mode`, `created_at` as
  `timestamp`, and `Output.error` as `is_error` plus `error_text`.
- **Breaking:** outbox rows no longer carry `model_id` or `prompt_version`. Both
  are `variant.knobs.get(...)` projections and travel inside `variant_knobs`.
- **Breaking:** `run_id` is now a dashed UUIDv7 rather than `uuid4().hex`, so it
  parses as a ClickHouse `UUID` and carries its own creation time. Uses
  `uuid.uuid7()` on 3.14 and an RFC 9562 implementation on 3.11 through 3.13.
- Outbox rows now carry the gate: `gate_verdict`, `gate_win`,
  `baseline_run_id`, `baseline_variant`, `win_baseline`, `win_candidate`,
  `win_delta` and `gate_summary` at run grain, plus `win`, `guardrail` and
  `guardrail_gap` on the metric each one refers to. Pass the `Comparison` to
  `score_rows`; without it a run reads as ungated.

### Added
- `RunResult.aggregate_scores`, retaining the `kind='aggregate'` scores so a
  store row keeps the grader that emitted them and what it reported. The
  scorecard kept only their values.
- `store.grader_lookups`, mapping grader names to a category and to a judge
  scale from a suite's grader specs, since a `Score` carries neither.
- Three outbox row shapes that previously had no representation: an aggregate
  metric (no `case_id`), an invocation that failed before any grader ran (no
  `grader` or `metric`), and a metric a guardrail or the win metric names but
  never scored (null value).

## [0.2.0] - 2026-07-16

### Changed
- **Breaking:** the import package and CLI are now `evalcore` (were `evalkit`).
  Update `import evalkit` to `import evalcore` and the `evalkit` command to
  `evalcore`. The distribution name (`evalcore`) is unchanged.
- Lowered the minimum Python to **3.11** (was 3.14).

### Added
- `evalcore.__version__`.
- Public `evalcore.adapters.expand_env` for `${VAR}` expansion in custom
  adapters (replaces the private `adapters._env` module).
- Exception hierarchy: `EvalcoreError` (base) and `ConfigError` (also a
  `ValueError`, so existing handlers keep working).
- Top-level convenience entry points: `load_suite`, `load_cases`, `run_suite`,
  `run_suite_sync`.
- HTML rendering for `sweep` and `pairwise` reports, and `--report` /
  `--report-out` on those CLI commands.

## [0.1.0] - 2026-07-16

- Initial public release: adapters (http/replay), graders (deterministic,
  numeric, classification, LLM judge + panel), runner (N-sampling, concurrency,
  retries, checkpoint/resume), compare/gate, sweep, pairwise, blind human
  rating + ranking with judge agreement, Markdown/HTML reporters, JSON +
  column-store outbox, and content-hash provenance.

[Unreleased]: https://github.com/scottpmiller/evalcore/compare/2.5.0...HEAD
[2.5.0]: https://github.com/scottpmiller/evalcore/compare/2.4.3...2.5.0
[2.4.3]: https://github.com/scottpmiller/evalcore/compare/2.4.2...2.4.3
[2.4.2]: https://github.com/scottpmiller/evalcore/compare/2.4.1...2.4.2
[2.4.1]: https://github.com/scottpmiller/evalcore/compare/2.4.0...2.4.1
[2.4.0]: https://github.com/scottpmiller/evalcore/compare/2.3.0...2.4.0
[2.3.0]: https://github.com/scottpmiller/evalcore/compare/2.2.0...2.3.0
[2.2.0]: https://github.com/scottpmiller/evalcore/compare/2.1.0...2.2.0
[2.1.0]: https://github.com/scottpmiller/evalcore/compare/2.0.0...2.1.0
[2.0.0]: https://github.com/scottpmiller/evalcore/compare/1.0.0...2.0.0
[1.0.0]: https://github.com/scottpmiller/evalcore/compare/0.3.0...1.0.0
[0.3.0]: https://github.com/scottpmiller/evalcore/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/scottpmiller/evalcore/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/scottpmiller/evalcore/releases/tag/0.1.0
