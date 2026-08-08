# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/scottpmiller/evalcore/compare/2.3.0...HEAD
[2.3.0]: https://github.com/scottpmiller/evalcore/compare/2.2.0...2.3.0
[2.2.0]: https://github.com/scottpmiller/evalcore/compare/2.1.0...2.2.0
[2.1.0]: https://github.com/scottpmiller/evalcore/compare/2.0.0...2.1.0
[2.0.0]: https://github.com/scottpmiller/evalcore/compare/1.0.0...2.0.0
[1.0.0]: https://github.com/scottpmiller/evalcore/compare/0.3.0...1.0.0
[0.3.0]: https://github.com/scottpmiller/evalcore/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/scottpmiller/evalcore/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/scottpmiller/evalcore/releases/tag/0.1.0
