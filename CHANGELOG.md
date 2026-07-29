# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
  the written `evaluation_scores` table, the run-grain `evaluations` view over
  it, the row grammar as `CONSTRAINT`s, and the trend queries. Also rendered as
  `docs/clickhouse-schema.html`.

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

[Unreleased]: https://github.com/scottpmiller/evalcore/compare/1.0.0...HEAD
[1.0.0]: https://github.com/scottpmiller/evalcore/compare/0.3.0...1.0.0
[0.3.0]: https://github.com/scottpmiller/evalcore/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/scottpmiller/evalcore/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/scottpmiller/evalcore/releases/tag/0.1.0
