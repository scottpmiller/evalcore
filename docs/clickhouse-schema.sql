-- eval results store - ClickHouse schema
--
-- One table, internal.evaluation_scores: one row per
-- (run, case, sample, grader, metric). Nothing derived is stored and no views
-- are defined; a run's scorecard is a read-time aggregation, so it cannot
-- disagree with the scores behind it. The canonical scorecard query:
--
--   SELECT grader, metric,
--          avg(value) AS value,
--          if(metric_kind = 'per_case' AND count(value) > 1,
--             stddevSamp(value), NULL) AS stddev
--   FROM internal.evaluation_scores FINAL
--   WHERE run_id = {run:UUID}
--     AND metric != ''  -- a failed invocation is not a scorecard entry
--   GROUP BY grader, metric, metric_kind;
--
-- FINAL collapses a redelivered insert before it is counted. Grouping by
-- grader keeps two graders that emit the same metric name apart. Run level,
-- guardrails_ok is countIf(guardrail = 'fail') = 0 over the same rows.
--
-- Row shapes. An empty key column means the row does not sit at that grain.
-- The CONSTRAINTs at the bottom of the table enforce this grammar, since
-- callers insert directly and the library cannot vouch for their rows.
--   per-case score      case_id, grader and metric all set. The common row.
--   aggregate metric    case_id = '', sample_idx = 0. Aggregate graders compute
--                       once over the whole run and produce no CaseResult.
--   failed invocation   grader = '', metric = '', value = NULL,
--                       metric_kind = 'none', is_error = true. The adapter
--                       failed before any grader ran.
--   named-but-unscored  a metric a guardrail or the win metric names that
--                       produced no score: metric set, value = NULL,
--                       metric_kind = 'none'. A guardrail on a missing metric
--                       is a breach, so it has to be representable.
--
-- Results are append-only and the schema is denormalized on purpose. The run
-- key repeats on every row, and things read together sit in Nested and JSON
-- columns rather than child tables. No joins within the eval data.
--
-- Types, names and formatting follow the shared ClickHouse conventions:
-- `timestamp` as the event-time column and partition key, `application` for
-- the system under test (what evalcore calls project), UUID for ids, Decimal
-- for durations, JSON for arbitrary dicts, codecs only on genuinely
-- large columns, two-space indent, short section comments, PARTITION BY before
-- ORDER BY.
--
-- Conventions we keep that the shared style does not use:
--   * Nullable on measurements that are genuinely optional, rather than a bare
--     DEFAULT 0. Those tables can default to 0 because their numbers are always
--     computed; ours are often absent, and 0 is a real score. avg/sum/count
--     skip nulls, so no query has to remember a filter.
--   * passed is Enum8('null','true','false'). Not a missing measurement: a
--     judge has no pass line by design.
--
-- The JSON columns set the server floor: the JSON type is GA in ClickHouse
-- 25.3, and 24.8+ needs allow_experimental_json_type. Callers do their own
-- inserts, so the floor is theirs to meet.
--
-- No TTL. At roughly 200 rows per run the numbers are cheap to keep, and the
-- run trend is read from them, so expiring rows would expire the trend.
--
-- Still NOT aligned, and deliberately out of scope here:
--   * house DDL is `CREATE OR REPLACE TABLE ... ON CLUSTER primary` with
--     ReplicatedReplacingMergeTree; this file still uses the single-node form
--   * house ingestion is a Kafka engine twin plus a materialized view, not
--     direct inserts
--   * house layout is one table per file under tables/<database>/, registered
--     in MANIFEST
--
-- Exporter notes:
--   * rows are published as JSONEachRow to Kafka by the internal-eval-results
--     library: internal.evaluation-scores. It owns the row shape, so this table
--     and its Pydantic model have to be changed together.
--   * send null for an absent measurement, not a (0, has_value=false) sentinel
--     pair. Nullable columns are the store convention here.
--   * key names must match the column names: Scorecard.project maps to
--     `application`, .mode to `adapter_mode`, .created_at to `timestamp`,
--     .revision to `application_revision`, and Output.error to `is_error`
--     plus `error_text`.
--   * model_id and prompt_version are not columns. Both are
--     variant.knobs.get(...) projections, so they live in variant_knobs.
--   * metric_kind carries Score.kind verbatim, 'per_case' or 'aggregate'.
--   * duration is Output.latency_ms divided by 1000.
--   * Output.tokens and Output.cost are not exported. Usage accounting is
--     captured outside the results store.
--   * a Nested column is one array column per subfield, so a message names
--     `judges.name` and not `judges`, and names every subfield even when the
--     list is empty.
--   * judges is populated only on the <grader>.overall metric. Every other row
--     carries empty arrays.
--   * the gate columns are run grain and repeat on every row, so the exporter
--     publishes once after the run completes, with the Comparison in hand. It
--     joins Comparison.guardrails onto the metric rows by metric name to fill
--     `guardrail` and `guardrail_gap`.
--   * run_id is UUIDv7. Python's uuid.uuid7() is 3.14; on 3.11 through 3.13 the
--     library generates it per RFC 9562. Send the dashed form; ClickHouse will
--     not parse .hex.
--   * timestamp is second precision, so the sub-second part of
--     datetime.isoformat() is dropped on the way in.
--
-- Open:
--   * metric names keep the dot (`quality.overall`). That delimiter is ours,
--     unlike the dots in judges.*, which the Nested type forces.
--   * Output.fields, the normalized output the graders scored, is not stored.
--     Neither are Output.raw, Case.input or Case.expected.
--
-- The sample data is illustrative, not a recorded run. It models an eval of
-- universal-builder's landing-pages AI builder: a gate comparing PromptLayer
-- editor template 186239 at v36, which production and staging are pinned to,
-- against v39, which testing resolves, over 12 page-generation prompts at 2
-- samples each.

CREATE DATABASE IF NOT EXISTS internal;


-- ===========================================================================
-- evaluation_scores - the only written table.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS internal.evaluation_scores
(
  -- Event Metadata
  timestamp              DateTime('UTC'),
  -- The system under test, and the tenant namespace. This is what evalcore
  -- calls suite.project.
  application            LowCardinality(String),
  -- UUIDv7. ClickHouse sorts the UUID type by its second half, so this cannot
  -- be range-pruned by time on its own; timestamp precedes it in the sort key.
  run_id                 UUID,

  -- Eval Identity. Constant within a run, so it repeats on every row and
  -- compresses to almost nothing.
  suite                  LowCardinality(String),
  variant                LowCardinality(String),
  -- A consumer-authored label, not an ordered version. dataset_hash is what
  -- proves two runs used the same cases.
  dataset_version        LowCardinality(String),
  -- Collected by the runner from graders that expose a pin, joined with ';'.
  -- Not a variant knob: it describes the measuring instrument, so a change
  -- invalidates comparability the same way suite_hash does.
  judge_version          LowCardinality(String),
  -- The application's revision as CI knows it: git SHA, image digest, release
  -- label. Opaque to the engine. Named for what it identifies; the run's id
  -- is run_id.
  application_revision   String DEFAULT '',
  -- Content digest of the suite config. Two runs sharing it ran the same
  -- graders and thresholds.
  suite_hash             String DEFAULT '',
  -- Content digest of the loaded cases, independent of file order and
  -- formatting.
  dataset_hash           String DEFAULT '',
  -- 'http' called the real service; 'replay' read recorded fixtures offline.
  adapter_mode           LowCardinality(String),
  -- Carried rather than counted. A case that produced no rows at all would be
  -- missing from uniqExact(case_id).
  n_cases                UInt16,
  n_samples              UInt16,
  -- The variant's config dict, stored as the consumer wrote it and opaque to
  -- the engine. Model and prompt live in here rather than in columns of their
  -- own; not every consumer under test is an LLM.
  variant_knobs          JSON DEFAULT '{}',

  -- Score
  -- Empty on aggregate rows.
  case_id                String,
  sample_idx             UInt16,
  -- Both empty on a failed invocation.
  grader                 LowCardinality(String),
  -- What kind of check this is, not which one. A closed set, because the
  -- grader registry is consumer-extensible but the categories are not. The
  -- names are the industry ones rather than evalcore's module names:
  --   heuristic     rule or code based - regex, length, non-empty, tolerance
  --   statistical   computed against ground truth - precision, recall, f1
  --   llm_as_judge  model graded against a rubric
  --   trajectory    scores the path, not the output - tool-use sanity, turn
  --                 efficiency, error recovery. Needs the adapter to surface
  --                 the trajectory.
  --   human         a person's rating. Empty today; human ratings stay in
  --                 files, but adding an Enum member later is a migration.
  -- Score carries no category, so the exporter maps the built-in registry keys
  -- off suite.graders; a custom grader that declares none lands on 'unknown'
  -- rather than being guessed at.
  grader_type            Enum8('unknown' = 0, 'heuristic' = 1,
                               'statistical' = 2, 'llm_as_judge' = 3,
                               'trajectory' = 4, 'human' = 5),
  metric                 LowCardinality(String),
  -- Score.kind verbatim. 'per_case' came from one case and gets averaged;
  -- 'aggregate' was computed once over the whole run. 'none' on a failed
  -- invocation and on a named-but-unscored metric.
  metric_kind            Enum8('none' = 0, 'per_case' = 1, 'aggregate' = 2),
  -- Null when the grader produced no number at all, which numeric graders do
  -- on a missing field and judges do on an unscored dimension. Distinct from a
  -- real 0.0, which is what a failing deterministic check scores.
  value                  Nullable(Float64),
  -- Three states. Deterministic graders set true or false; a judge has no pass
  -- line by design and leaves it null.
  passed                 Enum8('null' = 0, 'true' = 1, 'false' = 2),
  detail                 String DEFAULT '' CODEC(ZSTD(3)),
  -- The case's own labels, defined by the consumer and never interpreted.
  case_labels            JSON DEFAULT '{}',

  -- Gate, at metric grain. Every row for the metric carries these.
  -- True on the single metric the gate decides on. Guardrails can veto it, but
  -- only this one counts as winning.
  win                    Bool DEFAULT false,
  -- Whether this metric was a guardrail and how it came out. 'none' means it
  -- was not one.
  guardrail              Enum8('none' = 0, 'pass' = 1, 'fail' = 2),
  -- What the check measured against, in the engine's own words: '0.0000 ok'
  -- on a pass; '0.9583 < min 1.0', '1.0000 > max 0.0',
  -- 'increased 0.7100 -> 0.8200', or 'metric absent on candidate' on a
  -- breach. Empty when guardrail is 'none'.
  guardrail_gap          String DEFAULT '',

  -- Invocation. Grain is (case, sample), one level coarser than this table, so
  -- these repeat across a case's metric rows. Aggregate per (run_id, case_id,
  -- sample_idx) with any() before summing, or sums multiply by the metric
  -- count. All unset until an adapter populates them. No token or cost
  -- columns: usage accounting is captured elsewhere.
  duration               Nullable(Decimal(9, 3)),
  -- Whether the invocation itself failed, as opposed to succeeding and scoring
  -- badly.
  is_error               Bool DEFAULT false,
  error_text             String DEFAULT '' CODEC(ZSTD(3)),
  -- Whether the adapter marked the failure transient, a 429 or 5xx, which is
  -- what the runner's retry loop acts on.
  retryable              Bool DEFAULT false,
  artifacts              JSON DEFAULT '{}',

  -- Judge Verdicts. Populated only on the <grader>.overall metric; one entry
  -- per judge in a panel.
  --   points     each rubric dimension to the raw 1..judge_scale score that
  --              judge gave it. Keeping it is how a generous or drifting judge
  --              becomes visible, and it is the only place per-dimension
  --              disagreement is readable.
  --   score      that judge's mean over its own dimensions, normalized as
  --              points / judge_scale. Null when it scored nothing.
  --   rationale  why the judge scored it that way. This is what makes a dropped
  --              metric explainable from the database alone.
  judges                 Nested
  (
    name                 LowCardinality(String),
    version              LowCardinality(String),
    points               Map(LowCardinality(String), Nullable(Float32)),
    score                Nullable(Float32),
    rationale            String
  ),
  -- The scale the raw points sit on. Normalization is points / judge_scale, so
  -- a 1 on a 1..5 scale reads 0.2 rather than 0. Zero on rows no judge touched.
  judge_scale            UInt8 DEFAULT 0,

  -- Gate, at run grain. Repeats on every row. 'none' means the run was never
  -- gated, either a plain run or the baseline half of one.
  gate_verdict           Enum8('none' = 0, 'pass' = 1, 'warn' = 2, 'fail' = 3),
  gate_win               Enum8('none' = 0, 'neutral' = 1, 'improved' = 2,
                               'regressed' = 3),
  baseline_run_id        UUID DEFAULT '00000000-0000-0000-0000-000000000000',
  baseline_variant       LowCardinality(String) DEFAULT '',
  win_baseline           Nullable(Float64),
  win_candidate          Nullable(Float64),
  -- Candidate minus baseline. A genuine zero means no change; null means it was
  -- never computed.
  win_delta              Nullable(Float64),
  gate_summary           String DEFAULT '' CODEC(ZSTD(3)),

  -- Row-shape grammar, from the header. The shapes live in empty-string
  -- conventions and callers insert directly, so the table is the only place
  -- they can be enforced. A violation rejects the whole insert block.
  -- A row with no metric must be a failed invocation.
  CONSTRAINT shape_failed_invocation CHECK metric != '' OR is_error,
  -- A metric with no grader is only the named-but-unscored shape.
  CONSTRAINT shape_named_unscored    CHECK grader != '' OR metric = ''
                                           OR metric_kind = 'none',
  -- 'none' means no score was produced, so there is no value to carry.
  CONSTRAINT shape_unscored_value    CHECK metric_kind != 'none'
                                           OR value IS NULL,
  -- Raw judge points are unreadable without the scale they sit on.
  CONSTRAINT judge_points_readable   CHECK empty(`judges.name`)
                                           OR judge_scale > 0
)
ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (application, suite, variant, timestamp, run_id,
          case_id, sample_idx, grader, metric)
SETTINGS index_granularity = 8192;


-- ===========================================================================
-- Sample data - six rows of the prompt_v39 run, in sort-key order. The Values
-- parser rejects a comment between tuples, so the per-row notes sit up here:
--   1  (no case)               aggregate metric, and a passing guardrail
--   2  coaching_waitlist/1     per-case score, a heuristic that scored a real
--                              0.0
--   3  ebook_lead_magnet/0     per-case score, a judge dimension: no judges
--                              array, since that rides only on .overall
--   4  newsletter_signup/0     per-case score, a heuristic that scored a real
--                              0.0
--   5  pricing_page/0          per-case score, a heuristic that passed
--   6  webinar_registration/1  the win metric, and the one row carrying judge
--                              detail
-- ===========================================================================

INSERT INTO internal.evaluation_scores
(timestamp, application, run_id,
 suite, variant, dataset_version, judge_version, application_revision,
 suite_hash, dataset_hash, adapter_mode, n_cases, n_samples, variant_knobs,
 case_id, sample_idx, grader, grader_type, metric, metric_kind,
 value, passed, detail, case_labels,
 win, guardrail, guardrail_gap,
 duration, is_error, error_text, retryable, artifacts,
 `judges.name`, `judges.version`, `judges.points`, `judges.score`,
 `judges.rationale`, judge_scale,
 gate_verdict, gate_win, baseline_run_id, baseline_variant,
 win_baseline, win_candidate, win_delta, gate_summary)
VALUES
('2026-07-27 14:31:52', 'universal-builder',
 '019839a1-4c70-7d3e-95c2-b8134fa7e6d9',
 'landing-pages', 'prompt_v39', 'v3', 'quality@v2',
 '1f3c9ad7e4b2856c09fa31d5e8b47c2069af3d81', 'a3d19f2b7c04', '7e5c081da9f3',
 'http', 12, 2,
 '{"model": "claude-sonnet-4-6", "prompt_version": "186239@v39", "skill_collection_id": "1eb04771-9c62-4f38-b0a5-5d21e7c4a8f3"}',
 '', 0, 'save_failures', 'trajectory', 'save_failures', 'aggregate',
 0.0, 'null', '0 of 24 saves rejected', '{}',
 false, 'pass', '0.0000 ok',
 NULL, false, '', false, '{}',
 [], [], [], [], [], 0,
 'pass', 'improved', '019839a0-1e30-7a2f-9c3b-5e6a1d8f0427', 'prompt_v36',
 0.71, 0.82, 0.11,
 'quality.overall improved'),
('2026-07-27 14:31:52', 'universal-builder',
 '019839a1-4c70-7d3e-95c2-b8134fa7e6d9',
 'landing-pages', 'prompt_v39', 'v3', 'quality@v2',
 '1f3c9ad7e4b2856c09fa31d5e8b47c2069af3d81', 'a3d19f2b7c04', '7e5c081da9f3',
 'http', 12, 2,
 '{"model": "claude-sonnet-4-6", "prompt_version": "186239@v39", "skill_collection_id": "1eb04771-9c62-4f38-b0a5-5d21e7c4a8f3"}',
 'coaching_waitlist', 1, 'above_fold_cta', 'heuristic', 'above_fold_cta',
 'per_case', 0.0, 'false',
 'first CTA at 1032px, below the 900px fold',
 '{"goal": "waitlist", "sections": 6}',
 false, 'none', '',
 184.220, false, '', false,
 '{"screenshot": "out/coaching_waitlist.1.png", "page_html": "out/coaching_waitlist.1.html"}',
 [], [], [], [], [], 0,
 'pass', 'improved', '019839a0-1e30-7a2f-9c3b-5e6a1d8f0427', 'prompt_v36',
 0.71, 0.82, 0.11,
 'quality.overall improved'),
('2026-07-27 14:31:52', 'universal-builder',
 '019839a1-4c70-7d3e-95c2-b8134fa7e6d9',
 'landing-pages', 'prompt_v39', 'v3', 'quality@v2',
 '1f3c9ad7e4b2856c09fa31d5e8b47c2069af3d81', 'a3d19f2b7c04', '7e5c081da9f3',
 'http', 12, 2,
 '{"model": "claude-sonnet-4-6", "prompt_version": "186239@v39", "skill_collection_id": "1eb04771-9c62-4f38-b0a5-5d21e7c4a8f3"}',
 'ebook_lead_magnet', 0, 'quality', 'llm_as_judge', 'quality.conversion',
 'per_case', 0.8, 'null',
 'judges=quality@v2', '{"goal": "lead_magnet", "sections": 5}',
 false, 'none', '',
 176.940, false, '', false,
 '{"screenshot": "out/ebook_lead_magnet.0.png", "page_html": "out/ebook_lead_magnet.0.html"}',
 [], [], [], [], [], 0,
 'pass', 'improved', '019839a0-1e30-7a2f-9c3b-5e6a1d8f0427', 'prompt_v36',
 0.71, 0.82, 0.11,
 'quality.overall improved'),
('2026-07-27 14:31:52', 'universal-builder',
 '019839a1-4c70-7d3e-95c2-b8134fa7e6d9',
 'landing-pages', 'prompt_v39', 'v3', 'quality@v2',
 '1f3c9ad7e4b2856c09fa31d5e8b47c2069af3d81', 'a3d19f2b7c04', '7e5c081da9f3',
 'http', 12, 2,
 '{"model": "claude-sonnet-4-6", "prompt_version": "186239@v39", "skill_collection_id": "1eb04771-9c62-4f38-b0a5-5d21e7c4a8f3"}',
 'newsletter_signup', 0, 'above_fold_cta', 'heuristic', 'above_fold_cta',
 'per_case', 0.0, 'false', 'first CTA at 1140px, below the 900px fold',
 '{"goal": "subscribe", "sections": 4}',
 false, 'none', '',
 96.410, false, '', false,
 '{"screenshot": "out/newsletter_signup.0.png", "page_html": "out/newsletter_signup.0.html"}',
 [], [], [], [], [], 0,
 'pass', 'improved', '019839a0-1e30-7a2f-9c3b-5e6a1d8f0427', 'prompt_v36',
 0.71, 0.82, 0.11,
 'quality.overall improved'),
('2026-07-27 14:31:52', 'universal-builder',
 '019839a1-4c70-7d3e-95c2-b8134fa7e6d9',
 'landing-pages', 'prompt_v39', 'v3', 'quality@v2',
 '1f3c9ad7e4b2856c09fa31d5e8b47c2069af3d81', 'a3d19f2b7c04', '7e5c081da9f3',
 'http', 12, 2,
 '{"model": "claude-sonnet-4-6", "prompt_version": "186239@v39", "skill_collection_id": "1eb04771-9c62-4f38-b0a5-5d21e7c4a8f3"}',
 'pricing_page', 0, 'head_preserved', 'heuristic', 'head_preserved',
 'per_case', 1.0, 'true', 'tracking and font tags in <head> intact',
 '{"goal": "pricing", "sections": 8}',
 false, 'none', '',
 141.870, false, '', false,
 '{"screenshot": "out/pricing_page.0.png", "page_html": "out/pricing_page.0.html"}',
 [], [], [], [], [], 0,
 'pass', 'improved', '019839a0-1e30-7a2f-9c3b-5e6a1d8f0427', 'prompt_v36',
 0.71, 0.82, 0.11,
 'quality.overall improved'),
('2026-07-27 14:31:52', 'universal-builder',
 '019839a1-4c70-7d3e-95c2-b8134fa7e6d9',
 'landing-pages', 'prompt_v39', 'v3', 'quality@v2',
 '1f3c9ad7e4b2856c09fa31d5e8b47c2069af3d81', 'a3d19f2b7c04', '7e5c081da9f3',
 'http', 12, 2,
 '{"model": "claude-sonnet-4-6", "prompt_version": "186239@v39", "skill_collection_id": "1eb04771-9c62-4f38-b0a5-5d21e7c4a8f3"}',
 'webinar_registration', 1, 'quality', 'llm_as_judge', 'quality.overall',
 'per_case', 0.6, 'null',
 'judges=quality@v2', '{"goal": "registration", "sections": 7}',
 true, 'none', '',
 212.550, false, '', false,
 '{"screenshot": "out/webinar_registration.1.png", "page_html": "out/webinar_registration.1.html"}',
 ['quality'], ['v2'],
 [{'conversion': 3, 'clarity': 3, 'visual': 3}], [0.6],
 ['Three competing nav links pull attention off the signup form; only one CTA should survive.'],
 5,
 'pass', 'improved', '019839a0-1e30-7a2f-9c3b-5e6a1d8f0427', 'prompt_v36',
 0.71, 0.82, 0.11,
 'quality.overall improved');
