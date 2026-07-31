# evalcore - design overview

evalcore is a **consumer-agnostic** evaluation engine. It scores a candidate
against a baseline over a fixed dataset, applies guardrails plus one headline
win metric, and produces a gate verdict. It knows nothing about any particular
system under test; a *consumer* supplies the system-specific pieces as data and
small plug-ins.

This doc records the design: where the engine/consumer boundary sits, the core
domain model, and how results are stored. For hands-on usage see the top-level
`README.md` and the worked example under `examples/quickstart/`.

---

## 1. The boundary

The whole design rests on one line: **what the engine owns vs. what a consumer
owns.** If that boundary is clean, onboarding a second system is "write an
adapter + datasets + graders config," not "fork the framework." If it leaks,
every new consumer reopens the core.

This shape is well-trodden - promptfoo, OpenAI Evals, Braintrust, and LangSmith
all converge on the same pipeline (case -> target -> graders -> comparison ->
gate). evalcore is not inventing the abstraction; it is drawing the line so the
engine stays reusable across unrelated systems.

## 2. Generic vs. consumer-specific

| Concern | **Engine** | **Consumer** |
|---|---|---|
| Case / suite schema + loading | yes | uses it |
| Runner: sampling (N), concurrency, live/replay modes | yes | picks mode/config |
| Target-adapter protocol + built-ins (HTTP, replay) | yes | configures/extends |
| Variant model (baseline vs candidate knobs) | yes opaque dict | defines its knobs |
| Grader protocol + registry | yes | registers its graders |
| Generic graders: deterministic, LLM-judge, classification, numeric | yes | configures + writes custom |
| Comparison / regression engine (win metric, guardrails) | yes | sets thresholds |
| Gate decision (comparison -> pass/warn/fail) | yes | sets policy |
| Results store + row schema / outbox | yes | gets a `project` namespace |
| Reporters (Markdown / HTML) | yes | gets them free |
| Datasets (cases + labels) | no | yes owns |
| Judge **rubrics** | no | yes owns |
| Custom graders / adapters | no | yes owns |
| Thresholds / guardrail config | no | yes owns |
| Which suite runs on which change trigger | no | yes owns |

Everything in the left column is consumer-agnostic. The right column is what a
new team writes.

## 3. Core domain model

Eight concepts, none of which mention any particular system:

- **Suite** - a named eval belonging to a `project` (e.g. `my-service/reply`).
  The unit of versioning and gating.
- **Case** - `{ id, input, expected?, labels? }`. `input` and `expected` are
  **opaque blobs** the engine never interprets - only the adapter and graders
  do. This is what keeps the core generic.
- **Variant** - the thing being compared: a dict of knobs
  (`{model, prompt_version, ...}`). The engine treats it as opaque; the adapter
  understands it. "Baseline" and "candidate" are two variants.
- **Target adapter** - `invoke(case, variant) -> Output`. The one seam to the
  system under test. The built-in **HTTP adapter** (POST a templated body, read
  a response) covers most request/response APIs; the **replay adapter** returns
  recorded outputs so the whole pipeline runs offline. Consumers register their
  own for anything else.
- **Grader** - `grade(case, output) -> [Score]` where
  `Score = {metric, value, passed?, detail}`. Registry-based; tiered
  (deterministic -> numeric -> classification -> LLM judge).
- **Run** - execute `suite × variant` over all cases × N samples -> outputs +
  scores, persisted as a `RunResult`.
- **Scorecard** - aggregated scores for one run (mean/stdev per metric).
- **Comparison + Gate** - candidate scorecard vs baseline -> deltas, guardrail
  checks, win metric -> `pass | warn | fail`.

```python
class TargetAdapter(typing.Protocol):
    async def invoke(self, case: Case, variant: Variant) -> Output: ...

class Grader(typing.Protocol):
    name: str
    def grade(self, case: Case, output: Output) -> list[Score]: ...  # may be async
```

Two protocols (plus an `AggregateGrader` variant for whole-run metrics) are the
entire extension surface. Everything else is engine.

## 4. Where it lives - a library, not a central service

A tempting wrong turn is a central "eval service" that calls out to every
team's system. That inverts the dependency (the platform needs network + auth
into every consumer's environment) and becomes a bottleneck. Instead:

- **evalcore is a library.** It runs **in the consumer's CI**, right next to the
  system under test, where the network path and the service's own auth already
  exist. Runner, adapters, graders, and the comparison engine all ship here.
- **An optional shared results store is the only central component** - a column
  store holding multi-tenant scorecards keyed by `project`/`suite`, plus a
  dashboard for trend-over-time. No business logic, just storage + views.

So "the platform" = a library + (optionally) a shared store + conventions.
Coupling stays low; each consumer owns its runs.

## 5. The consumer contract

A consumer adds an eval tree (its own repo, or a directory like
`examples/quickstart/`) and implements exactly four things:

1. **Adapter config** - usually just declare the built-in HTTP adapter:
   endpoint, how to build the request body from `case.input`, how to extract
   `Output` fields from the response, auth, and the **variant knobs**.
2. **Datasets** - versioned `cases/*.yaml` with `input` + optional
   `expected`/`labels`.
3. **Graders** - pick generic ones (deterministic checks, numeric, judge with
   rubric, classification) + register any custom graders; supply judge rubrics.
4. **Suite + threshold config** - guardrail metrics, win metric + dead band,
   N samples, which triggers run which suite.

The engine supplies runner, comparison, gate, store, reporters, and the CLI.
That ratio - four data files vs. a whole engine - is the genericity test: if
onboarding a consumer ever requires editing `src/evalcore/`, that's an
abstraction leak to fix at the seam, not a fork.

## 6. Results store & outbox

Scorecards and comparisons serialize to JSON for CI artifacts and local files.
For trend tracking, results can land in a column store (e.g. ClickHouse) keyed
by `project`/`suite`. Rather than couple the engine to any particular database
driver, `store.JsonlOutboxExporter` flattens a run into a stable, flat row
shape and writes JSONL to an **outbox** a separate shipper drains: one feed at
`(run, case, sample, grader, metric)` grain, one row per score
(`run_id, case_id, sample_hash, grader, metric, metric_kind, value, passed,
detail`), carrying the invocation it came from and the gate verdict alongside.

Nothing derived is written. A run's scorecard and its trend are read-time
aggregations over the same rows, so they cannot disagree with the scores behind
them.

Every row repeats the full reproducibility key so a multi-tenant trend table can
filter/group on any dimension without joins.

A measurement that does not exist is `null`. A real `0.0` is a meaningful score -
what a failing deterministic check earns - so filling an absent one in with 0
would make the two unreadable apart, and `avg`/`sum`/`count` skip nulls anyway,
so no query has to remember a filter. `metric_kind` carries the engine's word
verbatim and says nothing about presence, so a metric some cases could not score
still reads as one metric; `'none'` is reserved for the two row shapes that hold
no score at all. `passed` is the tri-state string
`'true'|'false'|'null'`, since a judge has no pass line by design: a third
state, not a missing value.

A store that forbids nullable columns is free to fill those nulls in at ingest.
That mapping belongs at its boundary, not in the engine, which keeps absence
representable all the way out. Swap the exporter for a real database client
without touching the runner or any consumer.

## 7. Provenance

Every scorecard (and every outbox row) carries the full reproducibility key:
`project, suite, variant, dataset_version, model_id, prompt_version,
judge_version, revision, suite_hash, dataset_hash, mode, created_at`.

The declared versions state *intent*; the engine-computed content hashes prove
it. `suite_hash` digests the raw suite file; `dataset_hash` digests the loaded
cases (order- and formatting-independent). Two runs whose hashes match
evaluated the same config over the same data - so if a metric moved, the system
under test moved. If a hash changed, the eval itself changed and the comparison
is apples-to-oranges: re-baseline. `revision` ties a run to whatever provenance
scheme the consumer uses (commit SHA, image digest, release label); the engine
never interprets it.
