# evalcore

A small, **consumer-agnostic** evaluation engine for prompt, model, and API
outputs. It detects regressions and improvements as prompts and models change,
by scoring a candidate against a baseline over a fixed dataset and applying
guardrails + a headline win metric to produce a gate verdict.

evalcore knows nothing about any particular system under test. A consumer
supplies four things; evalcore supplies everything else:

| Consumer provides (data + small plug-ins) | evalcore provides |
| --- | --- |
| an **adapter** config (how to call the system + the knobs a variant sets) | runner (N-sampling), comparison/regression engine |
| **datasets** (cases with opaque `input`/`expected` blobs) | grader registry + generic graders |
| **graders** + judge rubrics | results store (JSON + column-store outbox) |
| a **suite + thresholds** config | Markdown reporting, CLI, gate exit codes |

See `docs/design.md` for the full design. A complete runnable consumer lives
in `examples/quickstart/` (a support-reply eval: custom adapter + custom
graders + deterministic/classification/LLM-judge checks, runnable fully
offline) and doubles as an end-to-end usage reference.

## Install

```bash
pip install evalcore
```

Imports as `evalcore` (`import evalcore`); CLI is `evalcore` (`evalcore --help`).
Extras: `evalcore[http]` (live HTTP adapter), `evalcore[judge]` (live LLM judge).

## Quickstart

Score a candidate against a baseline over a suite and read the gate verdict -
fully offline against recorded fixtures, so it runs with no network or API keys:

```python
import evalcore as ec

suite = ec.load_suite('suite.yaml')
baseline = ec.run_suite_sync(suite, 'baseline', mode='replay').scorecard
candidate = ec.run_suite_sync(suite, 'candidate', mode='replay').scorecard

result = ec.compare.compare(baseline, candidate, suite.thresholds)
print(result.verdict)            # 'pass' | 'warn' | 'fail'
print(ec.report.render_comparison(result))   # Markdown diff + guardrails
```

Or from the command line (exits non-zero on `fail`, so it drops into CI):

```bash
evalcore gate --suite suite.yaml --mode replay
```

`examples/quickstart/` is a complete, runnable version of the above (custom
adapter + graders + suite). The rest of this README is the reference for
writing your own.

## Develop

Standard [uv](https://docs.astral.sh/uv/) project; recipes via
[just](https://github.com/casey/just):

```bash
just sync          # editable install + all extras + dev deps
just test          # engine unit tests (coverage over src/)
just test-example  # run the quickstart example assertions offline
just example       # quickstart suite: scorecards + verdict (replay)
just example-api   # quickstart suite through the Python API (replay)
```

Extras: `http` (httpx, needed only for live adapter runs) and `judge`
(anthropic + openai SDKs, needed only for the live LLM judge). Replay/offline
runs need none of them.

---

# Writing a consumer

A consumer is a directory of data files plus (optionally) a small plug-in
module. Nothing about your system leaks into the engine; everything below
lives in *your* tree (your repo, or a directory under `examples/` here):

```
my_service/
  suite.yaml                     # the suite: adapter + graders + variants + thresholds
  graders.py                     # optional: custom graders/adapters (plug-in module)
  datasets/<name>/v1/cases/*.yaml
  fixtures/replay.yaml           # recorded outputs (offline runs / CI)
  fixtures/judge.yaml            # recorded judgments (offline judge)
```

The five steps, in dependency order:

## 1. Cases (the dataset)

A dataset is a directory containing `cases/`, holding one YAML (or JSON)
file per case. Every field except `id` is **opaque to the engine** - only
your adapter and graders interpret `input` and `expected`:

```yaml
# datasets/support_reply/v1/cases/refund_request.yaml
id: refund_request          # optional; defaults to the filename stem
labels: {category: billing} # optional metadata, useful for slicing later
input:                      # whatever YOUR adapter needs to call the system
  ticket_text: 'I was charged twice for March...'
  customer_tier: pro
expected:                   # optional ground truth for graders
  intent: refund
```

`loader.load_cases(dataset_dir)` reads every `cases/*.yaml|yml|json` in
sorted filename order and validates them into `models.Case`. Version the
dataset by directory (`v1`, `v2`, ...) and set `dataset_version` in the
suite to match; the engine also computes a `dataset_hash` at run time (see
"Provenance" below) so an edited case can't hide behind an unbumped version.

## 2. The adapter (how to call your system)

The adapter is the single seam to the system under test:

```python
class TargetAdapter(typing.Protocol):
    async def invoke(self, case: Case, variant: Variant) -> Output: ...
```

It receives one `Case` and one `Variant` and must return an
`Output` - never raise for a failed invocation; set `Output.error` instead
so graders can count it (see `errors` below). Set `Output.retryable = True`
alongside a *transient* error (a 429, a 5xx, a network timeout) and the
runner's retry loop will back off and try again (see "Retries" below); leave
it False for terminal failures (a bad request, a malformed response) so the
run doesn't burn attempts on something a retry can't fix.

### The built-in `http` adapter

Declares the whole call as data in the suite file:

```yaml
adapter:
  type: http
  base_url: ${MY_SERVICE_BASE_URL}     # ${VAR} expands from the environment
  path: /reply
  method: POST                          # default POST
  timeout: 30.0
  headers:
    Content-Type: application/json
    Authorization: ${MY_SERVICE_JWT}    # header dropped entirely if VAR unset
  body:                                 # template rendered per (case, variant)
    ticket: $input.ticket_text          # $-strings are references, resolved
    tier: $input.customer_tier          #   against the case/variant; anything
    model: $variant.model               #   else passes through literally
  extract:                              # Output.fields <- dotted paths into
    reply: choices.0.text               #   the JSON response (list indices ok)
    intent: analysis.intent
```

Reference roots available in `body`: `$input.*`, `$expected.*`,
`$variant.*` (the variant's knobs), `$case.*` (the whole case). A path that
doesn't resolve degrades to `null` rather than erroring. HTTP failures,
non-2xx statuses, and non-JSON bodies all become `Output.error` values with
the latency still recorded.

### Custom adapters

When the built-in isn't enough - auth dances, response post-processing,
non-HTTP targets - register your own under a config `type` and select it in
the suite. Subclassing the http adapter is often the shortest path.
`examples/quickstart/adapter.py` is a worked example: it turns each case's
input into structured `Output.fields` your graders can score:

```python
from evalcore import models
from evalcore.adapters import base, http

@base.register('my_service_json')
class MyAdapter(http.HTTPAdapter):
    async def invoke(self, case, variant):
        output = await super().invoke(case, variant)
        ...post-process output.fields...
        return output
```

Constructor kwargs come from the suite's `adapter:` mapping (everything
except `type`). Load the module at run time with `--plugins my_service.graders`
(CLI) or a plain `import` (Python API) - registration happens on import.

An adapter need not be HTTP-backed: it can grade *what a deployed system
already did* by reading from an observability store - turning an aggregated
result row into `Output.fields`.

An adapter that holds resources (a browser, an injected session, pooled
connections) may expose an optional async `aclose()`; the runner calls it
after the run, even on failure - so a browser-automation adapter (e.g. one
driving Playwright) can open its context once and tear it down cleanly.

## 3. Variants (what's being compared)

A variant is a named dict of **knobs** - opaque to the engine, interpreted
by your adapter (usually via `$variant.*` refs in the body template):

```yaml
variants:
  baseline:
    model: claude-haiku-4-5
    prompt_version: v3
  candidate:
    model: claude-sonnet-4-6
    prompt_version: v4
```

Two knob names get special treatment: `model` and `prompt_version` are
lifted onto the scorecard as `model_id` / `prompt_version` so results are
self-describing in the store. Everything else is yours.

## 4. Graders (how outputs are scored)

Two protocols; a class implements one or the other and the runner sorts
them into buckets automatically:

```python
class Grader(typing.Protocol):            # per-case; scores averaged
    name: str
    def grade(self, case, output) -> list[Score]: ...   # may be async

class AggregateGrader(typing.Protocol):   # whole-run; scores stored as-is
    name: str
    def aggregate(self, results: list[CaseResult]) -> list[Score]: ...
```

Suite config is a list of `{type, ...kwargs}` specs; `type` selects a
registered class, the rest becomes constructor kwargs. Built-ins:

```yaml
graders:
  # --- deterministic per-case checks (emit 1.0/0.0 + passed) -------------
  - type: max_chars          # len(field) <= maximum
    name: length_ok          # `name` doubles as the metric name
    field: output.reply
    maximum: 400
  - type: regex_absent       # field must NOT match pattern
    name: no_pii
    field: output.reply
    pattern: '\b\d{3}-\d{2}-\d{4}\b'
  - type: regex_present      # field must match EVERY pattern (all-of)
    name: required_markup
    field: output.html
    patterns: ['<form', 'type="email"']
  - type: non_empty          # field must be truthy
    field: output.reply

  # --- numeric (per-case): promote numeric output fields to metrics ------
  - type: numeric            # each field -> a scorecard metric (its mean)
    fields:
      - {ref: output.tool_error_rate, max: 0.09}  # bounded -> pass/fail too
      - {ref: output.hallucination_rate, max: 0.02}
      - output.cost_per_request                    # unbounded -> measurement

  # --- classification (aggregate): P/R/F1 + FN/FP rates ------------------
  - type: classification
    name: intent_detection
    predicted_ref: output.intent     # what the system produced
    expected_ref: expected.intent    # the human-authored answer key
    positive_labels: [refund]        # the class you must not miss
    negative_labels: [question, complaint]

  # --- LLM judge (per-case): rubric scoring, 1..scale -> 0..1 ------------
  - type: llm_judge
    name: quality
    content_ref: output.reply        # the text to judge
    scale: 5
    dimensions:
      - {key: empathy, description: "Acknowledges the customer's situation."}
      - {key: accuracy, description: 'Consistent with the ticket facts.'}
    rubric: |                        # optional free-text rubric for the judge
      Judge as a support-quality reviewer...
    context_refs:                    # extra context shown to the judge
      ticket: input.ticket_text
    model: claude-sonnet-4-6         # single-judge shorthand (live mode)
    judge_version: v1                # bump on ANY judge change; re-baseline
    replay_path: fixtures/judge.yaml # recorded judgments (replay mode)
```

Grader field selectors resolve against roots `input`, `expected`, `output`
(the adapter's extracted fields), `case`, and `artifacts` (the output's
saved files, e.g. `artifacts.screenshot`).

The `numeric` grader is what turns adapter-extracted numbers into scorecard
metrics: only `Score`s reach a scorecard, so a value the adapter merely put in
`Output.fields` (a cost, an error rate) needs a grader to promote it. Each
field's metric name defaults to the ref's leaf (`output.cost` -> `cost`); add
`min`/`max` to also emit per-case pass/fail. Absent or non-numeric fields
degrade to `null`. `compare`'s guardrails and a `win_metric` with
`win_higher_is_better: false` then gate on these directly - e.g. gating a
`generation_cost` or `tool_error_rate` alongside quality judges.

The judge runs live (`AnthropicJudgeClient` forced tool call, or
`OpenAIJudgeClient` `json_schema` - both temperature 0, needing the `judge`
extra plus `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`) or offline
(`ReplayJudgeClient`), chosen by the run mode like the adapter. Each
dimension becomes a metric `<name>.<key>` plus a `<name>.overall` mean.

**Panel + images.** Replace the single `model`/`replay_path` with a
`judges:` list to run a **panel** - each judge scores independently:

```yaml
  - type: llm_judge
    name: quality
    content_ref: output.html
    dimensions: [ ... ]
    image_refs:                      # images shown to judges (live only)
      screenshot: artifacts.screenshot
    disagreement_threshold: 2        # raw-point spread that flags a case
    judges:
      - {key: claude, provider: anthropic, model: claude-sonnet-4-6, replay_path: fixtures/judge_claude.yaml}
      - {key: gpt,    provider: openai,    model: 'openai:gpt-4o',    replay_path: fixtures/judge_gpt.yaml}
```

A panel emits, on top of the per-dimension panel means and `<name>.overall`:
`<name>.<judge>.overall` (each judge's own mean, so a systematically
generous judge is visible), `<name>.disagreement` (mean inter-judge spread
in raw points), and `<name>.flagged` (1.0 when any dimension's spread
reaches `disagreement_threshold` - averaged across cases, the fraction a
human should review). `image_refs` resolve to file paths (from
`output.artifacts`) or inline `{media_type, data}`; images are sent live
only, so replay stays offline. A single judge emits none of the panel-only
metrics, so existing single-judge suites are unchanged (a panel is the natural
fit for judging rendered screenshots with a Claude+GPT pair, for instance).

Custom graders register exactly like adapters - see
`examples/quickstart/graders.py` for one of each kind (a per-case
keyword check and a whole-run distinctness check). A grader that needs to
know the run mode (live vs replay) can expose `set_mode(mode: str)`; the
runner calls it before the run starts.

## 5. Fixtures (offline / CI runs)

Replay mode swaps the configured adapter for recorded outputs, keyed by
case id and variant name - the whole pipeline then runs with no network,
keys, or deployed service:

```yaml
# fixtures/replay.yaml
refund_request:
  baseline:  {reply: 'We can refund...', intent: refund}
  candidate: {reply: 'Refund issued...', intent: refund}
billing_question:
  baseline:  {error: 'HTTP 502'}        # recorded failures work too
```

Judge fixtures are keyed by the **exact content string under judgment**, so
different variants (different text) deterministically get different scores:

```yaml
# fixtures/judge.yaml
'We can refund...':
  scores: {empathy: 4, accuracy: 5}
  rationale: Correct but a little curt.
```

## 6. The suite + thresholds (the gate policy)

Ties it all together. Paths (`dataset`, `replay_fixtures`, grader
`replay_path`) resolve relative to the suite file, so the suite runs from
any working directory:

```yaml
project: my-service          # store namespace
suite: support_reply         # suite name within the project
dataset: datasets/support_reply/v1
dataset_version: v1
mode_default: http           # 'replay' to default offline
replay_fixtures: fixtures/replay.yaml
adapter: {...}               # step 2
graders: [...]               # step 4
variants: {...}              # step 3
n_samples: 1                 # invocations per case (sampling)
concurrency: 1               # max concurrent (case, sample) invocations
retry:                       # transient-failure retry (default: no retry)
  max_attempts: 3            # total tries per invocation (1 = off)
  backoff_base: 0.5          # seconds; delay = base * 2**(attempt-1)
  backoff_max: 30.0          # per-sleep cap
  jitter: 0.1                # +/- fractional randomization

thresholds:
  win_metric: quality.overall      # the ONE headline signal
  win_higher_is_better: true
  win_min_delta: 0.02              # dead band: |delta| <= this -> neutral
  on_regression: warn              # or 'fail' to hard-gate the win metric
  variants: {baseline: baseline, candidate: candidate}   # gate defaults
  guardrails:                      # hard constraints on the CANDIDATE
    - metric: false_negative_rate
      max: 0.10                    # absolute ceiling
      must_not_increase: true      # ...and no worse than baseline
    - metric: no_pii
      min: 1.0                     # absolute floor (pass-rates: 'all passed')
    - metric: errors
      max: 0                       # any failed invocation fails the gate
```

Guardrail rules compose: `max`, `min`, `must_not_increase`,
`must_not_decrease`. A guardrail whose metric is missing on the candidate
fails closed. Pick guardrails for the failures that must never ship, and
one win metric for the improvement you're hunting; everything else is
reported informationally.

## 7. Running it

**CLI** (plug-ins first, so custom types register):

```bash
# one variant -> scorecard (optionally saved)
evalcore --plugins my_service.graders run \
    --suite my_service/suite.yaml --variant candidate --mode replay \
    --out candidate.scorecard.json --revision "$GIT_SHA"

# the CI workhorse: run baseline+candidate, compare, exit 1 on 'fail'
evalcore --plugins my_service.graders gate \
    --suite my_service/suite.yaml --mode replay \
    --export outbox.jsonl --revision "$GIT_SHA"

# re-compare two previously saved runs (accepts --out scorecards OR
# --run-out run files)
evalcore compare --suite my_service/suite.yaml \
    --baseline old.run.json --candidate new.run.json
```

`gate` picks variant names from `thresholds.variants`, falling back to
variants literally named `baseline`/`candidate`. `--revision` is an opaque
provenance id (git SHA, image digest, release label - whatever your world
uses; the engine never interprets it).

`run`, `compare`, `gate`, `sweep`, and `pairwise` take `--report markdown`
(default) or `--report html` for a standalone, self-contained report document
(a CI artifact or PR attachment), plus `--report-out FILE` to write it instead
of printing. Reporters are a registry seam like adapters and graders - register
a custom format with `evalcore.reporters.base.register` and select it by name,
e.g. `--report pdf`. Every report type is also available as a plain library
call (`evalcore.reporters` / `evalcore.report`), returning the rendered string.

**The change loop - run, change, run, compare.** To measure whether a
change (a prompt edit, a new model, a frontend PR) helped or regressed,
run the *same variant* before and after and compare the two saved runs:

```bash
evalcore ... run --variant candidate --run-out before.run.json --revision before
# ... make the change (edit the prompt, point at the PR build, swap the model) ...
evalcore ... run --variant candidate --run-out after.run.json  --revision after
evalcore compare --suite my_service/suite.yaml \
    --baseline before.run.json --candidate after.run.json      # deltas + verdict
```

The comparison's guardrails + win metric then read as "did the change
regress?" For nondeterministic targets (LLMs, browsers) raise `n_samples`
so each metric is a mean (with `stdev`) over several generations - a single
run per side makes a small delta indistinguishable from run-to-run noise,
and `win_min_delta` is your noise floor.

**Python API** - everything the CLI does is a library call; the full worked
version is `examples/quickstart/run_eval.py`:

```python
import my_service.graders  # noqa: F401  (registers custom types)
from evalcore import compare, loader, report, runner, store

suite = loader.load_suite('my_service/suite.yaml')
baseline = runner.run_suite_sync(suite, 'baseline', mode='replay',
                                 revision='abc123', created_at=now)
candidate = runner.run_suite_sync(suite, 'candidate', mode='replay',
                                  revision='abc123', created_at=now)
# (async context: `await runner.run_suite(...)` is the same call)

result = compare.compare(baseline, candidate, suite.thresholds)
print(report.render_scorecard(candidate))
print(report.render_comparison(result))

store.write_scorecard('candidate.scorecard.json', candidate)
store.write_comparison('comparison.json', result)
store.JsonlOutboxExporter('outbox.jsonl').export(candidate)

raise SystemExit(0 if result.verdict != 'fail' else 1)
```

---

# Interpreting results

## The scorecard

One scorecard per (suite × variant) run. Header first:

```
### my-service/support_reply - `candidate`
- model: `claude-sonnet-4-6`  mode: `replay`  dataset: `v1`  cases: 6x2
```

`cases: 6x2` = 6 cases × `n_samples` 2 → **12 observations** behind every
number below it. Then one row per metric; there are three families, read
differently:

**Pass-rates** (deterministic + custom per-case graders). Each output
scored 1.0 or 0.0; the scorecard shows the mean. `1.0000` = every
observation passed; `0.9167` = 11 of 12. The metric name is the grader's
`name`.

**Judge scores** (`quality.empathy`, ..., `quality.overall`). Each output
rubric-scored 1..scale by the pinned judge, normalized to 0..1, averaged
across observations. `overall` is the per-output mean of the dimensions,
then averaged. An output the judge couldn't score (errored invocation,
missing content) contributes *nothing* - it is excluded from the mean, not
counted as zero - so always read judge means alongside `errors`.

**Set-level aggregates** (classification + custom aggregate graders).
Computed once over the whole run from a confusion matrix. The
`classification` grader maps predicted/expected labels to
positive/negative via your configured label sets, with three rules: an
errored output increments `errors` and is excluded; a label resolving to
nothing is an error too; an **unlisted** label counts as *negative*, so a
stray verdict can never masquerade as a catch. Then:

| metric | formula | question it answers |
| --- | --- | --- |
| `precision` | TP/(TP+FP) | of what it flagged positive, how much really was? |
| `recall` | TP/(TP+FN) | of the real positives, how many did it catch? |
| `f1` | 2PR/(P+R) | single-number balance of the two |
| `false_negative_rate` | FN/(FN+TP) | misses, as a fraction of real positives |
| `false_positive_rate` | FP/(FP+TN) | false alarms, as a fraction of real negatives |
| `accuracy` | (TP+TN)/all | overall fraction correct - flatters on imbalanced data; never guardrail it |
| `support_positive` / `support_negative` | TP+FN / TN+FP | the denominators: how much evidence backs the rates |
| `errors` | count | invocations that failed or produced no usable label |

FNR and FPR are first-class (not just `1-recall`) because they're the
operational failure modes gates hang guardrails on: FNR = "a positive
slipped through", FPR = "a negative got blocked". They have different
denominators, so they stay honest on imbalanced datasets where `accuracy`
lies. `support_*` doubles as a drift alarm: if it changes between runs on
the same `dataset_hash`, extraction or labels broke. `errors` is a raw
count and worth a `max: 0` guardrail - errored results are excluded from
every rate, so without it a variant that crashes on its hardest cases
would look *better*.

In JSON/outbox form each metric carries `kind` (`mean` vs `aggregate`) and
`n`. Downstream tooling may re-average `mean` metrics across runs (weighted
by `n`) but must never average two `aggregate` values (the mean of two F1s
is not the combined F1).

## The comparison

```
## **PASS** - my-service/support_reply
`candidate` vs `baseline` - quality.overall neutral
```

The badge is the verdict; the tail is why. The delta table lists every
metric with baseline / candidate / delta; the row marked
**(improved | regressed | neutral)** is the configured win metric. Its call
uses the dead band: `|delta| <= win_min_delta` → **neutral** - deliberate
protection against celebrating (or reverting on) noise from small samples.

The **Guardrails** section shows each rule as `[ok]` or `[BREACH]` with the
measured value. Verdict logic, in order:

1. any guardrail breach → **FAIL** (regardless of the win metric);
2. else win metric regressed → **WARN** (or **FAIL** if
   `on_regression: fail`);
3. else → **PASS**.

`gate` (and the example driver) exit non-zero exactly on **FAIL**, so the
verdict drops straight into CI. A **PASS with neutral win** is a perfectly
good outcome - it means "no regression, no proven improvement".

## Provenance (trusting a number later)

Every scorecard (and every outbox row) carries the full reproducibility
key: `project, suite, variant, dataset_version, model_id, prompt_version,
judge_version, revision, suite_hash, dataset_hash, mode, created_at`.

The declared versions state *intent*; the engine-computed content hashes
prove it: `suite_hash` digests the raw suite file, `dataset_hash` digests
the loaded cases (order-independent, formatting-independent). Two runs
whose hashes match evaluated the same config over the same data - if a
metric moved, the *system under test* moved. If a hash changed, the eval
itself changed and the comparison is apples-to-oranges: re-baseline.
`revision` ties the run to whatever provenance scheme you use (commit,
image digest, release label). A judge model/prompt/scale change is also a
re-baseline event - bump `judge_version`; the runner lifts each judge
grader's pin (`key@version`, a panel joins them) onto `Scorecard.judge_version`
so it rides the reproducibility key, and it stays recorded on every judge
score's `detail` too.

## Per-sample results

`runner.run_suite` returns a **`RunResult`** - the scorecard plus every
per-sample `CaseResult` (the output, its `artifacts`, and its scores). The
scorecard is the aggregate; the results are the ground truth it was folded
from. Persist the whole thing with `store.write_run` (CLI: `run --run-out`)
so transcript review, human rating, and judge-agreement analysis can read
back individual generations without re-running the suite. Every run gets a
`run_id` (a UUID) that threads onto the scorecard and every store row.

With `n_samples > 1`, each `mean` metric also carries a `stdev` over its
observations, so repeat-generation spread is visible, not just the average.
Set `concurrency: N` in the suite to run invocations concurrently (the
adapter and per-case graders must then tolerate concurrent calls).

**Retries.** Live targets fail transiently - a rate limit, a 5xx, a dropped
connection. A `retry:` block (above) makes the runner re-invoke the adapter
with exponential backoff (`backoff_base * 2**(attempt-1)`, capped at
`backoff_max`, ± `jitter`) when - and only when - the adapter marks the
failure `Output.retryable`. The built-in `http` adapter flags 429/5xx/network
errors and leaves other 4xx terminal; a custom adapter sets the flag for
whatever its transient failures are. The default (`max_attempts: 1`) is a
no-op, so existing suites are unchanged. Retries hold their concurrency slot
while backing off, so a rate-limited target naturally applies backpressure.

The **LLM judge** honors the same `retry:` policy: a transient judge-client
error (a 429/5xx/timeout raised by the Anthropic/OpenAI SDK) backs off and
retries, and only a sustained failure surfaces.

**Resume.** For long live runs, pass `run --checkpoint run.ckpt`: the runner
appends each `(case, sample)` result to that JSONL file as it completes, so an
interrupted run (Ctrl-C, crash, spot-instance reclaim) leaves a valid partial
trail. Re-run with `--resume` and it reuses the recorded results and invokes
only what's missing, reusing the original `run_id`:

```bash
evalcore run --suite suite.yaml --variant candidate --checkpoint run.ckpt
# ... interrupted after 40/100 cases ...
evalcore run --suite suite.yaml --variant candidate --checkpoint run.ckpt --resume
```

The checkpoint's meta line records `suite_hash`/`dataset_hash`, so a resume
against a changed suite, dataset, or variant refuses rather than mixing
incompatible results - delete the checkpoint to start over (which `--resume`
also does implicitly when the file is absent). A checkpointed `(case, sample)`
is treated as done whether it succeeded or errored; to redo just the failures,
drop their lines from the checkpoint first.

## Human rating & judge calibration

An LLM judge is only trustworthy as a win metric once you've checked it
tracks human taste. evalcore closes that loop over the persisted runs:

```bash
# blind rating web app over one or more saved runs (repeat --run to blind
# across variants: the browser never sees which model produced an output)
evalcore rate --run cand.run.json --run base.run.json \
    --ratings ratings.jsonl --dimensions visual_design,copy_quality \
    --content-ref output.html            # or a screenshot via artifacts.*

# how well the judge agreed with the humans, per dimension
evalcore agreement --run cand.run.json --ratings ratings.jsonl \
    --dimensions visual_design,copy_quality --judge-name quality
```

`rate` serves a dependency-free localhost page: a seeded-shuffled queue and
1..scale buttons per dimension. It renders each item as **typed panels**
derived from the output - `image`/`pdf` artifacts, `html` (sandboxed iframe
with a rendered/source toggle), `json`, or `text` - so plain-text
or JSON results render with zero config, and any number of artifacts become
that many panels. `--content-ref`/`--screenshot-ref` are the common
shorthand; a repeatable `--view label:kind:ref` gives explicit control.
Sessions are **resumable** (a rater only sees items they haven't scored).
**Blinding is enforced server-side** - the queue payload carries an opaque
item id and never the run/variant/model; ratings map back to
`(run_id, case_id, sample_hash)` only on the server. Ratings land in a JSONL
file (`models.Rating`) that is the **open interchange format**: any external
tool or spreadsheet export in the same shape feeds `agreement` too.

`agreement` reports, per dimension and overall, the mean-absolute-error and
correlation between the per-case human mean and the judge's score (both on
0..1). Low MAE + high correlation is the green light to trust that judge
dimension as a win metric; a panel's `flagged` cases (see the judge panel
above) are the natural first items to route through `rate`.

**Side-by-side preference (`rank`/`preferences`).** `rate` scores each output
in isolation; `rank` is its A-vs-B analog - the human counterpart of
`pairwise`. It shows both variants' outputs for the same case as neutral
"Option 1"/"Option 2" columns and the rater picks a winner overall and per
dimension:

```bash
# blind side-by-side ranking web app over two saved runs
evalcore rank --run-a base.run.json --run-b cand.run.json \
    --preferences prefs.jsonl --dimensions visual_design,copy_quality \
    --content-ref output.html

# human A-vs-B win-rate (overall + per dimension) from the collected file
evalcore preferences --run-a base.run.json --run-b cand.run.json \
    --preferences prefs.jsonl --report html --report-out prefs.html
```

Left/right sides are **shuffled per rater and un-blinded server-side**, so a
stored pick is always in *variant* terms (`variant_a`/`variant_b`) regardless
of which side it was shown on - position bias counterbalances across raters
exactly like `pairwise`'s order swap. Sessions are resumable and picks land
in a JSONL file (`models.Preference`), the **open A-vs-B interchange format**.
`preferences` aggregates it (ties count half); to check the LLM pairwise judge
against the human panel, pass the same file to `pairwise --preferences
prefs.jsonl`, which appends a per-case human-vs-judge agreement table - the
head-to-head calibration gate.

**Live judges on recorded data.** `--mode` drives the adapter; `--judge-mode`
drives the graders independently (default: same as `--mode`). So
`--mode replay --judge-mode live` re-scores recorded outputs with the live
judge panel - no regeneration - which is how you iterate on a rubric or
re-record judge fixtures cheaply.

## Sweeps & pairwise win-rate

`compare`/`gate` answer "candidate vs baseline". Two commands go beyond that:

```bash
# N-way: run every variant (or a subset) and rank them by the win metric
evalcore sweep --suite suite.yaml --mode replay          # or --variants a,b,c

# A-vs-B: a judge picks a winner per case -> A's win-rate
evalcore pairwise --suite suite.yaml --a baseline --b candidate --mode replay
```

`sweep` prints a ranked leaderboard plus a full metric × variant matrix (a
model × prompt-version grid is just several named variants) - reusing the
per-variant runner unchanged; it's pure orchestration + tabulation.

`pairwise` is the sharper subjective signal: instead of scoring each output
in isolation, a judge is shown **both** variants' outputs for the same case
and picks a winner, and evalcore reports A's win-rate (ties count half).
Order is **counterbalanced** - each pair is judged both ways and a pick that
flips when you swap the order collapses to a tie, so position bias can't
manufacture a winner. Configure it under `thresholds.pairwise` (`content_ref`,
`model`/`replay_path`, optional `rubric`/`context_refs`); it runs live
(Anthropic or OpenAI) or offline against recorded, order-independent
judgments, like the rubric judge. `examples/quickstart/suite.yaml` has an
offline pairwise config.

## The outbox

`JsonlOutboxExporter` flattens results into JSONL for a column-store shipper
(e.g. ClickHouse) to drain. `export_scores(run)` writes one feed at
`(run, case, sample, grader, metric)` grain - one row per score, carrying the
scored value, the invocation it came from, the gate verdict, and the full
reproducibility key (incl. `run_id`), so a multi-tenant trend table can
filter/group on any dimension without joins. Nothing derived is written: a
run's scorecard is a read-time aggregation over the same rows, so it cannot
disagree with the scores behind it. A measurement that does not exist is
`null`, not `0` - a real `0.0` is a meaningful score, and `avg`/`sum`/`count`
skip nulls, so no query has to remember a filter; `passed` is the tri-state
string `'true'|'false'|'null'`, since a judge has no pass line by design and
that is a third state rather than a missing value. Row keys are column names
rather than model attribute names, so a JSONEachRow-style feed lands in a flat
table without a mapping layer. A store that forbids nullable columns fills
those nulls in at ingest, on its side of the seam. Swap the exporter for a real
database client without touching the runner or any consumer.

---

## The two extension seams (recap)

```python
class TargetAdapter(typing.Protocol):       # how to call the system under test
    async def invoke(self, case, variant) -> Output: ...

class Grader(typing.Protocol):              # per-case (averaged)
    def grade(self, case, output) -> list[Score]: ...
class AggregateGrader(typing.Protocol):     # whole-run (P/R/F1, win-rate)
    def aggregate(self, results) -> list[Score]: ...
```

Built-ins: `http` + `replay` adapters; `classification`, `max_chars`,
`regex_absent`, `regex_present`, `non_empty`, and `llm_judge` graders; and
`markdown` + `html` reporters (single scorecards and comparative
comparisons; pick one with `--report`). Register more with
`evalcore.adapters.base.register` / `evalcore.graders.base.register` /
`evalcore.reporters.base.register` and load them with `--plugins your.module`
(CLI) or a plain import (Python API). If onboarding a new consumer ever
requires touching `src/evalcore/`, that's an abstraction leak - fix the engine
seam, don't fork it.

## Layout

```
src/evalcore/
  models.py      Case, Variant, Output, Score, Scorecard, Comparison (opaque-blob based)
  refs.py        $ref resolution (the only thing that opens a consumer's blobs)
  loader.py      suite + dataset loading, content hashes (YAML/JSON; suite-relative paths)
  adapters/      target seam - http, replay
  graders/       grader seam - deterministic, classification, llm_judge
  runner.py      suite x variant -> RunResult (scorecard + per-sample results)
  compare.py     candidate vs baseline -> guardrails + win -> verdict
  sweep.py       run N variants -> ranked leaderboard (metric x variant)
  pairwise.py    A-vs-B judging -> counterbalanced win-rate
  store.py       scorecard/run JSON + column-store outbox + ratings/prefs JSONL
  rating.py      blind rating + side-by-side ranking web apps + agreement
  reporters/     report seam - markdown / html (scorecard, comparison, ...)
  report.py      Markdown renderers (scorecard / comparison / sweep / pairwise / ...)
  cli.py         run - compare - gate - sweep - pairwise - rate - rank - report - ...
tests/               engine unit tests
examples/quickstart  a runnable consumer that doubles as an implementation test
docs/design.md       the design overview
```

## Status

2.0. The public API and the outbox row shape are stable; a breaking change to
either means a 3.0. Built: deterministic + classification + **LLM-judge**
(rubric scoring; single judge or a Claude/GPT **panel** with per-dimension
means, per-judge overalls, inter-judge disagreement flagging, and
image/screenshot inputs) graders, http/replay/browser adapters, runner
(N-sampling, optional concurrency, per-sample `RunResult` + `run_id` +
variance), comparison/gate, JSON + run + outbox store (one score-grain feed),
**N-way sweeps
+ counterbalanced pairwise A-vs-B win-rate** (`sweep`/`pairwise`), **blind
human-rating + side-by-side ranking web apps** with judge↔human agreement and
human-vs-judge pairwise agreement (`rate`/`agreement`, `rank`/`preferences`),
**pluggable reporters** (`markdown`/`html`, `--report`), provenance
(`revision` + suite/dataset content hashes), typed package (`py.typed`).

Known gaps / next:

- A real column-store client (the row shape + JSONL outbox exporter stand
  in, in `store.py`).
- Cost/token capture: `Output.tokens`/`cost` fields exist but nothing
  populates them (an adapter must fill them from whatever usage its target
  reports), and neither is exported to the results store - usage accounting
  is captured outside it.
- Run robustness lands: retry with exponential backoff on transient failures
  for both the adapter (suite `retry:` + `Output.retryable`) and the LLM judge
  client, plus idempotent mid-run resume from a `run --checkpoint`.

## Releasing

Releases publish to PyPI as **`evalcore`** via
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) - no API
tokens are stored. One-time setup on PyPI: add a *pending publisher* for project
`evalcore` pointing at owner `scottpmiller`, repo `evalcore`, workflow
`publish.yml`, environment `pypi`. Then to cut a release: bump `version` in
`pyproject.toml`, tag it, and publish a GitHub Release - `.github/workflows/publish.yml`
builds the sdist + wheel and uploads them. (Point it at TestPyPI first for a dry run.)
