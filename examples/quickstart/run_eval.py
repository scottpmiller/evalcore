"""Drive the quickstart suite through evalcore's Python API - no CLI involved.

Everything ``evalcore gate`` does, as plain library calls:

1. load the suite + dataset (``loader``);
2. run baseline and candidate (``runner``) with ``revision``/``created_at``;
3. render Markdown scorecards (``report``);
4. compare against the suite thresholds -> gate verdict (``compare``);
5. persist scorecards + comparison as JSON, prove the round-trip (``store``);
6. flatten to store rows and append the JSONL outbox (``store``);
7. return gate semantics as the exit code.

Run offline (recorded fixtures, no network, no keys):

    uv run python -m examples.quickstart.run_eval
    just example-api
"""

import argparse
import asyncio
import datetime
import os
import pathlib
import sys

from evalcore import compare, loader, report, runner, store

HERE = pathlib.Path(__file__).resolve().parent


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def main(argv: list[str] | None = None) -> int:
    """Run the whole pipeline; return a gate-style exit code."""
    parser = argparse.ArgumentParser(prog='quickstart-eval')
    parser.add_argument('--mode', default='replay')
    parser.add_argument('--out', default=str(HERE / 'results'))
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load the suite; dataset/fixture paths resolve against the suite file,
    # so this works from any working directory.
    suite = loader.load_suite(HERE / 'suite.yaml')
    cases = loader.load_cases(suite.dataset)
    print(
        f'suite {suite.project}/{suite.suite}: {len(cases)} cases '
        f'x {suite.n_samples} samples, mode={args.mode}, '
        f'variants={", ".join(suite.variants)}'
    )

    # 2. Run both variants. The async API and the sync wrapper are the same
    # call; both are shown here on purpose. `revision` is an opaque provenance
    # id - a git SHA, image digest, release label, whatever your world uses.
    revision = os.environ.get('EVAL_REVISION')
    baseline_run = runner.run_suite_sync(
        suite, 'baseline', mode=args.mode, revision=revision, created_at=_now()
    )
    candidate_run = asyncio.run(
        runner.run_suite(
            suite,
            'candidate',
            mode=args.mode,
            revision=revision,
            created_at=_now(),
        )
    )
    baseline, candidate = baseline_run.scorecard, candidate_run.scorecard

    # 3. Markdown scorecards - what a CI job pastes into a PR comment.
    print('\n' + report.render_scorecard(baseline))
    print('\n' + report.render_scorecard(candidate))

    # 4. Candidate vs baseline under the suite thresholds -> verdict.
    result = compare.compare(baseline, candidate, suite.thresholds)
    print('\n' + report.render_comparison(result))

    # 5. Persist everything and prove the JSON round-trips are lossless.
    baseline_path = out / 'baseline.scorecard.json'
    candidate_path = out / 'candidate.scorecard.json'
    store.write_scorecard(baseline_path, baseline)
    store.write_scorecard(candidate_path, candidate)
    store.write_run(out / 'candidate.run.json', candidate_run)
    store.write_comparison(out / 'comparison.json', result)
    if store.read_scorecard(baseline_path) != baseline:
        raise RuntimeError('scorecard JSON round-trip mismatch')
    if store.read_run(out / 'candidate.run.json') != candidate_run:
        raise RuntimeError('run JSON round-trip mismatch')

    # 6. Flatten to self-describing rows and append the outbox a column-store
    # shipper would drain. One grain only - per-sample score rows. A run's
    # scorecard is a read-time aggregation over these, so exporting it too
    # would store something derived that could disagree with them.
    exporter = store.JsonlOutboxExporter(out / 'scores.jsonl')
    n_scores = exporter.export_scores(baseline_run) + exporter.export_scores(
        candidate_run
    )
    print(f'\nscores: {n_scores} per-sample rows -> {exporter.outbox_path}')

    # 7. Gate semantics: non-zero exit on a failing verdict.
    print(f'\ngate verdict: {result.verdict} ({result.summary})')
    return 0 if result.verdict != 'fail' else 1


if __name__ == '__main__':
    sys.exit(main())
