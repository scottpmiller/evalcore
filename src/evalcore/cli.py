"""evalcore command-line interface.

    python -m evalcore.cli run  --suite S --variant V [--mode replay] [--out F]
                               [--checkpoint F [--resume]]
    python -m evalcore.cli gate --suite S [--baseline B --candidate C]
                                [--mode replay] [--export OUTBOX]
    python -m evalcore.cli compare --suite S --baseline F1 --candidate F2
    python -m evalcore.cli sweep --suite S [--variants a,b,c] [--mode replay]
    python -m evalcore.cli pairwise --suite S --a V1 --b V2 [--mode replay]
                                [--preferences P]
    python -m evalcore.cli rate  --run R --ratings F --dimensions a,b
    python -m evalcore.cli rank  --run-a A --run-b B --preferences P
                                --dimensions a,b
    python -m evalcore.cli agreement --run R --ratings F --dimensions a,b
    python -m evalcore.cli preferences --run-a A --run-b B --preferences P
                                [--report html] [--report-out F]
    python -m evalcore.cli report --run R [--ratings F --dimensions a,b]
                                [--report html] [--report-out F]

``gate`` is the CI workhorse: it runs the baseline and candidate variants and
compares them in one shot, exiting non-zero when the verdict is ``fail``.
``--plugins mod1,mod2`` imports consumer modules so their custom graders /
adapters register before the run.
"""

import argparse
import asyncio
import datetime
import importlib
import os
import pathlib
import sys

from evalcore import compare as compare_mod
from evalcore import loader, rating, report, reporters, runner, store
from evalcore import pairwise as pairwise_mod
from evalcore import sweep as sweep_mod


def _load_plugins(spec: str | None) -> None:
    if not spec:
        return
    # Consumers run the CLI from their repo root; the console script
    # (unlike `python -m`) does not put the cwd on sys.path, so add it
    # or `--plugins my_pkg.graders` could never import.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    for name in filter(None, spec.split(',')):
        importlib.import_module(name.strip())


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _reporter(args: argparse.Namespace) -> reporters.Reporter:
    """Build the reporter chosen by ``--report`` (default markdown)."""
    return reporters.build_reporter(
        getattr(args, 'report', None) or 'markdown'
    )


def _emit_report(args: argparse.Namespace, body: str) -> None:
    """Write the rendered report to ``--report-out``, else print it."""
    out = getattr(args, 'report_out', None)
    if out:
        pathlib.Path(out).write_text(body, encoding='utf-8')
        print(f'wrote report -> {out}', file=sys.stderr)
    else:
        print(body)


def _thresholds_variants(suite: loader.SuiteConfig) -> tuple[str, str]:
    variants = suite.thresholds.get('variants', {})
    baseline = variants.get('baseline')
    candidate = variants.get('candidate')
    names = list(suite.variants)
    if not baseline:
        baseline = 'baseline' if 'baseline' in suite.variants else names[0]
    if not candidate:
        candidate = 'candidate' if 'candidate' in suite.variants else names[-1]
    return baseline, candidate


def _cmd_run(args: argparse.Namespace) -> int:
    _load_plugins(args.plugins)
    suite = loader.load_suite(args.suite)
    run = runner.run_suite_sync(
        suite,
        args.variant,
        mode=args.mode,
        grader_mode=args.judge_mode,
        revision=args.revision,
        created_at=_now(),
        checkpoint=args.checkpoint,
        resume=args.resume,
    )
    rep = _reporter(args)
    _emit_report(args, reporters.render_run(rep, run))
    if args.out:
        store.write_scorecard(args.out, run.scorecard)
        print(f'\nwrote {args.out}', file=sys.stderr)
    if args.run_out:
        store.write_run(args.run_out, run)
        print(
            f'wrote {args.run_out} ({len(run.results)} results)',
            file=sys.stderr,
        )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    suite = loader.load_suite(args.suite)
    # Accept scorecard files (run --out) or full-run files (run --run-out).
    baseline = store.load_scorecard(args.baseline)
    candidate = store.load_scorecard(args.candidate)
    result = compare_mod.compare(baseline, candidate, suite.thresholds)
    rep = _reporter(args)
    _emit_report(args, reporters.wrap_document(rep, rep.comparison(result)))
    return 0 if result.verdict != 'fail' else 1


def _cmd_gate(args: argparse.Namespace) -> int:
    _load_plugins(args.plugins)
    suite = loader.load_suite(args.suite)
    baseline_name, candidate_name = _thresholds_variants(suite)
    if args.baseline:
        baseline_name = args.baseline
    if args.candidate:
        candidate_name = args.candidate

    baseline = runner.run_suite_sync(
        suite,
        baseline_name,
        mode=args.mode,
        grader_mode=args.judge_mode,
        revision=args.revision,
        created_at=_now(),
    )
    candidate = runner.run_suite_sync(
        suite,
        candidate_name,
        mode=args.mode,
        grader_mode=args.judge_mode,
        revision=args.revision,
        created_at=_now(),
    )
    result = compare_mod.compare(
        baseline.scorecard, candidate.scorecard, suite.thresholds
    )

    rep = _reporter(args)
    body = (
        rep.scorecard(baseline.scorecard)
        + '\n\n'
        + rep.scorecard(candidate.scorecard)
        + '\n\n'
        + rep.comparison(result)
    )
    _emit_report(args, reporters.wrap_document(rep, body))

    # One feed now, so --export and --export-scores name the same rows;
    # --export-scores is kept as an alias for existing invocations.
    outboxes = dict.fromkeys(
        path for path in (args.export, args.export_scores) if path
    )
    if outboxes:
        types, scales = store.grader_lookups(suite.graders)
    for outbox in outboxes:
        exporter = store.JsonlOutboxExporter(outbox)
        # The baseline half of a pair is not itself gated, so it carries no
        # comparison; the candidate carries the verdict and the guardrails.
        rows = exporter.export_scores(
            baseline, grader_types=types, judge_scales=scales
        )
        rows += exporter.export_scores(
            candidate,
            result,
            baseline_run_id=baseline.run_id,
            grader_types=types,
            judge_scales=scales,
        )
        print(f'\nexported {rows} store rows -> {outbox}', file=sys.stderr)

    return 0 if result.verdict != 'fail' else 1


def _cmd_sweep(args: argparse.Namespace) -> int:
    _load_plugins(args.plugins)
    suite = loader.load_suite(args.suite)
    names = (
        [v.strip() for v in args.variants.split(',') if v.strip()]
        if args.variants
        else None
    )
    runs = asyncio.run(
        sweep_mod.run_sweep(
            suite,
            names,
            mode=args.mode,
            grader_mode=args.judge_mode,
            revision=args.revision,
            created_at=_now(),
        )
    )
    result = sweep_mod.summarize_sweep(runs, suite.thresholds)
    rep = _reporter(args)
    _emit_report(
        args, reporters.wrap_document(rep, reporters.render_sweep(rep, result))
    )
    if args.export:
        types, scales = store.grader_lookups(suite.graders)
        exporter = store.JsonlOutboxExporter(args.export)
        rows = sum(
            exporter.export_scores(
                run, grader_types=types, judge_scales=scales
            )
            for run in runs
        )
        print(
            f'\nexported {rows} store rows -> {args.export}', file=sys.stderr
        )
    return 0


def _cmd_pairwise(args: argparse.Namespace) -> int:
    _load_plugins(args.plugins)
    suite = loader.load_suite(args.suite)
    config = dict(suite.thresholds.get('pairwise') or {})
    content_ref = args.content_ref or config.get('content_ref')
    if not content_ref:
        raise SystemExit('pairwise needs --content-ref or thresholds.pairwise')
    mode = args.mode or suite.mode_default
    run_a = runner.run_suite_sync(suite, args.a, mode=mode, created_at=_now())
    run_b = runner.run_suite_sync(suite, args.b, mode=mode, created_at=_now())
    client = pairwise_mod.build_pairwise_client(mode, config)
    result = asyncio.run(
        pairwise_mod.judge_pairwise(
            run_a,
            run_b,
            content_ref=content_ref,
            client=client,
            context_refs=config.get('context_refs'),
            rubric=config.get('rubric'),
            judge_version=config.get('judge_version', 'v1'),
        )
    )
    rep = _reporter(args)
    body = reporters.render_pairwise(rep, result)
    if args.preferences:
        # Fold in how the human panel agreed with this judge, per case.
        prefs = store.read_preferences(args.preferences)
        agreement = rating.compute_pairwise_agreement(prefs, result)
        body += '\n\n' + reporters.render_pairwise_agreement(rep, agreement)
    _emit_report(args, reporters.wrap_document(rep, body))
    return 0


def _parse_views(specs: list[str] | None) -> list[dict] | None:
    """Parse ``--view`` specs of the form ``label:kind:ref`` or ``label:ref``.

    ``kind`` (image/pdf/html/json/text) is optional; when omitted it is
    inferred from the resolved value. Refs use dots, never colons, so a
    plain split is unambiguous.
    """
    if not specs:
        return None
    views: list[dict] = []
    for spec in specs:
        parts = spec.split(':')
        if len(parts) == 3:
            views.append(
                {'label': parts[0], 'kind': parts[1], 'ref': parts[2]}
            )
        elif len(parts) == 2:
            views.append({'label': parts[0], 'ref': parts[1]})
        else:
            raise SystemExit(
                f'bad --view {spec!r}; use label:kind:ref or label:ref'
            )
    return views


def _cmd_rate(args: argparse.Namespace) -> int:
    runs = [store.read_run(path) for path in args.run]
    rating.serve(
        runs,
        args.ratings,
        [d.strip() for d in args.dimensions.split(',') if d.strip()],
        scale=args.scale,
        content_ref=args.content_ref,
        screenshot_ref=args.screenshot_ref,
        views=_parse_views(args.view),
        port=args.port,
        open_browser=not args.no_open,
    )
    return 0


def _cmd_rank(args: argparse.Namespace) -> int:
    run_a = store.read_run(args.run_a)
    run_b = store.read_run(args.run_b)
    rating.serve_rank(
        run_a,
        run_b,
        args.preferences,
        [d.strip() for d in args.dimensions.split(',') if d.strip()],
        content_ref=args.content_ref,
        screenshot_ref=args.screenshot_ref,
        views=_parse_views(args.view),
        port=args.port,
        open_browser=not args.no_open,
    )
    return 0


def _cmd_preferences(args: argparse.Namespace) -> int:
    run_a = store.read_run(args.run_a)
    run_b = store.read_run(args.run_b)
    prefs = store.read_preferences(args.preferences)
    dims = (
        [d.strip() for d in args.dimensions.split(',') if d.strip()]
        if args.dimensions
        else None
    )
    result = rating.aggregate_preferences(run_a, run_b, prefs, dims)
    rep = _reporter(args)
    _emit_report(
        args,
        reporters.wrap_document(
            rep, reporters.render_preferences(rep, result)
        ),
    )
    return 0


def _cmd_agreement(args: argparse.Namespace) -> int:
    run = store.read_run(args.run)
    ratings = store.read_ratings(args.ratings)
    result = rating.compute_agreement(
        run,
        ratings,
        [d.strip() for d in args.dimensions.split(',') if d.strip()],
        judge_name=args.judge_name,
        scale=args.scale,
    )
    print(report.render_agreement(result))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    run = store.read_run(args.run)
    rep = _reporter(args)
    body = reporters.run_body(rep, run)
    if args.ratings:
        if not args.dimensions:
            raise SystemExit('report --ratings also needs --dimensions')
        ratings = store.read_ratings(args.ratings)
        result = rating.compute_agreement(
            run,
            ratings,
            [d.strip() for d in args.dimensions.split(',') if d.strip()],
            judge_name=args.judge_name,
            scale=args.scale,
        )
        body += '\n\n' + reporters.render_agreement(rep, result)
    _emit_report(args, reporters.wrap_document(rep, body))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser (exposed for tests)."""
    parser = argparse.ArgumentParser(prog='evalcore')
    parser.add_argument(
        '--plugins',
        help='comma-separated modules to import (register custom graders)',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    run = sub.add_parser('run', help='run one variant of a suite')
    run.add_argument('--suite', required=True)
    run.add_argument('--variant', required=True)
    run.add_argument('--mode')
    run.add_argument(
        '--judge-mode',
        dest='judge_mode',
        help='grader mode when it differs from --mode '
        '(e.g. live judges on replayed data)',
    )
    run.add_argument('--out', help='write the scorecard as JSON')
    run.add_argument(
        '--run-out',
        dest='run_out',
        help='write the full run (scorecard + per-sample results) as JSON',
    )
    run.add_argument(
        '--revision',
        help='opaque provenance id stamped on the scorecard '
        '(git SHA, image digest, release label, ...)',
    )
    run.add_argument(
        '--report',
        default='markdown',
        help='report format: markdown (default), html, or a '
        'registered custom type',
    )
    run.add_argument(
        '--report-out',
        dest='report_out',
        help='write the rendered report to a file instead of stdout',
    )
    run.add_argument(
        '--checkpoint',
        help='JSONL checkpoint appended as each (case, sample) completes, so '
        'an interrupted run can be --resumed',
    )
    run.add_argument(
        '--resume',
        action='store_true',
        help='reuse completed results from --checkpoint (same suite/dataset/'
        'variant) and run only what is missing',
    )
    run.set_defaults(func=_cmd_run)

    cmp_ = sub.add_parser('compare', help='compare two saved scorecards')
    cmp_.add_argument('--suite', required=True)
    cmp_.add_argument('--baseline', required=True)
    cmp_.add_argument('--candidate', required=True)
    cmp_.add_argument(
        '--report',
        default='markdown',
        help='report format: markdown (default), html, or a '
        'registered custom type',
    )
    cmp_.add_argument(
        '--report-out',
        dest='report_out',
        help='write the rendered report to a file instead of stdout',
    )
    cmp_.set_defaults(func=_cmd_compare)

    gate = sub.add_parser('gate', help='run baseline+candidate and compare')
    gate.add_argument('--suite', required=True)
    gate.add_argument('--baseline')
    gate.add_argument('--candidate')
    gate.add_argument('--mode')
    gate.add_argument(
        '--judge-mode',
        dest='judge_mode',
        help='grader mode when it differs from --mode '
        '(e.g. live judges on replayed data)',
    )
    gate.add_argument(
        '--export', help='append results-store rows to a JSONL outbox'
    )
    gate.add_argument(
        '--export-scores',
        dest='export_scores',
        help='alias for --export, kept for existing invocations',
    )
    gate.add_argument(
        '--revision', help='opaque provenance id stamped on both scorecards'
    )
    gate.add_argument(
        '--report',
        default='markdown',
        help='report format: markdown (default), html, or a '
        'registered custom type',
    )
    gate.add_argument(
        '--report-out',
        dest='report_out',
        help='write the rendered report to a file instead of stdout',
    )
    gate.set_defaults(func=_cmd_gate)

    sweep = sub.add_parser(
        'sweep', help='run many variants and rank them (a leaderboard)'
    )
    sweep.add_argument('--suite', required=True)
    sweep.add_argument(
        '--variants', help='comma-separated subset (default: all)'
    )
    sweep.add_argument('--mode')
    sweep.add_argument(
        '--judge-mode',
        dest='judge_mode',
        help='grader mode when it differs from --mode '
        '(e.g. live judges on replayed data)',
    )
    sweep.add_argument(
        '--export', help='append results-store rows to a JSONL outbox'
    )
    sweep.add_argument('--revision')
    sweep.add_argument(
        '--report',
        default='markdown',
        help='report format: markdown (default), html, or a '
        'registered custom type',
    )
    sweep.add_argument(
        '--report-out',
        dest='report_out',
        help='write the rendered report to a file instead of stdout',
    )
    sweep.set_defaults(func=_cmd_sweep)

    pw = sub.add_parser(
        'pairwise', help='A-vs-B win-rate between two variants'
    )
    pw.add_argument('--suite', required=True)
    pw.add_argument('--a', required=True, help='variant A name')
    pw.add_argument('--b', required=True, help='variant B name')
    pw.add_argument('--mode')
    pw.add_argument(
        '--content-ref',
        dest='content_ref',
        help='ref to the text to compare (else thresholds.pairwise)',
    )
    pw.add_argument(
        '--preferences',
        help='JSONL human preferences (from `rank`) to append a '
        'human-panel vs judge agreement section',
    )
    pw.add_argument(
        '--report',
        default='markdown',
        help='report format: markdown (default), html, or a '
        'registered custom type',
    )
    pw.add_argument(
        '--report-out',
        dest='report_out',
        help='write the rendered report to a file instead of stdout',
    )
    pw.set_defaults(func=_cmd_pairwise)

    rate = sub.add_parser(
        'rate', help='blind human-rating web app over saved run(s)'
    )
    rate.add_argument(
        '--run',
        action='append',
        required=True,
        help='a run JSON from `run --run-out` (repeat to blind across runs)',
    )
    rate.add_argument('--ratings', required=True, help='JSONL ratings output')
    rate.add_argument(
        '--dimensions', required=True, help='comma-separated rubric keys'
    )
    rate.add_argument('--scale', type=int, default=5)
    rate.add_argument(
        '--content-ref',
        dest='content_ref',
        help='ref to HTML/text to render (e.g. output.html)',
    )
    rate.add_argument(
        '--screenshot-ref',
        dest='screenshot_ref',
        default='artifacts.screenshot',
        help='ref to a screenshot artifact path',
    )
    rate.add_argument(
        '--view',
        action='append',
        help='explicit panel as label:kind:ref (kind: image|pdf|html|json|'
        'text; optional). Repeatable; overrides --content-ref/--screenshot-'
        'ref. Omit all to auto-derive from artifacts + output fields.',
    )
    rate.add_argument('--port', type=int, default=8900)
    rate.add_argument('--no-open', action='store_true', dest='no_open')
    rate.set_defaults(func=_cmd_rate)

    rank = sub.add_parser(
        'rank',
        help='blind side-by-side A-vs-B ranking web app (human win-rate)',
    )
    rank.add_argument(
        '--run-a', dest='run_a', required=True, help='variant A run JSON'
    )
    rank.add_argument(
        '--run-b', dest='run_b', required=True, help='variant B run JSON'
    )
    rank.add_argument(
        '--preferences', required=True, help='JSONL preferences output'
    )
    rank.add_argument(
        '--dimensions',
        required=True,
        help='comma-separated rubric keys picked per pair (plus overall)',
    )
    rank.add_argument(
        '--content-ref',
        dest='content_ref',
        help='ref to HTML/text to render (e.g. output.html)',
    )
    rank.add_argument(
        '--screenshot-ref',
        dest='screenshot_ref',
        default='artifacts.screenshot',
        help='ref to a screenshot artifact path',
    )
    rank.add_argument(
        '--view',
        action='append',
        help='explicit panel as label:kind:ref (see `rate`); repeatable',
    )
    rank.add_argument('--port', type=int, default=8901)
    rank.add_argument('--no-open', action='store_true', dest='no_open')
    rank.set_defaults(func=_cmd_rank)

    prefs = sub.add_parser(
        'preferences',
        help='human A-vs-B win-rate from saved runs + a preferences file',
    )
    prefs.add_argument(
        '--run-a', dest='run_a', required=True, help='variant A run JSON'
    )
    prefs.add_argument(
        '--run-b', dest='run_b', required=True, help='variant B run JSON'
    )
    prefs.add_argument(
        '--preferences', required=True, help='a JSONL preferences file'
    )
    prefs.add_argument(
        '--dimensions',
        help='comma-separated rubric keys (default: all seen in the file)',
    )
    prefs.add_argument(
        '--report',
        default='markdown',
        help='report format: markdown (default), html, or a '
        'registered custom type',
    )
    prefs.add_argument(
        '--report-out',
        dest='report_out',
        help='write the rendered report to a file instead of stdout',
    )
    prefs.set_defaults(func=_cmd_preferences)

    rpt = sub.add_parser(
        'report',
        help='render a report from a saved run (no re-run); optionally '
        'fold in human ratings as a judge<->human agreement section',
    )
    rpt.add_argument(
        '--run', required=True, help='a run JSON from `run --run-out`'
    )
    rpt.add_argument(
        '--ratings',
        help='JSONL ratings to append a judge<->human agreement section',
    )
    rpt.add_argument(
        '--dimensions',
        help='comma-separated rubric keys (required with --ratings)',
    )
    rpt.add_argument('--judge-name', dest='judge_name', default='quality')
    rpt.add_argument('--scale', type=int, default=5)
    rpt.add_argument(
        '--report',
        default='markdown',
        help='report format: markdown (default), html, or a '
        'registered custom type',
    )
    rpt.add_argument(
        '--report-out',
        dest='report_out',
        help='write the rendered report to a file instead of stdout',
    )
    rpt.set_defaults(func=_cmd_report)

    agree = sub.add_parser(
        'agreement', help='judge<->human agreement over a run + ratings'
    )
    agree.add_argument('--run', required=True, help='a run JSON')
    agree.add_argument('--ratings', required=True, help='a JSONL ratings file')
    agree.add_argument(
        '--dimensions', required=True, help='comma-separated rubric keys'
    )
    agree.add_argument('--judge-name', dest='judge_name', default='quality')
    agree.add_argument('--scale', type=int, default=5)
    agree.set_defaults(func=_cmd_agreement)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
