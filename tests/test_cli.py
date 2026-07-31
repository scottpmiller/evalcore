"""End-to-end CLI tests, offline (replay mode).

One synthetic suite drives the whole pipeline through the CLI: loader,
runner, replay adapter, deterministic + classification graders, compare,
gate, sweep, pairwise, report rendering, and the store exporters.
"""

import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import yaml

from evalcore import cli, models, rating, store

SUITE = {
    'project': 'demo',
    'suite': 's',
    'dataset': 'data',
    'dataset_version': 'v1',
    'mode_default': 'replay',
    'replay_fixtures': 'fixtures.yaml',
    # A real (live-only) adapter spec; replay mode never builds it.
    'adapter': {
        'type': 'http',
        'base_url': 'http://x',
        'path': '/y',
        'body': {},
        'extract': {},
    },
    'graders': [
        {'type': 'non_empty', 'name': 'has_text', 'field': 'output.text'},
        {
            'type': 'max_chars',
            'name': 'len_ok',
            'field': 'output.text',
            'maximum': 100,
        },
        {
            'type': 'classification',
            'predicted_ref': 'output.label',
            'expected_ref': 'expected.label',
            'positive_labels': ['bad'],
            'negative_labels': ['good'],
        },
    ],
    'variants': {'baseline': {}, 'candidate': {'model': 'm2'}},
    'n_samples': 1,
    'thresholds': {
        'win_metric': 'f1',
        'win_higher_is_better': True,
        'variants': {'baseline': 'baseline', 'candidate': 'candidate'},
        'guardrails': [{'metric': 'false_negative_rate', 'max': 0.5}],
        'pairwise': {
            'content_ref': 'output.text',
            'replay_path': 'pairwise.yaml',
        },
    },
}

FIXTURES = {
    'c1': {
        'baseline': {'text': 'hello world', 'label': 'good'},
        'candidate': {'text': 'hi', 'label': 'good'},
    },
    'c2': {
        # baseline misses (says good, truth is bad) -> a false negative
        'baseline': {'text': 'ok', 'label': 'good'},
        'candidate': {'text': 'bad output here', 'label': 'bad'},
    },
}

PAIRWISE = [
    {'a': 'hello world', 'b': 'hi', 'winner': 'hello world'},
    {'a': 'ok', 'b': 'bad output here', 'winner': 'bad output here'},
]

CASES = {
    'c1': {'input': {'q': 'hi'}, 'expected': {'label': 'good'}},
    'c2': {'input': {'q': 'yo'}, 'expected': {'label': 'bad'}},
}


def _write_suite(root: pathlib.Path) -> pathlib.Path:
    (root / 'data' / 'cases').mkdir(parents=True)
    for cid, body in CASES.items():
        (root / 'data' / 'cases' / f'{cid}.yaml').write_text(
            yaml.safe_dump(body)
        )
    (root / 'fixtures.yaml').write_text(yaml.safe_dump(FIXTURES))
    (root / 'pairwise.yaml').write_text(yaml.safe_dump(PAIRWISE))
    suite_path = root / 'suite.yaml'
    suite_path.write_text(yaml.safe_dump(SUITE))
    return suite_path


def _run_cli(argv: list[str]) -> tuple[int, str]:
    """Invoke the CLI, capturing stdout; return (exit_code, stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


class CliPipelineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.suite = str(_write_suite(self.root))

    def tearDown(self):
        self._tmp.cleanup()

    def test_run_writes_scorecard_and_run(self):
        out = str(self.root / 'sc.json')
        run_out = str(self.root / 'run.json')
        code, text = _run_cli(
            [
                '--plugins',
                'evalcore.models',
                'run',
                '--suite',
                self.suite,
                '--variant',
                'candidate',
                '--mode',
                'replay',
                '--revision',
                'abc123',
                '--out',
                out,
                '--run-out',
                run_out,
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn('demo/s', text)
        card = store.read_scorecard(out)
        self.assertEqual(card.revision, 'abc123')
        self.assertEqual(card.metrics['f1'].value, 1.0)
        run = store.read_run(run_out)
        self.assertEqual(len(run.results), 2)

    def test_run_report_html_to_file(self):
        report_path = str(self.root / 'report.html')
        code, text = _run_cli(
            [
                'run',
                '--suite',
                self.suite,
                '--variant',
                'candidate',
                '--mode',
                'replay',
                '--report',
                'html',
                '--report-out',
                report_path,
            ]
        )
        self.assertEqual(code, 0)
        html = pathlib.Path(report_path).read_text()
        self.assertTrue(html.startswith('<!doctype html'))
        self.assertIn('<table>', html)
        # report went to the file, not stdout
        self.assertNotIn('<!doctype', text)
        self.assertIn('wrote report', text)

    def test_run_unknown_variant_raises(self):
        with self.assertRaises(KeyError):
            _run_cli(
                [
                    'run',
                    '--suite',
                    self.suite,
                    '--variant',
                    'nope',
                    '--mode',
                    'replay',
                ]
            )

    def test_gate_passes_and_exports_outbox(self):
        outbox = str(self.root / 'outbox.jsonl')
        scores = str(self.root / 'scores.jsonl')
        code, text = _run_cli(
            [
                'gate',
                '--suite',
                self.suite,
                '--mode',
                'replay',
                '--export',
                outbox,
                '--export-scores',
                scores,
                '--revision',
                'r1',
            ]
        )
        self.assertEqual(code, 0)  # candidate improves f1, guardrail ok
        self.assertIn('PASS', text)
        # Outbox has one row per metric per scorecard (baseline + candidate).
        rows = [
            json.loads(line)
            for line in pathlib.Path(outbox).read_text().splitlines()
        ]
        self.assertTrue(rows)
        self.assertTrue(all(r['application_revision'] == 'r1' for r in rows))
        # Per-case score rows carry case_id/metric.
        srows = [
            json.loads(line)
            for line in pathlib.Path(scores).read_text().splitlines()
        ]
        self.assertTrue(any(r['metric'] == 'has_text' for r in srows))

    def test_gate_explicit_baseline_candidate_flags(self):
        code, _ = _run_cli(
            [
                'gate',
                '--suite',
                self.suite,
                '--mode',
                'replay',
                '--baseline',
                'baseline',
                '--candidate',
                'candidate',
            ]
        )
        self.assertEqual(code, 0)

    def test_compare_two_scorecards(self):
        base = str(self.root / 'b.json')
        cand = str(self.root / 'c.json')
        _run_cli(
            [
                'run',
                '--suite',
                self.suite,
                '--variant',
                'baseline',
                '--mode',
                'replay',
                '--out',
                base,
            ]
        )
        _run_cli(
            [
                'run',
                '--suite',
                self.suite,
                '--variant',
                'candidate',
                '--mode',
                'replay',
                '--out',
                cand,
            ]
        )
        code, text = _run_cli(
            [
                'compare',
                '--suite',
                self.suite,
                '--baseline',
                base,
                '--candidate',
                cand,
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn('f1', text)

    def test_sweep_ranks_variants(self):
        code, text = _run_cli(
            [
                'sweep',
                '--suite',
                self.suite,
                '--mode',
                'replay',
                '--variants',
                'baseline,candidate',
                '--export',
                str(self.root / 'sw.jsonl'),
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn('sweep', text)
        self.assertIn('Leaderboard', text)

    def test_pairwise_win_rate(self):
        code, text = _run_cli(
            [
                'pairwise',
                '--suite',
                self.suite,
                '--a',
                'baseline',
                '--b',
                'candidate',
                '--mode',
                'replay',
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn('pairwise', text)

    def test_pairwise_requires_content_ref(self):
        # A suite whose thresholds carry no pairwise config.
        thin = dict(SUITE)
        thin['thresholds'] = {}
        p = self.root / 'thin.yaml'
        p.write_text(yaml.safe_dump(thin))
        with self.assertRaises(SystemExit):
            _run_cli(
                [
                    'pairwise',
                    '--suite',
                    str(p),
                    '--a',
                    'baseline',
                    '--b',
                    'candidate',
                    '--mode',
                    'replay',
                ]
            )

    def test_agreement_cli(self):
        # A run with a judge score + a ratings file the CLI reads back.
        run = models.RunResult(
            run_id='R',
            scorecard=models.Scorecard(
                run_id='R',
                project='p',
                suite='s',
                variant=models.Variant(name='v'),
                dataset_version='v1',
            ),
            results=[
                models.CaseResult(
                    case=models.Case(id='c1'),
                    variant_name='v',
                    sample_hash='h0',
                    output=models.Output(fields={}),
                    scores=[
                        models.Score(
                            grader='quality',
                            metric='quality.clarity',
                            value=0.8,
                            case_id='c1',
                        )
                    ],
                )
            ],
        )
        run_path = str(self.root / 'r.json')
        store.write_run(run_path, run)
        ratings = str(self.root / 'ratings.jsonl')
        store.append_rating(
            ratings,
            models.Rating(
                run_id='R',
                case_id='c1',
                # Must match the result's hash or agreement joins nothing.
                sample_hash='h0',
                rater='a',
                scores={'clarity': 4},
            ),
        )
        code, text = _run_cli(
            [
                'agreement',
                '--run',
                run_path,
                '--ratings',
                ratings,
                '--dimensions',
                'clarity',
                '--judge-name',
                'quality',
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn('agreement', text)

    def _saved_run_and_ratings(self) -> tuple[str, str]:
        """A saved run (with a judge score) + a ratings file on disk."""
        run = models.RunResult(
            run_id='R',
            scorecard=models.Scorecard(
                run_id='R',
                project='p',
                suite='s',
                variant=models.Variant(name='v'),
                dataset_version='v1',
            ),
            results=[
                models.CaseResult(
                    case=models.Case(id='c1'),
                    variant_name='v',
                    sample_hash='h0',
                    output=models.Output(fields={}),
                    scores=[
                        models.Score(
                            grader='quality',
                            metric='quality.clarity',
                            value=0.8,
                            case_id='c1',
                            kind='per_case',
                        )
                    ],
                )
            ],
        )
        run_path = str(self.root / 'r.json')
        store.write_run(run_path, run)
        ratings = str(self.root / 'ratings.jsonl')
        store.append_rating(
            ratings,
            models.Rating(
                run_id='R',
                case_id='c1',
                # Must match the result's hash or agreement joins nothing.
                sample_hash='h0',
                rater='a',
                scores={'clarity': 4},
            ),
        )
        return run_path, ratings

    def test_report_from_saved_run_no_rerun(self):
        # `report` renders straight from a saved run - no suite, no re-run.
        run_path, _ = self._saved_run_and_ratings()
        out = str(self.root / 'report.html')
        code, _ = _run_cli(
            [
                'report',
                '--run',
                run_path,
                '--report',
                'html',
                '--report-out',
                out,
            ]
        )
        self.assertEqual(code, 0)
        html = pathlib.Path(out).read_text()
        self.assertTrue(html.startswith('<!doctype html'))
        self.assertIn('quality.clarity', html)
        self.assertNotIn('agreement', html)  # no ratings -> no agreement card

    def test_report_folds_in_ratings_agreement(self):
        run_path, ratings = self._saved_run_and_ratings()
        out = str(self.root / 'review.html')
        code, _ = _run_cli(
            [
                'report',
                '--run',
                run_path,
                '--ratings',
                ratings,
                '--dimensions',
                'clarity',
                '--judge-name',
                'quality',
                '--report',
                'html',
                '--report-out',
                out,
            ]
        )
        self.assertEqual(code, 0)
        html = pathlib.Path(out).read_text()
        # one document holding both the scorecard and the agreement card
        self.assertEqual(html.count('<!doctype html'), 1)
        self.assertIn('quality.clarity', html)  # scorecard
        self.assertIn('agreement', html)  # folded-in review section

    def test_report_ratings_without_dimensions_errors(self):
        run_path, ratings = self._saved_run_and_ratings()
        with self.assertRaises(SystemExit):
            _run_cli(['report', '--run', run_path, '--ratings', ratings])

    def test_rate_cli_invokes_serve(self):
        run = models.RunResult(
            run_id='R',
            scorecard=models.Scorecard(
                project='p',
                suite='s',
                variant=models.Variant(name='v'),
                dataset_version='v1',
            ),
            results=[],
        )
        run_path = str(self.root / 'r.json')
        store.write_run(run_path, run)
        with mock.patch.object(rating, 'serve') as served:
            code, _ = _run_cli(
                [
                    'rate',
                    '--run',
                    run_path,
                    '--ratings',
                    str(self.root / 'out.jsonl'),
                    '--dimensions',
                    'a,b',
                    '--view',
                    'Answer:text:output.answer',
                    '--view',
                    'Shot:image:artifacts.shot',
                    '--no-open',
                ]
            )
        self.assertEqual(code, 0)
        served.assert_called_once()
        views = served.call_args.kwargs['views']
        self.assertEqual(
            views[0],
            {'label': 'Answer', 'kind': 'text', 'ref': 'output.answer'},
        )

    def test_rate_view_two_part_infers_kind(self):
        self.assertEqual(
            cli._parse_views(['Answer:output.answer']),
            [{'label': 'Answer', 'ref': 'output.answer'}],
        )
        self.assertIsNone(cli._parse_views(None))
        with self.assertRaises(SystemExit):
            cli._parse_views(['nocolons'])

    def test_run_checkpoint_and_resume(self):
        ckpt = str(self.root / 'ck.jsonl')
        code, _ = _run_cli(
            [
                'run',
                '--suite',
                self.suite,
                '--variant',
                'candidate',
                '--mode',
                'replay',
                '--checkpoint',
                ckpt,
            ]
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(store.read_checkpoint_samples(ckpt)), 2)  # c1, c2
        run_id = store.checkpoint_meta(ckpt)['run_id']
        # Resume with everything already done: a no-op that still succeeds and
        # keeps the same run id.
        code, _ = _run_cli(
            [
                'run',
                '--suite',
                self.suite,
                '--variant',
                'candidate',
                '--mode',
                'replay',
                '--checkpoint',
                ckpt,
                '--resume',
            ]
        )
        self.assertEqual(code, 0)
        self.assertEqual(store.checkpoint_meta(ckpt)['run_id'], run_id)

    def _ab_run_files(self):
        """Two saved run files (baseline, candidate) over c1/c2."""
        paths = {}
        for name in ('baseline', 'candidate'):
            run = models.RunResult(
                run_id=name,
                scorecard=models.Scorecard(
                    run_id=name,
                    project='demo',
                    suite='s',
                    variant=models.Variant(name=name),
                    dataset_version='v1',
                ),
                results=[
                    models.CaseResult(
                        case=models.Case(id=c),
                        variant_name=name,
                        sample_hash='h0',
                        output=models.Output(
                            fields={'html': f'<b>{name}:{c}</b>'}
                        ),
                    )
                    for c in ('c1', 'c2')
                ],
            )
            path = str(self.root / f'{name}.run.json')
            store.write_run(path, run)
            paths[name] = path
        return paths

    def _seed_prefs(self):
        path = str(self.root / 'prefs.jsonl')
        for cid, winner in [('c1', 'a'), ('c2', 'b')]:
            store.append_preference(
                path,
                models.Preference(
                    case_id=cid,
                    variant_a='baseline',
                    variant_b='candidate',
                    rater='r1',
                    winner=winner,
                    dims={'visual': winner},
                ),
            )
        return path

    def test_rank_cli_invokes_serve_rank(self):
        files = self._ab_run_files()
        with mock.patch.object(rating, 'serve_rank') as served:
            code, _ = _run_cli(
                [
                    'rank',
                    '--run-a',
                    files['baseline'],
                    '--run-b',
                    files['candidate'],
                    '--preferences',
                    str(self.root / 'p.jsonl'),
                    '--dimensions',
                    'visual',
                    '--content-ref',
                    'output.html',
                    '--no-open',
                ]
            )
        self.assertEqual(code, 0)
        served.assert_called_once()

    def test_preferences_cli_markdown(self):
        files = self._ab_run_files()
        prefs = self._seed_prefs()
        code, text = _run_cli(
            [
                'preferences',
                '--run-a',
                files['baseline'],
                '--run-b',
                files['candidate'],
                '--preferences',
                prefs,
                '--dimensions',
                'visual',
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn('human preferences', text)
        self.assertIn('A win-rate', text)

    def test_preferences_cli_html_to_file(self):
        files = self._ab_run_files()
        prefs = self._seed_prefs()
        out = str(self.root / 'pref.html')
        code, text = _run_cli(
            [
                'preferences',
                '--run-a',
                files['baseline'],
                '--run-b',
                files['candidate'],
                '--preferences',
                prefs,
                '--report',
                'html',
                '--report-out',
                out,
            ]
        )
        self.assertEqual(code, 0)
        html = pathlib.Path(out).read_text()
        self.assertTrue(html.startswith('<!doctype html'))
        self.assertIn('Human preferences', html)
        self.assertIn('wrote report', text)

    def test_pairwise_with_preferences_folds_agreement(self):
        prefs = self._seed_prefs()
        code, text = _run_cli(
            [
                'pairwise',
                '--suite',
                self.suite,
                '--a',
                'baseline',
                '--b',
                'candidate',
                '--mode',
                'replay',
                '--preferences',
                prefs,
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn('pairwise agreement', text)


if __name__ == '__main__':
    unittest.main()
