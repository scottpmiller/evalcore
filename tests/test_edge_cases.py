"""Targeted branch coverage for the small pure modules."""

import asyncio
import pathlib
import tempfile
import unittest

import yaml

from evalcore import compare, loader, models, pairwise, refs, sweep
from evalcore.adapters import http as http_adapter
from evalcore.graders import classification


def _card(name, **metrics):
    return models.Scorecard(
        project='p',
        suite='s',
        variant=models.Variant(name=name),
        dataset_version='v1',
        metrics={
            m: models.MetricValue(metric=m, value=v, kind='aggregate', n=1)
            for m, v in metrics.items()
        },
    )


class CompareBranchTests(unittest.TestCase):
    def test_guardrail_metric_absent_on_candidate(self):
        result = compare.compare(
            _card('b', x=1.0),
            _card('c'),
            {'guardrails': [{'metric': 'x', 'max': 1.0}]},
        )
        self.assertEqual(result.verdict, 'fail')
        self.assertIn('absent', result.guardrails[0].detail)

    def test_min_and_must_not_decrease_breaches(self):
        r1 = compare.compare(
            _card('b', x=0.9),
            _card('c', x=0.5),
            {'guardrails': [{'metric': 'x', 'min': 0.9}]},
        )
        self.assertFalse(r1.guardrails[0].passed)
        r2 = compare.compare(
            _card('b', x=0.9),
            _card('c', x=0.5),
            {'guardrails': [{'metric': 'x', 'must_not_decrease': True}]},
        )
        self.assertIn('decreased', r2.guardrails[0].detail)

    def test_no_win_metric_summary(self):
        result = compare.compare(_card('b', x=1.0), _card('c', x=1.0), {})
        self.assertEqual(result.summary, 'no win metric configured')
        self.assertEqual(result.win, 'neutral')

    def test_win_metric_absent_is_neutral(self):
        result = compare.compare(_card('b'), _card('c'), {'win_metric': 'f1'})
        self.assertEqual(result.win, 'neutral')

    def test_neutral_when_within_band(self):
        result = compare.compare(
            _card('b', f1=0.80),
            _card('c', f1=0.80),
            {'win_metric': 'f1', 'win_min_delta': 0.01},
        )
        self.assertEqual(result.win, 'neutral')


class ClassificationEdgeTests(unittest.TestCase):
    def test_confusion_with_errors_and_unknown_labels(self):
        grader = classification.Classification(
            predicted_ref='output.label',
            expected_ref='expected.label',
            positive_labels=['bad'],
            negative_labels=['good'],
        )

        def result(cid, predicted, expected, error=None):
            return models.CaseResult(
                case=models.Case(id=cid, expected={'label': expected}),
                variant_name='v',
                sample_hash='h0',
                output=models.Output(fields={'label': predicted}, error=error),
            )

        results = [
            result('tp', 'bad', 'bad'),
            result('tn', 'good', 'good'),
            result('fp', 'bad', 'good'),
            result('fn', 'good', 'bad'),
            result('unknown_pred', 'weird', 'good'),  # unlisted -> negative
            result('none_expected', 'bad', None),  # -> errors
            result('errored', None, 'bad', error='boom'),
        ]
        scores = {s.metric: s.value for s in grader.aggregate(results)}
        # 'errored' (output error) + 'none_expected' (unresolved label).
        self.assertEqual(scores['errors'], 2.0)
        # tp=1, fp=1, fn=1, tn=2 (unknown pred -> negative).
        self.assertAlmostEqual(scores['precision'], 0.5)  # tp / (tp+fp)
        self.assertAlmostEqual(scores['recall'], 0.5)  # tp / (tp+fn)


class RefsEdgeTests(unittest.TestCase):
    def test_none_midpath_and_attr_and_missing_root(self):
        self.assertIsNone(refs.resolve_path({'a': None}, 'a.b'))

        class Obj:
            x = 5

        self.assertEqual(refs.resolve_path(Obj(), 'x'), 5)
        self.assertIsNone(refs.resolve_ref({'input': {}}, 'nope.x'))

    def test_build_value_walks_lists(self):
        out = refs.build_value(['$a', 'lit', {'k': '$a'}], {'a': 1})
        self.assertEqual(out, [1, 'lit', {'k': 1}])


class HttpEnvTests(unittest.TestCase):
    def test_expand_env_recurses_into_list(self):
        self.assertEqual(http_adapter.expand_env(['${NOPE}', 2]), ['', 2])


class SweepMissingMetricTests(unittest.TestCase):
    def test_variant_missing_win_metric_sorts_last(self):
        run1 = models.RunResult(
            run_id='1', scorecard=_card('a', f1=0.9), results=[]
        )
        run2 = models.RunResult(
            run_id='2', scorecard=_card('b'), results=[]
        )  # no f1
        result = sweep.summarize_sweep([run1, run2], {'win_metric': 'f1'})
        self.assertEqual(result.entries[0].variant, 'a')
        self.assertEqual(result.entries[-1].variant, 'b')
        self.assertIsNone(result.entries[-1].win_value)


class LoaderEdgeTests(unittest.TestCase):
    def test_load_cases_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                loader.load_cases(tmp)  # no cases/ dir
            (pathlib.Path(tmp) / 'cases').mkdir()
            with self.assertRaises(FileNotFoundError):
                loader.load_cases(tmp)  # cases/ empty

    def test_load_suite_resolves_judge_replay_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            suite = {
                'project': 'p',
                'suite': 's',
                'dataset': 'data',
                'replay_fixtures': 'f.yaml',
                'adapter': {'type': 'replay'},
                'graders': [
                    {
                        'type': 'llm_judge',
                        'content_ref': 'output.text',
                        'dimensions': [{'key': 'q'}],
                        'replay_path': 'top.yaml',
                        'judges': [{'key': 'c', 'replay_path': 'c.yaml'}],
                    }
                ],
                'thresholds': {'pairwise': {'replay_path': 'pw.yaml'}},
            }
            path = root / 'suite.yaml'
            path.write_text(yaml.safe_dump(suite))
            cfg = loader.load_suite(path)
            self.assertTrue(cfg.graders[0]['replay_path'].endswith('top.yaml'))
            self.assertTrue(
                cfg.graders[0]['judges'][0]['replay_path'].startswith(
                    str(root)
                )
            )
            self.assertTrue(
                cfg.thresholds['pairwise']['replay_path'].startswith(str(root))
            )
            self.assertIsNotNone(cfg.suite_hash)


class PairwiseReplayTests(unittest.TestCase):
    def test_winner_not_in_pair_is_tie(self):
        rc = pairwise.ReplayPairwiseClient(
            [{'a': 'x', 'b': 'y', 'winner': 'z'}]
        )  # 'z' matches neither
        pick = asyncio.run(
            rc.compare(system='', user='', first='x', second='y')
        )
        self.assertEqual(pick, 'tie')

    def test_judge_pairwise_with_rubric_and_context(self):
        def _run(name, text):
            return models.RunResult(
                run_id=name,
                scorecard=models.Scorecard(
                    project='p',
                    suite='s',
                    variant=models.Variant(name=name),
                    dataset_version='v1',
                ),
                results=[
                    models.CaseResult(
                        case=models.Case(id='c1', input={'q': 'ask'}),
                        variant_name=name,
                        sample_hash='h0',
                        output=models.Output(fields={'text': text}),
                    )
                ],
            )

        client = pairwise.ReplayPairwiseClient(
            [{'a': 'A out', 'b': 'B out', 'winner': 'A out'}]
        )
        result = asyncio.run(
            pairwise.judge_pairwise(
                _run('a', 'A out'),
                _run('b', 'B out'),
                content_ref='output.text',
                client=client,
                rubric='be strict',
                context_refs={'request': 'input.q'},
            )
        )
        self.assertEqual(result.a_wins, 1)
        self.assertEqual(result.win_rate_a, 1.0)


if __name__ == '__main__':
    unittest.main()
