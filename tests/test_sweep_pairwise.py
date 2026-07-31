"""Unit tests for N-way sweeps and pairwise (A-vs-B) judging."""

import asyncio
import unittest

from evalcore import models, pairwise, sweep


def _card(variant, metrics):
    return models.Scorecard(
        run_id=variant,
        project='p',
        suite='s',
        variant=models.Variant(name=variant),
        dataset_version='v1',
        metrics={
            k: models.MetricValue(metric=k, value=v, kind='mean', n=1)
            for k, v in metrics.items()
        },
    )


def _run(variant, metrics, texts=None):
    """A RunResult; ``texts`` maps case_id -> content for pairwise tests."""
    results = []
    for case_id, text in (texts or {}).items():
        results.append(
            models.CaseResult(
                case=models.Case(id=case_id),
                variant_name=variant,
                sample_hash='h0',
                output=models.Output(fields={'text': text}),
            )
        )
    card = _card(variant, metrics)
    return models.RunResult(run_id=variant, scorecard=card, results=results)


class SweepTests(unittest.TestCase):
    def test_ranks_by_win_metric_and_builds_matrix(self):
        runs = [
            _run('haiku', {'f1': 0.7, 'latency': 100.0}),
            _run('sonnet', {'f1': 0.9, 'latency': 200.0}),
            _run('opus', {'f1': 0.8, 'latency': 300.0}),
        ]
        result = sweep.summarize_sweep(
            runs, {'win_metric': 'f1', 'win_higher_is_better': True}
        )
        self.assertEqual(
            [(e.variant, e.rank) for e in result.entries],
            [('sonnet', 1), ('opus', 2), ('haiku', 3)],
        )
        self.assertEqual(result.matrix['f1']['sonnet'], 0.9)
        self.assertEqual(result.matrix['latency']['haiku'], 100.0)

    def test_lower_is_better_flips_ranking(self):
        runs = [_run('a', {'cost': 0.5}), _run('b', {'cost': 0.2})]
        result = sweep.summarize_sweep(
            runs, {'win_metric': 'cost', 'win_higher_is_better': False}
        )
        self.assertEqual(result.entries[0].variant, 'b')


def _judge(run_a, run_b, client, **kw):
    return asyncio.run(
        pairwise.judge_pairwise(
            run_a, run_b, content_ref='output.text', client=client, **kw
        )
    )


class PairwiseTests(unittest.TestCase):
    def test_win_rate_from_replay(self):
        a = _run('base', {}, {'c1': 'weak one', 'c2': 'weak two'})
        b = _run('cand', {}, {'c1': 'strong one', 'c2': 'strong two'})
        client = pairwise.ReplayPairwiseClient(
            [
                {'a': 'weak one', 'b': 'strong one', 'winner': 'strong one'},
                {'a': 'weak two', 'b': 'strong two', 'winner': 'strong two'},
            ]
        )
        result = _judge(a, b, client)
        # B (cand) wins both -> A win-rate 0.
        self.assertEqual(result.n, 2)
        self.assertEqual(result.b_wins, 2)
        self.assertEqual(result.a_wins, 0)
        self.assertEqual(result.win_rate_a, 0.0)

    def test_tie_counts_as_half(self):
        a = _run('base', {}, {'c1': 'x'})
        b = _run('cand', {}, {'c1': 'y'})
        client = pairwise.ReplayPairwiseClient(
            [{'a': 'x', 'b': 'y', 'winner': 'tie'}]
        )
        result = _judge(a, b, client)
        self.assertEqual(result.ties, 1)
        self.assertEqual(result.win_rate_a, 0.5)

    def test_counterbalancing_collapses_position_bias_to_tie(self):
        # A client that always picks whichever is shown first is pure
        # position bias; counterbalancing must neutralize it to a tie.
        class FirstBiased:
            async def compare(self, *, system, user, first, second):
                return 'first'

        a = _run('base', {}, {'c1': 'x'})
        b = _run('cand', {}, {'c1': 'y'})
        result = _judge(a, b, FirstBiased())
        self.assertEqual(result.ties, 1)
        self.assertEqual(result.a_wins, 0)
        self.assertEqual(result.b_wins, 0)

    def test_no_counterbalance_keeps_first_order_pick(self):
        class FirstBiased:
            async def compare(self, *, system, user, first, second):
                return 'first'

        a = _run('base', {}, {'c1': 'x'})
        b = _run('cand', {}, {'c1': 'y'})
        result = _judge(a, b, FirstBiased(), counterbalance=False)
        # Without counterbalancing, order-1 'first' -> A wins.
        self.assertEqual(result.a_wins, 1)


if __name__ == '__main__':
    unittest.main()
