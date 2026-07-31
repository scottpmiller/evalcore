"""Extra pairwise coverage: client selection and content mapping."""

import asyncio
import unittest

from evalcore import models, pairwise, rating


class BuildClientTests(unittest.TestCase):
    def test_replay_requires_path(self):
        with self.assertRaises(ValueError):
            pairwise.build_pairwise_client('replay', {})

    def test_live_requires_model(self):
        with self.assertRaises(ValueError):
            pairwise.build_pairwise_client('http', {})

    def test_selects_anthropic_and_openai(self):
        a = pairwise.build_pairwise_client('http', {'model': 'claude-x'})
        self.assertIsInstance(a, pairwise.AnthropicPairwiseClient)
        o = pairwise.build_pairwise_client(
            'http', {'model': 'openai:gpt-4o', 'provider': 'openai'}
        )
        self.assertIsInstance(o, pairwise.OpenAIPairwiseClient)
        self.assertEqual(o.model, 'gpt-4o')

    def test_replay_client_from_pairs_mapping(self):
        # Construction from a {'pairs': [...]} mapping, then a lookup.
        rc = pairwise.ReplayPairwiseClient(
            {'pairs': [{'a': 'x', 'b': 'y', 'winner': 'x'}]}
        )
        pick = asyncio.run(
            rc.compare(system='', user='', first='x', second='y')
        )
        self.assertEqual(pick, 'first')


class ContentMapTests(unittest.TestCase):
    def test_skips_missing_content(self):
        run = models.RunResult(
            run_id='R',
            scorecard=models.Scorecard(
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
                    output=models.Output(fields={'text': 'hi'}),
                ),
                models.CaseResult(
                    case=models.Case(id='c2'),
                    variant_name='v',
                    sample_hash='h0',
                    output=models.Output(fields={}),
                ),  # no content -> skipped
            ],
        )
        mapped = pairwise.comparable_samples(run, 'output.text')
        self.assertEqual(set(mapped), {'c1'})

    def test_groups_a_cases_samples_in_run_order(self):
        run = _run('v', {'c1': ['first', 'second'], 'c2': ['only']})
        mapped = pairwise.comparable_samples(run, 'output.text')
        self.assertEqual(
            [content for content, _ in mapped['c1']], ['first', 'second']
        )
        self.assertEqual(len(mapped['c2']), 1)


def _run(variant, texts, run_id='R'):
    """A RunResult from ``{case_id: [content per sample]}``."""
    results = [
        models.CaseResult(
            case=models.Case(id=case_id),
            variant_name=variant,
            sample_hash=f'{variant}:{case_id}:{i}',
            output=models.Output(fields={'text': content}),
        )
        for case_id, contents in texts.items()
        for i, content in enumerate(contents)
    ]
    return models.RunResult(
        run_id=run_id,
        scorecard=models.Scorecard(
            project='p',
            suite='s',
            variant=models.Variant(name=variant),
            dataset_version='v1',
        ),
        results=results,
    )


class MultiSampleAlignmentTests(unittest.TestCase):
    """Two runs align by case, then sample order - never by output hash,
    which differs on both sides by construction."""

    def _judge(self, a, b, client):
        return asyncio.run(
            pairwise.judge_pairwise(
                a, b, content_ref='output.text', client=client
            )
        )

    def test_each_sample_becomes_its_own_pair(self):
        a = _run('base', {'c1': ['a one', 'a two']})
        b = _run('cand', {'c1': ['b one', 'b two']})
        client = pairwise.ReplayPairwiseClient(
            [
                {'a': 'a one', 'b': 'b one', 'winner': 'a one'},
                {'a': 'a two', 'b': 'b two', 'winner': 'b two'},
            ]
        )
        result = self._judge(a, b, client)
        self.assertEqual(result.n, 2)
        self.assertEqual(result.a_wins, 1)
        self.assertEqual(result.b_wins, 1)
        # Both outcomes name the same case but different samples, anchored on
        # the variant_a side so a Preference can join to them.
        self.assertEqual({o.case_id for o in result.outcomes}, {'c1'})
        self.assertEqual(
            [o.sample_hash for o in result.outcomes],
            ['base:c1:0', 'base:c1:1'],
        )

    def test_uneven_sample_counts_pair_to_the_thinner_side(self):
        # B lost a sample to an error, so only one pair is comparable.
        a = _run('base', {'c1': ['a one', 'a two']})
        b = _run('cand', {'c1': ['b one']})
        client = pairwise.ReplayPairwiseClient(
            [{'a': 'a one', 'b': 'b one', 'winner': 'tie'}]
        )
        result = self._judge(a, b, client)
        self.assertEqual(result.n, 1)
        self.assertEqual(result.outcomes[0].sample_hash, 'base:c1:0')

    def test_an_errored_sample_is_not_comparable(self):
        # A failed invocation may still carry content the ref resolves. It is
        # not a candidate answer, so it must not be judged as one.
        run = _run('base', {'c1': ['leftover']})
        run.results[0].output.error = 'boom'
        self.assertEqual(pairwise.comparable_samples(run, 'output.text'), {})

    def test_the_human_and_judge_paths_drop_the_same_samples(self):
        # The whole point of one shared filter: alignment inside a case is
        # positional, so if the ranking app and the judge disagreed about which
        # samples count, they would pair A's n-th against different B samples
        # and compute_pairwise_agreement would score two different comparisons
        # as one.
        a = _run('base', {'c1': ['a one', 'a two', 'a three']})
        b = _run('cand', {'c1': ['b one', 'b two', 'b three']})
        b.results[0].output.error = 'boom'  # errored, content still resolves
        a.results[1].output.fields = {}  # no content, no error

        app = rating._RankApp(
            a, b, 'prefs.jsonl', ['overall'], content_ref='output.text'
        )
        judged = self._judge(a, b, pairwise.ReplayPairwiseClient([]))

        # Same pairs, in the same order, on both paths.
        self.assertEqual(
            [(i['case_id'], i['sample_hash']) for i in app.items],
            [(o.case_id, o.sample_hash) for o in judged.outcomes],
        )
        # A's dropped sample is 'a two'; B's is 'b one'. What survives is
        # A[0]+A[2] against B[1]+B[2], and both paths agree on that.
        self.assertEqual(
            [i['sample_hash'] for i in app.items], ['base:c1:0', 'base:c1:2']
        )


if __name__ == '__main__':
    unittest.main()
