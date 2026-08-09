"""Unit tests for the LLM-judge rubric grader (offline, no API)."""

import asyncio
import unittest
from unittest import mock

from evalcore import loader, models
from evalcore.graders import base, judge

DIMENSIONS = [
    {'key': 'clarity', 'description': 'Is it clear?'},
    {'key': 'specificity', 'description': 'Is it specific?'},
]


def _grade(grader, case, output):
    return asyncio.run(grader.grade(case, output))


class RubricJudgeTests(unittest.TestCase):
    def _grader(self, fixtures):
        return judge.RubricJudge(
            content_ref='output.text',
            dimensions=DIMENSIONS,
            scale=5,
            client=judge.ReplayJudgeClient(fixtures),
        )

    def test_scores_normalized_and_overall_averaged(self):
        grader = self._grader(
            {'Great subject': {'scores': {'clarity': 5, 'specificity': 4}}}
        )
        case = models.Case(id='c1')
        output = models.Output(fields={'text': 'Great subject'})
        scores = {s.metric: s.value for s in _grade(grader, case, output)}
        self.assertAlmostEqual(scores['llm_judge.clarity'], 1.0)
        self.assertAlmostEqual(scores['llm_judge.specificity'], 0.8)
        self.assertAlmostEqual(scores['llm_judge.overall'], 0.9)

    def test_overall_retains_rationale_and_raw_points(self):
        grader = self._grader(
            {
                'Great subject': {
                    'scores': {'clarity': 5, 'specificity': 4},
                    'rationale': 'Clear and specific enough.',
                }
            }
        )
        case = models.Case(id='c1')
        output = models.Output(fields={'text': 'Great subject'})
        overall = next(
            s
            for s in _grade(grader, case, output)
            if s.metric == 'llm_judge.overall'
        )
        self.assertEqual(len(overall.judges), 1)
        jd = overall.judges[0]
        self.assertEqual(jd.rationale, 'Clear and specific enough.')
        # raw 1..scale points are retained, not just the normalized mean
        self.assertEqual(jd.points, {'clarity': 5, 'specificity': 4})
        self.assertAlmostEqual(jd.overall, 0.9)
        # the overall score carries the whole breakdown
        self.assertEqual(jd.points, {'clarity': 5, 'specificity': 4})

    def test_a_dimension_row_carries_its_own_verdict(self):
        """Every judge row describes what that row measures.

        The dimension row used to carry nothing, so `0.8` alone could not say
        who scored it, on what scale, or that the raw number was 4.
        """
        grader = self._grader(
            {
                'Great subject': {
                    'scores': {'clarity': 5, 'specificity': 4},
                    'rationale': 'Clear and specific enough.',
                }
            }
        )
        scores = _grade(
            grader,
            models.Case(id='c1'),
            models.Output(fields={'text': 'Great subject'}),
        )
        dim = next(s for s in scores if s.metric == 'llm_judge.specificity')
        self.assertEqual(len(dim.judges), 1)
        jd = dim.judges[0]
        # Scoped: only the point this row is about.
        self.assertEqual(jd.points, {'specificity': 4})
        self.assertAlmostEqual(jd.overall, 0.8)
        self.assertEqual(jd.rationale, 'Clear and specific enough.')

    def test_value_is_the_mean_of_the_judge_scores(self):
        """The invariant that holds on every judge row, dimensions included:
        value == mean(judges.score)."""
        grader = self._grader(
            {'X': {'scores': {'clarity': 5, 'specificity': 2}}}
        )
        scores = _grade(
            grader, models.Case(id='c'), models.Output(fields={'text': 'X'})
        )
        judged = [s for s in scores if s.judges and s.value is not None]
        self.assertTrue(judged)
        for s in judged:
            mean = sum(j.overall for j in s.judges) / len(s.judges)
            self.assertAlmostEqual(s.value, mean, msg=s.metric)

    def test_the_overall_row_keeps_the_full_map(self):
        grader = self._grader(
            {'X': {'scores': {'clarity': 5, 'specificity': 4}}}
        )
        scores = _grade(
            grader, models.Case(id='c'), models.Output(fields={'text': 'X'})
        )
        overall = next(s for s in scores if s.metric == 'llm_judge.overall')
        self.assertEqual(
            overall.judges[0].points, {'clarity': 5, 'specificity': 4}
        )

    def test_different_content_gets_different_scores(self):
        grader = self._grader(
            {
                'Good': {'scores': {'clarity': 4, 'specificity': 4}},
                'Bad': {'scores': {'clarity': 2, 'specificity': 1}},
            }
        )
        good = _grade(
            grader, models.Case(id='g'), models.Output(fields={'text': 'Good'})
        )
        bad = _grade(
            grader, models.Case(id='b'), models.Output(fields={'text': 'Bad'})
        )
        good_overall = next(
            s.value for s in good if s.metric == 'llm_judge.overall'
        )
        bad_overall = next(
            s.value for s in bad if s.metric == 'llm_judge.overall'
        )
        self.assertGreater(good_overall, bad_overall)

    def test_missing_content_yields_none_scores(self):
        grader = self._grader({})
        scores = _grade(grader, models.Case(id='c'), models.Output())
        self.assertTrue(all(s.value is None for s in scores))

    def test_missing_dimension_in_judgment_is_none(self):
        grader = self._grader({'X': {'scores': {'clarity': 3}}})
        graded = _grade(
            grader, models.Case(id='c'), models.Output(fields={'text': 'X'})
        )
        scores = {s.metric: s.value for s in graded}
        self.assertAlmostEqual(scores['llm_judge.clarity'], 0.6)
        self.assertIsNone(scores['llm_judge.specificity'])
        # overall averages only the present dimension.
        self.assertAlmostEqual(scores['llm_judge.overall'], 0.6)

    def test_records_judge_version_in_detail(self):
        grader = judge.RubricJudge(
            content_ref='output.text',
            dimensions=DIMENSIONS,
            judge_version='v3',
            client=judge.ReplayJudgeClient(
                {'X': {'scores': {'clarity': 5, 'specificity': 5}}}
            ),
        )
        scores = _grade(
            grader, models.Case(id='c'), models.Output(fields={'text': 'X'})
        )
        self.assertTrue(all('@v3' in s.detail for s in scores))

    def test_judge_version_pin_single_and_panel(self):
        single = judge.RubricJudge(
            content_ref='output.text',
            dimensions=DIMENSIONS,
            replay_path='f.yaml',
            judge_version='v3',
        )
        # No model configured (replay-only), so the pin is key@version.
        self.assertEqual(single.judge_version, 'anthropic@v3')
        panel = judge.RubricJudge(
            content_ref='output.text',
            dimensions=DIMENSIONS,
            judges=[
                {'key': 'claude', 'model': 'm', 'judge_version': 'v2'},
                {
                    'key': 'gpt',
                    'provider': 'openai',
                    'model': 'openai:g',
                    'judge_version': 'v5',
                },
            ],
        )
        self.assertEqual(panel.judge_version, 'claude:m@v2,gpt:openai:g@v5')

    def test_the_model_is_in_the_pin(self):
        """Swapping the judge's model has to move the pin.

        Without this, a model swap that left `judge_version` alone produced
        a byte-identical pin, so a comparison against a baseline scored by a
        different model passed every provenance check reading this field.
        """

        def _judge(model):
            return judge.RubricJudge(
                content_ref='output.text',
                dimensions=DIMENSIONS,
                model=model,
                judge_version='v1',
            )

        before = _judge('claude-sonnet-4-6').judge_version
        after = _judge('claude-opus-4-1').judge_version
        self.assertEqual(before, 'anthropic:claude-sonnet-4-6@v1')
        self.assertNotEqual(before, after)

    def test_a_single_judge_names_itself_by_provider(self):
        """Not the literal 'judge', which carried no information."""
        single = judge.RubricJudge(
            content_ref='output.text',
            dimensions=DIMENSIONS,
            replay_path='f.yaml',
        )
        self.assertEqual(single.judges[0]['key'], 'anthropic')

    def test_transient_client_error_retried_via_set_retry(self):
        class _RateLimit(Exception):
            status_code = 429

        class _FlakyClient:
            def __init__(self):
                self.calls = 0

            async def judge(self, **kw):
                self.calls += 1
                if self.calls <= 2:
                    raise _RateLimit()
                return {'scores': {'clarity': 5, 'specificity': 4}}

        client = _FlakyClient()
        grader = judge.RubricJudge(
            content_ref='output.text', dimensions=DIMENSIONS, client=client
        )
        grader.set_retry(loader.RetryConfig(max_attempts=5, jitter=0))
        with mock.patch('evalcore.retry.asyncio.sleep') as slept:
            graded = _grade(
                grader,
                models.Case(id='c'),
                models.Output(fields={'text': 'X'}),
            )
        scores = {s.metric: s.value for s in graded}
        self.assertEqual(client.calls, 3)  # 2 failures + 1 success
        self.assertAlmostEqual(scores['llm_judge.clarity'], 1.0)
        self.assertEqual(slept.call_count, 2)


class PanelJudgeTests(unittest.TestCase):
    def _panel(self, claude_scores, gpt_scores, threshold=2.0):
        grader = judge.RubricJudge(
            content_ref='output.text',
            dimensions=DIMENSIONS,
            scale=5,
            disagreement_threshold=threshold,
            judges=[
                {
                    'key': 'claude',
                    'provider': 'anthropic',
                    'replay_path': {'X': {'scores': claude_scores}},
                },
                {
                    'key': 'gpt',
                    'provider': 'openai',
                    'replay_path': {'X': {'scores': gpt_scores}},
                },
            ],
        )
        grader.set_mode('replay')
        graded = _grade(
            grader, models.Case(id='c'), models.Output(fields={'text': 'X'})
        )
        return {s.metric: s.value for s in graded}

    def test_panel_mean_and_per_judge_overall(self):
        scores = self._panel(
            {'clarity': 5, 'specificity': 5}, {'clarity': 3, 'specificity': 1}
        )
        # clarity mean(1.0, 0.6)=0.8; specificity mean(1.0, 0.2)=0.6.
        self.assertAlmostEqual(scores['llm_judge.clarity'], 0.8)
        self.assertAlmostEqual(scores['llm_judge.specificity'], 0.6)
        self.assertAlmostEqual(scores['llm_judge.overall'], 0.7)
        # Each judge's own overall is surfaced for bias inspection.
        self.assertAlmostEqual(scores['llm_judge.claude.overall'], 1.0)
        self.assertAlmostEqual(scores['llm_judge.gpt.overall'], 0.4)

    def test_panel_overall_retains_each_judges_breakdown(self):
        grader = judge.RubricJudge(
            content_ref='output.text',
            dimensions=DIMENSIONS,
            scale=5,
            judges=[
                {
                    'key': 'claude',
                    'provider': 'anthropic',
                    'replay_path': {
                        'X': {
                            'scores': {'clarity': 5, 'specificity': 5},
                            'rationale': 'claude liked it',
                        }
                    },
                },
                {
                    'key': 'gpt',
                    'provider': 'openai',
                    'replay_path': {
                        'X': {
                            'scores': {'clarity': 3, 'specificity': 1},
                            'rationale': 'gpt was harsher',
                        }
                    },
                },
            ],
        )
        grader.set_mode('replay')
        graded = _grade(
            grader, models.Case(id='c'), models.Output(fields={'text': 'X'})
        )
        overall = next(s for s in graded if s.metric == 'llm_judge.overall')
        by_key = {jd.key: jd for jd in overall.judges}
        self.assertEqual(set(by_key), {'claude', 'gpt'})
        self.assertEqual(by_key['claude'].rationale, 'claude liked it')
        self.assertEqual(
            by_key['gpt'].points, {'clarity': 3, 'specificity': 1}
        )
        self.assertAlmostEqual(by_key['claude'].overall, 1.0)
        self.assertAlmostEqual(by_key['gpt'].overall, 0.4)

    def test_disagreement_flags_when_over_threshold(self):
        scores = self._panel(
            {'clarity': 5, 'specificity': 5},
            {'clarity': 3, 'specificity': 1},
            threshold=2,
        )
        # spreads: clarity 2, specificity 4 -> mean 3.0, max 4 >= 2.
        self.assertAlmostEqual(scores['llm_judge.disagreement'], 3.0)
        self.assertEqual(scores['llm_judge.flagged'], 1.0)

    def test_no_flag_when_judges_agree(self):
        scores = self._panel(
            {'clarity': 4, 'specificity': 5},
            {'clarity': 5, 'specificity': 5},
            threshold=2,
        )
        # max spread is 1 (< 2) -> not flagged.
        self.assertEqual(scores['llm_judge.flagged'], 0.0)
        self.assertAlmostEqual(scores['llm_judge.disagreement'], 0.5)

    def test_single_judge_emits_no_panel_metrics(self):
        grader = judge.RubricJudge(
            content_ref='output.text',
            dimensions=DIMENSIONS,
            client=judge.ReplayJudgeClient(
                {'X': {'scores': {'clarity': 4, 'specificity': 4}}}
            ),
        )
        graded = _grade(
            grader, models.Case(id='c'), models.Output(fields={'text': 'X'})
        )
        metrics = {s.metric for s in graded}
        self.assertNotIn('llm_judge.disagreement', metrics)
        self.assertNotIn('llm_judge.flagged', metrics)

    def test_image_refs_loaded_from_inline_data(self):
        captured = {}

        class Recorder:
            async def judge(self, *, images, **kwargs):
                captured['images'] = images
                return {'scores': {'clarity': 5, 'specificity': 5}}

        grader = judge.RubricJudge(
            content_ref='output.text',
            dimensions=DIMENSIONS,
            image_refs={'shot': 'output.screenshot'},
            client=Recorder(),
        )
        _grade(
            grader,
            models.Case(id='c'),
            models.Output(
                fields={
                    'text': 'X',
                    'screenshot': {'media_type': 'image/png', 'data': 'AAAA'},
                }
            ),
        )
        self.assertEqual(len(captured['images']), 1)
        self.assertEqual(captured['images'][0]['media_type'], 'image/png')


class RegistryAndModeTests(unittest.TestCase):
    def test_built_from_registry(self):
        per_case, aggregate = base.build_graders(
            [
                {
                    'type': 'llm_judge',
                    'content_ref': 'output.text',
                    'dimensions': DIMENSIONS,
                    'model': 'claude-sonnet-4-6',
                }
            ]
        )
        self.assertEqual(len(per_case), 1)
        self.assertEqual(aggregate, [])

    def test_replay_mode_without_path_raises(self):
        grader = judge.RubricJudge(
            content_ref='output.text', dimensions=DIMENSIONS
        )
        grader.set_mode('replay')
        with self.assertRaises(ValueError):
            _grade(
                grader,
                models.Case(id='c'),
                models.Output(fields={'text': 'x'}),
            )


if __name__ == '__main__':
    unittest.main()
