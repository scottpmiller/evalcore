"""End-to-end offline run of the quickstart support-reply example.

Exercises every grader family through the engine (deterministic, custom
per-case + aggregate, classification, and the replay LLM judge) plus the
gate verdict, with no network. Importing the plug-in module registers both
the custom adapter and the custom graders.
"""

import pathlib
import unittest

from evalcore import compare, loader, runner

SUITE_PATH = pathlib.Path(__file__).resolve().parents[1] / 'suite.yaml'


class QuickstartReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = loader.load_suite(SUITE_PATH)
        cls.baseline_run = runner.run_suite_sync(
            cls.suite, 'baseline', mode='replay'
        )
        cls.candidate_run = runner.run_suite_sync(
            cls.suite, 'candidate', mode='replay'
        )
        cls.baseline = cls.baseline_run.scorecard
        cls.candidate = cls.candidate_run.scorecard

    def test_dataset_loaded(self):
        cases = loader.load_cases(self.suite.dataset)
        self.assertEqual(len(cases), 4)

    def test_all_grader_families_present(self):
        m = self.candidate.metrics
        for metric in (
            'length_ok',  # deterministic
            'no_pii',  # deterministic
            'reply_present',  # deterministic
            'false_negative_rate',  # classification
            'acknowledges_customer',  # custom per-case
            'distinct_reply_rate',  # custom aggregate
            'quality.overall',  # LLM judge
        ):
            self.assertIn(metric, m)

    def test_candidate_is_clean_and_high_quality(self):
        m = self.candidate.metrics
        self.assertAlmostEqual(m['quality.overall'].value, 0.95, places=4)
        self.assertAlmostEqual(m['distinct_reply_rate'].value, 1.0)
        self.assertAlmostEqual(m['acknowledges_customer'].value, 1.0)
        self.assertAlmostEqual(m['false_negative_rate'].value, 0.0)
        self.assertAlmostEqual(m['recall'].value, 1.0)
        self.assertAlmostEqual(m['no_pii'].value, 1.0)

    def test_baseline_collapses_and_misroutes(self):
        m = self.baseline.metrics
        # Mode collapse: one generic reply across 4 cases -> 1/4 distinct.
        self.assertAlmostEqual(m['distinct_reply_rate'].value, 0.25)
        self.assertAlmostEqual(m['acknowledges_customer'].value, 0.0)
        # Misses the "billed twice" refund -> FN on the positive class.
        self.assertAlmostEqual(m['false_negative_rate'].value, 0.5)
        self.assertLess(
            m['quality.overall'].value,
            self.candidate.metrics['quality.overall'].value,
        )

    def test_gate_verdict_is_pass_and_improved(self):
        result = compare.compare(
            self.baseline, self.candidate, self.suite.thresholds
        )
        self.assertEqual(result.win, 'improved')
        self.assertEqual(result.verdict, 'pass')
        self.assertTrue(all(g.passed for g in result.guardrails))

    def test_flipped_direction_fails_guardrail(self):
        # Treating the weak baseline as the candidate breaches the FN-rate
        # guardrail (0.5 > 0.10 and increases), so the gate fails.
        flipped = compare.compare(
            self.candidate, self.baseline, self.suite.thresholds
        )
        self.assertEqual(flipped.verdict, 'fail')
        self.assertFalse(all(g.passed for g in flipped.guardrails))

    def test_live_stub_adapter_matches_replay(self):
        # mode='live' builds the offline stub adapter; the judge still replays,
        # so the whole path stays offline and must agree with pure replay.
        live = runner.run_suite_sync(
            self.suite, 'candidate', mode='live', grader_mode='replay'
        ).scorecard
        self.assertAlmostEqual(
            live.metrics['quality.overall'].value,
            self.candidate.metrics['quality.overall'].value,
            places=6,
        )


if __name__ == '__main__':
    unittest.main()
