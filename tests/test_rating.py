"""Unit tests for the human-rating loop and judge<->human agreement."""

import pathlib
import tempfile
import unittest

from evalcore import models, rating, store

# The sample hash every fixture below shares. Agreement joins a run's results
# to the ratings on (case_id, sample_hash), so both sides have to name the
# same sample - in a real run this is the digest of the output the human saw.
_HASH = 'h0'


def _rating(case_id, rater, points, run_id='R', sample_hash=_HASH):
    """A visual_design rating of the sample :func:`_run` built for a case."""
    return models.Rating(
        run_id=run_id,
        case_id=case_id,
        sample_hash=sample_hash,
        rater=rater,
        scores={'visual_design': points},
    )


def _run(pairs, run_id='R'):
    """Build a RunResult with a quality.visual_design judge score per case."""
    results = []
    for case_id, judge_value in pairs:
        results.append(
            models.CaseResult(
                case=models.Case(id=case_id),
                variant_name='v',
                sample_hash=_HASH,
                output=models.Output(fields={'html': f'<i>{case_id}</i>'}),
                scores=[
                    models.Score(
                        grader='quality',
                        metric='quality.visual_design',
                        value=judge_value,
                        case_id=case_id,
                        kind='per_case',
                    )
                ],
            )
        )
    scorecard = models.Scorecard(
        run_id=run_id,
        project='p',
        suite='s',
        variant=models.Variant(name='v'),
        dataset_version='v1',
    )
    return models.RunResult(
        run_id=run_id, scorecard=scorecard, results=results
    )


class AgreementTests(unittest.TestCase):
    def test_mae_and_correlation(self):
        run = _run([('c1', 0.8), ('c2', 0.4)])
        ratings = [_rating('c1', 'a', 4), _rating('c2', 'a', 3)]
        result = rating.compute_agreement(
            run, ratings, ['visual_design'], judge_name='quality', scale=5
        )
        dim = result.dimensions[0]
        self.assertEqual(dim.n, 2)
        self.assertAlmostEqual(dim.human_mean, 0.7)  # (0.8 + 0.6) / 2
        self.assertAlmostEqual(dim.judge_mean, 0.6)  # (0.8 + 0.4) / 2
        self.assertAlmostEqual(dim.mae, 0.1)  # mean(|0|, |0.2|)
        self.assertAlmostEqual(dim.correlation, 1.0)  # both descend
        self.assertEqual(result.n_ratings, 2)
        self.assertEqual(result.n_raters, 1)

    def test_multiple_raters_are_averaged(self):
        run = _run([('c1', 0.8)])
        ratings = [_rating('c1', 'a', 5), _rating('c1', 'b', 3)]
        result = rating.compute_agreement(
            run, ratings, ['visual_design'], scale=5
        )
        # human mean = mean(1.0, 0.6) = 0.8, judge 0.8 -> MAE 0.
        self.assertAlmostEqual(result.dimensions[0].human_mean, 0.8)
        self.assertAlmostEqual(result.dimensions[0].mae, 0.0)
        self.assertEqual(result.n_raters, 2)

    def test_ratings_for_other_runs_ignored(self):
        run = _run([('c1', 0.8)], run_id='R')
        ratings = [_rating('c1', 'a', 1, run_id='OTHER')]
        result = rating.compute_agreement(
            run, ratings, ['visual_design'], scale=5
        )
        self.assertEqual(result.n_ratings, 0)
        self.assertEqual(result.dimensions[0].n, 0)
        self.assertIsNone(result.dimensions[0].mae)


class BlindQueueTests(unittest.TestCase):
    def _app(self, ratings_path):
        run = _run([('c1', 0.8), ('c2', 0.4)])
        return rating._RatingApp(
            [run],
            ratings_path,
            ['visual_design'],
            5,
            content_ref='output.html',
            screenshot_ref='artifacts.screenshot',
        )

    def test_queue_leaks_no_identity(self):
        app = self._app('/dev/null')
        payload = app.queue_for('rater-x')
        self.assertEqual(len(payload), 2)
        for item in payload:
            # Only opaque id + safe presentation fields reach the browser.
            self.assertEqual(set(item), {'id', 'input', 'views'})
            self.assertNotIn('run_id', item)
            self.assertNotIn('variant', item)
            self.assertNotIn('case_id', item)
            # Panels carry render kinds/values, never server file paths.
            self.assertIsInstance(item['views'], list)
            for view in item['views']:
                self.assertIn('kind', view)
                self.assertNotIn('files', view)

    def test_text_only_output_falls_back_to_fields(self):
        # No content_ref/screenshot_ref and no artifacts: a plain-text
        # field still yields a reviewable panel.
        run = _run([('c1', 0.8)])
        app = rating._RatingApp([run], '/dev/null', ['d'], 5)
        views = app.queue_for('r')[0]['views']
        self.assertEqual(len(views), 1)
        self.assertEqual(
            views[0]['kind'], 'html'
        )  # '<i>c1</i>' looks like html
        self.assertEqual(views[0]['value'], '<i>c1</i>')

    def test_seeded_shuffle_is_stable_per_rater(self):
        app = self._app('/dev/null')
        first = [i['id'] for i in app.queue_for('cv')]
        again = [i['id'] for i in app.queue_for('cv')]
        self.assertEqual(first, again)

    def test_record_writes_rating_mapped_back_to_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'ratings.jsonl'
            app = self._app(str(path))
            app.record(0, 'cv', {'visual_design': 4})
            saved = store.read_ratings(path)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].run_id, 'R')
            self.assertEqual(saved[0].case_id, 'c1')
            self.assertEqual(saved[0].rater, 'cv')
            self.assertEqual(saved[0].scores, {'visual_design': 4})


def _pref(
    case_id,
    winner,
    *,
    va='baseline',
    vb='candidate',
    rater='r1',
    dims=None,
    sample_hash=_HASH,
):
    return models.Preference(
        case_id=case_id,
        sample_hash=sample_hash,
        variant_a=va,
        variant_b=vb,
        rater=rater,
        winner=winner,
        dims=dims or {},
    )


def _ab_runs(cases=('c1', 'c2')):
    def one(name):
        return models.RunResult(
            run_id=name,
            scorecard=models.Scorecard(
                run_id=name,
                project='p',
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
                for c in cases
            ],
        )

    return one('baseline'), one('candidate')


class TallyTests(unittest.TestCase):
    def test_tally_counts_ties_as_half(self):
        a, b, t, wr = rating._tally(['a', 'a', 'b', 'tie'])
        self.assertEqual((a, b, t), (2, 1, 1))
        self.assertAlmostEqual(wr, (2 + 0.5) / 4)

    def test_tally_empty_is_none(self):
        self.assertEqual(rating._tally([]), (0, 0, 0, None))

    def test_majority_winner_plurality_and_tie(self):
        self.assertEqual(rating._majority_winner(['a', 'a', 'b']), 'a')
        self.assertEqual(rating._majority_winner(['a', 'b']), 'tie')
        self.assertEqual(rating._majority_winner(['tie', 'tie', 'a']), 'tie')


class AggregatePreferencesTests(unittest.TestCase):
    def test_overall_and_per_dimension_win_rate(self):
        run_a, run_b = _ab_runs()
        prefs = [
            _pref('c1', 'a', dims={'visual': 'a', 'copy': 'tie'}),
            _pref('c2', 'b', dims={'visual': 'a', 'copy': 'b'}),
        ]
        result = rating.aggregate_preferences(
            run_a, run_b, prefs, ['visual', 'copy']
        )
        self.assertEqual(
            (result.a_wins, result.b_wins, result.ties), (1, 1, 0)
        )
        self.assertAlmostEqual(result.win_rate_a, 0.5)
        self.assertEqual(result.n_raters, 1)
        visual = next(d for d in result.dimensions if d.dimension == 'visual')
        self.assertAlmostEqual(visual.win_rate_a, 1.0)  # a beat b on both
        copy = next(d for d in result.dimensions if d.dimension == 'copy')
        self.assertAlmostEqual(copy.win_rate_a, 0.25)  # one tie, one b

    def test_only_matching_orientation_counted(self):
        run_a, run_b = _ab_runs()
        prefs = [
            _pref('c1', 'a'),  # baseline vs candidate - counts
            _pref('c1', 'a', va='candidate', vb='baseline'),  # flipped - not
        ]
        result = rating.aggregate_preferences(run_a, run_b, prefs)
        self.assertEqual(result.n, 1)

    def test_dimensions_default_to_all_seen(self):
        run_a, run_b = _ab_runs()
        prefs = [_pref('c1', 'a', dims={'visual': 'a', 'copy': 'b'})]
        result = rating.aggregate_preferences(run_a, run_b, prefs)
        self.assertEqual(
            {d.dimension for d in result.dimensions}, {'visual', 'copy'}
        )

    def test_empty_preferences_win_rate_none(self):
        run_a, run_b = _ab_runs()
        result = rating.aggregate_preferences(run_a, run_b, [])
        self.assertEqual(result.n, 0)
        self.assertIsNone(result.win_rate_a)


class PairwiseAgreementTests(unittest.TestCase):
    def _pairwise(self, outcomes):
        return models.PairwiseResult(
            project='p',
            suite='s',
            variant_a='baseline',
            variant_b='candidate',
            judge_name='pairwise',
            outcomes=[
                # Same hash the preferences carry: both sides anchor on the
                # variant_a output, which is what lets them join.
                models.PairwiseOutcome(case_id=c, sample_hash=_HASH, winner=w)
                for c, w in outcomes
            ],
        )

    def test_human_majority_vs_judge(self):
        prefs = [
            _pref('c1', 'a', rater='r1'),
            _pref('c1', 'a', rater='r2'),
            _pref('c1', 'b', rater='r3'),  # majority a
            _pref('c2', 'b', rater='r1'),  # sole vote b
        ]
        pw = self._pairwise([('c1', 'a'), ('c2', 'a')])
        result = rating.compute_pairwise_agreement(prefs, pw)
        self.assertEqual(result.n, 2)
        self.assertEqual(result.agree, 1)  # c1 agrees, c2 disagrees
        self.assertAlmostEqual(result.agreement_rate, 0.5)
        self.assertAlmostEqual(result.human_win_rate_a, 0.5)  # a, b
        self.assertAlmostEqual(result.judge_win_rate_a, 1.0)  # a, a

    def test_only_cases_in_both_are_scored(self):
        prefs = [_pref('c1', 'a')]  # no c2 pref
        pw = self._pairwise([('c1', 'a'), ('c2', 'b')])
        result = rating.compute_pairwise_agreement(prefs, pw)
        self.assertEqual(result.n, 1)


if __name__ == '__main__':
    unittest.main()
