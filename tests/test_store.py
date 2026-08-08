"""Store tests: row flattening, the JSONL outbox exporter, and round-trips."""

import json
import pathlib
import tempfile
import unittest

from evalcore import errors, graders, models, store


def _run(with_failure: bool = False):
    card = models.Scorecard(
        run_id='R',
        project='p',
        suite='s',
        variant=models.Variant(name='cand', knobs={'model': 'm1'}),
        dataset_version='v1',
        revision='sha1',
        mode='replay',
        created_at='2026-07-27 14:31:52',
        n_cases=1,
        n_samples=1,
        metrics={
            'f1': models.MetricValue(
                metric='f1', value=0.9, kind='aggregate', n=2
            ),
            'gap': models.MetricValue(
                metric='gap', value=None, kind='mean', n=0
            ),
        },
    )
    result = models.CaseResult(
        case=models.Case(id='c1', labels={'goal': 'x'}),
        variant_name='cand',
        sample_hash='h0',
        output=models.Output(
            fields={'v': 1},
            error=None,
            latency_ms=1500.0,
            tokens={'input_tokens': 10, 'output_tokens': 20, 'cache_read': 5},
        ),
        scores=[
            models.Score(
                grader='det',
                metric='passed_check',
                value=1.0,
                passed=True,
                detail='ok',
                case_id='c1',
            ),
            models.Score(
                grader='j',
                metric='quality.overall',
                value=None,
                passed=None,
                case_id='c1',
                judges=[
                    models.JudgeDetail(
                        key='strict',
                        version='v3',
                        rationale='thin',
                        points={'clarity': 3.0, 'visual': None},
                        overall=0.6,
                    )
                ],
            ),
        ],
    )
    results = [result]
    if with_failure:
        results.append(
            models.CaseResult(
                case=models.Case(id='c2'),
                variant_name='cand',
                sample_hash='h0',
                output=models.Output(error='boom', retryable=True),
            )
        )
    return models.RunResult(
        run_id='R',
        scorecard=card,
        results=results,
        aggregate_scores=[
            models.Score(
                grader='cls',
                metric='f1',
                value=0.9,
                detail='2 results',
                kind='aggregate',
            )
        ],
    )


def _comparison():
    return models.Comparison(
        project='p',
        suite='s',
        baseline_variant='base',
        candidate_variant='cand',
        win_metric='passed_check',
        win='improved',
        verdict='fail',
        deltas=[
            models.MetricDelta(
                metric='passed_check', baseline=0.5, candidate=1.0, delta=0.5
            )
        ],
        guardrails=[
            models.GuardrailResult(
                metric='f1', passed=False, detail='0.9000 < min 1.0'
            ),
            models.GuardrailResult(
                metric='never_scored',
                passed=False,
                detail='metric absent on candidate',
            ),
        ],
        summary='guardrail breach: f1 (0.9000 < min 1.0)',
    )


class RowTests(unittest.TestCase):
    def _by_metric(self, **kwargs):
        return {r['metric']: r for r in store.score_rows(_run(), **kwargs)}

    def test_missing_value_is_null(self):
        """An absent score is ``null``, not a filled-in 0.

        A real 0.0 is a meaningful score, so an absence has to be a different
        value rather than the same value plus a flag. No ``has_value`` sidecar
        column either. ``metric_kind`` keeps saying what kind of metric this
        is - the null is what says it went unscored - so a partially scored
        metric reads as one metric, not two.
        """
        rows = self._by_metric()
        unscored = rows['quality.overall']
        self.assertIsNone(unscored['value'])
        self.assertEqual(unscored['metric_kind'], 'per_case')
        self.assertNotIn('has_value', unscored)
        self.assertEqual(rows['passed_check']['value'], 1.0)
        self.assertEqual(rows['passed_check']['metric_kind'], 'per_case')

    def test_only_measurements_are_ever_null(self):
        """Nulls mean "no measurement", so no key column may carry one.

        An absent number is null; an absent *identity* is an empty string or an
        enum member, since the row shapes are read off those columns.
        """
        nullable = {
            'value',
            'duration',
            'win_baseline',
            'win_candidate',
            'win_delta',
        }
        rows = store.score_rows(
            _run(with_failure=True), _comparison(), baseline_run_id='B'
        )
        rows += store.score_rows(_run(with_failure=True))
        for row in rows:
            for column, value in row.items():
                if column in nullable or column.startswith('judges.'):
                    continue
                self.assertIsNotNone(
                    value, f'{column} is null on metric {row["metric"]!r}'
                )

    def test_a_real_zero_keeps_its_kind(self):
        """A genuine 0.0 is a score, not an absence: kind stays per_case."""
        run = _run()
        run.results[0].scores[0].value = 0.0
        run.results[0].scores[0].passed = False
        row = {r['metric']: r for r in store.score_rows(run)}['passed_check']
        self.assertEqual(row['value'], 0.0)
        self.assertEqual(row['metric_kind'], 'per_case')

    def test_tristate_passed(self):
        rows = self._by_metric()
        self.assertEqual(rows['passed_check']['passed'], 'true')
        self.assertEqual(rows['quality.overall']['passed'], 'null')
        self.assertEqual(rows['passed_check']['case_id'], 'c1')

    def test_keys_are_column_names(self):
        row = self._by_metric()['passed_check']
        self.assertEqual(row['application'], 'p')
        self.assertEqual(row['adapter_mode'], 'replay')
        self.assertEqual(row['started_at'], '2026-07-27 14:31:52')
        self.assertEqual(row['application_revision'], 'sha1')
        for absent in ('project', 'mode', 'created_at', 'revision'):
            self.assertNotIn(absent, row)

    def test_model_and_prompt_travel_in_knobs(self):
        row = self._by_metric()['passed_check']
        self.assertEqual(row['variant_knobs'], {'model': 'm1'})
        self.assertNotIn('model_id', row)
        self.assertNotIn('prompt_version', row)

    def test_metric_kind_is_the_engines_word(self):
        rows = self._by_metric()
        self.assertEqual(rows['passed_check']['metric_kind'], 'per_case')
        self.assertEqual(rows['f1']['metric_kind'], 'aggregate')

    def test_aggregate_row_has_no_case(self):
        row = self._by_metric()['f1']
        self.assertEqual(row['case_id'], '')
        self.assertEqual(row['grader'], 'cls')
        self.assertEqual(row['detail'], '2 results')

    def test_invocation_columns(self):
        row = self._by_metric()['passed_check']
        self.assertEqual(row['duration'], 1.5)
        self.assertFalse(row['is_error'])
        self.assertEqual(row['case_labels'], {'goal': 'x'})
        for absent in ('input_tokens', 'output_tokens', 'cost'):
            self.assertNotIn(absent, row)

    def test_failed_invocation_gets_a_row(self):
        rows = store.score_rows(_run(with_failure=True))
        failed = [r for r in rows if r['case_id'] == 'c2']
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]['grader'], '')
        self.assertEqual(failed[0]['metric'], '')
        self.assertEqual(failed[0]['metric_kind'], 'none')
        self.assertIsNone(failed[0]['value'])
        self.assertTrue(failed[0]['is_error'])
        self.assertTrue(failed[0]['retryable'])
        self.assertEqual(failed[0]['error_text'], 'boom')

    def test_judges_only_where_the_engine_puts_them(self):
        rows = self._by_metric()
        judged = rows['quality.overall']
        self.assertEqual(judged['judges.name'], ['strict'])
        # 'visual' went unscored, and stays in the map as a null.
        self.assertEqual(
            judged['judges.points'], [{'clarity': 3.0, 'visual': None}]
        )
        self.assertEqual(judged['judges.score'], [0.6])
        self.assertEqual(judged['judge_scale'], 0)
        self.assertEqual(rows['passed_check']['judges.name'], [])

    def test_judge_scale_from_the_lookup(self):
        rows = self._by_metric(judge_scales={'j': 5})
        self.assertEqual(rows['quality.overall']['judge_scale'], 5)
        self.assertEqual(rows['passed_check']['judge_scale'], 0)

    def test_grader_type_from_the_lookup(self):
        rows = self._by_metric(grader_types={'j': 'llm_as_judge'})
        self.assertEqual(
            rows['quality.overall']['grader_type'], 'llm_as_judge'
        )
        self.assertEqual(rows['passed_check']['grader_type'], 'unknown')

    def test_ungated_run_reads_as_none(self):
        row = self._by_metric()['passed_check']
        self.assertEqual(row['gate_verdict'], 'none')
        self.assertEqual(row['gate_win'], 'none')
        self.assertFalse(row['win'])
        self.assertEqual(row['guardrail'], 'none')
        # Null rather than 0: on a gated run a 0 delta means no change.
        self.assertIsNone(row['win_delta'])
        self.assertIsNone(row['win_baseline'])
        self.assertIsNone(row['win_candidate'])

    def test_gate_stamped_on_every_row(self):
        rows = store.score_rows(_run(), _comparison(), baseline_run_id='B')
        self.assertTrue(all(r['gate_verdict'] == 'fail' for r in rows))
        self.assertTrue(all(r['baseline_run_id'] == 'B' for r in rows))
        by_metric = {r['metric']: r for r in rows}
        self.assertEqual(by_metric['passed_check']['win_delta'], 0.5)
        self.assertEqual(by_metric['passed_check']['win_baseline'], 0.5)

    def test_guardrail_lands_on_its_metric(self):
        rows = {
            r['metric']: r for r in store.score_rows(_run(), _comparison())
        }
        self.assertEqual(rows['f1']['guardrail'], 'fail')
        self.assertEqual(rows['f1']['guardrail_gap'], '0.9000 < min 1.0')
        self.assertEqual(rows['passed_check']['guardrail'], 'none')
        self.assertTrue(rows['passed_check']['win'])

    def test_guardrail_on_an_unscored_metric_gets_a_row(self):
        rows = {
            r['metric']: r for r in store.score_rows(_run(), _comparison())
        }
        row = rows['never_scored']
        self.assertEqual(row['case_id'], '')
        self.assertEqual(row['grader'], '')
        self.assertEqual(row['metric_kind'], 'none')
        self.assertIsNone(row['value'])
        self.assertEqual(row['guardrail'], 'fail')
        self.assertEqual(row['guardrail_gap'], 'metric absent on candidate')


class GraderLookupTests(unittest.TestCase):
    def test_maps_registry_types_to_categories(self):
        types, scales = store.grader_lookups(
            [
                {'type': 'regex_present', 'name': 'has_cta'},
                {'type': 'classification', 'name': 'cls'},
                {'type': 'llm_judge', 'name': 'quality', 'scale': 7},
                {'type': 'llm_judge', 'name': 'tone'},
                {'type': 'consumer_thing', 'name': 'custom'},
            ]
        )
        self.assertEqual(types['has_cta'], 'heuristic')
        self.assertEqual(types['cls'], 'statistical')
        self.assertEqual(types['quality'], 'llm_as_judge')
        self.assertEqual(types['custom'], 'unknown')
        self.assertEqual(scales['quality'], 7)
        self.assertEqual(scales['tone'], 5)
        self.assertNotIn('has_cta', scales)

    def test_falls_back_to_the_type_when_unnamed(self):
        types, _ = store.grader_lookups([{'type': 'non_empty'}])
        self.assertEqual(types['non_empty'], 'heuristic')

    def test_a_plugin_is_categorised_like_a_built_in(self):
        """The point of the registry lookup: a consumer grader that declares
        its category reads back as that category, not as 'unknown'.
        """

        @graders.base.register('_store_plugin', graders.GraderType.HEURISTIC)
        class _Plugin:
            name = '_store_plugin'

            def grade(self, case, output):  # pragma: no cover - not run
                return []

        types, _ = store.grader_lookups([{'type': '_store_plugin'}])
        self.assertEqual(types['_store_plugin'], 'heuristic')

    def test_a_type_cannot_register_twice(self):
        with self.assertRaises(errors.ConfigError):

            @graders.base.register('non_empty', graders.GraderType.HEURISTIC)
            class _Clash:
                name = 'non_empty'


class ScoreExporterProtocolTests(unittest.TestCase):
    """The seam another package implements to publish to a real store."""

    def test_the_builtin_exporter_satisfies_it(self):
        exporter = store.JsonlOutboxExporter('/dev/null')
        self.assertIsInstance(exporter, store.ScoreExporter)

    def test_an_outside_implementation_satisfies_it(self):
        """A store-side exporter: same method, its own transport."""

        class Collecting:
            def __init__(self):
                self.rows = []

            def export_scores(self, run, comparison=None, **kwargs):
                self.rows = store.score_rows(run, comparison, **kwargs)
                return len(self.rows)

        exporter = Collecting()
        self.assertIsInstance(exporter, store.ScoreExporter)
        self.assertEqual(exporter.export_scores(_run()), len(exporter.rows))

    def test_missing_the_method_does_not(self):
        class NotAnExporter:
            def export(self, run):  # pragma: no cover - never called
                return 0

        self.assertNotIsInstance(NotAnExporter(), store.ScoreExporter)


class ExporterTests(unittest.TestCase):
    def test_export_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'nested' / 'outbox.jsonl'
            exporter = store.JsonlOutboxExporter(path)
            written = exporter.export_scores(_run(), _comparison())
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), written)
            self.assertTrue(all(json.loads(line) for line in lines))
            self.assertTrue(
                all(
                    json.loads(line)['gate_verdict'] == 'fail'
                    for line in lines
                )
            )


class RoundTripTests(unittest.TestCase):
    def test_scorecard_and_comparison_and_ratings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            run = _run()
            store.write_scorecard(root / 'sc.json', run.scorecard)
            self.assertEqual(
                store.read_scorecard(root / 'sc.json').revision, 'sha1'
            )
            # load_scorecard accepts a full-run file too.
            store.write_run(root / 'run.json', run)
            self.assertEqual(
                store.load_scorecard(root / 'run.json').run_id, 'R'
            )

            comparison = models.Comparison(
                project='p',
                suite='s',
                baseline_variant='b',
                candidate_variant='c',
                summary='ok',
            )
            store.write_comparison(root / 'cmp.json', comparison)
            self.assertIn('"verdict"', (root / 'cmp.json').read_text())

    def test_read_ratings_missing_file_is_empty(self):
        self.assertEqual(store.read_ratings('/no/such/file.jsonl'), [])

    def test_preferences_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                pathlib.Path(tmp) / 'sub' / 'prefs.jsonl'
            )  # dir auto-created
            store.append_preference(
                path,
                models.Preference(
                    case_id='c1',
                    variant_a='a',
                    variant_b='b',
                    rater='r1',
                    winner='a',
                    dims={'visual': 'tie'},
                ),
            )
            store.append_preference(
                path,
                models.Preference(
                    case_id='c2',
                    variant_a='a',
                    variant_b='b',
                    rater='r1',
                    winner='b',
                ),
            )
            saved = store.read_preferences(path)
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved[0].winner, 'a')
            self.assertEqual(saved[0].dims, {'visual': 'tie'})
            self.assertEqual(saved[1].winner, 'b')

    def test_read_preferences_missing_file_is_empty(self):
        self.assertEqual(store.read_preferences('/no/such/file.jsonl'), [])

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'sub' / 'ck.jsonl'  # dir auto-created
            store.init_checkpoint(path, {'run_id': 'R', 'suite_hash': 'H'})
            self.assertEqual(store.checkpoint_meta(path)['run_id'], 'R')
            self.assertEqual(store.read_checkpoint_samples(path), [])
            for ordinal, cid in enumerate(('c1', 'c2')):
                store.append_checkpoint_result(
                    path,
                    models.CaseResult(
                        case=models.Case(id=cid),
                        variant_name='v',
                        sample_hash='h0',
                        output=models.Output(fields={'x': cid}),
                    ),
                    ordinal,
                )
            entries = store.read_checkpoint_samples(path)
            self.assertEqual([r.case.id for _, r in entries], ['c1', 'c2'])
            # The ordinal rides on the line, not on the CaseResult, so resume
            # can tell which samples are still owed.
            self.assertEqual([o for o, _ in entries], [0, 1])
            # meta survives the appends (still line 1)
            self.assertEqual(store.checkpoint_meta(path)['suite_hash'], 'H')

    def test_checkpoint_meta_absent_file_is_none(self):
        self.assertIsNone(store.checkpoint_meta('/no/such/ck.jsonl'))


if __name__ == '__main__':
    unittest.main()
