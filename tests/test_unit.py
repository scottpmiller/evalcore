"""Unit tests for the generic engine pieces."""

import pathlib
import unittest
from unittest import mock

from evalcore import compare, models, refs, store
from evalcore.adapters import expand_env
from evalcore.graders import base, classification, deterministic, numeric


def _card(name, **metric_values):
    return models.Scorecard(
        project='p',
        suite='s',
        variant=models.Variant(name=name),
        dataset_version='v1',
        metrics={
            m: models.MetricValue(metric=m, value=v, kind='aggregate', n=1)
            for m, v in metric_values.items()
        },
    )


class RefsTests(unittest.TestCase):
    def test_resolve_path_dict_list_and_missing(self):
        obj = {'a': {'b': [{'c': 7}]}}
        self.assertEqual(refs.resolve_path(obj, 'a.b.0.c'), 7)
        self.assertIsNone(refs.resolve_path(obj, 'a.b.9.c'))
        self.assertIsNone(refs.resolve_path(obj, 'nope'))

    def test_build_value_substitutes_refs(self):
        ctx = {'input': {'x': 'hi'}, 'variant': {'model': 'm1'}}
        template = {'a': '$input.x', 'b': '$variant.model', 'c': 'literal'}
        self.assertEqual(
            refs.build_value(template, ctx),
            {'a': 'hi', 'b': 'm1', 'c': 'literal'},
        )


class DeterministicGraderTests(unittest.TestCase):
    def test_max_chars(self):
        grader = deterministic.MaxChars(field='output.text', maximum=5)
        case = models.Case(id='c')
        long = models.Output(fields={'text': 'toolong'})
        short = models.Output(fields={'text': 'ok'})
        self.assertFalse(grader.grade(case, long)[0].passed)
        self.assertTrue(grader.grade(case, short)[0].passed)

    def test_regex_absent(self):
        grader = deterministic.RegexAbsent(
            field='output.text', pattern=r'\{\{.*?\}\}'
        )
        case = models.Case(id='c')
        token = models.Output(fields={'text': 'Hi {{first_name}}'})
        clean = models.Output(fields={'text': 'Hi there'})
        self.assertFalse(grader.grade(case, token)[0].passed)
        self.assertTrue(grader.grade(case, clean)[0].passed)

    def test_regex_present_all_of(self):
        grader = deterministic.RegexPresent(
            field='output.html',
            patterns=[r'type="email"', r'action="[^"]*addlead\.pl"'],
        )
        case = models.Case(id='c')
        wired = models.Output(
            fields={
                'html': '<form action="x/addlead.pl">'
                '<input type="EMAIL"></form>'
            }
        )
        partial = models.Output(fields={'html': '<input type="email">'})
        self.assertTrue(
            grader.grade(case, wired)[0].passed
        )  # case-insensitive
        result = grader.grade(case, partial)[0]
        self.assertFalse(result.passed)
        self.assertIn('addlead', result.detail)
        # single-pattern shorthand
        one = deterministic.RegexPresent(field='output.html', pattern=r'<form')
        self.assertTrue(one.grade(case, wired)[0].passed)


class ClassificationGraderTests(unittest.TestCase):
    def _result(self, predicted, actual):
        return models.CaseResult(
            case=models.Case(id='x', expected={'label': actual}),
            variant_name='v',
            sample_hash='h0',
            output=models.Output(fields={'verdict': predicted}),
        )

    def test_confusion_metrics(self):
        grader = classification.Classification(
            predicted_ref='output.verdict',
            expected_ref='expected.label',
            positive_labels=['malicious', 'suspicious'],
            negative_labels=['ok'],
        )
        results = [
            self._result('malicious', 'malicious'),  # TP
            self._result('ok', 'malicious'),  # FN
            self._result('suspicious', 'ok'),  # FP
            self._result('ok', 'ok'),  # TN
        ]
        scores = {s.metric: s.value for s in grader.aggregate(results)}
        self.assertAlmostEqual(scores['precision'], 0.5)
        self.assertAlmostEqual(scores['recall'], 0.5)
        self.assertAlmostEqual(scores['f1'], 0.5)
        self.assertAlmostEqual(scores['false_negative_rate'], 0.5)
        self.assertAlmostEqual(scores['false_positive_rate'], 0.5)


class CompareTests(unittest.TestCase):
    def _thresholds(self):
        return {
            'win_metric': 'f1',
            'win_min_delta': 0.01,
            'guardrails': [
                {
                    'metric': 'false_negative_rate',
                    'max': 0.10,
                    'must_not_increase': True,
                }
            ],
        }

    def test_improvement_passes(self):
        base_card = _card('baseline', f1=0.8, false_negative_rate=0.2)
        cand = _card('candidate', f1=1.0, false_negative_rate=0.0)
        result = compare.compare(base_card, cand, self._thresholds())
        self.assertEqual(result.verdict, 'pass')
        self.assertEqual(result.win, 'improved')

    def test_guardrail_breach_fails(self):
        base_card = _card('baseline', f1=0.8, false_negative_rate=0.0)
        cand = _card('candidate', f1=0.9, false_negative_rate=0.2)
        result = compare.compare(base_card, cand, self._thresholds())
        self.assertEqual(result.verdict, 'fail')
        self.assertFalse(result.guardrails[0].passed)

    def test_regression_warns(self):
        base_card = _card('baseline', f1=0.9, false_negative_rate=0.0)
        cand = _card('candidate', f1=0.5, false_negative_rate=0.0)
        result = compare.compare(base_card, cand, self._thresholds())
        self.assertEqual(result.verdict, 'warn')
        self.assertEqual(result.win, 'regressed')


class MetricDirectionTests(unittest.TestCase):
    """A delta's sign and its meaning are two different questions."""

    def _deltas(self, thresholds, **moves):
        """Move each metric from its first value to its second."""
        base = _card('baseline', **{m: v[0] for m, v in moves.items()})
        cand = _card('candidate', **{m: v[1] for m, v in moves.items()})
        result = compare.compare(base, cand, thresholds)
        return {d.metric: d for d in result.deltas}

    def test_ceiling_rule_means_lower_is_better(self):
        # Nobody writes a ceiling for a metric they want to go up, so an
        # existing suite states its directions without having to be edited.
        deltas = self._deltas(
            {'guardrails': [{'metric': 'errors', 'max': 3}]}, errors=(5.0, 1.0)
        )
        self.assertFalse(deltas['errors'].higher_is_better)
        self.assertEqual(deltas['errors'].direction, 'improved')

    def test_must_not_increase_means_lower_is_better(self):
        deltas = self._deltas(
            {'guardrails': [{'metric': 'fn_rate', 'must_not_increase': True}]},
            fn_rate=(0.1, 0.3),
        )
        self.assertEqual(deltas['fn_rate'].direction, 'regressed')

    def test_floor_rule_means_higher_is_better(self):
        deltas = self._deltas(
            {'guardrails': [{'metric': 'no_pii', 'min': 1.0}]},
            no_pii=(0.5, 1.0),
        )
        self.assertTrue(deltas['no_pii'].higher_is_better)
        self.assertEqual(deltas['no_pii'].direction, 'improved')

    def test_metric_fenced_on_both_sides_has_no_direction(self):
        # A band says neither way is the good way. Falling back to the
        # higher-is-better default here would be a confident wrong answer.
        deltas = self._deltas(
            {'guardrails': [{'metric': 'length', 'min': 0.2, 'max': 0.8}]},
            length=(0.3, 0.7),
        )
        self.assertIsNone(deltas['length'].higher_is_better)
        self.assertEqual(deltas['length'].direction, 'neutral')

    def test_declaration_beats_inference(self):
        deltas = self._deltas(
            {
                'metrics': {'cost': 'lower_is_better'},
                'guardrails': [{'metric': 'cost', 'min': 0.0}],
            },
            cost=(2.0, 1.0),
        )
        self.assertFalse(deltas['cost'].higher_is_better)
        self.assertEqual(deltas['cost'].direction, 'improved')

    def test_declared_neutral_stays_silent(self):
        deltas = self._deltas(
            {'metrics': {'tool_calls': 'neutral'}}, tool_calls=(10.0, 40.0)
        )
        self.assertIsNone(deltas['tool_calls'].higher_is_better)
        self.assertEqual(deltas['tool_calls'].direction, 'neutral')

    def test_unrecognised_word_is_ignored_not_guessed(self):
        deltas = self._deltas(
            {'metrics': {'score': 'sideways'}}, score=(0.5, 0.9)
        )
        self.assertTrue(deltas['score'].higher_is_better)

    def test_metric_nothing_speaks_to_defaults_to_higher_is_better(self):
        deltas = self._deltas({}, mystery=(0.5, 0.9))
        self.assertTrue(deltas['mystery'].higher_is_better)
        self.assertEqual(deltas['mystery'].direction, 'improved')

    def test_win_higher_is_better_false_reaches_the_delta_row(self):
        thresholds = {'win_metric': 'cost', 'win_higher_is_better': False}
        base = _card('baseline', cost=2.0)
        cand = _card('candidate', cost=1.0)
        result = compare.compare(base, cand, thresholds)
        self.assertEqual(result.win, 'improved')
        # The headline verdict and the metric's own row are one call, so
        # they cannot disagree about the same number.
        self.assertEqual(result.deltas[0].direction, 'improved')

    def test_dead_band_applies_to_the_win_metric_only(self):
        thresholds = {'win_metric': 'f1', 'win_min_delta': 0.05}
        deltas = self._deltas(thresholds, f1=(0.80, 0.82), other=(0.80, 0.82))
        self.assertEqual(deltas['f1'].direction, 'neutral')
        self.assertEqual(deltas['other'].direction, 'improved')


class NumericGraderTests(unittest.TestCase):
    def _grade(self, fields, output_fields):
        grader = numeric.Numeric(fields=fields)
        case = models.Case(id='c')
        scores = grader.grade(case, models.Output(fields=output_fields))
        return {s.metric: s for s in scores}

    def test_promotes_field_to_measurement_metric(self):
        scores = self._grade(['output.cost'], {'cost': 1.5})
        self.assertEqual(scores['cost'].value, 1.5)
        # No bound -> a pure measurement, no pass/fail.
        self.assertIsNone(scores['cost'].passed)

    def test_max_bound_sets_pass_fail(self):
        spec = [{'ref': 'output.err', 'max': 0.2}]
        self.assertTrue(self._grade(spec, {'err': 0.1})['err'].passed)
        self.assertFalse(self._grade(spec, {'err': 0.3})['err'].passed)

    def test_min_bound_and_name_override(self):
        spec = [{'ref': 'output.n', 'min': 10, 'name': 'volume'}]
        scores = self._grade(spec, {'n': 4})
        self.assertIn('volume', scores)
        self.assertEqual(scores['volume'].value, 4.0)
        self.assertFalse(scores['volume'].passed)

    def test_absent_field_degrades(self):
        # Unbounded absent -> value None, passed None.
        unbounded = self._grade(['output.missing'], {})['missing']
        self.assertIsNone(unbounded.value)
        self.assertIsNone(unbounded.passed)
        # Bounded absent -> a fail, not a silent pass.
        bounded = self._grade([{'ref': 'output.missing', 'max': 1}], {})
        self.assertFalse(bounded['missing'].passed)

    def test_non_numeric_is_not_coerced(self):
        scores = self._grade(['output.label'], {'label': 'malicious'})
        self.assertIsNone(scores['label'].value)


class EnvExpansionTests(unittest.TestCase):
    def test_expands_recursively_through_dict_and_list(self):
        env = {'A': 'aval', 'B': 'bval'}
        template = {
            'plain': 'no-vars',
            'one': '${A}',
            'nested': ['${A}', {'deep': '${B}'}],
            'number': 5,
        }
        with mock.patch.dict('os.environ', env):
            expanded = expand_env(template)
        self.assertEqual(
            expanded,
            {
                'plain': 'no-vars',
                'one': 'aval',
                'nested': ['aval', {'deep': 'bval'}],
                'number': 5,
            },
        )

    def test_unset_var_expands_to_empty(self):
        with mock.patch.dict('os.environ', {}, clear=True):
            self.assertEqual(expand_env('${MISSING}'), '')


class RegistryTests(unittest.TestCase):
    def test_unknown_grader_type_raises(self):
        with self.assertRaises(ValueError):
            base.build_graders([{'type': 'does-not-exist'}])


class StoreLoadTests(unittest.TestCase):
    """load_scorecard reads either a scorecard or a full-run file."""

    def test_loads_scorecard_and_run_files(self):
        import tempfile

        card = _card('candidate', f1=0.9)
        run = models.RunResult(run_id='r1', scorecard=card, results=[])
        with tempfile.TemporaryDirectory() as tmp:
            sc_path = pathlib.Path(tmp) / 'c.scorecard.json'
            run_path = pathlib.Path(tmp) / 'c.run.json'
            store.write_scorecard(sc_path, card)
            store.write_run(run_path, run)
            self.assertEqual(store.load_scorecard(sc_path), card)
            self.assertEqual(store.load_scorecard(run_path), card)


if __name__ == '__main__':
    unittest.main()
