"""Tests for the pluggable report seam (markdown + html built-ins)."""

import base64
import pathlib
import tempfile
import unittest

from evalcore import models, reporters
from evalcore.reporters import base


def _card(name='cand', **metrics):
    return models.Scorecard(
        project='p',
        suite='s',
        variant=models.Variant(name=name),
        dataset_version='v1',
        mode='replay',
        n_cases=2,
        metrics={
            m: models.MetricValue(metric=m, value=v, kind='aggregate', n=1)
            for m, v in metrics.items()
        },
    )


def _comparison():
    return models.Comparison(
        project='p',
        suite='s',
        baseline_variant='baseline',
        candidate_variant='candidate',
        win_metric='f1',
        win='improved',
        verdict='pass',
        deltas=[
            models.MetricDelta(
                metric='f1', baseline=0.8, candidate=1.0, delta=0.2
            )
        ],
        guardrails=[
            models.GuardrailResult(
                metric='error_rate', passed=True, detail='0.0000 ok'
            )
        ],
        summary='f1 improved',
    )


class RegistryTests(unittest.TestCase):
    def test_builtins_registered(self):
        self.assertIsInstance(
            reporters.build_reporter('markdown'), base.Reporter
        )
        self.assertIsInstance(reporters.build_reporter('html'), base.Reporter)

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            reporters.build_reporter('does-not-exist')

    def test_dict_spec_form(self):
        # {type: ...} spec form works like adapters/graders.
        self.assertIsInstance(
            reporters.build_reporter({'type': 'html'}), base.Reporter
        )

    def test_duplicate_registration_raises(self):
        with self.assertRaises(ValueError):

            @base.register('markdown')
            class _Dupe:
                pass


class MarkdownReporterTests(unittest.TestCase):
    def test_scorecard_and_comparison_are_markdown(self):
        rep = reporters.build_reporter('markdown')
        sc = rep.scorecard(_card(f1=1.0))
        self.assertIn('| metric | value |', sc)
        self.assertIn('p/s', sc)
        cmp_ = rep.comparison(_comparison())
        self.assertIn('PASS', cmp_)
        # document is identity for markdown
        self.assertEqual(reporters.wrap_document(rep, sc), sc)


class HtmlReporterTests(unittest.TestCase):
    def setUp(self):
        self.rep = reporters.build_reporter('html')

    def test_scorecard_fragment_and_document(self):
        frag = self.rep.scorecard(_card(f1=1.0))
        self.assertIn('<table>', frag)
        self.assertIn('1.0000', frag)
        self.assertNotIn('<!doctype', frag)  # fragment, not a full doc
        doc = reporters.wrap_document(self.rep, frag)
        self.assertTrue(doc.startswith('<!doctype html'))
        self.assertIn('<style>', doc)

    def test_comparison_has_verdict_and_deltas(self):
        frag = self.rep.comparison(_comparison())
        self.assertIn('verdict pass', frag)  # colored badge class
        self.assertIn('+0.2000', frag)  # signed delta
        self.assertIn('error_rate', frag)  # guardrail row

    def test_escapes_values(self):
        frag = self.rep.scorecard(_card(name='<x&y>'))
        self.assertNotIn('<x&y>', frag)
        self.assertIn('&lt;x&amp;y&gt;', frag)


def _run_result():
    def case_result(cid, acc, passed, error=None):
        return models.CaseResult(
            case=models.Case(id=cid),
            variant_name='cand',
            sample_hash='h0',
            output=models.Output(
                fields={'html': f'<form>{cid}</form>'}, error=error
            ),
            scores=[
                models.Score(
                    grader='wiring',
                    metric='wiring_complete',
                    value=acc,
                    passed=passed,
                    detail='missing [meta_adtracking]' if not passed else 'ok',
                    case_id=cid,
                    kind='per_case',
                )
            ],
        )

    return models.RunResult(
        run_id='R',
        scorecard=_card('cand', wiring_complete=0.5),
        results=[
            case_result('c1', 1.0, True),
            case_result('c2', 0.0, False),
            case_result('c3', None, None, error='boom'),
        ],
    )


class RunDetailTests(unittest.TestCase):
    def test_markdown_run_has_per_case_matrix_and_notes(self):
        md = reporters.build_reporter('markdown').run(_run_result())
        self.assertIn('#### Per-case', md)
        self.assertIn('wiring_complete', md)
        self.assertIn('c1', md)
        self.assertIn('c2', md)
        self.assertIn('**Notes**', md)
        # the failing case's "why" is surfaced, not just its number
        self.assertIn('missing [meta_adtracking]', md)
        self.assertIn('errored: boom', md)

    def test_html_run_has_per_case_table_and_outputs(self):
        rep = reporters.build_reporter('html')
        # run() now yields a composable fragment; the document wrap happens
        # once, in render_run.
        frag = rep.run(_run_result())
        self.assertNotIn('<!doctype', frag)
        html = base.render_run(rep, _run_result())
        self.assertTrue(html.startswith('<!doctype html'))
        self.assertIn('Per-case', html)
        self.assertIn('class="num fail"', html)  # the failing cell is flagged
        self.assertIn('<details>', html)  # collapsible per-case output
        self.assertIn('&lt;form&gt;c1&lt;/form&gt;', html)  # escaped output

    def test_render_run_falls_back_without_run_method(self):
        class _Bare:
            def scorecard(self, sc):
                return 'SC'

            def comparison(self, cmp):
                return 'CMP'

        self.assertEqual(base.render_run(_Bare(), _run_result()), 'SC')


# A 1x1 transparent PNG - the smallest real image to round-trip as a data URI.
_PNG = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk'
    'YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
)


class ArtifactEmbedTests(unittest.TestCase):
    """The html ``run`` report embeds saved artifacts inline (self-contained).

    Everything the run produced in the results dir - screenshots, rendered
    HTML - lands in the document, not just the artifact names.
    """

    def _run_with_artifacts(self, artifacts):
        result = models.CaseResult(
            case=models.Case(id='c1'),
            variant_name='cand',
            sample_hash='h0',
            output=models.Output(
                fields={'html': '<form>c1</form>'}, artifacts=artifacts or {}
            ),
            scores=[
                models.Score(
                    grader='wiring',
                    metric='wiring_complete',
                    value=1.0,
                    passed=True,
                    case_id='c1',
                    kind='per_case',
                )
            ],
        )
        return models.RunResult(
            run_id='R',
            scorecard=_card('cand', wiring_complete=1.0),
            results=[result],
        )

    def test_image_and_html_artifacts_embed_inline(self):
        with tempfile.TemporaryDirectory() as d:
            shot = pathlib.Path(d) / 'shot.png'
            shot.write_bytes(_PNG)
            form = pathlib.Path(d) / 'form.html'
            form.write_text('<h1>rendered form</h1>')
            html = reporters.build_reporter('html').run(
                self._run_with_artifacts(
                    {'screenshot': str(shot), 'html': str(form)}
                )
            )
        b64 = base64.b64encode(_PNG).decode('ascii')
        self.assertIn(f'data:image/png;base64,{b64}', html)  # image inlined
        self.assertIn('srcdoc=', html)  # html rendered in a sandboxed iframe
        self.assertIn('&lt;h1&gt;rendered form&lt;/h1&gt;', html)  # escaped
        self.assertIn('sandbox', html)

    def test_missing_artifact_degrades_to_a_note(self):
        html = reporters.build_reporter('html').run(
            self._run_with_artifacts({'screenshot': '/nope/gone.png'})
        )
        self.assertIn('missing:', html)
        self.assertIn('/nope/gone.png', html)

    def test_inline_html_field_renders_when_no_artifacts(self):
        # offline/replay runs save nothing to disk, but the html field is
        # still rendered so the report is visual without a live run.
        html = reporters.build_reporter('html').run(
            self._run_with_artifacts(None)
        )
        self.assertIn('html (inline)', html)
        self.assertIn('srcdoc=', html)


class JudgeDetailReportTests(unittest.TestCase):
    """The html run report surfaces the per-judge breakdown + stored detail."""

    def _run(self):
        result = models.CaseResult(
            case=models.Case(id='c1'),
            variant_name='cand',
            sample_hash='h0',
            output=models.Output(
                fields={'html': '<form>c1</form>'}, latency_ms=22160.0
            ),
            scores=[
                models.Score(
                    grader='wiring',
                    metric='wiring_complete',
                    value=0.0,
                    passed=False,
                    detail='missing [meta_adtracking]',
                    case_id='c1',
                    kind='per_case',
                ),
                models.Score(
                    grader='quality',
                    metric='quality.overall',
                    value=0.6,
                    detail='judges=claude@v2,gpt@v2',
                    case_id='c1',
                    kind='per_case',
                    judges=[
                        models.JudgeDetail(
                            key='claude',
                            version='v2',
                            rationale='clean wiring, weak contrast',
                            points={'accessibility': 2, 'visual_design': 3},
                            overall=0.5,
                        ),
                        models.JudgeDetail(
                            key='gpt',
                            version='v2',
                            rationale='mostly fine',
                            points={'accessibility': 4, 'visual_design': 3},
                            overall=0.7,
                        ),
                    ],
                ),
            ],
        )
        return models.RunResult(
            run_id='R',
            scorecard=_card('cand', wiring_complete=0.0),
            results=[result],
        )

    def test_judge_breakdown_rationales_notes_and_latency(self):
        html = reporters.build_reporter('html').run(self._run())
        self.assertIn('Judge panel', html)  # the breakdown block
        self.assertIn('clean wiring, weak contrast', html)  # rationale
        self.assertIn('mostly fine', html)
        self.assertIn('overall (0..1)', html)
        self.assertIn('accessibility', html)  # per-dimension raw points row
        self.assertIn('Notes', html)  # stored detail surfaced
        self.assertIn('missing [meta_adtracking]', html)  # the failing "why"
        self.assertIn('22.2s', html)  # latency label


def _agreement():
    return models.AgreementResult(
        judge_name='quality',
        scale=5,
        n_ratings=3,
        n_raters=1,
        overall_mae=0.2,
        overall_correlation=None,
        dimensions=[
            models.DimensionAgreement(
                dimension='clarity',
                n=3,
                human_mean=0.6,
                judge_mean=0.8,
                mae=0.2,
                correlation=None,
            )
        ],
    )


class AgreementReporterTests(unittest.TestCase):
    def test_markdown_agreement_delegates(self):
        md = reporters.build_reporter('markdown').agreement(_agreement())
        self.assertIn('judge<->human agreement', md)
        self.assertIn('clarity', md)

    def test_html_agreement_is_a_card_fragment(self):
        frag = reporters.build_reporter('html').agreement(_agreement())
        self.assertIn('class="card"', frag)
        self.assertIn('clarity', frag)
        self.assertIn('0.2000', frag)  # MAE formatted
        self.assertIn('n/a', frag)  # None correlation
        self.assertNotIn('<!doctype', frag)  # fragment, not a full doc

    def test_render_agreement_falls_back_to_markdown(self):
        class _Bare:
            def scorecard(self, sc):
                return 'SC'

            def comparison(self, cmp):
                return 'CMP'

        out = base.render_agreement(_Bare(), _agreement())
        self.assertIn('judge<->human agreement', out)  # markdown fallback


def _preferences():
    return models.PreferenceResult(
        project='p',
        suite='s',
        variant_a='baseline',
        variant_b='candidate',
        n_raters=2,
        n=4,
        a_wins=2,
        b_wins=1,
        ties=1,
        win_rate_a=0.625,
        dimensions=[
            models.DimensionPreference(
                dimension='visual',
                n=4,
                a_wins=3,
                b_wins=1,
                ties=0,
                win_rate_a=0.75,
            )
        ],
    )


def _pw_agreement():
    return models.PairwiseAgreement(
        variant_a='baseline',
        variant_b='candidate',
        judge_name='pairwise',
        n=2,
        agree=1,
        human_win_rate_a=0.5,
        judge_win_rate_a=1.0,
        agreement_rate=0.5,
        outcomes=[
            models.PairwiseAgreementCase(
                case_id='c1', human='a', judge='a', agree=True
            ),
            models.PairwiseAgreementCase(
                case_id='c2', human='b', judge='a', agree=False
            ),
        ],
    )


class PreferenceReporterTests(unittest.TestCase):
    def test_markdown_preferences_delegates(self):
        md = reporters.build_reporter('markdown').preferences(_preferences())
        self.assertIn('human preferences', md)
        self.assertIn('visual', md)
        self.assertIn('0.6250', md)

    def test_html_preferences_is_a_card_fragment(self):
        frag = reporters.build_reporter('html').preferences(_preferences())
        self.assertIn('class="card"', frag)
        self.assertIn('visual', frag)
        self.assertIn('0.7500', frag)
        self.assertNotIn('<!doctype', frag)

    def test_render_preferences_falls_back_to_markdown(self):
        class _Bare:
            def scorecard(self, sc):
                return 'SC'

            def comparison(self, cmp):
                return 'CMP'

        out = base.render_preferences(_Bare(), _preferences())
        self.assertIn('human preferences', out)

    def test_markdown_pairwise_agreement_delegates(self):
        md = reporters.build_reporter('markdown').pairwise_agreement(
            _pw_agreement()
        )
        self.assertIn('pairwise agreement', md)
        self.assertIn('c1', md)

    def test_html_pairwise_agreement_is_a_card_fragment(self):
        frag = reporters.build_reporter('html').pairwise_agreement(
            _pw_agreement()
        )
        self.assertIn('class="card"', frag)
        self.assertIn('0.5000', frag)  # agreement rate
        self.assertNotIn('<!doctype', frag)

    def test_render_pairwise_agreement_falls_back_to_markdown(self):
        class _Bare:
            def scorecard(self, sc):
                return 'SC'

            def comparison(self, cmp):
                return 'CMP'

        out = base.render_pairwise_agreement(_Bare(), _pw_agreement())
        self.assertIn('pairwise agreement', out)


def _sweep():
    return models.SweepResult(
        project='p',
        suite='s',
        win_metric='quality.overall',
        win_higher_is_better=True,
        entries=[
            models.SweepEntry(variant='candidate', win_value=0.9, rank=1),
            models.SweepEntry(variant='baseline', win_value=0.5, rank=2),
        ],
        matrix={
            'quality.overall': {'candidate': 0.9, 'baseline': 0.5},
            'errors': {'candidate': 0.0, 'baseline': None},
        },
    )


def _pairwise():
    return models.PairwiseResult(
        project='p',
        suite='s',
        variant_a='baseline',
        variant_b='candidate',
        judge_name='pairwise',
        judge_version='v1',
        n=4,
        a_wins=1,
        b_wins=3,
        ties=0,
        win_rate_a=0.25,
    )


class SweepReporterTests(unittest.TestCase):
    def test_markdown_sweep_delegates(self):
        md = reporters.build_reporter('markdown').sweep(_sweep())
        self.assertIn('## sweep', md)
        self.assertIn('Leaderboard', md)
        self.assertIn('candidate', md)

    def test_html_sweep_is_a_card_fragment(self):
        frag = reporters.build_reporter('html').sweep(_sweep())
        self.assertIn('class="card"', frag)
        self.assertIn('Leaderboard', frag)
        self.assertIn('quality.overall', frag)
        self.assertIn('0.9000', frag)  # win value formatted
        self.assertIn('n/a', frag)  # None cell in the matrix
        self.assertIn('<b>0.9000</b>', frag)  # best cell bolded
        self.assertNotIn('<!doctype', frag)  # fragment, not a full doc

    def test_render_sweep_falls_back_to_markdown(self):
        class _Bare:
            def scorecard(self, sc):
                return 'SC'

            def comparison(self, cmp):
                return 'CMP'

        out = base.render_sweep(_Bare(), _sweep())
        self.assertIn('## sweep', out)  # markdown fallback


class PairwiseReporterTests(unittest.TestCase):
    def test_markdown_pairwise_delegates(self):
        md = reporters.build_reporter('markdown').pairwise(_pairwise())
        self.assertIn('## pairwise', md)
        self.assertIn('A win-rate', md)

    def test_html_pairwise_is_a_card_fragment(self):
        frag = reporters.build_reporter('html').pairwise(_pairwise())
        self.assertIn('class="card"', frag)
        self.assertIn('0.2500', frag)  # win-rate formatted
        self.assertIn('baseline', frag)
        self.assertNotIn('<!doctype', frag)

    def test_render_pairwise_falls_back_to_markdown(self):
        class _Bare:
            def scorecard(self, sc):
                return 'SC'

            def comparison(self, cmp):
                return 'CMP'

        out = base.render_pairwise(_Bare(), _pairwise())
        self.assertIn('## pairwise', out)


if __name__ == '__main__':
    unittest.main()
