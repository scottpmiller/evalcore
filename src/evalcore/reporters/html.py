"""HTML reporter - standalone, self-contained report documents.

Renders a single scorecard or a candidate-vs-baseline comparison as an HTML
fragment, and wraps fragments in a minimal styled document via ``document``.
Values are escaped; the output has no external assets, so it drops straight
into a CI artifact, a PR attachment, or an inline preview.
"""

import base64
import html as _html
import json
import pathlib

from evalcore import models
from evalcore.reporters import base


def _dump(fields: dict) -> str:
    try:
        return json.dumps(fields, indent=2, default=str)[:6000]
    except (TypeError, ValueError):
        return str(fields)[:6000]


_IMG_EXT = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg'}


def _embed_artifact(name: str, path: str) -> str:
    """Embed one saved artifact inline so the report stays self-contained.

    Images become base64 ``data:`` URIs, HTML renders in a sandboxed iframe,
    PDFs embed via a data URI, and anything else falls back to its decoded
    text. A path that can't be read degrades to a note rather than erroring,
    so a report built on a machine without the artifacts still renders.
    """
    p = pathlib.Path(str(path))
    cap = (
        f'<figcaption class="meta">{_esc(name)} &middot; '
        f'<code>{_esc(p.name)}</code></figcaption>'
    )
    try:
        raw = p.read_bytes()
    except OSError:
        return (
            f'<figure class="art">{cap}'
            f'<p class="meta">missing: <code>{_esc(path)}</code></p></figure>'
        )
    ext = p.suffix.lower()
    if ext in _IMG_EXT:
        mime = (
            'image/svg+xml'
            if ext == '.svg'
            else 'image/jpeg'
            if ext in ('.jpg', '.jpeg')
            else f'image/{ext.lstrip(".")}'
        )
        b64 = base64.b64encode(raw).decode('ascii')
        media = (
            f'<img class="art-img" alt="{_esc(name)}" '
            f'src="data:{mime};base64,{b64}">'
        )
    elif ext in ('.html', '.htm'):
        media = _iframe(raw.decode('utf-8', 'replace'))
    elif ext == '.pdf':
        b64 = base64.b64encode(raw).decode('ascii')
        media = (
            f'<iframe class="art-frame" '
            f'src="data:application/pdf;base64,{b64}"></iframe>'
        )
    else:
        media = f'<pre>{_esc(raw.decode("utf-8", "replace")[:6000])}</pre>'
    return f'<figure class="art">{cap}{media}</figure>'


def _iframe(content: str) -> str:
    """A sandboxed iframe rendering an HTML fragment/document."""
    return (
        f'<iframe class="art-frame" sandbox loading="lazy" '
        f'srcdoc="{_html.escape(content, quote=True)}"></iframe>'
    )


def _pt(value) -> str:
    """A raw judge point: integer-clean when whole (``4``, not ``4.0``)."""
    return '&ndash;' if value is None else f'{value:g}'


def _judges_html(score: models.Score) -> str:
    """The per-judge breakdown behind an LLM-judge ``overall`` score:
    each judge's raw per-dimension points side by side, its own normalized
    overall, and its free-text rationale - the ``why`` the aggregate hides."""
    judges = score.judges
    if not judges:
        return ''
    dims = list(judges[0].points)
    head = ''.join(f'<th class="num">{_esc(j.key)}</th>' for j in judges)
    body = ''
    for dim in dims:
        cells = ''.join(
            f'<td class="num">{_pt(j.points.get(dim))}</td>' for j in judges
        )
        body += f'<tr><td>{_esc(dim)}</td>{cells}</tr>'
    body += (
        '<tr><td><b>overall (0..1)</b></td>'
        + ''.join(f'<td class="num">{_fmt(j.overall)}</td>' for j in judges)
        + '</tr>'
    )
    rationales = ''.join(
        f'<p class="rationale"><b>{_esc(j.key)}'
        f'{" @" + _esc(j.version) if j.version else ""}:</b> '
        f'{_esc(j.rationale)}</p>'
        for j in judges
        if j.rationale
    )
    return (
        f'<div class="judgment"><h4>Judge panel &mdash; '
        f'<code>{_esc(score.grader)}</code> &middot; raw points</h4>'
        f'<table class="judges"><thead><tr><th>dimension</th>{head}</tr>'
        f'</thead><tbody>{body}</tbody></table>{rationales}</div>'
    )


def _notes_html(cells: dict, error: str | None) -> str:
    """Surface stored per-case ``detail`` the matrix can't: the ``why`` of
    each failing deterministic check, plus an errored invocation's message."""
    items = []
    if error:
        items.append(f'<li class="bad">errored: {_esc(error)}</li>')
    for sc in cells.values():
        if sc.passed is False and sc.detail:
            items.append(
                f'<li><code>{_esc(sc.metric)}</code> &mdash; {_esc(sc.detail)}'
                '</li>'
            )
    if not items:
        return ''
    return f'<h4>Notes</h4><ul class="notes">{"".join(items)}</ul>'


_CSS = (
    'body{font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;'
    'margin:0;padding:24px;color:#1a1d21;background:#f6f7f9}'
    '.card{max-width:900px;margin:0 auto 20px;background:#fff;'
    'border:1px solid #e3e6ea;border-radius:12px;padding:20px 24px;'
    'box-shadow:0 1px 3px rgba(20,25,35,.06)}'
    'h2{margin:0 0 6px;font-size:20px}h3{margin:18px 0 8px;font-size:15px}'
    'code{background:#eef0f3;border-radius:5px;padding:1px 6px;font-size:.9em}'
    '.meta{color:#68707a;font-size:13px;margin:0 0 14px}'
    'table{width:100%;border-collapse:collapse;font-size:14px}'
    'th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #eceef1}'
    'th{color:#68707a;font-weight:600}.num{text-align:right;'
    'font-variant-numeric:tabular-nums}'
    '.up{color:#1f8f5f}.down{color:#c0392b}'
    '.verdict{display:inline-block;border-radius:6px;padding:2px 10px;color:#fff}'
    '.verdict.pass{background:#1f8f5f}.verdict.warn{background:#d0870b}'
    '.verdict.fail{background:#c0392b}'
    '.guards{list-style:none;padding:0;margin:0}'
    '.guards li{padding:4px 0;font-size:14px}'
    '.guards li.ok::before{content:"\\2713 ";color:#1f8f5f}'
    '.guards li.breach::before{content:"\\2717 ";color:#c0392b}'
    '.guards li.skipped::before{content:"\\2013 ";color:#8a8a8a}'
    'td.fail{color:#c0392b;font-weight:600}'
    'details{margin:6px 0}summary{cursor:pointer;color:#68707a;font-size:13px}'
    'pre{background:#f0f2f4;border-radius:6px;padding:10px;overflow:auto;'
    'max-height:340px;white-space:pre-wrap;word-break:break-word;'
    'font:12px ui-monospace,SFMono-Regular,Menlo,monospace}'
    '.scroll{overflow-x:auto}.scroll table{min-width:100%;width:max-content}'
    '.arts{display:grid;gap:14px;margin:10px 0}'
    'figure.art{margin:0;border:1px solid #e3e6ea;border-radius:8px;'
    'padding:10px 12px;background:#fafbfc}'
    'figure.art figcaption{margin:0 0 8px}'
    '.art-img{display:block;max-width:100%;height:auto;border-radius:6px;'
    'border:1px solid #e3e6ea;background:#fff}'
    '.art-frame{width:100%;height:520px;border:1px solid #e3e6ea;'
    'border-radius:6px;background:#fff}'
    'details.raw{margin-top:10px}'
    'h4{margin:14px 0 6px;font-size:14px}'
    '.judgment{margin:10px 0}'
    'table.judges{width:auto;min-width:60%}'
    'table.judges td:first-child,table.judges th:first-child{font-weight:600}'
    '.rationale{font-size:13px;color:#333;margin:6px 0;padding:8px 10px;'
    'background:#f0f2f4;border-radius:6px}'
    '.notes{margin:4px 0;padding-left:18px;font-size:13px}'
    '.notes li{padding:2px 0}.notes li.bad{color:#c0392b;font-weight:600}'
)


def _fmt(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:.4f}'


def _fmt_delta(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:+.4f}'


def _esc(value) -> str:
    return _html.escape(str(value))


def _guard_class(guard) -> str:
    """CSS class for one guardrail result.

    ``passed`` is tri-state: ``None`` is a rule that could not run - a
    relative guardrail with no baseline - which must not render as a breach.
    """
    if guard.passed is None:
        return 'skipped'
    return 'ok' if guard.passed else 'breach'


@base.register('html')
class HtmlReporter:
    """Render scorecards and comparisons as self-contained HTML."""

    def scorecard(self, scorecard: models.Scorecard) -> str:
        rows = ''.join(
            f'<tr><td>{_esc(m.metric)}</td>'
            f'<td class="num">{_fmt(m.value)}</td></tr>'
            for m in scorecard.metrics.values()
        )
        meta = (
            f'model <code>{_esc(scorecard.model_id)}</code> &middot; '
            f'mode <code>{_esc(scorecard.mode)}</code> &middot; '
            f'dataset <code>{_esc(scorecard.dataset_version)}</code> &middot; '
            f'{scorecard.n_cases}&times;{scorecard.n_samples} cases'
        )
        return (
            f'<section class="card"><h2>{_esc(scorecard.project)}/'
            f'{_esc(scorecard.suite)} &middot; '
            f'<code>{_esc(scorecard.variant.name)}</code></h2>'
            f'<p class="meta">{meta}</p>'
            f'<table><thead><tr><th>metric</th>'
            f'<th class="num">value</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></section>'
        )

    def comparison(self, comparison: models.Comparison) -> str:
        drows = ''
        for delta in comparison.deltas:
            mark = (
                f' <b>({_esc(comparison.win)})</b>'
                if delta.metric == comparison.win_metric
                else ''
            )
            direction = (
                'up'
                if (delta.delta or 0) > 0
                else 'down'
                if (delta.delta or 0) < 0
                else ''
            )
            drows += (
                f'<tr><td>{_esc(delta.metric)}{mark}</td>'
                f'<td class="num">{_fmt(delta.baseline)}</td>'
                f'<td class="num">{_fmt(delta.candidate)}</td>'
                f'<td class="num {direction}">{_fmt_delta(delta.delta)}</td>'
                f'</tr>'
            )
        guards = ''
        if comparison.guardrails:
            items = ''.join(
                f'<li class="{_guard_class(g)}">'
                f'<code>{_esc(g.metric)}</code> &mdash; {_esc(g.detail)}</li>'
                for g in comparison.guardrails
            )
            guards = f'<h3>Guardrails</h3><ul class="guards">{items}</ul>'
        return (
            f'<section class="card">'
            f'<h2><span class="verdict {_esc(comparison.verdict)}">'
            f'{_esc(comparison.verdict.upper())}</span> '
            f'{_esc(comparison.project)}/{_esc(comparison.suite)}</h2>'
            f'<p><code>{_esc(comparison.candidate_variant)}</code> vs '
            f'<code>{_esc(comparison.baseline_variant)}</code> &mdash; '
            f'{_esc(comparison.summary)}</p>'
            f'<table><thead><tr><th>metric</th>'
            f'<th class="num">baseline</th><th class="num">candidate</th>'
            f'<th class="num">delta</th></tr></thead>'
            f'<tbody>{drows}</tbody></table>{guards}</section>'
        )

    def agreement(self, result: models.AgreementResult) -> str:
        rows = ''.join(
            f'<tr><td>{_esc(d.dimension)}</td>'
            f'<td class="num">{d.n}</td>'
            f'<td class="num">{_fmt(d.human_mean)}</td>'
            f'<td class="num">{_fmt(d.judge_mean)}</td>'
            f'<td class="num">{_fmt(d.mae)}</td>'
            f'<td class="num">{_fmt(d.correlation)}</td></tr>'
            for d in result.dimensions
        )
        return (
            f'<section class="card"><h2>Judge&harr;human agreement &middot; '
            f'<code>{_esc(result.judge_name)}</code></h2>'
            f'<p class="meta">{result.n_ratings} ratings from '
            f'{result.n_raters} rater(s), scale 1..{result.scale} &middot; '
            f'overall MAE {_fmt(result.overall_mae)}, '
            f'r {_fmt(result.overall_correlation)}</p>'
            f'<table><thead><tr><th>dimension</th><th class="num">n</th>'
            f'<th class="num">human</th><th class="num">judge</th>'
            f'<th class="num">MAE</th><th class="num">corr</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></section>'
        )

    def preferences(self, result: models.PreferenceResult) -> str:
        rows = ''.join(
            f'<tr><td>{_esc(d.dimension)}</td>'
            f'<td class="num">{d.n}</td>'
            f'<td class="num">{d.a_wins}</td>'
            f'<td class="num">{d.b_wins}</td>'
            f'<td class="num">{d.ties}</td>'
            f'<td class="num">{_fmt(d.win_rate_a)}</td></tr>'
            for d in result.dimensions
        )
        body = (
            f'<table><thead><tr><th>dimension</th><th class="num">n</th>'
            f'<th class="num">A wins</th><th class="num">B wins</th>'
            f'<th class="num">ties</th><th class="num">A win-rate</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>'
            if rows
            else ''
        )
        return (
            f'<section class="card"><h2>Human preferences &middot; '
            f'{_esc(result.project)}/{_esc(result.suite)}</h2>'
            f'<p class="meta"><code>{_esc(result.variant_a)}</code> (A) vs '
            f'<code>{_esc(result.variant_b)}</code> (B) &middot; '
            f'{result.n} preferences from {result.n_raters} rater(s)</p>'
            f'<p><b>A win-rate {_fmt(result.win_rate_a)}</b> (ties = half) '
            f'&middot; A wins {result.a_wins} &middot; B wins {result.b_wins} '
            f'&middot; ties {result.ties}</p>{body}</section>'
        )

    def pairwise_agreement(self, result: models.PairwiseAgreement) -> str:
        rows = ''.join(
            f'<tr><td>{_esc(c.case_id)}</td>'
            f'<td class="num">{_esc(c.human)}</td>'
            f'<td class="num">{_esc(c.judge)}</td>'
            f'<td class="{"" if c.agree else "fail"}">'
            f'{"yes" if c.agree else "NO"}</td></tr>'
            for c in result.outcomes
        )
        return (
            f'<section class="card"><h2>Pairwise agreement &middot; human vs '
            f'<code>{_esc(result.judge_name)}</code></h2>'
            f'<p class="meta"><code>{_esc(result.variant_a)}</code> (A) vs '
            f'<code>{_esc(result.variant_b)}</code> (B)</p>'
            f'<p><b>agreement {_fmt(result.agreement_rate)}</b> '
            f'({result.agree}/{result.n} cases pick the same winner) &middot; '
            f'A win-rate: human {_fmt(result.human_win_rate_a)}, '
            f'judge {_fmt(result.judge_win_rate_a)}</p>'
            f'<table><thead><tr><th>case</th><th class="num">human</th>'
            f'<th class="num">judge</th><th>agree</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></section>'
        )

    def sweep(self, result: models.SweepResult) -> str:
        variants = [e.variant for e in result.entries] or sorted(
            {v for row in result.matrix.values() for v in row}
        )
        ranking = ''
        if result.win_metric:
            direction = 'higher' if result.win_higher_is_better else 'lower'
            rrows = ''.join(
                f'<tr><td class="num">{e.rank}</td>'
                f'<td><code>{_esc(e.variant)}</code></td>'
                f'<td class="num">{_fmt(e.win_value)}</td></tr>'
                for e in result.entries
            )
            ranking = (
                f'<p class="meta">ranked by <code>{_esc(result.win_metric)}'
                f'</code> ({direction} is better)</p>'
                f'<table><thead><tr><th class="num">rank</th><th>variant</th>'
                f'<th class="num">{_esc(result.win_metric)}</th></tr></thead>'
                f'<tbody>{rrows}</tbody></table>'
            )
        head = ''.join(f'<th class="num">{_esc(v)}</th>' for v in variants)
        mrows = ''
        for metric, row in result.matrix.items():
            present = [v for v in row.values() if v is not None]
            best = max(present) if present else None  # display hint only
            cells = ''
            for variant in variants:
                value = row.get(variant)
                text = _fmt(value)
                if value is not None and best is not None and value == best:
                    text = f'<b>{text}</b>'
                cells += f'<td class="num">{text}</td>'
            mrows += f'<tr><td>{_esc(metric)}</td>{cells}</tr>'
        return (
            f'<section class="card"><h2>Sweep &middot; '
            f'{_esc(result.project)}/{_esc(result.suite)}</h2>'
            f'{ranking}<h3>Leaderboard</h3>'
            f'<div class="scroll"><table><thead><tr><th>metric</th>{head}'
            f'</tr></thead><tbody>{mrows}</tbody></table></div></section>'
        )

    def pairwise(self, result: models.PairwiseResult) -> str:
        return (
            f'<section class="card"><h2>Pairwise &middot; '
            f'{_esc(result.project)}/{_esc(result.suite)}</h2>'
            f'<p class="meta"><code>{_esc(result.variant_a)}</code> (A) vs '
            f'<code>{_esc(result.variant_b)}</code> (B) &middot; judge '
            f'<code>{_esc(result.judge_name)}@'
            f'{_esc(result.judge_version)}</code></p>'
            f'<p><b>A win-rate {_fmt(result.win_rate_a)}</b> (ties = half) '
            f'over {result.n} cases &middot; A wins {result.a_wins} '
            f'&middot; B wins {result.b_wins} &middot; ties {result.ties}</p>'
            f'</section>'
        )

    def run(self, run: models.RunResult) -> str:
        """Scorecard summary + a per-case table and collapsible outputs.

        Returns a fragment; :func:`base.render_run` wraps it in a document."""
        body = self.scorecard(run.scorecard)
        metrics, rows = base.per_case_matrix(run)
        if rows:
            head = ''.join(f'<th class="num">{_esc(m)}</th>' for m in metrics)
            trs = ''
            details = ''
            for row in rows:
                if row['error']:
                    cells = (
                        f'<td colspan="{len(metrics)}" class="fail">'
                        f'error: {_esc(row["error"])}</td>'
                    )
                else:
                    cells = ''
                    for metric in metrics:
                        score = row['cells'].get(metric)
                        value = _fmt(score.value if score else None)
                        cls = (
                            'num fail'
                            if score and score.passed is False
                            else 'num'
                        )
                        cells += f'<td class="{cls}">{value}</td>'
                trs += f'<tr><td>{_esc(row["label"])}</td>{cells}</tr>'
                out = row['output']
                fields = out.fields or {}
                embeds = [
                    _embed_artifact(name, path)
                    for name, path in (out.artifacts or {}).items()
                ]
                # Fallback: render an inline HTML field when nothing was saved
                # to disk, so offline/replay runs still show the rendered form.
                if not embeds and isinstance(fields.get('html'), str):
                    embeds.append(
                        '<figure class="art"><figcaption class="meta">'
                        'html (inline)</figcaption>'
                        f'{_iframe(fields["html"])}</figure>'
                    )
                arts = (
                    f'<div class="arts">{"".join(embeds)}</div>'
                    if embeds
                    else ''
                )
                judgments = ''.join(
                    _judges_html(sc)
                    for sc in row['cells'].values()
                    if sc.judges
                )
                notes = _notes_html(row['cells'], row['error'])
                lat = (
                    f' &middot; <span class="meta">'
                    f'{out.latency_ms / 1000:.1f}s</span>'
                    if out.latency_ms is not None
                    else ''
                )
                details += (
                    f'<details><summary>{_esc(row["label"])}{lat}</summary>'
                    + arts
                    + judgments
                    + notes
                    + '<details class="raw"><summary>output fields (JSON)'
                    f'</summary><pre>{_esc(_dump(fields))}</pre></details>'
                    + '</details>'
                )
            body += (
                '<section class="card"><h3>Per-case</h3>'
                f'<div class="scroll"><table><thead><tr><th>case</th>{head}'
                f'</tr></thead><tbody>{trs}</tbody></table></div>'
                f'<h3>Outputs</h3>{details}</section>'
            )
        return body

    def document(self, body: str) -> str:
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,'
            'initial-scale=1"><title>evalcore report</title>'
            f'<style>{_CSS}</style></head><body>{body}</body></html>'
        )
