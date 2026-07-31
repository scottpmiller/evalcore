"""Human blind-rating loop and judge<->human agreement.

Two halves that share one open JSONL format (``models.Rating``):

* :func:`serve` - a dependency-free localhost web app that presents a
  *blind* queue over one or more persisted runs (no model/variant reaches
  the browser), renders each output's screenshot + HTML, collects 1..scale
  scores per rubric dimension, and appends ratings to a JSONL file. The
  same file can be produced by any external tool instead.
* :func:`compute_agreement` - reads a run + its ratings and reports how
  well the LLM judge tracks the humans (per-dimension MAE and correlation).
  A judge that doesn't agree with human taste isn't trustworthy as a win
  metric yet - this is the calibration gate before a sweep.

Blinding is enforced server-side: the queue payload carries an opaque item
index and the case input (shared across variants, so safe to show), never
the run/variant/model. Ratings map back to (run_id, case_id, sample_hash)
server-side.
"""

import http.server
import json
import pathlib
import random
import statistics
import threading
import urllib.parse
import webbrowser

from evalcore import models, refs, store
from evalcore.pairwise import comparable_samples

# Artifact rendering is type-driven, not consumer-specific: a panel's ``kind``
# decides how the browser shows it. File-backed kinds (image/pdf) stream
# through /api/artifact; inline kinds (html/json/text) ride in the queue.
_CONTENT_TYPES = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.svg': 'image/svg+xml',
    '.pdf': 'application/pdf',
    '.html': 'text/html',
    '.htm': 'text/html',
    '.json': 'application/json',
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.mjs': 'text/javascript',
    '.js': 'text/javascript',
}
_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg'}


def _kind_for_path(path: str) -> str:
    """Map an artifact file path to a render kind by extension."""
    ext = pathlib.Path(path).suffix.lower()
    if ext in _IMAGE_EXTS:
        return 'image'
    if ext == '.pdf':
        return 'pdf'
    if ext in ('.html', '.htm'):
        return 'html'
    if ext == '.json':
        return 'json'
    return 'text'


def _infer_kind(ref: str, value) -> str:
    """Best-effort kind for a view whose ``kind`` was not declared."""
    if isinstance(value, str) and ref.startswith('artifacts.'):
        return _kind_for_path(value)
    if isinstance(value, (dict, list)):
        return 'json'
    if isinstance(value, str) and value.lstrip().startswith('<'):
        return 'html'
    return 'text'


def _default_view_specs(
    content_ref: str | None, screenshot_ref: str | None
) -> list[dict]:
    """Back-compat views from the old content_ref/screenshot_ref params."""
    specs: list[dict] = []
    if content_ref:
        specs.append({'label': 'Rendered', 'kind': 'html', 'ref': content_ref})
    if screenshot_ref:
        specs.append(
            {'label': 'Screenshot', 'kind': 'image', 'ref': screenshot_ref}
        )
    return specs


def _prettify(name: str) -> str:
    return name.replace('_', ' ').replace('-', ' ').strip().title()


def _judge_scores(run: models.RunResult, judge_name: str) -> dict:
    """{(case_id, sample_hash): {dimension: normalized_value}} for the judge."""
    prefix = f'{judge_name}.'
    out: dict[tuple[str, str], dict[str, float]] = {}
    for result in run.results:
        dims: dict[str, float] = {}
        for score in result.scores:
            if score.metric.startswith(prefix) and score.value is not None:
                dims[score.metric[len(prefix) :]] = score.value
        out[result.case.id, result.sample_hash] = dims
    return out


def _human_scores(
    ratings: list[models.Rating], run_id: str, scale: int
) -> dict:
    """{(case_id, sample_hash): {dimension: [normalized human scores]}}.

    Keyed on the output digest, which is exact here: judge and humans scored
    the same run, so both sides carry the same hashes.
    """
    out: dict[tuple[str, str], dict[str, list[float]]] = {}
    for rating in ratings:
        if rating.run_id != run_id:
            continue
        key = (rating.case_id, rating.sample_hash)
        dims = out.setdefault(key, {})
        for dimension, raw in rating.scores.items():
            dims.setdefault(dimension, []).append(raw / scale)
    return out


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:  # zero variance on a side
        return None


def compute_agreement(
    run: models.RunResult,
    ratings: list[models.Rating],
    dimensions: list[str],
    *,
    judge_name: str = 'quality',
    scale: int = 5,
) -> models.AgreementResult:
    """Judge-vs-human agreement over one run's rated (case, sample) pairs."""
    judge = _judge_scores(run, judge_name)
    human = _human_scores(ratings, run.run_id, scale)
    raters = {r.rater for r in ratings if r.run_id == run.run_id}
    n_ratings = sum(1 for r in ratings if r.run_id == run.run_id)

    dim_results: list[models.DimensionAgreement] = []
    all_h: list[float] = []
    all_j: list[float] = []
    for dimension in dimensions:
        hs: list[float] = []
        js: list[float] = []
        for key, human_dims in human.items():
            values = human_dims.get(dimension)
            judged = judge.get(key, {}).get(dimension)
            if values and judged is not None:
                hs.append(statistics.fmean(values))
                js.append(judged)
        n = len(hs)
        mae = (
            statistics.fmean([abs(h - j) for h, j in zip(hs, js, strict=True)])
            if n
            else None
        )
        dim_results.append(
            models.DimensionAgreement(
                dimension=dimension,
                n=n,
                human_mean=statistics.fmean(hs) if hs else None,
                judge_mean=statistics.fmean(js) if js else None,
                mae=mae,
                correlation=_correlation(hs, js),
            )
        )
        all_h.extend(hs)
        all_j.extend(js)

    overall_mae = (
        statistics.fmean(
            [abs(h - j) for h, j in zip(all_h, all_j, strict=True)]
        )
        if all_h
        else None
    )
    return models.AgreementResult(
        judge_name=judge_name,
        scale=scale,
        n_ratings=n_ratings,
        n_raters=len(raters),
        dimensions=dim_results,
        overall_mae=overall_mae,
        overall_correlation=_correlation(all_h, all_j),
    )


def _tally(picks: list[str]) -> tuple[int, int, int, float | None]:
    """(a_wins, b_wins, ties, win_rate_a) over 'a'/'b'/'tie' picks; ties=half."""
    a = picks.count('a')
    b = picks.count('b')
    t = picks.count('tie')
    n = a + b + t
    return a, b, t, ((a + 0.5 * t) / n if n else None)


def aggregate_preferences(
    run_a: models.RunResult,
    run_b: models.RunResult,
    preferences: list[models.Preference],
    dimensions: list[str] | None = None,
) -> models.PreferenceResult:
    """Human side-by-side win-rate of A vs B, overall and per dimension.

    Only preferences whose ``(variant_a, variant_b)`` matches this pair's
    orientation are counted, so a preferences file mixing several comparisons
    is filtered to the one asked for.
    """
    va = run_a.scorecard.variant.name
    vb = run_b.scorecard.variant.name
    prefs = [p for p in preferences if p.variant_a == va and p.variant_b == vb]
    a_wins, b_wins, ties, win_rate = _tally([p.winner for p in prefs])

    keys = dimensions or sorted({d for p in prefs for d in p.dims})
    dim_results: list[models.DimensionPreference] = []
    for dimension in keys:
        picks = [p.dims[dimension] for p in prefs if dimension in p.dims]
        da, db, dt, dwr = _tally(picks)
        dim_results.append(
            models.DimensionPreference(
                dimension=dimension,
                n=da + db + dt,
                a_wins=da,
                b_wins=db,
                ties=dt,
                win_rate_a=dwr,
            )
        )
    return models.PreferenceResult(
        project=run_a.scorecard.project,
        suite=run_a.scorecard.suite,
        variant_a=va,
        variant_b=vb,
        n_raters=len({p.rater for p in prefs}),
        n=len(prefs),
        a_wins=a_wins,
        b_wins=b_wins,
        ties=ties,
        win_rate_a=win_rate,
        dimensions=dim_results,
    )


def _majority_winner(picks: list[str]) -> str:
    """The most-picked winner; a tie in the *counts* resolves to 'tie'."""
    counts = {w: picks.count(w) for w in ('a', 'b', 'tie')}
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    if ordered[0][1] == ordered[1][1]:  # no clear plurality
        return 'tie'
    return ordered[0][0]


def compute_pairwise_agreement(
    preferences: list[models.Preference], pairwise: models.PairwiseResult
) -> models.PairwiseAgreement:
    """How often the human panel and the LLM pairwise judge agree on the
    per-case winner. Human winner per case = majority of raters' overall picks.
    """
    # Both sides key on the variant_a output's digest, so the join lands even
    # though the two runs' outputs hash differently.
    va, vb = pairwise.variant_a, pairwise.variant_b
    by_case: dict[tuple[str, str], list[str]] = {}
    for p in preferences:
        if p.variant_a == va and p.variant_b == vb:
            by_case.setdefault((p.case_id, p.sample_hash), []).append(p.winner)
    human = {k: _majority_winner(v) for k, v in by_case.items()}
    judge = {(o.case_id, o.sample_hash): o.winner for o in pairwise.outcomes}

    keys = sorted(human.keys() & judge.keys())
    outcomes: list[models.PairwiseAgreementCase] = []
    agree = 0
    for key in keys:
        ok = human[key] == judge[key]
        agree += ok
        outcomes.append(
            models.PairwiseAgreementCase(
                case_id=key[0],
                sample_hash=key[1],
                human=human[key],
                judge=judge[key],
                agree=ok,
            )
        )
    n = len(keys)
    _, _, _, human_wr = _tally([human[k] for k in keys])
    _, _, _, judge_wr = _tally([judge[k] for k in keys])
    return models.PairwiseAgreement(
        variant_a=va,
        variant_b=vb,
        judge_name=pairwise.judge_name,
        n=n,
        agree=agree,
        human_win_rate_a=human_wr,
        judge_win_rate_a=judge_wr,
        agreement_rate=(agree / n if n else None),
        outcomes=outcomes,
    )


def _derive_panels(
    output: models.Output, specs: list[dict]
) -> tuple[list[dict], list[str]]:
    """Derive an output's render panels (shared by the rate + rank apps).

    Configured views first, then any *visual* artifacts (image/pdf) not
    already referenced, then a fields fallback so a plain text or JSON result
    still shows something with zero configuration. Returns ``(panels, files)``
    where ``panels`` is client-safe (no paths) and ``files`` is the server-only
    ``index -> path`` map that file-backed panels' ``v`` index into.
    """
    ctx = {
        'output': output.fields,
        'artifacts': output.artifacts,
        'case': None,
        'input': None,
    }
    panels: list[dict] = []
    files: list[str] = []
    covered: set[str] = set()

    def add_inline(label, kind, value):
        # The value stays exact for the rendered iframe, Source view, and copy.
        panels.append({'label': label, 'kind': kind, 'value': value})

    def add_file(label, kind, path):
        panels.append({'label': label, 'kind': kind, 'v': len(files)})
        files.append(path)

    for spec in specs:
        ref = spec['ref']
        value = refs.resolve_ref(ctx, ref)
        if value is None or value == '':
            continue
        kind = spec.get('kind') or _infer_kind(ref, value)
        if ref.startswith('artifacts.'):
            covered.add(ref.split('.', 1)[1])
        if kind in ('image', 'pdf'):
            if isinstance(value, str):
                add_file(spec['label'], kind, value)
        elif kind == 'json':
            add_inline(spec['label'], 'json', value)
        elif kind == 'html':
            add_inline(spec['label'], 'html', str(value))
        else:
            add_inline(spec['label'], 'text', str(value))

    # Any visual artifact not already shown -> its own panel (quantity).
    for name, path in (output.artifacts or {}).items():
        if name in covered or not isinstance(path, str):
            continue
        kind = _kind_for_path(path)
        if kind in ('image', 'pdf'):
            add_file(_prettify(name), kind, path)

    # Nothing configured/derived: fall back to the output fields, so a
    # text-only or JSON-only result is reviewable out of the box.
    if not panels:
        fields = output.fields or {}
        str_fields = {k: v for k, v in fields.items() if isinstance(v, str)}
        if len(fields) == 1 and str_fields:
            key, value = next(iter(str_fields.items()))
            add_inline(
                _prettify(key),
                'html' if value.lstrip().startswith('<') else 'text',
                value,
            )
        elif fields:
            add_inline('Output', 'json', fields)
    return panels, files


# -- the blind rating web app ------------------------------------------------

# Self-contained single page (no external assets): a production-minded blind
# rating console. ``__CONFIG__`` is replaced server-side with a JSON blob
# ``{scale, dims}`` - one robust injection point rather than string-replacing
# bare ``SCALE``/``DIMS`` tokens that could collide with page text.
_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>evalcore &middot; blind rating</title><style>
:root{
  --bg:#f5f6f8;--panel:#fff;--panel-2:#fafbfc;--ink:#1a1d21;--muted:#68707a;
  --line:#e3e6ea;--brand:#2f6fed;--brand-ink:#fff;--on:#1f8f5f;--on-ink:#fff;
  --shadow:0 1px 3px rgba(20,25,35,.08),0 8px 24px rgba(20,25,35,.06);--frame:#fff;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0d0f12;--panel:#16191e;--panel-2:#1b1f25;--ink:#eceef1;--muted:#9aa3ad;
  --line:#2a2f37;--brand:#4f8bff;--brand-ink:#06101f;--on:#3ecf8e;--on-ink:#052012;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 30px rgba(0,0,0,.35);--frame:#f4f5f7;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);
  border-bottom:1px solid var(--line);backdrop-filter:blur(6px)}
.hrow{display:flex;align-items:center;gap:14px;max-width:1180px;margin:0 auto;
  padding:12px 20px}
.brand{font-weight:700;letter-spacing:.2px}
.brand small{font-weight:500;color:var(--muted);margin-left:8px}
.who{margin-left:auto;display:flex;align-items:center;gap:8px;color:var(--muted);
  font-size:13px}
.chip{background:var(--panel-2);border:1px solid var(--line);border-radius:999px;
  padding:3px 10px;color:var(--ink);font-weight:600}
.progress{height:4px;background:var(--line)}
.progress i{display:block;height:100%;width:0;background:var(--brand);
  transition:width .25s ease}
main{max-width:1180px;margin:0 auto;padding:22px 20px 80px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  box-shadow:var(--shadow);overflow:hidden}
.panes{display:flex;flex-direction:column;gap:16px;margin:0 0 16px}
.pane{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  box-shadow:var(--shadow);overflow:hidden}
.phd{display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--panel);
  border-bottom:1px solid var(--line)}
.plabel{font-weight:700;font-size:13px}
.pkind{font-size:10px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);
  border:1px solid var(--line);border-radius:999px;padding:1px 8px}
.pacts{margin-left:auto;display:flex;gap:6px}
.pacts button{padding:4px 10px;font-size:12px;font-weight:600;border:1px solid var(--line);
  border-radius:6px;cursor:pointer;background:var(--panel-2);color:var(--ink)}
.pacts button.active{background:var(--brand);color:var(--brand-ink);border-color:var(--brand)}
.pbody{background:var(--frame)}
.pbody .rendered iframe{display:block;width:100%;height:720px;border:0;background:var(--frame)}
.pbody .embed{display:block;width:100%;height:640px;border:0;background:var(--frame)}
.pbody img{display:block;width:100%;height:auto}
.pbody pre.src{margin:0;padding:14px 16px;max-height:620px;overflow:auto;
  white-space:pre-wrap;word-break:break-word;tab-size:2;
  font:12.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink);
  background:var(--panel)}
.hidden{display:none}
.body{padding:18px 20px}
details{margin:0 0 16px;border:1px solid var(--line);border-radius:10px;
  background:var(--panel-2)}
details summary{cursor:pointer;padding:10px 14px;color:var(--muted);
  font-size:13px;font-weight:600;user-select:none}
details pre{margin:0;padding:0 14px 12px;white-space:pre-wrap;word-break:break-word;
  font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink)}
.dim{padding:14px 0;border-top:1px solid var(--line)}
.dim:first-of-type{border-top:0}
.dim .lab{display:flex;align-items:baseline;gap:10px;margin-bottom:8px}
.dim .lab b{font-size:15px}
.dim .lab .hint{color:var(--muted);font-size:12px}
.scale{display:flex;gap:8px;flex-wrap:wrap}
.scale button{min-width:46px;height:42px;padding:0 12px;border:1px solid var(--line);
  background:var(--panel-2);color:var(--ink);border-radius:10px;cursor:pointer;
  font-size:15px;font-weight:600;transition:transform .05s,background .12s,border-color .12s}
.scale button:hover{border-color:var(--brand)}
.scale button:active{transform:translateY(1px)}
.scale button.on{background:var(--on);color:var(--on-ink);border-color:var(--on)}
.actions{position:sticky;bottom:0;display:flex;align-items:center;gap:12px;
  padding:14px 20px;background:var(--panel);border-top:1px solid var(--line)}
.actions .rem{color:var(--muted);font-size:13px;margin-right:auto}
button.primary,button.ghost{border-radius:10px;font-size:15px;font-weight:600;
  cursor:pointer;padding:11px 20px;border:1px solid var(--line)}
button.primary{background:var(--brand);color:var(--brand-ink);border-color:var(--brand)}
button.primary:disabled{opacity:.45;cursor:not-allowed}
button.ghost{background:transparent;color:var(--muted)}
.kbd{font:12px ui-monospace,monospace;background:var(--panel-2);border:1px solid var(--line);
  border-radius:6px;padding:1px 6px;color:var(--muted)}
.empty{text-align:center;padding:70px 20px;color:var(--muted)}
.empty h2{color:var(--ink);margin:0 0 8px}
#toast{position:fixed;left:50%;bottom:24px;transform:translateX(-50%) translateY(20px);
  background:#c0392b;color:#fff;padding:10px 16px;border-radius:8px;opacity:0;
  transition:.2s;pointer-events:none}
#toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.foot{max-width:1180px;margin:14px auto 0;padding:0 20px;color:var(--muted);
  font-size:12px;display:flex;gap:16px;flex-wrap:wrap}
</style></head><body>
<header>
  <div class="hrow">
    <span class="brand">evalcore<small>blind rating</small></span>
    <span class="who">rater <span class="chip" id="rater"></span>
      <span id="pos"></span></span>
  </div>
  <div class="progress"><i id="pbar"></i></div>
</header>
<main><div id="card"><div class="empty">Loading queue&hellip;</div></div>
  <div class="foot" id="foot"></div>
</main>
<div id="toast"></div>
<script>
const CONFIG=__CONFIG__, scale=CONFIG.scale, dims=CONFIG.dims;
const q=new URLSearchParams(location.search), rater=q.get('rater')||'anon';
let queue=[], i=0, cur={}, total=0, focus=0;
document.getElementById('rater').textContent=rater;
const pretty=d=>d.replace(/[_-]+/g,' ').replace(/\\b\\w/g,c=>c.toUpperCase());
// A complete document (has <html>/<head>/<body>/<style>/doctype) already
// carries its own styling - render it AS-IS. Only bare fragments (e.g.
// replay fixtures: a lone <form> with no CSS) get wrapped in this neutral
// legibility frame, so we never override a real page's own layout.
const isDoc=h=>/<(?:!doctype|html|head|body|style)[\\s>]/i.test(h);
const FRAME='<!doctype html><html><head><meta charset="utf-8">'+
  '<meta name="viewport" content="width=device-width,initial-scale=1"><style>'+
  'html{color-scheme:light}body{margin:0;padding:28px;background:#f6f7f9;'+
  'font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1a1d21}'+
  'h1,h2,h3{margin:0 0 14px;line-height:1.25}'+
  'form{max-width:480px;margin:0 auto;background:#fff;border:1px solid #e3e6ea;'+
  'border-radius:12px;padding:24px;display:flex;flex-direction:column;gap:12px;'+
  'box-shadow:0 1px 3px rgba(20,25,35,.06)}'+
  'label{font-size:13px;font-weight:600;color:#57606a}'+
  'input,select,textarea{width:100%;padding:10px 12px;font-size:15px;'+
  'border:1px solid #cbd2da;border-radius:8px;background:#fff;color:#1a1d21}'+
  'button{padding:11px 16px;font-size:15px;font-weight:600;border:0;'+
  'border-radius:8px;background:#2f6fed;color:#fff;cursor:pointer}'+
  'p{color:#57606a;margin:2px 0;font-size:13px}</style></head><body>';
const frameDoc=html=>
  (isDoc(html)?html:FRAME+html+'</body></html>').replace(/"/g,'&quot;');
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
// Render one item's typed panels (image/pdf/html/json/text), stacked and
// full-width so nothing is compressed. html panels carry a rendered/source
// toggle; text/json/html carry a copy button.
function panelHTML(it){
  if(!it.views||!it.views.length)
    return '<div class="pbody"><pre class="src">(no renderable output)</pre></div>';
  const secs=it.views.map((vw,vi)=>{
    let body='',acts='';
    if(vw.kind==='image'){
      body='<img alt="'+esc(vw.label)+'" src="/api/artifact?id='+it.id+'&v='+vw.v+'">';
    }else if(vw.kind==='pdf'){
      body='<iframe class="embed" title="'+esc(vw.label)+
        '" src="/api/artifact?id='+it.id+'&v='+vw.v+'"></iframe>';
    }else if(vw.kind==='html'){
      body='<div class="rendered"><iframe sandbox title="'+esc(vw.label)+
        '" srcdoc="'+frameDoc(vw.value)+'"></iframe></div>'+
        '<div class="source hidden"><pre class="src">'+esc(vw.value)+'</pre></div>';
      acts='<button data-t="rendered" class="active">Rendered</button>'+
        '<button data-t="source">Source</button>'+
        '<button data-copy="'+vi+'">Copy</button>';
    }else if(vw.kind==='json'){
      body='<pre class="src">'+esc(JSON.stringify(vw.value,null,2))+'</pre>';
      acts='<button data-copy="'+vi+'">Copy</button>';
    }else{
      body='<pre class="src">'+esc(vw.value)+'</pre>';
      acts='<button data-copy="'+vi+'">Copy</button>';
    }
    return '<section class="pane" data-vi="'+vi+'"><div class="phd">'+
      '<span class="plabel">'+esc(vw.label)+'</span>'+
      '<span class="pkind">'+vw.kind+'</span><div class="pacts">'+acts+
      '</div></div><div class="pbody">'+body+'</div></section>';
  }).join('');
  return '<div class="panes">'+secs+'</div>';
}
function toast(msg){const t=document.getElementById('toast');t.textContent=msg;
  t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2600);}
function setProgress(){const done=total-queue.length+i;
  document.getElementById('pbar').style.width=total?(100*done/total)+'%':'0';
  document.getElementById('pos').textContent=total?((done+1>total?total:done+1)+' / '+total):'';}
async function load(){
  try{queue=await (await fetch('/api/queue?rater='+encodeURIComponent(rater))).json();}
  catch(e){toast('Could not load the queue');return;}
  total=queue.length;i=0;render();
}
function done(){
  document.getElementById('card').innerHTML=
    '<div class="empty"><h2>All done &mdash; thank you.</h2>'+
    '<p>'+total+' item'+(total===1?'':'s')+' rated as <b>'+rater+'</b>.</p>'+
    '<p>Reopen this page anytime to resume where you left off.</p></div>';
  document.getElementById('pbar').style.width='100%';
  document.getElementById('pos').textContent=total+' / '+total;
  document.getElementById('foot').innerHTML='';
}
function render(){
  const card=document.getElementById('card');
  if(i>=queue.length){done();return;}
  const it=queue[i];cur={};focus=0;setProgress();
  const media=panelHTML(it);
  const req=it.input?'<details><summary>Prompt / input</summary><pre>'+
    JSON.stringify(it.input,null,2).replace(/</g,'&lt;')+'</pre></details>':'';
  const rows=dims.map((d,di)=>'<div class="dim" data-di="'+di+'"><div class="lab">'+
    '<b>'+pretty(d)+'</b><span class="hint">1 = worst &middot; '+scale+' = best</span></div>'+
    '<div class="scale">'+Array.from({length:scale},(_,k)=>
      '<button type="button" data-d="'+d+'" data-v="'+(k+1)+'">'+(k+1)+
      '</button>').join('')+'</div></div>').join('');
  card.innerHTML=media+
    '<div class="card"><div class="body">'+req+rows+'</div>'+
    '<div class="actions"><span class="rem" id="rem"></span>'+
    '<button class="ghost" id="skip">Skip <span class="kbd">S</span></button>'+
    '<button class="primary" id="next" disabled>Submit <span class="kbd">Enter</span></button>'+
    '</div></div>';
  card.querySelectorAll('.scale button').forEach(b=>b.onclick=()=>pick(b.dataset.d,+b.dataset.v));
  document.getElementById('next').onclick=submit;
  document.getElementById('skip').onclick=()=>{i++;render();};
  card.querySelectorAll('.pane').forEach(pane=>{
    pane.querySelectorAll('.pacts [data-t]').forEach(b=>b.onclick=()=>{
      const t=b.dataset.t;
      pane.querySelector('.rendered').classList.toggle('hidden',t!=='rendered');
      pane.querySelector('.source').classList.toggle('hidden',t!=='source');
      pane.querySelectorAll('[data-t]').forEach(x=>x.classList.toggle('active',x===b));
    });
  });
  card.querySelectorAll('[data-copy]').forEach(b=>b.onclick=()=>{
    const vw=it.views[+b.dataset.copy];
    const text=typeof vw.value==='string'?vw.value:JSON.stringify(vw.value,null,2);
    navigator.clipboard.writeText(text).then(()=>{
      b.textContent='Copied';setTimeout(()=>{b.textContent='Copy';},1200);});
  });
  updateRem();
  document.getElementById('foot').innerHTML=
    'Keys: <span class="kbd">1</span>&ndash;<span class="kbd">'+scale+
    '</span> rate &middot; <span class="kbd">Tab</span> next field &middot; '+
    '<span class="kbd">Enter</span> submit &middot; <span class="kbd">S</span> skip';
}
function pick(d,v){
  cur[d]=v;
  document.querySelectorAll('[data-d="'+d+'"]').forEach(x=>x.classList.remove('on'));
  document.querySelector('[data-d="'+d+'"][data-v="'+v+'"]').classList.add('on');
  const di=dims.indexOf(d);if(di>-1&&di+1<dims.length&&di===focus)focus=di+1;
  updateRem();
}
function updateRem(){
  const left=dims.filter(d=>!(d in cur)).length;
  document.getElementById('next').disabled=left>0;
  document.getElementById('rem').textContent=left
    ?left+' dimension'+(left===1?'':'s')+' left':'ready to submit';
}
async function submit(){
  if(dims.some(d=>!(d in cur)))return;
  try{
    const r=await fetch('/api/rate',{method:'POST',
      headers:{'content-type':'application/json'},
      body:JSON.stringify({id:queue[i].id,rater:rater,scores:cur})});
    if(!r.ok)throw 0;
  }catch(e){toast('Save failed - not recorded');return;}
  i++;render();
}
document.addEventListener('keydown',e=>{
  if(i>=queue.length)return;
  if(e.key==='Enter'){if(!document.getElementById('next').disabled)submit();return;}
  if(e.key==='s'||e.key==='S'){i++;render();return;}
  const n=parseInt(e.key,10);
  if(!isNaN(n)&&n>=1&&n<=scale&&dims[focus])pick(dims[focus],n);
});
load();
</script></body></html>"""


class _RatingApp:
    """Server-side state: the blind queue and rating persistence."""

    def __init__(
        self,
        runs,
        ratings_path,
        dimensions,
        scale,
        content_ref=None,
        screenshot_ref=None,
        views=None,
    ):
        self.ratings_path = ratings_path
        self.dimensions = dimensions
        self.scale = scale
        specs = views or _default_view_specs(content_ref, screenshot_ref)
        # Full items keep the identity + file paths the client never sees.
        self.items = []
        for run in runs:
            for result in run.results:
                if result.output.error:
                    continue
                panels, files = self._panels(result.output, specs)
                self.items.append(
                    {
                        'run_id': run.run_id,
                        'case_id': result.case.id,
                        'sample_hash': result.sample_hash,
                        'input': result.case.input,
                        'panels': panels,  # client-safe (no paths)
                        'files': files,  # server-only: index -> path
                    }
                )

    def _panels(
        self, output: models.Output, specs: list[dict]
    ) -> tuple[list[dict], list[str]]:
        """Derive an item's render panels from its output (see
        :func:`_derive_panels`)."""
        return _derive_panels(output, specs)

    def _rated_keys(self, rater: str) -> set:
        """(run_id, case_id, sample_hash) this rater has already scored.

        Read from the ratings file so a reopened session resumes where the
        rater left off instead of re-presenting completed items.
        """
        return {
            (r.run_id, r.case_id, r.sample_hash)
            for r in store.read_ratings(self.ratings_path)
            if r.rater == rater
        }

    def queue_for(self, rater: str) -> list[dict]:
        """Seeded-shuffled blind payload: opaque id + safe fields only.

        Items this rater already rated are dropped (resumable sessions);
        the payload never carries run/variant/model identity.
        """
        order = list(range(len(self.items)))
        # Non-crypto: a stable per-rater presentation order, not a secret.
        random.Random(rater).shuffle(order)  # noqa: S311
        rated = self._rated_keys(rater)
        blind = []
        for idx in order:
            item = self.items[idx]
            key = (item['run_id'], item['case_id'], item['sample_hash'])
            if key in rated:
                continue
            blind.append(
                {'id': idx, 'input': item['input'], 'views': item['panels']}
            )
        return blind

    def record(self, idx: int, rater: str, scores: dict) -> None:
        item = self.items[idx]
        store.append_rating(
            self.ratings_path,
            models.Rating(
                run_id=item['run_id'],
                case_id=item['case_id'],
                sample_hash=item['sample_hash'],
                rater=rater,
                scores={k: int(v) for k, v in scores.items()},
            ),
        )


class _BaseHandler(http.server.BaseHTTPRequestHandler):
    """Shared plumbing for the rate + rank apps: silent logging, byte
    responses, static asset serving, and safe artifact-file streaming."""

    def log_message(self, *args):
        pass  # silence the default per-request stderr logging

    def _send(self, code, body, content_type='application/json'):
        data = body if isinstance(body, bytes) else body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, path: str) -> None:
        ctype = _CONTENT_TYPES.get(
            pathlib.Path(path).suffix.lower(), 'application/octet-stream'
        )
        with open(path, 'rb') as handle:
            self._send(200, handle.read(), ctype)


class _Handler(_BaseHandler):
    app: _RatingApp

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path == '/':
            config = json.dumps(
                {'scale': self.app.scale, 'dims': self.app.dimensions}
            )
            page = _PAGE.replace('__CONFIG__', config)
            self._send(200, page, 'text/html; charset=utf-8')
        elif parsed.path == '/api/queue':
            rater = (query.get('rater') or ['anon'])[0]
            self._send(200, json.dumps(self.app.queue_for(rater)))
        elif parsed.path == '/api/artifact':
            idx = int((query.get('id') or ['-1'])[0])
            view = int((query.get('v') or ['-1'])[0])
            try:
                self._serve_file(self.app.items[idx]['files'][view])
            except (OSError, IndexError, KeyError):
                self._send(404, b'', 'application/octet-stream')
        else:
            self._send(404, '{}')

    def do_POST(self):
        if self.path != '/api/rate':
            self._send(404, '{}')
            return
        length = int(self.headers.get('Content-Length', 0))
        payload = json.loads(self.rfile.read(length) or b'{}')
        self.app.record(
            int(payload['id']), payload['rater'], payload['scores']
        )
        self._send(200, '{"ok": true}')


def serve(
    runs: list[models.RunResult],
    ratings_path: str,
    dimensions: list[str],
    *,
    scale: int = 5,
    content_ref: str | None = None,
    screenshot_ref: str | None = 'artifacts.screenshot',
    views: list[dict] | None = None,
    host: str = '127.0.0.1',
    port: int = 8900,
    open_browser: bool = True,
) -> None:
    """Run the blind rating web app until interrupted (Ctrl-C).

    Artifacts are rendered by type. ``views`` is the general form: a list of
    ``{label, ref, kind?}`` where ``kind`` is image/pdf/html/json/text
    (inferred when omitted). ``content_ref``/``screenshot_ref`` remain as a
    shorthand for the common html+screenshot case. With neither, each item
    falls back to its output fields, so text- or JSON-only results still
    render.
    """
    _Handler.app = _RatingApp(
        runs,
        ratings_path,
        dimensions,
        scale,
        content_ref=content_ref,
        screenshot_ref=screenshot_ref,
        views=views,
    )
    server = http.server.ThreadingHTTPServer((host, port), _Handler)
    url = f'http://{host}:{port}/?rater=anon'
    n = len(_Handler.app.items)
    print(f'blind rating: {n} items on {url} (Ctrl-C to stop)')
    print(f'ratings -> {ratings_path}')
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


# -- the blind side-by-side ranking web app ---------------------------------

# The A-vs-B human analog of the rate app: two outputs for the same case are
# shown as neutral "Option 1"/"Option 2" columns and the rater picks a winner
# overall and per dimension. ``__CONFIG__`` is replaced with ``{dims}``.
_RANK_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>evalcore &middot; blind ranking</title><style>
:root{
  --bg:#f5f6f8;--panel:#fff;--panel-2:#fafbfc;--ink:#1a1d21;--muted:#68707a;
  --line:#e3e6ea;--brand:#2f6fed;--brand-ink:#fff;--on:#1f8f5f;--on-ink:#fff;
  --shadow:0 1px 3px rgba(20,25,35,.08),0 8px 24px rgba(20,25,35,.06);--frame:#fff;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0d0f12;--panel:#16191e;--panel-2:#1b1f25;--ink:#eceef1;--muted:#9aa3ad;
  --line:#2a2f37;--brand:#4f8bff;--brand-ink:#06101f;--on:#3ecf8e;--on-ink:#052012;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 30px rgba(0,0,0,.35);--frame:#f4f5f7;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);
  border-bottom:1px solid var(--line);backdrop-filter:blur(6px)}
.hrow{display:flex;align-items:center;gap:14px;max-width:1280px;margin:0 auto;
  padding:12px 20px}
.brand{font-weight:700;letter-spacing:.2px}
.brand small{font-weight:500;color:var(--muted);margin-left:8px}
.who{margin-left:auto;display:flex;align-items:center;gap:8px;color:var(--muted);
  font-size:13px}
.chip{background:var(--panel-2);border:1px solid var(--line);border-radius:999px;
  padding:3px 10px;color:var(--ink);font-weight:600}
.progress{height:4px;background:var(--line)}
.progress i{display:block;height:100%;width:0;background:var(--brand);
  transition:width .25s ease}
main{max-width:1280px;margin:0 auto;padding:22px 20px 80px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  box-shadow:var(--shadow);overflow:hidden}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:0 0 16px}
@media (max-width:820px){.cols{grid-template-columns:1fr}}
.col{display:flex;flex-direction:column;gap:12px}
.ch{font-weight:700;font-size:12px;color:var(--muted);text-transform:uppercase;
  letter-spacing:.7px;padding:2px}
.pane{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  box-shadow:var(--shadow);overflow:hidden}
.phd{display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--panel);
  border-bottom:1px solid var(--line)}
.plabel{font-weight:700;font-size:13px}
.pkind{font-size:10px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);
  border:1px solid var(--line);border-radius:999px;padding:1px 8px}
.pacts{margin-left:auto;display:flex;gap:6px}
.pacts button{padding:4px 10px;font-size:12px;font-weight:600;border:1px solid var(--line);
  border-radius:6px;cursor:pointer;background:var(--panel-2);color:var(--ink)}
.pacts button.active{background:var(--brand);color:var(--brand-ink);border-color:var(--brand)}
.pbody{background:var(--frame)}
.pbody .rendered iframe{display:block;width:100%;height:560px;border:0;background:var(--frame)}
.pbody .embed{display:block;width:100%;height:520px;border:0;background:var(--frame)}
.pbody img{display:block;width:100%;height:auto}
.pbody pre.src{margin:0;padding:14px 16px;max-height:520px;overflow:auto;
  white-space:pre-wrap;word-break:break-word;tab-size:2;
  font:12.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink);
  background:var(--panel)}
.hidden{display:none}
.body{padding:18px 20px}
details{margin:0 0 16px;border:1px solid var(--line);border-radius:10px;
  background:var(--panel-2)}
details summary{cursor:pointer;padding:10px 14px;color:var(--muted);
  font-size:13px;font-weight:600;user-select:none}
details pre{margin:0;padding:0 14px 12px;white-space:pre-wrap;word-break:break-word;
  font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink)}
.prow{display:flex;align-items:center;gap:12px;padding:12px 0;border-top:1px solid var(--line)}
.prow.head{border-top:0;color:var(--muted);font-weight:600;padding-bottom:2px}
.prow:first-of-type{border-top:0}
.plab{min-width:160px;font-weight:600}
.picks{display:flex;gap:8px;flex-wrap:wrap}
.picks button{min-width:118px;height:40px;padding:0 12px;border:1px solid var(--line);
  background:var(--panel-2);color:var(--ink);border-radius:10px;cursor:pointer;
  font-size:14px;font-weight:600;transition:transform .05s,background .12s,border-color .12s}
.picks button:hover{border-color:var(--brand)}
.picks button:active{transform:translateY(1px)}
.picks button.on{background:var(--brand);color:var(--brand-ink);border-color:var(--brand)}
.actions{position:sticky;bottom:0;display:flex;align-items:center;gap:12px;
  padding:14px 20px;background:var(--panel);border-top:1px solid var(--line)}
.actions .rem{color:var(--muted);font-size:13px;margin-right:auto}
button.primary,button.ghost{border-radius:10px;font-size:15px;font-weight:600;
  cursor:pointer;padding:11px 20px;border:1px solid var(--line)}
button.primary{background:var(--brand);color:var(--brand-ink);border-color:var(--brand)}
button.primary:disabled{opacity:.45;cursor:not-allowed}
button.ghost{background:transparent;color:var(--muted)}
.kbd{font:12px ui-monospace,monospace;background:var(--panel-2);border:1px solid var(--line);
  border-radius:6px;padding:1px 6px;color:var(--muted)}
.empty{text-align:center;padding:70px 20px;color:var(--muted)}
.empty h2{color:var(--ink);margin:0 0 8px}
#toast{position:fixed;left:50%;bottom:24px;transform:translateX(-50%) translateY(20px);
  background:#c0392b;color:#fff;padding:10px 16px;border-radius:8px;opacity:0;
  transition:.2s;pointer-events:none}
#toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.foot{max-width:1280px;margin:14px auto 0;padding:0 20px;color:var(--muted);
  font-size:12px;display:flex;gap:16px;flex-wrap:wrap}
</style></head><body>
<header>
  <div class="hrow">
    <span class="brand">evalcore<small>blind ranking</small></span>
    <span class="who">rater <span class="chip" id="rater"></span>
      <span id="pos"></span></span>
  </div>
  <div class="progress"><i id="pbar"></i></div>
</header>
<main><div id="card"><div class="empty">Loading queue&hellip;</div></div>
  <div class="foot" id="foot"></div>
</main>
<div id="toast"></div>
<script>
const CONFIG=__CONFIG__, dims=CONFIG.dims;
const q=new URLSearchParams(location.search), rater=q.get('rater')||'anon';
let queue=[], i=0, cur={winner:null,dims:{}}, total=0;
document.getElementById('rater').textContent=rater;
const pretty=d=>d.replace(/[_-]+/g,' ').replace(/\\b\\w/g,c=>c.toUpperCase());
const isDoc=h=>/<(?:!doctype|html|head|body|style)[\\s>]/i.test(h);
const FRAME='<!doctype html><html><head><meta charset="utf-8">'+
  '<meta name="viewport" content="width=device-width,initial-scale=1"><style>'+
  'html{color-scheme:light}body{margin:0;padding:28px;background:#f6f7f9;'+
  'font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#1a1d21}'+
  'h1,h2,h3{margin:0 0 14px;line-height:1.25}'+
  'form{max-width:480px;margin:0 auto;background:#fff;border:1px solid #e3e6ea;'+
  'border-radius:12px;padding:24px;display:flex;flex-direction:column;gap:12px;'+
  'box-shadow:0 1px 3px rgba(20,25,35,.06)}'+
  'label{font-size:13px;font-weight:600;color:#57606a}'+
  'input,select,textarea{width:100%;padding:10px 12px;font-size:15px;'+
  'border:1px solid #cbd2da;border-radius:8px;background:#fff;color:#1a1d21}'+
  'button{padding:11px 16px;font-size:15px;font-weight:600;border:0;'+
  'border-radius:8px;background:#2f6fed;color:#fff;cursor:pointer}'+
  'p{color:#57606a;margin:2px 0;font-size:13px}</style></head><body>';
const frameDoc=html=>
  (isDoc(html)?html:FRAME+html+'</body></html>').replace(/"/g,'&quot;');
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function artURL(id,side,v){
  return '/api/artifact?id='+id+'&side='+side+'&v='+v+
    '&rater='+encodeURIComponent(rater);
}
function panelHTML(views,side,id){
  if(!views||!views.length)
    return '<section class="pane"><div class="pbody"><pre class="src">'+
      '(no renderable output)</pre></div></section>';
  return views.map((vw,vi)=>{
    let body='',acts='';
    if(vw.kind==='image'){
      body='<img alt="'+esc(vw.label)+'" src="'+artURL(id,side,vw.v)+'">';
    }else if(vw.kind==='pdf'){
      body='<iframe class="embed" title="'+esc(vw.label)+'" src="'+
        artURL(id,side,vw.v)+'"></iframe>';
    }else if(vw.kind==='html'){
      body='<div class="rendered"><iframe sandbox title="'+esc(vw.label)+
        '" srcdoc="'+frameDoc(vw.value)+'"></iframe></div>'+
        '<div class="source hidden"><pre class="src">'+esc(vw.value)+'</pre></div>';
      acts='<button data-t="rendered" class="active">Rendered</button>'+
        '<button data-t="source">Source</button>';
    }else if(vw.kind==='json'){
      body='<pre class="src">'+esc(JSON.stringify(vw.value,null,2))+'</pre>';
    }else{
      body='<pre class="src">'+esc(vw.value)+'</pre>';
    }
    return '<section class="pane" data-vi="'+vi+'"><div class="phd">'+
      '<span class="plabel">'+esc(vw.label)+'</span>'+
      '<span class="pkind">'+vw.kind+'</span><div class="pacts">'+acts+
      '</div></div><div class="pbody">'+body+'</div></section>';
  }).join('');
}
function toast(msg){const t=document.getElementById('toast');t.textContent=msg;
  t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2600);}
function setProgress(){const done=total-queue.length+i;
  document.getElementById('pbar').style.width=total?(100*done/total)+'%':'0';
  document.getElementById('pos').textContent=total?((done+1>total?total:done+1)+' / '+total):'';}
async function load(){
  try{queue=await (await fetch('/api/queue?rater='+encodeURIComponent(rater))).json();}
  catch(e){toast('Could not load the queue');return;}
  total=queue.length;i=0;render();
}
function done(){
  document.getElementById('card').innerHTML=
    '<div class="empty"><h2>All done &mdash; thank you.</h2>'+
    '<p>'+total+' pair'+(total===1?'':'s')+' ranked as <b>'+rater+'</b>.</p>'+
    '<p>Reopen this page anytime to resume where you left off.</p></div>';
  document.getElementById('pbar').style.width='100%';
  document.getElementById('pos').textContent=total+' / '+total;
  document.getElementById('foot').innerHTML='';
}
function pickRow(key,label){
  return '<div class="prow" data-row="'+key+'"><span class="plab">'+label+
    '</span><div class="picks">'+
    '<button data-pick="left">&#9664; Option 1</button>'+
    '<button data-pick="tie">Tie</button>'+
    '<button data-pick="right">Option 2 &#9654;</button></div></div>';
}
function render(){
  const card=document.getElementById('card');
  if(i>=queue.length){done();return;}
  const it=queue[i];cur={winner:null,dims:{}};setProgress();
  const cols='<div class="cols">'+
    '<div class="col"><div class="ch">Option 1</div>'+
      panelHTML(it.left,'left',it.id)+'</div>'+
    '<div class="col"><div class="ch">Option 2</div>'+
      panelHTML(it.right,'right',it.id)+'</div></div>';
  const req=it.input?'<details><summary>Prompt / input</summary><pre>'+
    JSON.stringify(it.input,null,2).replace(/</g,'&lt;')+'</pre></details>':'';
  const rows=pickRow('__overall__','Overall')+
    dims.map(d=>pickRow(d,pretty(d))).join('');
  card.innerHTML=cols+
    '<div class="card"><div class="body">'+req+
    '<div class="prow head"><span class="plab">Which is better?</span></div>'+
    rows+'</div>'+
    '<div class="actions"><span class="rem" id="rem"></span>'+
    '<button class="ghost" id="skip">Skip <span class="kbd">S</span></button>'+
    '<button class="primary" id="next" disabled>Submit <span class="kbd">Enter</span></button>'+
    '</div></div>';
  card.querySelectorAll('.prow[data-row]').forEach(row=>{
    row.querySelectorAll('[data-pick]').forEach(b=>b.onclick=()=>{
      const key=row.dataset.row,val=b.dataset.pick;
      if(key==='__overall__')cur.winner=val; else cur.dims[key]=val;
      row.querySelectorAll('[data-pick]').forEach(x=>x.classList.toggle('on',x===b));
      updateRem();
    });
  });
  card.querySelectorAll('.pane').forEach(pane=>{
    pane.querySelectorAll('.pacts [data-t]').forEach(b=>b.onclick=()=>{
      const t=b.dataset.t;
      pane.querySelector('.rendered').classList.toggle('hidden',t!=='rendered');
      pane.querySelector('.source').classList.toggle('hidden',t!=='source');
      pane.querySelectorAll('[data-t]').forEach(x=>x.classList.toggle('active',x===b));
    });
  });
  document.getElementById('next').onclick=submit;
  document.getElementById('skip').onclick=()=>{i++;render();};
  updateRem();
  document.getElementById('foot').innerHTML=
    'Option 1 / Option 2 sides are shuffled per rater and un-blinded '+
    'server-side &middot; <span class="kbd">Enter</span> submit &middot; '+
    '<span class="kbd">S</span> skip';
}
function updateRem(){
  const left=(cur.winner===null?1:0)+dims.filter(d=>!(d in cur.dims)).length;
  document.getElementById('next').disabled=left>0;
  document.getElementById('rem').textContent=left
    ?left+' pick'+(left===1?'':'s')+' left':'ready to submit';
}
async function submit(){
  if(cur.winner===null||dims.some(d=>!(d in cur.dims)))return;
  try{
    const r=await fetch('/api/rank',{method:'POST',
      headers:{'content-type':'application/json'},
      body:JSON.stringify({id:queue[i].id,rater:rater,
        winner:cur.winner,dims:cur.dims})});
    if(!r.ok)throw 0;
  }catch(e){toast('Save failed - not recorded');return;}
  i++;render();
}
document.addEventListener('keydown',e=>{
  if(i>=queue.length)return;
  if(e.key==='Enter'){if(!document.getElementById('next').disabled)submit();return;}
  if(e.key==='s'||e.key==='S'){i++;render();}
});
load();
</script></body></html>"""


class _RankApp:
    """Server-side state for the blind side-by-side ranking app.

    Aligns two runs through the same
    :func:`~evalcore.pairwise.comparable_samples` the LLM judge uses - by
    ``case_id``, then the n-th sample of A against the n-th of B, since an
    output digest cannot match across two variants - shows each pair as neutral
    "Option 1"/"Option 2" columns whose A/B orientation is seeded per
    ``(rater, item)`` - so position bias is counterbalanced across raters -
    and un-blinds each pick back to *variant* terms server-side before it is
    written as a :class:`~evalcore.models.Preference`.
    """

    def __init__(
        self,
        run_a: models.RunResult,
        run_b: models.RunResult,
        preferences_path,
        dimensions,
        content_ref=None,
        screenshot_ref=None,
        views=None,
    ):
        self.preferences_path = preferences_path
        self.dimensions = dimensions
        self.variant_a = run_a.scorecard.variant.name
        self.variant_b = run_b.scorecard.variant.name
        specs = views or _default_view_specs(content_ref, screenshot_ref)
        # The same filter the LLM judge uses, so both sides of a comparison
        # keep and drop the same samples - see
        # :func:`~evalcore.pairwise.comparable_samples`. Only cases both
        # variants produced are comparable, and only as many samples as the
        # thinner side has.
        a_by = comparable_samples(run_a, content_ref)
        b_by = comparable_samples(run_b, content_ref)
        self.items = []
        for case_id in sorted(a_by.keys() & b_by.keys()):
            for (_, ra), (_, rb) in zip(
                a_by[case_id], b_by[case_id], strict=False
            ):
                panels_a, files_a = _derive_panels(ra.output, specs)
                panels_b, files_b = _derive_panels(rb.output, specs)
                self.items.append(
                    {
                        'case_id': case_id,
                        # A's digest is the pair's identity, matching what
                        # pairwise records, so human and judge picks join.
                        'sample_hash': ra.sample_hash,
                        'input': ra.case.input,
                        'a': {'panels': panels_a, 'files': files_a},
                        'b': {'panels': panels_b, 'files': files_b},
                    }
                )

    def _a_on_left(self, rater: str, idx: int) -> bool:
        """Whether variant A is shown on the left for this (rater, item).

        Deterministic, so orientation is stable across reloads (resumable)
        yet varies across raters - counterbalancing position bias in the
        aggregate without any server-side session state.
        """
        # Non-crypto: a presentation coin-flip, not a secret.
        return random.Random(f'{rater}:{idx}').random() < 0.5  # noqa: S311

    def _to_variant(self, pick: str, a_on_left: bool) -> str:
        """Map a display pick ('left'/'right'/'tie') to variant terms."""
        if pick not in ('left', 'right'):
            return 'tie'
        left_is_a = a_on_left
        if pick == 'left':
            return 'a' if left_is_a else 'b'
        return 'b' if left_is_a else 'a'

    def _ranked_keys(self, rater: str) -> set:
        """(case_id, sample_hash) this rater already ranked for THIS pair.

        Read back from the preferences file so a reopened session resumes;
        filtered to this A/B orientation so an unrelated comparison in the
        same file doesn't hide items.
        """
        return {
            (p.case_id, p.sample_hash)
            for p in store.read_preferences(self.preferences_path)
            if p.rater == rater
            and p.variant_a == self.variant_a
            and p.variant_b == self.variant_b
        }

    def queue_for(self, rater: str) -> list[dict]:
        """Seeded-shuffled blind payload: opaque id + both columns, no
        variant/model identity and no server file paths."""
        order = list(range(len(self.items)))
        # Non-crypto: a stable per-rater presentation order, not a secret.
        random.Random(rater).shuffle(order)  # noqa: S311
        ranked = self._ranked_keys(rater)
        blind = []
        for idx in order:
            item = self.items[idx]
            if (item['case_id'], item['sample_hash']) in ranked:
                continue
            a_left = self._a_on_left(rater, idx)
            left = item['a'] if a_left else item['b']
            right = item['b'] if a_left else item['a']
            blind.append(
                {
                    'id': idx,
                    'input': item['input'],
                    'left': left['panels'],
                    'right': right['panels'],
                }
            )
        return blind

    def file_path(self, idx: int, side: str, rater: str, view: int) -> str:
        """Resolve a display-side artifact request back to its file path."""
        a_left = self._a_on_left(rater, idx)
        if side == 'left':
            which = 'a' if a_left else 'b'
        else:
            which = 'b' if a_left else 'a'
        return self.items[idx][which]['files'][view]

    def record(self, idx: int, rater: str, winner: str, dims: dict) -> None:
        item = self.items[idx]
        a_left = self._a_on_left(rater, idx)
        store.append_preference(
            self.preferences_path,
            models.Preference(
                case_id=item['case_id'],
                sample_hash=item['sample_hash'],
                variant_a=self.variant_a,
                variant_b=self.variant_b,
                rater=rater,
                winner=self._to_variant(winner, a_left),
                dims={d: self._to_variant(p, a_left) for d, p in dims.items()},
            ),
        )


class _RankHandler(_BaseHandler):
    app: _RankApp

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path == '/':
            config = json.dumps({'dims': self.app.dimensions})
            page = _RANK_PAGE.replace('__CONFIG__', config)
            self._send(200, page, 'text/html; charset=utf-8')
        elif parsed.path == '/api/queue':
            rater = (query.get('rater') or ['anon'])[0]
            self._send(200, json.dumps(self.app.queue_for(rater)))
        elif parsed.path == '/api/artifact':
            idx = int((query.get('id') or ['-1'])[0])
            side = (query.get('side') or ['left'])[0]
            view = int((query.get('v') or ['-1'])[0])
            rater = (query.get('rater') or ['anon'])[0]
            try:
                self._serve_file(self.app.file_path(idx, side, rater, view))
            except (OSError, IndexError, KeyError):
                self._send(404, b'', 'application/octet-stream')
        else:
            self._send(404, '{}')

    def do_POST(self):
        if self.path != '/api/rank':
            self._send(404, '{}')
            return
        length = int(self.headers.get('Content-Length', 0))
        payload = json.loads(self.rfile.read(length) or b'{}')
        self.app.record(
            int(payload['id']),
            payload['rater'],
            payload.get('winner', 'tie'),
            payload.get('dims', {}),
        )
        self._send(200, '{"ok": true}')


def serve_rank(
    run_a: models.RunResult,
    run_b: models.RunResult,
    preferences_path: str,
    dimensions: list[str],
    *,
    content_ref: str | None = None,
    screenshot_ref: str | None = 'artifacts.screenshot',
    views: list[dict] | None = None,
    host: str = '127.0.0.1',
    port: int = 8901,
    open_browser: bool = True,
) -> None:
    """Run the blind side-by-side ranking web app until interrupted (Ctrl-C).

    ``run_a``/``run_b`` are the two variants' saved runs; the app aligns them
    by case and sample order (see :class:`_RankApp`) and appends one
    :class:`~evalcore.models.Preference` per ranked pair to
    ``preferences_path``. Panels are derived exactly like :func:`serve`.
    """
    _RankHandler.app = _RankApp(
        run_a,
        run_b,
        preferences_path,
        dimensions,
        content_ref=content_ref,
        screenshot_ref=screenshot_ref,
        views=views,
    )
    server = http.server.ThreadingHTTPServer((host, port), _RankHandler)
    url = f'http://{host}:{port}/?rater=anon'
    n = len(_RankHandler.app.items)
    print(f'blind ranking: {n} pairs on {url} (Ctrl-C to stop)')
    print(f'preferences -> {preferences_path}')
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
