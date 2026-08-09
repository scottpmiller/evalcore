"""Markdown rendering of scorecards and comparisons (CI / PR comments)."""

from evalcore import models


def _fmt(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:.4f}'


def _fmt_delta(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:+.4f}'


def render_scorecard(scorecard: models.Scorecard) -> str:
    """Render a single scorecard as a Markdown section."""
    lines = [
        f'### {scorecard.project}/{scorecard.suite} '
        f'- `{scorecard.variant.name}`',
        '',
        f'- model: `{scorecard.model_id}`  '
        f'mode: `{scorecard.mode}`  '
        f'dataset: `{scorecard.dataset_version}`  '
        f'cases: {scorecard.n_cases}x{scorecard.n_samples}',
        '',
        '| metric | value |',
        '| --- | ---: |',
    ]
    lines += [
        f'| {m.metric} | {_fmt(m.value)} |' for m in scorecard.metrics.values()
    ]
    return '\n'.join(lines)


_VERDICT_BADGE = {'pass': '**PASS**', 'warn': '**WARN**', 'fail': '**FAIL**'}


def render_comparison(comparison: models.Comparison) -> str:
    """Render a candidate-vs-baseline comparison as Markdown."""
    badge = _VERDICT_BADGE.get(comparison.verdict, comparison.verdict)
    lines = [
        f'## {badge} - {comparison.project}/{comparison.suite}',
        '',
        f'`{comparison.candidate_variant}` vs '
        f'`{comparison.baseline_variant}` - {comparison.summary}',
        '',
        '| metric | baseline | candidate | delta |',
        '| --- | ---: | ---: | ---: |',
    ]
    for delta in comparison.deltas:
        marker = (
            f' **({comparison.win})**'
            if delta.metric == comparison.win_metric
            else ''
        )
        lines.append(
            f'| {delta.metric}{marker} | {_fmt(delta.baseline)} '
            f'| {_fmt(delta.candidate)} | {_fmt_delta(delta.delta)} |'
        )
    if comparison.guardrails:
        lines += ['', '**Guardrails**', '']
        for guard in comparison.guardrails:
            # Tri-state: None is a rule that could not run (a relative
            # guardrail with no baseline), which is not a breach.
            mark = (
                'skipped'
                if guard.passed is None
                else 'ok'
                if guard.passed
                else 'BREACH'
            )
            lines.append(f'- [{mark}] `{guard.metric}` - {guard.detail}')
    return '\n'.join(lines)


def render_agreement(result: models.AgreementResult) -> str:
    """Render a judge<->human agreement result as Markdown."""
    lines = [
        f'### judge<->human agreement - `{result.judge_name}`',
        '',
        f'- {result.n_ratings} ratings from {result.n_raters} rater(s), '
        f'scale 1..{result.scale}',
        f'- overall: MAE {_fmt(result.overall_mae)}, '
        f'r {_fmt(result.overall_correlation)}',
        '',
        '| dimension | n | human | judge | MAE | corr |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for dim in result.dimensions:
        lines.append(
            f'| {dim.dimension} | {dim.n} | {_fmt(dim.human_mean)} '
            f'| {_fmt(dim.judge_mean)} | {_fmt(dim.mae)} '
            f'| {_fmt(dim.correlation)} |'
        )
    return '\n'.join(lines)


def render_sweep(result: models.SweepResult) -> str:
    """Render an N-way sweep as a ranking + a metric x variant leaderboard."""
    variants = [c.variant for c in result.entries] or sorted(
        {v for row in result.matrix.values() for v in row}
    )
    lines = [f'## sweep - {result.project}/{result.suite}', '']
    if result.win_metric:
        direction = 'higher' if result.win_higher_is_better else 'lower'
        lines += [
            f'ranked by `{result.win_metric}` ({direction} is better)',
            '',
            '| rank | variant | win |',
            '| ---: | --- | ---: |',
        ]
        lines += [
            f'| {e.rank} | `{e.variant}` | {_fmt(e.win_value)} |'
            for e in result.entries
        ]
        lines.append('')

    header = '| metric | ' + ' | '.join(variants) + ' |'
    sep = '| --- |' + ' ---: |' * len(variants)
    lines += ['**Leaderboard**', '', header, sep]
    for metric, row in result.matrix.items():
        present = [v for v in row.values() if v is not None]
        best = max(present) if present else None  # display hint only
        cells = []
        for variant in variants:
            value = row.get(variant)
            text = _fmt(value)
            if value is not None and best is not None and value == best:
                text = f'**{text}**'
            cells.append(text)
        lines.append(f'| {metric} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def render_pairwise(result: models.PairwiseResult) -> str:
    """Render an A-vs-B pairwise win-rate result as Markdown."""
    rate = _fmt(result.win_rate_a)
    return '\n'.join(
        [
            f'## pairwise - {result.project}/{result.suite}',
            '',
            f'`{result.variant_a}` (A) vs `{result.variant_b}` (B), '
            f'judge `{result.judge_name}@{result.judge_version}`',
            '',
            f'- **A win-rate: {rate}** (ties = half) over {result.n} cases',
            f'- A wins {result.a_wins} - B wins {result.b_wins} '
            f'- ties {result.ties}',
        ]
    )


def render_preferences(result: models.PreferenceResult) -> str:
    """Render a human side-by-side win-rate (overall + per dimension)."""
    lines = [
        f'## human preferences - {result.project}/{result.suite}',
        '',
        f'`{result.variant_a}` (A) vs `{result.variant_b}` (B), '
        f'{result.n} preferences from {result.n_raters} rater(s)',
        '',
        f'- **A win-rate: {_fmt(result.win_rate_a)}** (ties = half) '
        f'- A wins {result.a_wins} - B wins {result.b_wins} '
        f'- ties {result.ties}',
        '',
        '| dimension | n | A wins | B wins | ties | A win-rate |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for dim in result.dimensions:
        lines.append(
            f'| {dim.dimension} | {dim.n} | {dim.a_wins} | {dim.b_wins} '
            f'| {dim.ties} | {_fmt(dim.win_rate_a)} |'
        )
    return '\n'.join(lines)


def render_pairwise_agreement(result: models.PairwiseAgreement) -> str:
    """Render human-panel vs LLM-judge head-to-head agreement as Markdown."""
    lines = [
        f'## pairwise agreement - human vs `{result.judge_name}`',
        '',
        f'`{result.variant_a}` (A) vs `{result.variant_b}` (B)',
        '',
        f'- **agreement: {_fmt(result.agreement_rate)}** '
        f'({result.agree}/{result.n} cases pick the same winner)',
        f'- A win-rate: human {_fmt(result.human_win_rate_a)} '
        f'- judge {_fmt(result.judge_win_rate_a)}',
        '',
        '| case | human | judge | agree |',
        '| --- | :---: | :---: | :---: |',
    ]
    for case in result.outcomes:
        mark = 'yes' if case.agree else 'NO'
        lines.append(
            f'| {case.case_id} | {case.human} | {case.judge} | {mark} |'
        )
    return '\n'.join(lines)
