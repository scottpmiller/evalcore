"""Pairwise (A-vs-B) judging - the headline subjective win-rate.

Rubric scoring asks "how good is this output, 1..5"; pairwise asks the
sharper question "is A better than B for this case?" and reports A's
win-rate. It is a cross-variant operation the per-variant runner can't
express as a grader (it needs both variants' output for the same case at
once), so it lives here: align two runs by case, ask a judge to pick a winner
per case, aggregate.

Alignment is by ``case_id`` and then sample order within the case. It cannot
key on ``sample_hash``: that is a digest of the output, and the whole point of
a comparison is that the two variants produced different output, so the hashes
never match. A case's samples are repeat draws of one input, so pairing the
n-th of A with the n-th of B is as meaningful as any other pairing. Each
outcome records A's hash, so a pair traces back to a concrete response.

Position bias (LLMs favour whichever option they see first) is handled by
**counterbalancing**: each pair is judged in both orders and a pick that
flips with order collapses to a tie. The judge client is pluggable and
mode-selected (Anthropic / OpenAI / replay) exactly like the rubric judge.
"""

import json
import os
import typing

from evalcore import loader, models, refs

_SYSTEM = (
    'You are a careful, unbiased evaluator comparing two candidate '
    'outputs for the same request. Judge only on quality against the '
    'rubric; ignore which option is shown first. Pick the better option, '
    'or "tie" if they are genuinely equal.'
)

_Pick = typing.Literal['first', 'second', 'tie']


@typing.runtime_checkable
class PairwiseClient(typing.Protocol):
    """Pick the better of two outputs; return 'first' | 'second' | 'tie'."""

    async def compare(
        self, *, system: str, user: str, first: str, second: str
    ) -> str: ...


def _tool() -> dict:
    return {
        'name': 'pick_winner',
        'description': 'Pick the better output for this request.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'winner': {
                    'type': 'string',
                    'enum': ['first', 'second', 'tie'],
                },
                'rationale': {'type': 'string'},
            },
            'required': ['winner'],
        },
    }


class AnthropicPairwiseClient:
    """Live pairwise judge via the Anthropic SDK (forced tool call)."""

    def __init__(
        self,
        model: str,
        api_key_env: str = 'ANTHROPIC_API_KEY',
        max_tokens: int = 512,
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def compare(self, *, system, user, first, second) -> str:
        import anthropic  # optional dependency (the `judge` extra)

        tool = _tool()
        async with anthropic.AsyncAnthropic(
            api_key=os.environ[self.api_key_env]
        ) as client:
            response = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0,
                timeout=self.timeout,
                system=system,
                tools=[tool],
                tool_choice={'type': 'tool', 'name': tool['name']},
                messages=[{'role': 'user', 'content': user}],
            )
        for block in response.content:
            if (
                getattr(block, 'type', None) == 'tool_use'
                and block.name == tool['name']
            ):
                return dict(block.input).get('winner', 'tie')
        return 'tie'


class OpenAIPairwiseClient:
    """Live pairwise judge via the OpenAI SDK (``json_schema``)."""

    def __init__(
        self,
        model: str,
        api_key_env: str = 'OPENAI_API_KEY',
        max_tokens: int = 512,
        timeout: float = 30.0,
    ):
        self.model = model.split(':', 1)[1] if ':' in model else model
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def compare(self, *, system, user, first, second) -> str:
        import openai  # optional dependency (the `judge` extra)

        response_format = {
            'type': 'json_schema',
            'json_schema': {
                'name': 'pick_winner',
                'strict': True,
                'schema': {
                    'type': 'object',
                    'properties': {
                        'winner': {
                            'type': 'string',
                            'enum': ['first', 'second', 'tie'],
                        },
                        'rationale': {'type': 'string'},
                    },
                    'required': ['winner', 'rationale'],
                    'additionalProperties': False,
                },
            },
        }
        async with openai.AsyncOpenAI(
            api_key=os.environ[self.api_key_env]
        ) as client:
            response = await client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0,
                timeout=self.timeout,
                response_format=response_format,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user},
                ],
            )
        body = response.choices[0].message.content
        try:
            return (json.loads(body) if body else {}).get('winner', 'tie')
        except ValueError:
            return 'tie'


class ReplayPairwiseClient:
    """Offline pairwise judge keyed by the (unordered) pair of contents.

    Fixtures are a list of ``{a, b, winner, rationale?}`` where ``winner``
    is the winning *content* string (or ``'tie'``). Lookups are
    order-independent, so counterbalancing returns a stable result offline.
    """

    _SEP = '\n<VS>\n'

    def __init__(self, fixtures: str | list):
        if isinstance(fixtures, str):
            fixtures = loader.load_data_file(fixtures)
        # load_data_file returns {} for a bare list file under JSON fallback;
        # accept either a list or a {'pairs': [...]} mapping.
        rows = (
            fixtures
            if isinstance(fixtures, list)
            else (fixtures.get('pairs', []))
        )
        self._winners: dict[str, str] = {}
        for row in rows:
            self._winners[self._key(row['a'], row['b'])] = row['winner']

    def _key(self, x: str, y: str) -> str:
        return self._SEP.join(sorted([x, y]))

    async def compare(self, *, system, user, first, second) -> str:
        winner = self._winners.get(self._key(first, second))
        if winner is None or winner == 'tie':
            return 'tie'
        if winner == first:
            return 'first'
        if winner == second:
            return 'second'
        return 'tie'


def build_pairwise_client(mode: str, config: dict) -> PairwiseClient:
    """Pick a pairwise client for the run mode from a config mapping."""
    if mode == 'replay':
        if not config.get('replay_path'):
            raise ValueError('pairwise replay mode needs replay_path')
        return ReplayPairwiseClient(config['replay_path'])
    model = config.get('model')
    if not model:
        raise ValueError('pairwise live mode needs a model')
    if config.get('provider') == 'openai':
        return OpenAIPairwiseClient(model)
    return AnthropicPairwiseClient(model)


def comparable_samples(
    run: models.RunResult, content_ref: str | None
) -> dict[str, list[tuple[str, models.CaseResult]]]:
    """``case_id -> [(content, result)]`` for the samples a comparison can use.

    A sample is comparable when the invocation succeeded and - if a
    ``content_ref`` is given - that ref resolves to non-empty text. An errored
    output has nothing meaningful to judge or to show a rater, and empty
    content gives a judge nothing to weigh and a human an empty panel.
    ``content`` is ``''`` when no ``content_ref`` was asked for.

    Every path that pairs two runs has to filter through here, because
    alignment within a case is positional: a sample one side keeps and the
    other drops shifts every pair after it. The LLM judge would then compare
    A's n-th sample against B's n-th while the human panel compared that same
    A sample against a different B, and
    :func:`~evalcore.rating.compute_pairwise_agreement` joins the two on A's
    digest - so it would score two different comparisons as if they were one.
    """
    out: dict[str, list[tuple[str, models.CaseResult]]] = {}
    for result in run.results:
        if result.output.error:
            continue
        content = ''
        if content_ref:
            ctx = {
                'input': result.case.input,
                'expected': result.case.expected or {},
                'output': result.output.fields,
                'case': result.case.model_dump(),
                'artifacts': result.output.artifacts,
            }
            resolved = refs.resolve_ref(ctx, content_ref)
            if not (isinstance(resolved, str) and resolved):
                continue
            content = resolved
        out.setdefault(result.case.id, []).append((content, result))
    return out


def _prompt(first: str, second: str, context: dict, rubric: str | None) -> str:
    parts = []
    if rubric:
        parts.append(f'<rubric>\n{rubric}\n</rubric>')
    for label, value in context.items():
        parts.append(f'<{label}>\n{value}\n</{label}>')
    parts.append(f'<option_1>\n{first}\n</option_1>')
    parts.append(f'<option_2>\n{second}\n</option_2>')
    return '\n\n'.join(parts)


async def judge_pairwise(
    run_a: models.RunResult,
    run_b: models.RunResult,
    *,
    content_ref: str,
    client: PairwiseClient,
    context_refs: dict[str, str] | None = None,
    rubric: str | None = None,
    counterbalance: bool = True,
    judge_name: str = 'pairwise',
    judge_version: str = 'v1',
) -> models.PairwiseResult:
    """Compare two runs case-by-case and report A's win-rate."""
    a_map = comparable_samples(run_a, content_ref)
    b_map = comparable_samples(run_b, content_ref)
    context_refs = context_refs or {}

    # Comparable pairs: only cases both variants produced, and within a case
    # the n-th sample of A against the n-th of B. strict=False because an
    # uneven sample count is expected - a case that errored on one side has
    # fewer usable outputs there, and the extras have nothing to compare to.
    pairs = [
        (case_id, a_side, b_side)
        for case_id in sorted(a_map.keys() & b_map.keys())
        for a_side, b_side in zip(a_map[case_id], b_map[case_id], strict=False)
    ]

    outcomes: list[models.PairwiseOutcome] = []
    a_wins = b_wins = ties = 0
    for case_id, (content_a, result_a), (content_b, _) in pairs:
        context = {
            label: refs.resolve_ref(
                {
                    'input': result_a.case.input,
                    'expected': result_a.case.expected or {},
                    'case': result_a.case.model_dump(),
                },
                ref,
            )
            for label, ref in context_refs.items()
        }
        # Order 1: A shown first.
        pick1 = await client.compare(
            system=_SYSTEM,
            user=_prompt(content_a, content_b, context, rubric),
            first=content_a,
            second=content_b,
        )
        winner = {'first': 'a', 'second': 'b', 'tie': 'tie'}[pick1]
        detail = f'order1={pick1}'
        if counterbalance:
            # Order 2: B shown first; a flip vs order 1 means position bias.
            pick2 = await client.compare(
                system=_SYSTEM,
                user=_prompt(content_b, content_a, context, rubric),
                first=content_b,
                second=content_a,
            )
            winner2 = {'first': 'b', 'second': 'a', 'tie': 'tie'}[pick2]
            detail += f' order2={pick2}'
            if winner != winner2:
                winner = 'tie'  # inconsistent under swap -> not a real win

        if winner == 'a':
            a_wins += 1
        elif winner == 'b':
            b_wins += 1
        else:
            ties += 1
        outcomes.append(
            models.PairwiseOutcome(
                case_id=case_id,
                sample_hash=result_a.sample_hash,
                winner=winner,
                detail=detail,
            )
        )

    n = len(outcomes)
    win_rate_a = (a_wins + 0.5 * ties) / n if n else None
    return models.PairwiseResult(
        project=run_a.scorecard.project,
        suite=run_a.scorecard.suite,
        variant_a=run_a.scorecard.variant.name,
        variant_b=run_b.scorecard.variant.name,
        judge_name=judge_name,
        judge_version=judge_version,
        n=n,
        a_wins=a_wins,
        b_wins=b_wins,
        ties=ties,
        win_rate_a=win_rate_a,
        outcomes=outcomes,
    )
