"""LLM-as-judge (rubric scoring) grader, single judge or a panel.

The Tier-2 grader for subjective quality: one or more pinned, strong judge
models score each output 1..scale on a set of rubric dimensions via
structured output (Anthropic forced tool call, OpenAI ``json_schema``).
Scores are normalized to 0..1 and averaged by the runner, so a rubric
dimension - or the ``overall`` mean - can serve as a suite's win metric.

Configure ``judges`` for a **panel**: each judge scores independently, the
grader emits the panel mean per dimension plus each judge's ``overall``, an
inter-judge ``disagreement`` magnitude, and a ``flagged`` rate (cases where
judges disagree by >= ``disagreement_threshold`` raw points - the queue a
human reviewer should look at). A single judge (the default, back-compatible
config) emits just the per-dimension scores and ``overall``.

Judges take optional **image inputs** (``image_refs``) - screenshots or other
rendered artifacts the judge should see alongside the text (e.g. the builder's
rendered page). Images are sent only in live mode; replay is keyed by content.

Each judge call goes through a pluggable ``JudgeClient`` so the grader runs
live (``AnthropicJudgeClient`` / ``OpenAIJudgeClient``) or fully offline
against recorded judgments (``ReplayJudgeClient``), chosen by the run ``mode``
like the target adapter. Judge model + version are recorded on every score;
pin them and treat a judge change as a re-baseline event.

Pairwise (A-vs-B win-rate) judging is a cross-variant operation and lives in
:mod:`evalcore.pairwise`, not here - it needs both variants' per-case outputs
at once, which a per-case grader never sees.
"""

import base64
import functools
import json
import os
import pathlib
import re
import typing

from evalcore import loader, models, refs, retry
from evalcore.graders import base

_ENV_RE = re.compile(r'\$\{([A-Z0-9_]+)\}')


def _env_expand(value):
    """Expand ``${VAR}`` in a judge model string; passthrough otherwise.

    Lets a panel judge's model be env-gated - an unset var expands to ''
    and (in live mode) drops that judge, so 1-vs-2 judges is a config knob.
    """
    if not isinstance(value, str):
        return value
    return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ''), value)


_SYSTEM = (
    'You are a strict, consistent evaluator. Score the content on each '
    'dimension using the full 1..{scale} range, judging only against the '
    'rubric and dimension descriptions - never reward length or verbosity. '
    'Return only the structured per-dimension scores.'
)

_MEDIA_TYPES = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.webp': 'image/webp',
    '.gif': 'image/gif',
}


@typing.runtime_checkable
class JudgeClient(typing.Protocol):
    """Score one item; return ``{scores: {dim: int}, rationale: str}``."""

    async def judge(
        self,
        *,
        key: str,
        system: str,
        user: str,
        dimensions: list[tuple[str, str]],
        scale: int,
        images: list[dict] | None = None,
    ) -> dict: ...


def _score_properties(dimensions: list[tuple[str, str]], scale: int) -> dict:
    return {
        key: {
            'type': 'integer',
            'minimum': 1,
            'maximum': scale,
            'description': desc,
        }
        for key, desc in dimensions
    }


class AnthropicJudgeClient:
    """Live judge backed by the Anthropic SDK (forced single tool call)."""

    def __init__(
        self,
        model: str,
        api_key_env: str = 'ANTHROPIC_API_KEY',
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _tool(self, dimensions: list[tuple[str, str]], scale: int) -> dict:
        return {
            'name': 'score_rubric',
            'description': 'Report per-dimension rubric scores.',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'scores': {
                        'type': 'object',
                        'properties': _score_properties(dimensions, scale),
                        'required': [k for k, _ in dimensions],
                    },
                    'rationale': {'type': 'string'},
                },
                'required': ['scores'],
            },
        }

    async def judge(
        self, *, key, system, user, dimensions, scale, images=None
    ) -> dict:
        import anthropic  # optional dependency (the `judge` extra)

        tool = self._tool(dimensions, scale)
        content: list[dict] = [{'type': 'text', 'text': user}]
        for image in images or []:
            content.append(
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': image['media_type'],
                        'data': image['data'],
                    },
                }
            )
        # Context-managed so the connection pool closes inside the running
        # event loop instead of at GC time after asyncio.run() tore it down.
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
                messages=[{'role': 'user', 'content': content}],
            )
        for block in response.content:
            if (
                getattr(block, 'type', None) == 'tool_use'
                and block.name == tool['name']
            ):
                return dict(block.input)
        return {}


class OpenAIJudgeClient:
    """Live judge via the OpenAI SDK (``response_format`` json_schema)."""

    def __init__(
        self,
        model: str,
        api_key_env: str = 'OPENAI_API_KEY',
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ):
        # Accept a 'provider:model' id (e.g. 'openai:gpt-4o'); SDK wants bare.
        self.model = model.split(':', 1)[1] if ':' in model else model
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _response_format(
        self, dimensions: list[tuple[str, str]], scale: int
    ) -> dict:
        return {
            'type': 'json_schema',
            'json_schema': {
                'name': 'score_rubric',
                'strict': True,
                'schema': {
                    'type': 'object',
                    'properties': {
                        'scores': {
                            'type': 'object',
                            'properties': _score_properties(dimensions, scale),
                            'required': [k for k, _ in dimensions],
                            'additionalProperties': False,
                        },
                        'rationale': {'type': 'string'},
                    },
                    'required': ['scores', 'rationale'],
                    'additionalProperties': False,
                },
            },
        }

    async def judge(
        self, *, key, system, user, dimensions, scale, images=None
    ) -> dict:
        import openai  # optional dependency (the `judge` extra)

        content: list[dict] = [{'type': 'text', 'text': user}]
        for image in images or []:
            data_url = f'data:{image["media_type"]};base64,{image["data"]}'
            content.append(
                {'type': 'image_url', 'image_url': {'url': data_url}}
            )
        async with openai.AsyncOpenAI(
            api_key=os.environ[self.api_key_env]
        ) as client:
            response = await client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0,
                timeout=self.timeout,
                response_format=self._response_format(dimensions, scale),
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': content},
                ],
            )
        body = response.choices[0].message.content
        try:
            return json.loads(body) if body else {}
        except ValueError as exc:
            raise RuntimeError(f'judge returned non-JSON: {exc}') from exc


class ReplayJudgeClient:
    """Offline judge: return recorded judgments keyed by the judged content.

    Fixtures map the exact content string under judgement to a
    ``{scores: {dim: int}, rationale: str}`` dict, so different variants
    (which produce different content) get different scores deterministically.
    """

    def __init__(self, fixtures: str | dict):
        if isinstance(fixtures, str):
            self._fixtures = loader.load_data_file(fixtures)
        else:
            self._fixtures = fixtures

    async def judge(
        self, *, key, system, user, dimensions, scale, images=None
    ) -> dict:
        return dict(self._fixtures.get(key) or {})


def _load_images(refs_values: list) -> list[dict]:
    """Turn resolved image refs (file paths or dicts) into base64 blocks."""
    images: list[dict] = []
    for value in refs_values:
        if value is None:
            continue
        if isinstance(value, dict) and 'data' in value:
            images.append(
                {
                    'media_type': value.get('media_type', 'image/png'),
                    'data': value['data'],
                }
            )
            continue
        if isinstance(value, str):
            path = pathlib.Path(value)
            if not path.is_file():
                continue
            media_type = _MEDIA_TYPES.get(path.suffix.lower(), 'image/png')
            data = base64.b64encode(path.read_bytes()).decode('ascii')
            images.append({'media_type': media_type, 'data': data})
    return images


@base.register('llm_judge', base.GraderType.LLM_AS_JUDGE)
class RubricJudge:
    """Score an output's text on rubric dimensions with an LLM judge/panel.

    Config (suite ``graders`` entry):
        content_ref:  $ref to the text to judge (e.g. ``output.text``)
        dimensions:   list of {key, description}
        scale:        max score (default 5)
        rubric:       optional rubric text embedded in the judge prompt
        context_refs: optional {label: $ref} of extra context for the judge
        image_refs:   optional {label: $ref} of images (paths in
                      ``output.artifacts`` or inline data) shown to the judge
        judges:       optional list of {key, provider(anthropic|openai),
                      model, api_key_env?, replay_path?, judge_version?} - a
                      PANEL. Omit for a single judge configured by the
                      top-level model/replay_path/judge_version.
        disagreement_threshold: raw-point spread at/above which a case is
                      flagged for human review (default 2)
        model/replay_path/judge_version: single-judge shorthand
        client:       an explicit JudgeClient instance (tests); overrides mode
    """

    def __init__(
        self,
        content_ref: str,
        dimensions: list[dict],
        name: str = 'llm_judge',
        scale: int = 5,
        rubric: str | None = None,
        context_refs: dict[str, str] | None = None,
        image_refs: dict[str, str] | None = None,
        judges: list[dict] | None = None,
        disagreement_threshold: float = 2.0,
        model: str | None = None,
        judge_version: str = 'v1',
        replay_path: str | None = None,
        client: JudgeClient | None = None,
        max_tokens: int = 1024,
    ):
        self.name = name
        self.content_ref = content_ref
        self.dimensions = [
            (d['key'], d.get('description', '')) for d in dimensions
        ]
        self.scale = scale
        self.rubric = rubric
        self.context_refs = context_refs or {}
        self.image_refs = image_refs or {}
        self.disagreement_threshold = disagreement_threshold
        self.max_tokens = max_tokens
        self._explicit_client = client
        self._mode = 'http'
        # Normalize to a list of judge specs; a single judge is a panel of 1
        # and produces output identical to the pre-panel grader.
        if judges:
            self.judges = [
                {
                    'key': j.get('key') or j.get('provider') or 'judge',
                    'provider': j.get('provider', 'anthropic'),
                    'model': _env_expand(j.get('model')),
                    'api_key_env': j.get('api_key_env'),
                    'replay_path': j.get('replay_path') or replay_path,
                    'judge_version': j.get('judge_version', judge_version),
                }
                for j in judges
            ]
        else:
            self.judges = [
                {
                    'key': 'judge',
                    'provider': 'anthropic',
                    'model': _env_expand(model),
                    'api_key_env': None,
                    'replay_path': replay_path,
                    'judge_version': judge_version,
                }
            ]
        self._clients: dict[str, JudgeClient] = {}
        self._retry = retry.RetryConfig()

    def set_retry(self, config: retry.RetryConfig) -> None:
        """Runner hook: back off + retry transient judge-client failures."""
        self._retry = config

    def _active_judges(self) -> list[dict]:
        """Judges usable in the current mode - a panel may drop some.

        Only filters a PANEL (2+ judges): a judge with no model (live) or
        no replay_path (replay) is dropped, so an env-gated GPT judge that
        is unset simply leaves a Claude-only panel. A single judge is never
        filtered, so a misconfigured lone judge still raises loudly.
        """
        judges = self.judges
        if self._explicit_client is not None or len(judges) <= 1:
            return judges
        if self._mode == 'replay':
            return [j for j in judges if j['replay_path']] or judges
        return [j for j in judges if j['model']] or judges

    @property
    def is_panel(self) -> bool:
        return len(self._active_judges()) > 1

    @property
    def judge_version(self) -> str:
        """Provenance pin: ``key@version`` per configured judge (a panel
        joins them, comma-separated).

        The runner reads this onto ``Scorecard.judge_version`` so a judge
        model / prompt / scale change surfaces as a re-baseline event rather
        than hiding in each score's ``detail``. Uses the configured judges,
        not the mode-filtered active set, so the pin is stable across
        environments.
        """
        return ','.join(
            f'{j["key"]}@{j["judge_version"]}' for j in self.judges
        )

    def set_mode(self, mode: str) -> None:
        """Runner hook: pick the judge client to match the run mode."""
        self._mode = mode

    def _client_for(self, judge: dict) -> JudgeClient:
        if self._explicit_client is not None:
            return self._explicit_client
        cached = self._clients.get(judge['key'])
        if cached is not None:
            return cached
        if self._mode == 'replay':
            if not judge['replay_path']:
                raise ValueError(
                    f'judge {self.name}/{judge["key"]!r} needs a '
                    f'replay_path for replay mode'
                )
            client: JudgeClient = ReplayJudgeClient(judge['replay_path'])
        elif judge['provider'] == 'openai':
            if not judge['model']:
                raise ValueError(f'judge {judge["key"]!r} needs a model')
            client = OpenAIJudgeClient(
                judge['model'],
                api_key_env=judge['api_key_env'] or 'OPENAI_API_KEY',
                max_tokens=self.max_tokens,
            )
        else:
            if not judge['model']:
                raise ValueError(f'judge {judge["key"]!r} needs a model')
            client = AnthropicJudgeClient(
                judge['model'],
                api_key_env=judge['api_key_env'] or 'ANTHROPIC_API_KEY',
                max_tokens=self.max_tokens,
            )
        self._clients[judge['key']] = client
        return client

    def _prompt(self, content: str, context: dict) -> str:
        parts = []
        if self.rubric:
            parts.append(f'<rubric>\n{self.rubric}\n</rubric>')
        for label, value in context.items():
            parts.append(f'<{label}>\n{value}\n</{label}>')
        parts.append(f'<content>\n{content}\n</content>')
        return '\n\n'.join(parts)

    def _normalize(self, raw) -> float | None:
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return raw / self.scale
        return None

    async def grade(
        self, case: models.Case, output: models.Output
    ) -> list[models.Score]:
        ctx = {
            'input': case.input,
            'expected': case.expected or {},
            'output': output.fields,
            'case': case.model_dump(),
            'artifacts': output.artifacts,
        }
        content = refs.resolve_ref(ctx, self.content_ref)
        if not isinstance(content, str) or not content:
            return self._error_scores('no content to judge')

        context = {
            label: refs.resolve_ref(ctx, ref)
            for label, ref in self.context_refs.items()
        }
        images = _load_images(
            [refs.resolve_ref(ctx, ref) for ref in self.image_refs.values()]
        )
        system = _SYSTEM.format(scale=self.scale)
        user = self._prompt(content, context)

        # Build every client up front: a construction failure (e.g. no
        # replay_path in replay mode) is a configuration error and must
        # raise, unlike a failed judge call which degrades to error scores.
        clients = [(j, self._client_for(j)) for j in self._active_judges()]

        # raw[judge_key][dim] = integer score returned by that judge;
        # rationales[judge_key] = that judge's free-text justification.
        raw: dict[str, dict] = {}
        rationales: dict[str, str | None] = {}
        for judge, client in clients:
            # A transient client failure (429/5xx/timeout) backs off and
            # retries per the suite retry policy; a terminal one degrades to
            # error scores as before. functools.partial (not a closure) keeps
            # the retried call clean of loop-variable capture.
            call = functools.partial(
                client.judge,
                key=content,
                system=system,
                user=user,
                dimensions=self.dimensions,
                scale=self.scale,
                images=images,
            )
            try:
                result = await retry.call_with_retry(call, self._retry)
            except (KeyError, ValueError, RuntimeError) as exc:
                return self._error_scores(f'judge error: {exc}')
            result = result or {}
            raw[judge['key']] = result.get('scores') or {}
            rationales[judge['key']] = result.get('rationale')

        return self._build_scores(case, raw, rationales)

    def _build_scores(
        self,
        case: models.Case,
        raw: dict[str, dict],
        rationales: dict[str, str | None] | None = None,
    ) -> list[models.Score]:
        active = self._active_judges()
        versions = ','.join(f'{j["key"]}@{j["judge_version"]}' for j in active)
        detail = f'judges={versions}'
        rationales = rationales or {}
        out: list[models.Score] = []

        def score(
            metric: str,
            value: float | None,
            judges: list[models.JudgeDetail] | None = None,
        ) -> models.Score:
            return models.Score(
                grader=self.name,
                metric=metric,
                value=value,
                detail=detail,
                case_id=case.id,
                kind='per_case',
                judges=judges or [],
            )

        # Per-judge breakdown behind the aggregate means: the raw 1..scale
        # points each dimension got and the free-text rationale. Attached to
        # the overall score so review can read back *why* without re-running.
        judge_details: list[models.JudgeDetail] = []
        for judge in active:
            jkey = judge['key']
            points = {k: raw.get(jkey, {}).get(k) for k, _ in self.dimensions}
            norm = [
                v
                for k, _ in self.dimensions
                if (v := self._normalize(raw.get(jkey, {}).get(k))) is not None
            ]
            judge_details.append(
                models.JudgeDetail(
                    key=jkey,
                    version=judge['judge_version'],
                    rationale=rationales.get(jkey),
                    points=points,
                    overall=sum(norm) / len(norm) if norm else None,
                )
            )

        # Panel mean per dimension (identical to the single judge's value
        # when there is only one).
        panel_dims: list[float] = []
        for key, _ in self.dimensions:
            per_judge = [
                self._normalize(raw[j['key']].get(key)) for j in active
            ]
            present = [v for v in per_judge if v is not None]
            value = sum(present) / len(present) if present else None
            if value is not None:
                panel_dims.append(value)
            out.append(score(f'{self.name}.{key}', value))

        overall = sum(panel_dims) / len(panel_dims) if panel_dims else None
        out.append(
            score(f'{self.name}.overall', overall, judges=judge_details)
        )

        if len(active) <= 1:
            return out

        # Per-judge overall, so a systematically generous judge is visible.
        for jd in judge_details:
            out.append(score(f'{self.name}.{jd.key}.overall', jd.overall))

        # Inter-judge disagreement, in raw points, per dimension.
        spreads: list[float] = []
        for key, _ in self.dimensions:
            raws = [
                raw[j['key']].get(key)
                for j in active
                if isinstance(raw[j['key']].get(key), (int, float))
                and not isinstance(raw[j['key']].get(key), bool)
            ]
            if len(raws) >= 2:
                spreads.append(max(raws) - min(raws))
        mean_spread = sum(spreads) / len(spreads) if spreads else 0.0
        max_spread = max(spreads) if spreads else 0.0
        out.append(score(f'{self.name}.disagreement', mean_spread))
        out.append(
            score(
                f'{self.name}.flagged',
                1.0 if max_spread >= self.disagreement_threshold else 0.0,
            )
        )
        return out

    def _metric_names(self) -> list[str]:
        active = self._active_judges()
        names = [f'{self.name}.{k}' for k, _ in self.dimensions]
        names.append(f'{self.name}.overall')
        if len(active) > 1:
            names += [f'{self.name}.{j["key"]}.overall' for j in active]
            names += [f'{self.name}.disagreement', f'{self.name}.flagged']
        return names

    def _error_scores(self, detail: str) -> list[models.Score]:
        return [
            models.Score(
                grader=self.name,
                metric=metric,
                value=None,
                detail=detail,
                kind='per_case',
            )
            for metric in self._metric_names()
        ]
