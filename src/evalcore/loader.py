"""Loading of suite configs and datasets.

Data files are YAML when PyYAML is available (nice for multi-line content
payloads) and fall back to JSON otherwise, so the engine has no hard non-stdlib
data dependency.
"""

import hashlib
import json
import pathlib

import pydantic

from evalcore import models
from evalcore.retry import RetryConfig  # re-exported: SuiteConfig.retry type


def content_hash(data) -> str:
    """Stable short digest of parsed content (canonical JSON, sha256).

    Versions content independently of any VCS: two loads hash equal iff
    the parsed data is equal, regardless of file formatting or location.
    """
    text = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]


def dataset_hash(cases: list[models.Case]) -> str:
    """Content digest of a loaded case list (order-independent)."""
    dumped = sorted(
        (case.model_dump(mode='json') for case in cases),
        key=lambda data: str(data.get('id')),
    )
    return content_hash(dumped)


def load_data_file(path: str | pathlib.Path) -> dict:
    """Load a YAML or JSON mapping from ``path``."""
    text = pathlib.Path(path).read_text(encoding='utf-8')
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except ImportError:
        return json.loads(text) if text.strip() else {}


class SuiteConfig(pydantic.BaseModel):
    """Parsed ``evals/suites/<name>.yaml`` for one consumer suite."""

    project: str
    suite: str
    dataset: str
    dataset_version: str = 'v1'
    mode_default: str = 'http'
    replay_fixtures: str | None = None
    adapter: dict
    graders: list[dict] = pydantic.Field(default_factory=list)
    # Consumer modules to import before a run, so their registered custom
    # adapters/graders exist by the time a `type` here is looked up. Imported
    # by the runner, NOT by load_suite - loading a suite stays side-effect
    # free. See evalcore.plugins.
    plugins: list[str] = pydantic.Field(default_factory=list)
    variants: dict[str, dict] = pydantic.Field(default_factory=dict)
    n_samples: int = 1
    # Max concurrent (case, sample) invocations. Above 1, the adapter and
    # per-case graders must tolerate concurrent calls.
    concurrency: int = 1
    # Transient-failure retry policy for adapter calls (default: no retry).
    retry: RetryConfig = pydantic.Field(default_factory=RetryConfig)
    thresholds: dict = pydantic.Field(default_factory=dict)
    # Computed by load_suite (content digest of the raw file); not for
    # authors to set in the suite file.
    suite_hash: str | None = None


def load_suite(path: str | pathlib.Path) -> SuiteConfig:
    """Load and validate a suite config file.

    ``dataset`` and ``replay_fixtures`` are resolved relative to the suite
    file's own directory (unless already absolute), so a suite is portable and
    runnable from any working directory.
    """
    path = pathlib.Path(path)
    raw = load_data_file(path)
    config = SuiteConfig.model_validate(raw)
    # Hash the raw content BEFORE resolving paths so the digest is stable
    # across checkout locations.
    config.suite_hash = content_hash(raw)
    base = path.parent

    def _resolve(value: str) -> str:
        candidate = pathlib.Path(value)
        return value if candidate.is_absolute() else str(base / candidate)

    config.dataset = _resolve(config.dataset)
    if config.replay_fixtures:
        config.replay_fixtures = _resolve(config.replay_fixtures)
    # Grader specs may carry their own suite-relative fixtures (e.g. the LLM
    # judge's recorded judgments under ``replay_path``) - at the top level
    # for a single judge, or per judge in a ``judges`` panel list.
    for spec in config.graders:
        if isinstance(spec.get('replay_path'), str):
            spec['replay_path'] = _resolve(spec['replay_path'])
        for judge in spec.get('judges', []) or []:
            if isinstance(judge.get('replay_path'), str):
                judge['replay_path'] = _resolve(judge['replay_path'])
    # The pairwise judge's recorded judgments are suite-relative too.
    pairwise = config.thresholds.get('pairwise')
    if isinstance(pairwise, dict) and isinstance(
        pairwise.get('replay_path'), str
    ):
        pairwise['replay_path'] = _resolve(pairwise['replay_path'])
    return config


def load_cases(dataset_dir: str | pathlib.Path) -> list[models.Case]:
    """Load every ``cases/*.yaml|json`` file under a dataset directory."""
    cases_dir = pathlib.Path(dataset_dir) / 'cases'
    if not cases_dir.is_dir():
        raise FileNotFoundError(f'no cases directory at {cases_dir}')
    cases: list[models.Case] = []
    paths = sorted(
        p
        for p in cases_dir.iterdir()
        if p.suffix in ('.yaml', '.yml', '.json')
    )
    for path in paths:
        data = load_data_file(path)
        data.setdefault('id', path.stem)
        cases.append(models.Case.model_validate(data))
    if not cases:
        raise FileNotFoundError(f'no case files found in {cases_dir}')
    return cases
