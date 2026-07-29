"""evalcore - a generic, consumer-agnostic evaluation engine.

The engine knows nothing about any particular system under test. A consumer
supplies four things as data/plug-ins:

1. an **adapter** config (how to call its system + the knobs a variant sets),
2. **datasets** (cases with opaque ``input``/``expected`` blobs),
3. **graders** (generic ones here + any custom ones it registers), and
4. a **suite + thresholds** config.

Everything else - runner, comparison/regression engine, results store, and
reporting - lives here and is reused unchanged across consumers.

Quickstart (offline replay, no network or keys)::

    import evalcore as ec

    suite = ec.load_suite('suite.yaml')
    base = ec.run_suite_sync(
        suite, 'baseline', mode='replay'
    ).scorecard
    cand = ec.run_suite_sync(
        suite, 'candidate', mode='replay'
    ).scorecard
    verdict = ec.compare.compare(
        base, cand, suite.thresholds
    ).verdict  # pass | warn | fail

See ``docs/design.md`` for the full design.
"""

from importlib import metadata as _metadata

from evalcore import (
    adapters,
    compare,
    errors,
    graders,
    loader,
    models,
    pairwise,
    rating,
    refs,
    report,
    reporters,
    retry,
    runner,
    store,
    sweep,
)
from evalcore.errors import ConfigError, EvalcoreError
from evalcore.loader import load_cases, load_suite
from evalcore.runner import run_suite, run_suite_sync

try:
    __version__ = _metadata.version('evalcore')
except (
    _metadata.PackageNotFoundError
):  # pragma: no cover - running from source
    __version__ = '0.0.0+unknown'

__all__ = [
    'ConfigError',
    'EvalcoreError',
    '__version__',
    'adapters',
    'compare',
    'errors',
    'graders',
    'load_cases',
    'load_suite',
    'loader',
    'models',
    'pairwise',
    'rating',
    'refs',
    'report',
    'reporters',
    'retry',
    'run_suite',
    'run_suite_sync',
    'runner',
    'store',
    'sweep',
]
