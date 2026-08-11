"""Import consumer modules so their registrations take effect.

A custom adapter or grader reaches the engine through a registry keyed by the
``type`` string a suite names, and the registry entry is written by the
``@register`` decorator *at import time*. Nothing in evalcore imports a
consumer's module, so a suite naming ``type: my_grader`` fails as an unknown
type until something has imported the module that defines it.

Two things do that importing: a suite's own ``plugins:`` list, and the CLI's
``--plugins`` flag. Prefer the suite key. It keeps a suite self-contained -
the same file then works from the CLI, from the Python API, and from another
consumer's harness, with nothing to remember at the call site. The flag stays
for a suite that does not declare its own, and for adding a module to a run
without editing the suite (which would change ``suite_hash``).

Importing is deliberately **not** part of loading a suite.
:func:`~evalcore.loader.load_suite` parses and hashes YAML and executes
nothing, so reading a suite, rendering a report, or diffing two of them stays
free of side effects. The import happens when a run starts - which is already
the point where the suite's adapter gets to make network calls and spend
money, so it is the honest place to also let it run code.
"""

import importlib
import os
import sys

from evalcore import errors

#: Names already imported. ``sys.modules`` is the real cache - this only keeps
#: a repeated run in one process from re-attempting an import that failed, and
#: lets :func:`load` report what it actually did.
_loaded: set[str] = set()


def allow_cwd_imports() -> None:
    """Put the working directory on ``sys.path``.

    The console script (unlike ``python -m``) does not, so a
    ``plugins: [evals.graders]`` in a suite run from a repo root would never
    resolve. Only the CLI calls this: a library has no business editing the
    path of the process that imported it.
    """
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def load(names: list[str] | None) -> list[str]:
    """Import each module in ``names`` once; return those newly imported.

    Args:
        names: Module paths (``evals.graders``). Blanks are skipped, so a
            comma-split CLI string can be passed straight in.

    Returns:
        The names imported by this call, in order. Already-imported names are
        omitted rather than repeated.

    Raises:
        ConfigError: If a module cannot be imported. The name is quoted in the
            message, because the failure is nearly always a typo in the suite
            or a module that is not importable from the working directory.

    """
    fresh: list[str] = []
    for raw in names or ():
        name = raw.strip()
        if not name or name in _loaded:
            continue
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise errors.ConfigError(
                f'plugin module {name!r} could not be imported: {exc}'
            ) from exc
        _loaded.add(name)
        fresh.append(name)
    return fresh
