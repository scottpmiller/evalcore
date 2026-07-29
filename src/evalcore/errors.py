"""Exception types raised by evalcore.

Everything evalcore raises derives from :class:`EvalcoreError`, so a caller can
catch all of it with a single ``except EvalcoreError``. Where a builtin type is
also idiomatic, the subclass inherits it too - e.g. :class:`ConfigError` is
also a ``ValueError`` - so existing ``except ValueError`` handlers keep
working.
"""


class EvalcoreError(Exception):
    """Base class for every error raised by evalcore."""


class ConfigError(EvalcoreError, ValueError):
    """Invalid configuration.

    Raised for an unknown or duplicate registry ``type`` (adapter/grader/
    reporter) or a malformed suite/threshold config.
    """
