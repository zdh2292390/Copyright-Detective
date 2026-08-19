"""Metrics module - Evaluation tools and loggers."""

from .logger import RougeEvalLogger, FactRecallLogger

# Lazy import knowmem to avoid KeyError/circular import during common package load
def __getattr__(name):
    if name == "eval":
        from .knowmem import eval as _eval
        return _eval
    if name == "get_prefix_before_words_occur":
        from .knowmem import get_prefix_before_words_occur as _fn
        return _fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RougeEvalLogger",
    "FactRecallLogger",
    "eval",
    "get_prefix_before_words_occur",
]