"""Frozen provider connection settings shared by one task's API calls."""
from contextlib import contextmanager
from contextvars import ContextVar

_SETTINGS = ContextVar("analysis_api_connection_settings", default=None)


def current_api_task_settings():
    return _SETTINGS.get()


@contextmanager
def api_task_scope(settings):
    binding = _SETTINGS.set(dict(settings))
    try:
        yield
    finally:
        _SETTINGS.reset(binding)
