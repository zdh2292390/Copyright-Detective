"""Process-local cap on concurrent outbound LLM API calls."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

DEFAULT_MAX_CONCURRENT_API = 500
DEFAULT_ACQUIRE_TIMEOUT_SECONDS = 300.0


class ApiConcurrencyTimeout(RuntimeError):
    """Raised when no API slot becomes available before the wait timeout."""


def max_concurrent_api_calls() -> int:
    raw = os.environ.get("COPYRIGHT_DETECTIVE_MAX_CONCURRENT_API", "")
    if not str(raw).strip():
        return DEFAULT_MAX_CONCURRENT_API
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_MAX_CONCURRENT_API


def _acquire_timeout_seconds() -> Optional[float]:
    raw = os.environ.get("COPYRIGHT_DETECTIVE_API_ACQUIRE_TIMEOUT_SECONDS", "")
    if not str(raw).strip():
        return DEFAULT_ACQUIRE_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_ACQUIRE_TIMEOUT_SECONDS
    if value < 0:
        return None
    return value


_MAX_CONCURRENT_API = max_concurrent_api_calls()
_API_SEMAPHORE = threading.BoundedSemaphore(_MAX_CONCURRENT_API)


def acquire_api_concurrency(timeout: Optional[float] = None) -> None:
    wait = _acquire_timeout_seconds() if timeout is None else timeout
    if wait is None:
        _API_SEMAPHORE.acquire()
        return
    if not _API_SEMAPHORE.acquire(timeout=max(0.0, float(wait))):
        raise ApiConcurrencyTimeout(
            f"Too many concurrent API requests (limit {_MAX_CONCURRENT_API}). "
            "Please retry shortly."
        )


def release_api_concurrency() -> None:
    _API_SEMAPHORE.release()


@contextmanager
def limit_api_concurrency(timeout: Optional[float] = None) -> Iterator[None]:
    acquire_api_concurrency(timeout=timeout)
    try:
        yield
    finally:
        release_api_concurrency()
