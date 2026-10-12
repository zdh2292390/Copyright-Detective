"""Process-local cap on concurrent outbound LLM API calls."""

from __future__ import annotations

import os
import math
import threading
from contextlib import contextmanager
from contextvars import ContextVar
import time
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
    if not math.isfinite(value):
        return DEFAULT_ACQUIRE_TIMEOUT_SECONDS
    if value < 0:
        return None
    return value


_MAX_CONCURRENT_API = max_concurrent_api_calls()
_API_SEMAPHORE = threading.BoundedSemaphore(_MAX_CONCURRENT_API)


def acquire_api_concurrency(timeout: Optional[float] = None) -> None:
    from src.run_cancellation import check_cancellation, current_cancellation_token
    wait = _acquire_timeout_seconds() if timeout is None else timeout
    if wait is not None:
        wait = float(wait)
        if not math.isfinite(wait):
            raise ValueError("API slot timeout must be finite.")
        wait = max(0.0, wait)
    if current_cancellation_token() is None:
        acquired = _API_SEMAPHORE.acquire() if wait is None else _API_SEMAPHORE.acquire(timeout=wait)
    else:
        deadline = None if wait is None else time.monotonic() + wait
        while True:
            check_cancellation()
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            acquired = _API_SEMAPHORE.acquire(timeout=0.05 if remaining is None else min(0.05, remaining))
            if acquired:
                try:
                    check_cancellation()
                except BaseException:
                    _API_SEMAPHORE.release()
                    raise
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
    if not acquired:
        raise ApiConcurrencyTimeout(
            f"Too many concurrent API requests (limit {_MAX_CONCURRENT_API}). "
            "Please retry shortly."
        )


def release_api_concurrency() -> None:
    _API_SEMAPHORE.release()


class _ApiSlot:
    def __init__(self):
        self._lock = threading.Lock()
        self._users = 1

    def retain(self):
        with self._lock:
            if self._users == 0:
                return False
            self._users += 1
            return True

    def release(self):
        with self._lock:
            self._users -= 1
            finished = self._users == 0
        if finished:
            release_api_concurrency()


_CURRENT_SLOT = ContextVar("analysis_api_slot", default=None)


def retain_api_slot():
    """Keep an interruptible request's permit until its network worker exits."""
    slot = _CURRENT_SLOT.get()
    return slot.release if slot is not None and slot.retain() else None


@contextmanager
def limit_api_concurrency(timeout: Optional[float] = None) -> Iterator[None]:
    acquire_api_concurrency(timeout=timeout)
    slot = _ApiSlot()
    binding = _CURRENT_SLOT.set(slot)
    try:
        yield
    finally:
        _CURRENT_SLOT.reset(binding)
        slot.release()
