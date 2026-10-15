"""UI-free, per-run cancellation for API waits and background work."""
from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FutureTimeout
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from threading import Condition, Event, Lock, Thread
from typing import Callable


class AnalysisCancelled(BaseException):
    """A user stop signal must escape provider retry/error conversion handlers."""


class CancellationToken:
    def __init__(self):
        self._event = Event()
        self._condition = Condition()
        self._pending = 0

    @property
    def cancelled(self):
        return self._event.is_set()

    def is_cancelled(self):
        return self.cancelled

    def cancel(self):
        with self._condition:
            changed = not self.cancelled
            self._event.set()
            self._condition.notify_all()
        return changed

    def check(self):
        if self.cancelled:
            raise AnalysisCancelled("Analysis stopped by user. Saved progress is preserved.")

    @property
    def pending_count(self):
        with self._condition:
            return self._pending

    def wait_for_idle(self, timeout=None):
        with self._condition:
            return self._condition.wait_for(lambda: self._pending == 0, timeout)

    def _begin_call(self):
        with self._condition:
            self.check()
            self._pending += 1

    def _end_call(self):
        with self._condition:
            self._pending -= 1
            self._condition.notify_all()


_TOKEN = ContextVar("analysis_cancellation_token", default=None)
_IN_CALL = ContextVar("analysis_interruptible_api_call", default=False)


def current_cancellation_token():
    return _TOKEN.get()


def check_cancellation():
    token = _TOKEN.get()
    if token is not None:
        token.check()


@contextmanager
def cancellation_scope(token):
    binding = _TOKEN.set(token)
    try:
        yield token
    finally:
        _TOKEN.reset(binding)


def cancellation_sleep(seconds):
    import time
    token = _TOKEN.get()
    if token is None:
        time.sleep(max(0.0, float(seconds)))
        return
    token.check()
    token._event.wait(max(0.0, float(seconds)))
    token.check()


def cancellable_call(invoke: Callable, *args, **kwargs):
    """Interrupt waiting for a UI-free network request without killing threads.

    The HTTP request may finish at the provider after cancellation. Its worker
    retains the API concurrency permit until it exits. It never owns checkpoint
    writes or Streamlit state. Already saved responses remain authoritative.
    """
    token = _TOKEN.get()
    if token is None:
        return invoke(*args, **kwargs)
    token.check()
    if _IN_CALL.get():
        return invoke(*args, **kwargs)
    from src.api_concurrency import retain_api_slot
    release_slot = None
    closed_response = False
    close_lock = Lock()

    def close_discarded_response(value):
        nonlocal closed_response
        with close_lock:
            if closed_response:
                return
            closed_response = True
        # Disposal is best-effort: even an invalid response adapter's
        # attribute lookup or BaseException must not strand the result Future.
        try:
            close = getattr(value, "close", None)
            if callable(close):
                close()
        except BaseException:
            pass

    token._begin_call()
    try:
        # Every allocation after increasing pending_count belongs to rollback.
        release_slot = retain_api_slot()
        result = Future()
        context = copy_context()

        def discard_published_response(future):
            try:
                value = future.result()
            except BaseException:
                return
            close_discarded_response(value)

        def execute():
            binding = None
            value = None
            failure = None
            try:
                binding = _IN_CALL.set(True)
                token.check()
                value = invoke(*args, **kwargs)
                if token.cancelled:
                    close_discarded_response(value)
                    token.check()
            except BaseException as exc:
                failure = exc
            finally:
                try:
                    try:
                        if binding is not None:
                            _IN_CALL.reset(binding)
                    finally:
                        try:
                            if release_slot is not None:
                                release_slot()
                        finally:
                            token._end_call()
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                    close_discarded_response(value)
            # Publishing wakes the coordinator. Its checkpoint finalization
            # must observe an idle request and a released API permit.
            if failure is not None:
                result.set_exception(failure)
            else:
                result.set_result(value)

        Thread(target=lambda: context.run(execute), daemon=True, name="analysis-api-request").start()
    except BaseException:
        try:
            if release_slot is not None:
                release_slot()
        finally:
            token._end_call()
        raise

    try:
        while True:
            token.check()
            try:
                value = result.result(timeout=0.05)
            except FutureTimeout:
                if result.done():
                    # A provider TimeoutError differs from this polling timeout.
                    token.check()
                    return result.result()
                continue
            token.check()
            return value
    except AnalysisCancelled:
        # Cancellation can land after the worker's close check but before the
        # caller accepts the published response. A callback covers both already
        # published and still pending values, with exactly one cleanup owner.
        result.add_done_callback(discard_published_response)
        raise


_TASK_TOKENS = {}
_TASK_TOKENS_LOCK = Lock()


def bind_task_cancellation(owner_id, task_id, token):
    """Bind a verified owner's cloud task to its current local worker token."""
    key = (str(owner_id), str(task_id))
    if token is None:
        return lambda: None
    with _TASK_TOKENS_LOCK:
        _TASK_TOKENS[key] = token
    def unbind():
        with _TASK_TOKENS_LOCK:
            if _TASK_TOKENS.get(key) is token:
                _TASK_TOKENS.pop(key, None)
    return unbind


def cancel_task(owner_id, task_id):
    with _TASK_TOKENS_LOCK:
        token = _TASK_TOKENS.get((str(owner_id), str(task_id)))
    return token.cancel() if token is not None else False
