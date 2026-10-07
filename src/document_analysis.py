"""Checkpointed document analysis, independent of the Streamlit UI."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import CancelledError, Executor, FIRST_COMPLETED, ThreadPoolExecutor, wait
from email.utils import parsedate_to_datetime
from numbers import Real
from threading import Event
from typing import Any


MAX_RETRY_DELAY = 60.0


def analysis_fingerprint(text: str, settings: Mapping[str, Any]) -> str:
    """Bind saved results to the document contents and generation settings."""
    digest = hashlib.sha256(text.encode("utf-8"))
    digest.update(json.dumps(dict(settings), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def new_analysis_state(
    fingerprint: str, settings: Mapping[str, Any], total_chunks: int
) -> dict[str, Any]:
    return {
        "fingerprint": fingerprint,
        "settings": dict(settings),
        "total_chunks": total_chunks,
        "status": "incomplete",
        "results": {},
        "failures": {},
        "attempts": {},
        "current_chunk": None,
        "current_attempt": None,
        "retry_in_seconds": 0.0,
        "retry_attempt": None,
        "error": None,
    }


def analysis_results(state: Mapping[str, Any]) -> list:
    """Return successful comparisons in their original document order."""
    return [result for _, result in sorted(state["results"].items())]


def _systemic_error(message: str) -> bool:
    lowered = message.lower()
    return bool(re.search(r"\b(400|401|403|404)\b", lowered)) or any(
        marker in lowered
        for marker in (
            "api key", "authentication", "permission_denied", "permission denied",
            "invalid_argument", "model not found", "unsupported provider",
        )
    )


def _transient_error(message: str) -> bool:
    if _systemic_error(message):
        return False
    lowered = message.lower()
    return bool(re.search(r"\b(408|429|500|502|503|504)\b", lowered)) or any(
        marker in lowered
        for marker in (
            "rate limit", "resource_exhausted", "temporarily unavailable",
            "timeout", "timed out", "deadline_exceeded", "connection error",
            "connectionerror", "connectionreseterror", "connectionabortederror",
            "brokenpipeerror", "connection reset", "connection aborted", "overloaded",
            "concurrency limit", "too many concurrent", "unavailable", "internal server error",
        )
    )


def _chunk_specific_error(message: str) -> bool:
    lowered = message.lower()
    if _systemic_error(message):
        return False
    return any(
        marker in lowered
        for marker in (
            "returned empty content", "model returned no text", "model refused",
            "response blocked", "finish_reason=", "block_reason=",
            "empty vocabulary",
        )
    )


def _metrics_error(metrics: Any) -> str | None:
    if not isinstance(metrics, Mapping) or not metrics:
        return "Error: Chunk comparison did not produce similarity metrics."
    for name, value in metrics.items():
        if (
            not isinstance(name, str) or not name
            or isinstance(value, bool) or not isinstance(value, Real)
        ):
            return "Error: Chunk comparison produced invalid similarity metrics."
        try:
            finite = math.isfinite(value)
        except (OverflowError, TypeError, ValueError):
            finite = False
        if not finite:
            return "Error: Chunk comparison produced non-finite similarity metrics."
    return None


def _comparison_error(result: Any) -> str | None:
    # Normal Continuation returns (error_text, None), whereas some strategies
    # return a bare error string. Neither is a successful comparison.
    if isinstance(result, str):
        return result
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        return "Error: Invalid chunk analysis result."
    generated, metrics = result
    if (
        isinstance(generated, str) and generated.startswith("Error")
        and (not isinstance(metrics, Mapping) or not metrics)
    ):
        return generated
    if not isinstance(generated, str) or not generated.strip():
        return "Error: Model returned empty content."
    return _metrics_error(metrics)


def validate_analysis_state(
    chunk_pairs: Sequence[tuple[str, str]], state: Mapping[str, Any]
) -> None:
    """Reject mismatched or corrupt checkpoints before trusting saved coverage."""
    total = state.get("total_chunks")
    if type(total) is not int or total != len(chunk_pairs):
        raise ValueError("Saved analysis does not match the document chunk count.")
    for field in ("results", "failures", "attempts"):
        if not isinstance(state.get(field), dict):
            raise ValueError(f"Saved analysis has an invalid {field} checkpoint.")
        for index in state[field]:
            if type(index) is not int or not 0 <= index < total:
                raise ValueError(f"Saved analysis has an invalid {field} chunk index.")
    for index, result in state["results"].items():
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise ValueError(f"Saved analysis has an invalid result for chunk {index + 1}.")
        upper, lower, generated, metrics = result
        if (upper, lower) != tuple(chunk_pairs[index]):
            raise ValueError(f"Saved analysis does not match the document at chunk {index + 1}.")
        error = _comparison_error((generated, metrics))
        if error is not None:
            raise ValueError(f"Saved analysis has an invalid result for chunk {index + 1}: {error}")
    for index, message in state["failures"].items():
        if not isinstance(message, str) or not message.strip() or index in state["results"]:
            raise ValueError(f"Saved analysis has an invalid failure for chunk {index + 1}.")
    for index, count in state["attempts"].items():
        if type(count) is not int or count < 0:
            raise ValueError(f"Saved analysis has an invalid attempt count for chunk {index + 1}.")


def _provider_retry_delay(message: str) -> float | None:
    """Read common Retry-After / Gemini retryDelay hints from provider errors."""
    delays = []
    patterns = (
        r"(?:retry-after|retry_after|retrydelay|retry_delay)\s*[\"']?\s*[:=]\s*[\"']?\s*(\d+(?:\.\d+)?)\s*s?\b",
        r"retry(?:\s+again)?\s+in\s+(\d+(?:\.\d+)?)\s*(?:s|seconds?)\b",
        r"retry[_ ]?delay\s*[\"']?\s*[:=]?\s*\{\s*[\"']?seconds[\"']?\s*[:=]\s*[\"']?\s*(\d+(?:\.\d+)?)",
    )
    for pattern in patterns:
        delays.extend(float(match) for match in re.findall(pattern, message, flags=re.IGNORECASE))
    date_match = re.search(
        r"retry-after\s*[\"']?\s*[:=]\s*[\"']?\s*([A-Za-z]{3},[^\r\n]*?GMT)",
        message,
        flags=re.IGNORECASE,
    )
    if date_match is not None:
        try:
            delays.append(max(0.0, parsedate_to_datetime(date_match.group(1)).timestamp() - time.time()))
        except (TypeError, ValueError, OverflowError):
            pass
    return min(MAX_RETRY_DELAY, max(delays)) if delays else None


def _retry_delay(message: str, attempt: int, rng: Callable[[], float]) -> float:
    base = min(MAX_RETRY_DELAY, 2 ** min(attempt, 6))
    jitter = base * 0.25 * max(0.0, min(1.0, rng()))
    provider_delay = _provider_retry_delay(message) or 0.0
    return min(MAX_RETRY_DELAY, max(base + jitter, provider_delay))



def _rate_limited_error(message: str) -> bool:
    lowered = message.lower()
    return bool(re.search(r"\b429\b", lowered)) or any(
        marker in lowered
        for marker in ("rate limit", "resource_exhausted", "too many concurrent")
    )


def _run_parallel_chunk_analysis(
    chunk_pairs, analyze_chunk, state, *, on_update, max_attempts,
    max_concurrency, executor, sleep, rng, should_stop,
):
    """Coordinate bounded calls; only this thread mutates and publishes state."""
    missing = deque(index for index in range(len(chunk_pairs)) if index not in state["results"])
    pending = {}
    retries = {}
    halted = Event()
    skipped = object()
    fatal_error = None
    cooldown_until = 0.0
    effective_concurrency = max_concurrency
    owned_executor = executor is None

    def clear_retry():
        state["retry_in_seconds"] = 0.0
        state["retry_attempt"] = None

    def publish():
        nonlocal fatal_error
        if fatal_error is not None:
            return
        try:
            if on_update is not None:
                on_update(state)
        except BaseException as exc:
            fatal_error = exc
            halted.set()

    def requested_stop():
        nonlocal fatal_error
        if halted.is_set():
            return True
        try:
            requested = should_stop is not None and should_stop()
        except BaseException as exc:
            fatal_error = exc
            halted.set()
            return True
        if requested:
            halted.set()
            state["status"] = "incomplete"
            state["error"] = "Analysis stopped by request."
            clear_retry()
            return True
        return False

    def call_chunk(index):
        # A shared executor may leave our small window queued behind other jobs.
        # Cancelled/paused work must never start an outbound call from that queue.
        if halted.is_set() or (should_stop is not None and should_stop()):
            return skipped
        return analyze_chunk(*chunk_pairs[index])

    def restore_unstarted_attempt(index, previous, existed):
        if existed:
            state["attempts"][index] = previous
        else:
            state["attempts"].pop(index, None)

    def submit(index, attempt):
        nonlocal fatal_error
        if requested_stop():
            return
        previous = state["attempts"].get(index, 0)
        existed = index in state["attempts"]
        state["current_chunk"] = index + 1
        state["current_attempt"] = attempt + 1
        state["attempts"][index] = previous + 1
        state["active_chunks"] = sorted([meta[0] + 1 for meta in pending.values()] + [index + 1])
        clear_retry()
        # Commit the single index before sending it to the provider.
        publish()
        if requested_stop():
            restore_unstarted_attempt(index, previous, existed)
            return
        try:
            future = executor.submit(call_chunk, index)
        except BaseException as exc:
            restore_unstarted_attempt(index, previous, existed)
            fatal_error = exc
            halted.set()
            return
        pending[future] = (index, attempt, previous, existed)

    def collect(future):
        nonlocal cooldown_until, effective_concurrency, fatal_error
        index, attempt, previous, existed = pending.pop(future)
        state["current_chunk"] = index + 1
        state["current_attempt"] = attempt + 1
        state["active_chunks"] = sorted(meta[0] + 1 for meta in pending.values())
        try:
            result = future.result()
        except CancelledError:
            restore_unstarted_attempt(index, previous, existed)
            publish()
            return
        except Exception as exc:
            result = None
            error = f"Error analyzing chunk: {type(exc).__name__}: {exc}"
        except BaseException as exc:
            if fatal_error is None:
                fatal_error = exc
            halted.set()
            return
        else:
            if result is skipped:
                restore_unstarted_attempt(index, previous, existed)
                publish()
                return
            try:
                error = _comparison_error(result)
            except Exception as exc:
                error = f"Error analyzing chunk: {type(exc).__name__}: {exc}"
        if error is None:
            upper, lower = chunk_pairs[index]
            generated, metrics = result
            state["results"][index] = (upper, lower, generated, dict(metrics))
            state["failures"].pop(index, None)
            if not state["failures"] and not halted.is_set():
                state["error"] = None
            publish()
            return

        state["failures"][index] = error
        # Preserve the reason that first paused the job while draining other calls.
        if not halted.is_set():
            state["error"] = f"Chunk {index + 1}: {error}"
        transient = _transient_error(error)
        rate_limited = False
        if transient and not halted.is_set():
            delay = _retry_delay(error, attempt, rng)
            rate_limited = _rate_limited_error(error)
            if rate_limited:
                effective_concurrency = max(1, effective_concurrency // 2)
                state["effective_concurrency"] = effective_concurrency
                cooldown_until = max(cooldown_until, time.monotonic() + delay)
            if attempt + 1 < max_attempts:
                retries[index] = (attempt + 1, time.monotonic() + delay)
                state["retry_in_seconds"] = delay
                state["retry_attempt"] = attempt + 2
            else:
                halted.set()
                state["status"] = "incomplete"
                clear_retry()
        elif not halted.is_set() and not _chunk_specific_error(error):
            halted.set()
            state["status"] = "incomplete"
            clear_retry()
        # Publish the original failed index before changing any queued indices.
        publish()
        if rate_limited and not halted.is_set():
            # A shared pool may not have begun the rest of this window yet.
            # Cancel those calls during the shared cooldown and retain their
            # original attempt: no outbound request consumed that retry budget.
            for queued in list(pending):
                if queued.cancel():
                    queued_index, queued_attempt, _, _ = pending[queued]
                    collect(queued)
                    if halted.is_set():
                        break
                    retries[queued_index] = (queued_attempt, cooldown_until)

    state["status"] = "running"
    state["error"] = None
    state["active_chunks"] = []
    state["concurrency_limit"] = max_concurrency
    state["effective_concurrency"] = effective_concurrency
    clear_retry()
    publish()
    try:
        if executor is None and not halted.is_set():
            executor = ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="document-chunk")
        while missing or retries or pending:
            requested_stop()
            if halted.is_set():
                retries.clear()
                missing.clear()
                # Some calls are queued in the shared pool and have not started.
                for future in list(pending):
                    if future.cancel() or future.cancelled():
                        collect(future)
            else:
                now = time.monotonic()
                while len(pending) < effective_concurrency and now >= cooldown_until:
                    ready = [index for index, (_, deadline) in retries.items() if deadline <= now]
                    if ready:
                        index = min(ready)
                        attempt, _ = retries.pop(index)
                    elif missing and len(pending) + len(retries) < effective_concurrency:
                        index = missing.popleft()
                        attempt = 0
                    else:
                        break
                    submit(index, attempt)
                    if halted.is_set():
                        break
                    now = time.monotonic()

            if pending:
                finished, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in sorted(finished, key=lambda item: pending[item][0]):
                    collect(future)
                continue
            if halted.is_set():
                break
            if retries:
                next_index = min(retries, key=lambda index: retries[index][1])
                attempt, ready_at = retries[next_index]
                remaining = max(0.0, max(ready_at, cooldown_until) - time.monotonic())
                state["retry_in_seconds"] = remaining
                state["retry_attempt"] = attempt + 1
                if remaining:
                    # Event.wait-backed sleep can wake immediately for cancellation.
                    sleep(min(1.0, remaining) if should_stop is not None else remaining)
    except BaseException as exc:
        if fatal_error is None:
            fatal_error = exc
        halted.set()
        # Unforeseen coordinator failures still retain already running successes.
        for future in pending:
            future.cancel()
        for future in list(pending):
            collect(future)
    finally:
        if owned_executor and executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    state["active_chunks"] = []
    clear_retry()
    state["status"] = (
        "complete" if len(state["results"]) == len(chunk_pairs) else "incomplete"
    )
    if state["status"] == "complete":
        state["error"] = None
    publish()
    if fatal_error is not None:
        raise fatal_error
    return state


def run_chunk_analysis(
    chunk_pairs: Sequence[tuple[str, str]],
    analyze_chunk: Callable[[str, str], Any],
    state: dict[str, Any],
    *,
    on_update: Callable[[dict[str, Any]], None] | None = None,
    max_attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
    should_stop: Callable[[], bool] | None = None,
    max_concurrency: int = 1,
    executor: Executor | None = None,
) -> dict[str, Any]:
    """Analyze missing chunks, retry temporary failures, and retain checkpoints.

    Isolated blocked/empty responses do not stop the rest of the document.
    Persistent service/authentication/configuration errors pause the run so
    callers can fix the cause and resume without repeating successful calls.
    Rerun/stop signals (BaseException) intentionally escape, leaving a checkpoint.
    Cancellation pauses cooperatively; an in-flight provider call cannot be stopped.
    Concurrent runs keep a bounded window and publish each completed index from
    the coordinator. An injected executor is also used for a one-call window
    and remains owned by the caller.
    """
    validate_analysis_state(chunk_pairs, state)
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be at least one.")
    if type(max_concurrency) is not int or not 1 <= max_concurrency <= 8:
        raise ValueError("max_concurrency must be an integer between one and eight.")
    if max_concurrency > 1 or executor is not None:
        return _run_parallel_chunk_analysis(
            chunk_pairs, analyze_chunk, state, on_update=on_update,
            max_attempts=max_attempts, max_concurrency=max_concurrency,
            executor=executor, sleep=sleep, rng=rng, should_stop=should_stop,
        )

    def publish() -> None:
        if on_update is not None:
            on_update(state)

    def clear_retry() -> None:
        state["retry_in_seconds"] = 0.0
        state["retry_attempt"] = None

    def stop_requested() -> bool:
        if should_stop is None or not should_stop():
            return False
        state["status"] = "incomplete"
        state["error"] = "Analysis stopped by request."
        state["active_chunks"] = []
        clear_retry()
        publish()
        return True

    state["status"] = "running"
    state["error"] = None
    state["active_chunks"] = []
    state["concurrency_limit"] = 1
    state["effective_concurrency"] = 1
    clear_retry()
    publish()
    for index, (upper, lower) in enumerate(chunk_pairs):
        if stop_requested():
            return state
        if index in state["results"]:
            continue
        state["current_chunk"] = index + 1
        for attempt in range(max_attempts):
            if stop_requested():
                return state
            clear_retry()
            state["current_attempt"] = attempt + 1
            state["attempts"][index] = state["attempts"].get(index, 0) + 1
            state["active_chunks"] = [index + 1]
            # Persist the active chunk and attempt before the outbound call.
            publish()
            if stop_requested():
                return state
            try:
                result = analyze_chunk(upper, lower)
                error = _comparison_error(result)
            except Exception as exc:
                # An empty TimeoutError / ConnectionError still needs retry handling.
                error = f"Error analyzing chunk: {type(exc).__name__}: {exc}"
            state["active_chunks"] = []

            if error is None:
                generated, metrics = result
                state["results"][index] = (upper, lower, generated, dict(metrics))
                state["failures"].pop(index, None)
                if not state["failures"]:
                    state["error"] = None
                publish()
                break

            state["failures"][index] = error
            state["error"] = f"Chunk {index + 1}: {error}"
            publish()
            if stop_requested():
                return state
            transient = _transient_error(error)
            if transient and attempt + 1 < max_attempts:
                delay = _retry_delay(error, attempt, rng)
                state["retry_in_seconds"] = delay
                state["retry_attempt"] = attempt + 2
                publish()
                if should_stop is None:
                    sleep(delay)
                else:
                    remaining = delay
                    while remaining > 0:
                        if stop_requested():
                            return state
                        interval = min(1.0, remaining)
                        sleep(interval)
                        remaining -= interval
                continue
            if not transient and _chunk_specific_error(error):
                break
            state["status"] = "incomplete"
            clear_retry()
            publish()
            return state

    state["status"] = (
        "complete" if len(state["results"]) == len(chunk_pairs) else "incomplete"
    )
    state["active_chunks"] = []
    clear_retry()
    if state["status"] == "complete":
        state["error"] = None
    publish()
    return state

