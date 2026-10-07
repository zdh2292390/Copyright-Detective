"""Checkpointed document analysis, independent of the Streamlit UI."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from email.utils import parsedate_to_datetime
from numbers import Real
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
) -> dict[str, Any]:
    """Analyze missing chunks, retry temporary failures, and retain checkpoints.

    Isolated blocked/empty responses do not stop the rest of the document.
    Persistent service/authentication/configuration errors pause the run so
    callers can fix the cause and resume without repeating successful calls.
    Rerun/stop signals (BaseException) intentionally escape, leaving a checkpoint.
    Cancellation pauses cooperatively; an in-flight provider call cannot be stopped.
    """
    validate_analysis_state(chunk_pairs, state)
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be at least one.")

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
        clear_retry()
        publish()
        return True

    state["status"] = "running"
    state["error"] = None
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
    clear_retry()
    if state["status"] == "complete":
        state["error"] = None
    publish()
    return state

