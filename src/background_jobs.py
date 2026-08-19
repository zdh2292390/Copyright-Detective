"""Process-local background jobs that survive Streamlit script reruns."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Dict, Optional

import streamlit as st

from src.api_concurrency import max_concurrent_api_calls

ProgressReporter = Callable[[int, int, str], None]
JobRunner = Callable[[ProgressReporter], Any]

# Match the global API concurrency cap so Game 1/2 background runners can
# saturate up to COPYRIGHT_DETECTIVE_MAX_CONCURRENT_API in-flight calls.
_EXECUTOR = ThreadPoolExecutor(
    max_workers=max_concurrent_api_calls(),
    thread_name_prefix="copyright-game",
)
_LOCK = Lock()
_JOBS: Dict[str, Dict[str, Any]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def submit_background_job(key: str, label: str, runner: JobRunner) -> bool:
    """Submit once per key; an active job cannot be replaced by a rerun."""
    with _LOCK:
        existing = _JOBS.get(key)
        if existing and existing.get("status") in {"queued", "running"}:
            return False
        _JOBS[key] = {
            "key": key,
            "label": label,
            "status": "queued",
            "current": 0,
            "total": 1,
            "message": "Queued",
            "started_at": _now(),
            "finished_at": None,
            "error": None,
            "result": None,
        }

    def report(current: int, total: int, message: str = "") -> None:
        with _LOCK:
            state = _JOBS.get(key)
            if state is None:
                return
            state["current"] = max(0, int(current))
            state["total"] = max(1, int(total))
            if message:
                state["message"] = str(message)

    def execute() -> None:
        # A worker must not retain a ScriptRunContext after the app reruns. Doing
        # so makes Streamlit UI calls target fragment IDs from the previous run.
        try:
            with _LOCK:
                _JOBS[key]["status"] = "running"
                _JOBS[key]["message"] = "Starting"
            try:
                result = runner(report)
            except Exception as exc:
                with _LOCK:
                    state = _JOBS[key]
                    state["status"] = "failed"
                    state["error"] = str(exc)
                    state["message"] = "Run failed"
                    state["finished_at"] = _now()
            else:
                with _LOCK:
                    state = _JOBS[key]
                    state["status"] = "completed"
                    state["result"] = deepcopy(result)
                    state["current"] = state["total"]
                    state["message"] = "Completed"
                    state["finished_at"] = _now()
        finally:
            # Keep cleanup explicit without introducing Streamlit UI work here.
            pass

    _EXECUTOR.submit(execute)
    return True


def get_background_job(key: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        state = _JOBS.get(key)
        return deepcopy(state) if state is not None else None


def forget_background_job(key: str) -> bool:
    """Remove one finished process-local job without touching active work."""
    with _LOCK:
        state = _JOBS.get(key)
        if state and state.get("status") in {"queued", "running"}:
            return False
        return _JOBS.pop(key, None) is not None


def background_job_running(key: str) -> bool:
    state = get_background_job(key)
    return bool(state and state.get("status") in {"queued", "running"})


@st.fragment(run_every=2)
def _render_active_background_job_status(key: str) -> None:
    """Poll an active job and unregister the fragment once it finishes."""
    state = get_background_job(key)
    if not state:
        return
    status = str(state.get("status") or "")
    current = int(state.get("current") or 0)
    total = max(1, int(state.get("total") or 1))
    label = str(state.get("label") or "Game run")
    message = str(state.get("message") or "")
    if status in {"queued", "running"}:
        st.progress(min(current / total, 1.0), text=f"{label}: {message} ({current}/{total})")
        st.caption("This run continues in the background if the page reruns or you switch views.")
        return

    completion_token = str(state.get("finished_at") or status)
    st.session_state[f"_background_job_delivered:{key}"] = completion_token
    # A full rerun removes this timed fragment from the page. The completed
    # state is then rendered by the non-fragment function below.
    st.rerun()


def render_background_job_status(
    key: str,
    *,
    completed_message: Optional[str] = None,
    completed_message_ttl_seconds: Optional[float] = None,
) -> None:
    """Render a job, polling only while it is actively running."""
    state = get_background_job(key)
    if not state:
        return
    status = str(state.get("status") or "")
    if status in {"queued", "running"}:
        _render_active_background_job_status(key)
        return

    label = str(state.get("label") or "Game run")
    if status == "completed":
        if completed_message_ttl_seconds is not None:
            try:
                finished_at = datetime.fromisoformat(
                    str(state.get("finished_at") or "")
                )
                elapsed_seconds = (
                    datetime.now(timezone.utc) - finished_at
                ).total_seconds()
            except (TypeError, ValueError):
                elapsed_seconds = 0.0
            if elapsed_seconds >= max(0.0, completed_message_ttl_seconds):
                return
        st.success(completed_message or f"{label} completed and was saved.")
    else:
        st.error(f"{label} failed: {state.get('error') or 'Unknown error'}")