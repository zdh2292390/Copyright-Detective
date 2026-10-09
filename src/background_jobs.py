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
    from src.resumable_analysis import prepare_background_journal, journal_scope
    journal = prepare_background_journal(key, label)
    with _LOCK:
        existing = _JOBS.get(key)
        if existing and existing.get("status") in {"queued", "running"}:
            if journal is not None:
                try:
                    journal.finish(False)
                except Exception:
                    pass
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

    # Capture this exact record so callbacks from an older run cannot overwrite
    # a newer run using the same key.
    with _LOCK:
        record = _JOBS[key]

    def fail(exc: BaseException) -> None:
        with _LOCK:
            if _JOBS.get(key) is not record:
                return
            record.update(
                status="failed",
                error=str(exc).strip() or type(exc).__name__,
                message="Run failed",
                finished_at=_now(),
                result=None,
            )

    def report(current: int, total: int, message: str = "") -> None:
        # Convert before taking the lock; invalid progress must be reported as
        # a failed run rather than leaving a permanently running job.
        total_count = max(1, int(total))
        current_count = min(total_count, max(0, int(current)))
        with _LOCK:
            if _JOBS.get(key) is not record or record["status"] not in {"queued", "running"}:
                return
            record["current"] = current_count
            record["total"] = total_count
            if message:
                record["message"] = str(message)

    def execute() -> None:
        # Workers remain UI-free after their originating Streamlit run ends.
        try:
            with _LOCK:
                if _JOBS.get(key) is not record:
                    return
                record["status"] = "running"
                record["message"] = "Starting"
            with journal_scope(journal):
                result = runner(report)
            # Snapshotting is part of execution and can itself fail. Publish
            # completed only after the result can be safely delivered.
            saved_result = deepcopy(result)
            if journal is not None:
                journal.finish(True)
            with _LOCK:
                if _JOBS.get(key) is not record:
                    return
                record.update(
                    status="completed",
                    result=saved_result,
                    current=record["total"],
                    message="Completed",
                    finished_at=_now(),
                )
        except BaseException as exc:
            if journal is not None:
                try:
                    journal.finish(False)
                except Exception:
                    pass
            # SystemExit in a worker must also release the UI's active-job lock.
            fail(exc)

    def on_done(future: Any) -> None:
        if future.cancelled():
            if journal is not None:
                try:
                    journal.finish(False)
                except Exception:
                    pass
            fail(RuntimeError("The background task was cancelled before it could finish."))

    try:
        future = _EXECUTOR.submit(execute)
        future.add_done_callback(on_done)
    except Exception as exc:
        if journal is not None:
            try:
                journal.finish(False)
            except Exception:
                pass
        fail(exc)
    # True means this request was accepted, including an immediately visible
    # submission failure. False is reserved for an already active job.
    return True


def get_background_job(key: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        state = _JOBS.get(key)
        snapshot = dict(state) if state is not None else None
    return deepcopy(snapshot) if snapshot is not None else None


def forget_background_job(key: str) -> bool:
    """Remove one finished process-local job without touching active work."""
    with _LOCK:
        state = _JOBS.get(key)
        if state and state.get("status") in {"queued", "running"}:
            return False
        return _JOBS.pop(key, None) is not None


def background_job_running(key: str) -> bool:
    with _LOCK:
        state = _JOBS.get(key)
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