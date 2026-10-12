"""Process-local background jobs that survive Streamlit script reruns."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock, Thread
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

import streamlit as st

from src.api_concurrency import max_concurrent_api_calls
from src.run_cancellation import AnalysisCancelled, CancellationToken, cancellation_scope

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
_ACTIVE_STATUSES = frozenset({"queued", "running", "stopping"})
_SESSION_KEY = "_background_jobs_session_id"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _origin() -> tuple[Optional[str], str]:
    """Read identity only on the UI thread; workers keep the captured binding."""
    owner_id = str(st.session_state.get("user_id") or "") or None
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    context = get_script_run_ctx(suppress_warning=True)
    if context is not None:
        return owner_id, str(context.session_id)
    # Bare-mode tools/tests have no ScriptRunContext. The browser's session
    # state is still the fallback boundary for anonymous callers.
    if not st.session_state.get(_SESSION_KEY):
        st.session_state[_SESSION_KEY] = str(uuid4())
    return owner_id, str(st.session_state[_SESSION_KEY])


def submit_background_job(key: str, label: str, runner: JobRunner) -> bool:
    """Submit once per key; active and stopping jobs cannot be replaced."""
    with _LOCK:
        existing = _JOBS.get(key)
        if existing and existing.get("status") in _ACTIVE_STATUSES:
            return False
    from src.resumable_analysis import prepare_background_journal, journal_scope

    token = CancellationToken()
    # The journal's heartbeat starts during preparation. Bind the background
    # token now so a cloud stop also cancels queued work before its first call.
    with cancellation_scope(token):
        journal = prepare_background_journal(key, label)
    owner_id, session_id = _origin()
    with _LOCK:
        existing = _JOBS.get(key)
        if existing and existing.get("status") in _ACTIVE_STATUSES:
            if journal is not None:
                try:
                    journal.finish(False)
                except Exception:
                    pass
            return False
        record = {
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
            "owner_id": owner_id,
            "session_id": session_id,
            "_cancel_token": token,
            "_future": None,
        }
        _JOBS[key] = record

    def finish_incomplete() -> bool:
        if journal is None:
            return True
        try:
            journal.finish(False)
            return True
        except BaseException:
            # Cancellation must never skip local cleanup or keep the UI locked
            # because cloud lease release failed. The lease will expire.
            return False

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

    def stopped(checkpoint_released: bool) -> None:
        with _LOCK:
            if _JOBS.get(key) is not record:
                return
            record.update(
                status="stopped",
                error=None,
                message=("Stopped. Saved checkpoints remain available for recovery."
                         if checkpoint_released else
                         "Stopped. Saved steps remain available; the cloud lease will expire shortly."),
                finished_at=_now(),
                result=None,
            )

    def stop_after_cleanup() -> None:
        # Some SDKs cannot abort a request already accepted by the provider.
        # Keep this job locked while those calls hold their semaphore/journal
        # resources. The UI cancel handler itself never waits for cleanup.
        with _LOCK:
            if _JOBS.get(key) is not record:
                return
            record["status"] = "stopping"
            record["message"] = "Force stop requested; preserving saved progress"

        def cleanup() -> None:
            token.wait_for_idle()
            stopped(finish_incomplete())

        if token.pending_count:
            try:
                Thread(target=cleanup, daemon=True, name="background-stop-cleanup").start()
            except Exception:
                # This runs on the coordinator, never the UI cancel handler.
                # Resource exhaustion must retain the lock and lease until the
                # accepted request drains, then still complete local cleanup.
                cleanup()
        else:
            cleanup()

    def report(current: int, total: int, message: str = "") -> None:
        token.check()
        # Convert before taking the lock; invalid progress must be reported as
        # a failed run rather than leaving a permanently running job.
        total_count = max(1, int(total))
        current_count = min(total_count, max(0, int(current)))
        with _LOCK:
            token.check()
            if _JOBS.get(key) is not record or record["status"] not in {"queued", "running"}:
                return
            record["current"] = current_count
            record["total"] = total_count
            if message:
                record["message"] = str(message)

    def execute() -> None:
        # Workers remain UI-free after their originating Streamlit run ends.
        try:
            with cancellation_scope(token):
                token.check()
                with _LOCK:
                    token.check()
                    if _JOBS.get(key) is not record:
                        return
                    record["status"] = "running"
                    record["message"] = "Starting"
                with journal_scope(journal):
                    token.check()
                    result = runner(report)
                    token.check()
                # Snapshotting is part of execution and can itself fail. Publish
                # completed only after the result can be safely delivered.
                saved_result = deepcopy(result)
                token.check()
                if journal is not None:
                    journal.finish(True)
                with _LOCK:
                    token.check()
                    if _JOBS.get(key) is not record:
                        return
                    record.update(
                        status="completed",
                        result=saved_result,
                        current=record["total"],
                        message="Completed",
                        finished_at=_now(),
                    )
        except AnalysisCancelled:
            stop_after_cleanup()
        except BaseException as exc:
            # A stop requested while an operation fails is still a user stop,
            # rather than an unrelated error dialog for that cancelled run.
            if token.cancelled:
                stop_after_cleanup()
            else:
                finish_incomplete()
                # SystemExit must also release the UI's active-job lock.
                fail(exc)

    def on_done(future: Any) -> None:
        if future.cancelled():
            token.cancel()
            stop_after_cleanup()

    try:
        future = _EXECUTOR.submit(execute)
        with _LOCK:
            record["_future"] = future
        future.add_done_callback(on_done)
    except Exception as exc:
        finish_incomplete()
        fail(exc)
    # True means this request was accepted, including an immediately visible
    # submission failure. False is reserved for an already active job.
    return True


def force_stop_background_job(key: str, owner_id: Optional[str] = None) -> bool:
    """Cancel only this owner's job, retaining its lock until worker cleanup.

    A supplied owner ID narrows authorization; it never grants access to a job
    owned by a different login. Guests must use the originating browser session.
    Python/server threads are never terminated or asynchronously injected.
    """
    current_owner, current_session = _origin()
    if owner_id is not None and str(owner_id) != str(current_owner or ""):
        return False
    with _LOCK:
        record = _JOBS.get(key)
        if not record or record.get("status") not in _ACTIVE_STATUSES:
            return False
        recorded_owner = record.get("owner_id")
        if recorded_owner is not None:
            if current_owner != recorded_owner:
                return False
        elif current_owner is not None or current_session != record.get("session_id"):
            return False
        record["_cancel_token"].cancel()
        record["status"] = "stopping"
        record["message"] = "Force stop requested; preserving saved progress"
        future = record.get("_future")
    # Future.cancel executes callbacks synchronously. It must be outside _LOCK.
    # Running calls are stopped cooperatively and keep the key locked until
    # cleanup finishes; queued work can be removed before it starts.
    if future is not None:
        future.cancel()
    return True


def get_background_job(key: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        state = _JOBS.get(key)
        snapshot = {name: value for name, value in state.items() if not name.startswith("_")} if state is not None else None
    return deepcopy(snapshot) if snapshot is not None else None


def forget_background_job(key: str) -> bool:
    """Remove one finished process-local job without touching active work."""
    with _LOCK:
        state = _JOBS.get(key)
        if state and state.get("status") in _ACTIVE_STATUSES:
            return False
        return _JOBS.pop(key, None) is not None


def background_job_running(key: str) -> bool:
    with _LOCK:
        state = _JOBS.get(key)
        return bool(state and state.get("status") in _ACTIVE_STATUSES)


@st.fragment(run_every=2)
def _render_active_background_job_status(key: str, display_label: Optional[str] = None) -> None:
    """Poll an active job and unregister the fragment once it finishes."""
    state = get_background_job(key)
    if not state:
        return
    status = str(state.get("status") or "")
    current = int(state.get("current") or 0)
    total = max(1, int(state.get("total") or 1))
    label = str(display_label or state.get("label") or "Game run")
    message = str(state.get("message") or "")
    if status in _ACTIVE_STATUSES:
        from src.job_guard import render_force_stop_button

        progress_column, stop_column = st.columns([5, 1], vertical_alignment="center")
        with progress_column:
            st.progress(min(current / total, 1.0), text=f"{label}: {current}/{total}")
        with stop_column:
            render_force_stop_button(
                key=f"force_stop_background_job:{key}",
                label="Stopping…" if status == "stopping" else "Stop",
                on_click=lambda: force_stop_background_job(key),
                disabled=status == "stopping",
                help="Stop this run and keep saved progress to resume later.",
                width="stretch",
            )
        st.caption(
            "Stopping. Saved progress is kept."
            if status == "stopping" else
            message or "Runs in the background. Saved progress is kept."
        )
        return

    completion_token = str(state.get("finished_at") or status)
    st.session_state[f"_background_job_delivered:{key}"] = completion_token
    # A full rerun removes this timed fragment from the page. The completed
    # state is then rendered by the non-fragment function below.
    st.rerun()


def render_background_job_status(
    key: str,
    *,
    display_label: Optional[str] = None,
    completed_message: Optional[str] = None,
    completed_message_ttl_seconds: Optional[float] = None,
) -> None:
    """Render progress with an optional short label; saved task names stay intact."""
    state = get_background_job(key)
    if not state:
        return
    status = str(state.get("status") or "")
    if status in _ACTIVE_STATUSES:
        _render_active_background_job_status(key, display_label)
        return

    label = str(display_label or state.get("label") or "Game run")
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
    elif status == "stopped":
        st.warning(f"{label}: {state.get('message') or 'Stopped. Saved checkpoints remain available.'}")
    else:
        st.error(f"{label} failed: {state.get('error') or 'Unknown error'}")
