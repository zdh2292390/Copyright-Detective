"""Process-local background jobs that survive Streamlit script reruns."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock, Thread
from time import monotonic
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
_TERMINAL_STATUSES = frozenset({"completed", "failed", "stopped"})
# Keep recent or actively viewed results. Durable cloud tasks remain in Supabase;
# only abandoned process-local records are eligible for cleanup.
_TERMINAL_RETENTION_SECONDS = 24 * 60 * 60
_PRUNE_INTERVAL_SECONDS = 60
_NEXT_PRUNE_AT = 0.0


def _prune_finished_jobs_locked() -> None:
    """Remove terminal records idle for a day; callers hold _LOCK."""
    global _NEXT_PRUNE_AT
    now = monotonic()
    if now < _NEXT_PRUNE_AT:
        return
    _NEXT_PRUNE_AT = now + _PRUNE_INTERVAL_SECONDS
    for key, record in list(_JOBS.items()):
        if record.get("status") not in _TERMINAL_STATUSES:
            continue
        finished = record.get("_finished_monotonic")
        # Treat records without our timestamps conservatively as fresh.
        if finished is None:
            continue
        accessed = record.get("_last_accessed_monotonic", finished)
        if now - max(finished, accessed) >= _TERMINAL_RETENTION_SECONDS:
            _JOBS.pop(key, None)


def _clear_finished_handles_locked(record: Dict[str, Any]) -> None:
    """Break Future callback references to journals, inputs and cloud clients."""
    if record.get("status") in _TERMINAL_STATUSES:
        record["_future"] = None
        record["_cancel_token"] = None
        # A successful History deletion owns removal. Keep an active record
        # fenced until its worker has drained all accepted requests, then drop
        # the process-local result rather than publishing it again.
        if record.get("_delete_requested") and _JOBS.get(record.get("key")) is record:
            _JOBS.pop(record["key"], None)


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


def submit_background_job(key: str, label: str, runner: JobRunner, *, metadata: dict | None = None) -> bool:
    """Submit once per key; active and stopping jobs cannot be replaced."""
    if not isinstance(key, str) or not key.strip() or not callable(runner):
        raise ValueError("A job key and callable runner are required.")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("Job metadata must be a dictionary.")
    owner_id, session_id = _origin()
    with _LOCK:
        _prune_finished_jobs_locked()
        existing = _JOBS.get(key)
        if existing and (not _can_access(existing, owner_id, session_id)
                         or existing.get("status") in _ACTIVE_STATUSES):
            return False
    from src.resumable_analysis import prepare_background_journal, journal_scope, report_analysis_progress, _CURRENT, PageRun

    metadata = dict(metadata or {})
    scope = _CURRENT.get()
    if isinstance(scope, PageRun):
        metadata.setdefault("page_key", scope.page_key)
        metadata.setdefault("trigger_key", scope.trigger)
    token = CancellationToken()
    # The journal's heartbeat starts during preparation. Bind the background
    # token now so a cloud stop also cancels queued work before its first call.
    with cancellation_scope(token):
        cached_only = bool(metadata.get("source_type") == "page_job"
                           and isinstance(scope, PageRun) and scope.view_task_id)
        journal = prepare_background_journal(key, label, cached_only=cached_only)
    record = {
        "key": key,
        "label": label,
        "task_id": journal.task_id if journal is not None else None,
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
        "page_key": (metadata or {}).get("page_key"),
        "source_type": (metadata or {}).get("source_type"),
        "trigger_key": (metadata or {}).get("trigger_key"),
        "_cancel_token": token,
        "_future": None,
        "_last_accessed_monotonic": monotonic(),
    }
    current_origin = _origin()
    with _LOCK:
        _prune_finished_jobs_locked()
        existing = _JOBS.get(key)
        rejected = current_origin != (owner_id, session_id) or bool(
            existing and (not _can_access(existing, owner_id, session_id)
                          or existing.get("status") in _ACTIVE_STATUSES)
        )
        if not rejected:
            _JOBS[key] = record
    if rejected:
        # Cloud cleanup can block. Never hold the process registry lock while
        # releasing a journal that lost the race to reserve this key.
        if journal is not None:
            try:
                journal.finish(False)
            except BaseException:
                pass
        return False

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

    def fail(exc: BaseException, saved_result=None) -> None:
        nonlocal journal, runner
        with _LOCK:
            if _JOBS.get(key) is not record:
                return
            cancelled = token.cancelled
            if not cancelled:
                record.update(
                    status="failed",
                    error=str(exc).strip() or type(exc).__name__,
                    message="Partial results saved" if saved_result is not None else "Run failed",
                    finished_at=_now(),
                    result=saved_result,
                    _finished_monotonic=monotonic(),
                )
                _clear_finished_handles_locked(record)
                journal = None
                runner = None
                _prune_finished_jobs_locked()
        if cancelled:
            # Stop/Delete wins even if it arrives during cloud failure cleanup.
            stop_after_cleanup()

    def finish_failed(exc: BaseException, saved_result=None) -> None:
        def cleanup():
            token.wait_for_idle()
            checkpoint_released = finish_incomplete()
            if token.cancelled:
                stopped(checkpoint_released)
            else:
                fail(exc, saved_result)
        if token.pending_count:
            with _LOCK:
                if _JOBS.get(key) is not record:
                    return
                record["status"] = "stopping"
                record["message"] = "Finishing accepted requests after a failed step"
            try:
                Thread(target=cleanup, daemon=True, name="background-failure-cleanup").start()
            except Exception:
                cleanup()
        else:
            cleanup()

    def stopped(checkpoint_released: bool) -> None:
        nonlocal journal, runner
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
                _finished_monotonic=monotonic(),
            )
            _clear_finished_handles_locked(record)
            # A cancelled executor WorkItem can remain queued behind another
            # runner. Drop its closure's journal after checkpoint cleanup too.
            journal = None
            runner = None
            _prune_finished_jobs_locked()

    def stop_after_cleanup(*, defer=False) -> None:
        # Some SDKs cannot abort a request already accepted by the provider.
        # Keep this job locked while those calls hold their semaphore/journal
        # resources. The UI cancel handler itself never waits for cleanup.
        with _LOCK:
            if _JOBS.get(key) is not record:
                return
            record["status"] = "stopping"
            record["message"] = (
                "Deleted from Process Manager; finishing cancellation"
                if record.get("_delete_requested") else
                "Force stop requested; preserving saved progress"
            )

        def cleanup() -> None:
            token.wait_for_idle()
            stopped(finish_incomplete())

        if token.pending_count or defer:
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
        report_analysis_progress(
            current_count, total_count, unit="runs" if key.startswith("copyright_game2:") else "steps",
            stage="background",
        )

    def execute() -> None:
        # Workers remain UI-free after their originating Streamlit run ends.
        saved_result = None
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
                # A runner may return just before its accepted API threads
                # release their slots. Finalize only after those calls drain;
                # the coordinator stays in the background and retains its key.
                token.wait_for_idle()
                token.check()
                complete = bool(getattr(saved_result, "complete", True))
                result_error = getattr(saved_result, "error", None)
                if journal is not None:
                    if record.get("source_type") == "page_job":
                        from src.resumable_analysis import pack_snapshot, snapshot_session
                        final_snapshot = None
                        if not journal.read_only:
                            try:
                                final_snapshot = pack_snapshot(snapshot_session(
                                    saved_result, record.get("page_key"), max_bytes=8 * 1024 * 1024,
                                ))
                            except Exception:
                                # Final aggregates are optional. Keep the local
                                # result and reconstruct from cached calls later.
                                final_snapshot = None
                        complete = complete and not journal.failed and not journal.broken
                        complete = bool(journal.finish(complete, final_snapshot))
                    else:
                        from src.analysis_result_restore import encode_background_result
                        final_result = encode_background_result(saved_result, journal.task)
                        if final_result is not None:
                            journal.metadata["background_result"] = final_result
                        complete = complete and not journal.failed and not journal.broken
                        complete = bool(journal.finish(complete))
                with _LOCK:
                    token.check()
                    if _JOBS.get(key) is not record:
                        return
                    record.update(
                        status="completed" if complete else "failed",
                        result=saved_result,
                        error=None if complete else (result_error or "Some steps failed. Saved progress and partial results are available."),
                        current=record["total"] if complete else record["current"],
                        message="Completed" if complete else "Partial results saved",
                        finished_at=_now(),
                        _finished_monotonic=monotonic(),
                    )
        except AnalysisCancelled:
            stop_after_cleanup()
        except BaseException as exc:
            # A stop requested while an operation fails is still a user stop,
            # rather than an unrelated error dialog for that cancelled run.
            if token.cancelled:
                stop_after_cleanup()
            else:
                # A failed coordinator may still own accepted API requests.
                # Keep its key fenced until those calls and its lease drain.
                finish_failed(exc, saved_result)

    def on_done(future: Any) -> None:
        nonlocal journal, runner
        if future.cancelled():
            token.cancel()
            # Future.cancel invokes callbacks on its caller, often the UI.
            # Release the queued cloud journal on a cleanup thread as well.
            stop_after_cleanup(defer=True)
        with _LOCK:
            if record.get("status") in _TERMINAL_STATUSES:
                journal = None
                runner = None
            # Registering the callback after assigning _future also handles a
            # runner that finishes before executor.submit returns.
            if _JOBS.get(key) is record:
                _clear_finished_handles_locked(record)
                _prune_finished_jobs_locked()

    try:
        future = _EXECUTOR.submit(execute)
        with _LOCK:
            record["_future"] = future
        future.add_done_callback(on_done)
    except Exception as exc:
        finish_failed(exc)
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


def discard_background_task(owner_id: str, task_id: str) -> bool:
    """Forget a successfully deleted cloud task without unlocking active work.

    Call only after the owner-scoped delete RPC succeeds. A key may have been
    reused by a newer task, so both captured owner and task identity must match.
    Running requests keep their key fenced until cancellation cleanup is idle.
    """
    current_owner, _ = _origin()
    owner = str(owner_id or "")
    identity = str(task_id or "")
    if not owner or not identity or str(current_owner or "") != owner:
        return False
    futures = []
    found = False
    with _LOCK:
        for key, record in list(_JOBS.items()):
            if str(record.get("owner_id") or "") != owner or str(record.get("task_id") or "") != identity:
                continue
            found = True
            if record.get("status") not in _ACTIVE_STATUSES:
                _JOBS.pop(key, None)
                continue
            record["_delete_requested"] = True
            token = record.get("_cancel_token")
            if token is not None:
                token.cancel()
            record["status"] = "stopping"
            record["message"] = "Deleted from Process Manager; finishing cancellation"
            future = record.get("_future")
            if future is not None:
                futures.append(future)
    # Cancelling queued work runs its completion callback synchronously. Never
    # hold the registry lock while those callbacks perform cleanup.
    for future in futures:
        future.cancel()
    return found


def _can_access(record, owner_id, session_id) -> bool:
    recorded_owner = record.get("owner_id")
    if recorded_owner is not None:
        return owner_id == recorded_owner
    return owner_id is None and session_id == record.get("session_id")


def get_background_job(key: str, *, include_result: bool = True) -> Optional[Dict[str, Any]]:
    owner_id, session_id = _origin()
    with _LOCK:
        state = _JOBS.get(key)
        if state is None or not _can_access(state, owner_id, session_id):
            return None
        state["_last_accessed_monotonic"] = monotonic()
        _prune_finished_jobs_locked()
        snapshot = {name: value for name, value in state.items()
                    if not name.startswith("_") and (include_result or name != "result")}
        snapshot["has_result"] = isinstance(state.get("result"), dict)
    return deepcopy(snapshot)


def list_background_jobs(*, include_results: bool = False) -> list[dict]:
    """List this owner's jobs, omitting potentially large results by default."""
    owner_id, session_id = _origin()
    with _LOCK:
        _prune_finished_jobs_locked()
        records = [{**{name: value for name, value in record.items()
                       if not name.startswith("_") and (include_results or name != "result")},
                    "has_result": isinstance(record.get("result"), dict)}
                   for record in _JOBS.values() if _can_access(record, owner_id, session_id)]
    return deepcopy(sorted(records, key=lambda item: item.get("started_at") or "", reverse=True))


def forget_background_job(key: str) -> bool:
    """Remove this owner's finished job without touching any active work."""
    owner_id, session_id = _origin()
    with _LOCK:
        state = _JOBS.get(key)
        if not state or not _can_access(state, owner_id, session_id):
            return False
        if state.get("status") in _ACTIVE_STATUSES:
            return False
        return _JOBS.pop(key, None) is not None


def background_job_running(key: str) -> bool:
    state = get_background_job(key, include_result=False)
    return bool(state and state.get("status") in _ACTIVE_STATUSES)


@st.fragment(run_every=2)
def _render_active_background_job_status(key: str) -> None:
    """Poll an active job and unregister the fragment once it finishes."""
    state = get_background_job(key, include_result=False)
    if not state:
        return
    status = str(state.get("status") or "")
    current = int(state.get("current") or 0)
    total = max(1, int(state.get("total") or 1))
    label = str(state.get("label") or "Game run")
    message = str(state.get("message") or "")
    if status in _ACTIVE_STATUSES:
        from src.job_guard import render_running_status_row

        render_running_status_row(
            label, key=f"force_stop_background_job:{key}",
            status={"queued": "Queued", "running": "Running", "stopping": "Stopping"}[status],
            on_stop=lambda: force_stop_background_job(key),
            disabled=status == "stopping",
            help="Stop this run and keep saved progress to resume later.",
        )
        st.progress(min(current / total, 1.0), text=f"{message} ({current}/{total})")
        if status == "stopping":
            st.caption("Stopping. Saved progress is kept; Resume is available after current requests finish.")
        else:
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
    state = get_background_job(key, include_result=False)
    if not state:
        return
    status = str(state.get("status") or "")
    if status in _ACTIVE_STATUSES:
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
    elif status == "stopped":
        st.warning(f"{label}: {state.get('message') or 'Stopped. Saved checkpoints remain available.'}")
    else:
        st.error(f"{label} failed: {state.get('error') or 'Unknown error'}")
