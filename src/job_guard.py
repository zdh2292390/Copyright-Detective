"""Guard long-running detection jobs from accidental Streamlit reruns."""

from contextlib import contextmanager
from html import escape
from typing import Any, Iterator
from contextvars import ContextVar
from threading import Lock, Thread
from uuid import uuid4

from src.run_cancellation import CancellationToken, cancellation_scope, check_cancellation

import streamlit as st

JOB_RUNNING_KEY = "detection_job_running"
JOB_LABEL_KEY = "detection_job_label"
RUN_ARMED_KEY = "detection_run_armed"
JOB_UNLOCK_PENDING_KEY = "_detection_job_unlock_pending"
SIDEBAR_RENDERING_KEY = "_sidebar_rendering"
JOB_STYLE_SLOT_KEY = "_detection_job_style_slot"

JOB_RUN_ID_KEY = "detection_job_run_id"
STOP_NOTICE_KEY = "_analysis_force_stop_notice"
_RUNS = {}
_RUNS_LOCK = Lock()
_RUN_ID = ContextVar("detection_ui_run_id", default=None)


def get_detection_job_run_id():
    return st.session_state.get(JOB_RUN_ID_KEY)


def _run_token(run_id):
    with _RUNS_LOCK:
        record = _RUNS.get(run_id)
        if record and record["owner_id"] == str(st.session_state.get("user_id") or ""):
            return record["token"]
    return None


def _forget_run_when_idle(run_id):
    with _RUNS_LOCK:
        record = _RUNS.get(run_id)
        if not record or record.get("cleanup"):
            return
        record["cleanup"] = True
    def cleanup():
        record["token"].wait_for_idle()
        with _RUNS_LOCK:
            if _RUNS.get(run_id) is record:
                _RUNS.pop(run_id, None)
    if record["token"].pending_count:
        try:
            Thread(target=cleanup, daemon=True, name="analysis-stop-cleanup").start()
        except Exception:
            # Keep the stop callback responsive under thread exhaustion. The
            # interrupted coordinator's finally block will retry cleanup.
            with _RUNS_LOCK:
                record["cleanup"] = False
    else:
        cleanup()


def force_stop_detection_job():
    """Signal the current UI run and release its overlay without erasing data."""
    run_id = get_detection_job_run_id()
    token = _run_token(run_id)
    if token is None:
        return False
    token.cancel()
    for key in list(st.session_state):
        if str(key).startswith("detection_run_"):
            st.session_state.pop(key, None)
    from src.resumable_analysis import page_supports_analysis_recovery
    st.session_state[STOP_NOTICE_KEY] = (
        True if page_supports_analysis_recovery(st.session_state.get("main_navigation"))
        else "Stopped. Run this analysis again to restart."
    )
    finish_detection_job(run_id=run_id)
    return True


def render_force_stop_button(*, key, label="Stop", on_click=None, disabled=False, help=None, width="content"):
    """The explicit stop action remains clickable under the active-run overlay."""
    with st.container(key=f"force_stop_controls_{key}"):
        options = {"key": key, "disabled": disabled, "type": "secondary", "help": help, "width": width}
        if on_click is not None:
            options["on_click"] = on_click
        return st.button(label, **options)


def is_detection_job_running() -> bool:
    return bool(st.session_state.get(JOB_RUNNING_KEY))


def get_ui_disabled() -> bool:
    """Submitted work never disables page navigation or configuration."""
    return False

def wd(extra_disabled: bool = False) -> bool:
    """Preserve only page-specific business disabled conditions."""
    return extra_disabled


def install_widget_guards() -> None:
    """Create a per-script slot for clearing legacy interaction guards."""
    st.session_state[JOB_STYLE_SLOT_KEY] = st.empty()

def arm_detection_run(label: str, trigger_key: str) -> None:
    """Schedule one run without restricting any page controls."""
    from src.analysis_recovery_ui import discard_pending_analysis_restore
    discard_pending_analysis_restore()
    st.session_state.pop("_fcc_rendered_run", None)
    st.session_state[f"detection_run_{trigger_key}"] = True
    st.session_state[RUN_ARMED_KEY] = trigger_key



def should_execute_detection_run(trigger_key: str) -> bool:
    """Return True once for the rerun triggered by a specific Run button."""
    run_key = f"detection_run_{trigger_key}"
    should_run = bool(st.session_state.pop(run_key, False))
    if st.session_state.get(RUN_ARMED_KEY) == trigger_key:
        st.session_state.pop(RUN_ARMED_KEY, None)
    return should_run


def discard_unconsumed_detection_run() -> None:
    """Do not let a skipped Run click start work on an unrelated later rerun."""
    trigger = st.session_state.pop(RUN_ARMED_KEY, None)
    if isinstance(trigger, str) and trigger:
        st.session_state.pop(f"detection_run_{trigger}", None)


def render_run_button(label: str, trigger_key: str, button_label: str, **kwargs: Any) -> bool:
    """Render an independent submission action and consume its trigger once."""
    should_run = should_execute_detection_run(trigger_key)
    if should_run:
        from src.resumable_analysis import register_ui_run
        register_ui_run(label, trigger_key)

    extra_disabled = bool(kwargs.pop("disabled", False))
    kwargs.pop("type", None)
    kwargs.pop("use_container_width", None)
    kwargs["type"] = "primary"
    kwargs["width"] = "stretch"
    st.button(
        button_label.strip(),
        key=trigger_key,
        on_click=lambda: arm_detection_run(label, trigger_key),
        disabled=extra_disabled,
        **kwargs,
    )
    return should_run


def start_detection_job(label: str) -> None:
    if is_detection_job_running():
        return
    st.session_state.pop(JOB_UNLOCK_PENDING_KEY, None)
    run_id = uuid4().hex
    with _RUNS_LOCK:
        _RUNS[run_id] = {"token": CancellationToken(), "owner_id": str(st.session_state.get("user_id") or "")}
    st.session_state[JOB_RUN_ID_KEY] = run_id
    st.session_state[JOB_RUNNING_KEY] = True
    st.session_state[JOB_LABEL_KEY] = label


@contextmanager
def sidebar_rendering() -> Iterator[None]:
    """Mark sidebar widget rendering so job locks do not affect sidebar controls."""
    st.session_state[SIDEBAR_RENDERING_KEY] = True
    try:
        yield
    finally:
        st.session_state.pop(SIDEBAR_RENDERING_KEY, None)


def finish_detection_job(*, show_clear_cache: bool = False, run_id=None) -> None:
    bound_id = run_id or _RUN_ID.get() or get_detection_job_run_id()
    if bound_id:
        _forget_run_when_idle(bound_id)
        if bound_id != get_detection_job_run_id():
            return  # Cleanup from an interrupted script cannot unlock a newer run.
    st.session_state.pop(JOB_RUN_ID_KEY, None)
    was_running = is_detection_job_running()
    st.session_state[JOB_RUNNING_KEY] = False
    st.session_state.pop(JOB_LABEL_KEY, None)
    st.session_state.pop(RUN_ARMED_KEY, None)
    if was_running:
        # Streamlit may discard deltas emitted immediately before st.rerun().
        # Re-emit the unlock CSS once on the following complete script pass.
        st.session_state[JOB_UNLOCK_PENDING_KEY] = True
        _render_ui_unlock_styles()
    if not was_running:
        return
    if show_clear_cache and not st.session_state.get("_fcc_rendered_run"):
        from src.floating_clear_cache import queue_rerun_hint_after_run

        queue_rerun_hint_after_run(mount_key="job_done")
    st.session_state.pop("_fcc_rendered_run", None)


def reset_detection_job() -> None:
    """Clear any in-progress job lock (e.g. when the user clicks Clear Cache)."""
    token = _run_token(get_detection_job_run_id())
    if token is not None:
        token.cancel()
    finish_detection_job(show_clear_cache=False)


@contextmanager
def detection_job(label: str) -> Iterator[None]:
    if not is_detection_job_running():
        start_detection_job(label)
    run_id = get_detection_job_run_id()
    token = _run_token(run_id)
    binding = _RUN_ID.set(run_id)
    from src.resumable_analysis import finish_ui_run
    try:
        with cancellation_scope(token):
            try:
                check_cancellation()
                yield
                check_cancellation()
            except BaseException:
                finish_ui_run(False)
                raise
            else:
                finish_ui_run(True)
    finally:
        finish_detection_job(run_id=run_id)
        _RUN_ID.reset(binding)


def ensure_detection_job_finished() -> None:
    """Clear any lingering job lock once the current script run completes."""
    if is_detection_job_running():
        finish_detection_job(show_clear_cache=False)


def handle_interrupted_job() -> None:
    """Recover UI after a rerun killed an in-progress job."""
    if not is_detection_job_running():
        return
    # A Run-button callback intentionally starts the lock before the next script
    # pass. That first pass is not an interruption and must keep the lock active.
    if st.session_state.pop(RUN_ARMED_KEY, None):
        return
    job_label = st.session_state.get(JOB_LABEL_KEY, "Detection task")
    st.warning(
        f'⚠️ The previous run of "{job_label}" was interrupted '
        "(usually caused by switching pages, changing the sidebar, or adjusting parameters). "
        "Partial results below may still be available; review them before rerunning."
    )
    token = _run_token(get_detection_job_run_id())
    if token is not None:
        token.cancel()
    finish_detection_job(show_clear_cache=False)


def render_active_job_lock() -> None:
    """Show legacy foreground status while leaving controls available."""
    if not is_detection_job_running():
        st.session_state.pop(JOB_UNLOCK_PENDING_KEY, None)
        _render_ui_unlock_styles()
        notice = st.session_state.pop(STOP_NOTICE_KEY, False)
        if notice:
            st.info(notice if isinstance(notice, str) else
                    "Stopped. Saved progress is preserved. Choose Resume to continue.")
        return
    job_label = st.session_state.get(JOB_LABEL_KEY, "Detection task")
    from src.resumable_analysis import page_supports_analysis_recovery
    stop_help = ("Stop now and keep saved progress. Requests already sent may still finish."
                 if page_supports_analysis_recovery(st.session_state.get("main_navigation"))
                 else "Stop this run. Requests already sent may still finish.")
    render_running_status_row(
        job_label, key="force_stop_detection_job", on_stop=force_stop_detection_job,
        help=stop_help,
    )
    st.caption("You can configure other tasks while this task runs.")
    _render_ui_unlock_styles()


def render_background_job_lock(label: str) -> None:
    """Keep background tasks independent of the current page's controls."""
    _render_ui_unlock_styles()


def render_running_status_row(
    label: str, *, key: str, status: str = "Running", on_stop=None,
    disabled: bool = False, help: str | None = None,
) -> None:
    """Keep the active state and its Stop action in one compact horizontal row."""
    from src.history_ui import HISTORY_STYLES

    st.markdown(HISTORY_STYLES, unsafe_allow_html=True)
    st.markdown(
        """<style>
        [class*="st-key-running_status_"] {
            column-gap: 0.625rem !important;
            row-gap: 0.35rem !important;
            align-items: center !important;
        }
        [class*="st-key-running_status_"] .detection-job-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            min-height: 2.25rem;
            font-size: 0.875rem;
            line-height: 1.35;
        }
        [class*="st-key-running_status_"] .detection-job-label {
            overflow-wrap: anywhere;
            font-weight: 600;
        }
        [class*="st-key-running_status_"] .detection-job-state {
            flex-shrink: 0;
            padding: 0.2rem 0.55rem;
            border: 1px solid rgba(147, 197, 253, 0.6);
            border-radius: 6px;
            background: rgba(219, 234, 254, 0.6);
            color: #1e40af;
            font-size: 0.8125rem;
            font-weight: 600;
        }
        [class*="st-key-running_status_"] .detection-job-state[data-state="Stopping"],
        [class*="st-key-running_status_"] .detection-job-state[data-state="Queued"] {
            border-color: rgba(251, 191, 36, 0.5);
            background: rgba(254, 243, 199, 0.6);
            color: #92400e;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    with st.container(
        key=f"running_status_{key}", horizontal=True,
        vertical_alignment="center", gap="small",
    ):
        st.html(
            '<div class="detection-job-status" role="status" aria-live="polite">'
            f'<span class="detection-job-label">{escape(str(label))}</span>'
            f'<span class="detection-job-state" data-state="{escape(str(status), quote=True)}">'
            f'{escape(str(status))}</span></div>',
            width="content",
        )
        render_force_stop_button(
            key=key, on_click=on_stop, disabled=disabled,
            help=help, width="content",
        )




def _render_guard_styles(css: str) -> None:
    """Replace the current script pass's guard CSS without leaving stale rules."""
    slot = st.session_state.get(JOB_STYLE_SLOT_KEY)
    target = slot if slot is not None else st
    target.markdown(css, unsafe_allow_html=True)


def _render_ui_lock_styles() -> None:
    # Kept for older callers; submitted tasks never lock page controls.
    _render_ui_unlock_styles()


def _render_ui_unlock_styles() -> None:
    _render_guard_styles(
        """
        <style>
        :is(
            [data-testid="stMain"],
            .stMain,
            section.main,
            section[data-testid="stSidebar"]
        ) :is(
            button,
            a,
            summary,
            iframe,
            [role="button"],
            [role="tab"],
            [role="combobox"],
            [role="slider"],
            [contenteditable="true"],
            [data-testid="stSlider"],
            [data-testid="stNumberInput"],
            [data-testid="stTextArea"],
            [data-testid="stTextInput"],
            [data-testid="stSelectbox"],
            [data-testid="stRadio"],
            [data-testid="stCheckbox"],
            [data-testid="stToggle"],
            [data-testid="stFileUploader"],
            [data-testid="stMultiSelect"],
            [data-testid="stDateInput"],
            [data-testid="stTimeInput"],
            [data-testid="stColorPicker"],
            [data-testid="stDataEditor"],
            [data-testid="stDataFrame"]
        ) {
            pointer-events: revert !important;
            opacity: revert !important;
            filter: revert !important;
            cursor: revert !important;
        }
        </style>
        """
    )