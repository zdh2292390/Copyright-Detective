"""Guard long-running detection jobs from accidental Streamlit reruns."""

from contextlib import contextmanager
from typing import Any, Iterator

import streamlit as st

JOB_RUNNING_KEY = "detection_job_running"
JOB_LABEL_KEY = "detection_job_label"
RUN_ARMED_KEY = "detection_run_armed"
JOB_UNLOCK_PENDING_KEY = "_detection_job_unlock_pending"
SIDEBAR_RENDERING_KEY = "_sidebar_rendering"
JOB_STYLE_SLOT_KEY = "_detection_job_style_slot"


def is_detection_job_running() -> bool:
    return bool(st.session_state.get(JOB_RUNNING_KEY))


def get_ui_disabled() -> bool:
    """Return whether the active-job overlay should block interaction."""
    return is_detection_job_running()

def wd(extra_disabled: bool = False) -> bool:
    """Preserve only page-specific business disabled conditions."""
    return extra_disabled


def install_widget_guards() -> None:
    """Create the per-script CSS slot used to atomically lock controls."""
    st.session_state[JOB_STYLE_SLOT_KEY] = st.empty()

def arm_detection_run(label: str, trigger_key: str) -> None:
    """Schedule a run and lock the next script pass before any widgets render."""
    st.session_state.pop("_fcc_rendered_run", None)
    st.session_state[f"detection_run_{trigger_key}"] = True
    st.session_state[RUN_ARMED_KEY] = trigger_key
    start_detection_job(label)


def should_execute_detection_run(trigger_key: str) -> bool:
    """Return True once for the rerun triggered by a specific Run button."""
    run_key = f"detection_run_{trigger_key}"
    return bool(st.session_state.pop(run_key, False))


def render_run_button(label: str, trigger_key: str, button_label: str, **kwargs: Any) -> bool:
    """Render a Run button that locks the UI and returns True when it should execute."""
    should_run = should_execute_detection_run(trigger_key)
    if should_run:
        start_detection_job(label)

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


def finish_detection_job(*, show_clear_cache: bool = False) -> None:
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
    finish_detection_job(show_clear_cache=False)


@contextmanager
def detection_job(label: str) -> Iterator[None]:
    if not is_detection_job_running():
        start_detection_job(label)
    try:
        yield
    finally:
        finish_detection_job()


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
    finish_detection_job(show_clear_cache=False)


def render_active_job_lock() -> None:
    """Show the running banner and lock styles while a job is active."""
    if not is_detection_job_running():
        st.session_state.pop(JOB_UNLOCK_PENDING_KEY, None)
        _render_ui_unlock_styles()
        return
    job_label = st.session_state.get(JOB_LABEL_KEY, "Detection task")
    _render_running_banner(job_label)
    _render_ui_lock_styles()


def render_background_job_lock(label: str) -> None:
    """Lock interactive UI for a separately managed background job."""
    _render_running_banner(label)
    _render_ui_lock_styles()


def _render_running_banner(label: str) -> None:
    st.markdown(
        (
            '<div class="detection-job-status" '
            'style="padding:0.75rem 1rem;background:#fff3cd;border:1px solid #ffc107;'
            'border-radius:8px;margin-bottom:1rem;">'
            f"⏳ <strong>{label}</strong> is running. "
            "Please do not switch pages or adjust main-page parameters "
            "to avoid interrupting the task."
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_guard_styles(css: str) -> None:
    """Replace the current script pass's guard CSS without leaving stale rules."""
    slot = st.session_state.get(JOB_STYLE_SLOT_KEY)
    target = slot if slot is not None else st
    target.markdown(css, unsafe_allow_html=True)


def _render_ui_lock_styles() -> None:
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
            pointer-events: none !important;
            opacity: 0.68 !important;
            filter: saturate(0.72) !important;
            cursor: not-allowed !important;
        }
        </style>
        """
    )


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