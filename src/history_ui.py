"""Shared presentation for API and document Process Managers."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape

import streamlit as st
from uuid import uuid4


PROCESS_MANAGER_TITLE = "Process Manager"

HISTORY_STYLES = """<style>
/* Only history controls are styled; page Run buttons keep their existing UI. */
:is(.st-key-analysis_history_list, .st-key-document_history_list) [data-testid="stExpander"],
:is(.st-key-analysis_history_list, .st-key-document_history_list) [data-testid="stExpander"] > details {
    border-radius: 5px !important;
    box-shadow: none !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [data-testid="stExpander"] summary {
    min-height: 2.125rem;
    padding: 0.4rem 0.7rem !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [data-testid="stExpanderDetails"] {
    padding: 0.625rem 0.75rem !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [data-testid="stExpanderDetails"] > [data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_task_"] {
    padding: 0.625rem 0.75rem !important;
    gap: 0.35rem !important;
    border-radius: 5px !important;
}
/* One geometry and typography for Resume, View, Kill, Refresh and Stop. */
:is(.st-key-analysis_history_list, .st-key-document_history_list, .st-key-document_analysis_controls, [class*="st-key-running_status_"], .st-key-history_confirm_body) [data-testid="stButton"] button {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 6.25rem !important;
    min-width: 6.25rem !important;
    height: 2.25rem !important;
    min-height: 2.25rem !important;
    padding: 0.35rem 0.75rem !important;
    border: 2px solid rgba(147, 197, 253, 0.6) !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 255, 0.9)) !important;
    color: #1e40af !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    font-size: 0.8125rem !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
    letter-spacing: 0.3px !important;
    text-align: center !important;
    white-space: nowrap !important;
    transform: none !important;
    box-shadow: 0 4px 14px rgba(147, 197, 253, 0.2), 0 2px 6px rgba(15, 23, 42, 0.06) !important;
    transition: background 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list, .st-key-document_analysis_controls, [class*="st-key-running_status_"], .st-key-history_confirm_body) [data-testid="stButton"] button p {
    color: inherit !important;
    font-family: inherit !important;
    font-size: 0.8125rem !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
    letter-spacing: inherit !important;
    margin: 0 !important;
    text-align: center !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list, .st-key-document_analysis_controls, [class*="st-key-running_status_"], .st-key-history_confirm_body) [data-testid="stButton"] button:not(:disabled):hover {
    background: linear-gradient(135deg, rgba(147, 197, 253, 0.15), rgba(165, 180, 252, 0.12)) !important;
    border-color: #93c5fd !important;
    box-shadow: 0 6px 18px rgba(147, 197, 253, 0.3), 0 3px 8px rgba(15, 23, 42, 0.08) !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list, .st-key-document_analysis_controls, [class*="st-key-running_status_"], .st-key-history_confirm_body) [data-testid="stButton"] button:not(:disabled):active {
    box-shadow: 0 2px 8px rgba(147, 197, 253, 0.25) !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list, .st-key-document_analysis_controls, [class*="st-key-running_status_"], .st-key-history_confirm_body) [data-testid="stButton"] button:focus-visible {
    outline: 2px solid #93c5fd;
    outline-offset: 2px;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list, .st-key-document_analysis_controls, [class*="st-key-running_status_"], .st-key-history_confirm_body) [data-testid="stButton"] button:disabled {
    opacity: 0.5;
    transform: none !important;
    box-shadow: none !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list, .st-key-document_analysis_controls, [class*="st-key-running_status_"], .st-key-history_confirm_body) :is([class*="st-key-force_stop_controls_"][class*="kill"], .st-key-history_confirm_actions [data-testid="stButton"]:has(button[kind="primary"])) button {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(255, 245, 245, 0.9)) !important;
    color: #b91c1c !important;
    border-color: rgba(252, 165, 165, 0.6) !important;
    box-shadow: 0 4px 14px rgba(252, 165, 165, 0.2), 0 2px 6px rgba(15, 23, 42, 0.06) !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list, .st-key-document_analysis_controls, [class*="st-key-running_status_"], .st-key-history_confirm_body) :is([class*="st-key-force_stop_controls_"][class*="kill"], .st-key-history_confirm_actions [data-testid="stButton"]:has(button[kind="primary"])) button::before {
    background: linear-gradient(135deg, rgba(252, 165, 165, 0.1), transparent 50%);
}
:is(.st-key-analysis_history_list, .st-key-document_history_list, .st-key-document_analysis_controls, [class*="st-key-running_status_"], .st-key-history_confirm_body) :is([class*="st-key-force_stop_controls_"][class*="kill"], .st-key-history_confirm_actions [data-testid="stButton"]:has(button[kind="primary"])) button:not(:disabled):hover {
    background: linear-gradient(135deg, rgba(252, 165, 165, 0.15), rgba(254, 202, 202, 0.12)) !important;
    border-color: #fca5a5 !important;
    box-shadow: 0 6px 18px rgba(252, 165, 165, 0.3), 0 3px 8px rgba(15, 23, 42, 0.08) !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list, .st-key-document_analysis_controls, [class*="st-key-running_status_"], .st-key-history_confirm_body) :is([class*="st-key-force_stop_controls_"][class*="kill"], .st-key-history_confirm_actions [data-testid="stButton"]:has(button[kind="primary"])) button:focus-visible { outline-color: #fca5a5; }
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_actions_"] {
    gap: 0.5rem !important;
    justify-content: flex-end;
}
.st-key-history_confirm_actions {
    gap: 0.75rem !important;
    justify-content: flex-end !important;
    margin-top: 0.5rem;
}
[data-testid="stDialog"]:has(.st-key-history_confirm_body) [role="dialog"] {
    border-radius: 18px;
    border: 1px solid rgba(203, 213, 225, 0.65);
    box-shadow: 0 24px 64px rgba(15, 23, 42, 0.18);
}
.history-confirm-summary {
    padding: 0.875rem 1rem;
    border: 1px solid rgba(252, 165, 165, 0.45);
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(255, 245, 245, 0.9), rgba(255, 255, 255, 0.95));
    font-size: 0.9375rem;
    font-weight: 600;
    overflow-wrap: anywhere;
}
.st-key-history_confirm_body [data-testid="stMarkdownContainer"] p {
    font-size: 0.875rem;
    line-height: 1.5;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [data-testid="stCaptionContainer"] p {
    font-size: 0.8125rem !important;
    line-height: 1.35 !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_details_"] [data-testid="stExpander"],
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_details_"] [data-testid="stExpander"] > details {
    border: none !important;
    background: transparent !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_details_"] [data-testid="stExpander"] summary {
    min-height: 1.75rem;
    padding: 0.2rem 0 !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_details_"] [data-testid="stExpanderDetails"] {
    padding: 0.35rem 0 !important;
}
.history-details-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.35rem 1.25rem;
    font-size: 0.8125rem;
    line-height: 1.4;
}
.history-details-field { overflow-wrap: anywhere; }
.history-details-label { opacity: 0.65; margin-right: 0.5rem; }
.history-details-id { grid-column: 1 / -1; }
.history-details-id code { font-size: 0.75rem; padding: 0; background: transparent; }
@media (max-width: 600px) {
    .history-details-grid { grid-template-columns: minmax(0, 1fr); }
    :is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_actions_"] {
        justify-content: flex-start;
    }
}
</style>"""


def format_history_time(value):
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError):
        return None


def render_history_details(task, *, page_key=None, extra_fields=()):
    fields = []
    for key, label in (("created_at", "Created"), ("started_at", "Started"),
                       ("updated_at", "Updated"), ("completed_at", "Finished")):
        value = format_history_time(task.get(key))
        if value:
            fields.append((label, value))
    if page_key or task.get("page_key"):
        fields.append(("Page", task.get("page_key") or page_key))
    fields.extend((label, value) for label, value in extra_fields if value is not None)
    rows = [
        '<div class="history-details-field"><span class="history-details-label">'
        + escape(str(label)) + '</span>' + escape(str(value)) + '</div>'
        for label, value in fields
    ]
    task_id = task.get("id") or task.get("token")
    if task_id:
        rows.append('<div class="history-details-field history-details-id">'
                    '<span class="history-details-label">Task ID</span><code>'
                    + escape(str(task_id)) + '</code></div>')
    st.html('<div class="history-details-grid">' + ''.join(rows) + '</div>')


_KILL_PENDING = "_analysis_kill_pending"


def _dismiss_task_deletion():
    st.session_state.pop(_KILL_PENDING, None)


def _confirmation_owner_matches(pending):
    return str(st.session_state.get("user_id") or "") == str(pending.get("owner_id") or "")


def confirm_task_deletion(*, task_id, owner_id, label, on_confirm):
    """Schedule a native dialog; only its Confirm action may mutate the task."""
    pending = {
        "task_id": str(task_id), "owner_id": str(owner_id or ""),
        "label": str(label or "Analysis"), "nonce": uuid4().hex,
        "on_confirm": on_confirm,
    }
    if not _confirmation_owner_matches(pending):
        st.warning("Your account changed. Reopen this task from Process Manager before deleting it.")
        return False
    if not callable(on_confirm) or not pending["task_id"]:
        raise ValueError("A task and a deletion callback are required.")
    st.session_state[_KILL_PENDING] = pending
    # A polling fragment must return to the app-level dialog renderer. Normal
    # button reruns already reach it without a second full application rerun.
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    context = get_script_run_ctx(suppress_warning=True)
    if context is not None and getattr(context, "current_fragment_id", None):
        st.rerun(scope="app")
    return True


@st.dialog("Delete analysis?", width="small", on_dismiss=_dismiss_task_deletion)
def _render_task_deletion_dialog():
    pending = st.session_state.get(_KILL_PENDING)
    if not isinstance(pending, dict):
        st.rerun()
    if not _confirmation_owner_matches(pending):
        _dismiss_task_deletion()
        st.rerun()
    st.html(HISTORY_STYLES)
    with st.container(key="history_confirm_body", gap="small"):
        st.html('<div class="history-confirm-summary">' + escape(pending["label"]) + '</div>')
        st.markdown("This will stop this analysis and permanently delete its inputs, saved progress and results.")
        st.caption("This action cannot be undone.")
        if pending.get("error"):
            st.error(pending["error"])
        with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center",
                          gap="small", key="history_confirm_actions"):
            if st.button("Cancel", key="history_delete_cancel", width="content"):
                _dismiss_task_deletion()
                st.rerun()
            if st.button("Confirm", key="history_delete_confirm", type="primary", width="content"):
                current = st.session_state.get(_KILL_PENDING)
                if (not isinstance(current, dict) or current.get("nonce") != pending["nonce"]
                        or current.get("task_id") != pending["task_id"]
                        or not _confirmation_owner_matches(current)):
                    _dismiss_task_deletion()
                    st.rerun()
                # Consume before the callback so repeated/stale clicks cannot
                # cancel or delete again. A failed deletion may explicitly retry.
                st.session_state.pop(_KILL_PENDING, None)
                try:
                    success = current["on_confirm"]() is True
                except Exception:
                    success = False
                if success:
                    st.rerun()
                if _confirmation_owner_matches(current) and _KILL_PENDING not in st.session_state:
                    current["error"] = "Could not complete deletion. Retry Confirm or choose Cancel."
                    st.session_state[_KILL_PENDING] = current
                    st.rerun()
                st.rerun()


def render_pending_task_deletion():
    """Render at most one pending dialog after the application's page content."""
    pending = st.session_state.get(_KILL_PENDING)
    if not isinstance(pending, dict):
        return
    if not _confirmation_owner_matches(pending):
        _dismiss_task_deletion()
        return
    _render_task_deletion_dialog()
