"""Compact presentation shared by API and document analysis history."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape

import streamlit as st


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
:is(.st-key-analysis_history_list, .st-key-document_history_list) [data-testid="stButton"] button {
    min-height: 2.125rem !important;
    height: 2.125rem !important;
    padding: 0.3rem 0.65rem !important;
    border-width: 1px !important;
    border-radius: 4px !important;
    box-shadow: none !important;
    transform: none !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [data-testid="stButton"] button p {
    font-size: 0.8125rem !important;
    line-height: 1.2 !important;
}

/* Match the page Clear Cache buttons, at the compact History action size. */
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_actions_"] {
    gap: 0.5rem !important;
    justify-content: flex-end;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_actions_"] [data-testid="stButton"] button {
    width: 6.25rem !important;
    min-width: 6.25rem !important;
    height: 2.25rem !important;
    min-height: 2.25rem !important;
    padding: 0.35rem 0.8rem !important;
    border: 2px solid rgba(147, 197, 253, 0.6) !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 255, 0.9)) !important;
    color: #1e40af !important;
    font-weight: 700;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 14px rgba(147, 197, 253, 0.2), 0 2px 6px rgba(15, 23, 42, 0.06) !important;
    transition: background 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease, transform 0.25s ease;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_actions_"] [data-testid="stButton"] button p {
    color: inherit !important;
    font-weight: 700;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_actions_"] [data-testid="stButton"] button:not(:disabled):hover {
    background: linear-gradient(135deg, rgba(147, 197, 253, 0.15), rgba(165, 180, 252, 0.12)) !important;
    border-color: #93c5fd !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(147, 197, 253, 0.3), 0 4px 12px rgba(15, 23, 42, 0.08) !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_actions_"] [data-testid="stButton"] button:not(:disabled):active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 8px rgba(147, 197, 253, 0.25) !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_actions_"] [data-testid="stButton"] button:focus-visible {
    outline: 2px solid #93c5fd;
    outline-offset: 2px;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_actions_"] [data-testid="stButton"] button:disabled {
    opacity: 0.5;
    transform: none !important;
    box-shadow: none !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_actions_"] [class*="st-key-force_stop_controls_"] [data-testid="stButton"] button {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(255, 245, 245, 0.9)) !important;
    color: #b91c1c !important;
    border-color: rgba(252, 165, 165, 0.6) !important;
    box-shadow: 0 4px 14px rgba(252, 165, 165, 0.2), 0 2px 6px rgba(15, 23, 42, 0.06) !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_actions_"] [class*="st-key-force_stop_controls_"] [data-testid="stButton"] button::before {
    background: linear-gradient(135deg, rgba(252, 165, 165, 0.1), transparent 50%);
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_actions_"] [class*="st-key-force_stop_controls_"] [data-testid="stButton"] button:not(:disabled):hover {
    background: linear-gradient(135deg, rgba(252, 165, 165, 0.15), rgba(254, 202, 202, 0.12)) !important;
    border-color: #fca5a5 !important;
    box-shadow: 0 8px 24px rgba(252, 165, 165, 0.3), 0 4px 12px rgba(15, 23, 42, 0.08) !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_actions_"] [class*="st-key-force_stop_controls_"] [data-testid="stButton"] button:not(:disabled):active {
    box-shadow: 0 2px 8px rgba(252, 165, 165, 0.25) !important;
}
:is(.st-key-analysis_history_list, .st-key-document_history_list) [class*="st-key-history_actions_"] [class*="st-key-force_stop_controls_"] [data-testid="stButton"] button:focus-visible {
    outline-color: #fca5a5;
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
