"""Clear Cache helpers and API error display."""

from __future__ import annotations

import html
from typing import Callable, Dict, Optional

import streamlit as st
import streamlit.components.v1 as components

from src.job_guard import reset_detection_job

ACTIVE_CLEAR_ID_KEY = "_active_clear_cache_id"
PENDING_CLEAR_CACHE_MOUNT_KEY = "_pending_clear_cache_mount"
PENDING_RERUN_HINT_KEY = "_pending_rerun_hint"
CLEAR_CACHE_QUERY_PARAM = "_fcc"

RERUN_HINT_MESSAGE = (
    "✅ **Run complete.** To run again, use **Clear Cache** at the top of the page "
    "to reset cached inputs and results, then configure parameters and run again."
)

_HANDLERS: Dict[str, Callable[[], None]] = {}


def set_active_clear_cache_id(clear_id: str) -> None:
    st.session_state[ACTIVE_CLEAR_ID_KEY] = clear_id


def get_active_clear_cache_id() -> Optional[str]:
    value = st.session_state.get(ACTIVE_CLEAR_ID_KEY)
    return str(value) if value else None


def register_clear_cache_handler(clear_id: str, handler: Callable[[], None]) -> None:
    _HANDLERS[clear_id] = handler


def handle_clear_cache_query_param() -> bool:
    """Run a registered clear-cache handler when the draggable button is clicked."""
    clear_id = st.query_params.get(CLEAR_CACHE_QUERY_PARAM)
    if not clear_id:
        return False

    handler = _HANDLERS.get(clear_id)
    if handler:
        handler()

    try:
        del st.query_params[CLEAR_CACHE_QUERY_PARAM]
    except Exception:
        qp = dict(st.query_params)
        qp.pop(CLEAR_CACHE_QUERY_PARAM, None)
        st.query_params.clear()
        for key, value in qp.items():
            st.query_params[key] = value
    return True


def render_draggable_clear_cache_button(
    clear_id: Optional[str] = None,
    *,
    mount_key: str = "default",
    label: str = "Clear Cache",
) -> None:
    """Render a draggable Clear Cache pill fixed over the entire viewport."""
    cid = clear_id or get_active_clear_cache_id()
    if not cid or cid not in _HANDLERS:
        return

    st.session_state["_fcc_rendered_run"] = mount_key

    safe_id = html.escape(cid, quote=True)
    safe_label = html.escape(label, quote=True)
    storage_key = html.escape(f"{cid}_{mount_key}", quote=True)
    query_param = html.escape(CLEAR_CACHE_QUERY_PARAM, quote=True)

    components.html(
        "<!DOCTYPE html>"
        "<html>"
        "<head>"
        '<meta charset="utf-8" />'
        "<style>"
        "html, body {"
        "margin: 0;"
        "padding: 0;"
        "background: transparent;"
        "overflow: hidden;"
        "font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;"
        "}"
        ".fcc-shell {"
        "position: fixed;"
        "inset: 0;"
        "pointer-events: none;"
        "z-index: 1;"
        "}"
        ".fcc-btn {"
        "position: fixed;"
        "left: var(--fcc-x, auto);"
        "top: var(--fcc-y, auto);"
        "right: var(--fcc-right, 1.25rem);"
        "bottom: var(--fcc-bottom, 1.25rem);"
        "display: inline-flex;"
        "align-items: center;"
        "gap: 0.35rem;"
        "padding: 0.35rem 0.75rem;"
        "border-radius: 999px;"
        "border: 1px solid rgba(239, 68, 68, 0.35);"
        "background: linear-gradient(145deg, #fff5f5, #ffe4e6);"
        "color: #b91c1c;"
        "font-size: 0.78rem;"
        "font-weight: 600;"
        "line-height: 1;"
        "box-shadow: 0 8px 18px rgba(239, 68, 68, 0.16);"
        "cursor: grab;"
        "user-select: none;"
        "touch-action: none;"
        "white-space: nowrap;"
        "pointer-events: auto;"
        "z-index: 2;"
        "}"
        ".fcc-btn:active { cursor: grabbing; }"
        ".fcc-btn:hover {"
        "border-color: rgba(220, 38, 38, 0.55);"
        "box-shadow: 0 10px 22px rgba(239, 68, 68, 0.22);"
        "}"
        "</style>"
        "</head>"
        "<body>"
        '<div class="fcc-shell">'
        '<button type="button" class="fcc-btn" id="fcc-btn" title="Drag to move · Click to clear cache">'
        "🗑️ "
        + safe_label
        + "</button>"
        "</div>"
        "<script>"
        "(function () {"
        'const clearId = "'
        + safe_id
        + '";'
        'const storageKey = "copyright_detective_fcc_" + "'
        + storage_key
        + '";'
        'const queryParam = "'
        + query_param
        + '";'
        "const btn = document.getElementById('fcc-btn');"
        "let dragging = false;"
        "let moved = false;"
        "let offsetX = 0;"
        "let offsetY = 0;"
        "function promoteToGlobalOverlay() {"
        "const frame = window.frameElement;"
        "if (!frame) return;"
        "frame.style.cssText = 'position:fixed!important;top:0!important;left:0!important;width:100vw!important;height:100vh!important;z-index:999999!important;pointer-events:none!important;border:none!important;background:transparent!important;';"
        "}"
        "function applyPosition(x, y) {"
        "btn.style.setProperty('--fcc-x', x + 'px');"
        "btn.style.setProperty('--fcc-y', y + 'px');"
        "btn.style.setProperty('--fcc-right', 'auto');"
        "btn.style.setProperty('--fcc-bottom', 'auto');"
        "}"
        "function defaultPosition() {"
        "const margin = 20;"
        "const rect = btn.getBoundingClientRect();"
        "const width = rect.width || 120;"
        "const height = rect.height || 36;"
        "return {"
        "x: Math.max(margin, window.innerWidth - width - margin),"
        "y: Math.max(margin, window.innerHeight - height - margin),"
        "};"
        "}"
        "function loadPosition() {"
        "try {"
        "const raw = window.localStorage.getItem(storageKey);"
        "if (!raw) {"
        "const pos = defaultPosition();"
        "applyPosition(pos.x, pos.y);"
        "return;"
        "}"
        "const parsed = JSON.parse(raw);"
        "if (typeof parsed.x === 'number' && typeof parsed.y === 'number') {"
        "applyPosition(parsed.x, parsed.y);"
        "return;"
        "}"
        "} catch (err) {}"
        "const pos = defaultPosition();"
        "applyPosition(pos.x, pos.y);"
        "}"
        "function savePosition(x, y) {"
        "try {"
        'window.localStorage.setItem(storageKey, JSON.stringify({ x: x, y: y }));'
        "} catch (err) {}"
        "}"
        "function triggerClear() {"
        "const target = window.top || window.parent || window;"
        "const url = new URL(target.location.href);"
        "url.searchParams.set(queryParam, clearId);"
        "target.location.href = url.toString();"
        "}"
        "function currentPosition() {"
        "const rect = btn.getBoundingClientRect();"
        "return { x: rect.left, y: rect.top };"
        "}"
        "btn.addEventListener('mousedown', (event) => {"
        "dragging = true;"
        "moved = false;"
        "const pos = currentPosition();"
        "offsetX = event.clientX - pos.x;"
        "offsetY = event.clientY - pos.y;"
        "event.preventDefault();"
        "});"
        "window.addEventListener('mousemove', (event) => {"
        "if (!dragging) return;"
        "moved = true;"
        "const rect = btn.getBoundingClientRect();"
        "const width = rect.width;"
        "const height = rect.height;"
        "const margin = 8;"
        "const x = Math.min("
        "Math.max(margin, event.clientX - offsetX),"
        "window.innerWidth - width - margin"
        ");"
        "const y = Math.min("
        "Math.max(margin, event.clientY - offsetY),"
        "window.innerHeight - height - margin"
        ");"
        "applyPosition(x, y);"
        "});"
        "window.addEventListener('mouseup', () => {"
        "if (!dragging) return;"
        "dragging = false;"
        "if (moved) {"
        "const pos = currentPosition();"
        "savePosition(pos.x, pos.y);"
        "}"
        "});"
        "btn.addEventListener('click', (event) => {"
        "if (moved) {"
        "event.preventDefault();"
        "moved = false;"
        "return;"
        "}"
        "triggerClear();"
        "});"
        "window.addEventListener('resize', () => {"
        "const pos = currentPosition();"
        "const rect = btn.getBoundingClientRect();"
        "const margin = 8;"
        "const x = Math.min("
        "Math.max(margin, pos.x),"
        "window.innerWidth - rect.width - margin"
        ");"
        "const y = Math.min("
        "Math.max(margin, pos.y),"
        "window.innerHeight - rect.height - margin"
        ");"
        "applyPosition(x, y);"
        "savePosition(x, y);"
        "});"
        "promoteToGlobalOverlay();"
        "loadPosition();"
        "})();"
        "</script>"
        "</body>"
        "</html>",
        height=0,
        scrolling=False,
    )


def show_error_with_clear_cache(message: str, *, clear_id: Optional[str] = None, mount_key: str = "error") -> None:
    """Show an error message (Clear Cache remains at the top of the page)."""
    st.error(message)


def is_api_failure_message(message: str) -> bool:
    if not message:
        return False
    lowered = message.lower()
    return (
        message.startswith("Error")
        or "error calling api" in lowered
        or "api key" in lowered
        or "authentication" in lowered
        or "invalid_request_error" in lowered
        or "401" in message
        or "403" in message
    )


def show_api_failure_if_needed(message: str, *, clear_id: Optional[str] = None) -> bool:
    """Show API/auth failures as inline errors when applicable."""
    if not is_api_failure_message(message):
        return False
    display = message if message.startswith("❌") or message.startswith("⚠️") else f"❌ {message}"
    show_error_with_clear_cache(display, clear_id=clear_id)
    return True


def show_rerun_hint_after_run(*, mount_key: str = "done") -> None:
    """After a completed run, point users to the top Clear Cache button."""
    st.markdown("---")
    st.info(RERUN_HINT_MESSAGE)


def queue_rerun_hint_after_run(*, mount_key: str = "done") -> None:
    """Defer the post-run hint until the bottom of the current page."""
    st.session_state[PENDING_RERUN_HINT_KEY] = mount_key


def render_pending_rerun_hint() -> None:
    """Render a queued post-run hint at the bottom of the page."""
    mount_key = st.session_state.pop(PENDING_RERUN_HINT_KEY, None)
    if not mount_key:
        mount_key = st.session_state.pop(PENDING_CLEAR_CACHE_MOUNT_KEY, None)
    if mount_key:
        show_rerun_hint_after_run(mount_key=mount_key)


def show_clear_cache_after_run(*, clear_id: Optional[str] = None, mount_key: str = "done") -> None:
    """Deprecated: use queue_rerun_hint_after_run for post-run guidance."""
    queue_rerun_hint_after_run(mount_key=mount_key)


def queue_clear_cache_after_run(*, mount_key: str = "done") -> None:
    """Defer post-run hint rendering until the end of the page."""
    queue_rerun_hint_after_run(mount_key=mount_key)


def render_pending_clear_cache_button() -> None:
    """Backward-compatible alias for render_pending_rerun_hint."""
    render_pending_rerun_hint()


def build_reset_and_rerun_handler(
    clear_id: str,
    keys_to_clear: list[str],
    *,
    rerun: Callable[[], None],
) -> Callable[[], None]:
    """Factory for page-specific clear-cache handlers."""

    def _handler() -> None:
        for key in keys_to_clear:
            st.session_state.pop(key, None)
        reset_detection_job()
        rerun()

    register_clear_cache_handler(clear_id, _handler)
    return _handler
