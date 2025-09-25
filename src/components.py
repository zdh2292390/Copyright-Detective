from __future__ import annotations

import html
import json
from typing import Final

import streamlit.components.v1 as components


_COMPONENT_BASE_HEIGHT: Final[int] = 180
_COMPONENT_MAX_HEIGHT: Final[int] = 600


def _compute_height(prompt_text: str) -> int:
    """Estimate an appropriate iframe height for the custom component."""

    line_count = prompt_text.count("\n") + 1
    approx_height = _COMPONENT_BASE_HEIGHT + (line_count * 18)
    return min(_COMPONENT_MAX_HEIGHT, max(_COMPONENT_BASE_HEIGHT, approx_height))


def _build_component_html(prompt_text: str, expanded: bool) -> str:
    escaped_text = html.escape(prompt_text)
    copy_payload = json.dumps(prompt_text)
    open_attr = "open" if expanded else ""
    return f"""
    <div class="cd-prompt-preview">
        <details class="cd-prompt-preview__container" {open_attr}>
            <summary class="cd-prompt-preview__summary">Prompt Preview</summary>
            <div class="cd-prompt-preview__content">
                <button class="cd-prompt-preview__copy" onclick='navigator.clipboard.writeText({copy_payload}).then(() => {{
                    const btn = this;
                    const previous = btn.innerText;
                    btn.innerText = "Copied!";
                    setTimeout(() => btn.innerText = previous, 2000);
                }}); event.stopPropagation(); return false;'>Copy</button>
                <pre class="cd-prompt-preview__text">{escaped_text}</pre>
            </div>
        </details>
    </div>
    <style>
    .cd-prompt-preview__container {{
        border: 1px solid #d0d4dd;
        border-radius: 12px;
        background: linear-gradient(145deg, #f7f9fc 0%, #eef2f8 100%);
        padding: 14px;
        color: #1f2937;
        font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
        transition: box-shadow 0.2s ease;
    }}
    .cd-prompt-preview__container[open] {{
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12);
    }}
    .cd-prompt-preview__summary {{
        list-style: none;
        cursor: pointer;
        font-weight: 600;
        font-size: 1.05rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #1d4ed8;
    }}
    .cd-prompt-preview__summary::-webkit-details-marker {{
        display: none;
    }}
    .cd-prompt-preview__summary::before {{
        content: "\u25B6";
        display: inline-flex;
        transition: transform 0.2s ease;
    }}
    .cd-prompt-preview__container[open] > .cd-prompt-preview__summary::before {{
        transform: rotate(90deg);
    }}
    .cd-prompt-preview__content {{
        position: relative;
        background: #0f172a;
        color: #f8fafc;
        border-radius: 10px;
        margin-top: 14px;
        padding: 18px;
        overflow: auto;
        max-height: 100%;
    }}
    .cd-prompt-preview__copy {{
        position: absolute;
        top: 12px;
        right: 12px;
        background: rgba(148, 163, 184, 0.2);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 8px;
        color: #e2e8f0;
        padding: 6px 12px;
        font-size: 0.8rem;
        cursor: pointer;
        transition: background 0.2s ease, transform 0.2s ease;
    }}
    .cd-prompt-preview__copy:hover {{
        background: rgba(148, 163, 184, 0.35);
        transform: translateY(-1px);
    }}
    .cd-prompt-preview__text {{
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: "JetBrains Mono", "Fira Code", "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 0.95rem;
        line-height: 1.6;
    }}
    </style>
    """


def render_prompt_preview(prompt_text: str, *, expanded: bool = False) -> None:
    """Render the prompt preview through a custom HTML component."""

    safe_text = prompt_text or ""
    html_content = _build_component_html(safe_text, expanded)
    height = _compute_height(safe_text)
    components.html(html_content, height=height, scrolling=True)
