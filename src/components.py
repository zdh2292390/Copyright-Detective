from __future__ import annotations

import html
import json
from typing import Iterable, Optional, Tuple

import streamlit as st


def render_prompt_preview(prompt_text: str, *, expanded: bool = False) -> None:
    """Render the prompt preview inline so surrounding content flows naturally."""

    raw_text = prompt_text or ""
    escaped_text = html.escape(raw_text)
    copy_payload = json.dumps(raw_text).replace("'", "&apos;")
    open_attr = "open" if expanded else ""

    st.markdown(
        f"""
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
        """,
        unsafe_allow_html=True,
    )


AccordionSection = Tuple[str, str, Optional[str]]


def render_collapsible_panel(
    title: str,
    sections: Iterable[AccordionSection],
    *,
    meta: Optional[str] = None,
    expanded: bool = False,
) -> None:
    """Render a custom collapsible panel using native HTML details/summary."""

    open_attr = "open" if expanded else ""
    escaped_title = html.escape(title)
    meta_html = f'<span class="cd-accordion__meta">{html.escape(meta)}</span>' if meta else ""

    content_html_parts = []
    for heading, body, variant in sections:
        escaped_heading = html.escape(heading)
        escaped_body = html.escape(body or "").replace("\n", "<br/>")

        if variant == "generated":
            block_html = (
                f'<div class="cd-accordion__block">'
                f'<div class="cd-accordion__block-title">{escaped_heading}</div>'
                f'<div class="generated-text">{escaped_body}</div>'
                f"</div>"
            )
        else:
            block_html = (
                f'<div class="cd-accordion__block">'
                f'<div class="cd-accordion__block-title">{escaped_heading}</div>'
                f'<div class="cd-accordion__block-text">{escaped_body}</div>'
                f"</div>"
            )
        content_html_parts.append(block_html)

    content_html = "".join(content_html_parts)

    panel_html = f"""
    <details class="cd-accordion" {open_attr}>
        <summary class="cd-accordion__summary">
            <span class="cd-accordion__title">{escaped_title}</span>
            {meta_html}
        </summary>
        <div class="cd-accordion__content">
            {content_html}
        </div>
    </details>
    """

    st.markdown(panel_html, unsafe_allow_html=True)
