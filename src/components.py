from __future__ import annotations

import html
import json

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
