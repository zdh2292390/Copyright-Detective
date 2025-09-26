from __future__ import annotations

import html
import json
from typing import Iterable, Optional, Sequence, Tuple

import streamlit as st
import streamlit.components.v1 as components


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


def render_top_sample_distribution(records: Sequence[dict]) -> None:
    """Render the Top Sample Distribution table using a custom-styled component."""

    if not records:
        st.info("No top sample data available yet.")
        return

    header_html = """
        <tr>
            <th>#</th>
            <th>Frequency</th>
            <th>Probability</th>
            <th>Sample Preview</th>
        </tr>
    """

    row_html_parts = []
    for record in records:
        rank = html.escape(str(record.get("Rank", "")))
        frequency = html.escape(str(record.get("Frequency", "")))
        probability = record.get("Probability", 0.0)
        if isinstance(probability, (int, float)):
            probability_display = f"{probability:.3f}"
        else:
            probability_display = html.escape(str(probability))
        preview = html.escape(record.get("Sample Preview", "") or "")

        row_html_parts.append(
            f"""
            <tr>
                <td class="topk-cell-rank">{rank}</td>
                <td class="topk-cell-frequency">{frequency}</td>
                <td class="topk-cell-probability">{probability_display}</td>
                <td class="topk-cell-preview" title="{preview}">{preview}</td>
            </tr>
            """
        )

    rows_html = "".join(row_html_parts)

    style_block = """
        <style>
            :root { color-scheme: light; }
            * {
                box-sizing: border-box;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            body {
                margin: 0;
                padding: 0.25rem 0.4rem 0.6rem;
                background: transparent;
                color: #1e293b;
            }
            .topk-card {
                background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(240,245,255,0.85));
                border: 1px solid rgba(203, 213, 225, 0.7);
                border-radius: 14px;
                padding: 0.4rem 0.6rem 0.65rem;
                box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
            }
            .topk-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.82rem;
                color: #1e293b;
            }
            .topk-table thead th {
                text-transform: uppercase;
                font-size: 0.7rem;
                letter-spacing: 0.42px;
                font-weight: 700;
                color: #475569;
                padding: 0.35rem 0.45rem;
                border-bottom: 1px solid rgba(148, 163, 184, 0.45);
                background: rgba(226, 232, 240, 0.35);
            }
            .topk-table tbody td {
                padding: 0.36rem 0.45rem;
                border-bottom: 1px solid rgba(226, 232, 240, 0.5);
                vertical-align: middle;
            }
            .topk-table tbody tr:last-child td { border-bottom: none; }
            .topk-cell-rank {
                font-weight: 700;
                color: #1e40af;
                width: 2.8rem;
            }
            .topk-cell-frequency {
                font-weight: 600;
                color: #334155;
                width: 5.2rem;
            }
            .topk-cell-probability {
                font-weight: 700;
                color: #1d4ed8;
                width: 5.2rem;
            }
            .topk-cell-preview {
                font-size: 0.8rem;
                color: #1f2937;
                line-height: 1.45;
            }
            .topk-cell-preview:hover { color: #1e40af; }
        </style>
    """

    table_html = f"""
        <div class="topk-card">
            <table class="topk-table">
                <thead>{header_html}</thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
    """

    row_count = max(len(records), 1)
    estimated_height = min(420, 90 + row_count * 38)

    components.html(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset=\"utf-8\" />
            {style_block}
        </head>
        <body>
            {table_html}
        </body>
        </html>
        """,
        height=estimated_height,
        scrolling=False,
    )
