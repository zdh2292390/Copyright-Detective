from __future__ import annotations

import base64
import html
import json
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def render_prompt_preview(
    prompt_text: str,
    *,
    expanded: bool = False,
    title: str = "Prompt Preview",
) -> None:
    """Render a collapsible text preview with copy support."""

    raw_text = prompt_text or ""
    escaped_text = html.escape(raw_text)
    copy_payload = json.dumps(raw_text).replace("'", "&apos;")
    open_attr = "open" if expanded else ""
    summary_label = html.escape(title or "Prompt Preview")

    st.markdown(
        f"""
        <div class="cd-prompt-preview">
            <details class="cd-prompt-preview__container" {open_attr}>
                <summary class="cd-prompt-preview__summary">{summary_label}</summary>
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


_TABLE_COMPONENT_STYLE = """
    <style>
        :root { color-scheme: light; }
        .cd-table-shell { width: 100%; }
        .cd-table-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(241,248,255,0.88));
            border: 1px solid rgba(203, 213, 225, 0.8);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            box-shadow: 0 18px 38px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.1rem;
        }
        .cd-table-card--nested {
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(236,243,255,0.9));
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.9rem;
        }
        .cd-table-card__header {
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
            margin-bottom: 0.6rem;
        }
        .cd-table-card__title {
            font-size: 1.02rem;
            font-weight: 600;
            color: #0f172a;
        }
        .cd-table-card__subtitle {
            font-size: 0.85rem;
            color: #475569;
        }
        .cd-table-card__table-container {
            width: 100%;
            overflow-x: auto;
            border-radius: 12px;
            background: rgba(248, 250, 252, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.26);
        }
        .cd-table-card__table-container table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
            color: #1f2937;
        }
        .cd-table-card__table-container thead tr {
            background: rgba(226, 232, 240, 0.55);
        }
        .cd-table-card__table-container thead th {
            text-transform: uppercase;
            font-size: 0.7rem;
            letter-spacing: 0.4px;
            font-weight: 700;
            color: #475569;
            padding: 0.55rem 0.65rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.45);
        }
        .cd-table-card__table-container tbody td {
            padding: 0.55rem 0.65rem;
            border-bottom: 1px solid rgba(226, 232, 240, 0.5);
        }
        .cd-table-card__download {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            margin-top: 0.75rem;
            padding: 0.55rem 0.9rem;
            border-radius: 999px;
            background: linear-gradient(135deg, #1d4ed8, #2563eb);
            color: #f8fafc;
            font-weight: 600;
            text-decoration: none;
            font-size: 0.85rem;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.25);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .cd-table-card__download:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 28px rgba(37, 99, 235, 0.3);
        }
        .cd-table-card__extra {
            margin-top: 1.1rem;
        }
        .cd-table-card__extra-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: #1d4ed8;
            margin-bottom: 0.45rem;
        }
        .cd-table-card__empty {
            padding: 1.1rem 1.2rem;
            border-radius: 12px;
            border: 1px dashed rgba(148, 163, 184, 0.6);
            background: rgba(241, 245, 249, 0.6);
            color: #475569;
            font-size: 0.88rem;
        }
        .cd-table-card__badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.22rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            color: #1d4ed8;
            background: rgba(37, 99, 235, 0.12);
        }
        .cd-table-collapsible {
            border: 1px solid rgba(203, 213, 225, 0.7);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(241,245,255,0.92));
            margin-bottom: 1.2rem;
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
            overflow: hidden;
        }
        .cd-table-collapsible[open] {
            box-shadow: 0 18px 40px rgba(30, 64, 175, 0.12);
        }
        .cd-table-collapsible__summary {
            list-style: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            padding: 0.95rem 1.15rem;
        }
        .cd-table-collapsible__summary::-webkit-details-marker {
            display: none;
        }
        .cd-table-collapsible__summary::before {
            content: "";
            display: inline-flex;
            width: 1.1rem;
            height: 1.1rem;
            border-radius: 50%;
            background: linear-gradient(135deg, #1d4ed8, #2563eb);
            margin-right: 0.8rem;
            position: relative;
        }
        .cd-table-collapsible__summary::after {
            content: "";
            position: absolute;
            width: 0.45rem;
            height: 0.45rem;
            border-bottom: 2px solid #fff;
            border-right: 2px solid #fff;
            transform: rotate(45deg);
            margin-left: -1.38rem;
            margin-top: 0.15rem;
            transition: transform 0.2s ease;
        }
        .cd-table-collapsible[open] .cd-table-collapsible__summary::after {
            transform: rotate(225deg);
            margin-top: 0.1rem;
        }
        .cd-table-collapsible__title {
            font-size: 1rem;
            font-weight: 600;
            color: #0f172a;
        }
        .cd-table-collapsible__badge-area {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .cd-table-collapsible__body {
            padding: 0 1.15rem 1.15rem 1.15rem;
        }
        @media (max-width: 768px) {
            .cd-table-card {
                padding: 0.85rem;
            }
            .cd-table-collapsible__summary {
                flex-direction: column;
                align-items: flex-start;
                gap: 0.35rem;
            }
            .cd-table-collapsible__summary::before {
                margin-right: 0.4rem;
            }
            .cd-table-collapsible__summary::after {
                margin-left: -1.3rem;
            }
        }
    </style>
"""


def _sanitise_dataframe(dataframe: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if dataframe is None:
        return None
    sanitised = dataframe.copy()
    try:
        sanitised = sanitised.fillna("—")
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        sanitised = sanitised.replace({pd.NA: "—"})
    except Exception:  # pragma: no cover - defensive
        pass
    return sanitised


def _dataframe_to_html_table(dataframe: pd.DataFrame) -> str:
    sanitised = _sanitise_dataframe(dataframe)
    if sanitised is None:
        return ""
    return sanitised.to_html(
        index=False,
        escape=False,
        classes="cd-table-card__table",
        border=0,
        justify="left",
    )


def _build_download_anchor(dataframe: pd.DataFrame, filename: str, label: str) -> str:
    if not filename:
        return ""
    csv_bytes = dataframe.to_csv(index=False).encode("utf-8")
    payload = base64.b64encode(csv_bytes).decode("utf-8")
    escaped_label = html.escape(label)
    escaped_filename = html.escape(filename)
    return (
        f'<a class="cd-table-card__download" download="{escaped_filename}" '
        f'href="data:text/csv;base64,{payload}">{escaped_label}</a>'
    )


def _build_table_card_html(
    dataframe: Optional[pd.DataFrame],
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    download_filename: Optional[str] = None,
    download_label: str = "⬇️ Download CSV",
    empty_message: str = "No records available.",
    extra_sections: Optional[Sequence[Tuple[str, pd.DataFrame]]] = None,
    nested: bool = False,
) -> Tuple[str, int]:
    extra_sections = [
        (heading, section_df)
        for heading, section_df in (extra_sections or [])
        if section_df is not None and not section_df.empty
    ]

    classes = "cd-table-card cd-table-card--nested" if nested else "cd-table-card"
    header_parts = []
    if title:
        header_parts.append(f'<div class="cd-table-card__title">{html.escape(title)}</div>')
    if subtitle:
        header_parts.append(f'<div class="cd-table-card__subtitle">{html.escape(subtitle)}</div>')
    header_html = (
        f'<div class="cd-table-card__header">{"".join(header_parts)}</div>'
        if header_parts
        else ""
    )

    if dataframe is None or dataframe.empty:
        body_html = f'<div class="cd-table-card__empty">{html.escape(empty_message)}</div>'
        download_html = ""
        row_count = 0
    else:
        body_html = f'<div class="cd-table-card__table-container">{_dataframe_to_html_table(dataframe)}</div>'
        download_html = (
            _build_download_anchor(dataframe, download_filename, download_label)
            if download_filename
            else ""
        )
        row_count = len(dataframe)

    extras_html_parts = []
    extra_row_total = 0
    for heading, section_df in extra_sections:
        extras_html_parts.append(
            f'<div class="cd-table-card__extra">'
            f'<div class="cd-table-card__extra-title">{html.escape(heading)}</div>'
            f'<div class="cd-table-card__table-container">{_dataframe_to_html_table(section_df)}</div>'
            f"</div>"
        )
        extra_row_total += len(section_df)
    extras_html = "".join(extras_html_parts)

    card_html = (
        f'<div class="{classes}">{header_html}{body_html}{download_html}{extras_html}</div>'
    )

    return card_html, row_count + extra_row_total


def render_table_card(
    dataframe: Optional[pd.DataFrame],
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    download_filename: Optional[str] = None,
    download_label: str = "⬇️ Download CSV",
    empty_message: str = "No records available.",
) -> None:
    """Render a high-fidelity table card with optional download support."""

    card_html, total_rows = _build_table_card_html(
        dataframe,
        title=title,
        subtitle=subtitle,
        download_filename=download_filename,
        download_label=download_label,
        empty_message=empty_message,
    )

    estimated_height = min(740, 220 + max(total_rows, 1) * 36)

    components.html(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset=\"utf-8\" />
            {_TABLE_COMPONENT_STYLE}
        </head>
        <body>
            <div class=\"cd-table-shell\">{card_html}</div>
        </body>
        </html>
        """,
        height=estimated_height,
        scrolling=False,
    )


def render_collapsible_table_card(
    title: str,
    dataframe: Optional[pd.DataFrame],
    *,
    subtitle: Optional[str] = None,
    meta_badge: Optional[str] = None,
    download_filename: Optional[str] = None,
    download_label: str = "⬇️ Download CSV",
    empty_message: str = "No records captured for this configuration.",
    summary_sections: Optional[Sequence[Tuple[str, pd.DataFrame]]] = None,
    expanded: bool = False,
) -> None:
    """Render a collapsible table card with optional summary sections and download."""

    summary_sections = summary_sections or []
    card_html, total_rows = _build_table_card_html(
        dataframe,
        title=None,
        subtitle=None,
        download_filename=download_filename,
        download_label=download_label,
        empty_message=empty_message,
        extra_sections=summary_sections,
        nested=True,
    )

    badge_html = (
        f'<span class="cd-table-card__badge">{html.escape(meta_badge)}</span>'
        if meta_badge
        else ""
    )

    subtitle_html = (
        f'<div class="cd-table-card__subtitle" style="margin-bottom:0.9rem;">{html.escape(subtitle)}</div>'
        if subtitle
        else ""
    )

    open_attr = "open" if expanded else ""

    estimated_height = min(780, 260 + max(total_rows, 1) * 40)

    components.html(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset=\"utf-8\" />
            {_TABLE_COMPONENT_STYLE}
        </head>
        <body>
            <details class=\"cd-table-collapsible\" {open_attr}>
                <summary class=\"cd-table-collapsible__summary\">
                    <span class=\"cd-table-collapsible__title\">{html.escape(title)}</span>
                    <span class=\"cd-table-collapsible__badge-area\">{badge_html}</span>
                </summary>
                <div class=\"cd-table-collapsible__body\">{subtitle_html}{card_html}</div>
            </details>
        </body>
        </html>
        """,
        height=estimated_height,
        scrolling=False,
    )
