from __future__ import annotations

import base64
import html
import json
import math
from uuid import uuid4
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.direct_recall.comparison import compute_direct_recall_overlap, DiffToken


_COLLAPSIBLE_COMPONENT_STYLE = """
<style>
    :root {
        color-scheme: light;
        --primary: #2563eb;
        --primary-800: #1e40af;
        --muted: #64748b;
    }
    * {
        box-sizing: border-box;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    body {
        margin: 0;
        padding: 0.15rem 0.2rem 0.4rem;
        background: transparent;
        color: #0f172a;
    }
    .cd-accordion-shell {
        width: 100%;
    }
    .cd-accordion {
        border: 1px solid rgba(209, 213, 225, 0.65);
        border-radius: 14px;
        margin: 0.4rem 0;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(241, 246, 255, 0.82));
        box-shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
        overflow: hidden;
        transition: box-shadow 0.2s ease;
    }
    .cd-accordion.compact {
        margin: 0;
        border-radius: 0;
        border-bottom: none;
        box-shadow: none;
    }
    .cd-accordion.compact:first-of-type {
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
    }
    .cd-accordion.compact:last-of-type {
        border-bottom-left-radius: 8px;
        border-bottom-right-radius: 8px;
        border-bottom: 1px solid rgba(209, 213, 225, 0.65);
    }
    .cd-accordion.compact[open] {
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
        z-index: 1;
        position: relative;
        border-bottom: 1px solid rgba(209, 213, 225, 0.65);
        margin: 0.2rem 0;
    }
    .cd-accordion[open] {
        box-shadow: 0 20px 48px rgba(15, 23, 42, 0.12);
    }
    .cd-accordion__summary {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.95rem 1.25rem 0.95rem 2.5rem;
        cursor: pointer;
        list-style: none;
        font-weight: 700;
        color: #0f172a;
        position: relative;
    }
    .cd-accordion__summary::-webkit-details-marker { display: none; }
    .cd-accordion__summary::before {
        content: '';
        position: absolute;
        left: 1rem;
        top: 50%;
        transform: translateY(-50%);
        width: 0;
        height: 0;
        border-left: 6px solid var(--primary-800);
        border-top: 4px solid transparent;
        border-bottom: 4px solid transparent;
        transition: transform 0.2s ease;
    }
    .cd-accordion[open] > .cd-accordion__summary::before {
        transform: translateY(-50%) rotate(90deg);
    }
    .cd-accordion__title {
        font-size: 0.98rem;
        letter-spacing: 0.4px;
    }
    .cd-accordion__meta {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--muted);
    }
    .cd-accordion__content {
        padding: 0.85rem 1.25rem 1.2rem;
        border-top: 1px solid rgba(209, 213, 225, 0.45);
        background: rgba(248, 250, 252, 0.85);
    }
    .cd-accordion__block + .cd-accordion__block {
        margin-top: 0.85rem;
        padding-top: 0.85rem;
        border-top: 1px dashed rgba(148, 163, 184, 0.4);
    }
    .cd-accordion__block-title {
        font-weight: 700;
        font-size: 0.9rem;
        color: #1e293b;
        margin-bottom: 0.35rem;
    }
    .cd-accordion__block-text {
        font-size: 0.9rem;
        line-height: 1.6;
        color: #334155;
        white-space: pre-wrap;
    }
    .cd-accordion__empty {
        font-size: 0.9rem;
        font-style: italic;
        color: rgba(71, 85, 105, 0.9);
    }
    .generated-text {
        padding: 1rem 1.25rem;
        background: linear-gradient(135deg, rgba(239, 246, 255, 0.95), rgba(222, 235, 255, 0.8));
        border: 1px solid rgba(191, 219, 254, 0.85);
        border-left: 5px solid var(--primary);
        font-family: Georgia, serif;
        line-height: 1.75;
        color: #1d4ed8;
        margin: 0.75rem 0;
        border-radius: 0 16px 16px 0;
        font-size: 0.95rem;
        white-space: pre-wrap;
        word-wrap: break-word;
        box-shadow: 0 14px 28px rgba(30, 64, 175, 0.12);
    }
    .generated-text.sm {
        font-size: 0.85rem;
        line-height: 1.6;
        padding: 0.65rem 0.9rem;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 10px 20px rgba(30, 64, 175, 0.08);
    }
</style>
"""

_DIRECT_RECALL_DIFF_STYLE = """
<style>
    :root {
        --dr-match: linear-gradient(135deg, #22c55e, #16a34a);
        --dr-miss: linear-gradient(135deg, #f59e0b, #d97706);
        --dr-extra: linear-gradient(135deg, #f87171, #ef4444);
    }
    .dr-diff-wrapper {
        margin: 0.9rem 0 0.6rem;
        padding: 1rem 1.15rem 1.05rem;
        border-radius: 18px;
        border: 1px solid rgba(203, 213, 225, 0.75);
        background: linear-gradient(145deg, rgba(255,255,255,0.96), rgba(240, 245, 255, 0.9));
        box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
    }
    .dr-diff-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.65rem;
        letter-spacing: 0.3px;
    }
    .dr-diff-columns {
        display: flex;
        flex-wrap: wrap;
        gap: 0.85rem;
    }
    .dr-diff-column {
        flex: 1 1 0;
        min-width: 260px;
        background: rgba(248, 250, 252, 0.75);
        border: 1px solid rgba(191, 219, 254, 0.6);
        border-radius: 14px;
        padding: 0.75rem 0.85rem 0.8rem;
        position: relative;
    }
    .dr-diff-column::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: inherit;
        pointer-events: none;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.45);
    }
    .dr-diff-column__title {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #475569;
        letter-spacing: 0.42px;
        margin-bottom: 0.4rem;
    }
    .dr-diff-column__body {
        font-size: 0.9rem;
        line-height: 1.7;
        color: #1f2937;
        background: rgba(255, 255, 255, 0.85);
        border-radius: 12px;
        border-left: 4px solid #2563eb;
        padding: 0.65rem 0.75rem;
        white-space: normal;
        word-break: break-word;
    }
    .dr-diff-column__body--ground {
        border-left-color: #dc2626;
    }
    .dr-diff-column__body--generated {
        border-left-color: #2563eb;
    }
    .dr-token {
        display: inline;
        padding: 0.08rem 0.14rem;
        margin: 0 0.02rem;
        border-radius: 8px;
        font-weight: 510;
        box-decoration-break: clone;
        -webkit-box-decoration-break: clone;
        transition: background 0.18s ease;
    }
    .dr-token--match {
        background: rgba(34, 197, 94, 0.26);
        color: #14532d;
    }
    .dr-token--miss {
        background: rgba(250, 204, 21, 0.26);
        color: #78350f;
    }
    .dr-token--extra {
        background: rgba(248, 113, 113, 0.28);
        color: #7f1d1d;
    }
    .dr-token br {
        display: inline;
    }
    .dr-diff-legend {
        margin-top: 0.7rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        font-size: 0.77rem;
        color: #475569;
    }
    .dr-legend-item {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        background: rgba(241, 245, 249, 0.85);
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
        font-weight: 600;
        letter-spacing: 0.35px;
    }
    .dr-legend-chip {
        width: 0.75rem;
        height: 0.75rem;
        border-radius: 50%;
        background: rgba(148, 163, 184, 0.5);
    }
    .dr-legend-chip.match { background: var(--dr-match); }
    .dr-legend-chip.miss { background: var(--dr-miss); }
    .dr-legend-chip.extra { background: var(--dr-extra); }
    .dr-diff-metrics {
        margin-top: 0.65rem;
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        font-size: 0.78rem;
        color: #334155;
    }
    .dr-diff-metric {
        background: rgba(226, 232, 240, 0.35);
        border-radius: 12px;
        padding: 0.48rem 0.7rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65);
    }
    .dr-diff-metric strong {
        color: #0f172a;
        font-size: 0.82rem;
    }
    .dr-diff-metric-detail {
        display: block;
        margin-top: 0.18rem;
        font-size: 0.7rem;
        color: #64748b;
        font-weight: 500;
        letter-spacing: 0.25px;
    }
    @media (max-width: 820px) {
        .dr-diff-wrapper { padding: 0.95rem; }
        .dr-diff-column { min-width: 100%; }
    }
</style>
"""


def _escape_diff_token(token: str) -> str:
    escaped = html.escape(token)
    escaped = escaped.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
    escaped = escaped.replace(" ", "&nbsp;")
    return escaped.replace("\n", "<br />")


def _build_diff_html(tokens: List[DiffToken]) -> str:
    class_map = {
        "match": "dr-token dr-token--match",
        "miss": "dr-token dr-token--miss",
        "extra": "dr-token dr-token--extra",
    }
    parts: List[str] = []
    for token in tokens:
        text = _escape_diff_token(token.text)
        css_class = class_map.get(token.label)
        if css_class:
            parts.append(f'<span class="{css_class}">{text}</span>')
        else:
            parts.append(text)
    return "".join(parts)


def render_direct_recall_diff(
    ground_truth: str,
    generated_text: str,
    *,
    title: Optional[str] = None,
    metrics: Optional[Dict[str, float]] = None,
    rouge_score: Optional[float] = None,
    jaccard_index: Optional[float] = None,
    levenshtein_dist: Optional[int] = None,
) -> Dict[str, int]:
    """Render a side-by-side comparison highlighting token-level matches and errors."""

    overlap = compute_direct_recall_overlap(ground_truth or "", generated_text or "")
    ground_html = _build_diff_html(overlap["ground_tokens"])
    generated_html = _build_diff_html(overlap["generated_tokens"])
    counts = overlap["counts"]

    ground_total = overlap["ground_non_whitespace"] or 0
    generated_total = overlap["generated_non_whitespace"] or 0
    recall_pct = (
        f"{(counts['match'] / ground_total) * 100:.1f}%"
        if ground_total
        else "—"
    )
    precision_pct = (
        f"{(counts['match'] / generated_total) * 100:.1f}%"
        if generated_total
        else "—"
    )

    legend_html = (
        f'<div class="dr-diff-legend">'
        f'<span class="dr-legend-item"><span class="dr-legend-chip match"></span>Matches: {counts["match"]}</span>'
        f'<span class="dr-legend-item"><span class="dr-legend-chip miss"></span>Missed (Ground Truth Only): {counts["miss"]}</span>'
        f'<span class="dr-legend-item"><span class="dr-legend-chip extra"></span>Extra (Model Generation Only): {counts["extra"]}</span>'
        f"</div>"
    )

    metrics_data: Dict[str, float] = {}
    if metrics:
        metrics_data = {k: v for k, v in metrics.items() if v is not None}
    elif any(value is not None for value in (rouge_score, jaccard_index, levenshtein_dist)):
        if rouge_score is not None:
            metrics_data["rouge_l"] = rouge_score
        if jaccard_index is not None:
            metrics_data["jaccard_index"] = jaccard_index
        if levenshtein_dist is not None:
            metrics_data["levenshtein"] = float(levenshtein_dist)

    # Ensure numeric conversion for consistent formatting
    metrics_data = {
        key: float(value)
        for key, value in metrics_data.items()
        if isinstance(value, (int, float))
    }

    metric_entries: List[str] = []
    metric_spec = [
        ("rouge_1", "ROUGE-1", "{:.4f}", None),
        ("rouge_l", "ROUGE-L", "{:.4f}", None),
        ("lcs_char_ratio", "LCS (Character)", "{:.4f}", ("lcs_char_length", "len: {:.0f}")),
        ("lcs_word_ratio", "LCS (Word)", "{:.4f}", ("lcs_word_length", "len: {:.0f}")),
        ("acs_word", "ACS (Word)", "{:.4f}", None),
        ("jaccard_index", "Jaccard", "{:.4f}", None),
        ("levenshtein", "Levenshtein", "{:.0f}", None),
        ("semantic_similarity", "Semantic Similarity", "{:.4f}", None),
        ("minhash_similarity", "MinHash Similarity", "{:.4f}", None),
    ]

    for key, label, fmt, detail in metric_spec:
        value = metrics_data.get(key)
        if value is None:
            continue
        try:
            formatted_value = fmt.format(value)
        except (ValueError, TypeError):
            continue
        detail_html = ""
        if detail and detail[0] in metrics_data:
            detail_value = metrics_data[detail[0]]
            detail_html = f'<span class="dr-diff-metric-detail">{detail[1].format(detail_value)}</span>'
        metric_entries.append(
            f'<div class="dr-diff-metric">{label}: <strong>{formatted_value}</strong>{detail_html}</div>'
        )

    metric_entries.append(
        f'<div class="dr-diff-metric">Recall Coverage: <strong>{recall_pct}</strong></div>'
    )
    metric_entries.append(
        f'<div class="dr-diff-metric">Precision: <strong>{precision_pct}</strong></div>'
    )

    metrics_html = f'<div class="dr-diff-metrics">{"".join(metric_entries)}</div>'

    section_title = title or "Recall Comparison"

    st.markdown(
        f"""
        {_DIRECT_RECALL_DIFF_STYLE}<div class="dr-diff-wrapper">
            <div class="dr-diff-title">{html.escape(section_title)}</div>
            <div class="dr-diff-columns">
                <div class="dr-diff-column">
                    <div class="dr-diff-column__title">Ground Truth</div>
                    <div class="dr-diff-column__body dr-diff-column__body--ground">{ground_html}</div>
                </div>
                <div class="dr-diff-column">
                    <div class="dr-diff-column__title">Model Output</div>
                    <div class="dr-diff-column__body dr-diff-column__body--generated">{generated_html}</div>
                </div>
            </div>
            {legend_html}
            {metrics_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    return counts


def render_prompt_preview(
    prompt_text: str,
    *,
    expanded: bool = False,
    title: str = "Prompt Preview",
) -> None:
    """Render a collapsible text preview using Streamlit native components."""
    
    raw_text = prompt_text or ""
    
    # Use Streamlit's built-in expander component
    with st.expander(f"📝 {title}", expanded=expanded):
        # Display the prompt in a container with custom styling to prevent cursor-not-allowed
        st.markdown(
            """
            <style>
            div[data-testid="stTextArea"] textarea[disabled] {
                cursor: text !important;
                color: rgba(49, 51, 63, 1) !important;
                -webkit-text-fill-color: rgba(49, 51, 63, 1) !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Display the prompt in a read-only text area (normal text display, auto-wrapping)
        st.text_area(
            label="Prompt Content",
            value=raw_text,
            height=200,
            disabled=True,
            label_visibility="collapsed",
            key=f"prompt_preview_{hash(raw_text)}"  # Unique key to avoid conflicts
        )
        
        # Add a caption with character count
        char_count = len(raw_text)
        word_count = len(raw_text.split())
        st.caption(f"📊 {char_count} characters, {word_count} words")


def render_prompt_style_panel(
    title: str,
    sections: Iterable[AccordionSection],
    *,
    meta: Optional[str] = None,
    expanded: bool = False,
) -> None:
    """Render a collapsible panel reusing the prompt preview styling."""

    open_attr = "open" if expanded else ""
    safe_title = html.escape(title)

    meta_bits = []
    if meta:
        for part in meta.split("|"):
            item = part.strip()
            if item:
                meta_bits.append(html.escape(item))

    if meta_bits:
        chips_html = "".join(
            f'<span class="cd-prompt-panel__chip">{bit}</span>' for bit in meta_bits
        )
        meta_html = f'<div class="cd-prompt-panel__meta">{chips_html}</div>'
    else:
        meta_html = ""

    section_html_parts: List[str] = []
    for heading, body, variant in sections:
        safe_heading = html.escape(heading)
        body_text = body or ""
        safe_body = html.escape(body_text)

        if variant == "generated":
            body_block = f'<div class="generated-text sm">{safe_body}</div>'
        else:
            body_block = f'<div class="cd-prompt-panel__body">{safe_body}</div>'

        section_html_parts.append(
            f'<div class="cd-prompt-panel__section">'
            f'<div class="cd-prompt-panel__heading">{safe_heading}</div>'
            f'{body_block}'
            f'</div>'
        )

    if not section_html_parts:
        section_html_parts.append(
            '<div class="cd-prompt-panel__section">'
            '<div class="cd-prompt-panel__heading">No details available</div>'
            '<div class="cd-prompt-panel__body cd-prompt-panel__body--empty">No content supplied.</div>'
            '</div>'
        )

    st.markdown(
        f"""
        <div class="cd-prompt-preview cd-prompt-panel">
            <details class="cd-prompt-preview__container cd-prompt-panel__container" {open_attr}>
                <summary class="cd-prompt-preview__summary cd-prompt-panel__summary">
                    <span class="cd-prompt-panel__title">{safe_title}</span>
                    {meta_html}
                </summary>
                <div class="cd-prompt-preview__content cd-prompt-panel__content">
                    {''.join(section_html_parts)}
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
    compact: bool = False,
    max_height: Optional[int] = None,
) -> None:
    """Render a custom collapsible panel via an isolated HTML component."""

    open_attr = "open" if expanded else ""
    compact_class = " compact" if compact else ""
    escaped_title = html.escape(title)
    meta_html = f'<span class="cd-accordion__meta">{html.escape(meta)}</span>' if meta else ""

    content_html_parts = []
    total_lines = 0

    for heading, body, variant in sections:
        safe_heading = html.escape(heading)
        body_text = body or ""
        safe_body = html.escape(body_text)
        if body_text:
            line_breaks = body_text.count("\n") + 1
            soft_wrap_estimate = math.ceil(len(body_text) / 90)
            approx_lines = max(line_breaks, soft_wrap_estimate)
        else:
            approx_lines = 1
        total_lines += approx_lines

        if variant == "generated":
            block_html = (
                f'<div class="cd-accordion__block">'
                f'<div class="cd-accordion__block-title">{safe_heading}</div>'
                f'<div class="generated-text">{safe_body}</div>'
                f"</div>"
            )
        else:
            block_html = (
                f'<div class="cd-accordion__block">'
                f'<div class="cd-accordion__block-title">{safe_heading}</div>'
                f'<div class="cd-accordion__block-text">{safe_body}</div>'
                f"</div>"
            )
        content_html_parts.append(block_html)

    if not content_html_parts:
        content_html_parts.append('<div class="cd-accordion__empty">No content available.</div>')

    body_html = "".join(content_html_parts)

    panel_html = f"""
        <div class="cd-accordion-shell">
            <details class="cd-accordion{compact_class}" {open_attr}>
                <summary class="cd-accordion__summary">
                    <span class="cd-accordion__title">{escaped_title}</span>
                    {meta_html}
                </summary>
                <div class="cd-accordion__content">
                    {body_html}
                </div>
            </details>
        </div>
    """

    section_count = max(len(content_html_parts), 1)
    estimated_height = 160 + total_lines * 22 + section_count * 18
    max_height = max_height or 780
    height = min(max_height, max(220, estimated_height))
    scrolling = height >= max_height

    components.html(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset=\"utf-8\" />
            {_COLLAPSIBLE_COMPONENT_STYLE}
        </head>
        <body>{panel_html}</body>
        </html>
        """,
        height=height,
        scrolling=scrolling,
    )


def render_compact_mutation_panels(
    panels: Sequence[dict],
    *,
    columns: int = 2,
    expanded_index: Optional[int] = None,
) -> None:
    """Render a responsive grid of compact collapsible panels used for mutation judging results.

    Each panel dict should contain:
      - title: str
      - meta: Optional[str]
      - sections: Sequence[AccordionSection]
      - expanded (optional): bool

    Only one panel is expanded at a time (if expanded_index specified). Mimics the
    interaction pattern requested: tiles sit side-by-side (default 2 columns) and
    expanding one pushes the rest downward.
    """

    if not panels:
        st.info("No judging results to display yet.")
        return

    # Build HTML for all panels in one iframe so CSS grid works.
    # Clamp columns between 1 and 4 for safety
    try:
        columns = max(1, min(int(columns), 4))
    except Exception:  # pragma: no cover
        columns = 2

    style_block = """
    <style>
        :root {{ color-scheme: light; --primary:#2563eb; --primary-hover:#1d4ed8; --slate:#334155; }}
        body {{ margin:0; padding:0.25rem 0.15rem 0.15rem; background:transparent; font-family:'Inter',system-ui,sans-serif; }}
        .cd-mgrid {{ display:grid; grid-template-columns:repeat({columns}, 1fr); gap:0.65rem; align-items:start; margin-bottom:0; }}
        @media (max-width: 900px) {{ .cd-mgrid {{ grid-template-columns:1fr; }} }}
        details.cd-mtile {{
            position:relative;
            border:1px solid rgba(203,213,225,0.75);
            border-radius:16px;
            background:linear-gradient(145deg,rgba(255,255,255,0.98),rgba(248,250,252,0.92));
            box-shadow:0 8px 20px rgba(15,23,42,0.06), 0 2px 6px rgba(15,23,42,0.04);
            overflow:hidden;
            transition:all .22s cubic-bezier(0.4,0,0.2,1);
        }}
        details.cd-mtile:hover {{
            box-shadow:0 12px 28px rgba(15,23,42,0.1), 0 4px 10px rgba(15,23,42,0.06);
            transform:translateY(-2px);
        }}
        details.cd-mtile[open] {{
            box-shadow:0 20px 40px rgba(15,23,42,0.14), 0 8px 16px rgba(15,23,42,0.08);
            border-color:rgba(148,163,184,0.85);
            grid-column:span var(--span,1);
            transform:translateY(0);
        }}
        /* Enhanced status accent bars with glow effect */
        .cd-mtile[data-status="pass"] {{ 
            border-left:5px solid #16a34a; 
            background:linear-gradient(145deg,rgba(240,253,244,0.4),rgba(248,250,252,0.92));
        }}
        .cd-mtile[data-status="fail"] {{ 
            border-left:5px solid #dc2626; 
            background:linear-gradient(145deg,rgba(254,242,242,0.4),rgba(248,250,252,0.92));
        }}
        .cd-mtile[data-status="pending"] {{ 
            border-left:5px solid #f59e0b; 
            background:linear-gradient(145deg,rgba(255,251,235,0.4),rgba(248,250,252,0.92));
        }}
        .cd-mtile[data-status="unclear"] {{ 
            border-left:5px solid #6366f1; 
            background:linear-gradient(145deg,rgba(238,242,255,0.4),rgba(248,250,252,0.92));
        }}
        summary.cd-mtile__summary {{
            list-style:none; cursor:pointer; display:flex; flex-direction:column; gap:0.3rem;
            padding:0.75rem 0.95rem 0.8rem 2rem; position:relative; font-weight:600; color:#0f172a;
            background:linear-gradient(135deg,rgba(251,252,253,0.95),rgba(243,246,249,0.7));
            border-bottom:1px solid transparent;
            transition:all .18s ease;
        }}
        summary.cd-mtile__summary:hover {{
            background:linear-gradient(135deg,rgba(248,250,252,0.98),rgba(240,244,248,0.85));
        }}
        details.cd-mtile[open] > summary.cd-mtile__summary {{
            border-bottom-color:rgba(203,213,225,0.5);
        }}
        summary.cd-mtile__summary::-webkit-details-marker {{ display:none; }}
        summary.cd-mtile__summary::before {{
            content:''; position:absolute; left:0.75rem; top:0.95rem; width:0.7rem; height:0.7rem;
            border-right:2.5px solid var(--primary); border-bottom:2.5px solid var(--primary);
            transform:rotate(-45deg); transition:all .2s cubic-bezier(0.4,0,0.2,1);
        }}
        details.cd-mtile[open] > summary.cd-mtile__summary::before {{ 
            transform:rotate(45deg); top:1.05rem; 
            border-color:var(--primary-hover);
        }}
        .cd-mtile__title {{ font-size:0.9rem; letter-spacing:0.25px; color:#1e293b; }}
        .cd-mtile__meta {{ font-size:0.68rem; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:0.6px; }}
        .cd-mtile__chips {{ display:flex; flex-wrap:wrap; gap:0.32rem; margin-top:0.2rem; }}
        .cd-chip {{
            --chip-bg:rgba(226,232,240,0.7);
            --chip-color:#334155;
            display:inline-flex; align-items:center; gap:0.28rem;
            padding:0.2rem 0.5rem; border-radius:999px; font-size:0.6rem; font-weight:700;
            background:var(--chip-bg); color:var(--chip-color); letter-spacing:0.4px;
            box-shadow:0 1px 3px rgba(15,23,42,0.1);
            transition:all .15s ease;
        }}
        .cd-chip:hover {{ transform:translateY(-1px); box-shadow:0 2px 6px rgba(15,23,42,0.15); }}
        .cd-chip.score {{ --chip-bg:linear-gradient(135deg,#1d4ed8,#2563eb); --chip-color:#f1f5f9; }}
        .cd-chip.jaccard {{ --chip-bg:linear-gradient(135deg,#0891b2,#06b6d4); --chip-color:#ecfeff; }}
        .cd-chip.lev {{ --chip-bg:linear-gradient(135deg,#475569,#64748b); --chip-color:#f1f5f9; }}
        .cd-chip.pass {{ --chip-bg:linear-gradient(135deg,#16a34a,#15803d); --chip-color:#f0fdf4; }}
        .cd-chip.fail {{ --chip-bg:linear-gradient(135deg,#dc2626,#b91c1c); --chip-color:#fef2f2; }}
        .cd-chip.pending {{ --chip-bg:linear-gradient(135deg,#f59e0b,#d97706); --chip-color:#fffbeb; }}
        .cd-chip.unclear {{ --chip-bg:linear-gradient(135deg,#6366f1,#4f46e5); --chip-color:#eef2ff; }}
        .cd-mtile__content {{ 
            padding:0.75rem 0.95rem 0.9rem; 
            border-top:1px solid rgba(203,213,225,0.45); 
            background:linear-gradient(to bottom,rgba(249,250,251,0.85),rgba(248,250,252,0.6));
        }}
        .cd-mtile__block + .cd-mtile__block {{ margin-top:0.65rem; padding-top:0.7rem; border-top:1px dashed rgba(148,163,184,0.32); }}
        .cd-mtile__block-title {{ 
            font-size:0.75rem; font-weight:700; color:#1e293b; margin-bottom:0.28rem; 
            letter-spacing:0.6px; text-transform:uppercase; 
        }}
        .cd-mtile__text {{ font-size:0.78rem; line-height:1.6; color:#334155; white-space:pre-wrap; }}
        .cd-mtile__text.generated {{
            background:linear-gradient(135deg,rgba(239,246,255,0.96),rgba(224,237,255,0.88));
            border:1px solid rgba(191,219,254,0.75); border-left:4px solid var(--primary);
            padding:0.55rem 0.7rem; border-radius:0 14px 14px 0; font-family:Georgia,serif; 
            font-size:0.8rem; color:#1e40af; line-height:1.65;
            box-shadow:0 2px 8px rgba(29,78,216,0.08);
        }}
    </style>
    """
    # Inject the columns value into the CSS placeholder. Using replace here avoids needing to escape
    # all literal braces in the CSS block for an f-string.
    style_block = style_block.replace("{columns}", str(columns))

    panel_html_parts = []
    total_lines = 0
    for i, panel in enumerate(panels):
        title = html.escape(panel.get("title") or f"Panel {i+1}")
        meta = panel.get("meta")
        sections: Sequence[AccordionSection] = panel.get("sections") or []
        is_open = False
        if expanded_index is not None:
            is_open = (i == expanded_index)
        else:
            is_open = bool(panel.get("expanded"))
        open_attr = "open" if is_open else ""
        status = panel.get("status") or "pending"
        status_class = html.escape(status.lower())
        meta_html = f'<div class="cd-mtile__meta">{html.escape(meta)}</div>' if meta else ""

        # Metric chips
        chips_spec = panel.get("chips") or []  # list of (label, class)
        chip_html_parts = []
        for label, cls in chips_spec:
            chip_html_parts.append(f'<span class="cd-chip {html.escape(cls)}">{html.escape(label)}</span>')
        chips_html = f'<div class="cd-mtile__chips">{"".join(chip_html_parts)}</div>' if chip_html_parts else ""

        blocks_html = []
        for heading, body, variant in sections:
            h_safe = html.escape(heading)
            b_text = body or ""
            b_safe = html.escape(b_text)
            lines = b_text.count("\n") + 1 if b_text else 1
            approx_soft = max(1, math.ceil(len(b_text) / 90))
            total_lines += max(lines, approx_soft)
            text_class = "cd-mtile__text generated" if variant == "generated" else "cd-mtile__text"
            blocks_html.append(
                f'<div class="cd-mtile__block"><div class="cd-mtile__block-title">{h_safe}</div>'
                f'<div class="{text_class}">{b_safe}</div></div>'
            )

        if not blocks_html:
            blocks_html.append('<div class="cd-mtile__block"><div class="cd-mtile__block-title">Empty</div><div class="cd-mtile__text">No content</div></div>')

        panel_html_parts.append(
            f'<details class="cd-mtile" data-status="{status_class}" {open_attr}><summary class="cd-mtile__summary">'
            f'<span class="cd-mtile__title">{title}</span>{meta_html}{chips_html}</summary>'
            f'<div class="cd-mtile__content">{"".join(blocks_html)}</div></details>'
        )

    grid_html = "".join(panel_html_parts)

    estimated_height = min(1400, 120 + total_lines * 14)
    components.html(
        f"""
        <!DOCTYPE html>
        <html><head><meta charset='utf-8' />{style_block}</head>
        <body><div class='cd-mgrid'>{grid_html}</div></body></html>
        """,
        height=max(360, estimated_height),
        scrolling=True,
    )


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


from contextlib import contextmanager


@contextmanager
def render_streamlit_accordion(title: str, *, key: Optional[str] = None, expanded: bool = False, help: Optional[str] = None):
    """A lightweight Streamlit-native accordion helper.

    Usage:
        with render_streamlit_accordion("Advanced settings", key="acc1"):
            st.slider(...)

    This preserves Streamlit interactive widgets and their keys (unlike HTML/iframe components)
    while visually grouping controls under a summary line. It uses an st.expander under the hood
    but provides a single shared place to add extra styling or replace implementation later.
    """

    # For now, we wrap st.expander to ensure correct widget behavior. Using a contextmanager
    # keeps the call-site simple and makes it straightforward to later swap to a pure-Streamlit
    # implementation if desired.
    if help:
        summary = f"{title} — {help}"
    else:
        summary = title

    exp = st.expander(summary, expanded=expanded)
    try:
        with exp:
            yield
    finally:
        pass
