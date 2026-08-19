"""Single-stage GPT/Kimi direct-probing scaling playground."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.adversarial_persuasion_detection.plots import (
    build_rouge_l_strategy_histogram,
)
from src.background_jobs import (
    background_job_running,
    get_background_job,
    forget_background_job,
    render_background_job_status,
    submit_background_job,
)
from src.components import render_direct_recall_diff
from src.job_guard import finish_detection_job, render_run_button
from src.game_continuation import (
    BOOK_TITLE,
    DIRECT_PROBE_METHODS,
    FIXED_MODELS,
    GROUND_TRUTH,
    KIMI_PROVIDER,
    MAX_RUNS_PER_PROVIDER,
    OPENAI_PROVIDER,
    ScalingBatch,
    TARGET_WORD_COUNT,
    build_challenge_prompt,
    run_provider_scaling,
)
from src.auth import is_logged_in
from src.pages.sampling_controls import render_fragmented_temperature_top_p
from src.sidebar_utils import get_api_key_for_provider
from src.supabase_client import get_secret

_GAME_CSS = Path(__file__).resolve().parents[2] / "assets" / "game.css"
_LOGGER = logging.getLogger(__name__)
_PREFIX = "copyright_game2:"
_CLEAR_KEY = "_copyright_game2_clear_requested"
_SESSION_ID_KEY = "_copyright_game2_browser_session_id"
_KST = timezone(timedelta(hours=9))
_PROVIDERS = (OPENAI_PROVIDER, KIMI_PROVIDER)
_SIMILARITY_METRIC_SPECS = (
    ("rouge_1", "ROUGE-1"),
    ("rouge_l", "ROUGE-L"),
    ("jaccard_index", "Jaccard Index"),
    ("lcs_char_ratio", "LCS (Character Ratio)"),
    ("lcs_char_length", "LCS (Character Length)"),
    ("lcs_word_ratio", "LCS (Word Ratio)"),
    ("lcs_word_length", "LCS (Word Length)"),
    ("acs_word", "ACS (Word)"),
    ("levenshtein", "Levenshtein Distance"),
    ("semantic_similarity", "Semantic Similarity"),
    ("minhash_similarity", "MinHash Similarity"),
)
_PROVIDER_COMPARISON_METRICS = (
    ("rouge_l", "Avg ROUGE-L"),
    ("rouge_1", "Avg ROUGE-1"),
    ("jaccard_index", "Avg Jaccard"),
    ("lcs_word_ratio", "Avg LCS Word"),
    ("levenshtein", "Avg Levenshtein"),
)


def _normalize_metrics(metrics: Any) -> Dict[str, float]:
    if not metrics:
        return {}
    if isinstance(metrics, dict):
        normalized = dict(metrics)
    else:
        normalized = {}
        for key, _label in _SIMILARITY_METRIC_SPECS:
            if hasattr(metrics, key):
                normalized[key] = getattr(metrics, key)
        if hasattr(metrics, "jaccard"):
            normalized["jaccard"] = getattr(metrics, "jaccard")
    if "jaccard_index" not in normalized and "jaccard" in normalized:
        normalized["jaccard_index"] = normalized["jaccard"]

    cleaned: Dict[str, float] = {}
    for key, value in normalized.items():
        try:
            cleaned[key] = float(value)
        except (TypeError, ValueError):
            continue
    return cleaned


def _average_metric(result: ScalingBatch, metric_key: str) -> Optional[float]:
    values = []
    for attempt in result.attempts:
        metrics = _normalize_metrics(attempt.metrics)
        if metric_key in metrics:
            values.append(metrics[metric_key])
    if not values:
        return None
    return sum(values) / len(values)


def _format_optional(value: Optional[float], *, precision: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def _render_similarity_score_summary(metric_rows: List[Dict[str, Any]], *, expanded: bool = False) -> None:
    summary_rows: List[Dict[str, Any]] = []
    normalized_rows = [_normalize_metrics(row) for row in metric_rows]
    for metric_key, metric_label in _SIMILARITY_METRIC_SPECS:
        values = [row[metric_key] for row in normalized_rows if metric_key in row]
        if values:
            summary_rows.append(
                {
                    "Metric": metric_label,
                    "Minimum": min(values),
                    "Maximum": max(values),
                    "Average": sum(values) / len(values),
                }
            )
    if not summary_rows:
        return

    with st.expander("Similarity score summary", expanded=expanded):
        st.caption(
            "Same metric bundle used by Content Recall Detection. Levenshtein is a distance metric, so lower is closer."
        )
        st.dataframe(
            pd.DataFrame(summary_rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Metric": st.column_config.TextColumn("Metric"),
                "Minimum": st.column_config.NumberColumn("Minimum", format="%.4f"),
                "Maximum": st.column_config.NumberColumn("Maximum", format="%.4f"),
                "Average": st.column_config.NumberColumn("Average", format="%.4f"),
            },
        )


def _attempt_metric_caption(metrics: Any) -> str:
    normalized = _normalize_metrics(metrics)
    parts = []
    for key, label in (
        ("rouge_1", "ROUGE-1"),
        ("rouge_l", "ROUGE-L"),
        ("jaccard_index", "Jaccard"),
        ("levenshtein", "Levenshtein"),
    ):
        if key in normalized:
            parts.append(f"{label} {normalized[key]:.4f}")
    return " | ".join(parts)


def _key(name: str) -> str:
    return f"{_PREFIX}{name}"



def _browser_session_id() -> str:
    value = str(st.session_state.get(_SESSION_ID_KEY) or "")
    if not value:
        value = uuid4().hex
        st.session_state[_SESSION_ID_KEY] = value
    return value


def _job_key(provider: str) -> str:
    slug = "openai" if provider == OPENAI_PROVIDER else "kimi"
    return _key(f"{_browser_session_id()}:{slug}_background_job")


def _load_styles() -> None:
    try:
        st.markdown(
            f"<style>{_GAME_CSS.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )
    except OSError:
        pass


def _format_kst(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return datetime.now(_KST).strftime("%Y-%m-%d %H:%M:%S KST")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(_KST).strftime("%Y-%m-%d %H:%M:%S KST")
    except ValueError:
        return raw


def _provider_api_key(provider: str) -> str:
    api_keys = {
        "openai_api_key": str(
            st.session_state.get("sidebar_openai_api_key", "") or ""
        ),
        "kimi_api_key": str(
            st.session_state.get("sidebar_kimi_api_key", "") or ""
        ),
    }
    resolved = get_api_key_for_provider(provider, api_keys)
    if resolved:
        return resolved
    default_secret = {
        OPENAI_PROVIDER: "COPYRIGHT_GAME_OPENAI_API_KEY",
        KIMI_PROVIDER: "COPYRIGHT_GAME_KIMI_API_KEY",
    }.get(provider, "")
    return get_secret(default_secret).strip() if default_secret else ""

def _any_job_running() -> bool:
    return any(background_job_running(_job_key(provider)) for provider in _PROVIDERS)


def _request_clear() -> None:
    st.session_state[_CLEAR_KEY] = True


def _clear_local_cache() -> None:
    job_keys = [_job_key(provider) for provider in _PROVIDERS]
    for key in list(st.session_state.keys()):
        rendered = str(key)
        if rendered.startswith(_PREFIX) or rendered == _CLEAR_KEY:
            st.session_state.pop(key, None)
    for job_key in job_keys:
        forget_background_job(job_key)
        st.session_state.pop(f"_background_job_delivered:{job_key}", None)


@st.dialog("Clear Copyright Challenge 2 cache", dismissible=False)
def _render_clear_dialog() -> None:
    st.warning(
        "This clears only this browser session's Challenge 2 settings and scaling results."
    )
    st.caption("API keys saved through the sidebar are not removed.")
    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        confirmed = st.button(
            "Confirm",
            type="primary",
            width="stretch",
            key="_copyright_game2_clear_confirm",
        )
    with cancel_col:
        cancelled = st.button(
            "Cancel",
            width="stretch",
            key="_copyright_game2_clear_cancel",
        )
    if cancelled:
        st.session_state.pop(_CLEAR_KEY, None)
        st.rerun()
    if confirmed:
        _clear_local_cache()
        st.rerun()

def _render_toolbar() -> None:
    st.button(
        "\U0001F5D1\uFE0F Clear Cache",
        key="_copyright_game2_clear_cache",
        help="Clear only this session's Challenge 2 settings and results.",
        disabled=_any_job_running(),
        on_click=_request_clear,
        width="stretch",
    )
    if st.session_state.get(_CLEAR_KEY):
        _render_clear_dialog()


def _render_hero() -> None:
    st.markdown(
        """
        <h4 class="section-header">\U0001F4C8 Game 2: The Cross-Model Scaling Quest</h4>
        <p class="copyright-game-tagline">
            Scale the same recall probe across GPT and Kimi to reveal how memorization changes between models.
        </p>
        """,
        unsafe_allow_html=True,
    )


def _render_stage_brief(prompt_preview: str) -> None:
    with st.expander("Prompt preview and ground truth", expanded=False):
        st.markdown("### Game 2: The Cross-Model Scaling Quest")
        st.markdown(
            "Scale one Content Recall direct-probe prompt independently across the "
            "locked OpenAI and Kimi models. Every response is scored against the "
            f"same fixed {TARGET_WORD_COUNT}-word reference."
        )

        st.markdown("**Prompt preview**")
        st.code(
            prompt_preview or "The selected prompt could not be generated.",
            language=None,
            wrap_lines=True,
        )

        book_col, task_col, model_col, score_col = st.columns(4)
        with book_col:
            st.markdown("**Book**")
            st.caption(BOOK_TITLE)
        with task_col:
            st.markdown("**Task**")
            st.caption("Direct Probing")
        with model_col:
            st.markdown("**Models**")
            st.caption("OpenAI + Kimi")
        with score_col:
            st.markdown("**Score**")
            st.caption("ROUGE-L")

        st.markdown("**Ground truth**")
        st.code(GROUND_TRUTH, language=None, wrap_lines=True)

def _history_key(provider: str) -> str:
    slug = "openai" if provider == OPENAI_PROVIDER else "kimi"
    return _key(f"{slug}_history")


def _active_round_key(provider: str) -> str:
    slug = "openai" if provider == OPENAI_PROVIDER else "kimi"
    return _key(f"{slug}_active_round")


def _round_history() -> List[Dict[str, Any]]:
    value = st.session_state.get(_key("round_history"), [])
    return list(value) if isinstance(value, list) else []


def _record_round_provider(
    round_id: str,
    provider: str,
    *,
    completed_at: str,
    result: Optional[ScalingBatch] = None,
    error: Optional[str] = None,
    skipped: bool = False,
) -> None:
    history = _round_history()
    for payload in history:
        if str(payload.get("id") or "") != round_id:
            continue
        providers = payload.setdefault("providers", {})
        if not isinstance(providers, dict):
            providers = {}
            payload["providers"] = providers
        providers[provider] = {
            "completed_at": completed_at,
            "result": result,
            "error": str(error or "").strip(),
            "skipped": bool(skipped),
        }
        st.session_state[_key("round_history")] = history
        return


def _harvest_completed_job(provider: str) -> None:
    job_key = _job_key(provider)
    state = get_background_job(job_key)
    if not isinstance(state, dict) or state.get("status") not in {"completed", "failed"}:
        return
    token = str(state.get("finished_at") or state.get("status") or "completed")
    delivered_key = _key(f"delivered:{job_key}")
    if st.session_state.get(delivered_key) == token:
        return

    completed_at = _format_kst(state.get("finished_at"))
    result = state.get("result")
    round_id = str(st.session_state.get(_active_round_key(provider)) or "")
    if round_id:
        _record_round_provider(
            round_id,
            provider,
            completed_at=completed_at,
            result=result if isinstance(result, ScalingBatch) else None,
            error=(
                str(state.get("error") or "")
                if state.get("status") == "failed"
                else None
            ),
        )
    elif isinstance(result, ScalingBatch):
        # Preserve results produced by the earlier, one-provider-at-a-time UI.
        history = list(st.session_state.get(_history_key(provider), []))
        history.append({"completed_at": completed_at, "result": result})
        st.session_state[_history_key(provider)] = history
    st.session_state[delivered_key] = token


def _render_rouge_distribution(result: ScalingBatch) -> None:
    scores = [attempt.rouge_l for attempt in result.attempts]
    if not scores:
        st.info(f"No scored {result.provider} responses are available for distribution analysis.")
        return
    try:
        figure = build_rouge_l_strategy_histogram(
            {f"{result.provider} - {result.model}": scores}
        )
        figure.set_size_inches(5.2, 3.0, forward=True)
        st.pyplot(figure, width="content")
        plt.close(figure)
        st.caption(
            f"{len(scores)} scored response{'s' if len(scores) != 1 else ''} - "
            f"range {min(scores):.4f} to {max(scores):.4f}"
        )
    except Exception as exc:
        _LOGGER.exception("Unable to render %s ROUGE-L distribution", result.provider)
        st.info(f"{result.provider} ROUGE-L distribution is unavailable: {exc}")


def _render_round_score_overview(providers: Dict[str, Any]) -> None:
    results: Dict[str, ScalingBatch] = {}
    for provider in _PROVIDERS:
        payload = providers.get(provider)
        if not isinstance(payload, dict):
            continue
        result = payload.get("result")
        if isinstance(result, ScalingBatch):
            results[provider] = result

    if not results:
        return

    st.markdown("**Similarity summary**")
    metric_columns = st.columns(4)
    for provider_index, provider in enumerate(_PROVIDERS):
        result = results.get(provider)
        maximum = f"{result.max_rouge_l:.4f}" if result else "-"
        average = f"{result.avg_rouge_l:.4f}" if result else "-"
        metric_columns[provider_index * 2].metric(
            f"{provider} maximum ROUGE-L",
            maximum,
        )
        metric_columns[provider_index * 2 + 1].metric(
            f"{provider} average ROUGE-L",
            average,
        )

    comparison_rows: List[Dict[str, Any]] = []
    for provider in _PROVIDERS:
        result = results.get(provider)
        if result is None:
            continue
        row: Dict[str, Any] = {
            "Provider": provider,
            "Model": result.model,
            "Runs": f"{result.completed_runs}/{result.requested_runs}",
            "Max ROUGE-L": result.max_rouge_l,
        }
        for metric_key, label in _PROVIDER_COMPARISON_METRICS:
            row[label] = _average_metric(result, metric_key)
        comparison_rows.append(row)
    if comparison_rows:
        st.dataframe(
            pd.DataFrame(comparison_rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Provider": st.column_config.TextColumn("Provider", width="small"),
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "Runs": st.column_config.TextColumn("Runs", width="small"),
                "Max ROUGE-L": st.column_config.NumberColumn("Max ROUGE-L", format="%.4f"),
                "Avg ROUGE-L": st.column_config.NumberColumn("Avg ROUGE-L", format="%.4f"),
                "Avg ROUGE-1": st.column_config.NumberColumn("Avg ROUGE-1", format="%.4f"),
                "Avg Jaccard": st.column_config.NumberColumn("Avg Jaccard", format="%.4f"),
                "Avg LCS Word": st.column_config.NumberColumn("Avg LCS Word", format="%.4f"),
                "Avg Levenshtein": st.column_config.NumberColumn("Avg Levenshtein", format="%.4f"),
            },
        )

    st.markdown("**ROUGE-L distributions**")
    chart_columns = st.columns(2, gap="large")
    for column, provider in zip(chart_columns, _PROVIDERS):
        with column:
            result = results.get(provider)
            st.markdown(f"**{provider}**")
            if result is not None:
                _render_rouge_distribution(result)
                continue
            payload = providers.get(provider)
            if isinstance(payload, dict) and payload.get("skipped"):
                st.info(f"{provider} was skipped for this batch.")
            elif isinstance(payload, dict) and payload.get("error"):
                st.info(f"No {provider} distribution is available because the run failed.")
            else:
                st.info(f"Waiting for {provider} scores.")

    st.markdown("---")


def _render_batch(
    result: ScalingBatch,
    completed_at: str,
    *,
    show_distribution: bool = True,
) -> None:
    best_position = max(
        range(len(result.attempts)),
        key=lambda index: result.attempts[index].rouge_l,
    )
    score_col, avg_col, jaccard_col, runs_col, sampling_col = st.columns(5)
    score_col.metric("Maximum ROUGE-L", f"{result.max_rouge_l:.4f}")
    avg_col.metric("Average ROUGE-L", f"{result.avg_rouge_l:.4f}")
    jaccard_col.metric(
        "Average Jaccard",
        _format_optional(_average_metric(result, "jaccard_index")),
    )
    runs_col.metric(
        "Completed runs",
        f"{result.completed_runs}/{result.requested_runs}",
    )
    sampling_col.metric(
        "Temperature / top-p",
        f"{result.temperature:.2f} / {result.top_p:.2f}",
    )
    st.caption(
        f"{result.provider} \u00B7 {result.model} \u00B7 {result.prompt_method} \u00B7 "
        f"{result.prompt_mode} \u00B7 {completed_at}"
    )
    _render_similarity_score_summary(
        [attempt.metrics for attempt in result.attempts],
        expanded=False,
    )
    if show_distribution:
        st.markdown("**ROUGE-L distribution**")
        _render_rouge_distribution(result)

    with st.expander("Exact prompt sent to the provider", expanded=False):
        st.code(result.prompt, language=None, wrap_lines=True)

    with st.expander(
        f"All {result.provider} responses ({result.completed_runs})",
        expanded=False,
    ):
        for position, attempt in enumerate(result.attempts):
            best_label = " \u00B7 Best" if position == best_position else ""
            metric_caption = _attempt_metric_caption(attempt.metrics)
            title_metrics = metric_caption or f"ROUGE-L {attempt.rouge_l:.4f}"
            with st.expander(
                f"Run {attempt.run} \u00B7 {title_metrics}{best_label}",
                expanded=False,
            ):
                render_direct_recall_diff(
                    ground_truth=GROUND_TRUTH,
                    generated_text=attempt.response,
                    title=f"{result.provider} Run {attempt.run}",
                    metrics=attempt.metrics,
                )
    if result.errors:
        st.warning(
            "The provider returned a usable partial batch, then stopped: "
            + " | ".join(result.errors)
        )


def _render_history() -> None:
    rounds = _round_history()
    if rounds:
        st.markdown("#### Scaling run history")
    for number, payload in enumerate(rounds, start=1):
        providers = payload.get("providers")
        providers = providers if isinstance(providers, dict) else {}
        started_at = str(payload.get("started_at") or "Timestamp unavailable")
        completed_count = sum(
            1
            for provider in _PROVIDERS
            if isinstance(providers.get(provider), dict)
        )
        with st.expander(
            f"Scaling Run {number} - OpenAI + Kimi - {started_at}",
            expanded=(number == len(rounds)),
        ):
            if completed_count < len(_PROVIDERS):
                st.info(
                    "This dual-model run is still in progress. Completed provider "
                    "results will appear here automatically."
                )
            _render_round_score_overview(providers)
            for position, provider in enumerate(_PROVIDERS):
                provider_payload = providers.get(provider)
                if position:
                    st.markdown("---")
                provider_icon = (
                    "\U0001F916" if provider == OPENAI_PROVIDER else "\U0001F319"
                )
                st.markdown(f"### {provider_icon} {provider}")
                if not isinstance(provider_payload, dict):
                    st.caption("Waiting for this provider to finish.")
                    continue
                error = str(provider_payload.get("error") or "").strip()
                skipped = bool(provider_payload.get("skipped", False))
                result = provider_payload.get("result")
                completed_at = str(
                    provider_payload.get("completed_at") or "Timestamp unavailable"
                )
                if skipped:
                    st.info(f"{provider} was skipped for this batch (0 runs).")
                elif error:
                    st.error(f"{provider} scaling failed: {error}")
                elif isinstance(result, ScalingBatch):
                    _render_batch(
                        result,
                        completed_at,
                        show_distribution=False,
                    )

    # Older browser-session results remain reviewable after this UI upgrade.
    for provider in _PROVIDERS:
        history = list(st.session_state.get(_history_key(provider), []))
        if not history:
            continue
        st.markdown(f"#### {provider} scaling history from earlier runs")
        for number, payload in enumerate(history, start=1):
            result = payload.get("result")
            if not isinstance(result, ScalingBatch):
                continue
            completed_at = str(payload.get("completed_at") or "Timestamp unavailable")
            with st.expander(
                f"{provider} Run {number} \u00B7 {result.model} \u00B7 "
                f"{result.completed_runs} responses \u00B7 {completed_at}",
                expanded=False,
            ):
                _render_batch(result, completed_at)

def _render_prompt_settings(disabled: bool) -> tuple[str, str, Optional[str], str]:
    st.markdown(
        """
        <div class="copyright-game-section-heading">
            <div>
                <strong>Choose the direct-probe prompt</strong>
                <span>The selected direct request is shared by both providers.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    method = st.selectbox(
        "Direct-probing prompt",
        DIRECT_PROBE_METHODS,
        format_func=build_challenge_prompt,
        key=_key("prompt_method"),
        disabled=disabled,
    )
    prompt_mode = "Direct Probing"
    custom_template: Optional[str] = None
    preview = build_challenge_prompt(method, prompt_mode, custom_template)

    return method, prompt_mode, custom_template, preview

def _render_provider_settings(
    provider: str,
    *,
    disabled: bool,
) -> tuple[str, float, float, int, str]:
    slug = "openai" if provider == OPENAI_PROVIDER else "kimi"
    model = FIXED_MODELS[provider]
    provider_icon = "\U0001F916" if provider == OPENAI_PROVIDER else "\U0001F319"
    model_help = ""
    if provider == KIMI_PROVIDER and model.startswith("kimi-k2"):
        model_help = (
            '<span title="Kimi K2 models normalize sampling to temperature 1.00 '
            'and top-p 0.95 at request time; Moonshot v1 models use your selected '
            'values." aria-label="Kimi sampling behavior" style="display:inline-flex;'
            'align-items:center;justify-content:center;width:1.15rem;height:1.15rem;'
            'margin-left:0.35rem;border:1px solid #8a94a6;border-radius:50%;color:#5d687b;'
            'font-size:0.75rem;font-weight:700;cursor:help;vertical-align:middle;">?</span>'
        )
    st.markdown(
        f"""
        <div class="copyright-game2-provider-heading">
            <strong>{provider_icon} {provider}</strong>
            <span>Model: <code>{model}</code>{model_help}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    runs_col, sampling_col = st.columns(
        [0.9, 2.25],
        gap="medium",
        vertical_alignment="top",
    )
    with runs_col:
        runs = int(
            st.number_input(
                "Scaling runs",
                min_value=0,
                max_value=MAX_RUNS_PER_PROVIDER,
                value=1,
                step=1,
                key=_key(f"{slug}_runs"),
                disabled=disabled,
                help=(
                    f"Run the same prompt from 0 to {MAX_RUNS_PER_PROVIDER} times. "
                    "Set 0 to skip this provider for the current batch."
                ),
            )
        )
    with sampling_col:
        temperature, top_p = render_fragmented_temperature_top_p(
            container_key=f"copyright-game2-{slug}-sampling",
            gap="medium",
            temp_session_key=_key(f"{slug}_temperature"),
            top_p_session_key=_key(f"{slug}_top_p"),
            default_temp=0.7,
            default_top_p=0.9,
            temp_range=(0.0, 2.0),
            top_p_range=(0.05, 1.0),
            temp_step=0.01,
            top_p_step=0.01,
            disabled=disabled,
            help_temp="Controls randomness, matching Content Recall Detection.",
            help_top_p="Controls nucleus sampling, matching Content Recall Detection.",
        )
    api_key = _provider_api_key(provider)
    if runs > 0 and not api_key:
        st.warning(f"Add a {provider} API key in the sidebar before running.")

    return model, temperature, top_p, runs, api_key

def _start_scaling(
    provider: str,
    *,
    round_id: str,
    api_key: str,
    model: str,
    runs: int,
    temperature: float,
    top_p: float,
    prompt_method: str,
    prompt_mode: str,
    custom_template: Optional[str],
) -> bool:
    job_key = _job_key(provider)
    clicked_at = datetime.now(_KST).strftime("%Y-%m-%d %H:%M:%S KST")

    def execute(report) -> ScalingBatch:
        def update(current: int, total: int) -> None:
            report(current, total, f"Generated responses: {current}/{total}")

        return run_provider_scaling(
            api_key,
            provider=provider,
            model=model,
            runs=runs,
            temperature=temperature,
            top_p=top_p,
            prompt_method=prompt_method,
            prompt_mode=prompt_mode,
            custom_template=custom_template,
            on_progress=update,
        )

    started = submit_background_job(
        job_key,
        f"Challenge 2 \u00B7 {provider} \u00B7 {clicked_at}",
        execute,
    )
    if started:
        st.session_state[_active_round_key(provider)] = round_id
    return started


def _render_provider_workspace(
    method: str,
    mode: str,
    custom_template: Optional[str],
    prompt_preview: str,
    prompt_ready: bool,
    disabled: bool,
    settings_container: Any,
) -> None:
    with settings_container:
        st.markdown(
            """
            <div class="copyright-game-section-heading copyright-game2-section-divider">
                <div>
                    <strong>Choose sampling and scaling</strong>
                    <span>OpenAI and Kimi use separate sampling values, API keys, and run counts.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        openai_col, kimi_col = st.columns(2, gap="large")
        with openai_col:
            openai_settings = _render_provider_settings(
                OPENAI_PROVIDER,
                disabled=disabled,
            )
        with kimi_col:
            kimi_settings = _render_provider_settings(
                KIMI_PROVIDER,
                disabled=disabled,
            )

    _render_stage_brief(prompt_preview)
    any_running = _any_job_running()
    provider_settings = {
        OPENAI_PROVIDER: openai_settings,
        KIMI_PROVIDER: kimi_settings,
    }
    active_providers = [
        provider
        for provider, settings in provider_settings.items()
        if settings[3] > 0
    ]
    all_keys_ready = all(
        provider_settings[provider][4] for provider in active_providers
    )
    clicked = render_run_button(
        "Copyright Challenge 2 - Inference Scaling",
        _key("run_both_providers"),
        "Run: Inference Scaling",
        type="primary",
        disabled=(
            any_running
            or not prompt_ready
            or not active_providers
            or not all_keys_ready
        ),
        help="Start both locked models together with the settings configured above.",
    )

    status_columns = st.columns(2, gap="large")
    for column, provider in zip(status_columns, _PROVIDERS):
        with column:
            render_background_job_status(
                _job_key(provider),
                completed_message=(
                    f"{provider} scaling completed. Every response is available below."
                ),
            )

    if clicked:
        round_id = uuid4().hex
        started_at = datetime.now(_KST).strftime("%Y-%m-%d %H:%M:%S KST")
        history = _round_history()
        history.append(
            {
                "id": round_id,
                "started_at": started_at,
                "providers": {},
            }
        )
        st.session_state[_key("round_history")] = history

        failed_to_start: List[str] = []
        for provider in _PROVIDERS:
            model, temperature, top_p, runs, api_key = provider_settings[provider]
            if runs == 0:
                forget_background_job(_job_key(provider))
                st.session_state.pop(_active_round_key(provider), None)
                st.session_state.pop(
                    f"_background_job_delivered:{_job_key(provider)}",
                    None,
                )
                _record_round_provider(
                    round_id,
                    provider,
                    completed_at=started_at,
                    skipped=True,
                )
                continue
            started = _start_scaling(
                provider,
                round_id=round_id,
                api_key=api_key,
                model=model,
                runs=runs,
                temperature=temperature,
                top_p=top_p,
                prompt_method=method,
                prompt_mode=mode,
                custom_template=custom_template,
            )
            if not started:
                failed_to_start.append(provider)
                _record_round_provider(
                    round_id,
                    provider,
                    completed_at=started_at,
                    error="A background scaling job is already running.",
                )
        if failed_to_start:
            st.error("Could not start: " + ", ".join(failed_to_start) + ".")
        finish_detection_job()
        st.rerun()

def _render_unexpected_error() -> None:
    st.error(
        "Challenge 2 encountered an unexpected problem, but the rest of the "
        "application is still available."
    )
    st.caption("Use Clear Cache above if the problem repeats, then retry.")
    if st.button("Retry Challenge 2", key="_copyright_game2_retry"):
        st.rerun()


def render_copyright_game2_page() -> None:
    """Render the single-stage, no-leaderboard direct-probing duel."""
    _load_styles()
    header_col, toolbar_col = st.columns([4, 1], vertical_alignment="top")
    with header_col:
        _render_hero()
    with toolbar_col:
        _render_toolbar()

    if not is_logged_in():
        st.markdown(
            """
            <div class="copyright-game-card">
                <div class="copyright-game-card-label">Identity checkpoint</div>
                <h3 class="copyright-game-card-title">Sign in to enter Game 2</h3>
                <div class="copyright-game-card-copy">
                    GitHub login is required before using The Cross-Model Scaling Quest.
                    Your email and API credentials are never published.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("Use **Sign in with GitHub** in the sidebar to continue.")
        return

    for provider in _PROVIDERS:
        _harvest_completed_job(provider)

    try:
        running = _any_job_running()
        settings_container = st.container(
            border=True,
            key="copyright-game2-settings-card",
            gap="medium",
        )
        with settings_container:
            method, mode, custom_template, preview = _render_prompt_settings(running)
        _render_provider_workspace(
            method,
            mode,
            custom_template,
            preview,
            bool(preview),
            running,
            settings_container,
        )
        _render_history()
    except Exception:
        _LOGGER.exception("Unexpected Challenge 2 page error")
        _render_unexpected_error()


__all__ = ["render_copyright_game2_page"]
