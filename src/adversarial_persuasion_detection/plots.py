"""ROUGE-L distribution plots for persuasive jailbreak detection (exp1 boxplot, exp2 histogram+KDE)."""

from __future__ import annotations

import colorsys
import io
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from src.direct_recall.comparison import calculate_rouge_score

BASELINE_STRATEGY_LABEL = "Baseline (no strategy)"

# Shared figure/font config (aligned with Copyright-Detective-Exp demo exp1 & exp2)
FIG_SIZE = (5.5, 5.5)
MARGINS = {"left": 0.14, "right": 0.96, "top": 0.95, "bottom": 0.16}
BASE_STYLE = {
    "font.family": ["DejaVu Sans", "sans-serif"],
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}

plt.rcParams.update(BASE_STYLE)

# Base hues per strategy (well-separated; same strategy uses shades in mutation mode)
STRATEGY_PALETTE: Dict[str, str] = {
    # Core exp2 demo colors.
    "Baseline": "#015493",
    "Pathos": "#019092",
    "Alliance Building": "#999999",
    "Reciprocity": "#F4A99B",
    # Remaining persuasion strategies: one stable distinct color each.
    "Ethos": "#00838F",
    "Logos": "#B8860B",
    "Storytelling": "#AD1457",
    "Negotiation": "#5D4037",
    "Relationship Leverages": "#3949AB",
    "Loyalty Appeals": "#6A1B9A",
    "Affirmation": "#2E7D32",
    "Encouragement": "#F9A825",
    "Positive Motivation": "#EF6C00",
    "Negative Motivation": "#C62828",
    "Safety Needs": "#00695C",
    "Social Needs": "#5E35B1",
    "Self-Esteem Needs": "#8E24AA",
    "Foot-in-the-Door": "#455A64",
    "Door-in-the-Face": "#D84315",
    "Time Pressure": "#E65100",
    "Cognitive Dissonance": "#546E7A",
    "Priming": "#00897B",
    "Confirmation Bias": "#7B1FA2",
}

_EXP2_STRATEGY_ORDER = ("Baseline", "Pathos", "Alliance Building", "Reciprocity")
# Fallback only for unexpected/custom strategy names not listed above.
_DISTINCT_BASE_COLORS: Tuple[str, ...] = (
    "#1565A8",
    "#D84315",
    "#6B4C9A",
    "#1B7A4B",
    "#B8860B",
    "#00838F",
    "#AD1457",
    "#5D4037",
    "#3949AB",
    "#C62828",
)

_BIN_WIDTH = 0.05
_TICK_STEP = 0.1
_REFERENCE_GRID_POINTS = 400


@dataclass(frozen=True)
class MutationFootnote:
    """Maps a Mutation # (results panel index) to its persuasion strategy."""

    mutation_num: int
    strategy: str
    strategy_attempt: int


@dataclass
class DistributionPlotData:
    """Ordered ROUGE-L series for distribution charts."""

    groups: Dict[str, List[float]] = field(default_factory=dict)
    mutation_footnotes: Optional[List[MutationFootnote]] = None
    group_by_mutation: bool = False

    def has_data(self) -> bool:
        return any(scores for scores in self.groups.values())


def _display_strategy_label(strategy: str) -> str:
    if strategy == BASELINE_STRATEGY_LABEL:
        return "Baseline"
    return strategy.strip()


def _parse_rouge_l(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        try:
            score = float(str(value).strip())
        except (TypeError, ValueError):
            return None
    if math.isnan(score) or math.isinf(score):
        return None
    return max(0.0, min(1.0, score))


def _score_from_item(reference_text: str, llm_response: str, metrics: Any, rouge_fallback: Any) -> Optional[float]:
    llm_response = (llm_response or "").strip()
    if reference_text and llm_response:
        return calculate_rouge_score(reference_text, llm_response)
    if metrics is not None:
        return _parse_rouge_l(getattr(metrics, "rouge_l", None))
    return _parse_rouge_l(rouge_fallback)


def _order_strategy_groups(groups: Mapping[str, List[float]]) -> Dict[str, List[float]]:
    ordered: Dict[str, List[float]] = {}
    for name in _EXP2_STRATEGY_ORDER:
        if name in groups and groups[name]:
            ordered[name] = list(groups[name])
    for name in sorted(groups.keys()):
        if name not in ordered and groups[name]:
            ordered[name] = list(groups[name])
    return ordered


def _order_mutation_groups(groups: Mapping[str, List[float]]) -> Dict[str, List[float]]:
    def sort_key(label: str) -> int:
        if label.startswith("Mutation #"):
            try:
                return int(label.replace("Mutation #", "").strip())
            except ValueError:
                return 10**9
        return 10**9

    return {label: list(groups[label]) for label in sorted(groups.keys(), key=sort_key) if groups[label]}


def format_mutation_footnote_lines(footnotes: Sequence[MutationFootnote]) -> List[str]:
    return [
        f"Mutation #{item.mutation_num}: {item.strategy}"
        for item in footnotes
    ]


def collect_rouge_l_by_strategy(
    stored_panels: Sequence[Dict[str, Any]],
    reference_text: str,
    *,
    ranked_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, List[float]]:
    """Collect per-strategy ROUGE-L scores (legacy dict API)."""
    data = collect_distribution_plot_data(
        stored_panels,
        reference_text,
        ranked_rows=ranked_rows,
        attempts_per_strategy=1,
    )
    return data.groups if data else {}


def collect_rouge_l_by_mutation(
    stored_panels: Sequence[Dict[str, Any]],
    reference_text: str,
) -> DistributionPlotData:
    """
    Collect ROUGE-L per Mutation # (same numbering as results expanders).

    Each stored panel is one mutation; prompt-attempt scores form that mutation's distribution.
    """
    reference_text = (reference_text or "").strip()
    groups: Dict[str, List[float]] = {}
    footnotes: List[MutationFootnote] = []

    for idx, panel in enumerate(stored_panels, start=1):
        evaluation = panel.get("evaluation")
        if evaluation is None:
            continue
        strategy = getattr(getattr(evaluation, "mutation", None), "strategy", "") or ""
        strategy_attempt = int(getattr(evaluation, "attempt", 1) or 1)
        label = f"Mutation #{idx}"
        scores: List[float] = []

        for item in panel.get("group_items") or []:
            score = _score_from_item(
                reference_text,
                item.get("llm_response", ""),
                item.get("metrics"),
                None,
            )
            if score is not None:
                scores.append(score)

        if scores:
            groups[label] = scores
            footnotes.append(
                MutationFootnote(
                    mutation_num=idx,
                    strategy=_display_strategy_label(strategy),
                    strategy_attempt=strategy_attempt,
                )
            )

    return DistributionPlotData(
        groups=_order_mutation_groups(groups),
        mutation_footnotes=footnotes,
        group_by_mutation=True,
    )


def collect_distribution_plot_data(
    stored_panels: Sequence[Dict[str, Any]],
    reference_text: str,
    *,
    ranked_rows: Optional[Sequence[Dict[str, Any]]] = None,
    attempts_per_strategy: int = 1,
) -> Optional[DistributionPlotData]:
    """
    Build plot series grouped by strategy, or by Mutation # when attempts_per_strategy > 1.
    """
    if attempts_per_strategy > 1:
        data = collect_rouge_l_by_mutation(stored_panels, reference_text)
        return data if data.has_data() else None

    reference_text = (reference_text or "").strip()
    groups: Dict[str, List[float]] = {}

    if ranked_rows:
        for row in ranked_rows:
            strategy = row.get("strategy") or ""
            score = _score_from_item(
                reference_text,
                row.get("llm_response", ""),
                None,
                row.get("rouge_l"),
            )
            if score is not None:
                label = _display_strategy_label(strategy)
                groups.setdefault(label, []).append(score)
    else:
        for panel in stored_panels:
            evaluation = panel.get("evaluation")
            strategy = getattr(getattr(evaluation, "mutation", None), "strategy", None) or ""
            for item in panel.get("group_items") or []:
                score = _score_from_item(
                    reference_text,
                    item.get("llm_response", ""),
                    item.get("metrics"),
                    None,
                )
                if score is not None:
                    label = _display_strategy_label(strategy)
                    groups.setdefault(label, []).append(score)

    ordered = _order_strategy_groups(groups)
    if not ordered:
        return None
    return DistributionPlotData(groups=ordered, group_by_mutation=False)


def _resolve_groups(plot_data: DistributionPlotData | Mapping[str, Sequence[float]]) -> DistributionPlotData:
    if isinstance(plot_data, DistributionPlotData):
        if plot_data.group_by_mutation:
            return DistributionPlotData(
                groups=_order_mutation_groups(plot_data.groups),
                mutation_footnotes=plot_data.mutation_footnotes,
                group_by_mutation=True,
            )
        return DistributionPlotData(
            groups=_order_strategy_groups(plot_data.groups),
            group_by_mutation=False,
        )
    return DistributionPlotData(groups=_order_strategy_groups(plot_data), group_by_mutation=False)


def _hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16) / 255.0,
        int(hex_color[2:4], 16) / 255.0,
        int(hex_color[4:6], 16) / 255.0,
    )


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        int(max(0, min(255, round(r * 255)))),
        int(max(0, min(255, round(g * 255)))),
        int(max(0, min(255, round(b * 255)))),
    )


def _strategy_base_color(strategy: str, assigned: Dict[str, str]) -> str:
    if strategy in STRATEGY_PALETTE:
        return STRATEGY_PALETTE[strategy]
    if strategy in assigned:
        return assigned[strategy]
    idx = len(assigned) % len(_DISTINCT_BASE_COLORS)
    color = _DISTINCT_BASE_COLORS[idx]
    assigned[strategy] = color
    return color


def _shade_for_attempt(base_hex: str, attempt: int, total_attempts: int) -> str:
    """Same hue family: darker for early attempts, lighter for later ones."""
    r, g, b = _hex_to_rgb(base_hex)
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    saturation = min(0.92, max(0.42, saturation))

    if total_attempts <= 1:
        target_l = max(0.30, min(0.55, lightness))
    else:
        t = (attempt - 1) / (total_attempts - 1)
        target_l = 0.30 + t * 0.38

    r2, g2, b2 = colorsys.hls_to_rgb(hue, target_l, saturation)
    return _rgb_to_hex(r2, g2, b2)


def _build_label_color_map(resolved: DistributionPlotData) -> Dict[str, str]:
    labels = list(resolved.groups.keys())
    color_map: Dict[str, str] = {}
    assigned_bases: Dict[str, str] = {}

    if resolved.group_by_mutation and resolved.mutation_footnotes:
        max_attempt_by_strategy: Dict[str, int] = {}
        label_to_meta: Dict[str, MutationFootnote] = {}
        for footnote in resolved.mutation_footnotes:
            label = f"Mutation #{footnote.mutation_num}"
            label_to_meta[label] = footnote
            prev = max_attempt_by_strategy.get(footnote.strategy, 0)
            max_attempt_by_strategy[footnote.strategy] = max(prev, footnote.strategy_attempt)

        for label in labels:
            meta = label_to_meta.get(label)
            if meta is None:
                color_map[label] = _DISTINCT_BASE_COLORS[len(color_map) % len(_DISTINCT_BASE_COLORS)]
                continue
            base = _strategy_base_color(meta.strategy, assigned_bases)
            total = max_attempt_by_strategy.get(meta.strategy, meta.strategy_attempt)
            color_map[label] = _shade_for_attempt(base, meta.strategy_attempt, total)
    else:
        for label in labels:
            color_map[label] = _strategy_base_color(label, assigned_bases)

    return color_map


def _histogram_bins(*series_list: pd.Series) -> List[float]:
    combined_max = 0.0
    for series in series_list:
        if len(series) > 0:
            combined_max = max(combined_max, float(series.max()))
    max_edge = max(_BIN_WIDTH, math.ceil(combined_max / _BIN_WIDTH) * _BIN_WIDTH)
    return [i * _BIN_WIDTH for i in range(int(max_edge / _BIN_WIDTH) + 2)]


def _x_ticks_and_grid(max_edge: float) -> tuple[List[float], np.ndarray]:
    max_tick = max(_TICK_STEP, math.ceil(max_edge / _TICK_STEP) * _TICK_STEP)
    x_ticks = [round(i * _TICK_STEP, 10) for i in range(int(max_tick / _TICK_STEP) + 1)]
    x_grid = np.linspace(0, max_tick, _REFERENCE_GRID_POINTS)
    return x_ticks, x_grid


def build_rouge_l_strategy_histogram(
    plot_data: DistributionPlotData | Mapping[str, Sequence[float]],
) -> plt.Figure:
    """Overlaid ROUGE-L histograms with KDE curves (exp2 style)."""
    resolved = _resolve_groups(plot_data)
    if not resolved.groups:
        raise ValueError("plot_data must not be empty")

    ordered = resolved.groups
    series_list = [pd.Series(scores, dtype=float).dropna() for scores in ordered.values()]
    if not any(len(s) > 0 for s in series_list):
        raise ValueError("No valid ROUGE-L scores in plot_data")

    bins = _histogram_bins(*series_list)
    max_edge = bins[-1] if bins else _BIN_WIDTH
    x_ticks, x_grid = _x_ticks_and_grid(max_edge)
    x_grid_reshaped = x_grid[:, None]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.subplots_adjust(**MARGINS)
    color_map = _build_label_color_map(resolved)

    def add_kde_curve(series: pd.Series, color: str) -> None:
        values = series.to_numpy().reshape(-1, 1)
        std = float(series.std(ddof=0)) or 0.0
        bandwidth = max(0.01, std * 0.3, _BIN_WIDTH / 2)
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(values)
        densities = np.exp(kde.score_samples(x_grid_reshaped)) * _BIN_WIDTH
        ax.plot(x_grid, densities, color=color, linewidth=1.5)

    for label, scores in ordered.items():
        series = pd.Series(scores, dtype=float).dropna()
        if len(series) == 0:
            continue
        color = color_map[label]
        frequency_weight = 1.0 / len(series)
        ax.hist(
            series,
            bins=bins,
            alpha=0.5,
            label=label,
            color=color,
            weights=[frequency_weight] * len(series),
        )
        add_kde_curve(series, color)

    ax.set_xlabel("ROUGE-L")
    ax.set_ylabel("Frequency")
    ax.set_xticks(x_ticks)
    for freq in (0.2, 0.4, 0.6, 0.8):
        ax.axhline(y=freq, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)

    ncol = 1 if len(ordered) <= 6 else 2
    ax.legend(loc="upper right", ncol=ncol, fontsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(
        axis="both",
        labelsize=BASE_STYLE["xtick.labelsize"],
        colors="#333333",
        direction="out",
        length=3,
        width=0.6,
        pad=6,
    )

    return fig


def build_rouge_l_distribution_boxplot(
    plot_data: DistributionPlotData | Mapping[str, Sequence[float]],
) -> plt.Figure:
    """Vertical ROUGE-L boxplots (exp1 styling)."""
    resolved = _resolve_groups(plot_data)
    if not resolved.groups:
        raise ValueError("plot_data must not be empty")

    labels = list(resolved.groups.keys())
    data = [resolved.groups[label] for label in labels]
    color_map = _build_label_color_map(resolved)
    colors = [color_map[label] for label in labels]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.subplots_adjust(**MARGINS)

    box = ax.boxplot(
        data,
        tick_labels=labels,
        vert=True,
        patch_artist=True,
        widths=0.5,
        boxprops={"linewidth": 1.2},
        medianprops={"color": "#404040", "linewidth": 1.8, "solid_capstyle": "round"},
        whiskerprops={"color": "#606060", "linewidth": 1.1, "solid_capstyle": "round"},
        capprops={"color": "#606060", "linewidth": 1.1},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.7, "markeredgewidth": 0.8},
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#505050")
        patch.set_linewidth(1.2)
        patch.set_alpha(0.9)

    for flier, color in zip(box["fliers"], colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor("#505050")

    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle="-", alpha=0.4, color="#BBBBBB", linewidth=0.6, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#404040")

    ax.set_ylabel("ROUGE-L")
    x_label = "Mutation" if resolved.group_by_mutation else "Persuasion Strategy"
    ax.set_xlabel(x_label)
    rotation = 0 if resolved.group_by_mutation and len(labels) <= 8 else 15
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="center", va="top", rotation_mode="anchor")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(
        axis="both",
        labelsize=BASE_STYLE["xtick.labelsize"],
        colors="#333333",
        direction="out",
        length=3,
        width=0.6,
        pad=6,
    )

    return fig


def figure_to_png_bytes(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, facecolor="white", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()
