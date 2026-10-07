"""Reusable sampling control helpers (temperature, top-p) for Streamlit pages."""

from typing import Optional, Tuple
import math
from contextlib import nullcontext

import streamlit as st
from src.job_guard import get_ui_disabled


def _sampling_value(value, default, limits):
    try:
        number = float(value)
        if isinstance(value, bool) or not math.isfinite(number) or not limits[0] <= number <= limits[1]:
            raise ValueError("Invalid cached sampling value")
        return number
    except (TypeError, ValueError, OverflowError):
        return float(default)


def render_temperature_top_p(
    *,
    temp_session_key: str,
    top_p_session_key: str,
    default_temp: float = 0.7,
    default_top_p: float = 0.9,
    temp_label: str = "Temperature",
    top_p_label: str = "Top-p",
    temp_range: Tuple[float, float] = (0.0, 1.2),
    top_p_range: Tuple[float, float] = (0.0, 1.0),
    temp_step: float = 0.01,
    top_p_step: float = 0.01,
    help_temp: Optional[str] = None,
    help_top_p: Optional[str] = None,
    slider_key_prefix: str = "",
    col_temp=None,
    col_top_p=None,
    disabled: Optional[bool] = None,
) -> Tuple[float, float]:
    """
    Render temperature and top-p sliders with shared styling and state handling.

    Returns:
        (temperature, top_p)
    """
    if disabled is None:
        disabled = False
    st.session_state.setdefault(temp_session_key, default_temp)
    st.session_state.setdefault(top_p_session_key, default_top_p)

    # Allow caller to supply columns; fall back to page root.
    temp_container = col_temp if col_temp is not None else nullcontext()
    top_p_container = col_top_p if col_top_p is not None else nullcontext()

    temp_key = f"{slider_key_prefix}{temp_session_key}_slider"
    top_p_key = f"{slider_key_prefix}{top_p_session_key}_slider"

    st.session_state[temp_session_key] = _sampling_value(
        st.session_state[temp_session_key], default_temp, temp_range,
    )
    st.session_state[top_p_session_key] = _sampling_value(
        st.session_state[top_p_session_key], default_top_p, top_p_range,
    )
    for widget_key, default, limits in ((temp_key, default_temp, temp_range), (top_p_key, default_top_p, top_p_range)):
        if widget_key in st.session_state:
            value = st.session_state[widget_key]
            recovered = _sampling_value(value, default, limits)
            if not isinstance(value, float) or not math.isfinite(value) or value != recovered:
                st.session_state[widget_key] = recovered

    with temp_container:
        temperature = st.slider(
            temp_label,
            min_value=float(temp_range[0]),
            max_value=float(temp_range[1]),
            value=float(st.session_state[temp_session_key]),
            step=float(temp_step),
            help=help_temp,
            key=temp_key,
            disabled=disabled,
        )
    st.session_state[temp_session_key] = temperature

    with top_p_container:
        top_p = st.slider(
            top_p_label,
            min_value=float(top_p_range[0]),
            max_value=float(top_p_range[1]),
            value=float(st.session_state[top_p_session_key]),
            step=float(top_p_step),
            help=help_top_p,
            key=top_p_key,
            disabled=disabled,
        )
    st.session_state[top_p_session_key] = top_p

    return temperature, top_p


@st.fragment
def render_fragmented_temperature_top_p(
    *,
    container_key: Optional[str] = None,
    gap: str = "large",
    **kwargs,
) -> Tuple[float, float]:
    """Render sampling sliders in an isolated rerun scope.

    Widget changes update the canonical session keys through
    ``render_temperature_top_p`` without rerunning authentication, remote data
    reads, leaderboards, or the rest of the page. Run buttons intentionally stay
    outside this fragment so the global job guard still owns execution reruns.
    """
    with st.container(key=container_key):
        col_temp, col_top_p = st.columns(2, gap=gap)
        return render_temperature_top_p(
            col_temp=col_temp,
            col_top_p=col_top_p,
            **kwargs,
        )