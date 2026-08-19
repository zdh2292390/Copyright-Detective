"""Reusable sampling control helpers (temperature, top-p) for Streamlit pages."""

from typing import Optional, Tuple

import streamlit as st
from src.job_guard import get_ui_disabled


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
    temp_container = col_temp or st
    top_p_container = col_top_p or st

    temp_key = f"{slider_key_prefix}{temp_session_key}_slider"
    top_p_key = f"{slider_key_prefix}{top_p_session_key}_slider"

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