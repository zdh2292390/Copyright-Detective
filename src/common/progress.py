"""Utility helpers for displaying progress of LLM calls in Streamlit."""

from __future__ import annotations

import time
from typing import Any, Optional, Tuple

try:
    import streamlit as st  # type: ignore[import]
except ImportError:  # pragma: no cover - Streamlit should be available in the app runtime
    st = None  # type: ignore

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional runtime
    get_script_run_ctx = None  # type: ignore[assignment]


ProgressArtifacts = Tuple[Optional[Any], Optional[Any], Optional[Any]]


def _has_streamlit_script_context() -> bool:
    """Return whether UI commands can safely target the current app run."""
    if st is None or get_script_run_ctx is None:
        return False
    try:
        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:  # pragma: no cover - tolerate Streamlit internal API changes
        return False


def start_llm_progress(message: str) -> ProgressArtifacts:
    """Render a progress bar with an accompanying label if Streamlit is available."""
    if not _has_streamlit_script_context():
        return None, None, None

    try:
        label_placeholder = st.empty()
        bar_placeholder = st.empty()
        label_placeholder.markdown(f"**{message}**")
        progress_bar = bar_placeholder.progress(0)
        return label_placeholder, bar_placeholder, progress_bar
    except Exception:  # pragma: no cover - defensive against runtime rendering issues
        return None, None, None


def update_llm_progress(progress_bar: Optional[Any], *, value: int) -> None:
    """Advance the progress bar if it was created successfully."""
    if progress_bar is None:
        return
    try:
        bounded_value = max(0, min(100, value))
        progress_bar.progress(bounded_value)
    except Exception:  # pragma: no cover - ignore rendering failures
        pass


def complete_llm_progress(
    label_placeholder: Optional[Any],
    bar_placeholder: Optional[Any],
    progress_bar: Optional[Any],
    *,
    final_message: str,
    success: bool,
    linger: float = 0.35,
) -> None:
    """Finalize the progress display, briefly showing completion state before clearing it."""
    icon = "✅" if success else "❌"

    if progress_bar is not None:
        try:
            progress_bar.progress(100)
        except Exception:  # pragma: no cover
            pass

    if label_placeholder is not None:
        try:
            label_placeholder.markdown(f"{icon} {final_message}")
        except Exception:  # pragma: no cover
            pass

    if (label_placeholder is not None or bar_placeholder is not None) and linger > 0:
        time.sleep(linger)

    if label_placeholder is not None:
        try:
            label_placeholder.empty()
        except Exception:  # pragma: no cover
            pass
    if bar_placeholder is not None:
        try:
            bar_placeholder.empty()
        except Exception:  # pragma: no cover
            pass
