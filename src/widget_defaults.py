"""Use widget defaults only until a keyed SessionState value is available."""

import streamlit as st


def widget_defaults(key: str, **defaults) -> dict:
    """Preserve restored widget state without duplicate default warnings."""
    return {} if key in st.session_state else defaults
