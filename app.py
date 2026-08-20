# -*- coding: utf-8 -*-
import streamlit as st
import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
# Get the directory where app.py is located
project_root = Path(__file__).parent.absolute()
project_root_str = str(project_root)
logger = logging.getLogger(__name__)

# Ensure project root is in sys.path (at the beginning)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
elif sys.path[0] != project_root_str:
    # Move to front if already in path
    sys.path.remove(project_root_str)
    sys.path.insert(0, project_root_str)

# Ensure we're in the right directory
os.chdir(project_root_str)

from src.auth import init_auth
from src.job_guard import (
    finish_detection_job,
    handle_interrupted_job,
    install_widget_guards,
    is_detection_job_running,
    render_active_job_lock,
)
from src.floating_clear_cache import handle_clear_cache_query_param
from src.ui import (
    GAMES_ENABLED,
    render_header,
    render_sidebar,
    render_snippet_to_document_page,
    render_knowledge_memorization_page,
    render_adversarial_persuasion_page,
    render_unlearning_detection_page,
    render_legal_case_display_page,
)

# Configure Streamlit page - sidebar will be expandable/collapsible via native toggle
st.set_page_config(
    page_title="Copyright Detective",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",  # Sidebar starts expanded; users can toggle via button
)

install_widget_guards()

init_auth()

# Handle draggable Clear Cache button clicks before rendering the page
if handle_clear_cache_query_param():
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
    else:
        experimental_rerun = getattr(st, "experimental_rerun", None)
        if callable(experimental_rerun):
            experimental_rerun()

st.markdown(
    """
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    """,
    unsafe_allow_html=True
)

# Load custom CSS
with open("assets/styles.css", "r") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Keep CSS-only approach; removed JS ligature fallback to avoid side effects

# Render header
render_header()

# Recover from accidental reruns that interrupted a long-running job
handle_interrupted_job()

# Lock controls before sidebar widgets render. The page itself remains scrollable.
render_active_job_lock()

# Render sidebar and get configuration
api_key, model_choice, provider, page = render_sidebar()

# Render main content behind a final safety boundary. Streamlit rerun/stop signals
# inherit from BaseException, so this catches real page failures without blocking navigation.
try:
    if page == "Content Recall Detection":
        render_snippet_to_document_page(api_key, model_choice, provider)
    elif page == "Knowledge Memorization Detection":
        render_knowledge_memorization_page(api_key, model_choice, provider)
    elif page == "Persuasive Jailbreak Detection":
        render_adversarial_persuasion_page(api_key, model_choice, provider)
    elif page == "Unlearning Detection":
        render_unlearning_detection_page(api_key, model_choice, provider)
    elif page == "Legal Cases Display":
        render_legal_case_display_page()
    elif GAMES_ENABLED and page in {
        "Game 1: The Hidden Passage Hunt",
        "Game 2: The Hidden Passage Hunt",
        "Copyright Challenge",
        "Copyright Challenge 1",
    }:
        from src.pages.copyright_game import render_copyright_game_page

        render_copyright_game_page()
    elif GAMES_ENABLED and page in {
        "Game 2: The Cross-Model Scaling Quest",
        "Game 1: The Cross-Model Scaling Quest",
        "Game 2: The Twin Oracle Duel",
        "Game 2: The Two-Model Continuation Duel",
        "Copyright Challenge 2",
    }:
        from src.pages.copyright_game2 import render_copyright_game2_page

        render_copyright_game2_page()
    elif GAMES_ENABLED and page in {
        "Game 3: The Memory Vault Hunt",
        "Copyright Challenge 3",
        "Game 3: The Knowledge Memorization Challenge",
    }:
        from src.pages.copyright_game3 import render_copyright_game3_page

        render_copyright_game3_page(api_key, model_choice)
except Exception:
    logger.exception("Unhandled page error on %s", page)
    st.error(
        "This page encountered an unexpected problem, but the application is still running. "
        "Your saved competition scores and account data were not cleared."
    )
    st.caption("Retry once. If it happens again, use the page's Clear cache button.")
    if st.button("Retry page", key="_global_page_retry"):
        st.rerun()
finally:
    # Always release the run lock when this script finishes (success or error).
    if is_detection_job_running():
        finish_detection_job(show_clear_cache=False)

# Footer (commented out as it's not defined)
# render_footer()
