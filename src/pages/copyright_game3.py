"""Game 3 free-exploration workspace backed by the shared GPT API."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.auth import is_logged_in
from src.game.storage import get_shared_api_key
from src.game2.constants import GAME_PROVIDER

_GAME_CSS = Path(__file__).resolve().parents[2] / "assets" / "game.css"


def _load_styles() -> None:
    if not _GAME_CSS.exists():
        return
    st.markdown(
        f"<style>{_GAME_CSS.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def render_copyright_game3_page(model_choice: str = "gpt-4o-mini") -> None:
    """Reuse Knowledge Memorization Detection without competition or ranking."""

    if not is_logged_in():
        _load_styles()
        st.markdown(
            """
            <h4 class="section-header">\U0001F9ED Game 3: The Memory Vault Hunt</h4>
            <p class="copyright-game-tagline">
                Probe whether a model has memorized specific source material.
            </p>
            <div class="copyright-game-card">
                <div class="copyright-game-card-label">Identity checkpoint</div>
                <h3 class="copyright-game-card-title">Sign in to enter Game 3</h3>
                <div class="copyright-game-card-copy">
                    GitHub login is required before using The Memory Vault Hunt.
                    Your email and API credentials are never published.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("Use **Sign in with GitHub** in the sidebar to continue.")
        return

    api_key = get_shared_api_key()
    if not api_key:
        st.error("Game 3 is unavailable because the shared GPT API key is not configured.")
        st.caption("Configure `COPYRIGHT_GAME_OPENAI_API_KEY` to enable this workspace.")
        return

    # Imported lazily to avoid a module cycle while app.py initializes page modules.
    from src.ui import render_knowledge_memorization_page

    render_knowledge_memorization_page(
        api_key,
        "gpt-4o-mini",
        GAME_PROVIDER,
        show_page_header=True,
        page_title="\U0001F9ED Game 3: The Memory Vault Hunt",
        page_description=(
            "Probe whether a model has memorized specific source material. "
            "Ask open-ended questions and score the answers, or use single-choice items "
            "that mix a verbatim passage with close paraphrases and see which option the model prefers."
        ),
    )