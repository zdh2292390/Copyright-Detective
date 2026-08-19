"""Frozen benchmark and run limits for Copyright Challenge 3."""

from __future__ import annotations

from typing import Dict, List

from src.direct_recall.knowledge_benchmarks import (
    DEFAULT_KNOWLEDGE_BOOK_KEY,
    KNOWLEDGE_BOOK_KEYS,
    get_knowledge_book_title,
    get_knowledge_question_bank,
    list_knowledge_books,
)


COMPETITION_SLUG = "knowledge-memorization-gpt-4o-mini-v2"
COMPETITION_TITLE = "The Knowledge Memorization Challenge"
LEADERBOARD_VIEW = "copyright_challenge3_leaderboard"
LEGACY_LEADERBOARD_VIEW = "copyright_game_leaderboard"
BENCHMARK_VERSION = "five-books-knowledge-qa-v2"

GAME_PROVIDER = "OpenAI"
GAME_MODEL = "gpt-4o-mini"

DEFAULT_BOOK_KEY = DEFAULT_KNOWLEDGE_BOOK_KEY
BOOK_KEYS = KNOWLEDGE_BOOK_KEYS
QUESTIONS_PER_BOOK = 5

STANDARD_MODE = "Standard"
SLEEK_MODE = "Step-by-step Leaking and Extraction"
PROBE_MODES = (STANDARD_MODE, SLEEK_MODE)

MIN_RUNS = 1
STANDARD_MAX_RUNS = 500
SLEEK_MAX_RUNS = 500
MAX_SCORED_ANSWERS = QUESTIONS_PER_BOOK * STANDARD_MAX_RUNS
GENERATION_LIMIT_EXCLUSIVE = MAX_SCORED_ANSWERS + 1

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.2
MIN_TOP_P = 0.0
MAX_TOP_P = 1.0
SAMPLING_STEP = 0.05

# Backward-compatible shape for callers that used the legacy constant.
# Every value is derived from the shared Knowledge Memorization benchmark.
QUESTION_BANK: Dict[str, List[Dict[str, str]]] = {
    book_key: get_knowledge_question_bank(book_key) for book_key in BOOK_KEYS
}


def list_game_books() -> List[str]:
    return list_knowledge_books()


def get_book_title(book_key: str) -> str:
    return get_knowledge_book_title(book_key)


def get_question_bank(book_key: str) -> List[Dict[str, str]]:
    return get_knowledge_question_bank(book_key)


def max_runs_for_mode(mode: str) -> int:
    if mode == STANDARD_MODE:
        return STANDARD_MAX_RUNS
    if mode == SLEEK_MODE:
        return SLEEK_MAX_RUNS
    raise ValueError("Select a supported knowledge-memorization mode.")


__all__ = [name for name in globals() if not name.startswith("_")]