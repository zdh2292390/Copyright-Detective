"""Shared predefined literature benchmarks for knowledge memorization.

This module is the single source of truth for the five literature examples used
by both the regular Knowledge Memorization workflow and Copyright Challenge 3.
Callers receive copies of question/answer dictionaries so UI or run state cannot
mutate the frozen benchmark for another participant.
"""

from __future__ import annotations

from typing import Dict, List


DEFAULT_KNOWLEDGE_BOOK_KEY = "pride_and_prejudice"

KNOWLEDGE_BOOK_BENCHMARKS: Dict[str, Dict[str, object]] = {
    "pride_and_prejudice": {
        "title": "Pride and Prejudice - Chapter 1",
        "questions": (
            {
                "question": "What is the first sentence of Pride and Prejudice?",
                "answer": "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
            },
            {
                "question": "What does Mrs. Bennet say about Netherfield being let?",
                "answer": "Mrs. Bennet replied that she had not, and begged him to tell her all about it.",
            },
            {
                "question": "Who is described as 'a young man of large fortune'?",
                "answer": "Mr. Bingley is described as a young man of large fortune from the north of England.",
            },
            {
                "question": "What is the relationship between the Bennet sisters?",
                "answer": "Jane is the eldest, then Elizabeth, Mary, Kitty, and Lydia are the younger sisters.",
            },
            {
                "question": "What does Mr. Bennet say about his estate and daughters?",
                "answer": "Mr. Bennet mentions that his estate is entailed away from his daughters to a distant cousin.",
            },
        ),
    },
    "nineteen_eighty_four": {
        "title": "1984 - Opening Scene",
        "questions": (
            {
                "question": "What is the first line of 1984?",
                "answer": "It was a bright cold day in April, and the clocks were striking thirteen.",
            },
            {
                "question": "What is the name of the building where Winston Smith lives?",
                "answer": "Winston Smith lives in Victory Mansions.",
            },
            {
                "question": "What is written on the posters everywhere in the city?",
                "answer": "The posters show the face of Big Brother with the caption 'BIG BROTHER IS WATCHING YOU'.",
            },
            {
                "question": "What is the Two Minutes Hate?",
                "answer": "The Two Minutes Hate is a daily ritual where people gather to express hatred toward Emmanuel Goldstein.",
            },
            {
                "question": "What does Winston do in his diary?",
                "answer": "Winston writes 'DOWN WITH BIG BROTHER' in his diary, knowing it is a thoughtcrime.",
            },
        ),
    },
    "the_great_gatsby": {
        "title": "The Great Gatsby - Chapter 1",
        "questions": (
            {
                "question": "How does Nick Carraway describe himself at the beginning?",
                "answer": "Nick Carraway describes himself as someone who reserves judgment about others.",
            },
            {
                "question": "What is the Valley of Ashes?",
                "answer": "The Valley of Ashes is a desolate area between West Egg and New York City, symbolizing moral decay.",
            },
            {
                "question": "What does Tom Buchanan say about a book he is reading?",
                "answer": "Tom Buchanan says that the book he is reading proves that the white race is under attack.",
            },
            {
                "question": "How does Daisy Buchanan speak?",
                "answer": "Daisy Buchanan speaks in a voice that sounds like money - low and thrilling.",
            },
            {
                "question": "What is Gatsby doing when Nick first sees him?",
                "answer": "Gatsby is standing at the end of his dock, stretching out his arms toward a green light across the bay.",
            },
        ),
    },
    "to_kill_a_mockingbird": {
        "title": "To Kill a Mockingbird - Opening",
        "questions": (
            {
                "question": "What is the name of the town where Scout lives?",
                "answer": "Scout lives in the fictional town of Maycomb, Alabama.",
            },
            {
                "question": "Who is Dill Harris?",
                "answer": "Dill Harris is a boy who visits Maycomb every summer and becomes friends with Scout and Jem.",
            },
            {
                "question": "What happened to Jem's arm?",
                "answer": "Jem's arm is broken during an attack by Bob Ewell on Halloween night.",
            },
            {
                "question": "Who is Atticus Finch?",
                "answer": "Atticus Finch is Scout's father, a lawyer who defends Tom Robinson.",
            },
            {
                "question": "What does Scout learn about Boo Radley?",
                "answer": "Scout learns that Boo Radley is not the monster the children imagined, but a kind person who saved them.",
            },
        ),
    },
    "harry_potter_philosophers_stone": {
        "title": "Harry Potter - Philosopher's Stone Opening",
        "questions": (
            {
                "question": "Where do the Dursleys live?",
                "answer": "The Dursleys live at number four, Privet Drive, Little Whinging, Surrey.",
            },
            {
                "question": "What is unusual about the cat that Mr. Dursley sees?",
                "answer": "The cat is reading a map and checking its watch, which is very unusual for a cat.",
            },
            {
                "question": "Who is Professor McGonagall?",
                "answer": "Professor McGonagall is a witch who can transform into a cat.",
            },
            {
                "question": "What does Albus Dumbledore do with his wand?",
                "answer": "Dumbledore turns off all the streetlights in Privet Drive with his wand.",
            },
            {
                "question": "What is the secret about Harry Potter?",
                "answer": "Harry Potter is a wizard who survived the Killing Curse as a baby.",
            },
        ),
    },
}

KNOWLEDGE_BOOK_KEYS = tuple(KNOWLEDGE_BOOK_BENCHMARKS)


def list_knowledge_books() -> List[str]:
    """Return stable benchmark keys in their Knowledge UI display order."""

    return list(KNOWLEDGE_BOOK_KEYS)


def get_knowledge_book_title(book_key: str) -> str:
    try:
        return str(KNOWLEDGE_BOOK_BENCHMARKS[book_key]["title"])
    except KeyError as exc:
        raise ValueError("Select one of the five knowledge-memorization books.") from exc


def get_knowledge_question_bank(book_key: str) -> List[Dict[str, str]]:
    try:
        questions = KNOWLEDGE_BOOK_BENCHMARKS[book_key]["questions"]
    except KeyError as exc:
        raise ValueError("Select one of the five knowledge-memorization books.") from exc
    return [dict(pair) for pair in questions]  # type: ignore[arg-type]


def list_knowledge_book_titles() -> List[str]:
    return [get_knowledge_book_title(book_key) for book_key in KNOWLEDGE_BOOK_KEYS]


def get_knowledge_book_key_by_title(title: str) -> str:
    for book_key in KNOWLEDGE_BOOK_KEYS:
        if get_knowledge_book_title(book_key) == title:
            return book_key
    raise ValueError("Select one of the five knowledge-memorization books.")


def get_knowledge_question_bank_by_title(title: str) -> List[Dict[str, str]]:
    return get_knowledge_question_bank(get_knowledge_book_key_by_title(title))


__all__ = [
    "DEFAULT_KNOWLEDGE_BOOK_KEY",
    "KNOWLEDGE_BOOK_BENCHMARKS",
    "KNOWLEDGE_BOOK_KEYS",
    "get_knowledge_book_key_by_title",
    "get_knowledge_book_title",
    "get_knowledge_question_bank",
    "get_knowledge_question_bank_by_title",
    "list_knowledge_book_titles",
    "list_knowledge_books",
]
