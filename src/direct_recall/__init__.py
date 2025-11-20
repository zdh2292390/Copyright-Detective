"""Recall Test Module - Text and document memorization detection."""

from .comparison import (
    compare_texts,
    enforce_exact_char_count,
    get_llm_completion,
    calculate_rouge_score,
    calculate_jaccard_index,
)
from .pdf_utils import extract_text_from_document, split_text_into_chunks
from .knowledge_qa import (
    generate_qa_pairs_from_document,
    run_knowledge_qa_evaluation,
    calculate_aggregate_metrics,
)
from .single_choice import (
    generate_single_choice_questions_from_document,
    generate_single_choice_questions_from_text,
    list_dataset_documents,
    load_dataset_excerpt,
    run_single_choice_evaluation,
    summarize_single_choice_results,
)
from .decop_analysis import get_available_datasets

__all__ = [
    # comparison
    "compare_texts",
    "enforce_exact_char_count",
    "get_llm_completion",
    "calculate_rouge_score",
    "calculate_jaccard_index",
    # pdf_utils
    "extract_text_from_document",
    "split_text_into_chunks",
    # knowledge_qa
    "generate_qa_pairs_from_document",
    "run_knowledge_qa_evaluation",
    "calculate_aggregate_metrics",
    # single_choice
    "generate_single_choice_questions_from_document",
    "generate_single_choice_questions_from_text",
    "list_dataset_documents",
    "load_dataset_excerpt",
    "run_single_choice_evaluation",
    "summarize_single_choice_results",
    # dataset management
    "get_available_datasets",
]
