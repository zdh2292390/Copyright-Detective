"""Content Recall Test Module - Text and document memorization detection."""

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
    generate_single_choice_questions_from_fragments,
    generate_single_choice_questions_from_document_fragments,
    list_dataset_documents,
    load_dataset_excerpt,
    load_predefined_examples,
    parse_question_indices,
    run_single_choice_evaluation,
    summarize_single_choice_results,
    get_predefined_examples_index,
)
from .decop_analysis import get_available_datasets
# Import persuasion functions from the independent adversarial_persuasion_detection module
# These are still used in content recall test for persuasion strategies
from src.adversarial_persuasion_detection import (
    run_persuasion_probe,
    get_persuasion_template,
    get_persuasion_prompt,
)

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
    "generate_single_choice_questions_from_fragments",
    "generate_single_choice_questions_from_document_fragments",
    "list_dataset_documents",
    "load_dataset_excerpt",
    "load_predefined_examples",
    "parse_question_indices",
    "run_single_choice_evaluation",
    "summarize_single_choice_results",
    "get_predefined_examples_index",
    # dataset management
    "get_available_datasets",
    # persuasion functions (imported from adversarial_persuasion_detection for backward compatibility)
    "run_persuasion_probe",
    "get_persuasion_template",
    "get_persuasion_prompt",
]
