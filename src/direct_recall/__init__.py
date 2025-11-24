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
from .persuasive_jailbreak import (
    run_persuasion_probe,
    get_persuasion_template,
    get_persuasion_prompt,
    list_persuasion_strategies,
    get_mutation_instruction,
    run_inference_scaling,
    assess_intention_preservation,
    mutate_strategies,
    rank_by_rouge,
    list_baseline_prompts,
    ExperimentMode,
    DEFAULT_HP_REFERENCE_EXCERPT,
    serialise_mutation_with_judge,
    deserialise_mutation_with_judge,
    MutationWithJudge,
    MutationEvaluation,
    SimilarityMetrics,
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
    # persuasive_jailbreak
    "run_persuasion_probe",
    "get_persuasion_template",
    "get_persuasion_prompt",
    "list_persuasion_strategies",
    "get_mutation_instruction",
    "run_inference_scaling",
    "assess_intention_preservation",
    "mutate_strategies",
    "rank_by_rouge",
    "list_baseline_prompts",
    "ExperimentMode",
    "DEFAULT_HP_REFERENCE_EXCERPT",
    "serialise_mutation_with_judge",
    "deserialise_mutation_with_judge",
    "MutationWithJudge",
    "MutationEvaluation",
    "SimilarityMetrics",
]
