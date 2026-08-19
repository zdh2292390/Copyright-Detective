"""Persuasive Jailbreak Detection Test Module - Adversarial prompting and jailbreak detection."""

from .jailbreak_probe import (
    run_persuasion_probe,
    get_persuasion_template,
    get_persuasion_prompt,
)
from .plots import (
    BASELINE_STRATEGY_LABEL,
    DistributionPlotData,
    MutationFootnote,
    build_rouge_l_distribution_boxplot,
    build_rouge_l_strategy_histogram,
    collect_distribution_plot_data,
    collect_rouge_l_by_strategy,
    format_mutation_footnote_lines,
    figure_to_png_bytes,
)
from .adversarial_prompting import (
    list_persuasion_strategies,
    get_mutation_instruction,
    run_inference_scaling,
    assess_intention_preservation,
    mutate_strategies,
    rank_by_rouge,
    list_baseline_prompts,
    ExperimentMode,
    DEFAULT_HP_REFERENCE_EXCERPT,
    DEFAULT_HB_REFERENCE_EXCERPT,
    DEFAULT_GA_REFERENCE_EXCERPT,
    serialise_mutation_with_judge,
    deserialise_mutation_with_judge,
    MutationWithJudge,
    MutationEvaluation,
    SimilarityMetrics,
    # Custom input mutation support
    list_custom_mutation_strategies,
    get_custom_mutation_instruction,
    run_custom_mutation,
    parse_custom_mutation_output,
    mutate_custom_strategies,
    CUSTOM_MUTATION_STRATEGIES,
)

__all__ = [
    # jailbreak_probe
    "run_persuasion_probe",
    "get_persuasion_template",
    "get_persuasion_prompt",
    # adversarial_prompting
    "list_persuasion_strategies",
    "get_mutation_instruction",
    "run_inference_scaling",
    "assess_intention_preservation",
    "mutate_strategies",
    "rank_by_rouge",
    "list_baseline_prompts",
    "ExperimentMode",
    "DEFAULT_HP_REFERENCE_EXCERPT",
    "DEFAULT_HB_REFERENCE_EXCERPT",
    "DEFAULT_GA_REFERENCE_EXCERPT",
    "serialise_mutation_with_judge",
    "deserialise_mutation_with_judge",
    "MutationWithJudge",
    "MutationEvaluation",
    "SimilarityMetrics",
    # Custom input mutation support
    "list_custom_mutation_strategies",
    "get_custom_mutation_instruction",
    "run_custom_mutation",
    "parse_custom_mutation_output",
    "mutate_custom_strategies",
    "CUSTOM_MUTATION_STRATEGIES",
    # plots
    "BASELINE_STRATEGY_LABEL",
    "DistributionPlotData",
    "MutationFootnote",
    "build_rouge_l_distribution_boxplot",
    "build_rouge_l_strategy_histogram",
    "collect_distribution_plot_data",
    "collect_rouge_l_by_strategy",
    "format_mutation_footnote_lines",
    "figure_to_png_bytes",
]
