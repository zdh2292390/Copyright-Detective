"""Neighborhood Comparison for Memorization Detection.

This module implements a "neighborhood comparison" technique inspired by 
uncertainty quantification (UQ) methods. By perturbing the input text slightly
and measuring how the model's response changes, we can detect mechanical memorization.

Key Insight:
- If a model has mechanically memorized content, small perturbations to the input
  will cause dramatic changes in the output (high loss/low similarity).
- This is because the model only "recognizes" the exact memorized sequence,
  not variations of it.
- Conversely, if the model has genuinely learned the content/patterns, small
  perturbations should produce similar outputs.

Perturbation Strategies:
1. Character-level: swap adjacent characters, introduce typos
2. Word-level: swap adjacent words, replace with synonyms
3. Semantic: paraphrase while preserving meaning
"""

from __future__ import annotations

import random
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from rouge_score import rouge_scorer

from src.direct_recall.comparison import (
    get_llm_completion,
    calculate_similarity_metrics,
)
from src.common.progress import (
    start_llm_progress,
    update_llm_progress,
    complete_llm_progress,
)


# Initialize ROUGE scorer
_ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


@dataclass
class PerturbedInput:
    """Represents a perturbed version of the input text."""
    original_text: str
    perturbed_text: str
    perturbation_type: str
    perturbation_description: str
    edit_distance: int = 0  # Number of edits made


@dataclass
class NeighborhoodResponse:
    """Response from the model for a perturbed input."""
    perturbed_input: PerturbedInput
    generated_text: str
    similarity_to_original: float  # Similarity to the original output
    rouge_l: float
    rouge_1: float
    jaccard: float
    error: Optional[str] = None


@dataclass
class NeighborhoodAnalysisResult:
    """Complete result of neighborhood comparison analysis."""
    original_input: str
    original_output: str
    perturbations: List[NeighborhoodResponse]
    avg_similarity_drop: float  # Average drop in similarity compared to original
    max_similarity_drop: float  # Maximum drop
    min_similarity: float  # Minimum similarity observed
    stability_score: float  # 0-1, higher = more stable (less likely memorized)
    memorization_score: float  # 0-1, higher = more likely memorized
    
    # Per-perturbation-type statistics
    type_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_input": self.original_input,
            "original_output": self.original_output,
            "avg_similarity_drop": self.avg_similarity_drop,
            "max_similarity_drop": self.max_similarity_drop,
            "min_similarity": self.min_similarity,
            "stability_score": self.stability_score,
            "memorization_score": self.memorization_score,
            "num_perturbations": len(self.perturbations),
            "type_statistics": self.type_statistics,
            "perturbations": [
                {
                    "type": p.perturbed_input.perturbation_type,
                    "description": p.perturbed_input.perturbation_description,
                    "original_text": p.perturbed_input.original_text[:100] + "..." if len(p.perturbed_input.original_text) > 100 else p.perturbed_input.original_text,
                    "perturbed_text": p.perturbed_input.perturbed_text[:100] + "..." if len(p.perturbed_input.perturbed_text) > 100 else p.perturbed_input.perturbed_text,
                    "generated_text": p.generated_text[:200] + "..." if len(p.generated_text) > 200 else p.generated_text,
                    "similarity_to_original": p.similarity_to_original,
                    "rouge_l": p.rouge_l,
                    "error": p.error,
                }
                for p in self.perturbations
            ],
        }


# ============================================================================
# Perturbation Functions
# ============================================================================

def _swap_adjacent_chars(text: str, num_swaps: int = 1) -> Tuple[str, int]:
    """Swap adjacent characters in the text.
    
    Returns:
        Tuple of (perturbed_text, actual_num_swaps)
    """
    if len(text) < 2:
        return text, 0
    
    chars = list(text)
    actual_swaps = 0
    
    # Find valid swap positions (not at word boundaries, not spaces)
    valid_positions = []
    for i in range(len(chars) - 1):
        if chars[i].isalnum() and chars[i + 1].isalnum():
            valid_positions.append(i)
    
    if not valid_positions:
        return text, 0
    
    # Perform swaps
    swap_positions = random.sample(valid_positions, min(num_swaps, len(valid_positions)))
    for pos in swap_positions:
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        actual_swaps += 1
    
    return "".join(chars), actual_swaps


def _introduce_typos(text: str, num_typos: int = 1) -> Tuple[str, int]:
    """Introduce typos by replacing characters with nearby keyboard keys.
    
    Returns:
        Tuple of (perturbed_text, actual_num_typos)
    """
    # Simple keyboard neighbor mapping
    keyboard_neighbors = {
        'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'erfcxs', 'e': 'rdsw',
        'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg', 'i': 'uojk', 'j': 'uikmnh',
        'k': 'iolmj', 'l': 'opk', 'm': 'njk', 'n': 'bhjm', 'o': 'iplk',
        'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'wedxza', 't': 'rfgy',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qeas', 'x': 'zsdc', 'y': 'tghu',
        'z': 'asx',
    }
    
    chars = list(text)
    actual_typos = 0
    
    # Find valid positions (alphabetic characters)
    valid_positions = [i for i, c in enumerate(chars) if c.lower() in keyboard_neighbors]
    
    if not valid_positions:
        return text, 0
    
    typo_positions = random.sample(valid_positions, min(num_typos, len(valid_positions)))
    
    for pos in typo_positions:
        char = chars[pos]
        is_upper = char.isupper()
        lower_char = char.lower()
        
        if lower_char in keyboard_neighbors:
            neighbors = keyboard_neighbors[lower_char]
            new_char = random.choice(neighbors)
            chars[pos] = new_char.upper() if is_upper else new_char
            actual_typos += 1
    
    return "".join(chars), actual_typos


def _swap_adjacent_words(text: str, num_swaps: int = 1) -> Tuple[str, int]:
    """Swap adjacent words in the text.
    
    Returns:
        Tuple of (perturbed_text, actual_num_swaps)
    """
    words = text.split()
    
    if len(words) < 2:
        return text, 0
    
    actual_swaps = 0
    valid_positions = list(range(len(words) - 1))
    
    swap_positions = random.sample(valid_positions, min(num_swaps, len(valid_positions)))
    
    for pos in sorted(swap_positions, reverse=True):
        words[pos], words[pos + 1] = words[pos + 1], words[pos]
        actual_swaps += 1
    
    return " ".join(words), actual_swaps


def _delete_random_words(text: str, num_deletions: int = 1) -> Tuple[str, int]:
    """Delete random words from the text.
    
    Returns:
        Tuple of (perturbed_text, actual_num_deletions)
    """
    words = text.split()
    
    if len(words) <= num_deletions + 2:  # Keep at least 2 words
        return text, 0
    
    # Don't delete first or last word
    valid_positions = list(range(1, len(words) - 1))
    
    if not valid_positions:
        return text, 0
    
    delete_positions = set(random.sample(valid_positions, min(num_deletions, len(valid_positions))))
    
    new_words = [w for i, w in enumerate(words) if i not in delete_positions]
    
    return " ".join(new_words), len(delete_positions)


def _replace_with_synonym_placeholder(text: str, num_replacements: int = 1) -> Tuple[str, int]:
    """Replace words with simple variations (add/remove 's', change tense).
    
    This is a simplified version that doesn't require external synonym databases.
    
    Returns:
        Tuple of (perturbed_text, actual_num_replacements)
    """
    words = text.split()
    
    if len(words) < 3:
        return text, 0
    
    # Simple word transformations
    actual_replacements = 0
    valid_positions = [i for i, w in enumerate(words) if len(w) > 3 and w.isalpha()]
    
    if not valid_positions:
        return text, 0
    
    replace_positions = random.sample(valid_positions, min(num_replacements, len(valid_positions)))
    
    for pos in replace_positions:
        word = words[pos]
        # Simple transformations
        if word.endswith('s') and len(word) > 4:
            words[pos] = word[:-1]  # Remove 's'
            actual_replacements += 1
        elif word.endswith('ed') and len(word) > 4:
            words[pos] = word[:-2] + 'ing'  # Change tense
            actual_replacements += 1
        elif word.endswith('ing') and len(word) > 5:
            words[pos] = word[:-3] + 'ed'  # Change tense
            actual_replacements += 1
        elif not word.endswith('s'):
            words[pos] = word + 's'  # Add 's'
            actual_replacements += 1
    
    return " ".join(words), actual_replacements


def _shuffle_middle_chars(text: str) -> Tuple[str, int]:
    """Shuffle middle characters of words while keeping first and last chars.
    
    Based on the phenomenon that humans can read text with shuffled middle chars.
    
    Returns:
        Tuple of (perturbed_text, num_words_shuffled)
    """
    words = text.split()
    num_shuffled = 0
    
    new_words = []
    for word in words:
        # Only shuffle words with 4+ characters
        if len(word) >= 4 and word.isalpha():
            middle = list(word[1:-1])
            random.shuffle(middle)
            new_word = word[0] + "".join(middle) + word[-1]
            if new_word != word:
                num_shuffled += 1
            new_words.append(new_word)
        else:
            new_words.append(word)
    
    return " ".join(new_words), num_shuffled


def generate_perturbations(
    text: str,
    num_perturbations_per_type: int = 2,
    seed: Optional[int] = None,
) -> List[PerturbedInput]:
    """Generate various perturbations of the input text.
    
    Args:
        text: Original input text.
        num_perturbations_per_type: Number of perturbations to generate per type.
        seed: Random seed for reproducibility.
    
    Returns:
        List of PerturbedInput objects.
    """
    if seed is not None:
        random.seed(seed)
    
    perturbations: List[PerturbedInput] = []
    
    # Character-level perturbations
    for i in range(num_perturbations_per_type):
        perturbed, num_edits = _swap_adjacent_chars(text, num_swaps=i + 1)
        if perturbed != text:
            perturbations.append(PerturbedInput(
                original_text=text,
                perturbed_text=perturbed,
                perturbation_type="char_swap",
                perturbation_description=f"Swapped {num_edits} adjacent character pair(s)",
                edit_distance=num_edits,
            ))
    
    for i in range(num_perturbations_per_type):
        perturbed, num_edits = _introduce_typos(text, num_typos=i + 1)
        if perturbed != text:
            perturbations.append(PerturbedInput(
                original_text=text,
                perturbed_text=perturbed,
                perturbation_type="typo",
                perturbation_description=f"Introduced {num_edits} typo(s)",
                edit_distance=num_edits,
            ))
    
    # Word-level perturbations
    for i in range(num_perturbations_per_type):
        perturbed, num_edits = _swap_adjacent_words(text, num_swaps=i + 1)
        if perturbed != text:
            perturbations.append(PerturbedInput(
                original_text=text,
                perturbed_text=perturbed,
                perturbation_type="word_swap",
                perturbation_description=f"Swapped {num_edits} adjacent word pair(s)",
                edit_distance=num_edits,
            ))
    
    for i in range(num_perturbations_per_type):
        perturbed, num_edits = _delete_random_words(text, num_deletions=i + 1)
        if perturbed != text:
            perturbations.append(PerturbedInput(
                original_text=text,
                perturbed_text=perturbed,
                perturbation_type="word_deletion",
                perturbation_description=f"Deleted {num_edits} word(s)",
                edit_distance=num_edits,
            ))
    
    for i in range(num_perturbations_per_type):
        perturbed, num_edits = _replace_with_synonym_placeholder(text, num_replacements=i + 1)
        if perturbed != text:
            perturbations.append(PerturbedInput(
                original_text=text,
                perturbed_text=perturbed,
                perturbation_type="word_variation",
                perturbation_description=f"Applied {num_edits} word variation(s)",
                edit_distance=num_edits,
            ))
    
    # Middle character shuffle (single perturbation)
    perturbed, num_edits = _shuffle_middle_chars(text)
    if perturbed != text:
        perturbations.append(PerturbedInput(
            original_text=text,
            perturbed_text=perturbed,
            perturbation_type="middle_shuffle",
            perturbation_description=f"Shuffled middle chars of {num_edits} word(s)",
            edit_distance=num_edits,
        ))
    
    return perturbations


def _compute_similarity(text1: str, text2: str) -> Tuple[float, float, float]:
    """Compute similarity metrics between two texts.
    
    Returns:
        Tuple of (rouge_l, rouge_1, jaccard)
    """
    if not text1.strip() or not text2.strip():
        return 0.0, 0.0, 0.0
    
    scores = _ROUGE_SCORER.score(text1, text2)
    rouge_l = scores["rougeL"].fmeasure
    rouge_1 = scores["rouge1"].fmeasure
    
    # Jaccard
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    jaccard = len(intersection) / len(union) if union else 0.0
    
    return rouge_l, rouge_1, jaccard


def run_neighborhood_comparison(
    input_text: str,
    prompt_template: str,
    api_key: str,
    model_name: str,
    provider: str = "OpenAI",
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_perturbations_per_type: int = 2,
    max_tokens: int = 500,
    seed: Optional[int] = 42,
    progress_message: Optional[str] = None,
) -> NeighborhoodAnalysisResult:
    """Run neighborhood comparison analysis.
    
    This function:
    1. Generates the original output from the input text
    2. Creates perturbations of the input text
    3. Generates outputs for each perturbation
    4. Compares all outputs to detect memorization patterns
    
    Args:
        input_text: The original input text to analyze.
        prompt_template: Template for the prompt, with {input} placeholder.
        api_key: API key for the LLM provider.
        model_name: Name of the model to use.
        provider: LLM provider.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        num_perturbations_per_type: Number of perturbations per type.
        max_tokens: Maximum tokens to generate.
        seed: Random seed for reproducibility.
        progress_message: Optional progress message.
    
    Returns:
        NeighborhoodAnalysisResult with analysis details.
    """
    label_placeholder, bar_placeholder, progress_bar = start_llm_progress(
        progress_message or f"Running neighborhood analysis · {model_name}"
    )
    update_llm_progress(progress_bar, value=5)
    
    # Generate original output
    original_prompt = prompt_template.format(input=input_text)
    original_output = get_llm_completion(
        original_prompt,
        api_key,
        model_name,
        provider,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    
    if isinstance(original_output, str) and original_output.startswith("Error"):
        complete_llm_progress(
            label_placeholder,
            bar_placeholder,
            progress_bar,
            final_message="Neighborhood analysis failed",
            success=False,
            linger=0.5,
        )
        return NeighborhoodAnalysisResult(
            original_input=input_text,
            original_output="",
            perturbations=[],
            avg_similarity_drop=0.0,
            max_similarity_drop=0.0,
            min_similarity=0.0,
            stability_score=0.0,
            memorization_score=0.0,
        )
    
    update_llm_progress(progress_bar, value=15)
    
    # Generate perturbations
    perturbations = generate_perturbations(
        input_text,
        num_perturbations_per_type=num_perturbations_per_type,
        seed=seed,
    )
    
    if not perturbations:
        complete_llm_progress(
            label_placeholder,
            bar_placeholder,
            progress_bar,
            final_message="No perturbations could be generated",
            success=False,
            linger=0.5,
        )
        return NeighborhoodAnalysisResult(
            original_input=input_text,
            original_output=original_output,
            perturbations=[],
            avg_similarity_drop=0.0,
            max_similarity_drop=0.0,
            min_similarity=1.0,
            stability_score=1.0,
            memorization_score=0.0,
        )
    
    # Process perturbations
    responses: List[NeighborhoodResponse] = []
    total_perturbations = len(perturbations)
    
    for i, perturbed_input in enumerate(perturbations):
        progress_value = 15 + int((i / total_perturbations) * 75)
        update_llm_progress(progress_bar, value=progress_value)
        
        # Generate output for perturbed input
        perturbed_prompt = prompt_template.format(input=perturbed_input.perturbed_text)
        perturbed_output = get_llm_completion(
            perturbed_prompt,
            api_key,
            model_name,
            provider,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )
        
        error = None
        if isinstance(perturbed_output, str) and perturbed_output.startswith("Error"):
            error = perturbed_output
            perturbed_output = ""
        
        # Compute similarity to original output
        rouge_l, rouge_1, jaccard = _compute_similarity(original_output, perturbed_output)
        similarity = rouge_l  # Use ROUGE-L as primary similarity metric
        
        responses.append(NeighborhoodResponse(
            perturbed_input=perturbed_input,
            generated_text=perturbed_output,
            similarity_to_original=similarity,
            rouge_l=rouge_l,
            rouge_1=rouge_1,
            jaccard=jaccard,
            error=error,
        ))
    
    update_llm_progress(progress_bar, value=95)
    
    # Calculate statistics
    valid_responses = [r for r in responses if r.error is None]
    
    if not valid_responses:
        complete_llm_progress(
            label_placeholder,
            bar_placeholder,
            progress_bar,
            final_message="All perturbation queries failed",
            success=False,
            linger=0.5,
        )
        return NeighborhoodAnalysisResult(
            original_input=input_text,
            original_output=original_output,
            perturbations=responses,
            avg_similarity_drop=0.0,
            max_similarity_drop=0.0,
            min_similarity=0.0,
            stability_score=0.0,
            memorization_score=0.0,
        )
    
    similarities = [r.similarity_to_original for r in valid_responses]
    avg_similarity = statistics.mean(similarities)
    min_similarity = min(similarities)
    
    # Similarity drop (1 - similarity, since 1 would be perfect match)
    similarity_drops = [1.0 - s for s in similarities]
    avg_similarity_drop = statistics.mean(similarity_drops)
    max_similarity_drop = max(similarity_drops)
    
    # Stability score: how consistent are outputs across perturbations
    # Higher = more stable (outputs similar despite perturbations)
    stability_score = avg_similarity
    
    # Memorization score: inverse of stability
    # High drops in similarity with small perturbations = memorization
    # We also consider variance in responses
    variance = statistics.variance(similarities) if len(similarities) > 1 else 0.0
    memorization_score = min(1.0, avg_similarity_drop + variance * 0.5)
    
    # Per-type statistics
    type_statistics: Dict[str, Dict[str, float]] = {}
    type_groups: Dict[str, List[float]] = {}
    
    for r in valid_responses:
        ptype = r.perturbed_input.perturbation_type
        if ptype not in type_groups:
            type_groups[ptype] = []
        type_groups[ptype].append(r.similarity_to_original)
    
    for ptype, sims in type_groups.items():
        type_statistics[ptype] = {
            "avg_similarity": statistics.mean(sims),
            "min_similarity": min(sims),
            "max_similarity": max(sims),
            "avg_drop": 1.0 - statistics.mean(sims),
        }
    
    complete_llm_progress(
        label_placeholder,
        bar_placeholder,
        progress_bar,
        final_message=f"Neighborhood analysis complete · {len(valid_responses)} perturbations analyzed",
        success=True,
    )
    
    return NeighborhoodAnalysisResult(
        original_input=input_text,
        original_output=original_output,
        perturbations=responses,
        avg_similarity_drop=avg_similarity_drop,
        max_similarity_drop=max_similarity_drop,
        min_similarity=min_similarity,
        stability_score=stability_score,
        memorization_score=memorization_score,
        type_statistics=type_statistics,
    )


def format_neighborhood_analysis_summary(result: NeighborhoodAnalysisResult) -> str:
    """Format the neighborhood analysis result as a human-readable summary."""
    lines = [
        "🔍 **Neighborhood Comparison Analysis**",
        "",
        f"- **Memorization Score**: {result.memorization_score:.2%}",
        f"- **Stability Score**: {result.stability_score:.2%}",
        f"- **Average Similarity Drop**: {result.avg_similarity_drop:.2%}",
        f"- **Maximum Similarity Drop**: {result.max_similarity_drop:.2%}",
        f"- **Minimum Similarity**: {result.min_similarity:.2%}",
        f"- **Perturbations Tested**: {len(result.perturbations)}",
        "",
    ]
    
    if result.type_statistics:
        lines.append("**Per-Perturbation-Type Results:**")
        for ptype, stats in result.type_statistics.items():
            type_label = ptype.replace("_", " ").title()
            lines.append(
                f"  - {type_label}: avg drop = {stats['avg_drop']:.2%}, "
                f"min sim = {stats['min_similarity']:.2%}"
            )
    
    # Interpretation
    lines.append("")
    if result.memorization_score > 0.6:
        lines.append(
            "🚨 **High memorization likelihood!** The model shows significant sensitivity to input "
            "perturbations, suggesting mechanical memorization of the original text."
        )
    elif result.memorization_score > 0.35:
        lines.append(
            "⚠️ **Moderate memorization signals.** The model shows some sensitivity to perturbations, "
            "which may indicate partial memorization."
        )
    else:
        lines.append(
            "✅ **Low memorization likelihood.** The model produces consistent outputs across "
            "perturbations, suggesting learned patterns rather than memorization."
        )
    
    return "\n".join(lines)


# Default prompt template for continuation tasks
DEFAULT_CONTINUATION_TEMPLATE = """Continue the following text in the same style and tone:

{input}

Continuation:"""
