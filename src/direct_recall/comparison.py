import hashlib
import openai
import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional

import anthropic
import google.generativeai as genai
from Levenshtein import distance
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.prompt_utils import get_full_prompt
from src.common.progress import (
    start_llm_progress,
    update_llm_progress,
    complete_llm_progress,
)


def _normalize_spaces(s: str) -> str:
    return " ".join(s.strip().split())


_TOKEN_SPLIT_RE = re.compile(r"\w+|\s+|[^\w\s]", flags=re.UNICODE)

_ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

_MINHASH_PERMUTATIONS = 128
_MINHASH_SHINGLE_SIZE = 3
_MINHASH_PRIME = 4_294_967_311
_MINHASH_RANDOM = random.Random(314159)
_MINHASH_COEFFICIENTS = [
    (
        _MINHASH_RANDOM.randint(1, _MINHASH_PRIME - 1),
        _MINHASH_RANDOM.randint(0, _MINHASH_PRIME - 1),
    )
    for _ in range(_MINHASH_PERMUTATIONS)
]


@dataclass(frozen=True)
class DiffToken:
    text: str
    label: str  # match, miss, extra, neutral


def _tokenize_for_diff(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return _TOKEN_SPLIT_RE.findall(text)


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    if not seq1 or not seq2:
        return 0

    len1, len2 = len(seq1), len(seq2)
    previous = [0] * (len2 + 1)
    current = [0] * (len2 + 1)

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous, current = current, [0] * (len2 + 1)

    return previous[-1]


def _compute_semantic_similarity(text1: str, text2: str) -> float:
    stripped_1 = text1.strip()
    stripped_2 = text2.strip()
    if not stripped_1 and not stripped_2:
        return 1.0
    if not stripped_1 or not stripped_2:
        return 0.0

    vectorizer = TfidfVectorizer()
    try:
        matrix = vectorizer.fit_transform([text1, text2])
    except ValueError:
        # Empty vocabulary or other tokenization issues
        return 0.0

    if matrix.nnz == 0:
        return 0.0

    similarity = cosine_similarity(matrix[0], matrix[1])[0][0]
    # Clamp to [0, 1] to guard against floating point drift
    return float(max(0.0, min(1.0, similarity)))


def _generate_word_shingles(text: str, size: int = _MINHASH_SHINGLE_SIZE) -> List[str]:
    tokens = text.lower().split()
    if not tokens:
        return []
    if len(tokens) <= size:
        return [" ".join(tokens)]
    return [" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]


def _hash_shingle(shingle: str) -> int:
    digest = hashlib.sha1(shingle.encode("utf-8")).hexdigest()
    return int(digest, 16) % _MINHASH_PRIME


def _compute_minhash_signature(shingles: List[str]) -> List[int]:
    if not shingles:
        return [_MINHASH_PRIME] * _MINHASH_PERMUTATIONS

    shingle_hashes = [_hash_shingle(shingle) for shingle in shingles]
    signature: List[int] = []
    for a, b in _MINHASH_COEFFICIENTS:
        min_value = min(((a * value) + b) % _MINHASH_PRIME for value in shingle_hashes)
        signature.append(min_value)
    return signature


def _compute_minhash_similarity(text1: str, text2: str) -> float:
    shingles1 = _generate_word_shingles(text1)
    shingles2 = _generate_word_shingles(text2)

    if not shingles1 and not shingles2:
        return 1.0
    if not shingles1 or not shingles2:
        return 0.0

    signature1 = _compute_minhash_signature(shingles1)
    signature2 = _compute_minhash_signature(shingles2)
    matches = sum(1 for h1, h2 in zip(signature1, signature2) if h1 == h2)
    return matches / _MINHASH_PERMUTATIONS


def compute_direct_recall_overlap(
    reference_text: str,
    candidate_text: str,
) -> Dict[str, object]:
    """Compute token-level overlap between ground truth and generated text.

    Returns a mapping containing token sequences for both texts with highlight labels
    and aggregate counts for match/miss/extra tokens. Whitespace-only tokens are
    preserved for rendering but labelled as ``neutral`` and excluded from counts.
    """

    reference_tokens = _tokenize_for_diff(reference_text)
    candidate_tokens = _tokenize_for_diff(candidate_text)

    matcher = SequenceMatcher(None, reference_tokens, candidate_tokens, autojunk=False)

    reference_alignment: List[DiffToken] = []
    candidate_alignment: List[DiffToken] = []
    def add_tokens(
        target: List[DiffToken],
        tokens: List[str],
        label: str,
    ) -> None:
        for token in tokens:
            if token == "":
                continue
            effective_label = label if token.strip() else "neutral"
            target.append(DiffToken(token, effective_label))

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            add_tokens(reference_alignment, reference_tokens[i1:i2], "match")
            add_tokens(candidate_alignment, candidate_tokens[j1:j2], "match")
        elif tag == "delete":
            add_tokens(reference_alignment, reference_tokens[i1:i2], "miss")
        elif tag == "insert":
            add_tokens(candidate_alignment, candidate_tokens[j1:j2], "extra")
        elif tag == "replace":
            add_tokens(reference_alignment, reference_tokens[i1:i2], "miss")
            add_tokens(candidate_alignment, candidate_tokens[j1:j2], "extra")

    def token_length(tokens: List[DiffToken]) -> int:
        return sum(1 for token in tokens if token.label != "neutral")

    def token_count(tokens: List[DiffToken], label: str) -> int:
        return sum(1 for token in tokens if token.label == label)

    ground_match = token_count(reference_alignment, "match")
    ground_miss = token_count(reference_alignment, "miss")
    generated_match = token_count(candidate_alignment, "match")
    generated_extra = token_count(candidate_alignment, "extra")

    counts = {
        "match": min(ground_match, generated_match),
        "miss": ground_miss,
        "extra": generated_extra,
    }

    return {
        "ground_tokens": reference_alignment,
        "generated_tokens": candidate_alignment,
        "counts": counts,
        "ground_non_whitespace": token_length(reference_alignment),
        "generated_non_whitespace": token_length(candidate_alignment),
    }


def enforce_exact_char_count(text: str, target: Optional[int]) -> str:
    """Ensure output has at most `target` characters by truncating."""
    if not target:
        return _normalize_spaces(text)

    normalized_text = _normalize_spaces(text)
    if len(normalized_text) > target:
        return normalized_text[:target]
    
    return normalized_text


def enforce_exact_word_count(text: str, target_words: Optional[int]) -> str:
    """Clip the text to the first ``target_words`` tokens (space-delimited)."""
    if not target_words:
        return text

    words = _normalize_spaces(text).split()
    if len(words) <= target_words:
        return " ".join(words)

    return " ".join(words[:target_words])

# Backward-compatible alias (internal use in legacy imports)
_enforce_exact_char_count = enforce_exact_char_count

def get_llm_completion(
    prompt,
    api_key,
    model_name,
    provider="OpenAI",
    temperature=0.7,
    top_p=1.0,
    *,
    progress_message: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
):
    """
    Gets a completion from the specified LLM.
    """
    label_placeholder, bar_placeholder, progress_bar = start_llm_progress(
        progress_message or f"Calling {provider} · {model_name}"
    )
    update_llm_progress(progress_bar, value=15)

    try:
        if provider == "OpenAI":
            client = openai.OpenAI(api_key=api_key)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            request_kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
            }
            if max_output_tokens is not None:
                request_kwargs["max_tokens"] = max_output_tokens
            if stop_sequences:
                request_kwargs["stop"] = stop_sequences
            response = client.chat.completions.create(**request_kwargs)
            result_text = response.choices[0].message.content.strip()
        
        elif provider == "OpenRouter":
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            request_kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "extra_headers": {
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "Copyright Detective"
                },
            }
            if max_output_tokens is not None:
                request_kwargs["max_tokens"] = max_output_tokens
            if stop_sequences:
                request_kwargs["stop"] = stop_sequences
            response = client.chat.completions.create(**request_kwargs)
            result_text = response.choices[0].message.content.strip()
        
        elif provider == "Anthropic":
            client = anthropic.Anthropic(api_key=api_key)
            request_kwargs = {
                "model": model_name,
                "max_tokens": max_output_tokens or 1000,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "top_p": top_p,
            }
            if stop_sequences:
                request_kwargs["stop_sequences"] = stop_sequences
            response = client.messages.create(**request_kwargs)
            result_text = response.content[0].text.strip()
        
        elif provider == "Google Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            generation_config_kwargs = {
                "temperature": temperature,
                "top_p": top_p,
            }
            if max_output_tokens is not None:
                generation_config_kwargs["max_output_tokens"] = max_output_tokens
            if stop_sequences:
                generation_config_kwargs["stop_sequences"] = stop_sequences
            generation_config = genai.types.GenerationConfig(**generation_config_kwargs)
            response = model.generate_content(prompt, generation_config=generation_config)
            result_text = response.text.strip()
        
        else:
            error_message = f"Error: Unsupported provider {provider}"
            complete_llm_progress(
                label_placeholder,
                bar_placeholder,
                progress_bar,
                final_message=error_message,
                success=False,
                linger=0.5,
            )
            return error_message
    
    except Exception as e:
        complete_llm_progress(
            label_placeholder,
            bar_placeholder,
            progress_bar,
            final_message="LLM request failed",
            success=False,
            linger=0.6,
        )
        return f"Error calling API: {e}"

    update_llm_progress(progress_bar, value=80)
    complete_llm_progress(
        label_placeholder,
        bar_placeholder,
        progress_bar,
        final_message=f"LLM request completed ({provider} · {model_name})",
        success=True,
    )
    return result_text

def calculate_rouge_score(text1, text2):
    """
    Calculates the ROUGE-L score between two texts.
    """
    scores = _ROUGE_SCORER.score(text1, text2)
    return scores['rougeL'].fmeasure

def calculate_jaccard_index(text1, text2):
    """
    Calculates the Jaccard Index between two texts.
    """
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(intersection) / len(union)


def calculate_similarity_metrics(reference_text: str, candidate_text: str) -> Dict[str, float]:
    """Calculate a bundle of overlap and distance metrics between two texts."""

    rouge_scores = _ROUGE_SCORER.score(reference_text, candidate_text)

    # Character-level LCS
    lcs_char_length = _lcs_length(list(reference_text), list(candidate_text))
    max_char_length = max(len(reference_text), len(candidate_text))
    lcs_char_ratio = (lcs_char_length / max_char_length) if max_char_length else 0.0

    # Word-level LCS and derived ACS
    ref_words = reference_text.split()
    cand_words = candidate_text.split()
    lcs_word_length = _lcs_length(ref_words, cand_words)
    max_word_length = max(len(ref_words), len(cand_words))
    lcs_word_ratio = (lcs_word_length / max_word_length) if max_word_length else 0.0
    recall = (lcs_word_length / len(ref_words)) if ref_words else 0.0
    precision = (lcs_word_length / len(cand_words)) if cand_words else 0.0
    acs_word = (recall + precision) / 2 if (ref_words or cand_words) else 0.0

    metrics: Dict[str, float] = {
        "rouge_1": rouge_scores["rouge1"].fmeasure,
        "rouge_l": rouge_scores["rougeL"].fmeasure,
        "lcs_char_length": float(lcs_char_length),
        "lcs_char_ratio": lcs_char_ratio,
        "lcs_word_length": float(lcs_word_length),
        "lcs_word_ratio": lcs_word_ratio,
        "acs_word": acs_word,
        "jaccard_index": calculate_jaccard_index(reference_text, candidate_text),
        "levenshtein": float(distance(reference_text, candidate_text)),
        "semantic_similarity": _compute_semantic_similarity(reference_text, candidate_text),
        "minhash_similarity": _compute_minhash_similarity(reference_text, candidate_text),
    }

    return metrics

def compare_texts(
    input_text,
    reference_text,
    api_key,
    model_name,
    provider="OpenAI",
    prompt_type="Next-Passage Prediction",
    chunk_size=None,
    temperature=0.7,
    top_p=1.0,
    continuation_method="Normal Continuation",
    custom_template: Optional[str] = None,
    mode: str = "Zero-Shot",
    target_word_count: Optional[int] = None,
    extra_prompt_instructions: Optional[str] = None,
):
    """
    Generates text based on the input_text according to prompt_type and compares it to reference_text.
        prompt_type choices:
      - "Next-Passage Prediction": continue from the given prefix (input_text)
      - "Prior-Context Reconstruction": infer the preceding sentence given a continuation (input_text)
      - "Title Prediction": infer a likely title/attribution from the snippet (input_text)
        continuation_method selects the strategy template for reconstruction prompts.

        Returns a tuple ``(generated_text, metrics)`` where ``metrics`` is a dictionary containing
        ROUGE-1, ROUGE-L, character/word LCS (length + ratio), ACS (word), Levenshtein distance,
        semantic similarity, MinHash similarity, and the Jaccard index. If no reference text is
        supplied the metrics entry is ``None``.
    """
    # Determine target length for generation
    if reference_text and prompt_type != "Title Prediction":
        target_char_count = len(reference_text)
    else:
        # Default small size for titles or missing reference
        target_char_count = 80  # A reasonable default for titles

    # Build prompt according to selected probe
    if prompt_type == "Next-Passage Prediction":
        prompt = get_full_prompt(
            prompt_type,
            input_text,
            chunk_size=chunk_size,
            continuation_method=continuation_method,
            char_count=target_char_count,
            custom_template=custom_template,
            mode=mode,
        )
    elif prompt_type == "Prior-Context Reconstruction":
        word_target = (
            chunk_size
            if chunk_size is not None
            else (len(reference_text.split()) if reference_text else None)
        )
        prompt = get_full_prompt(
            prompt_type,
            input_text,
            chunk_size=word_target,
            continuation_method=continuation_method,
            char_count=target_char_count,
            custom_template=custom_template,
        )
    elif prompt_type == "User-Defined Evaluation":
        # For user-defined evaluation, use the custom template directly as the prompt
        prompt = custom_template if custom_template else ""
    else:  # Title Prediction
        prompt = (
            "Provide only a short, likely title or attribution for the following text snippet. Do NOT include commentary, summaries, or extra formatting — return only the inferred title/attribution.\n\nSnippet:\n" + input_text
        )

    if extra_prompt_instructions:
        prompt = f"{prompt}\n\n{extra_prompt_instructions.strip()}"

    generated_text = get_llm_completion(prompt, api_key, model_name, provider, temperature=temperature, top_p=top_p)
    # Return early if API error to avoid post-processing masking the error message
    if isinstance(generated_text, str) and generated_text.startswith("Error"):
        return generated_text, None

    # Enforce exact character length to match the ground truth
    generated_text = enforce_exact_char_count(
        generated_text,
        target_char_count,
    )

    generated_text = enforce_exact_word_count(
        generated_text,
        target_word_count,
    )

    # If a reference/target text is provided, compute similarity metrics; otherwise return zeros.
    metrics = (
        calculate_similarity_metrics(reference_text, generated_text)
        if reference_text
        else None
    )

    return generated_text, metrics
