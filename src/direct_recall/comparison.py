import hashlib
import json
import math
import openai
import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import google.genai as genai
import streamlit as st
from Levenshtein import distance
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.api_concurrency import ApiConcurrencyTimeout, limit_api_concurrency
from src.prompt_utils import get_full_prompt
from src.kimi_utils import normalize_kimi_sampling_params
from src.openai_utils import apply_openai_request_compat, unsupported_openai_sampling_param
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


_GEMINI_MODEL_ALIASES = {
    # Retired 1.5 / 2.0 names → current GA model
    "gemini-1.5-flash": "gemini-3.5-flash",
    "gemini-1.5-pro": "gemini-3.5-flash",
    "gemini-pro": "gemini-3.5-flash",
    "gemini-1.5-flash-001": "gemini-3.5-flash",
    "gemini-1.5-pro-001": "gemini-3.5-flash",
    "gemini-1.5-flash-latest": "gemini-3.5-flash",
    "gemini-1.5-pro-latest": "gemini-3.5-flash",
    "gemini-2.0-flash": "gemini-3.5-flash",
    "gemini-2.0-flash-001": "gemini-3.5-flash",
    "gemini-2.0-flash-lite": "gemini-3.1-flash-lite",
    "gemini-3-pro-preview": "gemini-3.5-flash",
}

_GEMINI_DEFAULT_MODEL = "gemini-3.5-flash"


def _normalize_gemini_model(model_name: Optional[str]) -> str:
    """Map legacy Gemini names to current v1 identifiers."""

    if not model_name:
        return _GEMINI_DEFAULT_MODEL
    return _GEMINI_MODEL_ALIASES.get(model_name, model_name)


def _is_gemma_model(model_name: str) -> bool:
    """Check if the model is a Gemma model that doesn't support system messages."""
    if not model_name:
        return False
    model_lower = model_name.lower()
    return "gemma" in model_lower


def _build_messages_for_model(
    model_name: str,
    system_content: str,
    user_content: str,
) -> List[Dict[str, str]]:
    """Build messages list, merging system message into user message for Gemma models.
    
    Args:
        model_name: The model name
        system_content: System message content
        user_content: User message content
        
    Returns:
        List of message dictionaries
    """
    if _is_gemma_model(model_name):
        # For Gemma models, merge system message into user message
        combined_content = f"{system_content}\n\n{user_content}"
        return [{"role": "user", "content": combined_content}]
    else:
        # For other models, use separate system and user messages
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]


def _require_api_key(api_key: Optional[str], provider: str) -> Optional[str]:
    if str(api_key or "").strip():
        return None
    return (
        f"Error: {provider} API key is missing. "
        "Enter your key in the sidebar under API Configuration."
    )


def _extract_chat_message_text(response) -> str:
    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", None)
    refusal = getattr(message, "refusal", None)
    if content:
        return str(content).strip()
    if refusal:
        return f"Error: Model refused the request: {refusal}"
    finish_reason = getattr(choice, "finish_reason", "") or "unknown"
    return f"Error: Model returned empty content (finish_reason={finish_reason})."


def _is_unsupported_max_tokens_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "max_tokens" in text and ("max_completion_tokens" in text or "unsupported" in text)


def _is_logprobs_error(exc: Exception) -> bool:
    return "logprob" in str(exc).lower()


def create_openai_chat_completion(client, request_kwargs: Dict[str, Any]):
    """Call OpenAI chat completions with compatibility retries."""
    kwargs = apply_openai_request_compat(dict(request_kwargs))
    last_exc: Optional[Exception] = None
    for _ in range(5):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            changed = False
            if "max_tokens" in kwargs and _is_unsupported_max_tokens_error(exc):
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                changed = True
            elif kwargs.get("logprobs") and _is_logprobs_error(exc):
                kwargs.pop("logprobs", None)
                kwargs.pop("top_logprobs", None)
                changed = True
            else:
                param = unsupported_openai_sampling_param(exc)
                if param and param in kwargs:
                    kwargs.pop(param, None)
                    changed = True
            if not changed:
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("OpenAI chat completion failed without an exception")


@dataclass(frozen=True)
class DiffToken:
    text: str
    label: str  # match, miss, extra, neutral


def _tokenize_for_diff(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return _TOKEN_SPLIT_RE.findall(text)


def _normalize_token_for_matching(token: str) -> str:
    """Normalize a token for matching purposes.
    
    Similar to JavaScript's normalizeToken function:
    - Convert to lowercase
    - Remove non-alphanumeric characters
    - Returns empty string for whitespace-only tokens
    """
    if not token or not token.strip():
        return ""
    # Convert to lowercase and remove non-word characters (keep only alphanumeric)
    normalized = re.sub(r'[^\w]', '', token.lower())
    return normalized


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
    
    This implementation follows the JavaScript matching logic:
    - Tokens are normalized (lowercase, non-alphanumeric removed) for matching
    - Only non-empty normalized tokens are considered for matching
    - Original tokens are preserved for display purposes
    """

    reference_tokens = _tokenize_for_diff(reference_text)
    candidate_tokens = _tokenize_for_diff(candidate_text)

    # Create normalized versions for matching (similar to JavaScript normalizeToken)
    # Only normalize actual tokens (non-whitespace), whitespace tokens remain as-is
    reference_normalized = [
        _normalize_token_for_matching(token) if token.strip() else token
        for token in reference_tokens
    ]
    candidate_normalized = [
        _normalize_token_for_matching(token) if token.strip() else token
        for token in candidate_tokens
    ]

    # Use normalized tokens for matching, but we'll map back to original tokens for display
    matcher = SequenceMatcher(None, reference_normalized, candidate_normalized, autojunk=False)

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
            # For "equal" opcodes, normalized tokens match.
            # Following JavaScript logic: only count as "match" if normalized token is non-empty.
            # Empty normalized tokens (whitespace) will be labeled as "neutral" by add_tokens.
            # Since SequenceMatcher guarantees ref_segment and cand_segment have same length in "equal",
            # we can process them together.
            for idx in range(i2 - i1):
                ref_token = reference_tokens[i1 + idx]
                cand_token = candidate_tokens[j1 + idx]
                ref_norm = reference_normalized[i1 + idx]
                
                # Only mark as match if normalized version is non-empty
                # (matching JavaScript: seq1[i-1] === seq2[j-1] && seq1[i-1] !== '')
                # Since we're in "equal" opcode, ref_norm == cand_norm is guaranteed
                if ref_norm != "":
                    add_tokens(reference_alignment, [ref_token], "match")
                    add_tokens(candidate_alignment, [cand_token], "match")
                else:
                    # Whitespace tokens: add with "match" label, but add_tokens will convert to "neutral"
                    add_tokens(reference_alignment, [ref_token], "match")
                    add_tokens(candidate_alignment, [cand_token], "match")
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

    # ground_match and generated_match should always be equal since they come from
    # the same "equal" opcodes in SequenceMatcher. However, due to whitespace token
    # handling (tokens with only whitespace are labeled "neutral" and excluded from counts),
    # they might differ slightly in edge cases.
    # 
    # For correctness:
    # - match: number of tokens that appear in both texts (from ground truth perspective)
    # - miss: number of tokens in ground truth that don't appear in generated text
    # - extra: number of tokens in generated text that don't appear in ground truth
    #
    # We use ground_match as the authoritative source since it represents matches from
    # the ground truth perspective, which is more appropriate for recall calculation.
    # If ground_match and generated_match differ, it indicates a potential issue with
    # whitespace handling, but we use ground_match to maintain consistency with the
    # ground truth perspective.
    counts = {
        "match": ground_match,
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
    top_p=0.9,
    base_url: Optional[str] = None,
    *,
    progress_message: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    stop_sequences: Optional[List[str]] = None,
    return_logprobs: bool = False,
    request_timeout: Optional[float] = None,
    request_max_retries: Optional[int] = None,
) -> Any:
    """
    Gets a completion from the specified LLM.
    
    Args:
        prompt: The prompt to send to the LLM.
        api_key: API key for the provider.
        model_name: Name of the model to use.
    provider: LLM provider ("OpenAI", "OpenRouter", "Anthropic", "Google Gemini", "Kimi", "Local vLLM").
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        base_url: Optional base URL for OpenAI-compatible endpoints (used for Local vLLM).
        progress_message: Optional progress message.
        max_output_tokens: Maximum tokens to generate.
        stop_sequences: Stop sequences.
        return_logprobs: If True, return (text, logprobs_data) tuple. Only works for OpenAI/OpenRouter.
    
        request_timeout: Optional provider request timeout in seconds.
        request_max_retries: Optional OpenAI SDK retry limit.
    Returns:
        If return_logprobs is False: str (generated text or error message)
        If return_logprobs is True: Tuple[str, Optional[List[Dict]]] (text, logprobs data)
    """
    label_placeholder, bar_placeholder, progress_bar = start_llm_progress(
        progress_message or f"Calling {provider} · {model_name}"
    )
    update_llm_progress(progress_bar, value=15)

    missing_key = _require_api_key(api_key, provider)
    if missing_key:
        complete_llm_progress(
            label_placeholder,
            bar_placeholder,
            progress_bar,
            final_message="Missing API key",
            success=False,
            linger=0.5,
        )
        if return_logprobs:
            return missing_key, None
        return missing_key

    try:
        with limit_api_concurrency():
            return _execute_llm_completion(
                prompt=prompt,
                api_key=api_key,
                model_name=model_name,
                provider=provider,
                temperature=temperature,
                top_p=top_p,
                base_url=base_url,
                max_output_tokens=max_output_tokens,
                stop_sequences=stop_sequences,
                return_logprobs=return_logprobs,
                request_timeout=request_timeout,
                request_max_retries=request_max_retries,
                label_placeholder=label_placeholder,
                bar_placeholder=bar_placeholder,
                progress_bar=progress_bar,
            )
    except ApiConcurrencyTimeout as exc:
        complete_llm_progress(
            label_placeholder,
            bar_placeholder,
            progress_bar,
            final_message="API concurrency limit reached",
            success=False,
            linger=0.6,
        )
        error_msg = str(exc)
        if return_logprobs:
            return error_msg, None
        return error_msg


def _execute_llm_completion(
    *,
    prompt,
    api_key,
    model_name,
    provider,
    temperature,
    top_p,
    base_url,
    max_output_tokens,
    stop_sequences,
    return_logprobs,
    request_timeout,
    request_max_retries,
    label_placeholder,
    bar_placeholder,
    progress_bar,
) -> Any:
    logprobs_data: Optional[List[Dict[str, Any]]] = None
    try:
        if provider == "OpenAI":
            client_options: Dict[str, Any] = {"api_key": str(api_key).strip()}
            if request_timeout is not None:
                client_options["timeout"] = max(1.0, float(request_timeout))
            if request_max_retries is not None:
                client_options["max_retries"] = max(
                    0, int(request_max_retries)
                )
            client = openai.OpenAI(**client_options)
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
            if return_logprobs:
                request_kwargs["logprobs"] = True
                request_kwargs["top_logprobs"] = 5
            response = create_openai_chat_completion(client, request_kwargs)
            result_text = _extract_chat_message_text(response)
            
            if return_logprobs:
                logprobs_data = _extract_logprobs_from_response(response)
        
        elif provider == "OpenRouter":
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            messages = _build_messages_for_model(
                model_name,
                "You are a helpful assistant.",
                prompt
            )
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
            # Request logprobs if needed
            if return_logprobs:
                request_kwargs["logprobs"] = True
                request_kwargs["top_logprobs"] = 5
            
            # Try the original model first
            try:
                response = create_openai_chat_completion(client, request_kwargs)
            except Exception as e:
                # Handle 429 rate limit error for gemma-4-31b by falling back to gemma-4-26b
                error_str = str(e)
                if "429" in error_str and "gemma-4-31b" in model_name.lower():
                    # Automatically switch to gemma-4-26b as fallback
                    fallback_model = "google/gemma-4-26b-a4b-it:free"
                    request_kwargs["model"] = fallback_model
                    # Rebuild messages for the fallback model
                    messages = _build_messages_for_model(
                        fallback_model,
                        "You are a helpful assistant.",
                        prompt
                    )
                    request_kwargs["messages"] = messages
                    # Retry with fallback model
                    response = create_openai_chat_completion(client, request_kwargs)
                else:
                    # Re-raise if it's not a 429 for gemma-4-31b
                    raise
            
            result_text = _extract_chat_message_text(response)
            
            # Extract logprobs if requested
            if return_logprobs:
                logprobs_data = _extract_logprobs_from_response(response)
        
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
            # Anthropic doesn't support logprobs in the same way
        
        elif provider == "Google Gemini":
            client = genai.Client(api_key=api_key)
            normalized_model = _normalize_gemini_model(model_name)
            config = {}
            if temperature is not None:
                config["temperature"] = temperature
            if top_p is not None:
                config["top_p"] = top_p
            if max_output_tokens is not None:
                config["max_output_tokens"] = max_output_tokens
            if stop_sequences:
                config["stop_sequences"] = stop_sequences
            response = client.models.generate_content(
                model=normalized_model,
                contents=prompt,
                config=config or None
            )
            result_text = response.text.strip()
            # Google Gemini doesn't support logprobs in the same way

        elif provider == "Kimi":
            # Kimi (Moonshot) exposes an OpenAI-compatible endpoint
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.moonshot.cn/v1",
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            kimi_temperature, kimi_top_p = normalize_kimi_sampling_params(
                model_name, temperature, top_p
            )
            request_kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": kimi_temperature,
                "top_p": kimi_top_p,
            }
            if max_output_tokens is not None:
                request_kwargs["max_tokens"] = max_output_tokens
            if stop_sequences:
                request_kwargs["stop"] = stop_sequences

            # Kimi currently does not expose logprobs; ignore if requested
            response = create_openai_chat_completion(client, request_kwargs)
            result_text = _extract_chat_message_text(response)
            if return_logprobs:
                logprobs_data = _extract_logprobs_from_response(response)

        elif provider == "Local vLLM":
            # Local vLLM via OpenAI-compatible endpoint
            resolved_base = base_url or st.session_state.get("sidebar_local_vllm_base_url", "http://localhost:8000/v1")
            resolved_key = api_key or st.session_state.get("sidebar_local_vllm_api_key", "")
            client = openai.OpenAI(
                api_key=resolved_key or None,
                base_url=resolved_base,
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
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

            # Many local vLLM deployments may not support logprobs; attempt only if requested
            if return_logprobs:
                request_kwargs["logprobs"] = True
                request_kwargs["top_logprobs"] = 5
            response = create_openai_chat_completion(client, request_kwargs)
            result_text = _extract_chat_message_text(response)
            if return_logprobs:
                logprobs_data = _extract_logprobs_from_response(response)
        
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
            if return_logprobs:
                return error_message, None
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
        error_msg = f"Error calling API: {e}"
        if return_logprobs:
            return error_msg, None
        return error_msg

    update_llm_progress(progress_bar, value=80)
    complete_llm_progress(
        label_placeholder,
        bar_placeholder,
        progress_bar,
        final_message=f"LLM request completed ({provider} · {model_name})",
        success=True,
    )
    
    if return_logprobs:
        return result_text, logprobs_data
    return result_text


def _extract_logprobs_from_response(response) -> Optional[List[Dict[str, Any]]]:
    """Extract logprobs data from OpenAI-compatible API response.
    
    Returns a list of dicts with 'token', 'logprob', and 'linear_prob' keys.
    """
    try:
        logprobs_content = response.choices[0].logprobs
        if not logprobs_content or not hasattr(logprobs_content, 'content') or not logprobs_content.content:
            return None
        
        result = []
        for token_info in logprobs_content.content:
            logprob = token_info.logprob
            linear_prob = math.exp(logprob) if logprob > -100 else 0.0
            result.append({
                "token": token_info.token,
                "logprob": logprob,
                "linear_prob": linear_prob,
            })
        return result
    except Exception:
        return None


def _truncate_logprobs_to_text(
    logprobs_data: List[Dict[str, Any]], 
    target_text: str
) -> List[Dict[str, Any]]:
    """Truncate logprobs data to match the truncated generated text.
    
    This ensures that the logprobs data corresponds exactly to the text
    that is displayed to the user after any truncation operations.
    
    Args:
        logprobs_data: List of dicts with 'token', 'logprob', and 'linear_prob' keys.
        target_text: The truncated/processed text to match.
    
    Returns:
        Truncated logprobs data that matches the target text.
    """
    if not logprobs_data or not target_text:
        return logprobs_data
    
    # Reconstruct text from tokens and find where to truncate
    reconstructed = ""
    truncate_index = len(logprobs_data)
    
    for i, token_data in enumerate(logprobs_data):
        token = token_data.get("token", "")
        reconstructed += token
        
        # Check if we've reached or exceeded the target text length
        # We need to normalize both to handle whitespace differences
        normalized_reconstructed = _normalize_spaces(reconstructed)
        if len(normalized_reconstructed) >= len(target_text):
            truncate_index = i + 1
            break
    
    return logprobs_data[:truncate_index]


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
    top_p=0.9,
    continuation_method="Normal Continuation",
    custom_template: Optional[str] = None,
    mode: str = "Zero-Shot",
    target_word_count: Optional[int] = None,
    extra_prompt_instructions: Optional[str] = None,
    return_logprobs: bool = False,
):
    """
    Generates text based on the input_text according to prompt_type and compares it to reference_text.
    
    Args:
        prompt_type choices:
          - "Next-Passage Prediction": continue from the given prefix (input_text)
          - "Direct Probing": directly ask the model to retrieve specific content from a book (input_text is the book title)
        continuation_method selects the strategy template for prompts.
        return_logprobs: If True, also return logprobs data (only works for OpenAI/OpenRouter).

    Returns:
        If return_logprobs is False:
            Tuple ``(generated_text, metrics)`` where ``metrics`` is a dictionary containing
            ROUGE-1, ROUGE-L, character/word LCS (length + ratio), ACS (word), Levenshtein distance,
            semantic similarity, MinHash similarity, and the Jaccard index.
        If return_logprobs is True:
            Tuple ``(generated_text, metrics, logprobs_data)`` where logprobs_data is a list of
            dicts with 'token', 'logprob', and 'linear_prob' keys, or None if unavailable.
    """
    # Determine target length for generation
    if reference_text and prompt_type != "Direct Probing":
        target_char_count = len(reference_text)
    else:
        # For Direct Probing, use reference text length if available, otherwise default
        target_char_count = len(reference_text) if reference_text else 500  # Default for direct probing

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
    elif prompt_type == "User-Defined Evaluation":
        # For user-defined evaluation, use the custom template directly as the prompt
        prompt = custom_template if custom_template else ""
    elif prompt_type == "Direct Probing":
        # For Direct Probing, input_text could be either:
        # 1. A complete prompt (from examples) - contains phrases like "Find the first 100 words"
        # 2. A book title (from custom input) - needs to generate prompt from template
        if any(phrase in input_text for phrase in [
            "Find the first 100 words",
            "Extract and list the first 100 words",
            "Reproduce verbatim",
            "Could you retrieve",
            "I need a verbatim extraction",
            "Give me the first 100 words"
        ]):
            # input_text is already a complete prompt
            prompt = input_text
        else:
            # input_text is a book title, generate prompt from template
            prompt = get_full_prompt(
                prompt_type,
                input_text,  # This is the book title
                chunk_size=chunk_size,
                char_count=target_char_count,
            )
    else:
        # Fallback for unknown prompt types
        prompt = input_text

    if extra_prompt_instructions:
        prompt = f"{prompt}\n\n{extra_prompt_instructions.strip()}"

    # Call LLM with optional logprobs
    logprobs_data = None
    if return_logprobs:
        result = get_llm_completion(
            prompt, api_key, model_name, provider, 
            temperature=temperature, top_p=top_p,
            return_logprobs=True
        )
        if isinstance(result, tuple):
            generated_text, logprobs_data = result
        else:
            generated_text = result
    else:
        generated_text = get_llm_completion(
            prompt, api_key, model_name, provider, 
            temperature=temperature, top_p=top_p
        )
    
    # Return early if API error to avoid post-processing masking the error message
    if isinstance(generated_text, str) and generated_text.startswith("Error"):
        if return_logprobs:
            return generated_text, None, None
        return generated_text, None

    # For Direct Probing, use the full output without truncation
    if prompt_type != "Direct Probing":
        # Enforce exact character length to match the ground truth
        generated_text = enforce_exact_char_count(
            generated_text,
            target_char_count,
        )

        generated_text = enforce_exact_word_count(
            generated_text,
            target_word_count,
        )

    # Truncate logprobs to match the truncated generated_text
    if logprobs_data and generated_text:
        logprobs_data = _truncate_logprobs_to_text(logprobs_data, generated_text)

    # If a reference/target text is provided, compute similarity metrics; otherwise return zeros.
    metrics = (
        calculate_similarity_metrics(reference_text, generated_text)
        if reference_text
        else None
    )

    if return_logprobs:
        return generated_text, metrics, logprobs_data
    return generated_text, metrics


def load_literal_examples(indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Load predefined text examples from data.literal.json for Next-Passage Prediction.
    
    Args:
        indices: Optional list of example indices to load (0-based). If None, load all.
        
    Returns:
        List of dictionaries with 'id', 'title', 'input', and 'reference' keys
    """
    # Get the path to data.literal.json relative to this file
    current_dir = Path(__file__).parent
    json_path = current_dir / "data.literal.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    for idx, item in enumerate(data):
        # Skip if indices specified and current index not in the list
        if indices is not None and idx not in indices:
            continue
        
        example = {
            "id": item.get("id", ""),
            "title": item.get("title", ""),
            "input": item.get("input", ""),
            "reference": item.get("reference", ""),
            "index": idx + 1  # 1-based indexing for display
        }
        examples.append(example)
    
    return examples


def parse_example_indices(indices_str: str) -> List[int]:
    """Parse example indices string into a list of integers.
    
    Supports formats like:
    - "1,5,10" (individual indices)
    - "10-15" (ranges)
    - "1,5,10-15,20" (mixed)
    
    Args:
        indices_str: String containing indices specification
        
    Returns:
        List of 0-based indices
        
    Raises:
        ValueError: If the format is invalid
    """
    indices = []
    parts = indices_str.split(',')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if '-' in part:
            # Handle range
            range_parts = part.split('-')
            if len(range_parts) != 2:
                raise ValueError(f"Invalid range format: {part}")
            
            try:
                start = int(range_parts[0].strip())
                end = int(range_parts[1].strip())
                if start > end:
                    raise ValueError(f"Range start ({start}) must be <= end ({end})")
                # Convert to 0-based indices
                indices.extend(range(start - 1, end))
            except ValueError as e:
                raise ValueError(f"Invalid range values in '{part}': {e}")
        else:
            # Handle single index
            try:
                idx = int(part)
                if idx < 1:
                    raise ValueError(f"Index must be >= 1, got {idx}")
                # Convert to 0-based index
                indices.append(idx - 1)
            except ValueError as e:
                raise ValueError(f"Invalid index '{part}': {e}")
    
    return sorted(set(indices))  # Remove duplicates and sort
