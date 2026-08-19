"""Single-choice question generation and evaluation utilities."""

from __future__ import annotations

import json
import math
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from openai import OpenAI
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

from src.direct_recall.comparison import get_llm_completion
from src.kimi_utils import normalize_kimi_sampling_params
from src.openai_utils import apply_openai_request_compat
from src.direct_recall.pdf_utils import extract_text_from_document

REPO_ROOT = Path(__file__).resolve().parents[2]
SC_DATA_DIR = REPO_ROOT / "src" / "direct_recall" / "decop" / "data"
SC_QA_PROMPT = (
    "You are entering a multiple choice questions exam. You should directly answer each question by choosing "
    "the correct option. Be concise and straight to the point in your answer. Output only the letter corresponding "
    "to the correct answer."
)
ANSWER_PREFIX = "Answer:"
OPTION_LABELS = ["A", "B", "C", "D"]
LETTER_PATTERN = re.compile(r"[A-D]", re.IGNORECASE)


def _clean_text(value: Any) -> str:
    return str(value).strip()


def _normalize_options(raw_options: Any) -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = []

    if isinstance(raw_options, dict):
        for key in sorted(raw_options.keys()):
            label = key.strip().upper()[:1]
            text = _clean_text(raw_options[key])
            if label and text:
                options.append({"label": label, "text": text})
    elif isinstance(raw_options, list):
        for idx, option in enumerate(raw_options):
            label = OPTION_LABELS[idx] if idx < len(OPTION_LABELS) else None
            text = ""
            if isinstance(option, dict):
                if "label" in option:
                    label = str(option["label"]).strip().upper()[:1] or label
                text = option.get("text") or option.get("option") or option.get("content") or ""
            else:
                text = option
            text = _clean_text(text)
            if label and text:
                options.append({"label": label, "text": text})
    else:
        return []

    # Keep only the first four labeled options A-D
    filtered: List[Dict[str, str]] = []
    seen_labels = set()
    for option in options:
        label = option["label"].upper()
        if label in OPTION_LABELS and label not in seen_labels:
            filtered.append({"label": label, "text": option["text"]})
            seen_labels.add(label)
        if len(filtered) == 4:
            break

    return filtered if len(filtered) == 4 else []


def _normalize_mcq_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    question = _clean_text(entry.get("question", ""))
    if not question:
        return None

    options = _normalize_options(entry.get("options") or entry.get("choices"))
    if not options:
        return None

    correct_option = str(entry.get("correct_option") or entry.get("answer") or "").strip().upper()
    if correct_option.startswith("OPTION "):
        correct_option = correct_option.split()[-1]
    correct_option = correct_option[:1]

    if correct_option not in OPTION_LABELS:
        answer_text = entry.get("correct_answer") or entry.get("correct_answer_text")
        if answer_text:
            answer_text_normalized = _clean_text(answer_text).lower()
            for option in options:
                if option["text"].lower() == answer_text_normalized:
                    correct_option = option["label"]
                    break

    if correct_option not in OPTION_LABELS:
        return None

    explanation = _clean_text(entry.get("explanation", "")) or None

    return {
        "question": question,
        "options": options,
        "correct_option": correct_option,
        "explanation": explanation,
    }


def _extract_json_array(response_text: str) -> Optional[List[Any]]:
    response_text = response_text.strip()
    start_idx = response_text.find("[")
    end_idx = response_text.rfind("]") + 1
    if start_idx == -1 or end_idx <= start_idx:
        return None

    json_str = response_text[start_idx:end_idx]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def generate_single_choice_questions_from_text(
    text: str,
    api_key: str,
    model_choice: str,
    provider: str,
    num_questions: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[Dict[str, Any]]:
    """Generate multiple single-choice questions from raw text."""

    prompt = f"""You are constructing a copyright-detection multiple-choice exam.
Given the source text below, craft EXACTLY {num_questions} single-choice questions that probe for
verbatim memorization. Requirements:
- Each question must have four options (A, B, C, D) with nearly identical structure and only subtle keyword differences.
- Only one option should be correct and supported by the source text.
- Provide an optional explanation referencing the text.

Return the questions as pure JSON using the format:
[
  {{
    "question": "...",
    "options": [
      {{"label": "A", "text": "..."}},
      {{"label": "B", "text": "..."}},
      {{"label": "C", "text": "..."}},
      {{"label": "D", "text": "..."}}
    ],
    "correct_option": "B",
    "explanation": "..."
  }}
]

Source text:
{text}
"""

    response = get_llm_completion(
        prompt,
        api_key,
        model_choice,
        provider,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=2500,
    )

    if isinstance(response, str) and response.startswith("Error"):
        return []

    if not isinstance(response, str):
        return []

    data = _extract_json_array(response)
    if not isinstance(data, list):
        return []

    mcq_bank: List[Dict[str, Any]] = []
    for entry in data:
        if isinstance(entry, dict):
            normalized = _normalize_mcq_entry(entry)
            if normalized:
                mcq_bank.append(normalized)
        if len(mcq_bank) == num_questions:
            break

    return mcq_bank


def extract_text_fragments(text: str, fragment_size: int = 30, num_fragments: int = 5) -> List[str]:
    """Extract text fragments of specified word count from the input text."""
    words = text.split()
    if len(words) < fragment_size:
        return [text]  # Return the whole text if it's shorter than fragment size
    
    fragments = []
    step = max(1, (len(words) - fragment_size) // max(1, num_fragments - 1))
    
    for i in range(0, min(len(words) - fragment_size + 1, num_fragments * step), step):
        fragment = " ".join(words[i:i + fragment_size])
        fragments.append(fragment)
        if len(fragments) >= num_fragments:
            break
    
    return fragments


def generate_distractors_for_fragment(
    correct_fragment: str,
    api_key: str,
    model_choice: str,
    provider: str,
    num_distractors: int = 3,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[str]:
    """Generate distractor options for a correct text fragment."""
    
    prompt = f"""Given the correct text fragment below, generate EXACTLY {num_distractors} distractor options that are very similar but incorrect. The distractors should:

1. Have nearly identical word count and structure
2. Contain subtle but meaningful differences
3. Be plausible but not verbatim from any real source
4. Maintain similar vocabulary and style

Correct fragment: "{correct_fragment}"

Return ONLY a JSON array of strings, like: ["distractor 1", "distractor 2", "distractor 3"]
"""

    response = get_llm_completion(
        prompt,
        api_key,
        model_choice,
        provider,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=1000,
    )

    if isinstance(response, str) and response.startswith("Error"):
        return []

    if not isinstance(response, str):
        return []

    # Try to extract JSON array
    start_idx = response.find("[")
    end_idx = response.rfind("]") + 1
    if start_idx == -1 or end_idx <= start_idx:
        return []

    json_str = response[start_idx:end_idx]
    try:
        distractors = json.loads(json_str)
        if isinstance(distractors, list):
            return [str(d) for d in distractors[:num_distractors]]
    except json.JSONDecodeError:
        pass

    return []


def generate_single_choice_questions_from_fragments(
    text: str,
    api_key: str,
    model_choice: str,
    provider: str,
    num_questions: int = 5,
    fragment_size: int = 30,
    num_distractors: int = 3,
    temperature: float = 0.7,
    top_p: float = 0.9,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """Generate single-choice questions by extracting text fragments and creating distractors."""
    
    # Extract text fragments
    fragments = extract_text_fragments(text, fragment_size, num_questions)
    if not fragments:
        return []
    
    mcq_bank: List[Dict[str, Any]] = []
    total_operations = len(fragments) * (num_distractors + 1)  # +1 for fragment extraction
    current_operation = 0
    
    for i, correct_fragment in enumerate(fragments):
        # Generate distractors for this fragment
        distractors = generate_distractors_for_fragment(
            correct_fragment,
            api_key,
            model_choice,
            provider,
            num_distractors=num_distractors,
            temperature=temperature,
            top_p=top_p,
        )
        
        current_operation += num_distractors
        if progress_callback:
            progress_callback(current_operation, total_operations, i + 1)
        
        if len(distractors) < num_distractors:
            continue  # Skip if we couldn't generate enough distractors
        
        # Create the question
        question_text = f"Which of the following passages is verbatim from the source text?"
        
        # Combine correct answer and distractors, then shuffle with synced label.
        provisional_options = [
            {"label": OPTION_LABELS[j], "text": text}
            for j, text in enumerate([correct_fragment] + distractors)
            if j < len(OPTION_LABELS)
        ]
        options, correct_label = _shuffle_mcq_options(
            provisional_options,
            "A",  # correct_fragment is placed first above
            rng=random.Random(42 + i),
        )
        
        mcq = {
            "question": question_text,
            "options": options,
            "correct_option": correct_label,
            "explanation": f"The correct option matches the verbatim text from the source.",
            "source_fragment": correct_fragment
        }
        
        mcq_bank.append(mcq)
        current_operation += 1
        if progress_callback:
            progress_callback(current_operation, total_operations, i + 1)
        
        if len(mcq_bank) >= num_questions:
            break
    
    return mcq_bank


def generate_single_choice_questions_from_document_fragments(
    document_file,
    api_key: str,
    model_choice: str,
    provider: str,
    num_questions: int = 5,
    fragment_size: int = 30,
    num_distractors: int = 3,
    temperature: float = 0.7,
    top_p: float = 0.9,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Extract text from a document and generate single-choice questions from fragments."""

    text = extract_text_from_document(document_file)
    if isinstance(text, str) and text.startswith("Error"):
        return [], text

    words = text.split()
    if len(words) > 3500:
        text = " ".join(words[:3500])

    questions = generate_single_choice_questions_from_fragments(
        text,
        api_key,
        model_choice,
        provider,
        num_questions=num_questions,
        fragment_size=fragment_size,
        num_distractors=num_distractors,
        temperature=temperature,
        top_p=top_p,
        progress_callback=progress_callback,
    )

    return questions, text


def generate_single_choice_questions_from_document(
    document_file,
    api_key: str,
    model_choice: str,
    provider: str,
    num_questions: int = 5,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[List[Dict[str, Any]], str]:
    """Extract text from a document and generate single-choice questions."""

    text = extract_text_from_document(document_file)
    if isinstance(text, str) and text.startswith("Error"):
        return [], text

    words = text.split()
    if len(words) > 3500:
        text = " ".join(words[:3500])

    questions = generate_single_choice_questions_from_text(
        text,
        api_key,
        model_choice,
        provider,
        num_questions=num_questions,
        temperature=temperature,
        top_p=top_p,
    )

    return questions, text


@lru_cache(maxsize=16)
def list_dataset_documents(dataset_name: str, limit: int = 50) -> List[str]:
    """List a subset of document IDs available in a dataset."""

    data_path = SC_DATA_DIR / f"{dataset_name}.csv"
    if not data_path.exists():
        return []

    try:
        df = pd.read_csv(data_path, usecols=["ID"])
    except Exception:
        return []

    doc_ids = sorted(df["ID"].dropna().astype(str).str.strip().unique().tolist())
    if limit:
        doc_ids = doc_ids[:limit]
    return doc_ids


@lru_cache(maxsize=64)
def load_dataset_excerpt(
    dataset_name: str,
    document_id: Optional[str] = None,
    max_rows: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Load a dataset document for question generation.

    By default this function returns the full document (all rows matching
    `document_id` or the sampled document). If `max_rows` is set to an int,
    only the first `max_rows` rows will be returned (backwards-compatible
    truncation control). If `max_chars` is set, the combined text will be
    truncated to that many characters.
    """

    data_path = SC_DATA_DIR / f"{dataset_name}.csv"
    if not data_path.exists():
        return "", {}

    try:
        df = pd.read_csv(data_path)
    except Exception:
        return "", {}

    df["ID"] = df["ID"].astype(str).str.strip()

    doc_df = df
    if document_id and document_id in df["ID"].values:
        doc_df = df[df["ID"] == document_id]
    else:
        doc_df = df.sample(n=1, random_state=42)

    # If max_rows is provided, keep only the first `max_rows` rows. If
    # max_rows is None (the default), return the full document.
    if max_rows is not None:
        doc_df = doc_df.head(max_rows)

    example_columns = [col for col in doc_df.columns if col and col.lower().startswith("example")]
    if not example_columns:
        example_columns = ["Example_A", "Example_B"]

    row_texts: List[str] = []
    for _, row in doc_df.iterrows():
        passages: List[str] = []
        for column in example_columns:
            value = row.get(column, "")
            if isinstance(value, str) and value.strip():
                passages.append(_clean_text(value))
        if passages:
            row_texts.append(" ".join(passages))

    rows_available = len(row_texts)
    rows_returned = 0
    partial_row = False
    was_truncated = False

    if max_chars is not None and max_chars > 0:
        remaining = max_chars
        chunks: List[str] = []
        for row_text in row_texts:
            if remaining <= 0:
                was_truncated = True
                break
            addition = row_text if not chunks else f"\n\n{row_text}"
            addition_len = len(addition)
            if addition_len <= remaining:
                chunks.append(addition)
                remaining -= addition_len
                rows_returned += 1
            else:
                if remaining > 0:
                    chunks.append(addition[:remaining])
                    rows_returned += 1
                    partial_row = True
                was_truncated = True
                remaining = 0
                break
        combined_text = "".join(chunks)
        if remaining == 0 and rows_returned < rows_available:
            was_truncated = True
    else:
        combined_text = "\n\n".join(row_texts)
        rows_returned = rows_available

    chars_returned = len(combined_text)

    # Build simple metadata from the first row but include details so callers
    # know if they received the full document or a truncated slice.
    meta = {
        "document_id": doc_df.iloc[0].get("ID", "") if not doc_df.empty else "",
        "label": doc_df.iloc[0].get("Label", "") if not doc_df.empty else "",
        "length": doc_df.iloc[0].get("Length", "") if not doc_df.empty else "",
        "rows_sampled": len(doc_df),
        "rows_available": rows_available,
        "rows_returned": rows_returned,
        "chars_returned": chars_returned,
        "was_truncated": was_truncated,
        "partial_row": partial_row,
        "max_rows": max_rows,
        "max_chars": max_chars,
    }

    return combined_text, meta


def _build_sc_prompt_body(question: Dict[str, Any]) -> Tuple[str, str]:
    """Create single-choice prompt body for both OpenAI and Anthropic evaluators.

    The downstream LLM must *never* see the source passage. We therefore follow
    standard multiple-choice exam format where the evaluator only receives the 
    synthetic question plus four options and is forced to reply with a single letter.
    """

    prompt_question = _clean_text(question.get("question", "")) or (
        "Which of the following options best matches the reference text?"
    )
    option_lines = []
    for option in question.get("options", []):
        label = option.get("label") or ""
        text = _clean_text(option.get("text", ""))
        if not label or not text:
            continue
        option_lines.append(f"{label}. {text}")

    options_block = "\n".join(option_lines)
    question_block = f"Question: {prompt_question}\nOptions:\n{options_block}\n"

    openai_prompt = f"{SC_QA_PROMPT} {question_block}{ANSWER_PREFIX} "
    anthropic_prompt = f"{HUMAN_PROMPT} {SC_QA_PROMPT} {question_block}{AI_PROMPT} {ANSWER_PREFIX} "
    return openai_prompt, anthropic_prompt


def _extract_option_from_text(text: str) -> str:
    if not text:
        return ""
    match = LETTER_PATTERN.search(text)
    return match.group(0).upper() if match else ""


def _normalize_option_probabilities(raw: Dict[str, float]) -> Optional[Dict[str, float]]:
    if not raw:
        return None
    total = sum(value for value in raw.values() if value is not None)
    if total <= 0:
        return None
    return {label: value / total for label, value in raw.items() if value is not None}


def _parse_openai_top_logprobs(top_logprobs: Any) -> Optional[Dict[str, float]]:
    if not top_logprobs:
        return None

    entries: List[Any] = []
    first_entry = top_logprobs[0] if isinstance(top_logprobs, (list, tuple)) else top_logprobs
    # OpenAI returns a list of dict-like objects per token step
    if isinstance(first_entry, (list, tuple)):
        entries = list(first_entry)
    elif isinstance(first_entry, dict):
        entries = [first_entry]
    else:
        entries = [first_entry]

    option_scores: Dict[str, float] = {}
    for entry in entries:
        token = getattr(entry, "token", None)
        if token is None and isinstance(entry, dict):
            token = entry.get("token") or entry.get("text")
        if not token:
            continue
        candidate = token.strip().upper()
        if candidate not in OPTION_LABELS:
            continue
        logprob = getattr(entry, "logprob", None)
        if logprob is None and isinstance(entry, dict):
            logprob = entry.get("logprob")
        if logprob is None:
            continue
        option_scores[candidate] = math.exp(float(logprob))

    return _normalize_option_probabilities(option_scores)


def _try_openai_style_completion(
    prompt: str,
    api_key: str,
    model_choice: str,
    provider: str,
    temperature: float,
    top_p: float,
):
    """Attempt to call the OpenAI-compatible completions API with logprobs."""

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    request_kwargs: Dict[str, Any] = {
        "model": model_choice,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "top_p": 1,
        "logprobs": 4,
        "stop": ["\n"],
    }

    if provider == "OpenRouter":
        client_kwargs["base_url"] = "https://openrouter.ai/api/v1"
        request_kwargs["extra_headers"] = {
            "HTTP-Referer": "http://localhost",
            "X-Title": "Copyright Detective",
        }
    elif provider == "Kimi":
        client_kwargs["base_url"] = "https://api.moonshot.cn/v1"
        kimi_temperature, kimi_top_p = normalize_kimi_sampling_params(
            model_choice,
            request_kwargs["temperature"],
            request_kwargs["top_p"],
        )
        request_kwargs["temperature"] = kimi_temperature
        request_kwargs["top_p"] = kimi_top_p

    request_kwargs = apply_openai_request_compat(request_kwargs)

    try:
        client = OpenAI(**client_kwargs)
        # Try the original model first
        try:
            response = client.completions.create(**request_kwargs)
        except Exception as e:
            # Handle 429 rate limit error for gemma-4-31b by falling back to gemma-4-26b
            error_str = str(e)
            if "429" in error_str and "gemma-4-31b" in model_choice.lower():
                # Automatically switch to gemma-4-26b as fallback
                fallback_model = "google/gemma-4-26b-a4b-it:free"
                request_kwargs["model"] = fallback_model
                # Retry with fallback model
                response = client.completions.create(**request_kwargs)
            else:
                # Re-raise if it's not a 429 for gemma-4-31b
                raise
    except Exception:
        return None

    choice = response.choices[0]
    token_text = (choice.text or "").strip()
    probabilities = None
    logprobs = getattr(choice, "logprobs", None)
    if logprobs and getattr(logprobs, "top_logprobs", None):
        probabilities = _parse_openai_top_logprobs(logprobs.top_logprobs)

    selected = _extract_option_from_text(token_text)
    if not selected and probabilities:
        selected = max(probabilities, key=probabilities.get)

    return {
        "choice": selected or "?",
        "option_probabilities": probabilities,
        "raw_response": token_text,
        "logit_mode": "logprobs" if probabilities else "text",
    }


def _try_anthropic_style_completion(
    prompt: str,
    api_key: str,
    model_choice: str,
) -> Optional[Dict[str, Any]]:
    client = Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model_choice,
            max_tokens=1,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
    except Exception:
        return None

    token_text = response.content[0].text.strip()
    selected = _extract_option_from_text(token_text)
    return {
        "choice": selected or "?",
        "option_probabilities": None,
        "raw_response": token_text,
        "logit_mode": "text",
    }


def _evaluate_with_basic_completion(
    prompt: str,
    api_key: str,
    model_choice: str,
    provider: str,
    temperature: float,
    top_p: float,
):
    response = get_llm_completion(
        prompt,
        api_key,
        model_choice,
        provider,
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=1,
        stop_sequences=["\n"],
        progress_message="Running single-choice evaluation",
    )

    text = response.strip() if isinstance(response, str) else ""
    return {
        "choice": _extract_option_from_text(text) or "?",
        "option_probabilities": None,
        "raw_response": text,
        "logit_mode": "text",
    }


def evaluate_single_choice_question(
    question: Dict[str, Any],
    api_key: str,
    model_choice: str,
    provider: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    openai_prompt, anthropic_prompt = _build_sc_prompt_body(question)

    if provider in {"OpenAI", "OpenRouter", "Kimi"}:
        result = _try_openai_style_completion(openai_prompt, api_key, model_choice, provider, temperature, top_p)
        if result:
            return result

    if provider == "Anthropic":
        result = _try_anthropic_style_completion(anthropic_prompt, api_key, model_choice)
        if result:
            return result

    return _evaluate_with_basic_completion(openai_prompt, api_key, model_choice, provider, temperature, top_p)


def run_single_choice_evaluation(
    questions: List[Dict[str, Any]],
    api_key: str,
    model_choice: str,
    provider: str,
    num_runs: int = 1,
    temperature: float = 0.7,
    top_p: float = 0.9,
    progress_callback: Optional[Callable[[int, int, int, int, int], None]] = None,
) -> List[List[Dict[str, Any]]]:
    """Evaluate generated MCQs using the target LLM."""

    all_results: List[List[Dict[str, Any]]] = []
    total_items = max(1, num_runs * len(questions))
    current = 0

    for run_idx in range(num_runs):
        run_results: List[Dict[str, Any]] = []
        for question_idx, mcq in enumerate(questions):
            evaluation = evaluate_single_choice_question(
                mcq,
                api_key,
                model_choice,
                provider,
                temperature=temperature,
                top_p=top_p,
            )
            choice = evaluation.get("choice", "?")

            result = {
                "question": mcq.get("question"),
                "options": mcq.get("options"),
                "correct_option": mcq.get("correct_option"),
                "llm_choice": choice,
                "is_correct": choice == mcq.get("correct_option"),
                "raw_response": evaluation.get("raw_response", ""),
                "option_probabilities": evaluation.get("option_probabilities"),
                "logit_mode": evaluation.get("logit_mode", "text"),
                "explanation": mcq.get("explanation"),
            }

            run_results.append(result)
            current += 1
            if progress_callback:
                progress_callback(current, total_items, run_idx + 1, question_idx + 1, len(questions))

        all_results.append(run_results)

    return all_results


def summarize_single_choice_results(all_results: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Aggregate accuracy and preference metrics across runs."""

    if not all_results:
        return {}

    total_attempts = sum(len(run) for run in all_results)
    if total_attempts == 0:
        return {}

    total_correct = sum(1 for run in all_results for item in run if item.get("is_correct"))
    option_distribution: Dict[str, int] = {label: 0 for label in OPTION_LABELS}
    option_distribution["?"] = 0

    correct_confidences: List[float] = []
    for run in all_results:
        for item in run:
            choice = item.get("llm_choice", "?")
            if choice not in option_distribution:
                option_distribution[choice] = 0
            option_distribution[choice] += 1
            probs = item.get("option_probabilities")
            if isinstance(probs, dict):
                confidence = probs.get(item.get("correct_option"), 0.0)
                if confidence is not None:
                    correct_confidences.append(float(confidence))

    max_questions = max(len(run) for run in all_results)
    per_question = []
    for idx in range(max_questions):
        attempts = 0
        correct = 0
        question_text = ""
        for run in all_results:
            if idx < len(run):
                attempts += 1
                question_text = run[idx].get("question", question_text)
                if run[idx].get("is_correct"):
                    correct += 1
        if attempts:
            per_question.append(
                {
                    "index": idx,
                    "question": question_text,
                    "accuracy": correct / attempts,
                    "attempts": attempts,
                }
            )

    return {
        "total_runs": len(all_results),
        "total_attempts": total_attempts,
        "overall_accuracy": total_correct / total_attempts,
        "option_distribution": option_distribution,
        "per_question": per_question,
        "avg_correct_confidence": (sum(correct_confidences) / len(correct_confidences))
        if correct_confidences
        else None,
    }


def _shuffle_mcq_options(
    options: List[Dict[str, str]],
    correct_option: str,
    *,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Dict[str, str]], str]:
    """Randomly reorder option texts and keep the correct answer letter in sync.

    Shuffling is done by option index (not text equality) so the correct label
    always tracks the same underlying choice after the permutation.
    """
    if not options:
        return options, correct_option

    labels = [str(option.get("label") or "").strip().upper()[:1] for option in options]
    texts = [str(option.get("text") or "") for option in options]
    correct_label = str(correct_option or "").strip().upper()[:1]
    try:
        correct_index = labels.index(correct_label)
    except ValueError:
        # DE-COP releases store the verbatim passage in Example_A by default.
        correct_index = 0

    order = list(range(len(texts)))
    (rng or random).shuffle(order)

    shuffled_options = [
        {"label": OPTION_LABELS[new_index], "text": texts[old_index]}
        for new_index, old_index in enumerate(order)
    ]
    new_correct_label = OPTION_LABELS[order.index(correct_index)]
    return shuffled_options, new_correct_label


def shuffle_tections_dataframe(
    df: pd.DataFrame,
    *,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Return a copy of a DE-COP CSV frame with options shuffled per row.

    Raw arXivTection / BookTection files keep the verbatim passage in Example_A
    (Answer=A). This helper permutes Example_A..D and rewrites Answer so UIs and
    exports are not dominated by letter-position bias.
    """
    if df is None or df.empty:
        return df

    example_cols = [f"Example_{label}" for label in OPTION_LABELS]
    if any(column not in df.columns for column in example_cols):
        return df.copy()

    shuffled = df.copy()
    base_seed = 0 if seed is None else int(seed)
    for position, (idx, row) in enumerate(shuffled.iterrows()):
        options = []
        for label in OPTION_LABELS:
            text = _clean_text(row.get(f"Example_{label}", ""))
            if text:
                options.append({"label": label, "text": text})
        if len(options) < 2:
            continue

        original_correct = _clean_text(row.get("Answer", "")).upper()[:1]
        if original_correct not in OPTION_LABELS:
            original_correct = options[0]["label"]

        rng = random.Random(base_seed + position * 97_003 + position * 17)
        shuffled_options, new_correct = _shuffle_mcq_options(
            options,
            original_correct,
            rng=rng,
        )
        for option in shuffled_options:
            shuffled.at[idx, f"Example_{option['label']}"] = option["text"]
        shuffled.at[idx, "Answer"] = new_correct
        if "True Answer" in shuffled.columns:
            shuffled.at[idx, "True Answer"] = new_correct

    return shuffled


def load_predefined_examples(
    dataset_name: str,
    indices: Optional[List[int]] = None,
    *,
    shuffle_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load predefined single-choice questions from CSV files for evaluation.

    Args:
        dataset_name: Name of the dataset ("arXivTection" or "BookTection")
        indices: Optional list of question indices to load (0-based). If None, load all.
        shuffle_seed: Optional seed controlling option order. When omitted, a fresh
            random seed is used so each Load reshuffles letter positions.

    Returns:
        List of MCQ dictionaries with question, options, correct_option, and label.
        Option order is randomly shuffled per question, with ``correct_option``
        updated to the new letter so evaluation stays synchronized.
    """
    csv_filename = f"{dataset_name}.csv"
    csv_path = SC_DATA_DIR / csv_filename

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Always shuffle before materializing MCQs. Raw CSVs park the verbatim
    # passage in Example_A (Answer=A), which creates letter-selection bias.
    effective_seed = (
        int(shuffle_seed)
        if shuffle_seed is not None
        else random.SystemRandom().randint(0, 2**31 - 1)
    )
    df = shuffle_tections_dataframe(df, seed=effective_seed)

    mcqs = []
    for idx, row in df.iterrows():
        # Skip if indices specified and current index not in the list
        if indices is not None and idx not in indices:
            continue

        # Extract options from columns Example_A, Example_B, Example_C, Example_D
        options = []
        for label in OPTION_LABELS:
            option_text = _clean_text(row.get(f"Example_{label}", ""))
            if option_text:
                options.append({"label": label, "text": option_text})

        if not options:
            continue

        correct_option = _clean_text(row.get("Answer", "")).upper()[:1]
        if correct_option not in OPTION_LABELS:
            correct_option = options[0]["label"]

        # Create MCQ format with dataset-specific question ID
        question_id = f"{dataset_name}_{idx + 1}"  # 1-based indexing for display
        mcq = {
            "question": question_id,  # Use formatted ID as question identifier
            "options": options,
            "correct_option": correct_option,
            "original_correct_option": "A",  # DE-COP source files always use A
            "original_id": _clean_text(row.get("ID", "")),
            "shuffle_seed": effective_seed,
        }

        mcqs.append(mcq)

    return mcqs


def get_predefined_examples_index() -> Dict[str, List[Dict[str, Any]]]:
    """Get index information for all predefined examples from CSV files.
    
    Returns:
        Dictionary with dataset names as keys and list of example info as values.
        Each example info contains: index, id, source_type, dataset
    """
    datasets = ["arXivTection", "BookTection"]
    index_info = {}
    
    for dataset_name in datasets:
        csv_filename = f"{dataset_name}.csv"
        csv_path = SC_DATA_DIR / csv_filename
        
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path)
        examples = []
        
        for idx, row in df.iterrows():
            # Determine source type based on dataset
            if dataset_name == "arXivTection":
                source_type = "arXiv Paper"
            elif dataset_name == "BookTection":
                source_type = "Book"
            else:
                source_type = "Unknown"
            
            example_info = {
                "index": idx + 1,  # 1-based indexing for display
                "id": row.get("ID", "").strip(),
                "source_type": source_type,
                "dataset": dataset_name
            }
            examples.append(example_info)
        
        index_info[dataset_name] = examples
    
    return index_info


def parse_question_indices(indices_str: str) -> List[int]:
    """Parse question indices string into a list of integers.
    
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
                    raise ValueError(f"Invalid range: start > end in {part}")
                # Convert to 0-based indexing
                indices.extend(range(start - 1, end))
            except ValueError:
                raise ValueError(f"Invalid range values: {part}")
        else:
            # Handle single index
            try:
                idx = int(part)
                indices.append(idx - 1)  # Convert to 0-based
            except ValueError:
                raise ValueError(f"Invalid index: {part}")
    
    # Remove duplicates and sort
    indices = sorted(list(set(indices)))
    
    # Validate indices are non-negative
    if any(idx < 0 for idx in indices):
        raise ValueError("Question indices must be positive")
        
    return indices
