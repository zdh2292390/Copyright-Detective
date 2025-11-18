"""Single-choice question generation and evaluation utilities."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from openai import OpenAI
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

from src.direct_recall.comparison import get_llm_completion
from src.direct_recall.pdf_utils import extract_text_from_document

REPO_ROOT = Path(__file__).resolve().parents[2]
DECOP_DATA_DIR = REPO_ROOT / "src" / "direct_recall" / "decop" / "data"
DECOP_QA_PROMPT = (
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
    temperature: float = 0.5,
    top_p: float = 0.8,
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


def generate_single_choice_questions_from_document(
    document_file,
    api_key: str,
    model_choice: str,
    provider: str,
    num_questions: int = 5,
    temperature: float = 0.5,
    top_p: float = 0.8,
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

    data_path = DECOP_DATA_DIR / f"{dataset_name}.csv"
    if not data_path.exists():
        return []

    try:
        df = pd.read_csv(data_path, usecols=["ID"])
    except Exception:
        return []

    doc_ids = sorted(df["ID"].dropna().unique().tolist())
    if limit:
        doc_ids = doc_ids[:limit]
    return doc_ids


@lru_cache(maxsize=64)
def load_dataset_excerpt(
    dataset_name: str,
    document_id: Optional[str] = None,
    max_rows: int = 3,
) -> Tuple[str, Dict[str, Any]]:
    """Load a short excerpt from the DECOP datasets for question generation."""

    data_path = DECOP_DATA_DIR / f"{dataset_name}.csv"
    if not data_path.exists():
        return "", {}

    try:
        df = pd.read_csv(data_path)
    except Exception:
        return "", {}

    doc_df = df
    if document_id and document_id in df["ID"].values:
        doc_df = df[df["ID"] == document_id]
    else:
        doc_df = df.sample(n=1, random_state=42)

    doc_df = doc_df.head(max_rows)

    excerpts: List[str] = []
    for _, row in doc_df.iterrows():
        passages = [row.get(col, "") for col in ["Example_A", "Example_B"]]
        passages = [_clean_text(p) for p in passages if isinstance(p, str) and p.strip()]
        if passages:
            excerpts.append(" ".join(passages))

    combined_text = "\n\n".join(excerpts)
    meta = {
        "document_id": doc_df.iloc[0].get("ID", ""),
        "label": doc_df.iloc[0].get("Label", ""),
        "length": doc_df.iloc[0].get("Length", ""),
    }

    return combined_text, meta


def _build_decop_prompt_body(question: Dict[str, Any]) -> Tuple[str, str]:
    """Create DECOP-style prompt body for both OpenAI and Anthropic evaluators.

    The downstream LLM must *never* see the source passage. We therefore reuse the
    exact instructions from ``src/direct_recall/decop/2_decop_blackbox.py`` so the
    evaluator only receives the synthetic question plus four options and is forced
    to reply with a single letter.
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

    openai_prompt = f"{DECOP_QA_PROMPT} {question_block}{ANSWER_PREFIX} "
    anthropic_prompt = f"{HUMAN_PROMPT} {DECOP_QA_PROMPT} {question_block}{AI_PROMPT} {ANSWER_PREFIX} "
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

    try:
        client = OpenAI(**client_kwargs)
        response = client.completions.create(**request_kwargs)
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
        completion = client.completions.create(
            model=model_choice,
            prompt=prompt,
            max_tokens_to_sample=1,
            temperature=0,
        )
    except Exception:
        return None

    token_text = (completion.completion or "").strip()
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
        temperature=0.0,
        top_p=1.0,
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
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Dict[str, Any]:
    openai_prompt, anthropic_prompt = _build_decop_prompt_body(question)

    if provider in {"OpenAI", "OpenRouter"}:
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
    temperature: float = 0.0,
    top_p: float = 1.0,
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
