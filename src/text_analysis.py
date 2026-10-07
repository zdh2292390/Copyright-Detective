"""Validation and repeated inference helpers for the text analysis page."""

from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Real


class TextAnalysisError(ValueError):
    """A text inference did not produce a usable comparison."""


def safe_numeric_setting(value, default, minimum, maximum, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        return default
    try:
        valid = math.isfinite(value) and minimum <= value <= maximum
    except (OverflowError, ValueError):
        valid = False
    if not valid or (integer and type(value) is not int):
        return default
    return value if integer else float(value)


def safe_selection_index(value, count):
    if type(value) is not int:
        return 0
    return max(0, min(value, count - 1))


def validate_text_parameters(inference_runs, temperature, top_p):
    if type(inference_runs) is not int or not 1 <= inference_runs <= 1000:
        raise TextAnalysisError("Number of inference runs must be an integer from 1 to 1000.")
    for name, value, maximum in (("Temperature", temperature, 1.2), ("Top-p", top_p, 1.0)):
        if safe_numeric_setting(value, None, 0.0, maximum) is None:
            raise TextAnalysisError(f"{name} must be a finite number from 0 to {maximum}.")


def unpack_text_result(result):
    if isinstance(result, str):
        raise TextAnalysisError(result or "The model returned empty content.")
    if not isinstance(result, (tuple, list)) or len(result) not in (2, 3):
        raise TextAnalysisError("The model returned an unexpected analysis result format.")
    generated, metrics = result[:2]
    if isinstance(generated, str) and generated.startswith("Error") and not metrics:
        raise TextAnalysisError(generated)
    if not isinstance(generated, str) or not generated.strip():
        raise TextAnalysisError("The model returned empty content.")
    if not isinstance(metrics, Mapping) or not metrics:
        raise TextAnalysisError("The comparison did not produce similarity metrics.")
    for name, value in metrics.items():
        if not isinstance(name, str) or not name or isinstance(value, bool) or not isinstance(value, Real):
            raise TextAnalysisError("The comparison produced invalid similarity metrics.")
        try:
            finite = math.isfinite(value)
        except (OverflowError, ValueError):
            finite = False
        if not finite:
            raise TextAnalysisError("The comparison produced non-finite similarity metrics.")
    return generated, dict(metrics), result[2] if len(result) == 3 else None


def run_text_inferences(inference_runs, analyze_run, on_success):
    """Checkpoint each success; stop on failure without discarding earlier runs."""
    if type(inference_runs) is not int or not 1 <= inference_runs <= 1000:
        raise TextAnalysisError("Number of inference runs must be an integer from 1 to 1000.")
    for index in range(inference_runs):
        try:
            generated, metrics, logprobs = unpack_text_result(analyze_run(index))
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            raise TextAnalysisError(f"Run {index + 1}/{inference_runs} failed: {message}") from exc
        on_success(index, generated, metrics, logprobs)


def text_report_fingerprint(results_data):
    """Bind a PDF cache to the results and their captured analysis metadata."""
    payload = json.dumps(results_data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
