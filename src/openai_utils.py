"""OpenAI Chat Completions parameter compatibility without changing budgets.

Contracts: https://developers.openai.com/api/docs/guides/latest-model
https://developers.openai.com/api/docs/guides/gpt-5.4.md
https://developers.openai.com/api/docs/models/gpt-5.5
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


# Match model aliases and dated snapshots, not unrelated OpenRouter models.
_DEFAULT_NONE = re.compile(r"^gpt-5\.(?:1|2|4)(?:-(?:mini|nano))?(?:-\d{4}-\d{2}-\d{2})?$")
_OPTIONAL_NONE = re.compile(r"^(?:gpt-5\.5|gpt-6-(?:sol|luna))(?:-\d{4}-\d{2}-\d{2})?$")
_REASONING_FAMILY = re.compile(r"^(?:gpt-(?:5|6)(?:[.\-]|$)|o\d(?:[.\-]|$))")


def _model_name(model_name: Any) -> str:
    name = str(model_name or "").lower().strip()
    if name.startswith("openai/"):
        name = name[len("openai/"):]
    return name


def _is_reasoning_model(name: str) -> bool:
    # Chat/instant models have a different API contract from reasoning aliases.
    return bool(_REASONING_FAMILY.match(name)) and "chat" not in name and not name.endswith("-instant")


def _supports_none(name: str) -> bool:
    return bool(_DEFAULT_NONE.fullmatch(name) or _OPTIONAL_NONE.fullmatch(name))


def openai_model_rejects_sampling_params(
    model_name: str, reasoning_effort: Optional[str] = None,
) -> bool:
    """Whether this model's effective reasoning mode rejects sampling controls.

    GPT-5.1/5.2/5.4 defaults allow sampling. GPT-5.5 and the GPT-6 reasoning
    aliases do not default to none. An explicit none is allowed only for
    models whose published API contract supports it.
    """
    name = _model_name(model_name)
    if not _is_reasoning_model(name):
        return False
    if reasoning_effort == "none":
        return not _supports_none(name)
    if reasoning_effort is not None:
        return True
    return not bool(_DEFAULT_NONE.fullmatch(name))


def apply_openai_request_compat(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust Chat Completions parameters for the effective reasoning mode.

    Keep the model's default effort and the caller's token budget. In
    particular, this does not turn GPT-5.5's default medium into none.
    Legacy Completions requests must not use this Chat-specific adapter.
    """
    result = dict(kwargs)
    name = _model_name(result.get("model"))
    if not _is_reasoning_model(name):
        return result

    effort = result.get("reasoning_effort")
    if effort == "none" and not _supports_none(name):
        raise ValueError(
            f"{result.get('model')} does not support reasoning_effort='none'. "
            "Use a supported reasoning effort or select gpt-4o-mini/gpt-6-luna."
        )
    if openai_model_rejects_sampling_params(name, effort):
        if result.get("logprobs") or result.get("top_logprobs") is not None:
            if _supports_none(name):
                raise ValueError(
                    f"Logprobs for {result.get('model')} require reasoning_effort='none'. "
                    "Use an explicit non-reasoning request or select gpt-4o-mini."
                )
            raise ValueError(
                f"{result.get('model')} does not support logprobs in its reasoning mode. "
                "Select gpt-4o-mini or gpt-6-luna with reasoning_effort='none'."
            )
        for key in ("top_p", "temperature", "logprobs", "top_logprobs"):
            result.pop(key, None)
    if "max_tokens" in result:
        budget = result.pop("max_tokens")
        result.setdefault("max_completion_tokens", budget)
    return result


def apply_openai_short_answer_compat(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Use non-reasoning mode explicitly for single-choice/one-token answers.

    This dedicated policy enables none only on models that support it, so the
    unchanged tiny budget is available for the answer and token probabilities.
    Models requiring reasoning, or an explicit non-none effort, fail before a
    request instead of spending a larger hidden budget or inventing probabilities.
    Ordinary generation must use apply_openai_request_compat instead.
    """
    result = dict(kwargs)
    name = _model_name(result.get("model"))
    if _is_reasoning_model(name):
        if not _supports_none(name):
            raise ValueError(
                f"{result.get('model')} requires reasoning and cannot perform a "
                "one-token single-choice evaluation. Select gpt-4o-mini or gpt-6-luna."
            )
        if result.get("reasoning_effort") not in (None, "none"):
            raise ValueError(
                "Single-choice evaluation requires reasoning_effort='none' with "
                "its one-token budget. Use a non-reasoning request or another model."
            )
        result["reasoning_effort"] = "none"
        # A one-token answer is already bounded. Avoid requiring stop-sequence
        # support on reasoning models; the single-choice caller parses its letter.
        result.pop("stop", None)
    return apply_openai_request_compat(result)


def unsupported_openai_sampling_param(exc: Exception) -> Optional[str]:
    """Return the unsupported sampling param name from an OpenAI API error."""
    text = str(exc).lower()
    if "unsupported" not in text and "not supported" not in text:
        return None
    for param in ("top_p", "temperature"):
        if param in text:
            return param
    return None
