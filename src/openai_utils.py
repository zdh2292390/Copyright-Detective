"""OpenAI API parameter helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


def openai_model_rejects_sampling_params(model_name: str) -> bool:
    """Return True for OpenAI reasoning models that reject top_p/temperature."""
    name = (model_name or "").lower().strip()
    if not name:
        return False
    # Chat/instant variants accept sampling params (e.g. gpt-5-chat-latest).
    if "chat" in name or name.endswith("-instant"):
        return False
    if name.startswith("gpt-5"):
        return True
    if re.match(r"^o\d", name):
        return True
    return False


def apply_openai_request_compat(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust OpenAI chat/completions kwargs for model-specific API constraints."""
    result = dict(kwargs)
    model = str(result.get("model") or "")
    if not openai_model_rejects_sampling_params(model):
        return result

    result.pop("top_p", None)
    result.pop("temperature", None)
    if "max_tokens" in result and "max_completion_tokens" not in result:
        result["max_completion_tokens"] = result.pop("max_tokens")
    return result


def unsupported_openai_sampling_param(exc: Exception) -> Optional[str]:
    """Return the unsupported sampling param name from an OpenAI API error."""
    text = str(exc).lower()
    if "unsupported" not in text and "not supported" not in text:
        return None
    for param in ("top_p", "temperature"):
        if param in text:
            return param
    return None
