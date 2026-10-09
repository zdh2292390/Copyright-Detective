"""Claude Messages API compatibility for documented model capabilities."""

from __future__ import annotations

import inspect
import re
from collections.abc import Mapping
from typing import Any, Dict

# Official capability references:
# https://platform.claude.com/docs/en/about-claude/model-deprecations
# https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide
# https://platform.claude.com/docs/en/models/haiku-5-5/whats-new-haiku-5-5
# https://platform.claude.com/docs/en/models/opus-5-5/migration-guide
# https://platform.claude.com/docs/en/models/haiku-4-5/migration-guide
_SAMPLING_RESTRICTED_MODELS = (
    "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5-5",
    "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1",
)
_ALWAYS_THINKING_MODELS = ("claude-opus-5-5", "claude-fable-5-1")
_LEGACY_SAMPLING_MODELS = (
    "claude-opus-4-5", "claude-opus-4-6",
    "claude-sonnet-4-5", "claude-sonnet-4-6", "claude-haiku-4-5",
)


def _matches_model(model_name: str, names: tuple[str, ...]) -> bool:
    name = str(model_name or "").strip().lower()
    return any(
        name == model or re.fullmatch(re.escape(model) + r"-(?:\d{8}|\d{4}-\d{2}-\d{2})", name)
        for model in names
    )


def anthropic_short_answer_error(model_name: str) -> str | None:
    """A one-token answer cannot reserve space for always-on thinking."""
    if _matches_model(model_name, _ALWAYS_THINKING_MODELS):
        return (
            f"Error: {model_name} requires thinking and does not support this "
            "one-token single-choice evaluation. Choose Claude Sonnet 5.5 "
            "or Claude Haiku 5.5 for text-only single-choice evaluation."
        )
    return None


def apply_anthropic_request_compat(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep budgets/prompts and select the application's former text-only mode.

    Sampling is omitted only where the API rejects it. New Sonnet/Haiku
    defaults would otherwise spend the caller's text budget on thinking.
    Explicit thinking settings supplied by other callers remain intact.
    """
    result = dict(kwargs)
    model = str(result.get("model") or "")
    if _matches_model(model, _SAMPLING_RESTRICTED_MODELS):
        for param in ("temperature", "top_p", "top_k"):
            result.pop(param, None)
    elif _matches_model(model, ("claude-haiku-4-5",)):
        # Haiku 4.5 rejects specifying both controls, even without thinking.
        # Keep the selected temperature; a top_p-only caller remains intact.
        if result.get("temperature") is not None:
            result.pop("top_p", None)
        elif result.get("temperature") is None:
            result.pop("temperature", None)
    if "thinking" not in result:
        if _matches_model(model, ("claude-sonnet-5-5",)):
            result["thinking"] = {"type": "between_tools"}
        elif _matches_model(model, ("claude-haiku-5-5",)):
            result["thinking"] = {"type": "disabled"}
    return result


def create_anthropic_message(client: Any, kwargs: Dict[str, Any]) -> Any:
    """Use Messages on old/new SDKs without dropping legacy sampling controls."""
    request = apply_anthropic_request_compat(kwargs)
    # SDK 1.0 removes sampling kwargs even for still-active older models.
    # extra_body preserves the values on the wire for those models.
    try:
        parameters = inspect.signature(client.messages.create).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if parameters and not accepts_kwargs:
        extra_body = dict(request.get("extra_body") or {})
        for param in ("temperature", "top_p", "top_k"):
            if param in request and param not in parameters:
                extra_body[param] = request.pop(param)
        if extra_body:
            request["extra_body"] = extra_body
    try:
        return client.messages.create(**request)
    except Exception as exc:
        # Preserve effective controls on older models unless the server
        # specifically rejects this combination. Never retry another 400,
        # an authentication error, a rate limit, or an unavailable service.
        extra_body = dict(request.get("extra_body") or {})
        temperature = request.get("temperature", extra_body.get("temperature"))
        has_top_p = "top_p" in request or "top_p" in extra_body
        if (
            not _matches_model(str(request.get("model") or ""), _LEGACY_SAMPLING_MODELS)
            or temperature is None or not has_top_p
            or not _is_sampling_combination_error(exc)
        ):
            raise
        request.pop("top_p", None)
        extra_body.pop("top_p", None)
        if extra_body:
            request["extra_body"] = extra_body
        else:
            request.pop("extra_body", None)
        return client.messages.create(**request)


def _is_sampling_combination_error(exc: Exception) -> bool:
    message = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if status is not None:
        if status != 400:
            return False
    elif not re.search(r"\b400\b", message):
        return False
    return (
        "temperature" in message and "top_p" in message
        and any(marker in message for marker in (
            "both", "only one", "one of", "at most one", "cannot combine",
        ))
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def extract_anthropic_response_text(response: Any) -> str:
    """Only answer text is eligible for scoring, never reasoning blocks."""
    stop_reason = _field(response, "stop_reason") or "unknown"
    if stop_reason == "refusal":
        return "Error: Anthropic refused the request (finish_reason=refusal)."
    parts = []
    for block in _field(response, "content", []) or []:
        block_type = _field(block, "type")
        text = _field(block, "text")
        # Missing type is tolerated for old SDK-compatible client adapters.
        if block_type in (None, "text") and isinstance(text, str):
            parts.append(text)
    text = "".join(parts).strip()
    if text:
        return text
    return f"Error: Anthropic returned empty content (finish_reason={stop_reason})."
