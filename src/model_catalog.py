"""Provider-scoped text model catalog, checked against official sources.

Availability is a catalog check, not a guarantee of access for every account.
Never remap an inference request: saved results must retain their model identity.
"""

CATALOG_CHECKED_ON = "2026-10-07"
CATALOG_SOURCES = {
    "OpenAI": "https://developers.openai.com/api/docs/models/all",
    "OpenRouter": "https://openrouter.ai/api/v1/models",
    "Anthropic": "https://platform.claude.com/docs/en/models/overview",
    "Google Gemini": "https://ai.google.dev/gemini-api/docs/models",
    "Kimi": "https://platform.kimi.ai/docs/models",
}

DEFAULT_MODELS = {
    "OpenAI": "gpt-4o-mini",
    "OpenRouter": "google/gemma-4-26b-a4b-it:free",
    "Anthropic": "claude-sonnet-5-5",
    "Google Gemini": "gemini-3.8-flash",
    "Kimi": "kimi-k2.6",
}


def _config(provider, models, key, help_text):
    return {
        "models": models,
        "key": key,
        "default_index": models.index(DEFAULT_MODELS[provider]),
        "help": help_text,
    }


MODEL_CONFIG = {
    "OpenAI": _config("OpenAI", [
        "gpt-6-astra", "gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna",
        "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano",
        "gpt-5.2", "gpt-5.1", "gpt-4o", "gpt-4o-mini",
    ], "sidebar_openai_model_selectbox",
        "Default: gpt-4o-mini. Latest: GPT-6 Astra / GPT-6.1 Sol / GPT-6 Luna. "
        "gpt-5.1 and gpt-5.4-nano remain available until 2027-04-01. "
        "Reasoning models have sampling and token-probability restrictions."),
    "OpenRouter": _config("OpenRouter", [
        "google/gemma-4-26b-a4b-it:free", "google/gemma-4-31b-it:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "cohere/north-mini-code:free", "openrouter/free",
        "moonshotai/kimi-k2.6", "moonshotai/kimi-k2.5",
        "qwen/qwen3-235b-a22b-thinking-2507",
    ], "sidebar_openrouter_model_selectbox",
        "Verified against the official model API on 2026-10-07. "
        "The free catalog changes often; openrouter/free routes automatically. "
        "Models without :free may incur charges; token probabilities depend on the model and endpoint."),
    "Anthropic": _config("Anthropic", [
        "claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5",
        "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6",
        "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001",
    ], "sidebar_anthropic_model_selectbox",
        "Default: claude-sonnet-5-5. Latest: Fable 5.1 and Claude 5.5. "
        "Sonnet 4.5 is deprecated and retires on 2026-11-30. "
        "Always-thinking models cannot perform one-token choice requests."),
    "Google Gemini": _config("Google Gemini", [
        "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
        "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview",
        "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
    ], "sidebar_google_model_selectbox",
        "Recommended: gemini-3.8-flash (stable), or gemini-3.5-flash-lite for high volume. "
        "Gemini 2.5 requires prior account usage; it is still served. "
        "Preview models may have tighter limits."),
    "Kimi": _config("Kimi", [
        "kimi-k3", "kimi-k2.6", "kimi-k2.7-code-highspeed", "kimi-k2.7-code",
    ], "sidebar_kimi_model_selectbox",
        "Default: kimi-k2.6 in non-thinking mode (temperature=0.6, top_p=0.95). "
        "K3 always reasons (temperature=1, top_p=0.95). K2.7 models target coding. "
        "kimi-k2.5 and moonshot-v1 shut down on 2026-08-31."),
}

# These particular free routes are absent from the official OpenRouter catalog.
# Their paid counterparts are separate IDs; do not silently substitute them.
_UNLISTED_OPENROUTER_MODELS = frozenset({
    "inclusionai/ling-3.0-flash:free", "openai/gpt-oss-20b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free", "nvidia/nemotron-nano-12b-v2-vl:free",
    "nvidia/nemotron-nano-9b-v2:free",
})


# Confirmed past retirements only; future deprecation dates do not block runs.
# https://platform.claude.com/docs/en/about-claude/model-deprecations
_RETIRED_ANTHROPIC_MODELS = {
    "claude-opus-4-1-20250805": "2026-08-05",
    "claude-opus-4-20250514": "2026-06-15",
    "claude-sonnet-4-20250514": "2026-06-15",
    "claude-3-7-sonnet-20250219": "2026-02-19",
    "claude-3-5-haiku-20241022": "2026-02-19",
    "claude-3-haiku-20240307": "2026-04-20",
    "claude-3-5-sonnet-20240620": "2025-10-28",
    "claude-3-5-sonnet-20241022": "2025-10-28",
    "claude-3-opus-20240229": "2026-01-05",
    "claude-3-sonnet-20240229": "2025-07-21",
    "claude-2.0": "2025-07-21", "claude-2.1": "2025-07-21",
    **{f"claude-1.{i}": "2024-11-06" for i in range(4)},
    **{f"claude-instant-1.{i}": "2024-11-06" for i in range(3)},
}
# https://developers.openai.com/api/docs/deprecations
_RETIRED_OPENAI_MODELS = {
    "gpt-5-chat-latest": "2026-07-23", "gpt-5-codex": "2026-07-23",
    "gpt-5.1-chat-latest": "2026-07-23", "gpt-5.1-codex": "2026-07-23",
    "gpt-5.1-codex-mini": "2026-07-23", "gpt-5.1-codex-max": "2026-07-23",
    "gpt-5.2-codex": "2026-07-23", "gpt-5.2-chat-latest": "2026-08-10",
    "gpt-5.3-chat-latest": "2026-08-10",
    "gpt-3.5-turbo-instruct": "2026-09-28", "gpt-3.5-turbo-1106": "2026-09-28",
    "babbage-002": "2026-09-28", "davinci-002": "2026-09-28",
    "gpt-4-0314": "2026-03-26", "gpt-4-0125-preview": "2026-03-26",
    "gpt-4-turbo-preview": "2026-03-26",
    "gpt-4-turbo-preview-completions": "2026-03-26",
    "chatgpt-4o-latest": "2026-02-17",
}


def _unavailable_reason(provider, model):
    if not isinstance(model, str):
        return None
    name = model.strip()
    if provider == "Anthropic" and name in _RETIRED_ANTHROPIC_MODELS:
        return f"was retired by Anthropic on {_RETIRED_ANTHROPIC_MODELS[name]}"
    if provider == "OpenAI" and name in _RETIRED_OPENAI_MODELS:
        return f"was shut down by OpenAI on {_RETIRED_OPENAI_MODELS[name]}"
    if provider == "Kimi" and (name == "kimi-k2.5" or name.startswith("moonshot-v1")):
        return "was shut down by Kimi on 2026-08-31"
    if provider == "Google Gemini":
        name = name.removeprefix("models/")
        if (name == "gemini-pro" or name.startswith(("gemini-1.5-", "gemini-2.0-"))
                or name in {"gemini-3-pro-preview", "gemini-3.1-flash-lite-preview"}):
            return "has been shut down by Google"
    if provider == "OpenRouter" and name in _UNLISTED_OPENROUTER_MODELS:
        return "is no longer listed in OpenRouter's official model catalog as of 2026-10-07"
    return None


def model_replacement(provider, model):
    """Suggest a new selection; callers must never substitute it in a saved run."""
    if _unavailable_reason(provider, model) is None:
        return None
    if provider == "Anthropic":
        if "haiku" in model or model.startswith(("claude-1.", "claude-instant-")):
            return "claude-haiku-5-5"
        if "opus" in model:
            return "claude-opus-5-5"
    if provider == "Google Gemini":
        name = model.strip().removeprefix("models/")
        if name == "gemini-3-pro-preview":
            return "gemini-3.1-pro-preview"
        if "lite" in name:
            return "gemini-3.5-flash-lite"
    return DEFAULT_MODELS.get(provider)


def model_unavailability_error(provider, model):
    """Reject only verified unavailable routes, preserving custom/local IDs."""
    reason = _unavailable_reason(provider, model)
    if reason is None:
        return None
    replacement = model_replacement(provider, model)
    return (
        f"Model {model} {reason}. Select {replacement} or another available model "
        "and start a new analysis. Existing saved results retain their original model; "
        "a saved analysis cannot resume with a different model."
    )
