"""Sidebar utilities for API configuration and model selection."""

from typing import Any, Dict, List, Tuple

import streamlit as st

from src.auth import (
    auth_enabled,
    is_logged_in,
    render_api_configuration_auth,
    render_keep_button,
)
from src.config import DEFAULT_KIMI_KEY, DEFAULT_OPENROUTER_KEY

HIDDEN_SIDEBAR_API_KEY_PROVIDERS = ("OpenAI", "Kimi")

SIDEBAR_DEFAULTS = {
    "sidebar_openai_api_key": "",
    "sidebar_openrouter_api_key": "",
    "sidebar_anthropic_api_key": "",
    "sidebar_google_api_key": "",
    "sidebar_kimi_api_key": "",
    "sidebar_local_vllm_base_url": "http://localhost:8000/v1",
    "sidebar_local_vllm_api_key": "",
    "sidebar_local_vllm_model": "",
    "sidebar_provider_selectbox": "OpenAI",
}

# Model configuration for sidebar
MODEL_CONFIG = {
    "OpenAI": {
        "models": [
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.2",
            "gpt-5.1",
            "gpt-4o",
            "gpt-4o-mini",
        ],
        "help": "Default: gpt-4o-mini. Newest: gpt-5.5 / gpt-5.4.",
        "key": "sidebar_openai_model_selectbox",
        "default_index": 7,
    },
    "OpenRouter": {
        "models": [
            "google/gemma-4-26b-a4b-it:free",
            "inclusionai/ling-3.0-flash:free",
            "openai/gpt-oss-20b:free",
            "google/gemma-4-31b-it:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "nvidia/nemotron-3-nano-30b-a3b:free",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            "nvidia/nemotron-nano-12b-v2-vl:free",
            "nvidia/nemotron-nano-9b-v2:free",
            "cohere/north-mini-code:free",
            "openrouter/free",
            "moonshotai/kimi-k2.6",
            "moonshotai/kimi-k2.5",
            "qwen/qwen3-235b-a22b-thinking-2507",
        ],
        "help": (
            "Free models verified against OpenRouter's official model API. "
            "The free catalog changes often; openrouter/free provides automatic routing."
        ),
        "key": "sidebar_openrouter_model_selectbox",
        "default_index": 0,
    },    "Anthropic": {
        "models": [
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
        ],
        "help": "Newest flagship: claude-opus-4-8. claude-sonnet-4-6 is the balanced default for everyday tasks.",
        "key": "sidebar_anthropic_model_selectbox",
        "default_index": 3,
    },
    "Google Gemini": {
        "models": [
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-3-flash-preview",
        ],
        "help": "Recommended: gemini-3.5-flash (GA). gemini-2.5-* remains available; Gemini 1.5/2.0 are shut down.",
        "key": "sidebar_google_model_selectbox",
        "default_index": 0,
    },
    "Kimi": {
        "models": [
            "kimi-k2.7-code-highspeed",
            "kimi-k2.7-code",
            "kimi-k2.6",
            "kimi-k2.5",
            "moonshot-v1-8k",
            "moonshot-v1-32k",
            "moonshot-v1-128k",
            "moonshot-v1-8k-vision-preview",
            "moonshot-v1-32k-vision-preview",
            "moonshot-v1-128k-vision-preview",
        ],
        "help": "Default: moonshot-v1-128k. K2 models auto-use temperature=1 and top_p=0.95.",
        "key": "sidebar_kimi_model_selectbox",
        "default_index": 6,
    },
}

API_KEY_CONFIG = [
    {
        "label": "OpenAI API Key",
        "key": "sidebar_openai_api_key",
        "help": "Enter your OpenAI API key",
        "provider": "OpenAI",
    },
    {
        "label": "OpenRouter API Key",
        "key": "sidebar_openrouter_api_key",
        "help": "Enter your OpenRouter API key (leave blank to use default key)",
        "provider": "OpenRouter",
    },
    {
        "label": "Anthropic API Key",
        "key": "sidebar_anthropic_api_key",
        "help": "Enter your Anthropic API key",
        "provider": "Anthropic",
    },
    {
        "label": "Google Gemini API Key",
        "key": "sidebar_google_api_key",
        "help": "Enter your Google Gemini API key",
        "provider": "Google Gemini",
    },
    {
        "label": "Kimi API Key",
        "key": "sidebar_kimi_api_key",
        "help": "Enter your Kimi (Moonshot) API key (leave blank to use default key)",
        "provider": "Kimi",
    },
]


def init_sidebar_defaults() -> None:
    for key, default in SIDEBAR_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def render_api_key_input(
    label: str,
    key: str,
    help_text: str,
    *,
    provider: str = "",
    disabled: bool = False,
) -> str:
    if auth_enabled() and is_logged_in():
        help_text = (
            f"{help_text} Use the eye icon to show or hide the key. "
            "Clear the field and click Keep to remove a saved key."
        )
    return st.text_input(
        label,
        type="password",
        help=help_text,
        key=key,
        disabled=disabled,
    )


def render_model_selectbox(provider: str, config: Dict[str, Any], *, disabled: bool = False) -> str:
    models = list(config["models"])
    widget_key = str(config["key"])
    current_model = st.session_state.get(widget_key)
    if current_model is not None and current_model not in models:
        # Hot deployments can leave a removed OpenRouter model in this session.
        st.session_state.pop(widget_key, None)

    kwargs = {
        "label": "Model name",
        "options": models,
        "key": widget_key,
        "disabled": disabled,
    }
    if config.get("default_index") is not None:
        kwargs["index"] = config["default_index"]
    if config.get("help"):
        kwargs["help"] = config["help"]
    return st.selectbox(**kwargs)

def get_api_key_for_provider(provider: str, api_keys: Dict[str, str]) -> str:
    from src.user_settings import get_secure_api_key

    provider_key_map = {
        "OpenAI": "openai_api_key",
        "OpenRouter": "openrouter_api_key",
        "Anthropic": "anthropic_api_key",
        "Google Gemini": "google_api_key",
        "Kimi": "kimi_api_key",
        "Local vLLM": "local_vllm_api_key",
    }

    key_name = provider_key_map.get(provider)
    if not key_name:
        return ""

    api_key = get_secure_api_key(provider, api_keys)
    if not api_key:
        api_key = api_keys.get(key_name, "")

    if provider == "Kimi":
        return api_key.strip() if api_key.strip() else DEFAULT_KIMI_KEY
    if provider == "OpenRouter":
        return api_key.strip() if api_key.strip() else DEFAULT_OPENROUTER_KEY

    return str(api_key or "").strip()


def _current_model_name(provider: str) -> str:
    if provider in MODEL_CONFIG:
        model_key = MODEL_CONFIG[provider]["key"]
        return str(st.session_state.get(model_key, "") or "")
    if provider == "Local vLLM":
        return str(st.session_state.get("sidebar_local_vllm_model", "") or "")
    return ""


def render_api_configuration_section(
    *,
    disabled: bool = False,
    include_auth: bool = True,
) -> Dict[str, str]:
    from src.user_settings import apply_pending_sidebar_widget_resets

    apply_pending_sidebar_widget_resets()
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)

    init_sidebar_defaults()
    if include_auth:
        render_api_configuration_auth(disabled=disabled)

    api_keys: Dict[str, str] = {}
    provider_var_map = {
        "OpenAI": "openai_api_key",
        "OpenRouter": "openrouter_api_key",
        "Anthropic": "anthropic_api_key",
        "Google Gemini": "google_api_key",
        "Kimi": "kimi_api_key",
    }

    for config in API_KEY_CONFIG:
        if config["provider"] in HIDDEN_SIDEBAR_API_KEY_PROVIDERS:
            api_keys[provider_var_map[config["provider"]]] = st.session_state.get(
                config["key"], ""
            )
            continue
        api_key = render_api_key_input(
            config["label"],
            config["key"],
            config["help"],
            provider=config["provider"],
            disabled=disabled,
        )
        api_keys[provider_var_map[config["provider"]]] = api_key

    st.markdown('<div style="margin-top:8px; font-size: 0.9rem; font-weight: 600;">Local vLLM</div>', unsafe_allow_html=True)
    st.text_input(
        "Base URL",
        help="OpenAI-compatible endpoint for your vLLM server (e.g., http://<host>:8000/v1)",
        key="sidebar_local_vllm_base_url",
        disabled=disabled,
    )
    st.text_input(
        "API Key (optional)",
        type="password",
        help=(
            "Leave blank if your vLLM endpoint does not require a key. "
            "Use the eye icon to show or hide the key. "
            "Clear the field and click Keep to remove a saved key."
        ),
        key="sidebar_local_vllm_api_key",
        disabled=disabled,
    )
    api_keys["local_vllm_api_key"] = st.session_state.get("sidebar_local_vllm_api_key", "")

    provider = st.session_state.get("sidebar_provider_selectbox", "OpenAI")
    model_name = _current_model_name(provider)
    render_keep_button(disabled=disabled, provider=provider, model_name=model_name, api_keys=api_keys)

    st.markdown("</div>", unsafe_allow_html=True)
    return api_keys


def render_model_selection_section(api_keys: Dict[str, str], *, disabled: bool = False) -> Tuple[str, str, str]:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)

    provider_options = [
        "OpenAI",
        "Kimi",
        "OpenRouter",
        "Anthropic",
        "Google Gemini",
        "Local vLLM",
    ]
    current_provider = st.session_state.get("sidebar_provider_selectbox")
    if current_provider not in provider_options:
        st.session_state.pop("sidebar_provider_selectbox", None)
        provider_index = provider_options.index("OpenAI")
    else:
        provider_index = None

    provider = st.selectbox(
        "Select provider",
        provider_options,
        index=provider_index,
        help="Choose your AI provider",
        key="sidebar_provider_selectbox",
        disabled=disabled,
    )

    model_choice = None
    if provider in MODEL_CONFIG:
        model_choice = render_model_selectbox(provider, MODEL_CONFIG[provider], disabled=disabled)
        api_key = get_api_key_for_provider(provider, api_keys)
    elif provider == "Local vLLM":
        model_choice = st.text_input(
            "Model Choice (Optional)",
            placeholder="Just to identify the model you are using (e.g., 'meta-llama/Llama-3-70B-Instruct-v3')",
            key="sidebar_local_vllm_model",
            disabled=disabled,
        )
        api_key = get_api_key_for_provider(provider, api_keys)

    st.markdown("</div>", unsafe_allow_html=True)
    return model_choice, api_key, provider
