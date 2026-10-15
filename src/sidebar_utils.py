"""Sidebar utilities for API configuration and model selection."""

from typing import Any, Dict, List, Tuple

import streamlit as st
from src.widget_defaults import widget_defaults
from src.asu_rc import ASU_PROVIDER, ASU_DEFAULT_MODEL, AsuCatalogError, fetch_asu_models

from src.auth import (
    auth_enabled,
    is_logged_in,
    render_api_configuration_auth,
    render_keep_button,
)
from src.config import DEFAULT_KIMI_KEY, DEFAULT_OPENROUTER_KEY
from src.model_catalog import MODEL_CONFIG, model_replacement

HIDDEN_SIDEBAR_API_KEY_PROVIDERS = ()

SIDEBAR_DEFAULTS = {
    "sidebar_openai_api_key": "",
    "sidebar_openrouter_api_key": "",
    "sidebar_anthropic_api_key": "",
    "sidebar_google_api_key": "",
    "sidebar_kimi_api_key": "",
    "sidebar_asu_rc_api_key": "",
    "sidebar_asu_rc_custom_model": "",
    "sidebar_local_vllm_base_url": "http://localhost:8000/v1",
    "sidebar_local_vllm_api_key": "",
    "sidebar_local_vllm_model": "",
    "sidebar_provider_selectbox": "OpenAI",
}

API_KEY_CONFIG = [
    {
        "label": "ASU Research Computing API Key",
        "key": "sidebar_asu_rc_api_key",
        "help": "Enter your personal ASU Research Computing API key. This provider has no shared default key.",
        "provider": ASU_PROVIDER,
    },
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
    restored_models = st.session_state.get("_analysis_restored_models")
    saved_model = restored_models.get(widget_key) if isinstance(restored_models, dict) else None
    preserve_saved_model = (
        isinstance(current_model, str) and bool(current_model.strip())
        and saved_model == current_model
    )
    if isinstance(restored_models, dict) and widget_key in restored_models and not preserve_saved_model:
        # A changed selection is a new configuration, not the saved run's model.
        updated_models = dict(restored_models)
        updated_models.pop(widget_key)
        st.session_state["_analysis_restored_models"] = updated_models
    if preserve_saved_model and current_model not in models:
        # Restore the exact request identity even after the public catalog changes.
        # The provider layer still reports models that are no longer available.
        models.append(current_model)
    if widget_key in st.session_state and current_model not in models:
        # Saved profiles and hot deployments can retain removed IDs or None.
        replacement = model_replacement(provider, current_model)
        if replacement in models:
            st.session_state[widget_key] = replacement
        else:
            st.session_state.pop(widget_key, None)
        if isinstance(current_model, str) and current_model:
            selected = replacement if replacement in models else models[config.get("default_index", 0)]
            st.info(f"The saved model {current_model} is no longer an option. "
                    f"Selected {selected} for new analyses. Process Manager tasks keep their submitted model.")

    kwargs = {
        "label": "Model name",
        "options": models,
        "key": widget_key,
        "disabled": disabled,
    }
    if config.get("default_index") is not None:
        kwargs.update(widget_defaults(widget_key, index=config["default_index"]))
    if config.get("help"):
        kwargs["help"] = config["help"]
    return st.selectbox(**kwargs)


def render_asu_model_selection(api_key: str, *, disabled: bool = False) -> str:
    """Use ASU's live catalog when available without changing a restored model ID."""
    st.caption("ASU Research Computing provides this gateway at no cost. Request limits may apply.")
    config = dict(MODEL_CONFIG[ASU_PROVIDER])
    models = [model for model in config["models"] if model != "Custom model"]
    if api_key and not disabled:
        try:
            available = fetch_asu_models(api_key)
            if available:
                models = list(available)
            else:
                st.caption("Live catalog unavailable; showing ASU's published model list.")
        except AsuCatalogError as exc:
            st.warning(str(exc))
            st.caption("Showing ASU's published model list.")
        except Exception:
            # Never include the transport exception: it may contain credentials.
            st.caption("Live catalog unavailable; showing ASU's published model list.")
    current = st.session_state.get(config["key"])
    if isinstance(current, str) and current.strip() and current != "Custom model" and current not in models:
        models.append(current)
    models.append("Custom model")
    config["models"] = models
    config["default_index"] = models.index(ASU_DEFAULT_MODEL) if ASU_DEFAULT_MODEL in models else 0
    selection = render_model_selectbox(ASU_PROVIDER, config, disabled=disabled)
    if selection == "Custom model":
        return st.text_input(
            "Custom model ID",
            key="sidebar_asu_rc_custom_model",
            placeholder="Exact model ID from your ASU model catalog",
            help="Requests stay on the ASU Research Computing endpoint.",
            disabled=disabled,
        ).strip()
    return selection


def get_api_key_for_provider(provider: str, api_keys: Dict[str, str]) -> str:
    from src.user_settings import get_secure_api_key

    provider_key_map = {
        "OpenAI": "openai_api_key",
        "OpenRouter": "openrouter_api_key",
        "Anthropic": "anthropic_api_key",
        "Google Gemini": "google_api_key",
        "Kimi": "kimi_api_key",
        ASU_PROVIDER: "asu_rc_api_key",
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
    if provider == ASU_PROVIDER and st.session_state.get("sidebar_asu_rc_model_selectbox") == "Custom model":
        return str(st.session_state.get("sidebar_asu_rc_custom_model", "") or "").strip()
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
        ASU_PROVIDER: "asu_rc_api_key",
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
        ASU_PROVIDER,
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
    if provider == ASU_PROVIDER:
        api_key = get_api_key_for_provider(provider, api_keys)
        model_choice = render_asu_model_selection(api_key, disabled=disabled)
    elif provider in MODEL_CONFIG:
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
