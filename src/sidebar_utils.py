"""Sidebar utilities for API configuration and model selection.

This module contains reusable components for rendering the sidebar,
including API key inputs, model selection dropdowns, and configuration data.
"""

from typing import Any, Dict, List, Tuple
import streamlit as st
from src.config import DEFAULT_OPENROUTER_KEY


# Model configuration for sidebar
MODEL_CONFIG = {
    "OpenAI": {
        "models": [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-instruct",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-5.1",
            "gpt-5.2",
        ],
        "help": "Select an OpenAI model. Perplexity probes work best with instruct-style or mini models that support logprobs.",
        "key": "sidebar_openai_model_selectbox",
        "default_index": None,
    },
    "OpenRouter": {
        "models": [
            "meta-llama/llama-3.3-70b-instruct:free",  # Default model (index 0)
            "meta-llama/llama-3.1-70b-instruct:free",
            "allenai/olmo-3.1-32b-think:free",
            "allenai/olmo-3-32b-think:free",
            "openai/gpt-oss-120b:free",
            "z-ai/glm-4.5-air:free",
            "moonshotai/kimi-k2:free",
            "deepseek/deepseek-r1-0528:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "nousresearch/hermes-3-llama-3.1-405b:free",
            "meta-llama/llama-3.1-405b-instruct:free"
        ],
        "help": None,
        "key": "sidebar_openrouter_model_selectbox",
        "default_index": 0,  # Default to meta-llama/llama-3.3-70b-instruct:free
    },
    "Anthropic": {
        "models": [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229"
        ],
        "help": None,
        "key": "sidebar_anthropic_model_selectbox",
        "default_index": None,
    },
    "Google Gemini": {
        "models": [
            "gemini-1.5-flash-001",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-001",
            "gemini-1.5-pro-latest",
        ],
        "help": "Use the latest Gemini model identifiers supported by the v1 API.",
        "key": "sidebar_google_model_selectbox",
        "default_index": None,
    },
    "Kimi": {
        "models": [
            "kimi-k2-0905-preview",
            "kimi-k2-turbo-preview",
            "kimi-k2-thinking",
            "kimi-k2-thinking-turbo",
        ],
        "help": None,
        "key": "sidebar_kimi_model_selectbox",
        "default_index": None,
    },
}


# API key input configuration
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
        "help": "Enter your Kimi (Moonshot) API key",
        "provider": "Kimi",
    },
]


def render_api_key_input(label: str, key: str, help_text: str) -> str:
    """Render a single API key input field.
    
    Args:
        label: The label for the input field
        key: The Streamlit key for the input field
        help_text: The help text to display
        
    Returns:
        The API key value entered by the user
    """
    return st.text_input(label, type="password", help=help_text, key=key)


def render_model_selectbox(provider: str, config: Dict[str, Any]) -> str:
    """Render model selection selectbox for a given provider.
    
    Args:
        provider: The name of the provider
        config: Configuration dictionary containing models, help, key, and default_index
        
    Returns:
        The selected model name
    """
    kwargs = {
        "label": "Model name",
        "options": config["models"],
        "key": config["key"],
    }
    if config.get("default_index") is not None:
        kwargs["index"] = config["default_index"]
    if config.get("help"):
        kwargs["help"] = config["help"]
    
    return st.selectbox(**kwargs)


def get_api_key_for_provider(provider: str, api_keys: Dict[str, str]) -> str:
    """Get the appropriate API key for the selected provider.
    
    Args:
        provider: The name of the provider
        api_keys: Dictionary mapping provider variable names to API key values
        
    Returns:
        The API key for the provider, or empty string if not found
    """
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
    
    api_key = api_keys.get(key_name, "")
    
    # Special handling for OpenRouter: use default key if empty
    if provider == "OpenRouter":
        return api_key.strip() if api_key.strip() else DEFAULT_OPENROUTER_KEY
    
    return api_key


def render_api_configuration_section() -> Dict[str, str]:
    """Render the API Configuration section and return all API keys.
    
    Returns:
        Dictionary mapping provider variable names to API key values
    """
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    
    api_keys = {}
    
    # Render API key inputs using configuration
    for config in API_KEY_CONFIG:
        api_key = render_api_key_input(config["label"], config["key"], config["help"])
        # Map to original variable names for compatibility
        provider_var_map = {
            "OpenAI": "openai_api_key",
            "OpenRouter": "openrouter_api_key",
            "Anthropic": "anthropic_api_key",
            "Google Gemini": "google_api_key",
            "Kimi": "kimi_api_key",
        }
        api_keys[provider_var_map[config["provider"]]] = api_key
    
    # Local vLLM configuration (special case)
    st.markdown('<div style="margin-top:8px; font-size: 0.9rem; font-weight: 600;">Local vLLM</div>', unsafe_allow_html=True)
    local_vllm_base_url = st.text_input(
        "Base URL",
        value=st.session_state.get("sidebar_local_vllm_base_url", "http://localhost:8000/v1"),
        help="OpenAI-compatible endpoint for your vLLM server (e.g., http://<host>:8000/v1)",
        key="sidebar_local_vllm_base_url",
    )
    local_vllm_api_key = st.text_input(
        "API Key (optional)",
        value=st.session_state.get("sidebar_local_vllm_api_key", ""),
        type="password",
        help="Leave blank if your vLLM endpoint does not require a key",
        key="sidebar_local_vllm_api_key",
    )
    api_keys["local_vllm_api_key"] = local_vllm_api_key
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return api_keys


def render_model_selection_section(api_keys: Dict[str, str]) -> Tuple[str, str, str]:
    """Render the Model Selection section and return model_choice, api_key, and provider.
    
    Args:
        api_keys: Dictionary mapping provider variable names to API key values
        
    Returns:
        Tuple of (model_choice, api_key, provider) for the selected provider
    """
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    
    provider = st.selectbox(
        "Select provider",
        ["OpenAI", "OpenRouter", "Anthropic", "Google Gemini", "Kimi", "Local vLLM"],
        index=1,  # Default to OpenRouter (index 1)
        help="Choose your AI provider",
        key="sidebar_provider_selectbox",
    )
    
    model_choice = None
    # Use configuration for standard providers
    if provider in MODEL_CONFIG:
        model_choice = render_model_selectbox(provider, MODEL_CONFIG[provider])
        api_key = get_api_key_for_provider(provider, api_keys)
    elif provider == "Local vLLM":
        # Special handling for Local vLLM
        model_choice = st.text_input(
            "Model Choice (Optional)",
            value=st.session_state.get("sidebar_local_vllm_model", ""),
            placeholder="Just to identify the model you are using (e.g., 'meta-llama/Llama-3-70B-Instruct-v3')",
            key="sidebar_local_vllm_model",
        )
        api_key = get_api_key_for_provider(provider, api_keys)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return model_choice, api_key, provider

