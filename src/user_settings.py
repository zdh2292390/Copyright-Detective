"""Encrypted user settings persistence in Supabase."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import streamlit as st
from cryptography.fernet import Fernet, InvalidToken

from src.supabase_client import get_authenticated_client, get_fernet_key

PROVIDER_KEY_MAP = {
    "OpenAI": "openai_api_key",
    "OpenRouter": "openrouter_api_key",
    "Anthropic": "anthropic_api_key",
    "Google Gemini": "google_api_key",
    "Kimi": "kimi_api_key",
    "Local vLLM": "local_vllm_api_key",
}

SIDEBAR_STORAGE_KEYS = [
    "sidebar_openai_api_key",
    "sidebar_openrouter_api_key",
    "sidebar_anthropic_api_key",
    "sidebar_google_api_key",
    "sidebar_kimi_api_key",
    "sidebar_local_vllm_base_url",
    "sidebar_local_vllm_api_key",
    "sidebar_local_vllm_model",
    "sidebar_provider_selectbox",
]

SECURE_KEYS_SESSION = "_secure_api_keys"
PLAIN_STORAGE_PREFIX = "plain:"
PENDING_FILL_API_KEY_INPUTS = "_pending_fill_api_key_inputs"
PENDING_RESET_SIDEBAR_SETTINGS = "_pending_reset_sidebar_settings"


def schedule_fill_api_key_inputs(payload: Dict[str, Any]) -> None:
    st.session_state[PENDING_FILL_API_KEY_INPUTS] = {
        "api_keys": dict(payload.get("api_keys") or {}),
        "extra_config": dict(payload.get("extra_config") or {}),
        "provider": str(payload.get("provider") or ""),
    }


def schedule_reset_sidebar_settings() -> None:
    st.session_state[PENDING_RESET_SIDEBAR_SETTINGS] = True


def apply_pending_sidebar_widget_resets() -> None:
    """Apply deferred sidebar widget updates before widgets are instantiated."""
    pending = st.session_state.pop(PENDING_FILL_API_KEY_INPUTS, None)
    if pending:
        reverse_map = _sidebar_key_to_storage()
        for storage_key, session_key in reverse_map.items():
            if storage_key == "provider":
                provider = str(pending.get("provider") or "")
                if provider:
                    st.session_state[session_key] = provider
                continue
            value = ""
            api_keys = pending.get("api_keys") or {}
            extra = pending.get("extra_config") or {}
            if storage_key in api_keys:
                value = str(api_keys.get(storage_key) or "")
            elif storage_key in extra:
                value = str(extra.get(storage_key) or "")
            if value:
                st.session_state[session_key] = value

    if st.session_state.pop(PENDING_RESET_SIDEBAR_SETTINGS, False):
        for session_key in SIDEBAR_STORAGE_KEYS:
            if session_key.endswith("_api_key") or session_key == "sidebar_local_vllm_api_key":
                st.session_state[session_key] = ""
            elif session_key == "sidebar_local_vllm_base_url":
                st.session_state[session_key] = "http://localhost:8000/v1"
            elif session_key == "sidebar_provider_selectbox":
                st.session_state[session_key] = "Kimi"
            elif session_key == "sidebar_local_vllm_model":
                st.session_state[session_key] = ""


def _fernet() -> Optional[Fernet]:
    key = get_fernet_key().strip()
    if not key:
        return None
    try:
        return Fernet(key.encode())
    except (ValueError, TypeError):
        return None


def mask_api_key(api_key: str) -> str:
    value = (api_key or "").strip()
    if not value:
        return ""
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def _sidebar_key_to_storage() -> Dict[str, str]:
    return {
        "openai_api_key": "sidebar_openai_api_key",
        "openrouter_api_key": "sidebar_openrouter_api_key",
        "anthropic_api_key": "sidebar_anthropic_api_key",
        "google_api_key": "sidebar_google_api_key",
        "kimi_api_key": "sidebar_kimi_api_key",
        "local_vllm_base_url": "sidebar_local_vllm_base_url",
        "local_vllm_api_key": "sidebar_local_vllm_api_key",
        "local_vllm_model": "sidebar_local_vllm_model",
        "provider": "sidebar_provider_selectbox",
    }


def collect_settings_payload(
    provider: str,
    model_name: str,
    api_keys: Dict[str, str],
) -> Dict[str, Any]:
    """Build settings from current sidebar widgets.

    API key fields are taken as-is: an empty input clears that saved key on Keep.
    """
    mapping = _sidebar_key_to_storage()
    secure = st.session_state.get(SECURE_KEYS_SESSION) or {}
    payload: Dict[str, Any] = {
        "provider": provider,
        "model_name": model_name or "",
        "api_keys": {},
        "extra_config": {},
    }
    for storage_key, session_key in mapping.items():
        if storage_key == "provider":
            payload["provider"] = str(st.session_state.get(session_key, provider) or provider)
            continue
        typed = str(st.session_state.get(session_key, "") or "").strip()
        if not typed and storage_key in api_keys:
            typed = str(api_keys.get(storage_key, "") or "").strip()
        is_secret = storage_key.endswith("_api_key") or storage_key == "local_vllm_api_key"
        if is_secret:
            # Widget value is authoritative so clearing + Keep removes the key.
            payload["api_keys"][storage_key] = typed
        elif storage_key in ("local_vllm_base_url", "local_vllm_model"):
            saved = str(secure.get(storage_key, "") or "").strip()
            payload["extra_config"][storage_key] = typed or saved
    return payload


def _encrypt_payload(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload).encode()
    fernet = _fernet()
    if fernet is None:
        raise RuntimeError(
            "FERNET_KEY is missing or invalid. API keys were not saved. "
            "Configure FERNET_KEY in the server secrets and try Keep again."
        )
    return fernet.encrypt(raw).decode()


def _decrypt_payload(encrypted: str) -> Dict[str, Any]:
    if encrypted.startswith(PLAIN_STORAGE_PREFIX):
        raw = encrypted[len(PLAIN_STORAGE_PREFIX) :]
    else:
        fernet = _fernet()
        if fernet is None:
            raise RuntimeError("Saved settings are encrypted but FERNET_KEY is missing or invalid.")
        raw = fernet.decrypt(encrypted.encode()).decode()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Invalid saved settings payload.")
    return data


def apply_settings_to_session(payload: Dict[str, Any], *, update_widgets: bool = True) -> None:
    secure: Dict[str, str] = {}
    api_keys = payload.get("api_keys") or {}
    extra = payload.get("extra_config") or {}

    reverse_map = _sidebar_key_to_storage()
    for storage_key, session_key in reverse_map.items():
        if storage_key == "provider":
            provider = str(payload.get("provider") or "")
            if provider and update_widgets:
                st.session_state[session_key] = provider
            continue
        value = ""
        if storage_key in api_keys:
            value = str(api_keys.get(storage_key) or "")
        elif storage_key in extra:
            value = str(extra.get(storage_key) or "")
        is_secret = storage_key.endswith("_api_key") or storage_key == "local_vllm_api_key"
        if is_secret and value:
            secure[storage_key] = value
            if update_widgets:
                st.session_state[session_key] = value
        elif value and update_widgets:
            st.session_state[session_key] = value

    st.session_state[SECURE_KEYS_SESSION] = secure
    st.session_state["_saved_provider"] = str(payload.get("provider") or "")
    st.session_state["_saved_model_name"] = str(payload.get("model_name") or "")


def get_secure_api_key(provider: str, api_keys: Dict[str, str]) -> str:
    storage = PROVIDER_KEY_MAP.get(provider, "")
    if not storage:
        return ""

    session_key = f"sidebar_{storage}"
    typed = str(st.session_state.get(session_key, "") or "").strip()
    if typed:
        return typed

    secure = st.session_state.get(SECURE_KEYS_SESSION) or {}
    saved = str(secure.get(storage, "") or "").strip()
    if saved:
        return saved

    return str(api_keys.get(storage, "") or "").strip()


def get_masked_saved_key(provider: str) -> str:
    storage = PROVIDER_KEY_MAP.get(provider, "")
    if not storage:
        return ""
    secure = st.session_state.get(SECURE_KEYS_SESSION) or {}
    return mask_api_key(str(secure.get(storage, "") or ""))


def has_saved_keys() -> bool:
    secure = st.session_state.get(SECURE_KEYS_SESSION) or {}
    return any(str(v).strip() for v in secure.values())


def _format_settings_storage_error(exc: Exception) -> str:
    message = str(exc)
    if "PGRST205" in message or "Could not find the table" in message and "user_settings" in message:
        return (
            "Database table public.user_settings is missing. "
            "Open Supabase → SQL Editor, run supabase/user_settings.sql from this repo, then try Keep again."
        )
    return message


def load_user_settings(*, force: bool = False) -> bool:
    if not st.session_state.get("user_id"):
        return False
    if st.session_state.get("_user_settings_loaded") and not force:
        return bool(has_saved_keys())

    client = get_authenticated_client()
    if client is None:
        return False

    try:
        result = (
            client.table("user_settings")
            .select("encrypted_api_key, provider, model_name, extra_config")
            .eq("user_id", st.session_state.user_id)
            .maybe_single()
            .execute()
        )
        row = result.data
        if not row or not row.get("encrypted_api_key"):
            st.session_state[SECURE_KEYS_SESSION] = {}
            st.session_state["_user_settings_loaded"] = True
            return False

        payload = _decrypt_payload(row["encrypted_api_key"])
        if not payload.get("provider"):
            payload["provider"] = row.get("provider") or ""
        if not payload.get("model_name"):
            payload["model_name"] = row.get("model_name") or ""
        apply_settings_to_session(payload)
        st.session_state["_user_settings_loaded"] = True
        return True
    except InvalidToken:
        st.warning("Saved API key could not be decrypted. Please enter a new key and click Keep.")
    except Exception as exc:
        message = _format_settings_storage_error(exc)
        if "user_settings" not in message and "PGRST" not in message:
            st.warning(f"Failed to load saved settings: {message}")
        elif "missing" in message.lower():
            st.warning(message)
    st.session_state["_user_settings_loaded"] = True
    return False


def save_user_settings(provider: str, model_name: str, api_keys: Dict[str, str]) -> str:
    """Persist current sidebar API keys. Returns 'saved', 'updated', or 'cleared'."""
    if not st.session_state.get("user_id"):
        raise RuntimeError("Not signed in.")

    client = get_authenticated_client()
    if client is None:
        raise RuntimeError("Authentication session is missing.")

    had_saved_keys = has_saved_keys()
    payload = collect_settings_payload(provider, model_name, api_keys)
    saved_keys = payload.get("api_keys") or {}
    if not any(str(value).strip() for value in saved_keys.values()):
        delete_user_settings()
        return "cleared"

    encrypted = _encrypt_payload(payload)
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "user_id": st.session_state.user_id,
        "encrypted_api_key": encrypted,
        "provider": payload.get("provider") or provider,
        "model_name": payload.get("model_name") or model_name or "",
        "extra_config": payload.get("extra_config") or {},
        "updated_at": now,
    }
    try:
        client.table("user_settings").upsert(row, on_conflict="user_id").execute()
    except Exception as exc:
        raise RuntimeError(_format_settings_storage_error(exc)) from exc
    apply_settings_to_session(payload, update_widgets=False)
    st.session_state["_user_settings_loaded"] = True
    schedule_fill_api_key_inputs(payload)
    return "updated" if had_saved_keys else "saved"


def delete_user_settings() -> None:
    if not st.session_state.get("user_id"):
        raise RuntimeError("Not signed in.")

    client = get_authenticated_client()
    if client is None:
        raise RuntimeError("Authentication session is missing.")

    try:
        client.table("user_settings").delete().eq("user_id", st.session_state.user_id).execute()
    except Exception as exc:
        raise RuntimeError(_format_settings_storage_error(exc)) from exc
    clear_secure_keys()


def clear_secure_keys() -> None:
    st.session_state.pop(SECURE_KEYS_SESSION, None)
    st.session_state.pop("_user_settings_loaded", None)
    st.session_state.pop("_saved_provider", None)
    st.session_state.pop("_saved_model_name", None)
    schedule_reset_sidebar_settings()
