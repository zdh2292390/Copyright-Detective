"""Supabase client helpers."""

from __future__ import annotations

import os
from threading import Lock
from typing import Dict, Optional

import streamlit as st
from supabase import Client, create_client


_SECRET_CACHE: Dict[str, str] = {}
_SECRET_CACHE_LOCK = Lock()


def get_secret(name: str, default: str = "") -> str:
    """Read a secret once on the Streamlit thread, then serve workers from memory."""
    with _SECRET_CACHE_LOCK:
        cached = _SECRET_CACHE.get(name)
    if cached is not None:
        return cached

    env_value = os.environ.get(name, "")
    if env_value:
        resolved = str(env_value)
    else:
        try:
            value = st.secrets[name]
        except (FileNotFoundError, KeyError, TypeError):
            value = default
        except Exception:
            value = default
        resolved = default if value is None or value == "" else str(value)

    with _SECRET_CACHE_LOCK:
        existing = _SECRET_CACHE.setdefault(name, resolved)
    return existing

def preload_secrets(names: list[str]) -> None:
    """Resolve Streamlit-backed secrets before work moves to background threads."""

    for name in names:
        get_secret(name)


def auth_enabled() -> bool:
    return bool(get_secret("SUPABASE_URL") and get_secret("SUPABASE_ANON_KEY"))


def get_app_url() -> str:
    return get_secret("APP_URL", "http://localhost:8501").rstrip("/") + "/"


def get_fernet_key() -> str:
    return get_secret("FERNET_KEY")


def create_supabase_client() -> Client:
    return create_client(get_secret("SUPABASE_URL"), get_secret("SUPABASE_ANON_KEY"))


def authenticated_client(access_token: str, refresh_token: str = "") -> Client:
    client = create_supabase_client()
    if access_token:
        client.auth.set_session(access_token, refresh_token or "")
    return client


def get_authenticated_client() -> Optional[Client]:
    access_token = st.session_state.get("access_token")
    if not access_token:
        return None
    refresh_token = st.session_state.get("refresh_token") or ""
    return authenticated_client(access_token, refresh_token)
