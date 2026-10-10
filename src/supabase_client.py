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
    """Restore this UI account and retain tokens rotated by Supabase.

    Keep worker clients explicit via ``authenticated_client``. A UI client is
    never cached/shared across users, and an in-flight refresh cannot replace a
    different account or a newer session in Streamlit state.
    """
    access_token = st.session_state.get("access_token")
    if not access_token:
        return None
    from src.analysis_checkpoints import AnalysisCheckpointError
    refresh_token = st.session_state.get("refresh_token") or ""
    owner_id = str(st.session_state.get("user_id") or "")
    client = create_supabase_client()
    response = client.auth.set_session(access_token, refresh_token)
    session = getattr(response, "session", None)
    user = getattr(response, "user", None) or getattr(session, "user", None)
    verified_owner = str(getattr(user, "id", "") or "")
    session_owner = str(getattr(getattr(session, "user", None), "id", "") or "")
    if not owner_id or verified_owner != owner_id or (session is not None and session_owner != owner_id):
        raise AnalysisCheckpointError("Your Supabase account could not be verified. Sign in again.", kind="auth")
    if (str(st.session_state.get("user_id") or "") != owner_id
            or st.session_state.get("access_token") != access_token
            or (st.session_state.get("refresh_token") or "") != refresh_token):
        raise AnalysisCheckpointError("Your Supabase session changed while being verified. Retry the action.", kind="auth")
    next_access = getattr(session, "access_token", None)
    next_refresh = getattr(session, "refresh_token", None)
    if not next_access or next_refresh is None:
        raise AnalysisCheckpointError("Supabase returned an incomplete session. Sign in again.", kind="auth")
    if next_access != access_token or next_refresh != refresh_token:
        from src.auth import _set_user_session, sync_auth_browser_storage
        _set_user_session(user, next_access, next_refresh)
        sync_auth_browser_storage()
    return client
