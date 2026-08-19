"""GitHub OAuth and API Configuration auth UI."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from urllib.parse import parse_qs, parse_qsl, quote, unquote, urlencode, urlparse, urlunparse

import streamlit as st

from src.supabase_client import (
    auth_enabled,
    create_supabase_client,
    get_app_url,
    get_authenticated_client,
)
from src.user_settings import clear_secure_keys, load_user_settings, save_user_settings

AUTH_BROWSER_STORAGE_KEY = "copyright_detective_auth"
PKCE_VERIFIER_BROWSER_KEY = "copyright_detective_pkce_verifier"
RETURN_URL_BROWSER_KEY = "copyright_detective_return_url"


def _pkce_cache_keys(provider: str, *, for_popup: bool) -> Tuple[str, str]:
    suffix = f"{provider}_{'popup' if for_popup else 'main'}"
    return f"_oauth_login_url_{suffix}", f"_oauth_pkce_verifier_{suffix}"


def _read_pkce_verifier_from_client(client: Any) -> Optional[str]:
    try:
        auth = client.auth
        value = auth._storage.get_item(f"{auth._storage_key}-code-verifier")
        if value is None:
            return None
        text = str(value).strip()
        return text or None
    except Exception:
        return None


def _append_query_param(url: str, key: str, value: str) -> str:
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}{quote(key, safe='')}={quote(value, safe='')}"


def _embed_verifier_in_oauth_url(oauth_url: str, code_verifier: str) -> str:
    """Put PKCE verifier into redirect_to so the OAuth callback URL carries cd_cv."""
    parsed = urlparse(oauth_url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    redirect_to = (params.get("redirect_to") or [get_app_url()])[0]
    redirect_to = _append_query_param(redirect_to, OAUTH_PKCE_PARAM, code_verifier)
    params["redirect_to"] = [redirect_to]
    flat = {key: values[0] if values else "" for key, values in params.items()}
    return urlunparse(parsed._replace(query=urlencode(flat)))


def _get_pkce_verifier(*, provider: str = "github", for_popup: bool = False) -> Optional[str]:
    url_verifier = _oauth_query_param(OAUTH_PKCE_PARAM)
    if url_verifier:
        return url_verifier
    _, verifier_key = _pkce_cache_keys(provider, for_popup=for_popup)
    stored = st.session_state.get(verifier_key)
    if stored:
        text = str(stored).strip()
        if text:
            return text
    return None


def _oauth_callback_in_progress() -> bool:
    if is_logged_in():
        return False
    return _has_oauth_callback_params()


def _clear_pkce_state() -> None:
    for key in list(st.session_state.keys()):
        key_str = str(key)
        if key_str.startswith("_oauth_pkce_verifier_") or key_str.startswith("_oauth_login_url_"):
            st.session_state.pop(key, None)


def _clear_pkce_browser_storage() -> None:
    _auth_iframe(
        f"""
        <script>
        {_AUTH_CLIENT_SCRIPT}
        (function () {{
            const bridge = window.__copyrightDetectiveAuth;
            const storage = bridge && bridge.appStorage ? bridge.appStorage() : null;
            if (storage) {{
                try {{ storage.removeItem({json.dumps(PKCE_VERIFIER_BROWSER_KEY)}); }} catch (e) {{}}
            }}
        }})();
        </script>
        """,
    )

SESSION_AUTH_KEYS = (
    "user_id",
    "user_email",
    "user_name",
    "access_token",
    "refresh_token",
    "_browser_auth_synced",
    "_clear_browser_auth_pending",
)

OAUTH_MESSAGE_TYPE = "copyright_detective_oauth"
OAUTH_POPUP_NAME = "copyright_detective_github_oauth"
OAUTH_TOKEN_PARAM = "cd_at"
OAUTH_REFRESH_PARAM = "cd_rt"
OAUTH_PKCE_PARAM = "cd_cv"
OAUTH_POPUP_FLAG = "oauth_window"
OAUTH_CALLBACK_PARAMS = (
    "code",
    OAUTH_TOKEN_PARAM,
    OAUTH_REFRESH_PARAM,
    OAUTH_PKCE_PARAM,
    "access_token",
    "refresh_token",
    OAUTH_POPUP_FLAG,
    "expires_in",
    "token_type",
    "provider_token",
)

# Shared browser-side auth bridge: same-window cache restore, popup OAuth, postMessage back to opener.
_AUTH_CLIENT_SCRIPT = f"""
(function () {{
    const STORAGE_KEY = {json.dumps(AUTH_BROWSER_STORAGE_KEY)};
    const PKCE_KEY = {json.dumps(PKCE_VERIFIER_BROWSER_KEY)};
    const MESSAGE_TYPE = {json.dumps(OAUTH_MESSAGE_TYPE)};
    const POPUP_NAME = {json.dumps(OAUTH_POPUP_NAME)};

    function appStorage() {{
        try {{
            if (window.top && window.top.localStorage) return window.top.localStorage;
        }} catch (e) {{}}
        try {{
            return window.localStorage;
        }} catch (e2) {{}}
        return null;
    }}

    function readPkceVerifier() {{
        const storage = appStorage();
        if (!storage) return null;
        try {{
            const value = storage.getItem(PKCE_KEY);
            if (value) return value;
        }} catch (e) {{}}
        return null;
    }}

    function persistPkceVerifier(verifier) {{
        if (!verifier) return;
        const storage = appStorage();
        if (!storage) return;
        try {{ storage.setItem(PKCE_KEY, verifier); }} catch (e) {{}}
    }}

    function persistReturnUrl(url) {{
        if (!url) return;
        const storage = appStorage();
        if (!storage) return;
        try {{ storage.setItem({json.dumps(RETURN_URL_BROWSER_KEY)}, url); }} catch (e) {{}}
    }}

    function readReturnUrl() {{
        const storage = appStorage();
        if (!storage) return null;
        try {{
            const value = storage.getItem({json.dumps(RETURN_URL_BROWSER_KEY)});
            if (value) return value;
        }} catch (e) {{}}
        return null;
    }}

    function clearReturnUrl() {{
        const storage = appStorage();
        if (!storage) return;
        try {{ storage.removeItem({json.dumps(RETURN_URL_BROWSER_KEY)}); }} catch (e) {{}}
    }}

    function stripOAuthParams(url) {{
        const oauthParams = [
            "code",
            {json.dumps(OAUTH_TOKEN_PARAM)},
            {json.dumps(OAUTH_REFRESH_PARAM)},
            {json.dumps(OAUTH_PKCE_PARAM)},
            "access_token",
            "refresh_token",
            {json.dumps(OAUTH_POPUP_FLAG)},
            "expires_in",
            "token_type",
            "provider_token",
        ];
        oauthParams.forEach(function (name) {{ url.searchParams.delete(name); }});
        return url;
    }}

    function restoreReturnUrlAfterLogin(targetWin) {{
        const app = appWindow(targetWin);
        if (!app || !app.location) return false;
        const saved = readReturnUrl();
        if (!saved) return false;

        let target;
        try {{
            target = stripOAuthParams(new URL(saved.split("#")[0]));
        }} catch (e) {{
            clearReturnUrl();
            return false;
        }}

        const current = stripOAuthParams(new URL(app.location.href.split("#")[0]));
        clearReturnUrl();
        if (target.toString() !== current.toString()) {{
            app.location.replace(target.toString());
            return true;
        }}
        return false;
    }}

    function appWindow(win) {{
        try {{
            if (win && win.top && win.top.location) return win.top;
        }} catch (e) {{}}
        return (win && win.parent) ? win.parent : win;
    }}

    function persistAuthCache(accessToken, refreshToken) {{
        if (!accessToken) return;
        const storage = appStorage();
        if (!storage) return;
        try {{
            storage.setItem(
                STORAGE_KEY,
                JSON.stringify({{
                    access_token: accessToken,
                    refresh_token: refreshToken || "",
                    user_id: "",
                }})
            );
        }} catch (e) {{}}
    }}

    function applyTokensToUrl(targetWin, accessToken, refreshToken) {{
        if (!targetWin || !targetWin.location || !accessToken) return false;
        const search = targetWin.location.search || "";
        if (search.includes({json.dumps(OAUTH_TOKEN_PARAM)} + "=") || search.includes("access_token=")) {{
            return false;
        }}
        persistAuthCache(accessToken, refreshToken);
        const url = new URL(targetWin.location.href.split("#")[0]);
        url.searchParams.delete("code");
        url.searchParams.delete({json.dumps(OAUTH_POPUP_FLAG)});
        url.searchParams.set({json.dumps(OAUTH_TOKEN_PARAM)}, accessToken);
        if (refreshToken) url.searchParams.set({json.dumps(OAUTH_REFRESH_PARAM)}, refreshToken);
        targetWin.location.href = url.toString();
        return true;
    }}

    function forwardAuthorizationCodeToOpener() {{
        const app = appWindow(window);
        if (!app || !app.location) return false;
        if (!window.opener || window.opener.closed) return false;
        const params = new URLSearchParams(app.location.search || "");
        const code = params.get("code");
        if (!code) return false;
        const verifier = readPkceVerifier();
        if (verifier) {{
            try {{
                window.opener.postMessage(
                    {{
                        type: MESSAGE_TYPE,
                        auth_code: code,
                        code_verifier: verifier,
                    }},
                    window.location.origin
                );
            }} catch (e) {{}}
        }}
        const openerUrl = new URL(window.opener.location.href.split("#")[0]);
        openerUrl.searchParams.set("code", code);
        if (verifier) openerUrl.searchParams.set({json.dumps(OAUTH_PKCE_PARAM)}, verifier);
        openerUrl.searchParams.delete({json.dumps(OAUTH_POPUP_FLAG)});
        window.opener.location.href = openerUrl.toString();
        try {{ window.close(); }} catch (e) {{}}
        return true;
    }}

    function readCachedAuth() {{
        const storage = appStorage();
        if (!storage) return null;
        let authRaw = null;
        try {{ authRaw = storage.getItem(STORAGE_KEY); }} catch (e) {{ return null; }}
        if (!authRaw) return null;
        try {{
            const auth = JSON.parse(authRaw);
            if (auth && auth.access_token) return auth;
        }} catch (e) {{}}
        return null;
    }}

    function restoreSessionInWindow(targetWin) {{
        const auth = readCachedAuth();
        if (!auth) return false;
        return applyTokensToUrl(targetWin, auth.access_token, auth.refresh_token);
    }}

    function deliverOAuthToOpener(accessToken, refreshToken) {{
        if (!window.opener || window.opener.closed) return false;
        persistAuthCache(accessToken, refreshToken);
        try {{
            window.opener.postMessage(
                {{
                    type: MESSAGE_TYPE,
                    access_token: accessToken,
                    refresh_token: refreshToken || "",
                }},
                window.location.origin
            );
        }} catch (e) {{
            return false;
        }}
        try {{ window.close(); }} catch (e) {{}}
        return true;
    }}

    function captureHashTokens(targetWin) {{
        if (!targetWin || !targetWin.location || !targetWin.location.hash) return false;
        const hash = targetWin.location.hash.substring(1);
        if (!hash) return false;
        const params = new URLSearchParams(hash);
        const accessToken = params.get("access_token");
        if (!accessToken) return false;
        const refreshToken = params.get("refresh_token") || "";

        if (window.opener && !window.opener.closed) {{
            return deliverOAuthToOpener(accessToken, refreshToken);
        }}
        return applyTokensToUrl(targetWin, accessToken, refreshToken);
    }}

    function restorePkceVerifierToUrl(targetWin) {{
        if (!targetWin || !targetWin.location) return false;
        const params = new URLSearchParams(targetWin.location.search || "");
        if (!params.get("code") || params.get({json.dumps(OAUTH_PKCE_PARAM)})) return false;
        const verifier = readPkceVerifier();
        if (!verifier) return false;
        const url = new URL(targetWin.location.href.split("#")[0]);
        url.searchParams.set({json.dumps(OAUTH_PKCE_PARAM)}, verifier);
        targetWin.location.href = url.toString();
        return true;
    }}

    function captureOAuthReturn(targetWin) {{
        if (forwardAuthorizationCodeToOpener()) return true;
        return captureHashTokens(targetWin);
    }}

    function installOAuthListener(targetWin) {{
        if (!targetWin || targetWin.__copyrightDetectiveOAuthListener) return;
        targetWin.__copyrightDetectiveOAuthListener = true;
        targetWin.addEventListener("message", function (event) {{
            if (event.origin !== window.location.origin) return;
            const data = event.data;
            if (!data || data.type !== MESSAGE_TYPE) return;
            if (data.code_verifier) persistPkceVerifier(data.code_verifier);
            if (data.auth_code) {{
                const url = new URL(targetWin.location.href.split("#")[0]);
                url.searchParams.set("code", data.auth_code);
                if (data.code_verifier) {{
                    url.searchParams.set({json.dumps(OAUTH_PKCE_PARAM)}, data.code_verifier);
                }}
                url.searchParams.delete({json.dumps(OAUTH_POPUP_FLAG)});
                targetWin.location.href = url.toString();
                return;
            }}
            if (!data.access_token) return;
            applyTokensToUrl(targetWin, data.access_token, data.refresh_token || "");
        }});
    }}

    function signInWithGitHub(targetWin, oauthUrl) {{
        if (!oauthUrl) return;
        const app = appWindow(targetWin);
        if (restoreSessionInWindow(app)) return;

        const popup = app.open(
            oauthUrl,
            POPUP_NAME,
            "popup=yes,width=520,height=720,menubar=no,toolbar=no,location=yes,status=no,resizable=yes,scrollbars=yes"
        );
        if (!popup) {{
            app.location.href = oauthUrl;
        }}
    }}

    window.__copyrightDetectiveAuth = {{
        appWindow,
        appStorage,
        applyTokensToUrl,
        readCachedAuth,
        readPkceVerifier,
        persistPkceVerifier,
        restoreSessionInWindow,
        captureHashTokens,
        captureOAuthReturn,
        forwardAuthorizationCodeToOpener,
        persistAuthCache,
        installOAuthListener,
        restorePkceVerifierToUrl,
        persistReturnUrl,
        restoreReturnUrlAfterLogin,
        signInWithGitHub,
        deliverOAuthToOpener,
    }};
}})();
"""


def _oauth_query_param(name: str) -> Optional[str]:
    raw = st.query_params.get(name)
    if raw is None:
        return None
    if isinstance(raw, list):
        value = raw[0] if raw else ""
    else:
        value = raw
    text = str(value).strip()
    return text or None


def _has_oauth_callback_params() -> bool:
    return bool(_oauth_query_param("code") or _oauth_query_param(OAUTH_TOKEN_PARAM) or _oauth_query_param("access_token"))


def _auth_iframe(html: str, *, height: int = 1) -> None:
    iframe_fn = getattr(st, "iframe", None)
    if callable(iframe_fn):
        iframe_fn(html, height=max(height, 1))
        return
    import streamlit.components.v1 as components

    components.html(html, height=max(height, 1), scrolling=False)


def is_logged_in() -> bool:
    return bool(st.session_state.get("user_id") and st.session_state.get("access_token"))


def _set_user_session(user, access_token: str, refresh_token: str) -> None:
    metadata = user.user_metadata or {}
    st.session_state.user_id = user.id
    st.session_state.user_email = user.email or ""
    st.session_state.user_name = (
        metadata.get("user_name")
        or metadata.get("preferred_username")
        or metadata.get("name")
        or user.email
        or "GitHub User"
    )
    st.session_state.access_token = access_token
    st.session_state.refresh_token = refresh_token or ""
    st.session_state.pop("_user_settings_loaded", None)
    st.session_state.pop("_browser_auth_synced", None)


def _force_logout_local() -> None:
    for key in SESSION_AUTH_KEYS:
        st.session_state.pop(key, None)
    st.session_state["_clear_browser_auth_pending"] = True
    clear_secure_keys()


def _run_auth_client(
    *,
    capture_hash: bool = False,
    install_listener: bool = False,
    restore_cache: bool = False,
    restore_pkce: bool = False,
    restore_return_url: bool = False,
) -> None:
    flags = []
    if install_listener:
        flags.append("bridge.installOAuthListener(app);")
    if capture_hash:
        flags.append("bridge.captureOAuthReturn(app);")
    if restore_pkce:
        flags.append("bridge.restorePkceVerifierToUrl(app);")
    if restore_return_url:
        flags.append("bridge.restoreReturnUrlAfterLogin(app);")
    if restore_cache:
        flags.append(
            "if (!window.opener && !(app.location.search || '').includes('"
            + OAUTH_TOKEN_PARAM
            + "=') && !(app.location.search || '').includes('access_token=') "
            "&& !(app.location.search || '').includes('code=')) "
            "{ bridge.restoreSessionInWindow(app); }"
        )
    body = "\n            ".join(flags) if flags else ""
    _auth_iframe(
        f"""
        <script>
        {_AUTH_CLIENT_SCRIPT}
        (function () {{
            const bridge = window.__copyrightDetectiveAuth;
            if (!bridge) return;
            const app = bridge.appWindow(window);
            {body}
        }})();
        </script>
        """,
    )


def install_oauth_popup_listener() -> None:
    if not auth_enabled():
        return
    _run_auth_client(install_listener=True)


def capture_oauth_tokens_from_url_hash() -> None:
    if not auth_enabled() or is_logged_in() or _has_oauth_callback_params():
        return
    _run_auth_client(capture_hash=True)


def restore_auth_from_browser() -> None:
    if not auth_enabled() or is_logged_in() or _has_oauth_callback_params():
        return
    _run_auth_client(restore_cache=True)


def sync_auth_browser_storage() -> None:
    if st.session_state.pop("_clear_browser_auth_pending", False):
        _clear_pkce_browser_storage()
        _auth_iframe(
            f"""
            <script>
            {_AUTH_CLIENT_SCRIPT}
            (function () {{
                const storage = window.__copyrightDetectiveAuth && window.__copyrightDetectiveAuth.appStorage
                    ? window.__copyrightDetectiveAuth.appStorage()
                    : null;
                if (storage) {{
                    try {{ storage.removeItem({json.dumps(AUTH_BROWSER_STORAGE_KEY)}); }} catch (e) {{}}
                }}
            }})();
            </script>
            """,
        )
        return

    if not is_logged_in():
        return

    access_token = st.session_state.get("access_token") or ""
    refresh_token = st.session_state.get("refresh_token") or ""
    sync_key = f"{access_token[:24]}:{refresh_token[:24]}"
    if st.session_state.get("_browser_auth_synced") == sync_key:
        return

    payload = json.dumps(
        {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user_id": st.session_state.get("user_id") or "",
        }
    )
    _auth_iframe(
        f"""
        <script>
        {_AUTH_CLIENT_SCRIPT}
        (function () {{
            const storage = window.__copyrightDetectiveAuth && window.__copyrightDetectiveAuth.appStorage
                ? window.__copyrightDetectiveAuth.appStorage()
                : null;
            if (storage) {{
                try {{
                    storage.setItem({json.dumps(AUTH_BROWSER_STORAGE_KEY)}, {json.dumps(payload)});
                }} catch (e) {{}}
            }}
        }})();
        </script>
        """,
    )
    st.session_state["_browser_auth_synced"] = sync_key


def _clear_oauth_query_params() -> None:
    for param in OAUTH_CALLBACK_PARAMS:
        if param in st.query_params:
            del st.query_params[param]


def _complete_login(user, access_token: str, refresh_token: str) -> None:
    st.session_state.pop("_pkce_restore_attempted", None)
    _set_user_session(user, access_token, refresh_token)
    _clear_oauth_query_params()
    _clear_pkce_state()
    _clear_pkce_browser_storage()
    sync_auth_browser_storage()
    load_user_settings(force=True)
    _run_auth_client(restore_return_url=True)
    st.rerun()


def _user_from_auth_response(response: Any) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    session = getattr(response, "session", None)
    if session is None and isinstance(response, dict):
        session = response.get("session")
    user = getattr(response, "user", None)
    if user is None and session is not None:
        user = getattr(session, "user", None)
    if session is None:
        return user, None, None
    access_token = getattr(session, "access_token", None) or ""
    refresh_token = getattr(session, "refresh_token", None) or ""
    return user, access_token, refresh_token


def handle_oauth_callback() -> None:
    if not auth_enabled():
        return

    auth_code = _oauth_query_param("code")
    if auth_code:
        if is_logged_in():
            _clear_oauth_query_params()
            return

        # Popup callback: forward ?code= to opener via JS; do not consume the code in this tab.
        if _oauth_query_param(OAUTH_POPUP_FLAG) == "popup":
            _run_auth_client(capture_hash=True)
            return

        code_verifier = _get_pkce_verifier()
        if not code_verifier:
            st.error(
                "GitHub sign-in expired before completion. Please click Sign in with GitHub again."
            )
            _clear_oauth_query_params()
            st.session_state.pop("_pkce_restore_attempted", None)
            return

        try:
            client = create_supabase_client()
            response = client.auth.exchange_code_for_session(
                {"auth_code": auth_code, "code_verifier": code_verifier}
            )
            user, access_token, refresh_token = _user_from_auth_response(response)
            if user is None or not access_token:
                user_response = client.auth.get_user()
                user = user_response.user if user_response else None
            if user is None or not access_token:
                st.error("GitHub sign-in failed: could not retrieve user information.")
                _clear_oauth_query_params()
                return
            _complete_login(user, access_token, refresh_token or "")
        except Exception as exc:
            _clear_oauth_query_params()
            _clear_pkce_state()
            _clear_pkce_browser_storage()
            st.session_state["_clear_browser_auth_pending"] = True
            st.error(f"GitHub sign-in callback failed: {exc}")
        return

    access_token = _oauth_query_param(OAUTH_TOKEN_PARAM) or _oauth_query_param("access_token")
    if not access_token:
        return

    refresh_token = _oauth_query_param(OAUTH_REFRESH_PARAM) or _oauth_query_param("refresh_token") or ""

    if st.session_state.get("user_id") and st.session_state.get("access_token") == access_token:
        _clear_oauth_query_params()
        return

    try:
        client = create_supabase_client()
        session = client.auth.set_session(access_token, refresh_token)
        user = session.user if session else None
        if user is None:
            user_response = client.auth.get_user()
            user = user_response.user if user_response else None
        if user is None:
            st.error("GitHub sign-in failed: could not retrieve user information.")
            return

        _complete_login(user, access_token, refresh_token)
    except Exception as exc:
        _clear_oauth_query_params()
        st.session_state["_clear_browser_auth_pending"] = True
        st.error(f"GitHub sign-in callback failed: {exc}")


def ensure_valid_session() -> bool:
    if not is_logged_in():
        return False

    access_token = st.session_state.get("access_token") or ""
    refresh_token = st.session_state.get("refresh_token") or ""

    try:
        client = create_supabase_client()
        client.auth.set_session(access_token, refresh_token)
        user_response = client.auth.get_user()
        if user_response and user_response.user:
            return True
    except Exception:
        pass

    if not refresh_token:
        return True

    try:
        client = create_supabase_client()
        refreshed = client.auth.refresh_session(refresh_token)
        session = refreshed.session if refreshed else None
        user = refreshed.user if refreshed else None
        if session and user:
            _set_user_session(user, session.access_token, session.refresh_token or refresh_token)
            return True
    except Exception:
        pass

    return True


def get_oauth_redirect_url(provider: str = "github", *, for_popup: bool = False) -> Optional[str]:
    if not auth_enabled():
        return None
    url_key, verifier_key = _pkce_cache_keys(provider, for_popup=for_popup)
    cached_url = st.session_state.get(url_key)
    cached_verifier = st.session_state.get(verifier_key)
    if cached_url and cached_verifier:
        return str(cached_url)

    redirect_to = get_app_url()
    if for_popup:
        joiner = "&" if "?" in redirect_to else "?"
        redirect_to = f"{redirect_to}{joiner}{OAUTH_POPUP_FLAG}=popup"
    try:
        client = create_supabase_client()
        response = client.auth.sign_in_with_oauth(
            {"provider": provider, "options": {"redirect_to": redirect_to}}
        )
        url = response.url
        code_verifier = _read_pkce_verifier_from_client(client)
        if not url or not code_verifier:
            st.error("Failed to start GitHub sign-in: missing PKCE verifier.")
            return None
        url = _embed_verifier_in_oauth_url(str(url), code_verifier)
        st.session_state[url_key] = url
        st.session_state[verifier_key] = code_verifier
        return url
    except Exception as exc:
        st.error(f"Failed to start {provider} sign-in: {exc}")
        return None


def logout() -> None:
    try:
        client = get_authenticated_client()
        if client is not None:
            client.auth.sign_out()
    except Exception:
        pass

    _force_logout_local()
    _clear_pkce_state()
    _clear_pkce_browser_storage()
    st.rerun()


def _handle_missing_pkce_on_callback() -> bool:
    """Legacy fallback when callback URL lacks cd_cv. Returns True if init_auth should stop."""
    if not _oauth_query_param("code") or is_logged_in():
        return False
    if _oauth_query_param(OAUTH_POPUP_FLAG) == "popup":
        return False
    if _get_pkce_verifier():
        return False

    if not st.session_state.get("_pkce_restore_attempted"):
        st.session_state["_pkce_restore_attempted"] = True
        _run_auth_client(restore_pkce=True)
        return True

    st.session_state.pop("_pkce_restore_attempted", None)
    st.error("GitHub sign-in expired before completion. Please click Sign in with GitHub again.")
    _clear_oauth_query_params()
    return True


def init_auth() -> None:
    if not auth_enabled():
        return

    install_oauth_popup_listener()

    if _handle_missing_pkce_on_callback():
        return

    handle_oauth_callback()
    capture_oauth_tokens_from_url_hash()
    restore_auth_from_browser()

    if is_logged_in() and ensure_valid_session():
        load_user_settings()

    sync_auth_browser_storage()
    st.session_state.pop("_pkce_restore_attempted", None)


def _bind_return_url_on_sign_in_click(login_url: str) -> None:
    login_url_json = json.dumps(login_url)
    _auth_iframe(
        f"""
        <script>
        {_AUTH_CLIENT_SCRIPT}
        (function () {{
            const bridge = window.__copyrightDetectiveAuth;
            if (!bridge) return;
            const app = bridge.appWindow(window.top);
            const loginUrl = {login_url_json};
            const sidebar = app.document.querySelector('section[data-testid="stSidebar"]');
            if (!sidebar) return;
            sidebar.querySelectorAll('[data-testid="stLinkButton"] a').forEach(function (link) {{
                if (link.dataset.cdAuthBound) return;
                const href = link.getAttribute("href") || "";
                if (href !== loginUrl) return;
                link.classList.add("github-sign-in-btn");
                link.dataset.cdAuthBound = "1";
                link.setAttribute("aria-label", "Sign in with GitHub");
                link.addEventListener("click", function () {{
                    try {{
                        bridge.persistReturnUrl(app.location.href.split("#")[0]);
                    }} catch (e) {{}}
                }});
            }});
        }})();
        </script>
        """,
        height=1,
    )


def render_github_sign_in_button(*, disabled: bool = False) -> None:
    if _oauth_callback_in_progress():
        return

    login_url = get_oauth_redirect_url("github", for_popup=False)
    if not login_url:
        return

    st.link_button(
        "Sign in with GitHub",
        login_url,
        width="stretch",
        disabled=disabled,
    )
    if not disabled:
        _bind_return_url_on_sign_in_click(login_url)


def render_api_configuration_auth(*, disabled: bool = False) -> None:
    """Sign-in status, GitHub sign-in / log out inside API Configuration."""
    if not auth_enabled():
        return

    if is_logged_in():
        email = st.session_state.get("user_email") or st.session_state.get("user_name") or "your account"
        st.caption(f"Signed in as {email}")
        if st.button("Log out", width="stretch", disabled=disabled, key="auth_logout_btn", type="secondary"):
            logout()
    else:
        if _oauth_callback_in_progress():
            st.caption("Signing in with GitHub…")
        else:
            st.caption("Not signed in. Sign in with GitHub to keep your API key for future sessions.")
            render_github_sign_in_button(disabled=disabled)


def render_keep_button(*, disabled: bool = False, provider: str = "", model_name: str = "", api_keys: Optional[dict] = None) -> None:
    """Keep button to save, update, or clear encrypted API keys for the signed-in user."""
    if not auth_enabled():
        return

    if st.button("Keep", width="stretch", disabled=disabled, key="auth_keep_btn", type="primary"):
        if not is_logged_in():
            st.warning("Please sign in with GitHub before keeping your API key.")
            return
        if not ensure_valid_session():
            st.warning("Your session expired. Please sign in again.")
            return
        try:
            result = save_user_settings(provider or "Kimi", model_name or "", api_keys or {})
            if result == "cleared":
                st.success("Saved API keys cleared.")
            elif result == "updated":
                st.success("Saved API keys updated.")
            else:
                st.success("API keys saved.")
            st.rerun()
        except ValueError as exc:
            st.warning(str(exc))
        except RuntimeError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Failed to save API key: {exc}")
