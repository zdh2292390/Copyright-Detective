"""Small, credential-free runtime summaries for saved analysis cards."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit


def _label(value: Any, limit: int = 256) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value[:limit] or None


def _endpoint(value: Any) -> str | None:
    """Never duplicate URL credentials, query parameters, or fragments."""
    if not isinstance(value, str) or len(value) > 4096:
        return None
    try:
        parsed = urlsplit(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))
    except ValueError:
        return None


def _api(payload: dict) -> dict | None:
    endpoint = next((safe for name in ("base_url", "endpoint", "agent_url")
                     if (safe := _endpoint(payload.get(name)))), None)
    provider = _label(payload.get("provider"))
    model = _label(payload.get("model")) or _label(payload.get("model_name"))
    if payload.get("agent_url"):
        provider = provider or "Deployment agent"
        model = model or _label(payload.get("model_path"))
    elif not provider and endpoint:
        hostname = urlsplit(endpoint).hostname
        provider = {
            "api.openai.com": "OpenAI", "api.anthropic.com": "Anthropic",
            "openrouter.ai": "OpenRouter", "api.moonshot.cn": "Kimi",
            "api.moonshot.ai": "Kimi", "localhost": "Local vLLM",
            "127.0.0.1": "Local vLLM", "::1": "Local vLLM",
        }.get(hostname, "OpenAI-compatible API")
    if not any((provider, model, endpoint)):
        return None
    return {"provider": provider, "model": model, "endpoint": endpoint}


def update_task_display(previous_display: Any, request: dict, index: int, phase: str) -> dict:
    """Describe one actual journal call without copying any request content."""
    if phase not in {"running", "complete", "failed"}:
        raise ValueError("Unknown analysis step phase.")
    previous = previous_display if isinstance(previous_display, dict) else {}
    apis = []
    previous_apis = previous.get("apis") or []
    for value in previous_apis if isinstance(previous_apis, list) else []:
        if isinstance(value, dict):
            api = _api(value)
            if api is not None and api not in apis:
                apis.append(api)
    payload = request.get("request")
    if isinstance(payload, dict):
        api = _api(payload)
        if api is not None and api not in apis:
            apis.append(api)
    scope = request.get("scope") or {}
    path = scope.get("path", []) if isinstance(scope, dict) else []
    return {
        "apis": apis,
        "current_step": {
            "operation": _label(request.get("operation")) or "Analysis call",
            "scope": [part if type(part) is int else _label(part)
                      for part in path if type(part) in (str, int)] if isinstance(path, list) else [],
            "index": index,
            "phase": phase,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def summarize_saved_items(items: Iterable[dict]) -> dict:
    """Recover display details from legacy journal rows, using only safe fields."""
    display = {"apis": []}
    current_step = None
    newest_time = None
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        request = item.get("payload") or item.get("input")
        if not isinstance(request, dict):
            continue
        index = item.get("index", item.get("item_index", position))
        if type(index) is not int or index < 0:
            continue
        phase = item.get("status")
        phase = phase if phase in {"complete", "failed"} else "running"
        display = update_task_display(display, request, index, phase)
        step = display["current_step"]
        timestamp = item.get("updated_at")
        try:
            modified = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            modified = modified.replace(tzinfo=timezone.utc) if modified.tzinfo is None else modified.astimezone(timezone.utc)
        except (ValueError, TypeError, AttributeError):
            modified = None
        if modified is not None:
            step["updated_at"] = modified.isoformat()
        else:
            step["updated_at"] = None
        if current_step is None or (modified is not None and (newest_time is None or modified >= newest_time)) or (modified is None and newest_time is None):
            current_step = step
            newest_time = modified
    if current_step is not None:
        display["current_step"] = current_step
    return display
