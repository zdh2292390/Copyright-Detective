"""ASU Research Computing's free, fixed-origin OpenAI-compatible gateway."""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
import json
import re
from threading import RLock
from time import monotonic

import requests

ASU_PROVIDER = "ASU Research Computing API"
ASU_BASE_URL = "https://openai.rc.asu.edu/v1"
ASU_DEFAULT_MODEL = "gemma4-31b-it"
ASU_DOCS_URL = "https://docs.rc.asu.edu/ai/api/"
ASU_FREE_SERVICE_URL = "https://docs.rc.asu.edu/ai/getting-started/"
ASU_MODEL_IDS = (
    "gemma4-31b-it", "glm-5-2", "glm-5-3", "glm-5-3-flash",
    "kimi-k2-7-code", "muse-glimmer-30b", "north-mini-code", "qwen38-27b",
    "deepseek-v4-1-flash", "devstral2-123b", "gemma3-27b-it", "glm-4-5v",
    "gpt-oss-120b", "granite41-30b", "hemmingway-1", "inkling-small",
    "laguna-s-2-1", "llama3-groq-70b-tool-use", "llama4-maverick-17b",
    "llama4-scout-17b", "mimo-v2-6-flash", "minimax-m2-7", "minimax-m3",
    "north-mini-code-glm", "ornith-1-0-35b", "ornith-1-5-35b-a3b",
    "qwen-agentworld-35b-a3b", "qwen3-coder-30b-a3b-instruct", "qwen3-coder-next",
    "qwen35-122b-a10b", "qwen35-27b", "qwen35-35b-a3b", "qwen35-397b-a17b",
    "qwen36-27b", "qwen36-35b-a3b", "router-classifier-v2",
)
# Only a digest of the account key indexes the bounded process-local cache.
# Neither credentials nor model catalogs are saved to analysis checkpoints.
_CACHE_TTL = 300
_ERROR_TTL = 30
_CACHE_LIMIT = 32
_MAX_CATALOG_BYTES = 2 * 1024 * 1024
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,255}")
_MODEL_CACHE = OrderedDict()
_MODEL_CACHE_LOCK = RLock()


class AsuCatalogError(ValueError):
    """Safe UI feedback without HTTP bodies, tokens or credential-bearing URLs."""


def _parse_model_ids(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise AsuCatalogError("ASU returned an invalid model catalog. You can enter a model ID from the ASU portal.")
    models = []
    seen = set()
    for entry in payload["data"]:
        model = entry.get("id") if isinstance(entry, dict) else None
        if not isinstance(model, str) or _MODEL_ID.fullmatch(model) is None:
            continue
        if model not in seen:
            models.append(model)
            seen.add(model)
        if len(models) >= 2000:
            break
    if not models:
        raise AsuCatalogError("ASU returned no usable models. Check your key and the model list in the ASU portal.")
    return tuple(models)


def _remember(cache_key, *, models=None, error=None):
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[cache_key] = (monotonic() + (_ERROR_TTL if error else _CACHE_TTL), models, error)
        _MODEL_CACHE.move_to_end(cache_key)
        while len(_MODEL_CACHE) > _CACHE_LIMIT:
            _MODEL_CACHE.popitem(last=False)


def fetch_asu_models(api_key):
    """Read available IDs from ASU /models; no inference requests or paid fallback.

    A failed fetch preserves UI choice via its caller's portal catalog/custom ID.
    Requests never follow redirects, keeping account credentials on ASU's origin.
    """
    if not isinstance(api_key, str) or not api_key.strip():
        raise AsuCatalogError("Enter your ASU API key to load the current model catalog.")
    credential = api_key.strip()
    if any(char in credential for char in "\r\n"):
        raise AsuCatalogError("The ASU API key contains invalid characters. Paste only the key value.")
    cache_key = sha256(credential.encode()).digest()
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached and cached[0] > monotonic():
            _MODEL_CACHE.move_to_end(cache_key)
            if cached[2]:
                raise AsuCatalogError(cached[2])
            return list(cached[1])
        _MODEL_CACHE.pop(cache_key, None)
    try:
        with requests.get(
            ASU_BASE_URL + "/models", headers={"Authorization": "Bearer " + credential},
            timeout=(3.05, 5), allow_redirects=False, stream=True,
        ) as response:
            code = response.status_code
            if code in (401, 403):
                raise AsuCatalogError("ASU rejected this key. Check that it is valid and has LLM API access.")
            if code == 429:
                raise AsuCatalogError("ASU's model catalog is rate limited. Retry shortly or choose a portal model below.")
            if code != 200:
                raise AsuCatalogError("ASU's model catalog is temporarily unavailable. Choose a portal model or enter its ID.")
            body = bytearray()
            for chunk in response.iter_content(chunk_size=65536):
                body.extend(chunk)
                if len(body) > _MAX_CATALOG_BYTES:
                    raise AsuCatalogError("ASU's model catalog is too large. Enter a model ID from the portal instead.")
            models = _parse_model_ids(json.loads(body))
    except AsuCatalogError as exc:
        _remember(cache_key, error=str(exc))
        raise
    except (requests.RequestException, ValueError, TypeError):
        message = "Could not load ASU's model catalog. You can still select a portal model or enter its ID."
        _remember(cache_key, error=message)
        raise AsuCatalogError(message) from None
    _remember(cache_key, models=models)
    return list(models)
