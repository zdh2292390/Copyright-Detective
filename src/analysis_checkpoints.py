"""Owner-scoped durable API-analysis checkpoints; no Streamlit calls in workers.

The caller must verify the Supabase user before constructing a service-role store.
Credentials belong to the worker, never to task settings, source, or item payloads.
Install ``supabase/analysis_checkpoints.sql`` before using this store. Database
leases fence stale writers; a crash after an API response but before its commit
can still require repeating that unfinished request.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qsl, urlsplit
from uuid import UUID, uuid4


_SQLSTATE_CLASSES = frozenset({
    "00", "01", "02", "03", "08", "09", "0A", "0B", "0F", "0L", "0P", "0Z",
    "20", "21", "22", "23", "24", "25", "26", "27", "28", "2B", "2D", "2F",
    "34", "38", "39", "3B", "3D", "3F", "40", "42", "44", "53", "54", "55",
    "57", "58", "F0", "HV", "P0", "XX",
})


def _safe_backend_code(value: Any) -> str | None:
    # Backend details may contain credentials or arbitrary user content. Only
    # recognize the fixed PostgreSQL/PostgREST diagnostic-code formats.
    if not isinstance(value, str):
        return None
    if re.fullmatch(r"PGRST[0-9]{3}", value):
        return value
    if re.fullmatch(r"[A-Z0-9]{5}", value) and value[:2] in _SQLSTATE_CLASSES:
        return value
    return None


class AnalysisCheckpointError(RuntimeError):
    """Safe checkpoint failure with an optional backend diagnostic code.

    ``kind`` selects a recovery action. Raw backend messages, hints and request
    URLs must never become this error's user-visible text.
    """

    def __init__(self, message: str, *, kind: str = "checkpoint", code: str | None = None,
                 status_code: int | None = None):
        self.kind = kind
        self.code = _safe_backend_code(code)
        self.status_code = status_code if type(status_code) is int and 100 <= status_code <= 599 else None
        diagnostic = f" (code: {self.code})" if self.code else ""
        super().__init__(message + diagnostic)


class AnalysisLeaseError(AnalysisCheckpointError):
    """Another worker owns the task, or this worker's lease has expired."""


def classify_checkpoint_error(exc: Exception, *, operation: str = "write") -> AnalysisCheckpointError:
    """Translate storage/auth/network failures into fixed, actionable diagnostics.

    SQLSTATE and PostgREST codes are retained in safe fields and visible text.
    Messages are inspected only for classification; they are never interpolated.
    This also handles HTTP clients whose error carries an HTTP response instead
    of a PostgREST ``code`` attribute.
    """
    if isinstance(exc, AnalysisCheckpointError):
        return exc
    code = None
    status = None
    messages: list[str] = []
    raw_codes: list[str] = []
    errors: list[BaseException] = []
    current: BaseException | None = exc
    # Connection/auth helpers sometimes wrap the SDK exception. Inspect only
    # explicit causes, with a bound and cycle guard, to retain its safe code.
    for _ in range(4):
        if current is None or any(current is previous for previous in errors):
            break
        if isinstance(current, AnalysisCheckpointError):
            return current
        errors.append(current)
        raw_code = getattr(current, "code", None)
        if isinstance(raw_code, str):
            raw_codes.append(raw_code)
        code = code or _safe_backend_code(raw_code)
        raw_message = getattr(current, "message", None)
        if not isinstance(raw_message, str):
            raw_message = str(current)
        messages.append(raw_message[:8192].lower())
        current_status = getattr(current, "status_code", None)
        if type(current_status) is not int:
            current_status = getattr(current, "status", None)  # Supabase AuthApiError
        if type(current_status) is not int:
            current_status = getattr(getattr(current, "response", None), "status_code", None)
        if status is None and type(current_status) is int and 100 <= current_status <= 599:
            status = current_status
        # SDK auth/gateway failures may have an HTTP string code instead of a
        # response. It is used for classification, never displayed raw.
        if status is None and isinstance(raw_code, str) and raw_code in {"401", "403", "408", "429", "500", "502", "503", "504"}:
            status = int(raw_code)
        current = current.__cause__
    message = "\n".join(messages)
    def failure(text: str, kind: str, error_type: type[AnalysisCheckpointError] = AnalysisCheckpointError):
        return error_type(text, kind=kind, code=code, status_code=status)

    if "analysis_untrusted_game" in message:
        return failure(
            "This older official-game checkpoint has no verified server origin and cannot be replayed. "
            "Saved official scores remain available; start a new game task.", "untrusted_game")
    if "analysis_official_game" in message:
        return failure(
            "Official game checkpoints require the server's Supabase service-role configuration.", "official_game")
    if "analysis_lease" in message:
        return failure(
            "The analysis task is owned by another worker or its lease expired.", "lease", AnalysisLeaseError)
    if code in {"PGRST202", "PGRST204", "PGRST205", "42P01", "42703", "42883"}:
        return failure(
            "Cloud checkpoint schema is unavailable or outdated. Run the full supabase/analysis_checkpoints.sql "
            "migration in the app's Supabase project, then reload its PostgREST schema cache.", "schema")
    if any(marker in message for marker in ("invalid api key", "invalid apikey", "invalid supabase_key", "invalid supabase key")):
        return failure(
            "The Supabase API key was rejected. Check that the app's Supabase URL and API key belong to "
            "the same project, then restart the app.", "api_key")
    if code in {"PGRST301", "PGRST303", "28000", "28P01"} or status == 401 or any(
        marker in message for marker in ("jwt expired", "jwt is expired", "invalid jwt", "expired jwt", "invalid refresh token")
    ) or any(value in {"bad_jwt", "session_not_found", "refresh_token_not_found", "refresh_token_already_used"} for value in raw_codes):
        return failure(
            "Your Supabase login session is expired or invalid. Sign out and sign in with GitHub again "
            "before restoring or starting analysis.", "auth")
    if code == "42501" and "owner authorization failed" in message:
        return failure(
            "The checkpoint account does not match your Supabase login. Sign out and sign in with GitHub "
            "again before restoring or starting analysis.", "auth")
    if code in {"42501", "PGRST302"} or status == 403:
        return failure(
            "Cloud checkpoint permissions are missing. Reapply supabase/analysis_checkpoints.sql to "
            "restore the table grants and RPC permissions; keep row level security enabled.", "permission")
    network_type = any(cls.__name__ in {
        "RequestError", "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
        "NetworkError", "ReadError", "WriteError", "RemoteProtocolError", "ConnectionError", "TimeoutError",
    } for error in errors for cls in type(error).__mro__)
    if code in {"PGRST000", "PGRST001", "PGRST002", "PGRST003", "08000", "08001", "08003", "08006", "53300", "57P01", "57P02", "57P03"} or network_type or status in {408, 429, 500, 502, 503, 504}:
        return failure(
            "Supabase checkpoint storage is temporarily unreachable. Check the project status and the "
            "app's network connection, then retry; saved progress remains available.", "network")
    return failure(
        "Cloud checkpoints could not be read. Check the Supabase connection and retry." if operation == "read"
        else "Cloud checkpoint operation failed; saved progress was not replaced. Check the Supabase connection and retry.",
        "checkpoint")


_CREDENTIAL_KEYS = frozenset({
    "apikey", "accesskey", "secretkey", "accesskeyid", "secretaccesskey",
    "accesstoken", "refreshtoken", "authorization", "password", "passwd",
    "clientsecret", "servicerolekey", "supabaseservicerolekey", "anonkey",
    "supabaseanonkey", "credentials", "credential", "bearertoken", "apikeys", "apitoken",
})
# All application aliases that execute official Game 1 scoring. These
# journals can only be written by the trusted server, never a browser client.
OFFICIAL_GAME_PAGE_KEYS = frozenset({
    'Game 1: The Hidden Passage Hunt',
    'Game 2: The Hidden Passage Hunt',
    'Copyright Challenge',
    'Copyright Challenge 1',
})
_TASK_STATUSES = frozenset({"creating", "queued", "running", "incomplete", "complete"})
_ITEM_STATUSES = frozenset({"pending", "complete", "failed"})
_MAX_ITEMS = 500_000
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_BATCH_BYTES = 4 * 1024 * 1024
_TASK_COLUMNS = (
    "id,owner_id,page_key,fingerprint,settings,source,metadata,dynamic_items,server_managed,"
    "total_items,status,completed_items,failed_items,stop_requested,"
    "lease_token,lease_generation,lease_expires_at,revision,created_at,updated_at"
)
_SUMMARY_COLUMNS = (
    "id,owner_id,page_key,settings,dynamic_items,server_managed,total_items,status,completed_items,"
    "failed_items,stop_requested,lease_expires_at,revision,created_at,updated_at"
)
_ITEM_COLUMNS = "task_id,owner_id,item_index,input,status,result,error,attempts,updated_at"


def _uuid(value: str, name: str) -> str:
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise AnalysisCheckpointError(f"{name} must be a valid UUID.") from exc


def _integer(value: Any, name: str, low: int = 0, high: int = _MAX_ITEMS) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise AnalysisCheckpointError(f"{name} must be an integer between {low} and {high}.")
    return value


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _check_credentials(value: Any) -> None:
    if isinstance(value, dict):
        for name, child in value.items():
            if not isinstance(name, str):
                raise AnalysisCheckpointError("Checkpoint JSON keys must be strings.")
            if (_key(name) in _CREDENTIAL_KEYS or re.search(
                    r"(?:apikeys?|apitoken|accesstoken|refreshtoken|secretkey|servicerolekey)$", _key(name))):
                raise AnalysisCheckpointError("Credentials cannot be stored in analysis checkpoints.")
            if isinstance(child, str) and (_key(name).endswith("url") or _key(name) in {"endpoint", "baseuri", "apibase"}):
                try:
                    parsed = urlsplit(child)
                    credential_query = any(_key(key) in _CREDENTIAL_KEYS for key, _ in parse_qsl(parsed.query))
                except ValueError as exc:
                    raise AnalysisCheckpointError("Checkpoint URL is invalid.") from exc
                if parsed.username or parsed.password or credential_query:
                    raise AnalysisCheckpointError("Credential-bearing URLs cannot be stored in checkpoints.")
            _check_credentials(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_credentials(child)


def _json(value: Any, name: str, *, object_only: bool = False) -> Any:
    if object_only and not isinstance(value, Mapping):
        raise AnalysisCheckpointError(f"{name} must be a JSON object.")
    try:
        # A serialization round trip detaches state and rejects NaN/custom objects.
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > _MAX_JSON_BYTES:
            raise AnalysisCheckpointError(f"{name} exceeds the checkpoint size limit.")
        copied = json.loads(encoded)
        _check_credentials(copied)
        return copied
    except AnalysisCheckpointError:
        raise
    except (TypeError, ValueError, RecursionError) as exc:
        raise AnalysisCheckpointError(f"{name} must contain finite JSON values.") from exc


def _data(response: Any) -> Any:
    return getattr(response, "data", response.get("data") if isinstance(response, dict) else None)


def _row(response: Any) -> dict[str, Any]:
    data = _data(response)
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    if not isinstance(data, dict):
        raise AnalysisCheckpointError("Checkpoint storage returned an invalid task response.")
    return data


def _rows(response: Any) -> list[dict[str, Any]]:
    data = _data(response)
    if data is None:
        return []
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        raise AnalysisCheckpointError("Checkpoint storage returned invalid rows.")
    return data


class SupabaseAnalysisCheckpointStore:
    """A fixed-owner store usable by background threads with an injected client.

    Even service-role reads include the owner predicate before execution. All
    mutations go through transactional RPCs that check the owner and lease.
    The instance does not read Streamlit state or cache any model API key.
    """

    def __init__(self, client: Any, owner_id: str, *, batch_size: int = 200):
        self.client = client
        self.owner_id = _uuid(owner_id, "owner_id")
        self.batch_size = _integer(batch_size, "batch_size", 1, 1000)

    def _rpc(self, method: str, **params: Any) -> dict[str, Any]:
        arguments = {"p_owner_id": self.owner_id, **params}
        try:
            return _row(self.client.rpc(f"analysis_{method}", arguments).execute())
        except AnalysisCheckpointError:
            raise
        except Exception as exc:
            raise classify_checkpoint_error(exc, operation="write") from exc

    def _query(self, query: Any) -> list[dict[str, Any]]:
        try:
            return _rows(query.execute())
        except AnalysisCheckpointError:
            raise
        except Exception as exc:
            raise classify_checkpoint_error(exc, operation="read") from exc

    def create_task(
        self, page_key: str, settings: Mapping[str, Any], source: Any,
        work_items: Iterable[Any], *, task_id: str | None = None,
        fingerprint: str | None = None, metadata: Mapping[str, Any] | None = None,
        dynamic_items: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(page_key, str) or not re.fullmatch(r"[A-Za-z0-9_ ./:-]{1,100}", page_key):
            raise AnalysisCheckpointError("page_key must be a short stable page identifier.")
        if not isinstance(dynamic_items, bool):
            raise AnalysisCheckpointError("dynamic_items must be a boolean.")
        safe_settings = _json(settings, "settings", object_only=True)
        safe_source = _json(source, "source")
        safe_metadata = _json(metadata or {}, "metadata", object_only=True)
        items = []
        digest = hashlib.sha256()
        for value in (page_key, safe_settings, safe_source, dynamic_items):
            digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\0")
        for index, payload in enumerate(work_items):
            _integer(index, "item index", 0, _MAX_ITEMS - 1)
            safe_payload = _json(payload, "work item")
            encoded = json.dumps(safe_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            if len(encoded) > _MAX_BATCH_BYTES:
                raise AnalysisCheckpointError("A work item exceeds the batch size limit.")
            digest.update(encoded)
            digest.update(b"\0")
            items.append({"index": index, "input": safe_payload})
        if dynamic_items and items:
            raise AnalysisCheckpointError("Dynamic tasks must start with an empty work item list.")
        if fingerprint is None:
            fingerprint = digest.hexdigest()
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[a-f0-9]{64}", fingerprint):
            raise AnalysisCheckpointError("fingerprint must be a lowercase SHA-256 digest.")
        identity = _uuid(task_id or str(uuid4()), "task_id")
        task = self._rpc("create_task", p_task_id=identity, p_page_key=page_key,
                         p_settings=safe_settings, p_source=safe_source,
                         p_total_items=len(items), p_fingerprint=fingerprint,
                         p_metadata=safe_metadata, p_dynamic_items=dynamic_items)
        # Creation can be retried with the same UUID after a partial upload.
        if task.get("status") != "creating":
            if not dynamic_items:
                # An explicit caller fingerprint must not conceal changed inputs.
                for offset in range(0, len(items), self.batch_size):
                    stored = self.load_items(identity, offset=offset, limit=self.batch_size)
                    expected = items[offset:offset + self.batch_size]
                    if len(stored) != len(expected) or any(
                        row.get("item_index") != item["index"] or row.get("input") != item["input"]
                        for row, item in zip(stored, expected)
                    ):
                        raise AnalysisCheckpointError("Analysis work inputs cannot be changed when resuming a task.")
            return task
        batch: list[dict[str, Any]] = []
        byte_count = 0
        for item in items:
            size = len(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            if batch and (len(batch) >= self.batch_size or byte_count + size > _MAX_BATCH_BYTES):
                self._rpc("put_items", p_task_id=identity, p_items=batch)
                batch, byte_count = [], 0
            batch.append(item)
            byte_count += size
        if batch:
            self._rpc("put_items", p_task_id=identity, p_items=batch)
        return self._rpc("finalize_task", p_task_id=identity)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        task_id = _uuid(task_id, "task_id")
        rows = self._query(self.client.table("analysis_tasks").select(_TASK_COLUMNS)
                           .eq("owner_id", self.owner_id).eq("id", task_id).limit(1))
        return rows[0] if rows else None

    def list_tasks(self, *, page_key: str | None = None, statuses: Iterable[str] | None = None,
                   offset: int = 0, limit: int = 50, summary_only: bool = False) -> list[dict[str, Any]]:
        offset = _integer(offset, "offset")
        limit = _integer(limit, "limit", 1, 1000)
        if not isinstance(summary_only, bool):
            raise AnalysisCheckpointError("summary_only must be a boolean.")
        columns = _SUMMARY_COLUMNS if summary_only else _TASK_COLUMNS
        query = self.client.table("analysis_tasks").select(columns).eq("owner_id", self.owner_id)
        if page_key is not None:
            query = query.eq("page_key", page_key)
        if statuses is not None:
            values = list(statuses)
            if any(value not in _TASK_STATUSES for value in values):
                raise AnalysisCheckpointError("Invalid task status filter.")
            if not values:
                return []
            query = query.in_("status", values)
        return self._query(query.order("updated_at", desc=True).order("id").range(offset, offset + limit - 1))

    def load_items(self, task_id: str, *, offset: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        task_id = _uuid(task_id, "task_id")
        offset = _integer(offset, "offset")
        limit = _integer(limit, "limit", 1, 1000)
        return self._query(self.client.table("analysis_items").select(_ITEM_COLUMNS)
                           .eq("owner_id", self.owner_id).eq("task_id", task_id)
                           .order("item_index").range(offset, offset + limit - 1))

    def claim(self, task_id: str, *, ttl_seconds: int = 180, lease_token: str | None = None) -> dict[str, Any]:
        return self._rpc("claim", p_task_id=_uuid(task_id, "task_id"),
                         p_lease_token=_uuid(lease_token or str(uuid4()), "lease_token"),
                         p_ttl_seconds=_integer(ttl_seconds, "ttl_seconds", 60, 900))

    def heartbeat(self, task_id: str, lease_token: str, *, ttl_seconds: int = 180) -> dict[str, Any]:
        return self._rpc("heartbeat", p_task_id=_uuid(task_id, "task_id"),
                         p_lease_token=_uuid(lease_token, "lease_token"),
                         p_ttl_seconds=_integer(ttl_seconds, "ttl_seconds", 60, 900))

    def append_item(self, task_id: str, index: int, payload: Any, *, lease_token: str) -> dict[str, Any]:
        return self._rpc("append_item", p_task_id=_uuid(task_id, "task_id"),
                         p_index=_integer(index, "index", 0, _MAX_ITEMS - 1),
                         p_input=_json(payload, "work item"),
                         p_lease_token=_uuid(lease_token, "lease_token"))

    def save_items(self, task_id: str, updates: Iterable[Mapping[str, Any]], *,
                   metadata: Mapping[str, Any] | None = None, status: str | None = None,
                   lease_token: str) -> dict[str, Any]:
        if status is not None and status not in {"running", "incomplete", "complete"}:
            raise AnalysisCheckpointError("Invalid saved task status.")
        safe_updates = []
        indices = set()
        for update in updates:
            if not isinstance(update, Mapping):
                raise AnalysisCheckpointError("Item updates must be JSON objects.")
            index = _integer(update.get("index"), "index", 0, _MAX_ITEMS - 1)
            if index in indices:
                raise AnalysisCheckpointError("A batch cannot update an item twice.")
            indices.add(index)
            item_status = update.get("status")
            if item_status not in _ITEM_STATUSES:
                raise AnalysisCheckpointError("Invalid item status.")
            result = _json(update.get("result"), "result")
            error = update.get("error")
            if error is not None and (not isinstance(error, str) or not error.strip() or len(error) > 16000):
                raise AnalysisCheckpointError("Item error must be a nonempty short string.")
            if ((item_status == "complete" and (result is None or error is not None))
                    or (item_status == "failed" and (result is not None or error is None))
                    or (item_status == "pending" and (result is not None or error is not None))):
                raise AnalysisCheckpointError("Item result/error is inconsistent with its status.")
            safe_updates.append({"index": index, "status": item_status, "result": result,
                                 "error": error, "attempts": _integer(update.get("attempts", 0), "attempts", 0, 1_000_000)})
        if len(safe_updates) > 1000 or len(json.dumps(safe_updates, ensure_ascii=False).encode("utf-8")) > _MAX_JSON_BYTES:
            raise AnalysisCheckpointError("Result batch exceeds the checkpoint size limit.")
        safe_metadata = None if metadata is None else _json(metadata, "metadata", object_only=True)
        return self._rpc("save_items", p_task_id=_uuid(task_id, "task_id"), p_updates=safe_updates,
                         p_metadata=safe_metadata, p_status=status,
                         p_lease_token=_uuid(lease_token, "lease_token"))

    def save_item(self, task_id: str, index: int, *, result: Any = None, error: str | None = None,
                  attempts: int = 0, metadata: Mapping[str, Any] | None = None,
                  status: str | None = None, task_status: str | None = None,
                  lease_token: str) -> dict[str, Any]:
        item_status = status or ("failed" if error is not None else "complete" if result is not None else "pending")
        return self.save_items(task_id, [{"index": index, "status": item_status, "result": result,
                                         "error": error, "attempts": attempts}],
                               metadata=metadata, status=task_status, lease_token=lease_token)

    def save_progress(self, task_id: str, *, metadata: Mapping[str, Any] | None = None,
                      status: str | None = None, lease_token: str) -> dict[str, Any]:
        return self.save_items(task_id, [], metadata=metadata, status=status, lease_token=lease_token)

    def release(self, task_id: str, lease_token: str, *, status: str = "incomplete",
                metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if status not in {"incomplete", "complete"}:
            raise AnalysisCheckpointError("A released task must be incomplete or complete.")
        return self._rpc("release", p_task_id=_uuid(task_id, "task_id"),
                         p_lease_token=_uuid(lease_token, "lease_token"), p_status=status,
                         p_metadata=None if metadata is None else _json(metadata, "metadata", object_only=True))

    def request_stop(self, task_id: str) -> dict[str, Any]:
        return self._rpc("request_stop", p_task_id=_uuid(task_id, "task_id"))

    def delete(self, task_id: str) -> None:
        self._rpc("delete", p_task_id=_uuid(task_id, "task_id"))