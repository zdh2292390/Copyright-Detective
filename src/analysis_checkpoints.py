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


class AnalysisCheckpointError(RuntimeError):
    """Checkpoint storage is unavailable or a checkpoint is invalid."""


class AnalysisLeaseError(AnalysisCheckpointError):
    """Another worker owns the task, or this worker's lease has expired."""


_CREDENTIAL_KEYS = frozenset({
    "apikey", "accesskey", "secretkey", "accesskeyid", "secretaccesskey",
    "accesstoken", "refreshtoken", "authorization", "password", "passwd",
    "clientsecret", "servicerolekey", "supabaseservicerolekey", "anonkey",
    "supabaseanonkey", "credentials", "credential", "bearertoken", "apikeys", "apitoken",
})
_TASK_STATUSES = frozenset({"creating", "queued", "running", "incomplete", "complete"})
_ITEM_STATUSES = frozenset({"pending", "complete", "failed"})
_MAX_ITEMS = 500_000
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_BATCH_BYTES = 4 * 1024 * 1024
_TASK_COLUMNS = (
    "id,owner_id,page_key,fingerprint,settings,source,metadata,dynamic_items,"
    "total_items,status,completed_items,failed_items,stop_requested,"
    "lease_token,lease_generation,lease_expires_at,revision,created_at,updated_at"
)
_SUMMARY_COLUMNS = (
    "id,owner_id,page_key,settings,dynamic_items,total_items,status,completed_items,"
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
            code = str(getattr(exc, "code", ""))
            message = str(getattr(exc, "message", ""))
            if "analysis_lease" in message:
                raise AnalysisLeaseError("The analysis task is owned by another worker or its lease expired.") from exc
            if code in {"PGRST202", "PGRST204", "PGRST205", "42P01", "42883"}:
                raise AnalysisCheckpointError(
                    "Cloud checkpoint schema is unavailable. Apply supabase/analysis_checkpoints.sql before running analyses."
                ) from exc
            raise AnalysisCheckpointError("Cloud checkpoint operation failed; saved progress was not replaced.") from exc

    def _query(self, query: Any) -> list[dict[str, Any]]:
        try:
            return _rows(query.execute())
        except AnalysisCheckpointError:
            raise
        except Exception as exc:
            code = str(getattr(exc, "code", ""))
            if code in {"PGRST202", "PGRST204", "PGRST205", "42P01", "42883"}:
                raise AnalysisCheckpointError(
                    "Cloud checkpoint schema is unavailable. Apply supabase/analysis_checkpoints.sql before running analyses."
                ) from exc
            raise AnalysisCheckpointError("Cloud checkpoints could not be read.") from exc

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