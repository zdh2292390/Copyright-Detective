"""Account-scoped document checkpoints backed by the shared Supabase task store.

Only source chunks, whitelisted generation settings and analysis results are
persisted. Provider credentials and worker lease tokens remain in memory.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import json
from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from uuid import UUID

from src.document_checkpoints import (
    CheckpointError, DocumentCheckpointStore, _chunk_rows, _integer, _metadata,
    _pair_hash, _pairs, _read_json,
)


DOCUMENT_PAGE_KEY = "document_memorization"
LEASE_SECONDS = 180
_CHANGED_SNAPSHOT = object()
DEFAULT_CACHED_TASKS = 4
DEFAULT_CACHE_BYTES = 64 * 1024 * 1024
WRITE_BATCH_ITEMS = 1000
WRITE_BATCH_BYTES = 4 * 1024 * 1024
MAX_UPDATE_BYTES = 16 * 1024 * 1024


def is_cloud_document_token(token):
    if not isinstance(token, str):
        return False
    try:
        return str(UUID(token)) == token
    except (ValueError, AttributeError):
        return False


def _active(task):
    expiry = task.get("lease_expires_at")
    if not expiry or not task.get("lease_token"):
        return False
    try:
        end = datetime.fromisoformat(str(expiry).replace("Z", "+00:00"))
        if end.tzinfo is None:
            raise ValueError("Timezone missing")
        return end > datetime.now(timezone.utc)
    except (TypeError, ValueError):
        raise CheckpointError("The saved cloud analysis has an invalid worker lease.")


class SupabaseDocumentCheckpointStore:
    """Adapt the SQLite checkpoint contract to an authenticated account store."""

    remote = True

    def __init__(self, store, *, max_cached_tasks=DEFAULT_CACHED_TASKS, max_cache_bytes=DEFAULT_CACHE_BYTES):
        owner = getattr(store, "owner_id", None)
        if not isinstance(owner, str) or not owner.strip():
            raise CheckpointError("A verified account is required for cloud document recovery.")
        if type(max_cached_tasks) is not int or max_cached_tasks < 1 or type(max_cache_bytes) is not int or max_cache_bytes < 1:
            raise ValueError("Document cache limits must be positive integers.")
        self.owner_id = owner
        self.store = store
        self._max_cached_tasks = max_cached_tasks
        self._max_cache_bytes = max_cache_bytes
        self._cache_order = OrderedDict()
        self._source_bytes = {}
        self._row_bytes = {}
        self._state_bytes = {}
        self._identity_bytes = {}
        self._known_revisions = {}
        self._lock = RLock()
        self._leases = {}
        self._sources = {}
        self._identities = {}
        self._persisted = {}
        self._states = {}
        self._revisions = {}

    def bind_store(self, store):
        """Refresh this owner's client without replacing worker lease state."""
        if getattr(store, "owner_id", None) != self.owner_id:
            raise CheckpointError("The cloud document client has a different owner.")
        with self._lock:
            self.store = store

    @staticmethod
    def _row_size(row):
        if row is None:
            return 0
        result, error, _, _ = row
        return 128 + 4 * (len(result or "") + len(error or ""))

    @staticmethod
    def _identity_size(metadata):
        settings = metadata.get("settings", {})
        return 1024 + 4 * (
            sum(len(value) for value in metadata.values() if isinstance(value, str))
            + sum(len(value) for value in settings.values() if isinstance(value, str))
        )

    @staticmethod
    def _state_size(state):
        # Conservative payload estimates avoid serializing/copying all results
        # on every progress write. Strings need up to four bytes per character.
        return (
            256 * sum(len(state.get(key, {})) for key in ("results", "failures", "attempts"))
            + 4 * sum(len(result[2]) for result in state.get("results", {}).values())
        )

    def _touch_locked(self, token):
        self._cache_order[token] = None
        self._cache_order.move_to_end(token)

    def _drop_cache_locked(self, token):
        for cache in (
            self._sources, self._identities, self._persisted, self._states, self._revisions,
            self._source_bytes, self._row_bytes, self._state_bytes, self._identity_bytes,
            self._known_revisions, self._cache_order,
        ):
            cache.pop(token, None)

    def _prune_locked(self):
        # Active leases pin the source and acknowledged result state needed by
        # their workers. Inactive entries can always be reloaded from Supabase.
        def retained_bytes():
            return sum(self._source_bytes.values()) + sum(self._row_bytes.values()) + sum(self._state_bytes.values()) + sum(self._identity_bytes.values())
        while len(self._cache_order) > self._max_cached_tasks or retained_bytes() > self._max_cache_bytes:
            oldest = next((token for token in self._cache_order if token not in self._leases), None)
            if oldest is None:
                return
            self._drop_cache_locked(oldest)

    def _stale_revision_locked(self, token, revision):
        return type(revision) is int and self._known_revisions.get(token, -1) > revision

    def _remember_revision_locked(self, token, revision):
        if type(revision) is int:
            self._known_revisions[token] = max(revision, self._known_revisions.get(token, -1))

    def _cache_source_locked(self, token, pairs):
        self._sources[token] = pairs
        self._source_bytes[token] = 128 * len(pairs) + 4 * sum(len(upper) + len(lower) for upper, lower in pairs)
        self._touch_locked(token)

    @staticmethod
    def _token(token):
        if not is_cloud_document_token(token):
            raise CheckpointError("Invalid cloud document recovery token.")
        return token

    @staticmethod
    def _call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except CheckpointError:
            raise
        except Exception as exc:
            # The shared store's errors deliberately omit database responses and
            # credentials. Preserve its actionable setup/lease diagnostics.
            from src.analysis_checkpoints import AnalysisCheckpointError
            message = str(exc) if isinstance(exc, AnalysisCheckpointError) else "Cloud document checkpoint storage is unavailable. Try again shortly."
            raise CheckpointError(message) from exc

    def _task(self, token):
        token = self._token(token)
        task = self._call(self.store.get_task, token)
        if task is None:
            return None
        if not isinstance(task, Mapping) or task.get("id") != token or task.get("page_key") != DOCUMENT_PAGE_KEY:
            raise CheckpointError("This recovery link is not a saved document analysis.")
        if task.get("owner_id") != self.owner_id:
            raise CheckpointError("Sign in to the account that created this analysis to restore it.")
        return task

    def _state_metadata(self, task):
        value = task.get("metadata")
        metadata = _metadata(value)
        if metadata != value:
            raise CheckpointError("The cloud analysis contains unexpected checkpoint metadata.")
        if metadata.get("owner_id") != self.owner_id:
            raise CheckpointError("The cloud document checkpoint has an invalid owner.")
        if task.get("settings") != metadata["settings"] or task.get("total_items") != metadata["total_chunks"] or task.get("fingerprint") != metadata["fingerprint"]:
            raise CheckpointError("The cloud document checkpoint settings or fingerprint do not match.")
        # Task columns are the lifecycle authority. RPCs deliberately leave
        # document metadata untouched when an owner kills an inactive task.
        return self._project_task_state(task, metadata)

    @staticmethod
    def _project_task_state(task, state):
        state = dict(state)
        status = task.get("status")
        if status not in {"queued", "running", "incomplete", "complete", "stopping", "killed"}:
            raise CheckpointError("The cloud document checkpoint has an invalid status.")
        state["status"] = "incomplete" if status == "queued" else status
        state["stop_requested"] = bool(task.get("stop_requested"))
        state["kill_requested"] = bool(task.get("kill_requested")) or status in {"stopping", "killed"}
        for key in ("kill_requested_at", "killed_at", "started_at", "completed_at", "updated_at"):
            if key in task:
                state[key] = task[key]
        if state["kill_requested"]:
            state["stop_requested"] = True
            state["retry_in_seconds"] = 0
            state["retry_attempt"] = None
        if status == "killed":
            state["active_chunks"] = []
        return state

    def create(self, state, chunk_pairs):
        metadata = _metadata(state)
        if metadata.get("owner_id") != self.owner_id:
            raise CheckpointError("The cloud document checkpoint has an invalid owner.")
        pairs = _pairs(chunk_pairs, metadata["total_chunks"])
        rows, contexts = _chunk_rows(state, metadata["total_chunks"])
        if any(pairs[index] != context for index, context in contexts.items()):
            raise CheckpointError("Checkpoint results do not match the document source pairs.")
        task = self._call(
            self.store.create_task, DOCUMENT_PAGE_KEY, metadata["settings"],
            {"pairs_hash": _pair_hash(pairs)},
            ({"upper_text": upper, "lower_text": lower} for upper, lower in pairs),
            fingerprint=metadata["fingerprint"], metadata=metadata,
        )
        token = self._token(task.get("id") if isinstance(task, Mapping) else None)
        with self._lock:
            self._cache_source_locked(token, pairs)
            self._identities[token] = deepcopy(metadata)
            self._identity_bytes[token] = self._identity_size(metadata)
            self._persisted[token] = {}
            self._row_bytes[token] = 0
            self._state_bytes[token] = 0
            self._prune_locked()
        # Boundary imports may already contain successes. Preserve them without
        # storing an ever-growing result blob in task metadata.
        if rows or metadata["status"] == "complete":
            if not self.claim(token):
                raise CheckpointError("The newly saved analysis is already being processed.")
            try:
                self.save(token, state)
            finally:
                self.release(token)
        return token

    def load(self, token):
        for _ in range(3):
            result = self._load_once(token)
            if result is not _CHANGED_SNAPSHOT:
                return result
        raise CheckpointError("Analysis progress changed while loading. Try opening it again shortly.")

    def _load_once(self, token):
        task = self._task(token)
        if task is None:
            return None
        if task.get("status") == "creating":
            raise CheckpointError("The document is still being saved. Try opening it again shortly.")
        state = self._state_metadata(task)
        with self._lock:
            if token in self._states and task.get("revision") == self._revisions[token]:
                cached = self._states[token]
                if task.get("completed_items") != len(cached["results"]) or task.get("failed_items") != len(cached["failures"]):
                    raise CheckpointError("The cloud checkpoint progress counts do not match its saved results.")
                self._touch_locked(token)
                return self._project_task_state(task, deepcopy(cached)), list(self._sources[token])
        total = state["total_chunks"]
        items = []
        for offset in range(0, total, 1000):
            batch = self._call(self.store.load_items, token, offset=offset, limit=min(1000, total - offset))
            if not isinstance(batch, list):
                raise CheckpointError("The cloud document checkpoint contains invalid chunk data.")
            items.extend(batch)
        latest = self._task(token)
        if latest is None:
            return None
        if latest.get("revision") != task.get("revision"):
            return _CHANGED_SNAPSHOT
        if len(items) != total or any(not isinstance(item, Mapping) for item in items) or [item.get("item_index") for item in items] != list(range(total)):
            raise CheckpointError("The cloud checkpoint has missing or invalid document chunks.")
        pairs = []
        state.update(results={}, failures={}, attempts={})
        for index, item in enumerate(items):
            source = item.get("input")
            if not isinstance(source, Mapping) or set(source) != {"upper_text", "lower_text"}:
                raise CheckpointError("The cloud checkpoint contains invalid document source chunks.")
            pairs.append((source["upper_text"], source["lower_text"]))
            status = item.get("status")
            if status == "complete":
                result = item.get("result")
                if not isinstance(result, Mapping) or set(result) != {"generated", "metrics"} or item.get("error") is not None:
                    raise CheckpointError("The cloud checkpoint contains an invalid successful result.")
                state["results"][index] = (*pairs[-1], result["generated"], result["metrics"])
            elif status == "failed":
                if item.get("result") is not None or not isinstance(item.get("error"), str):
                    raise CheckpointError("The cloud checkpoint contains an invalid failed chunk.")
                state["failures"][index] = item["error"]
            elif status != "pending" or item.get("result") is not None or item.get("error") is not None:
                raise CheckpointError("The cloud checkpoint contains an invalid chunk status.")
            attempts = _integer(item.get("attempts", 0), "attempt count", maximum=2**63 - 1)
            if attempts:
                state["attempts"][index] = attempts
        pairs = _pairs(pairs, total)
        source = task.get("source")
        if not isinstance(source, Mapping) or source.get("pairs_hash") != _pair_hash(pairs):
            raise CheckpointError("The cloud document source chunks failed their integrity check.")
        stored_status = task.get("status")
        if stored_status not in {"queued", "running", "incomplete", "complete", "stopping", "killed"}:
            raise CheckpointError("The cloud document checkpoint has an invalid status.")
        state["status"] = "incomplete" if stored_status == "queued" else stored_status
        state["stop_requested"] = bool(task.get("stop_requested"))
        rows, _ = _chunk_rows(state, total)
        if task.get("completed_items") != len(state["results"]) or task.get("failed_items") != len(state["failures"]):
            raise CheckpointError("The cloud checkpoint progress counts do not match its saved results.")
        with self._lock:
            # A worker may have acknowledged a newer write after the final
            # server read but before this cache fill acquired the lock.
            if self._stale_revision_locked(token, task.get("revision")):
                return _CHANGED_SNAPSHOT
            self._remember_revision_locked(token, task.get("revision"))
            self._cache_source_locked(token, pairs)
            self._identities[token] = deepcopy(_metadata(state))
            self._identity_bytes[token] = self._identity_size(self._identities[token])
            self._persisted[token] = rows
            self._row_bytes[token] = sum(self._row_size(row) for row in rows.values())
            self._states[token] = deepcopy(state)
            self._state_bytes[token] = self._state_size(state)
            self._revisions[token] = task.get("revision")
            self._prune_locked()
        return state, pairs

    def _identity(self, token, metadata):
        self._token(token)
        if metadata.get("owner_id") != self.owner_id:
            raise CheckpointError("The cloud document checkpoint has an invalid owner.")
        with self._lock:
            stored = self._identities.get(token)
        if stored is None:
            if self.load(token) is None:
                raise CheckpointError("The saved cloud document analysis is no longer available.")
            stored = self._identities[token]
        DocumentCheckpointStore._check_identity(stored, metadata)

    def _lease(self, token):
        with self._lock:
            lease = self._leases.get(token)
        if not lease:
            raise CheckpointError("This cloud analysis is not claimed by the current worker.")
        return lease

    @staticmethod
    def _update(index, row):
        result_json, error, attempts, _ = row if row is not None else (None, None, 0, None)
        result = None
        if result_json is not None:
            generated, metrics = _read_json(result_json)
            result = {"generated": generated, "metrics": metrics}
        return {
            "index": index, "status": "complete" if result is not None else "failed" if error is not None else "pending",
            "result": result, "error": error, "attempts": attempts,
        }

    def _write_batches(self, changed):
        """Decode only a bounded batch, including a valid large singleton."""
        entries = iter(sorted(changed.items()))
        entry = next(entries, None)
        if entry is None:
            yield [], {}, True
            return
        prepared = None
        while entry is not None:
            updates = []
            acknowledged = {}
            byte_count = 2  # JSON array brackets.
            while entry is not None and len(updates) < WRITE_BATCH_ITEMS:
                index, row = entry
                update, size = prepared if prepared is not None else (None, None)
                if update is None:
                    update = self._update(index, row)
                    size = len(json.dumps(update, ensure_ascii=False).encode("utf-8"))
                prepared = (update, size)
                next_size = byte_count + size + (2 if updates else 0)
                if updates and next_size > WRITE_BATCH_BYTES:
                    break
                if next_size > MAX_UPDATE_BYTES:
                    raise CheckpointError("A document result exceeds the cloud checkpoint size limit.")
                updates.append(update)
                acknowledged[index] = row
                byte_count = next_size
                entry = next(entries, None)
                prepared = None
            yield updates, acknowledged, entry is None

    def _write(self, token, metadata, changed):
        task = None
        for updates, acknowledged, final_batch in self._write_batches(changed):
            batch_metadata = metadata
            if not final_batch and metadata["status"] == "complete":
                batch_metadata = {**metadata, "status": "incomplete"}
            task = self._call(
                self.store.save_items, token, updates,
                metadata=batch_metadata, status=batch_metadata["status"], lease_token=self._lease(token),
            )
            with self._lock:
                revision = task.get("revision") if isinstance(task, Mapping) else None
                if self._stale_revision_locked(token, revision):
                    # A newer full snapshot already includes this committed
                    # write; its status and acknowledged rows take precedence.
                    continue
                self._remember_revision_locked(token, revision)
                persisted = self._persisted[token]
                for index, row in acknowledged.items():
                    self._row_bytes[token] += self._row_size(row) - self._row_size(persisted.get(index))
                    if row is None:
                        persisted.pop(index, None)
                    else:
                        persisted[index] = row
                self._identities[token] = deepcopy(batch_metadata)
                self._identity_bytes[token] = self._identity_size(batch_metadata)
                self._states.pop(token, None)
                self._state_bytes[token] = 0
                self._revisions.pop(token, None)
                self._touch_locked(token)
                self._prune_locked()
        return task

    def save(self, token, state):
        self._lease(token)
        metadata = _metadata(state)
        self._identity(token, metadata)
        rows, contexts = _chunk_rows(state, metadata["total_chunks"])
        pairs = self._sources[token]
        if any(pairs[index] != context for index, context in contexts.items()):
            raise CheckpointError("Checkpoint results do not match the document source pairs.")
        with self._lock:
            previous = dict(self._persisted[token])
        changed = {index: row for index, row in rows.items() if previous.get(index) != row}
        changed.update({index: None for index in previous.keys() - rows.keys()})
        return self._write(token, metadata, changed)

    def save_progress(self, token, state, chunk_index=None):
        self._lease(token)
        metadata = _metadata(state)
        self._identity(token, metadata)
        changed = {}
        if chunk_index is not None:
            _integer(chunk_index, "progress chunk index", maximum=metadata["total_chunks"] - 1)
            partial = {"status": metadata["status"]}
            for key in ("results", "failures", "attempts"):
                values = state.get(key, {})
                if not isinstance(values, Mapping):
                    raise CheckpointError(f"Checkpoint {key} must be a mapping.")
                if chunk_index in values and str(chunk_index) in values:
                    raise CheckpointError(f"Checkpoint contains duplicate {key} indices.")
                partial[key] = {chunk_index: values[chunk_index]} if chunk_index in values else {chunk_index: values[str(chunk_index)]} if str(chunk_index) in values else {}
            rows, contexts = _chunk_rows(partial, metadata["total_chunks"], check_complete=False)
            if chunk_index in contexts and self._sources[token][chunk_index] != contexts[chunk_index]:
                raise CheckpointError("Checkpoint result does not match its source chunk.")
            row = rows.get(chunk_index)
            if self._persisted[token].get(chunk_index) != row:
                changed[chunk_index] = row
        return self._write(token, metadata, changed)

    def claim(self, token):
        self._token(token)
        from src.analysis_checkpoints import AnalysisLeaseError
        try:
            task = self.store.claim(token, ttl_seconds=LEASE_SECONDS)
        except AnalysisLeaseError:
            return False
        except Exception as exc:
            from src.analysis_checkpoints import AnalysisCheckpointError
            message = str(exc) if isinstance(exc, AnalysisCheckpointError) else "Cloud document checkpoint storage is unavailable. Try again shortly."
            raise CheckpointError(message) from exc
        if not isinstance(task, Mapping) or not isinstance(task.get("lease_token"), str) or not task["lease_token"]:
            raise CheckpointError("The cloud analysis did not return a valid worker lease.")
        with self._lock:
            self._leases[token] = task["lease_token"]
            if token in self._cache_order:
                self._touch_locked(token)
        return True

    def heartbeat(self, token):
        return self._call(self.store.heartbeat, token, self._lease(token), ttl_seconds=LEASE_SECONDS)

    def release(self, token):
        with self._lock:
            lease = self._leases.get(token)
            metadata = self._identities.get(token)
        if lease is None:
            return
        try:
            status = ("killed" if metadata and (metadata.get("kill_requested") or metadata.get("status") in {"stopping", "killed"})
                      else "complete" if metadata and metadata.get("status") == "complete" else "incomplete")
            self._call(self.store.release, token, lease, status=status)
        finally:
            with self._lock:
                self._leases.pop(token, None)
                self._prune_locked()

    def is_active(self, token):
        task = self._task(token)
        return bool(task and _active(task))

    def request_stop(self, token):
        task = self._task(token)
        if task is None or not _active(task):
            return False
        self._call(self.store.request_stop, token)
        return True

    def request_kill(self, token, *, active=False):
        # Kill also terminates queued and paused tasks. Unlike resumable Stop,
        # it must reach the owner-scoped RPC without requiring a live lease.
        if self._task(token) is None:
            return False
        task = self._call(self.store.kill_task, token)
        if not isinstance(task, Mapping):
            raise CheckpointError("The cloud analysis did not acknowledge Kill.")
        state = self._state_metadata(task)
        if not state.get("kill_requested") and state["status"] not in {"stopping", "killed"}:
            return False
        with self._lock:
            self._remember_revision_locked(token, task.get("revision"))
            self._states.pop(token, None)
            self._revisions.pop(token, None)
            self._state_bytes[token] = 0
            if token in self._identities:
                self._identities[token] = deepcopy(state)
        return state

    def list_for_owner(self):
        tasks = self._call(self.store.list_tasks, page_key=DOCUMENT_PAGE_KEY, statuses=["queued", "running", "incomplete"], limit=100)
        result = []
        for task in tasks:
            if not isinstance(task, Mapping) or task.get("owner_id") != self.owner_id or task.get("page_key") != DOCUMENT_PAGE_KEY:
                raise CheckpointError("The saved document list contains an invalid account scope.")
            if task.get("kill_requested") or task.get("status") in {"stopping", "killed"}:
                continue
            metadata = self._state_metadata(task)
            result.append({
                "token": self._token(task.get("id")), "settings": metadata["settings"],
                "status": task["status"], "total_chunks": metadata["total_chunks"],
                "completed_chunks": task["completed_items"], "failed_chunks": task["failed_items"],
                "updated_at": task.get("updated_at"), "active": _active(task),
            })
        return result

    def delete(self, token):
        self._token(token)
        self._call(self.store.delete, token)
        with self._lock:
            self._leases.pop(token, None)
            self._drop_cache_locked(token)
