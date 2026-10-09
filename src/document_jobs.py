"""Persistent document tasks that are independent of Streamlit script reruns."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import os
from threading import Event, RLock, Thread
from typing import Any
from urllib.parse import urlsplit

from src.document_analysis import run_chunk_analysis, validate_analysis_state
from src.document_checkpoints import CheckpointError, DocumentCheckpointStore
from src.model_catalog import model_unavailability_error


DEFAULT_CHUNK_CONCURRENCY = 3
DEFAULT_PARALLEL_THRESHOLD = 8
DEFAULT_CHUNK_WORKERS = 8
LEASE_HEARTBEAT_SECONDS = 20


def _configured_limit(name, default, maximum):
    """Keep administrator overrides bounded; ignore malformed settings."""
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if 1 <= value <= maximum else default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_error(message: Any, api_key: str) -> str:
    text = str(message)
    if api_key:
        text = text.replace(api_key, "[redacted]")
    return text


def _snapshot(state):
    # Completed result tuples and their metrics are immutable in the worker.
    # Copy mappings here; make the defensive deep copy only when a UI reads.
    return {
        **state,
        "settings": dict(state["settings"]),
        "results": dict(state["results"]),
        "failures": dict(state["failures"]),
        "attempts": dict(state["attempts"]),
        "active_chunks": list(state.get("active_chunks", [])),
    }


def _analyze_chunk(settings, api_key, upper, lower):
    # Import providers only in workers. No Streamlit session state is read here.
    from src.direct_recall.comparison import compare_texts
    from src.adversarial_persuasion_detection import run_persuasion_probe

    words = len(lower.split()) or settings["chunk_size"]
    options = {
        "chunk_size": words,
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "custom_template": settings.get("custom_template"),
        "target_word_count": words,
        "extra_prompt_instructions": settings.get("extra_prompt_instructions"),
        "request_timeout": 120,
        "request_max_retries": 0,
        "base_url": settings.get("base_url"),
    }
    if settings["continuation_method"] == "Normal Continuation":
        return compare_texts(
            upper, lower, api_key, model_name=settings["model"],
            provider=settings["provider"],
            continuation_method=settings["continuation_method"], **options,
        )
    return run_persuasion_probe(
        api_key, settings["model"], settings["provider"],
        settings["continuation_method"], upper, lower, **options,
    )


class DocumentAnalysisJobs:
    """Bounded worker pool with durable per-chunk progress and duplicate protection."""

    def __init__(
        self, store=None, *, max_workers=4, max_pending=16, analyze_chunk=None,
        chunk_concurrency=None, parallel_threshold=DEFAULT_PARALLEL_THRESHOLD,
        max_chunk_workers=None,
    ):
        if chunk_concurrency is None:
            chunk_concurrency = _configured_limit(
                "COPYRIGHT_DETECTIVE_DOCUMENT_CHUNK_CONCURRENCY", DEFAULT_CHUNK_CONCURRENCY, 8
            )
        if max_chunk_workers is None:
            max_chunk_workers = _configured_limit(
                "COPYRIGHT_DETECTIVE_DOCUMENT_CHUNK_WORKERS", DEFAULT_CHUNK_WORKERS, 32
            )
        for name, value, maximum in (
            ("max_workers", max_workers, 32),
            ("max_pending", max_pending, 128),
            ("chunk_concurrency", chunk_concurrency, 8),
            ("parallel_threshold", parallel_threshold, 500_000),
            ("max_chunk_workers", max_chunk_workers, 32),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer between 1 and {maximum}.")
        self.store = store if store is not None else DocumentCheckpointStore()
        self.chunk_concurrency = min(chunk_concurrency, max_chunk_workers)
        self.parallel_threshold = parallel_threshold
        self.max_chunk_workers = max_chunk_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="document-analysis"
        )
        # All documents share this pool; each coordinator keeps only a small
        # rolling window, so a large document never queues all of its chunks.
        self._chunk_executor = ThreadPoolExecutor(
            max_workers=max_chunk_workers, thread_name_prefix="document-chunk"
        )
        self._lock = RLock()
        self._max_pending = max_pending
        self._jobs = {}
        self._cloud_stores = {}
        self._analyze_chunk = analyze_chunk or _analyze_chunk

    def configure_cloud_store(self, cloud_store):
        """Bind a store supplied by the UI after Supabase verifies its user."""
        if cloud_store is None:
            return
        from src.supabase_document_checkpoints import SupabaseDocumentCheckpointStore
        adapter = SupabaseDocumentCheckpointStore(cloud_store)
        with self._lock:
            self._cloud_stores.setdefault(adapter.owner_id, adapter)

    def cloud_enabled_for(self, owner_id):
        with self._lock:
            return bool(owner_id and owner_id in self._cloud_stores)

    def _store_for(self, token=None, *, owner_id=None):
        # Existing random hexadecimal links remain local. New cloud links are
        # UUIDs and must be routed through the authenticated owner's binding.
        from src.supabase_document_checkpoints import is_cloud_document_token
        if token is not None and not is_cloud_document_token(token):
            return self.store
        with self._lock:
            cloud = self._cloud_stores.get(owner_id) if owner_id else None
        if token is not None and cloud is None:
            raise CheckpointError("Sign in to the account that created this cloud analysis to restore it.")
        return cloud or self.store

    def list_saved(self, *, owner_id=None):
        with self._lock:
            store = self._cloud_stores.get(owner_id) if owner_id else None
        return store.list_for_owner() if store is not None else []

    def is_active(self, token, *, owner_id=None):
        with self._lock:
            job = self._jobs.get(token)
            if job is not None and not job.get("finished"):
                self._check_owner(job["state"], owner_id)
                return True
        store = self._store_for(token, owner_id=owner_id)
        return bool(getattr(store, "remote", False) and store.is_active(token))

    def concurrency_for(self, remaining_chunks):
        if remaining_chunks < self.parallel_threshold:
            return 1
        return min(self.chunk_concurrency, remaining_chunks)

    @staticmethod
    def _validate_settings(state):
        settings = state.get("settings", {})
        required = ("filename", "model", "provider", "chunk_size", "continuation_method", "temperature", "top_p")
        if not isinstance(settings, dict) or any(key not in settings for key in required):
            raise CheckpointError("The saved analysis is missing generation settings. Start a new run.")
        for key in ("filename", "model", "provider", "continuation_method"):
            if not isinstance(settings[key], str) or not settings[key].strip():
                raise CheckpointError("The saved analysis has invalid generation settings. Start a new run.")

    @staticmethod
    def _check_model_available(state):
        settings = state["settings"]
        error = model_unavailability_error(settings["provider"], settings["model"])
        if error:
            raise ValueError(error)

    @staticmethod
    def _check_owner(state, owner_id):
        if state.get("owner_id") and state["owner_id"] != owner_id:
            raise CheckpointError("Sign in to the account that created this analysis to restore it.")

    def create(self, state, pairs, *, owner_id=None, use_cloud=True):
        validate_analysis_state(pairs, state)
        self._validate_settings(state)
        self._check_model_available(state)
        base_url = state["settings"].get("base_url")
        if base_url:
            endpoint = urlsplit(base_url)
            if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
                raise ValueError("Use an endpoint URL without credentials or query parameters; enter credentials in the API key field.")
        state = deepcopy(state)
        state["owner_id"] = owner_id
        state["updated_at"] = _now()
        store = self._store_for(owner_id=owner_id) if use_cloud else self.store
        return store.create(state, pairs)

    def is_running(self, token):
        with self._lock:
            job = self._jobs.get(token)
            return bool(job is not None and not job.get("finished"))

    def get(self, token, *, owner_id=None):
        # Keep restoration atomic with submit, so a concurrent UI read cannot
        # label a newly submitted worker as a server interruption.
        with self._lock:
            job = self._jobs.get(token)
            if job is not None:
                self._check_owner(job["state"], owner_id)
                return deepcopy(job["state"])
            store = self._store_for(token, owner_id=owner_id)
            saved = store.load(token)
            if saved is None:
                return None
            state, pairs = saved
            self._check_owner(state, owner_id)
            validate_analysis_state(pairs, state)
            self._validate_settings(state)
            remote = getattr(store, "remote", False)
            active = remote and store.is_active(token)
            if remote and state["status"] == "running" and not active:
                # Completion may have committed after the first snapshot.
                latest = store.load(token)
                if latest is None:
                    return None
                state, pairs = latest
                self._check_owner(state, owner_id)
                validate_analysis_state(pairs, state)
            if state["status"] == "running" and not active:
                state["status"] = "incomplete"
                state["error"] = "The server stopped before this analysis finished. Resume to process the remaining chunks."
                state["retry_in_seconds"] = 0
                state["stop_requested"] = False
                state["active_chunks"] = []
                if not getattr(store, "remote", False):
                    store.save(token, state)
            return state

    def submit(self, token, api_key, *, owner_id=None):
        with self._lock:
            existing = self._jobs.get(token)
            if existing is not None:
                self._check_owner(existing["state"], owner_id)
                if not existing.get("finished"):
                    return False
            active_count = sum(not job.get("finished") for job in self._jobs.values())
            if active_count >= self._max_pending:
                raise CheckpointError("Document analysis capacity is busy. Try starting this saved run again shortly.")
            store = self._store_for(token, owner_id=owner_id)
            saved = store.load(token)
            if saved is None:
                raise CheckpointError("The saved analysis is no longer available.")
            state, pairs = saved
            self._check_owner(state, owner_id)
            validate_analysis_state(pairs, state)
            self._validate_settings(state)
            if not str(api_key or "").strip() and state["settings"]["provider"] != "Local vLLM":
                raise ValueError("Enter an API key before starting or resuming analysis.")
            if state["status"] == "complete":
                return False
            self._check_model_available(state)
            claimed = False
            try:
                if getattr(store, "remote", False):
                    if not store.claim(token):
                        return False
                    claimed = True
                    # Claim fences competing processes. Re-read progress after
                    # acquiring it so a just-finished chunk is never repeated.
                    state, pairs = store.load(token)
                    self._check_owner(state, owner_id)
                    validate_analysis_state(pairs, state)
                    self._validate_settings(state)
                if existing is not None:
                    retained = existing["state"]
                    if not getattr(store, "remote", False):
                        state = deepcopy(retained)
                    else:
                        # A newer worker's persisted successes take precedence
                        # over results retained during a previous storage outage.
                        for index, result in retained["results"].items():
                            state["results"].setdefault(index, deepcopy(result))
                            state["failures"].pop(index, None)
                        for index, attempts in retained["attempts"].items():
                            state["attempts"][index] = max(attempts, state["attempts"].get(index, 0))
                state["status"] = "running"
                state["stop_requested"] = False
                state["active_chunks"] = []
                limit = self.concurrency_for(state["total_chunks"] - len(state["results"]))
                state["concurrency_limit"] = limit
                state["effective_concurrency"] = limit
                state["error"] = None
                state["started_at"] = _now()
                state["updated_at"] = state["started_at"]
                store.save(token, state)
                stop = Event()
                self._jobs[token] = {"state": _snapshot(state), "stop": stop, "store": store}
                try:
                    self._executor.submit(self._run, token, state, pairs, str(api_key or "").strip(), stop, store)
                except Exception:
                    self._jobs.pop(token, None)
                    state["status"] = "incomplete"
                    state["error"] = "The analysis worker could not start. Retry the run."
                    store.save(token, state)
                    raise
                return True
            except BaseException:
                if claimed:
                    store.release(token)
                raise

    def _run(self, token, state, pairs, api_key, stop, store):
        heartbeat_done = Event()
        lease_errors = []

        def heartbeat():
            while not heartbeat_done.wait(LEASE_HEARTBEAT_SECONDS):
                try:
                    if store.heartbeat(token).get("stop_requested"):
                        stop.set()
                except Exception as exc:
                    lease_errors.append(exc)
                    stop.set()
                    return

        heartbeat_thread = None
        if getattr(store, "remote", False):
            heartbeat_thread = Thread(target=heartbeat, name="document-lease-heartbeat", daemon=True)
            heartbeat_thread.start()

        def checkpoint(updated):
            updated["updated_at"] = _now()
            if updated["status"] == "complete" and not updated.get("completed_at"):
                updated["completed_at"] = updated["updated_at"]
            updated["stop_requested"] = stop.is_set()
            updated["error"] = _redact_error(updated.get("error"), api_key) if updated.get("error") else None
            updated["failures"] = {
                index: _redact_error(error, api_key)
                for index, error in updated["failures"].items()
            }
            # Stop outbound work if progress cannot be persisted.
            try:
                current = updated.get("current_chunk")
                if lease_errors:
                    raise CheckpointError("The cloud task lease could not be renewed. Resume after the active lease expires.") from lease_errors[0]
                saved = store.save_progress(token, updated, current - 1 if current else None)
                if isinstance(saved, dict) and saved.get("stop_requested"):
                    stop.set()
                    updated["stop_requested"] = True
            finally:
                with self._lock:
                    if token in self._jobs:
                        self._jobs[token]["state"] = _snapshot(updated)

        try:
            run_chunk_analysis(
                pairs,
                lambda upper, lower: self._analyze_chunk(state["settings"], api_key, upper, lower),
                state, on_update=checkpoint, should_stop=stop.is_set, sleep=stop.wait,
                max_concurrency=state["concurrency_limit"], executor=self._chunk_executor,
            )
        except BaseException as exc:
            state["status"] = "incomplete"
            state["error"] = _redact_error(f"Analysis paused: {type(exc).__name__}: {exc}", api_key)
            state["retry_in_seconds"] = 0
            state["retry_attempt"] = None
            state["active_chunks"] = []
            state["updated_at"] = _now()
            state["stop_requested"] = stop.is_set()
            state["failures"] = {
                index: _redact_error(error, api_key)
                for index, error in state["failures"].items()
            }
            try:
                # An interrupted parallel coordinator may have drained several
                # in-flight results after its first checkpoint failure. Persist
                # all of them atomically instead of only the last changed row.
                store.save(token, state)
            except Exception:
                # Keep every drained success available while storage is down.
                with self._lock:
                    if token in self._jobs:
                        self._jobs[token]["storage_failed"] = True
            finally:
                with self._lock:
                    if token in self._jobs:
                        self._jobs[token]["state"] = _snapshot(state)
        finally:
            heartbeat_done.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1)
                try:
                    store.release(token)
                except Exception:
                    # Expiry provides recovery if the server cannot release a
                    # lease. Never overwrite another worker's fenced progress.
                    pass
            with self._lock:
                job = self._jobs.get(token)
                if job is not None and not job.get("storage_failed"):
                    self._jobs.pop(token, None)
                elif job is not None:
                    job["finished"] = True

    def stop(self, token, *, owner_id=None):
        with self._lock:
            job = self._jobs.get(token)
            if job is not None and not job.get("finished"):
                self._check_owner(job["state"], owner_id)
                job["stop"].set()
                job["state"]["stop_requested"] = True
                store = job["store"]
                if getattr(store, "remote", False):
                    store.request_stop(token)
                return True
            store = self._store_for(token, owner_id=owner_id)
            if getattr(store, "remote", False):
                return store.request_stop(token)
            return False

    def delete(self, token, *, owner_id=None):
        with self._lock:
            if self.is_active(token, owner_id=owner_id):
                raise CheckpointError("Stop the analysis and wait for its in-flight requests before clearing it.")
            self.get(token, owner_id=owner_id)
            self._store_for(token, owner_id=owner_id).delete(token)
            self._jobs.pop(token, None)

    def close(self):
        with self._lock:
            for job in self._jobs.values():
                job["stop"].set()
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._chunk_executor.shutdown(wait=True, cancel_futures=True)


DOCUMENT_JOBS = DocumentAnalysisJobs()
