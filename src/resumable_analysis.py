"""Owner-bound, durable API call journals shared by the analysis pages.

Only JSON data is stored. Credentials and Streamlit objects never cross this
boundary. A saved response is replayed only for the exact original request.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from hashlib import sha256
import json
import math
import re
import base64
import gzip
import io
from threading import Event, RLock, Thread
from typing import Any, Callable

_CURRENT: ContextVar[Any] = ContextVar("analysis_journal", default=None)
_SUPPRESSED: ContextVar[bool] = ContextVar("analysis_journal_suppressed", default=False)
_CALL_SCOPE: ContextVar[tuple] = ContextVar("analysis_work_item_scope", default=())
PAGE_KEY = "_analysis_page_key"
PENDING_RESTORE = "_analysis_pending_restore"
RESUME_TASK = "_analysis_resume_task"
RUN_TRIGGER = "_analysis_run_trigger"
RUN_LABEL = "_analysis_run_label"
LEASE_TTL = 180
HEARTBEAT_SECONDS = 20


class AnalysisCheckpointError(RuntimeError):
    """Persistence failed; further paid calls must stop until recovery."""


class AnalysisResumeMismatch(AnalysisCheckpointError):
    """Saved requests cannot be mixed with changed inputs or models."""


# Canonical page inputs and outputs; account/authentication state is excluded.
_SAFE_PREFIXES = (
    "text_", "qa_", "sc_", "sleek_", "persuasion_", "mink_", "min_k_",
    "unlearn_", "representational_", "copyright_game", "_copyright_game",
    "sidebar_", "preview_", "probe_", "muse_",
)
_SAFE_KEYS = frozenset({
    "main_navigation", "detection_navigation", "game_navigation",
    "content_recall_mode", "knowledge_detection_mode", "unlearning_method", "unlearning_detection_mode",
    "custom_user_prompt", "custom_continuation_prompt", "current_prompt_template",
    "previous_prompting_method", "previous_prompting_method_example",
    "previous_input_method_example", "continuation_method_selector", "prompt_mode_selector",
    "num_qa_pairs", "num_eval_runs", "eval_temperature", "eval_top_p", "enable_llm_judge",
    "input_prompt", "reference", "generation_mode", "strategies", "attempts",
    "attempts_per_prompt", "generation_checklist", "baseline_selector", "prompt_preview",
    "generation_results", "evaluation_results", "mutation_store", "last_run_config",
    "confidence_analysis_result", "judge_provider", "judge_model", "judge_temperature",
    "judge_top_p", "secondary_judge_provider", "secondary_judge_model",
})
_SECRET_PARTS = ("api_key", "apikey", "access_token", "refresh_token", "password",
                 "secret", "authorization", "service_role", "agent_key", "fernet", "credential")
_UNSAFE_PARTS = ("pdf_report", "pdf_bytes", "uploader", "clear_cache", "confirm_reset")
_BOOL_STATE_KEYS = frozenset({"enable_llm_judge", "qa_enable_llm_judge", "sc_logit_mode", "text_return_logprobs"})


# Only the selected page's inputs/results are restored; preview datasets and
# other pages' histories are caches, not part of the frozen analysis input.
_PAGE_PREFIXES = {
    "Content Recall Detection": ("text_",),
    "Knowledge Memorization Detection": ("qa_", "sc_", "sleek_", "muse_"),
    "Persuasive Jailbreak Detection": ("persuasion_", "preview_", "probe_"),
    "Unlearning Detection": ("min_k_", "mink_", "unlearn_", "representational_"),
    "Game 1: The Hidden Passage Hunt": ("copyright_game:", "_copyright_game_"),
    "Game 2: The Cross-Model Scaling Quest": ("copyright_game2:", "_copyright_game2_"),
    "Game 3: The Memory Vault Hunt": ("qa_", "sc_", "sleek_", "muse_"),
}
_PAGE_EXTRA_KEYS = {
    "Content Recall Detection": {"content_recall_mode", "custom_user_prompt", "custom_continuation_prompt", "current_prompt_template", "previous_prompting_method", "previous_prompting_method_example", "previous_input_method_example", "continuation_method_selector", "prompt_mode_selector", "confidence_analysis_result"},
    "Knowledge Memorization Detection": {"knowledge_detection_mode", "num_qa_pairs", "num_eval_runs", "eval_temperature", "eval_top_p", "enable_llm_judge"},
    "Unlearning Detection": {"unlearning_method", "unlearning_detection_mode"},
    "Persuasive Jailbreak Detection": {"input_prompt", "reference", "generation_mode", "strategies", "attempts", "attempts_per_prompt", "generation_checklist", "baseline_selector", "prompt_preview", "generation_results", "evaluation_results", "mutation_store", "last_run_config", "confidence_analysis_result", "judge_provider", "judge_model", "judge_temperature", "judge_top_p", "secondary_judge_provider", "secondary_judge_model"},
}
_PAGE_EXTRA_KEYS["Game 3: The Memory Vault Hunt"] = _PAGE_EXTRA_KEYS["Knowledge Memorization Detection"]


def key_belongs_to_page(key: str, page_key: str | None) -> bool:
    if page_key not in _PAGE_PREFIXES:
        return True
    if key in {"main_navigation", "detection_navigation", "game_navigation"} or key.startswith("sidebar_"):
        return True
    return key.startswith(_PAGE_PREFIXES[page_key]) or key in _PAGE_EXTRA_KEYS.get(page_key, set())


def is_safe_state_key(key: str) -> bool:
    low = key.lower()
    if low.startswith("min_k_full_data_") or low == "text_literal_examples":
        return False
    if low.startswith("copyright_game2:") and (low.endswith("history") or "round_history" in low):
        return False
    if any(part in low for part in _SECRET_PARTS):
        return False
    if any(part in low for part in _UNSAFE_PARTS if part != "upload"):
        return False
    return key in _SAFE_KEYS or key.startswith(_SAFE_PREFIXES)


def encode(value: Any) -> Any:
    """A small safe codec preserving tuples, with no pickle or dynamic imports."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Checkpoint numbers must be finite.")
        return value
    if isinstance(value, tuple):
        return {"__analysis_tuple__": [encode(item) for item in value]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Checkpoint keys must be strings.")
        if any(any(part in key.lower() for part in _SECRET_PARTS) for key in value):
            raise ValueError("Credentials cannot be stored in an analysis checkpoint.")
        return {key: encode(item) for key, item in value.items()}
    raise ValueError(f"Unsupported checkpoint value: {type(value).__name__}")


def decode(value: Any) -> Any:
    if isinstance(value, list):
        return [decode(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__analysis_tuple__"} and isinstance(value["__analysis_tuple__"], list):
            return tuple(decode(item) for item in value["__analysis_tuple__"])
        return {key: decode(item) for key, item in value.items()}
    return value


def pack_snapshot(snapshot: dict) -> dict:
    """Compress large JSON snapshots; task rows stay within PostgREST limits."""
    from src.analysis_checkpoints import _check_credentials
    _check_credentials(snapshot)  # Validate URLs/fields before compression.
    raw = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(raw) > 64 * 1024 * 1024:
        raise AnalysisCheckpointError("The saved input is too large. Divide this analysis into smaller batches.")
    if len(raw) < 512 * 1024:
        return snapshot
    compressed = base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")
    if len(compressed) > 12 * 1024 * 1024:
        raise AnalysisCheckpointError("The saved input is too large. Divide this analysis into smaller batches.")
    return {"__analysis_snapshot_gzip__": compressed}


def unpack_snapshot(value: Any) -> dict:
    if not isinstance(value, dict):
        raise AnalysisCheckpointError("The saved analysis snapshot is invalid.")
    if set(value) != {"__analysis_snapshot_gzip__"}:
        return value
    try:
        compressed = base64.b64decode(value["__analysis_snapshot_gzip__"], validate=True)
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            raw = stream.read(64 * 1024 * 1024 + 1)
        if len(raw) > 64 * 1024 * 1024:
            raise ValueError("snapshot exceeds limit")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("snapshot must be an object")
        return data
    except Exception as exc:
        raise AnalysisCheckpointError("The saved analysis snapshot cannot be decoded safely.") from exc


def snapshot_session(state: Any, page_key: str | None = None, *, max_bytes: int | None = None) -> dict:
    saved = {}
    remaining = max_bytes
    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    for key in list(state):
        if not isinstance(key, str) or not is_safe_state_key(key) or not key_belongs_to_page(key, page_key):
            continue
        try:
            value = state[key]
            if value is None or (isinstance(value, bool) and key not in _BOOL_STATE_KEYS):
                continue  # Action-button and upload-widget state cannot be restored.
            encoded = encode(value)
            if remaining is not None:
                size = len(key.encode("utf-8")) + 8
                for chunk in encoder.iterencode(encoded):
                    size += len(chunk.encode("utf-8"))
                    if size > remaining:
                        break
                if size > remaining:
                    continue  # Large aggregates can be rebuilt from saved calls.
                remaining -= size
            saved[key] = encoded
        except (ValueError, TypeError):
            # File objects, reports, dataframes and UI handles are regenerated.
            continue
    return saved


def get_cloud_store_for_current_user():
    """Verify account identity before constructing an owner-bound cloud store.

    Guests retain existing behavior. Signed-in accounts always use cloud
    recovery: an authenticated client is sufficient for ordinary analyses,
    while a configured service client supports privileged competition jobs.
    Authentication/configuration failures never fall back to temporary files.
    """
    import streamlit as st
    from src.supabase_client import get_authenticated_client, get_secret
    if not st.session_state.get("access_token"):
        return None
    url = str(get_secret("SUPABASE_URL", "") or "").strip()
    service_key = str(get_secret("SUPABASE_SERVICE_ROLE_KEY", "") or "").strip()
    if not url:
        raise AnalysisCheckpointError(
            "Cloud recovery requires SUPABASE_URL. Configure Supabase before starting a signed-in analysis."
        )
    try:
        authenticated = get_authenticated_client()
        user = authenticated.auth.get_user().user if authenticated else None
        owner_id = str(getattr(user, "id", "") or "")
        if not owner_id or owner_id != str(st.session_state.get("user_id") or ""):
            raise AnalysisCheckpointError("Sign in again before recovering an analysis task.")
        from src.analysis_checkpoints import SupabaseAnalysisCheckpointStore
        if service_key:
            from supabase import create_client
            client = create_client(url, service_key)
        else:
            client = authenticated
        return SupabaseAnalysisCheckpointStore(client, owner_id)
    except AnalysisCheckpointError:
        raise
    except Exception as exc:
        raise AnalysisCheckpointError(
            "Cloud recovery could not verify your account. Sign in again or check Supabase availability."
        ) from exc


def _valid_response(result: Any) -> bool:
    if result is None:
        return False
    if isinstance(result, str):
        return bool(result.strip()) and not re.match(
            r"^error(?:\s*:|\s+calling\s+api\b|\s*$)", result.lstrip(), re.IGNORECASE,
        )
    if isinstance(result, tuple):
        return bool(result) and _valid_response(result[0])
    if isinstance(result, dict):
        return not result.get("error") and result.get("status") not in {"error", "failed"}
    return True


class CallJournal:
    """One fenced writer; cached responses survive a fresh application process."""

    def __init__(self, store: Any, task: dict, *, heartbeat: bool = True):
        self.store = store
        self.task = task
        self.task_id = str(task["id"])
        self.index = 0
        self.unscoped_index = 0
        self.scope_counters = {}
        self.scoped_items = {}
        self.unscoped_items = []
        self.visited_items = set()
        self.scopes_enabled = (task.get("source") or {}).get("journal_version", 1) == 2
        from src.analysis_checkpoints import OFFICIAL_GAME_PAGE_KEYS
        if task.get("page_key") in OFFICIAL_GAME_PAGE_KEYS and task.get("server_managed") is not True:
            raise AnalysisCheckpointError(
                "This legacy official-game checkpoint cannot be verified. Start a new task; existing saved competition scores are preserved."
            )
        self.failed = False
        self.broken = False
        self.closed = False
        self.verified_complete = False
        self.lock = RLock()
        self.stop_event = Event()
        self.lease_token = None
        self.read_only = task.get("status") == "complete"
        try:
            if not self.read_only:
                self.task = store.claim(self.task_id, ttl_seconds=LEASE_TTL)
                self.lease_token = self.task["lease_token"]
            self.items = []
            while True:
                batch = store.load_items(self.task_id, offset=len(self.items), limit=200)
                self.items.extend({**row, "index": row.get("index", row.get("item_index")), "payload": row.get("payload", row.get("input"))} for row in batch)
                if len(batch) < 200:
                    break
            if any(int(row.get("index", row.get("item_index", -1))) != i for i, row in enumerate(self.items)):
                raise AnalysisCheckpointError("The saved call journal has invalid item indices.")
            for index, item in enumerate(self.items):
                scope = item["payload"].get("scope")
                if scope is None:
                    self.unscoped_items.append(index)
                else:
                    if not isinstance(scope, dict) or not isinstance(scope.get("path"), list):
                        raise AnalysisCheckpointError("The saved work item identity is invalid.")
                    path = tuple(scope["path"])
                    ordinal = scope.get("ordinal")
                    if not path or any(type(part) not in (str, int) for part in path) or type(ordinal) is not int or ordinal < 0:
                        raise AnalysisCheckpointError("The saved work item identity is invalid.")
                    identity = (path, ordinal)
                    if identity in self.scoped_items:
                        raise AnalysisCheckpointError("The saved work item identity is duplicated.")
                    self.scoped_items[identity] = index
                    self.scopes_enabled = True
        except Exception as exc:
            if self.lease_token:
                try:
                    store.release(self.task_id, self.lease_token, status="incomplete")
                except Exception:
                    pass
            raise AnalysisCheckpointError("Unable to load or claim the saved task. It may still be running.") from exc
        self.thread = None
        if heartbeat and not self.read_only:
            self.thread = Thread(target=self._heartbeat, daemon=True, name="analysis-checkpoint-heartbeat")
            self.thread.start()

    def _heartbeat(self):
        while not self.stop_event.wait(HEARTBEAT_SECONDS):
            try:
                task = self.store.heartbeat(self.task_id, self.lease_token, ttl_seconds=LEASE_TTL)
                if task.get("stop_requested"):
                    self.broken = True
                    return
            except Exception:
                self.broken = True
                return

    def call(self, operation: str, payload: dict, invoke: Callable, is_success: Callable | None = None):
        # Current page runners are sequential. The lock also protects accidental
        # concurrent use, so request order and repeated identical calls stay stable.
        with self.lock:
            if self.broken or self.closed:
                raise AnalysisCheckpointError("Checkpoint saving was interrupted. Resume this task before making more API calls.")
            try:
                request = {"operation": operation, "request": encode(payload)}
            except Exception as exc:
                self.failed = self.broken = True
                raise AnalysisCheckpointError("The next checkpoint request contains invalid data or credential fields. No API call was started.") from exc
            # A stable work-item path isolates branches: adding a judge/retry
            # for one item never displaces another item's saved response.
            scope = _CALL_SCOPE.get() if self.scopes_enabled else ()
            if scope:
                ordinal = self.scope_counters.get(scope, 0)
                self.scope_counters[scope] = ordinal + 1
                identity = (scope, ordinal)
                request["scope"] = {"path": list(scope), "ordinal": ordinal}
                index = self.scoped_items.get(identity, len(self.items))
            else:
                ordinal = self.unscoped_index
                self.unscoped_index += 1
                index = self.unscoped_items[ordinal] if ordinal < len(self.unscoped_items) else len(self.items)
            request["fingerprint"] = sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            self.index += 1
            try:
                if index < len(self.items):
                    item = self.items[index]
                    if item["payload"] != request:
                        self.failed = self.broken = True
                        raise AnalysisResumeMismatch(
                            "The resumed request differs from the saved input, model or parameters. "
                            "Restore the task's original settings; start a new task for changed settings."
                        )
                    self.visited_items.add(index)
                    if item.get("status") == "complete":
                        return decode(deepcopy(item["result"]))
                else:
                    if self.read_only:
                        raise AnalysisResumeMismatch("This completed task has no saved response for the requested call.")
                    self.store.append_item(self.task_id, index, request, lease_token=self.lease_token)
                    item = {"index": index, "payload": request, "status": "pending", "attempts": 0}
                    self.items.append(item)
                    if scope:
                        self.scoped_items[identity] = index
                    else:
                        self.unscoped_items.append(index)
                    self.visited_items.add(index)
                # Refresh/check fencing immediately before every paid call.
                task = self.store.heartbeat(self.task_id, self.lease_token, ttl_seconds=LEASE_TTL)
                if task.get("stop_requested"):
                    raise AnalysisCheckpointError("This saved task was stopped.")
            except AnalysisCheckpointError:
                self.failed = True
                raise
            except Exception as exc:
                self.failed = self.broken = True
                raise AnalysisCheckpointError("Unable to save the next request. No API call was started.") from exc

            token = _SUPPRESSED.set(True)
            try:
                result = invoke()
            except BaseException:
                self.failed = True
                raise
            finally:
                _SUPPRESSED.reset(token)
            success = (is_success or _valid_response)(result)
            if not success:
                self.failed = True
            try:
                encoded = encode(result) if success else None
                self.task = self.store.save_item(
                    self.task_id, index, result=encoded,
                    status="complete" if success else "failed",
                    attempts=int(item.get("attempts") or 0) + 1,
                    error=None if success else "The API call did not return a successful result.",
                    lease_token=self.lease_token,
                )
                item.update(status="complete" if success else "failed", result=encoded,
                            attempts=int(item.get("attempts") or 0) + 1)
            except Exception as exc:
                self.failed = self.broken = True
                raise AnalysisCheckpointError(
                    "The API returned, but its result could not be saved to Supabase. "
                    "Further calls have stopped; this unsaved call may need to run again."
                ) from exc
            return result

    def finish(self, success: bool, final_snapshot: dict | None = None):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.stop_event.set()
            if self.read_only:
                return
            complete = success and not self.failed and not self.broken and (len(self.visited_items) == len(self.items) or self.verified_complete)
            metadata = {"final_session": final_snapshot} if complete and final_snapshot is not None else {}
            try:
                self.store.release(self.task_id, self.lease_token,
                                   status="complete" if complete else "incomplete", metadata=metadata)
            except Exception as exc:
                self.broken = True
                raise AnalysisCheckpointError("The final checkpoint could not be saved. Saved calls remain available for recovery.") from exc


class PageRun:
    def __init__(self, page_key: str, state: Any):
        self.page_key = page_key
        self.state = state
        self.journal = None
        self.checked_store = False
        self.store = None
        self.label = None
        self.trigger = None
        self.broken = False

    def ensure_journal(self):
        if self.broken:
            raise AnalysisCheckpointError("Cloud checkpoints are unavailable. Restore the task before making more calls.")
        if self.journal is not None:
            return self.journal
        if not self.trigger:
            if self.state.get("access_token"):
                self.broken = True
                raise AnalysisCheckpointError(
                    "Use this page's Run action before making an API request. "
                    "Viewing saved results cannot start an untracked signed-in analysis."
                )
            return None  # Guests retain their existing local behavior.
        if not self.checked_store:
            self.checked_store = True
            try:
                self.store = get_cloud_store_for_current_user()
            except Exception:
                self.broken = True
                raise
        if self.store is None:
            return None
        try:
            resume_id = self.state.pop(RESUME_TASK, None)
            if resume_id:
                task = self.store.get_task(resume_id)
                if not task or task.get("page_key") != self.page_key:
                    raise AnalysisCheckpointError("The saved task is not available on this page for your account.")
                source = task.get("source") or {}
                if source.get("trigger_key") != self.trigger:
                    raise AnalysisResumeMismatch("Use the original run action to resume this task.")
                saved_snapshot = unpack_snapshot(source.get("initial_session", {}))
                current = snapshot_session(self.state, self.page_key)
                # Widgets may coerce harmless scalar types; requests are checked
                # exactly before replay. Provider/model/endpoint changes fail here.
                for key in saved_snapshot:
                    if key.startswith("sidebar_") and key in current and current[key] != saved_snapshot[key]:
                        raise AnalysisResumeMismatch("Restore the saved provider, model and endpoint before resuming.")
            else:
                task = self.store.create_task(
                    page_key=self.page_key,
                    settings={"label": self.label or self.trigger},
                    source={"initial_session": pack_snapshot(snapshot_session(self.state, self.page_key)),
                            "trigger_key": self.trigger, "label": self.label or self.trigger,
                            "journal_version": 2},
                    work_items=[], dynamic_items=True,
                )
            self.journal = CallJournal(self.store, task)
            self.state["_analysis_current_task_id"] = self.journal.task_id
            return self.journal
        except AnalysisCheckpointError:
            self.broken = True
            raise
        except Exception as exc:
            self.broken = True
            raise AnalysisCheckpointError(
                "Cloud checkpoints are unavailable. Apply supabase/analysis_checkpoints.sql "
                "or check the database connection before starting analysis."
            ) from exc

    def finish(self, success: bool):
        if self.journal is not None and not self.journal.closed:
            final_snapshot = None
            if success:
                try:
                    final_snapshot = pack_snapshot(snapshot_session(self.state, self.page_key, max_bytes=8 * 1024 * 1024))
                except Exception:
                    # The per-call journal remains authoritative. An optional
                    # report snapshot must not prevent completion or leave a
                    # heartbeat running forever; rebuild from saved calls.
                    final_snapshot = None
            self.journal.finish(success, final_snapshot)


def register_ui_run(label: str, trigger_key: str):
    current = _CURRENT.get()
    if isinstance(current, PageRun):
        current.label, current.trigger = label, trigger_key


@contextmanager
def page_analysis_scope(page_key: str, state: Any):
    scope = PageRun(page_key, state)
    token = _CURRENT.set(scope)
    try:
        yield scope
    except BaseException:
        scope.finish(False)
        raise
    else:
        scope.finish(True)
    finally:
        _CURRENT.reset(token)


def finish_ui_run(success: bool):
    scope = _CURRENT.get()
    if isinstance(scope, PageRun):
        scope.finish(success)


def checkpoint_call(operation: str, payload: dict, invoke: Callable, *, is_success: Callable | None = None):
    if _SUPPRESSED.get():
        return invoke()
    current = _CURRENT.get()
    journal = current.ensure_journal() if isinstance(current, PageRun) else current
    if journal is None:
        return invoke()
    return journal.call(operation, payload, invoke, is_success)


def checkpoint_local_value(operation: str, payload: dict, invoke: Callable, *, is_success: Callable | None = None):
    """Freeze generated local inputs alongside active version-two API work.

    Previews stay local, nested provider wrappers own their input/result, and
    legacy journals keep their original sequence of API calls.
    """
    if _SUPPRESSED.get():
        return invoke()
    current = _CURRENT.get()
    if isinstance(current, PageRun):
        if not current.trigger or (current.journal is not None and current.journal.closed):
            return invoke()
        journal = current.ensure_journal()
    else:
        journal = current
    if journal is None or not getattr(journal, "scopes_enabled", False) or getattr(journal, "closed", False):
        return invoke()
    return journal.call(operation, payload, invoke, is_success)


@contextmanager
def checkpoint_scope(*identity):
    """Identify one frozen work item independently of other items' branches.

    Use stable indices/phase names, never credentials or request-content hashes.
    Existing version-one tasks retain their original strict call ordering.
    """
    if any(type(part) not in (str, int) or (isinstance(part, str) and (not part or len(part) > 256)) for part in identity):
        raise AnalysisCheckpointError("A checkpoint work item needs stable string/integer identifiers.")
    token = _CALL_SCOPE.set(_CALL_SCOPE.get() + tuple(identity))
    try:
        yield
    finally:
        _CALL_SCOPE.reset(token)


@contextmanager
def journal_scope(journal):
    token = _CURRENT.set(journal)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current_task_id():
    current = _CURRENT.get()
    if isinstance(current, PageRun):
        current = current.journal
    return current.task_id if isinstance(current, CallJournal) else None


def prepare_background_journal(key: str, label: str):
    """Capture verified identity and frozen inputs on the UI thread, not workers."""
    scope = _CURRENT.get()
    if not isinstance(scope, PageRun) or not scope.trigger:
        return None
    if scope.broken:
        raise AnalysisCheckpointError("Cloud checkpoints are unavailable. Resume this task before making more calls.")
    try:
        if not scope.checked_store:
            scope.checked_store = True
            scope.store = get_cloud_store_for_current_user()
        store = scope.store
        if store is None:
            return None
        resume_id = scope.state.pop(RESUME_TASK, None)
        if resume_id:
            task = store.get_task(resume_id)
            if not task or task.get("page_key") != scope.page_key:
                raise AnalysisCheckpointError("This background task is not available for your account.")
            source = task.get("source") or {}
            if source.get("background_key") != key or source.get("trigger_key") != scope.trigger:
                raise AnalysisResumeMismatch("Use the saved background task's original run action.")
        else:
            task = store.create_task(
                page_key=scope.page_key, settings={"label": label},
                source={"initial_session": pack_snapshot(snapshot_session(scope.state, scope.page_key)),
                        "trigger_key": scope.trigger, "label": label, "background_key": key,
                        "journal_version": 2},
                work_items=[], dynamic_items=True,
            )
        return CallJournal(store, task)
    except AnalysisCheckpointError:
        scope.broken = True
        raise
    except Exception as exc:
        scope.broken = True
        raise AnalysisCheckpointError("Unable to prepare a cloud checkpoint for the background task.") from exc


def acknowledge_task_completion():
    """An authoritative saved final result can close a replay without more calls.

    This is used only after the competition backend verifies a completed official
    record for this exact task, owner and configuration. All call results must
    already be present; this cannot turn partial analyses into complete ones.
    """
    current = _CURRENT.get()
    journal = current.journal if isinstance(current, PageRun) else current
    if not isinstance(journal, CallJournal):
        return
    with journal.lock:
        if journal.broken or journal.failed or any(item.get("status") != "complete" for item in journal.items):
            raise AnalysisCheckpointError("The saved task still has unfinished API calls.")
        journal.verified_complete = True
