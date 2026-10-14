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
import logging
from threading import Event, RLock, Thread
from typing import Any, Callable

from src.run_cancellation import (
    AnalysisCancelled, current_cancellation_token, check_cancellation, bind_task_cancellation,
)

from src.analysis_checkpoints import (
    AnalysisCheckpointError, OFFICIAL_GAME_PAGE_KEYS, classify_checkpoint_error,
)
from src.analysis_task_details import summarize_saved_items, update_task_display

_CURRENT: ContextVar[Any] = ContextVar("analysis_journal", default=None)
_SUPPRESSED: ContextVar[bool] = ContextVar("analysis_journal_suppressed", default=False)
_CALL_SCOPE: ContextVar[tuple] = ContextVar("analysis_work_item_scope", default=())
PAGE_KEY = "_analysis_page_key"
PENDING_RESTORE = "_analysis_pending_restore"
RESUME_TASK = "_analysis_resume_task"
VIEW_TASK = "_analysis_view_task"
VIEW_PAGE = "_analysis_view_page"
RUN_TRIGGER = "_analysis_run_trigger"
RUN_LABEL = "_analysis_run_label"
LEASE_TTL = 180
HEARTBEAT_SECONDS = 20
_LOGGER = logging.getLogger(__name__)


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
    "generated_persuasion_mutations", "stage1_reference_texts", "last_stage1_prompt",
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
    "Persuasive Jailbreak Detection": {"input_prompt", "reference", "generation_mode", "strategies", "attempts", "attempts_per_prompt", "generation_checklist", "baseline_selector", "prompt_preview", "generation_results", "evaluation_results", "mutation_store", "last_run_config", "generated_persuasion_mutations", "stage1_reference_texts", "last_stage1_prompt", "confidence_analysis_result", "judge_provider", "judge_model", "judge_temperature", "judge_top_p", "secondary_judge_provider", "secondary_judge_model"},
}
_PAGE_EXTRA_KEYS["Game 3: The Memory Vault Hunt"] = _PAGE_EXTRA_KEYS["Knowledge Memorization Detection"]


_PAGE_ALIASES = {
    "Copyright Challenge": "Game 1: The Hidden Passage Hunt",
    "Copyright Challenge 1": "Game 1: The Hidden Passage Hunt",
    "Game 2: The Hidden Passage Hunt": "Game 1: The Hidden Passage Hunt",
    "Copyright Challenge 2": "Game 2: The Cross-Model Scaling Quest",
    "Game 1: The Cross-Model Scaling Quest": "Game 2: The Cross-Model Scaling Quest",
    "Game 2: The Twin Oracle Duel": "Game 2: The Cross-Model Scaling Quest",
    "Game 2: The Two-Model Continuation Duel": "Game 2: The Cross-Model Scaling Quest",
    "Copyright Challenge 3": "Game 3: The Memory Vault Hunt",
    "Game 3: The Knowledge Memorization Challenge": "Game 3: The Memory Vault Hunt",
}


def canonical_page_key(page_key: str | None) -> str | None:
    return _PAGE_ALIASES.get(page_key, page_key)


def page_supports_analysis_recovery(page_key: str | None) -> bool:
    """Unlearning runs do not persist cloud tasks, including remote execution."""
    return canonical_page_key(page_key) != "Unlearning Detection"


def key_belongs_to_page(key: str, page_key: str | None) -> bool:
    page_key = canonical_page_key(page_key)
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
    """Compress large snapshots without allocating their entire JSON twice."""
    from src.analysis_checkpoints import _check_credentials
    _check_credentials(snapshot)
    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    buffered = bytearray()
    compressed = io.BytesIO()
    zipper = None
    total = 0
    # Base64 expands three input bytes to four output bytes.
    max_compressed = 9 * 1024 * 1024
    size_error = "The saved input is too large. Divide this analysis into smaller batches."
    try:
        for chunk in encoder.iterencode(snapshot):
            for offset in range(0, len(chunk), 65536):
                data = chunk[offset:offset + 65536].encode("utf-8")
                total += len(data)
                if total > 64 * 1024 * 1024:
                    raise AnalysisCheckpointError(size_error)
                if zipper is None and total >= 512 * 1024:
                    zipper = gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0)
                    zipper.write(buffered)
                    buffered.clear()
                buffered.extend(data)
                if zipper is not None and len(buffered) >= 65536:
                    zipper.write(buffered)
                    buffered.clear()
                    if compressed.tell() > max_compressed:
                        raise AnalysisCheckpointError(size_error)
        if zipper is None:
            return snapshot
        zipper.write(buffered)
        buffered.clear()
        zipper.close()
        zipper = None
        if compressed.tell() > max_compressed:
            raise AnalysisCheckpointError(size_error)
        encoded = base64.b64encode(compressed.getvalue()).decode("ascii")
        return {"__analysis_snapshot_gzip__": encoded}
    finally:
        if zipper is not None:
            zipper.close()
        compressed.close()


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


def get_cloud_store_for_current_user(page_key: str | None = None):
    """Verify account identity before constructing an owner-bound cloud store.

    Guests and excluded pages retain existing behavior. Signed-in accounts
    use cloud recovery on supported pages: an authenticated client suffices
    while official competition jobs require the server-only service client.
    Authentication/configuration failures never fall back to temporary files.
    """
    if not page_supports_analysis_recovery(page_key):
        return None
    import streamlit as st
    from src.supabase_client import get_authenticated_client, get_secret
    if not st.session_state.get("access_token"):
        return None
    url = str(get_secret("SUPABASE_URL", "") or "").strip()
    service_key = str(get_secret("SUPABASE_SERVICE_ROLE_KEY", "") or "").strip()
    if not url:
        raise AnalysisCheckpointError(
            "Cloud recovery requires SUPABASE_URL. Configure Supabase before starting a signed-in analysis.",
            kind="configuration",
        )
    try:
        authenticated = get_authenticated_client()
        user = authenticated.auth.get_user().user if authenticated else None
        owner_id = str(getattr(user, "id", "") or "")
        if not owner_id or owner_id != str(st.session_state.get("user_id") or ""):
            raise AnalysisCheckpointError("Sign in again before recovering an analysis task.", kind="auth")
        from src.analysis_checkpoints import SupabaseAnalysisCheckpointStore
        if page_key in OFFICIAL_GAME_PAGE_KEYS:
            if not service_key:
                raise AnalysisCheckpointError(
                    "Official game recovery requires SUPABASE_SERVICE_ROLE_KEY on the server.",
                    kind="configuration",
                )
            from supabase import create_client
            client = create_client(url, service_key)
        else:
            client = authenticated
        return SupabaseAnalysisCheckpointStore(client, owner_id)
    except AnalysisCheckpointError:
        raise
    except Exception as exc:
        raise classify_checkpoint_error(exc, operation="read") from exc


def _checkpoint_error_with_context(exc: Exception, context: str) -> AnalysisCheckpointError:
    diagnostic = classify_checkpoint_error(exc)
    detail = str(diagnostic)
    if diagnostic.code:
        detail = detail.removesuffix(f" (code: {diagnostic.code})")
    return AnalysisCheckpointError(
        f"{context} {detail}", kind=diagnostic.kind, code=diagnostic.code,
        status_code=diagnostic.status_code,
    )


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
        if task.get("kill_requested") or task.get("status") in {"stopping", "killed"}:
            raise AnalysisCheckpointError("This task was killed and cannot be resumed. Start a new analysis.", kind="killed")
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
        self._fresh_call_started = False
        self.lock = RLock()
        self.stop_event = Event()
        self.lease_token = None
        self.read_only = task.get("status") == "complete"
        self.cancellation_token = current_cancellation_token()
        # A failed/racing claim must never replace the active worker's binding.
        self._unbind_cancellation = lambda: None
        self.cleanup_thread = None
        self.metadata = deepcopy(task.get("metadata") or {})
        self.thread = None
        try:
            if not self.read_only:
                self.task = store.claim(self.task_id, ttl_seconds=LEASE_TTL)
                self.lease_token = self.task["lease_token"]
                if self.task.get("kill_requested") or self.task.get("status") in {"stopping", "killed"}:
                    raise AnalysisCheckpointError("This task was killed and cannot be resumed. Start a new analysis.", kind="killed")
                self.metadata.update(deepcopy(self.task.get("metadata") or {}))
                # A large restored journal can take longer than its lease to
                # download. Renew it during loading, before any paid API call.
                if heartbeat:
                    self.thread = Thread(target=self._heartbeat, daemon=True, name="analysis-checkpoint-heartbeat")
                    self.thread.start()
            self.items = []
            while True:
                if self.cancellation_token is not None:
                    self.cancellation_token.check()
                if self.broken:
                    raise AnalysisCheckpointError("Checkpoint saving was interrupted. Resume this task before making more API calls.")
                batch = store.load_items(self.task_id, offset=len(self.items), limit=200)
                if self.cancellation_token is not None:
                    self.cancellation_token.check()
                if self.broken:
                    raise AnalysisCheckpointError("Checkpoint saving was interrupted. Resume this task before making more API calls.")
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
            # Old checkpoints have no card summary. Recover their API history
            # locally; a cached replay or completed task never writes it back.
            restored_display = summarize_saved_items(self.items)
            existing_display = self.metadata.get("display") or {}
            if isinstance(existing_display, dict):
                existing_apis = existing_display.get("apis") or []
                for api in existing_apis if isinstance(existing_apis, list) else []:
                    if isinstance(api, dict) and api not in restored_display["apis"]:
                        restored_display["apis"].append(deepcopy(api))
                if isinstance(existing_display.get("current_step"), dict):
                    restored_display["current_step"] = deepcopy(existing_display["current_step"])
            self.metadata["display"] = restored_display
            self._last_api_was_cached = bool(self.items)
        except BaseException as exc:
            self.stop_event.set()
            if self.lease_token:
                try:
                    store.release(self.task_id, self.lease_token, status="incomplete")
                except Exception:
                    pass
            self._unbind_cancellation()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, AnalysisCheckpointError):
                raise
            raise classify_checkpoint_error(exc) from exc
        self._unbind_cancellation = bind_task_cancellation(
            getattr(store, "owner_id", ""), self.task_id, self.cancellation_token,
        )

    def report_progress(self, current: int, total: int, *, unit: str = "items", stage: str = "analysis"):
        """Persist business work positions independently of provider call counts."""
        from src.analysis_work_progress import normalize_work_progress
        progress = normalize_work_progress({"current": current, "total": total, "unit": unit, "stage": stage})
        if progress is None:
            raise AnalysisCheckpointError("Invalid analysis work progress.")
        with self.lock:
            if self.read_only or self.closed:
                return
            check_cancellation()
            if self.cancellation_token is not None:
                self.cancellation_token.check()
            if self.broken:
                raise AnalysisCheckpointError("Progress saving was interrupted. Resume this task before making more API calls.")
            previous = normalize_work_progress(self.metadata.get("work_progress"))
            # Cached replay revisits already completed positions. Keep the latest
            # persisted position until replay reaches the remaining work.
            stored = normalize_work_progress((self.task.get("metadata") or {}).get("work_progress"))
            for candidate in (previous, stored):
                if candidate is not None and all(candidate[key] == progress[key] for key in ("total", "unit", "stage")):
                    progress["current"] = max(progress["current"], candidate["current"])
            if progress == previous:
                return
            self.metadata["work_progress"] = progress
            if self._last_api_was_cached:
                # Avoid one cloud write per replayed unit. The next fresh call
                # or finish() commits this reconstructed position with its data.
                return
            try:
                self.task = self.store.save_progress(
                    self.task_id, metadata=deepcopy(self.metadata), lease_token=self.lease_token,
                )
            except Exception as exc:
                self.broken = True
                raise _checkpoint_error_with_context(
                    exc, "The analysis progress could not be saved. Saved calls remain available for recovery."
                ) from exc
            if self.task.get("stop_requested"):
                if self.cancellation_token is not None:
                    self.cancellation_token.cancel()
                raise AnalysisCancelled("Analysis stopped. Saved progress is preserved.")

    def _display_metadata(self, request: dict, index: int, phase: str) -> dict:
        # Keep locally recovered API history when an old task's heartbeat
        # returns metadata that does not yet contain all of its saved calls.
        previous_display = self.metadata.get("display")
        self.metadata.update({key: deepcopy(value) for key, value in (self.task.get("metadata") or {}).items()
                              if key != "work_progress"})
        self.metadata["display"] = update_task_display(previous_display, request, index, phase)
        return deepcopy(self.metadata)

    def _heartbeat(self):
        while not self.stop_event.wait(HEARTBEAT_SECONDS):
            try:
                task = self.store.heartbeat(self.task_id, self.lease_token, ttl_seconds=LEASE_TTL)
                if task.get("stop_requested"):
                    if self.cancellation_token is not None:
                        self.cancellation_token.cancel()
                    self.broken = True
                    # Keep renewing until finish() releases the lease. A stopped
                    # background runner may still be draining an accepted request;
                    # another process must not resume it during that cleanup.
            except Exception:
                self.broken = True
                # Nested provider calls must also stop when their checkpoint
                # lease fails, rather than spending more API calls unsaved.
                if self.cancellation_token is not None:
                    self.cancellation_token.cancel()
                return

    def call(self, operation: str, payload: dict, invoke: Callable, is_success: Callable | None = None):
        # Current page runners are sequential. The lock also protects accidental
        # concurrent use, so request order and repeated identical calls stay stable.
        with self.lock:
            token = current_cancellation_token()
            if token is not None and token is not self.cancellation_token:
                self._unbind_cancellation()
                self.cancellation_token = token
                self._unbind_cancellation = bind_task_cancellation(
                    getattr(self.store, "owner_id", ""), self.task_id, token,
                )
            check_cancellation()
            if self.cancellation_token is not None:
                self.cancellation_token.check()
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
                        self._last_api_was_cached = True
                        return decode(item["result"])
                    if self.read_only:
                        raise AnalysisResumeMismatch("This completed task has an unfinished saved response. Viewing results cannot retry API calls.")
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
                self.task = self.store.heartbeat(self.task_id, self.lease_token, ttl_seconds=LEASE_TTL)
                if self.task.get("stop_requested"):
                    if self.cancellation_token is not None:
                        self.cancellation_token.cancel()
                    raise AnalysisCancelled("Analysis stopped. Saved progress is preserved.")
                self.task = self.store.save_progress(
                    self.task_id, metadata=self._display_metadata(request, index, "running"),
                    lease_token=self.lease_token,
                )
                if self.task.get("stop_requested"):
                    if self.cancellation_token is not None:
                        self.cancellation_token.cancel()
                    raise AnalysisCancelled("Analysis stopped. Saved progress is preserved.")
            except AnalysisCheckpointError:
                self.failed = self.broken = True
                raise
            except Exception as exc:
                self.failed = self.broken = True
                raise _checkpoint_error_with_context(exc, "Unable to save the next request. No API call was started.") from exc

            self._last_api_was_cached = False
            self._fresh_call_started = True
            token = _SUPPRESSED.set(True)
            try:
                check_cancellation()
                result = invoke()
            except BaseException:
                self.failed = True
                try:
                    self.task = self.store.save_progress(
                        self.task_id, metadata=self._display_metadata(request, index, "failed"),
                        lease_token=self.lease_token,
                    )
                except Exception:
                    self.broken = True
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
                    metadata=self._display_metadata(request, index, "complete" if success else "failed"),
                    lease_token=self.lease_token,
                )
                item.update(status="complete" if success else "failed", result=encoded,
                            attempts=int(item.get("attempts") or 0) + 1)
            except Exception as exc:
                self.failed = self.broken = True
                raise _checkpoint_error_with_context(
                    exc, "The API returned, but its result could not be saved to Supabase. "
                    "Further calls have stopped; this unsaved call may need to run again."
                ) from exc
            check_cancellation()
            return result

    def finish(self, success: bool, final_snapshot: dict | None = None):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.read_only:
                self.stop_event.set()
                self._unbind_cancellation()
                if success and (self.failed or self.broken or len(self.visited_items) != len(self.items)):
                    self.failed = self.broken = True
                    raise AnalysisResumeMismatch("The saved report could not be fully reconstructed because some saved calls were not replayed. Viewing results cannot create missing API calls.")
                return
            success = success and not (self.cancellation_token is not None and self.cancellation_token.cancelled)
            complete = success and not self.failed and not self.broken and (len(self.visited_items) == len(self.items) or self.verified_complete)
            metadata = deepcopy(self.metadata)
            metadata.update({key: deepcopy(value) for key, value in (self.task.get("metadata") or {}).items()
                             if key not in {"display", "work_progress"}})
            if not complete and not self._fresh_call_started:
                # An interrupted replay may still be rebuilding an earlier
                # stage. Retain the last durable stage until real work resumes.
                from src.analysis_work_progress import normalize_work_progress
                stored = normalize_work_progress((self.task.get("metadata") or {}).get("work_progress"))
                if stored is not None:
                    metadata["work_progress"] = stored
            if complete and final_snapshot is not None:
                metadata["final_session"] = final_snapshot

            def release_checkpoint():
                self.stop_event.set()
                self._unbind_cancellation()
                try:
                    self.store.release(self.task_id, self.lease_token,
                                       status="complete" if complete else "incomplete", metadata=metadata)
                except Exception as exc:
                    self.broken = True
                    raise _checkpoint_error_with_context(
                        exc, "The final checkpoint could not be saved. Saved calls remain available for recovery."
                    ) from exc

            if self.cancellation_token is not None and self.cancellation_token.pending_count:
                # The UI can stop immediately. Retain and renew the lease until
                # accepted requests drain, so another browser cannot repeat the
                # same pending paid call while its old request is still running.
                token = self.cancellation_token
                def cleanup():
                    token.wait_for_idle()
                    try:
                        release_checkpoint()
                    except AnalysisCheckpointError as exc:
                        # No Streamlit objects cross into this cleanup thread.
                        # A failed release leaves the saved task recoverable
                        # after the fenced lease expires.
                        _LOGGER.warning("Stopped analysis checkpoint cleanup failed: %s", exc)
                try:
                    self.cleanup_thread = Thread(
                        target=cleanup, daemon=True, name="analysis-checkpoint-stop-cleanup",
                    )
                    self.cleanup_thread.start()
                except Exception:
                    # Resource exhaustion must not reopen a still-running paid
                    # request. The already-stopped coordinator can drain it;
                    # the browser's force-stop callback does not wait here.
                    token.wait_for_idle()
                    release_checkpoint()
            else:
                release_checkpoint()


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
        # Capture the selection before widgets or a replay can consume its marker.
        self.view_task_id = str(state.get(VIEW_TASK) or "") or None
        selected_page = state.get(VIEW_PAGE)
        if not isinstance(selected_page, str) or not selected_page.strip():
            # Existing sessions already bind feedback to the verified task/page.
            # Orphaned legacy markers without that binding remain conservative.
            feedback = state.get("_analysis_restore_feedback")
            if (isinstance(feedback, dict)
                    and str(feedback.get("task_id") or "") == self.view_task_id):
                selected_page = feedback.get("page_key")
        if (self.view_task_id and isinstance(selected_page, str) and selected_page.strip()
                and canonical_page_key(selected_page) != canonical_page_key(page_key)):
            previous_view = self.view_task_id
            self.view_task_id = None
            if str(state.get(VIEW_TASK) or "") == previous_view:
                state.pop(VIEW_TASK, None)
                state.pop(VIEW_PAGE, None)
            if str(state.get(RESUME_TASK) or "") == previous_view:
                state.pop(RESUME_TASK, None)

    def ensure_journal(self):
        if not page_supports_analysis_recovery(self.page_key):
            return None
        if self.broken:
            raise AnalysisCheckpointError("Cloud checkpoints are unavailable. Restore the task before making more calls.")
        if self.journal is not None:
            return self.journal
        if not self.trigger:
            if self.view_task_id or self.state.get("access_token"):
                self.broken = True
                raise AnalysisCheckpointError(
                    "Use this page's Run action before making an API request. "
                    "Viewing saved results cannot start an untracked signed-in analysis."
                )
            return None  # Guests retain their existing local behavior.
        if not self.checked_store:
            self.checked_store = True
            try:
                self.store = get_cloud_store_for_current_user(page_key=self.page_key)
            except Exception:
                self.broken = True
                raise
        if self.store is None:
            if self.view_task_id:
                raise AnalysisCheckpointError("Sign in before viewing this task's saved results. No API call was started.", kind="auth")
            return None
        try:
            resume_id = self.view_task_id or self.state.get(RESUME_TASK)
            if resume_id:
                task = self.store.get_task(resume_id)
                if not task or canonical_page_key(task.get("page_key")) != canonical_page_key(self.page_key):
                    raise AnalysisCheckpointError("The saved task is not available on this page for your account.")
                if self.view_task_id and task.get("status") != "complete":
                    raise AnalysisCheckpointError("Only completed tasks can be viewed without running an analysis.")
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
            # Failed validation or lease startup must retain the task selected
            # for Resume; the next Run must not silently create a new task.
            if not self.view_task_id and resume_id and self.state.get(RESUME_TASK) == resume_id:
                self.state.pop(RESUME_TASK, None)
            self.state["_analysis_current_task_id"] = self.journal.task_id
            return self.journal
        except AnalysisCheckpointError:
            self.broken = True
            raise
        except Exception as exc:
            self.broken = True
            raise classify_checkpoint_error(exc) from exc

    def finish(self, success: bool):
        if self.journal is not None and not self.journal.closed:
            final_snapshot = None
            if success and not self.journal.read_only:
                try:
                    final_snapshot = pack_snapshot(snapshot_session(self.state, self.page_key, max_bytes=8 * 1024 * 1024))
                except Exception:
                    # The per-call journal remains authoritative. An optional
                    # report snapshot must not prevent completion or leave a
                    # heartbeat running forever; rebuild from saved calls.
                    final_snapshot = None
            self.journal.finish(success, final_snapshot)
            if success and self.view_task_id and not self.journal.failed and not self.journal.broken:
                if self.state.get(VIEW_TASK) == self.view_task_id:
                    self.state.pop(VIEW_TASK, None)
                    self.state.pop(VIEW_PAGE, None)
                if self.state.get(RESUME_TASK) == self.view_task_id:
                    self.state.pop(RESUME_TASK, None)
        elif success and self.journal is None and self.view_task_id and self.state.get(RESUME_TASK) != self.view_task_id:
            # Direct final snapshots have no journal to close. Release their
            # per-render safety marker so a later explicit Run creates new work.
            if self.state.get(VIEW_TASK) == self.view_task_id:
                self.state.pop(VIEW_TASK, None)
                self.state.pop(VIEW_PAGE, None)


def is_viewing_saved_results() -> bool:
    """Identify both cached UI replay and direct readonly journal consumers."""
    scope = _CURRENT.get()
    return bool((isinstance(scope, PageRun) and scope.view_task_id)
                or (isinstance(scope, CallJournal) and scope.read_only))


def saved_view_api_key(api_key):
    """Let cached-only replay pass old input guards without saving a credential.

    Provider keys are absent from request fingerprints. The selected completed
    task still authorizes every exact cached request, and a missing response or
    invalid login is rejected before its live provider callback can execute.
    """
    scope = _CURRENT.get()
    if isinstance(scope, PageRun) and scope.view_task_id and not str(api_key or "").strip():
        return "saved-response-only"
    return api_key


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


def report_analysis_progress(current: int, total: int, *, unit: str = "items", stage: str = "analysis"):
    """Record the runner's real current/total without changing checkpoint identity."""
    if _SUPPRESSED.get():
        # Nested SDK/provider work may run on another thread while its parent
        # owns the journal lock. Only the business coordinator publishes units.
        return
    scope = _CURRENT.get()
    if isinstance(scope, PageRun):
        if not scope.trigger or (scope.journal is not None and scope.journal.closed):
            return
        journal = scope.ensure_journal()
    else:
        journal = scope
    if isinstance(journal, CallJournal):
        journal.report_progress(current, total, unit=unit, stage=stage)


def checkpoint_call(operation: str, payload: dict, invoke: Callable, *, is_success: Callable | None = None):
    check_cancellation()
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
    if not page_supports_analysis_recovery(scope.page_key):
        return None
    if scope.broken:
        raise AnalysisCheckpointError("Cloud checkpoints are unavailable. Resume this task before making more calls.")
    if scope.view_task_id:
        raise AnalysisCheckpointError("Completed background results must be restored locally. Viewing results cannot start a worker.")
    try:
        if not scope.checked_store:
            scope.checked_store = True
            scope.store = get_cloud_store_for_current_user(page_key=scope.page_key)
        store = scope.store
        if store is None:
            return None
        resume_id = scope.state.get(RESUME_TASK)
        if resume_id:
            task = store.get_task(resume_id)
            if not task or canonical_page_key(task.get("page_key")) != canonical_page_key(scope.page_key):
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
        journal = CallJournal(store, task)
        if resume_id and scope.state.get(RESUME_TASK) == resume_id:
            scope.state.pop(RESUME_TASK, None)
        return journal
    except AnalysisCheckpointError:
        scope.broken = True
        raise
    except Exception as exc:
        scope.broken = True
        raise classify_checkpoint_error(exc) from exc


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
