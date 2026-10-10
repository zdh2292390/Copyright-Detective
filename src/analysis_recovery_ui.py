"""Restore durable analysis inputs before Streamlit creates their widgets."""
from __future__ import annotations

from datetime import datetime, timezone
import streamlit as st

from src.analysis_checkpoints import OFFICIAL_GAME_PAGE_KEYS
from src.resumable_analysis import (
    AnalysisCheckpointError, PENDING_RESTORE, RESUME_TASK,
    decode, get_cloud_store_for_current_user, is_safe_state_key, key_belongs_to_page, unpack_snapshot,
)


ANALYSIS_SESSION_OWNER = "_analysis_session_owner"
_ROUTE_KEYS = frozenset({
    "main_navigation", "detection_navigation", "game_navigation",
    "content_recall_mode", "knowledge_detection_mode", "unlearning_method",
    "unlearning_detection_mode",
})
_ACCOUNT_ANALYSIS_PREFIXES = (
    "text_", "qa_", "sc_", "sleek_", "persuasion_", "min_k_", "mink_",
    "unlearn_", "representational_", "muse_", "probe_", "preview_",
    "pdf_", "_pdf_", "copyright_game", "_copyright_game", "sidebar_",
    "_analysis_", "_background_job_", "detection_run_", "detection_job_",
    "adv_", "jailbreak_",
)
_ACCOUNT_ANALYSIS_KEYS = frozenset({
    "generated_persuasion_mutations", "stage1_reference_texts", "last_prompt",
    "results_prompt_selector", "knowledge_qa_pdf_upload",
    "_pending_fill_api_key_inputs", "_detection_job_unlock_pending",
})


def clear_account_analysis_state() -> None:
    """Detach account-specific UI state without deleting saved tasks or scores.

    File widgets and background-job identifiers are also cleared: they are not
    serializable snapshots but can still expose the previous account's input or
    cause the next account to harvest an old process-local job. Running workers
    retain their own verified owner and are not canceled by this UI reset.
    """
    for key in list(st.session_state):
        name = str(key)
        if name in _ROUTE_KEYS:
            continue
        if (is_safe_state_key(name) or name.startswith(_ACCOUNT_ANALYSIS_PREFIXES)
                or name in _ACCOUNT_ANALYSIS_KEYS):
            st.session_state.pop(key, None)
    st.query_params.pop("document_analysis", None)


def apply_pending_analysis_restore():
    pending = st.session_state.pop(PENDING_RESTORE, None)
    if not isinstance(pending, dict):
        return
    store = get_cloud_store_for_current_user()
    if store is None:
        raise AnalysisCheckpointError("Sign in to the task's account before restoring analysis.")
    task = store.get_task(pending.get("task_id"))
    if not task:
        raise AnalysisCheckpointError("The saved task is not available for this account.")
    if task.get("page_key") in OFFICIAL_GAME_PAGE_KEYS and task.get("server_managed") is not True:
        raise AnalysisCheckpointError(
            "This older official-game checkpoint cannot be verified. Start a new task; "
            "your saved official scores are retained."
        )
    source = task.get("source") or {}
    initial = unpack_snapshot(source.get("initial_session") or {})
    if not isinstance(initial, dict) or not isinstance(source.get("trigger_key"), str):
        raise AnalysisCheckpointError("This task cannot be restored by this page.")
    final = unpack_snapshot((task.get("metadata") or {}).get("final_session") or {})
    result_keys = {
        "run_snippet_analysis_button": ("text_analysis_results",),
        "generate_qa_button": ("qa_generated_qa_pairs",),
        "run_knowledge_eval_button": ("qa_evaluation_results",),
        "run_sleek_eval_button": ("qa_sleek_results",),
        "run_sleek_button": ("sleek_evaluation_results",),
        "sc_generate_mcq_button": ("sc_generated_mcqs",),
        "sc_run_eval_button": ("sc_evaluation_results",),
        "qa_run_knowmem_eval": (),
        "run_generation": (), "run_probe_button": (), "unlearn_rep_submit_run": (),
    }.get(source["trigger_key"], ("text_analysis_results", "qa_evaluation_results", "sc_evaluation_results"))
    if source["trigger_key"] == "run_min_k_prob_analysis_button":
        result_keys = tuple(key for key in final if key.startswith("min_k_") and key.endswith(("_batch_results", "_last_result")))
    restore_results = task.get("status") == "complete" and isinstance(final, dict) and any(final.get(key) for key in result_keys)
    restored = {**initial, **final} if restore_results else initial
    page_key = str(task.get("page_key") or "")
    for key in list(st.session_state):
        if is_safe_state_key(str(key)) and key_belongs_to_page(str(key), page_key):
            st.session_state.pop(key, None)
    for key, value in restored.items():
        if isinstance(key, str) and is_safe_state_key(key) and key_belongs_to_page(key, page_key):
            st.session_state[key] = decode(value)
    if key_belongs_to_page("qa_generated_qa_pairs", page_key):
        st.session_state.pop("_analysis_restored_qa_bank", None)
        saved_title = st.session_state.get("qa_literature_selection")
        if (st.session_state.get("qa_source_mode") == "Predefined Examples"
                and st.session_state.get("qa_pairs_source") == "predefined"
                and st.session_state.get("qa_generated_qa_pairs")
                and isinstance(saved_title, str) and saved_title):
            st.session_state["_analysis_restored_qa_bank"] = {"title": saved_title}
    st.session_state.pop(RESUME_TASK, None)
    if not restore_results:
        st.session_state[RESUME_TASK] = str(task["id"])
    st.session_state.pop("_analysis_resume_background_key", None)
    if source.get("background_key"):
        st.session_state["_analysis_resume_background_key"] = source["background_key"]
    st.session_state["_analysis_restored_inputs"] = True
    trigger = source["trigger_key"]
    st.session_state.pop(f"detection_run_{trigger}", None)
    if not restore_results:
        st.session_state[f"detection_run_{trigger}"] = True
    st.session_state["detection_job_running"] = False
    st.session_state.pop("detection_run_armed", None)


def _lease_active(task):
    expires = task.get("lease_expires_at")
    if not expires:
        return False
    try:
        parsed = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
        return parsed > datetime.now(timezone.utc)
    except (TypeError, ValueError):
        return True


def render_analysis_recovery(page_key: str):
    if page_key == "Legal Cases Display":
        return
    try:
        store = get_cloud_store_for_current_user()
        if store is None:
            return
        limit_key = f"_analysis_task_list_limit:{page_key}"
        limit = int(st.session_state.get(limit_key, 50))
        tasks = store.list_tasks(page_key=page_key, limit=limit, summary_only=True)
    except Exception:
        st.warning(
            "Cloud recovery is unavailable. Apply supabase/analysis_checkpoints.sql "
            "and check the Supabase connection. New cloud analyses will wait until checkpoints are available."
        )
        return
    has_more = len(tasks) >= limit
    tasks = [task for task in tasks if task.get("dynamic_items")]
    if not tasks:
        return
    with st.expander("Saved analysis tasks · resume after restarting", expanded=any(t.get("status") != "complete" for t in tasks)):
        st.caption("Restores the original input, model and sampling settings. Saved successful API calls are reused. API keys are not stored with tasks.")
        for task in tasks:
            label = (task.get("settings") or {}).get("label", "Analysis")
            completed = int(task.get("completed_items") or 0)
            total = int(task.get("total_items") or 0)
            status = str(task.get("status") or "incomplete")
            active = _lease_active(task)
            st.write(f"{label} — {completed}/{total} saved steps · {status}")
            st.caption(f"Task {str(task['id'])[:8]} · {task.get('updated_at') or ''}")
            if active:
                st.caption("A worker still holds this task. After an unexpected server shutdown, recovery becomes available when its lease expires (up to 3 minutes).")
            if st.button(
                "Restore saved results" if status == "complete" else "Restore and continue",
                key=f"_analysis_resume:{task['id']}", disabled=active,
            ):
                st.session_state[PENDING_RESTORE] = {"task_id": str(task["id"])}
                st.rerun()
            if active and st.button("Stop after the current call", key=f"_analysis_stop:{task['id']}"):
                store.request_stop(str(task["id"]))
                st.rerun()
        if has_more and limit < 1000 and st.button("Show older tasks", key="_analysis_show_older"):
            st.session_state[limit_key] = min(1000, limit + 50)
            st.rerun()
