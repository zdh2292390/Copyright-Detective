"""Restore durable analysis inputs before Streamlit creates their widgets."""
from __future__ import annotations

from datetime import datetime, timezone
import re
from urllib.parse import urlsplit
import streamlit as st

from src.analysis_checkpoints import OFFICIAL_GAME_PAGE_KEYS, classify_checkpoint_error
from src.analysis_work_progress import format_work_progress
from src.analysis_result_restore import completed_result_snapshot
from src.resumable_analysis import (
    AnalysisCheckpointError, PENDING_RESTORE, RESUME_TASK, VIEW_TASK, VIEW_PAGE,
    canonical_page_key, decode, get_cloud_store_for_current_user, is_safe_state_key, key_belongs_to_page,
    page_supports_analysis_recovery, unpack_snapshot,
)


ANALYSIS_SESSION_OWNER = "_analysis_session_owner"
_RESTORE_FEEDBACK = "_analysis_restore_feedback"
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
    if not page_supports_analysis_recovery(task.get("page_key")):
        # Old queued restores must not unpack inputs or restart excluded pages.
        if st.session_state.get(RESUME_TASK) == str(task["id"]):
            st.session_state.pop(RESUME_TASK, None)
        return
    if task.get("kill_requested") or task.get("status") in {"stopping", "killed"}:
        raise AnalysisCheckpointError("This task was killed and cannot be resumed. Start a new analysis.", kind="killed")
    if task.get("page_key") in OFFICIAL_GAME_PAGE_KEYS and task.get("server_managed") is not True:
        raise AnalysisCheckpointError(
            "This older official-game checkpoint cannot be verified. Start a new task; "
            "your saved official scores are retained."
        )
    if task.get("status") != "complete" and _lease_active(task):
        raise AnalysisCheckpointError(
            "This task is still active or finishing earlier requests. Refresh its status before resuming.",
            kind="lease",
        )
    source = task.get("source") or {}
    initial = unpack_snapshot(source.get("initial_session") or {})
    if not isinstance(initial, dict) or not isinstance(source.get("trigger_key"), str):
        raise AnalysisCheckpointError("This task cannot be restored by this page.")
    completed = task.get("status") == "complete"
    final = completed_result_snapshot(task, store=store) if completed else None
    restore_results = final is not None
    final = final or {}
    restored = {**initial, **final} if restore_results else initial
    page_key = canonical_page_key(str(task.get("page_key") or ""))
    for key in list(st.session_state):
        if is_safe_state_key(str(key)) and key_belongs_to_page(str(key), page_key):
            st.session_state.pop(key, None)
    for key, value in restored.items():
        if isinstance(key, str) and is_safe_state_key(key) and key_belongs_to_page(key, page_key):
            st.session_state[key] = decode(value)
    _restore_execution_mode(page_key, source["trigger_key"], owner_id=task.get("owner_id") or store.owner_id)
    st.session_state["_analysis_restored_models"] = {
        key: st.session_state[key] for key in restored
        if isinstance(key, str) and key.startswith("sidebar_") and key.endswith("_model_selectbox")
        and isinstance(st.session_state.get(key), str) and st.session_state[key].strip()
    }
    if key_belongs_to_page("qa_generated_qa_pairs", page_key):
        st.session_state.pop("_analysis_restored_qa_bank", None)
        saved_title = st.session_state.get("qa_literature_selection")
        if (st.session_state.get("qa_source_mode") == "Predefined Examples"
                and st.session_state.get("qa_pairs_source") == "predefined"
                and st.session_state.get("qa_generated_qa_pairs")
                and isinstance(saved_title, str) and saved_title):
            st.session_state["_analysis_restored_qa_bank"] = {"title": saved_title}
    st.session_state.pop(RESUME_TASK, None)
    st.session_state.pop(VIEW_TASK, None)
    st.session_state.pop(VIEW_PAGE, None)
    st.session_state.pop("_analysis_current_task_id", None)
    if completed:
        st.session_state[VIEW_TASK] = str(task["id"])
        st.session_state[VIEW_PAGE] = page_key
    if not restore_results:
        st.session_state[RESUME_TASK] = str(task["id"])
    st.session_state.pop("_analysis_resume_background_key", None)
    if source.get("background_key") and not completed:
        st.session_state["_analysis_resume_background_key"] = source["background_key"]
    st.session_state["_analysis_restored_inputs"] = True
    trigger = source["trigger_key"]
    for key in list(st.session_state):
        if str(key).startswith("detection_run_"):
            st.session_state.pop(key, None)
    if not restore_results:
        st.session_state[f"detection_run_{trigger}"] = True
    st.session_state["detection_job_running"] = False
    st.session_state.pop("detection_run_armed", None)
    st.session_state[_RESTORE_FEEDBACK] = {
        "task_id": str(task["id"]), "page_key": page_key, "trigger_key": trigger,
        "action": "results" if completed else "resume", "replay": completed and not restore_results,
        "status": "restored",
    }


def _restore_execution_mode(page_key, trigger, *, owner_id=None):
    # The saved action identifies its original branch, including older snapshots
    # without submode widgets. Restore this before any page widgets are created.
    if page_key in {"Content Recall Detection", "Knowledge Memorization Detection",
                    "Persuasive Jailbreak Detection", "Unlearning Detection", "Legal Cases Display"}:
        st.session_state["main_navigation"] = page_key
        st.session_state["detection_navigation"] = page_key
        st.session_state["game_navigation"] = None
    elif page_key in {"Game 1: The Hidden Passage Hunt", "Game 2: The Cross-Model Scaling Quest", "Game 3: The Memory Vault Hunt"}:
        st.session_state["main_navigation"] = page_key
        st.session_state["game_navigation"] = page_key
        st.session_state["detection_navigation"] = None
        if page_key == "Game 1: The Hidden Passage Hunt" and owner_id:
            stages = {f"copyright_game:{owner_id}:run_stage_one": "stage_one",
                      f"copyright_game:{owner_id}:run_stage_two": "stage_two"}
            if trigger in stages:
                st.session_state[f"copyright_game:{owner_id}:active_view"] = stages[trigger]
    if page_key == "Unlearning Detection":
        mode = {"run_min_k_prob_analysis_button": "MIN-K% PROB",
                "unlearn_rep_submit_run": "Representational Analysis"}.get(trigger)
        if mode:
            st.session_state["unlearning_detection_mode"] = mode
    if page_key == "Content Recall Detection" and trigger in {
        "run_snippet_analysis_button", "text_generate_report_narrative",
    }:
        st.session_state["content_recall_mode"] = "Text Memorization Detection"
    elif page_key in {"Knowledge Memorization Detection", "Game 3: The Memory Vault Hunt"}:
        if trigger in {"sc_generate_mcq_button", "sc_run_eval_button"}:
            st.session_state["knowledge_detection_mode"] = "Single-choice Question"
        elif trigger in {"generate_qa_button", "run_knowledge_eval_button", "run_sleek_eval_button", "qa_run_knowmem_eval"}:
            st.session_state["knowledge_detection_mode"] = "Open-ended Question"
            mode = {"run_knowledge_eval_button": "Standard",
                    "run_sleek_eval_button": "Step-by-step Leaking and Extraction"}.get(trigger)
            if mode:
                st.session_state["qa_evaluation_mode"] = mode
                st.session_state["qa_evaluation_mode_radio"] = mode


def _queue_analysis_restore(task_id):
    # A Streamlit callback runs before the next script creates page widgets.
    st.session_state[PENDING_RESTORE] = {"task_id": str(task_id)}


def render_analysis_restore_feedback(page_key):
    if not page_supports_analysis_recovery(page_key):
        return None
    notice = st.session_state.get(_RESTORE_FEEDBACK)
    if not isinstance(notice, dict) or notice.get("page_key") != page_key:
        return None
    slot = st.empty()
    if notice.get("status") in {"blocked", "error"}:
        slot.warning(notice["message"])
    elif notice.get("action") == "results":
        if notice.get("replay"):
            slot.info("Rebuilding final results from saved calls.")
        else:
            slot.success("Saved analysis results restored.")
    else:
        slot.info("Saved inputs restored. Resuming the original task and reusing its successful calls.")
    return slot


def finish_analysis_restore_feedback(page_key, slot, *, failed=False):
    notice = st.session_state.get(_RESTORE_FEEDBACK)
    if slot is None or not isinstance(notice, dict) or notice.get("page_key") != page_key:
        return
    task_id = notice.get("task_id")
    if failed:
        message = ("Saved results could not be displayed. Retry View results; saved calls remain unchanged."
                   if notice.get("action") == "results" else
                   "The saved task could not finish resuming. Review the error below and retry Resume. Saved progress is preserved.")
        notice.update(status="error", message=message)
        slot.warning(message)
    elif st.session_state.get(RESUME_TASK) == task_id:
        pending_action = st.session_state.get(f"detection_run_{notice.get('trigger_key')}")
        message = (
            "The saved task has not restarted because its Run action is unavailable with the restored inputs. "
            "Review the analysis mode and input, then retry Resume. Saved progress is preserved."
            if pending_action else
            "The saved task has not restarted. Complete the required API settings or inputs shown below, "
            "then retry Resume. Saved progress is preserved."
        )
        if notice.get("action") == "results":
            message = "The saved result view has not finished. Retry View results; no new API calls will be made."
        notice.update(status="blocked", message=message)
        slot.warning(message)
    else:
        slot.success("Saved analysis results restored." if notice.get("action") == "results" else
                     "The original saved task resumed using its existing checkpoints.")
        st.session_state.pop(_RESTORE_FEEDBACK, None)


def _lease_active(task):
    expires = task.get("lease_expires_at")
    if not expires:
        return False
    try:
        parsed = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
        return parsed > datetime.now(timezone.utc)
    except (TypeError, ValueError):
        return True


def _task_display(task):
    """Prefer live metadata; read recent compact checkpoints for older tasks."""
    from src.analysis_task_details import summarize_saved_items
    display = task.get("display") or (task.get("metadata") or {}).get("display")
    if isinstance(display, dict) and display.get("current_step"):
        return display
    items = []
    for row in task.get("latest_items") or []:
        request = {key: row.get(key) for key in (
            "provider", "model", "model_name", "endpoint", "base_url", "agent_url", "model_path",
        ) if row.get(key)}
        item = {**row, "input": {"operation": row.get("operation"), "request": request}}
        if row.get("scope"):
            item["input"]["scope"] = row["scope"]
        items.append(item)
    return summarize_saved_items(items) if items else {}


_OPERATION_LABELS = {
    "llm.completion": "Generate model response",
    "single_choice.evaluate": "Evaluate multiple-choice answer",
    "decop.openai": "Evaluate DE-COP options",
    "decop.anthropic": "Evaluate DE-COP options",
    "min_k.completion_probe": "Measure token probabilities",
    "prompt.full": "Prepare sampled prompt",
    "agent.deploy": "Deploy model for analysis",
    "representational.submit": "Submit representational analysis",
    "representational.result": "Wait for representational results",
    "game.stage_two.reserve": "Prepare game stage 2",
}
_SCOPE_LABELS = {
    "knowledge_qa": "Evaluate knowledge Q&A",
    "knowmem_answer": "Evaluate knowledge memory",
    "sleek_qa": "Evaluate SLEEK Q&A",
    "sleek_probe": "Run SLEEK probe",
    "text_report_narrative": "Generate report explanation",
    "persuasion_evaluate": "Evaluate persuasion response",
    "persuasion_judge": "Judge persuasion response",
    "custom_mutation": "Generate custom prompt mutation",
    "mutation": "Generate prompt mutation",
    "min_k_example": "Analyze Min-K example",
    "min_k_stage": "Min-K probability analysis",
    "agent_deploy": "Deploy analysis model",
    "game_direct": "Game direct recall",
    "game_mutations": "Game prompt mutations",
    "game_evaluation": "Game response evaluation",
    "game_knowledge_direct": "Game knowledge recall",
    "game_knowledge_evaluation": "Game knowledge evaluation",
}


def _step_description(task, display):
    step = display.get("current_step") or {}
    if not isinstance(step, dict) or not step.get("operation"):
        return "Preparing the first checkpoint" if _lease_active(task) else "No checkpoint recorded yet"
    operation = str(step["operation"])
    scope = step.get("scope") or []
    name = _OPERATION_LABELS.get(operation, operation.replace(".", " / ").replace("_", " "))
    if scope:
        phase = str(scope[0])
        name = _SCOPE_LABELS.get(phase, phase.replace("_", " ").capitalize())
        parts = scope[1:]
        if phase == "knowledge_qa" and len(parts) >= 2 and all(type(part) is int for part in parts[:2]):
            name += f" / run {parts[0] + 1}, question {parts[1] + 1}"
        elif phase == "sleek_qa" and len(parts) >= 2 and all(type(part) is int for part in parts[:2]):
            name += f" / question {parts[0] + 1}, run {parts[1] + 1}"
        else:
            details = [f"item {part + 1}" if type(part) is int else str(part).replace("_", " ") for part in parts]
            if details:
                name += " / " + ", ".join(details)
    if task.get("status") == "complete":
        return "Finished / " + name
    if not _lease_active(task):
        return "Stopped at / " + name if task.get("stop_requested") else "Paused at / " + name
    if task.get("stop_requested"):
        return "Stopping at / " + name
    if step.get("phase") == "complete":
        return "Last completed / " + name
    if step.get("phase") == "failed":
        return "Failed / " + name
    return "Current / " + name


def _kill_saved_task(store, task_id, *, task_completed=False):
    """Archive completed results or terminate this owner's unfinished task."""
    if not task_completed:
        from src.run_cancellation import cancel_task
        cancel_task(store.owner_id, str(task_id))
    error_key = f"_analysis_kill_error:{task_id}"
    try:
        killed = store.kill_task(str(task_id))
        if killed.get("status") == "complete" and not killed.get("kill_requested"):
            if not killed.get("archived_at"):
                raise AnalysisCheckpointError("Apply supabase/analysis_completed_archive.sql before archiving completed results.", kind="schema")
        elif not killed.get("kill_requested") and killed.get("status") not in {"stopping", "killed"}:
            raise AnalysisCheckpointError("The cloud task did not acknowledge Kill.")
    except Exception as exc:
        diagnostic = classify_checkpoint_error(exc)
        st.session_state[error_key] = (
            "The cloud Kill request failed. Retry Kill to terminate and remove this task. "
            f"Saved progress is preserved. {diagnostic}"
        )
        return False
    st.session_state.pop(error_key, None)
    if st.session_state.get(RESUME_TASK) == str(task_id):
        st.session_state.pop(RESUME_TASK, None)
    if st.session_state.get(VIEW_TASK) == str(task_id):
        st.session_state.pop(VIEW_TASK, None)
        st.session_state.pop(VIEW_PAGE, None)
    pending = st.session_state.get(PENDING_RESTORE)
    if isinstance(pending, dict) and pending.get("task_id") == str(task_id):
        st.session_state.pop(PENDING_RESTORE, None)
    notice = st.session_state.get(_RESTORE_FEEDBACK)
    if isinstance(notice, dict) and notice.get("task_id") == str(task_id):
        st.session_state.pop(_RESTORE_FEEDBACK, None)
    return True


_SAVED_ANALYSIS_STYLES = """<style>
/* Scoped to task lists; existing Run controls are unaffected. */
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [data-testid="stExpander"],
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [data-testid="stExpander"] > details {
    border-radius: 6px !important;
    box-shadow: none !important;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [data-testid="stExpander"] summary {
    min-height: 2.25rem;
    padding: 0.5rem 0.75rem !important;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [data-testid="stExpander"] summary p {
    font-size: 0.875rem;
    line-height: 1.25;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [data-testid="stExpanderDetails"] {
    padding: 0.625rem 0.75rem !important;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [data-testid="stExpanderDetails"] > [data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [data-testid="stButton"] button {
    height: 2.125rem !important;
    min-height: 2.125rem !important;
    padding: 0.3rem 0.65rem !important;
    border-width: 1px !important;
    border-radius: 4px !important;
    box-shadow: none !important;
    transform: none !important;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [data-testid="stButton"] button p {
    font-size: 0.8125rem !important;
    line-height: 1.2 !important;
}
.st-key-analysis_saved_list .st-key-analysis_saved_toolbar {
    gap: 0.375rem 0.75rem !important;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [class*="st-key-analysis_saved_info_"] [data-testid="stHorizontalBlock"] {
    gap: 0.125rem 0.75rem !important;
}
.st-key-analysis_saved_list .st-key-analysis_saved_toolbar [data-testid="stCaptionContainer"] p {
    font-size: 0.8125rem;
    line-height: 1.35;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [class*="st-key-analysis_saved_task_"] {
    padding: 0.625rem 0.75rem !important;
    gap: 0.5rem !important;
    border-radius: 6px !important;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [class*="st-key-analysis_saved_info_"] p {
    font-size: 0.875rem;
    line-height: 1.4;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [class*="st-key-analysis_saved_info_"] [data-testid="stCaptionContainer"] p {
    font-size: 0.8125rem;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [class*="st-key-analysis_saved_task_"] [data-testid="stButton"] button {
    width: 6rem !important;
    min-width: 6rem !important;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [class*="st-key-analysis_saved_actions_"] {
    gap: 0.5rem !important;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [class*="st-key-analysis_saved_actions_"] > [data-testid="stLayoutWrapper"]:has(> [class*="st-key-analysis_saved_details_"]) {
    flex: 1 1 10rem !important;
    min-width: 10rem;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [class*="st-key-force_stop_controls_"] [data-testid="stButton"] button:not(:disabled) {
    background: #fff7f7 !important;
    border-color: #fca5a5 !important;
    color: #b91c1c !important;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [class*="st-key-force_stop_controls_"] [data-testid="stButton"] button:not(:disabled):hover {
    background: #fee2e2 !important;
    border-color: #ef4444 !important;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [class*="st-key-force_stop_controls_"] [data-testid="stButton"] button:not(:disabled) p {
    color: #b91c1c !important;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [class*="st-key-analysis_saved_details_"] [data-testid="stExpander"],
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [class*="st-key-analysis_saved_details_"] [data-testid="stExpander"] > details {
    border: none !important;
    background: transparent !important;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [class*="st-key-analysis_saved_details_"] [data-testid="stExpander"] summary {
    min-height: 2.125rem;
    padding: 0.3rem 0.5rem !important;
    border-radius: 4px !important;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [class*="st-key-analysis_saved_details_"] [data-testid="stExpanderDetails"] {
    padding: 0.5rem 0 !important;
}
:is(.st-key-analysis_saved_list, .st-key-analysis_archived_list) [class*="st-key-analysis_saved_details_"] [data-testid="stExpanderDetails"] [data-testid="stVerticalBlock"] {
    gap: 0.375rem !important;
}
</style>"""


def render_analysis_recovery(page_key: str):
    if page_key == "Legal Cases Display" or not page_supports_analysis_recovery(page_key):
        return
    try:
        store = get_cloud_store_for_current_user(page_key=page_key)
        if store is None:
            return
        limit_key = f"_analysis_task_list_limit:{page_key}"
        limit = int(st.session_state.get(limit_key, 50))
        tasks = store.list_tasks(
            page_key=page_key, limit=limit, summary_only=True, exclude_killed=True, exclude_archived=True,
            statuses=["creating", "queued", "running", "incomplete", "complete"],
        )
    except Exception as exc:
        diagnostic = classify_checkpoint_error(exc, operation="read")
        st.warning(f"Cloud recovery is unavailable. {diagnostic}")
        from src.supabase_client import get_secret
        try:
            hostname = urlsplit(get_secret("SUPABASE_URL")).hostname or ""
            match = re.fullmatch(r"([a-z0-9]{20})\.supabase\.co", hostname)
        except (TypeError, ValueError):
            match = None
        if match:
            st.caption(f"Configured Supabase project: {match.group(1)}. Apply migrations in this same project.")
        st.caption("New signed-in analyses will wait until cloud checkpoints are available.")
        return
    has_more = len(tasks) >= limit
    tasks = [task for task in tasks if task.get("dynamic_items")
             and not task.get("kill_requested") and not task.get("archived_at")
             and task.get("status") not in {"stopping", "killed"}]
    st.html(_SAVED_ANALYSIS_STYLES)
    if tasks:
        with st.container(key="analysis_saved_list", gap=None):
            with st.expander("Saved analyses", expanded=any(t.get("status") != "complete" for t in tasks)):
                with st.container(horizontal=True, vertical_alignment="center", gap="small",
                                  key="analysis_saved_toolbar"):
                    st.caption("Resume reuses saved progress. Kill removes the task from this list.")
                    if st.button("Refresh", key=f"_analysis_refresh:{page_key}", width="content"):
                        st.rerun()
                for task in tasks:
                    _render_saved_analysis_card(task, store, page_key)
                if has_more and limit < 1000 and st.button("Show more", key=f"_analysis_show_older:{page_key}"):
                    st.session_state[limit_key] = min(1000, limit + 50)
                    st.rerun()

    _render_archived_analysis_results(store, page_key)


def _render_saved_analysis_card(task, store, page_key, *, archived=False):
    label = (task.get("settings") or {}).get("label", "Analysis")
    status = str(task.get("status") or "incomplete")
    active = _lease_active(task)
    stop_requested = bool(task.get("stop_requested"))
    stopping = stop_requested and active
    display_status = "Stopping" if stopping else "Stopped" if stop_requested and status != "complete" else {
        "creating": "Preparing", "queued": "Queued", "running": "Running" if active else "Paused",
        "incomplete": "Paused", "complete": "Complete",
    }.get(status, status.capitalize())
    if archived:
        display_status = "Archived"
    status_color = {"Running": "blue", "Complete": "green", "Stopping": "orange",
                    "Paused": "orange", "Stopped": "gray"}.get(display_status, "gray")
    display = _task_display(task)
    apis = [api for api in display.get("apis") or [] if isinstance(api, dict)]
    with st.container(border=True, key=f"analysis_saved_task_{task['id']}", gap="small"):
        with st.container(gap=None, key=f"analysis_saved_info_{task['id']}"):
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(f"**{label}**", width="content")
                st.badge(display_status, color=status_color)
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                if apis:
                    api = apis[0]
                    provider = api.get("provider") or "Custom API"
                    model = api.get("model") or "Not recorded"
                    extra = f" · +{len(apis) - 1} in Task details" if len(apis) > 1 else ""
                    st.caption(f"API: {provider} / Model: {model}{extra}", width="content")
                else:
                    st.caption("API / Model: not recorded yet", width="content")
                st.caption(format_work_progress(task), width="content")
            st.markdown(_step_description(task, display))
            if active and status != "complete":
                st.caption("Finishing current requests…" if stopping else
                           "Active task · Resume is temporarily unavailable.")
        with st.container(horizontal=True, vertical_alignment="top", gap="small",
                          key=f"analysis_saved_actions_{task['id']}"):
            st.button(
                "View results" if status == "complete" else "Resume",
                key=f"_analysis_resume:{task['id']}", disabled=active and status != "complete", width="content",
                on_click=_queue_analysis_restore, args=(str(task["id"]),),
                help=("This task is active. After a server restart, Resume becomes available within 3 minutes."
                      if active and status != "complete" else "Restore the original inputs and view saved results."),
            )
            if not archived:
                # Keep the force-stop wrapper while avoiding a stretched
                # column that separates the two task actions.
                with st.container(width="content"):
                    from src.job_guard import render_force_stop_button
                    if render_force_stop_button(
                        label="Kill", key=f"_analysis_kill:{task['id']}",
                        on_click=lambda task_id=str(task["id"]): _kill_saved_task(store, task_id, task_completed=status == "complete"),
                        disabled=False, width="content",
                        help=("Archive this completed task; its final results remain in Archived results."
                          if status == "complete" else
                          "Terminate this task and remove it from Saved analyses. Its run record and saved results remain in the database."),
                    ):
                        st.rerun()
            with st.container(key=f"analysis_saved_details_{task['id']}", gap=None):
                with st.expander("Task details", expanded=False):
                    identity, timestamps = st.columns(2, gap="medium")
                    with identity:
                        st.caption(f"Task ID: {task['id']}")
                        st.caption(f"Page: {task.get('page_key') or page_key}")
                        step = display.get("current_step") or {}
                        if step.get("operation"):
                            st.caption(f"Operation: {step['operation']}")
                            if type(step.get("index")) is int:
                                st.caption(f"Saved step: {step['index'] + 1}")
                            phase = {"running": "Started", "complete": "Saved", "failed": "Failed; Resume can retry"}.get(step.get("phase"))
                            if phase:
                                st.caption(f"Checkpoint state: {phase}")
                    with timestamps:
                        for field, title in (("created_at", "Created"), ("started_at", "Started"), ("updated_at", "Last updated"), ("completed_at", "Completed"), ("archived_at", "Archived")):
                            value = task.get(field)
                            if value:
                                try:
                                    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                                    st.caption(f"{title}: {stamp.astimezone(timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
                                except (TypeError, ValueError):
                                    pass
                    for api in apis:
                        if len(apis) > 1:
                            st.caption(f"API: {api.get('provider') or 'Custom API'} / Model: {api.get('model') or 'Not recorded'}")
                        if api.get("endpoint"):
                            st.text(f"{api.get('provider') or 'Custom API'} endpoint: {api['endpoint']}")
                    st.caption("Progress tracks analysis work, such as runs or questions; API calls are saved separately.")
                    st.caption("Kill retains the task record and saved results in the database.")
                    if not task.get("display") and apis:
                        st.caption("This older task shows APIs from recent checkpoints and the latest saved step.")
                    if active and status != "complete":
                        st.caption(
                            "Stop requested. Resume is available after current requests finish."
                            if stopping else
                            "This task is active. After a server restart, Resume becomes available within 3 minutes."
                        )
        error = st.session_state.get(f"_analysis_kill_error:{task['id']}")
        if error:
            st.warning(error)


def _render_archived_analysis_results(store, page_key):
    with st.container(key="analysis_archived_list", gap=None):
        with st.expander("Archived results", expanded=False):
            if not st.checkbox("Show archived analyses", key=f"_analysis_show_archived:{page_key}"):
                return
            limit_key = f"_analysis_archived_list_limit:{page_key}"
            limit = int(st.session_state.get(limit_key, 50))
            try:
                tasks = store.list_tasks(page_key=page_key, statuses=["complete"], limit=limit,
                                         summary_only=True, exclude_killed=True, archived_only=True)
            except Exception as exc:
                st.warning(f"Archived results are unavailable. {classify_checkpoint_error(exc, operation='read')}")
                return
            has_more = len(tasks) >= limit
            tasks = [task for task in tasks if task.get("dynamic_items") and task.get("archived_at")
                     and task.get("status") == "complete" and not task.get("kill_requested")]
            if not tasks:
                st.caption("No archived results on this page.")
            for task in tasks:
                _render_saved_analysis_card(task, store, page_key, archived=True)
            if has_more and limit < 1000 and st.button("Show more", key=f"_analysis_archived_show_more:{page_key}"):
                st.session_state[limit_key] = min(1000, limit + 50)
                st.rerun()
