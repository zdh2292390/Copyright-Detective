"""Restore durable analysis inputs before Streamlit creates their widgets."""
from __future__ import annotations

from datetime import datetime, timezone
import re
from urllib.parse import urlsplit
import streamlit as st

from src.analysis_checkpoints import OFFICIAL_GAME_PAGE_KEYS, classify_checkpoint_error
from src.analysis_work_progress import format_work_progress
from src.history_ui import HISTORY_STYLES, PROCESS_MANAGER_TITLE, render_history_details, confirm_task_deletion
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
    "_analysis_", "_background_job_", "_page_job_", "detection_run_", "detection_job_",
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


def discard_pending_analysis_restore():
    """Abandon queued restore intent without changing submitted workers."""
    notice = st.session_state.get(_RESTORE_FEEDBACK)
    if isinstance(notice, dict) and isinstance(notice.get("trigger_key"), str):
        st.session_state.pop(f"detection_run_{notice['trigger_key']}", None)
    for key in (PENDING_RESTORE, RESUME_TASK, VIEW_TASK, VIEW_PAGE, _RESTORE_FEEDBACK,
                "_analysis_resume_background_key", "_analysis_restored_inputs", "_page_job_pending_view"):
        st.session_state.pop(key, None)


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
    discard_pending_analysis_restore()
    from src.page_jobs import detach_page_jobs
    detach_page_jobs(page_key)
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
    # Restoring one task must not clear another submitted action or change
    # the lifecycle of a worker that is still using its captured inputs.
    if not restore_results:
        st.session_state[f"detection_run_{trigger}"] = True
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
        message = ("Saved results could not be displayed. Retry View; saved calls remain unchanged."
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
            message = "The saved result view has not finished. Retry View; no new API calls will be made."
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
        return "Preparing analysis" if _lease_active(task) else "No saved progress yet"
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
        return "Finished: " + name
    if not _lease_active(task):
        return "Paused at: " + name if task.get("stop_requested") else "Paused at: " + name
    if task.get("stop_requested"):
        return "Stopping: " + name
    if step.get("phase") == "complete":
        return "Last completed: " + name
    if step.get("phase") == "failed":
        return "Retry needed: " + name
    return "Current step: " + name


def _task_action_store(store, page_key):
    # The shared list uses the authenticated read client. Official-game writes
    # retain their original server-side client and verified owner boundary.
    current_owner = str(st.session_state.get("user_id") or "")
    if not current_owner or current_owner != str(store.owner_id or ""):
        raise AnalysisCheckpointError("Your account changed. Reopen this task from Process Manager.", kind="auth")
    if page_key not in OFFICIAL_GAME_PAGE_KEYS:
        return store
    action_store = get_cloud_store_for_current_user(page_key=page_key)
    if action_store is None or str(action_store.owner_id) != str(store.owner_id):
        raise AnalysisCheckpointError("Sign in to this task's account before managing it.", kind="auth")
    return action_store


def _kill_saved_task(store, task_id, *, task_completed=False, page_key=None):
    """Cancel this task and delete its durable inputs, checkpoints and results."""
    from src.run_cancellation import cancel_task
    task_id = str(task_id)
    cancelled = False
    error_key = f"_analysis_kill_error:{task_id}"
    try:
        store = _task_action_store(store, page_key)
        cancelled = cancel_task(store.owner_id, task_id)
        deleted = store.kill_task(task_id)
        if not isinstance(deleted, dict) or deleted.get("deleted") is not True:
            raise AnalysisCheckpointError(
                "Apply supabase/analysis_history_delete.sql before deleting tasks.", kind="schema",
            )
    except Exception as exc:
        st.session_state[error_key] = (
            "Kill could not delete this task. Retry Kill. "
            + str(classify_checkpoint_error(exc))
        )
        return False
    from src.background_jobs import discard_background_task
    discard_background_task(store.owner_id, task_id)
    st.session_state.pop(error_key, None)
    if st.session_state.get(RESUME_TASK) == task_id:
        st.session_state.pop(RESUME_TASK, None)
        st.session_state.pop("_analysis_resume_background_key", None)
    if st.session_state.get(VIEW_TASK) == task_id:
        st.session_state.pop(VIEW_TASK, None)
        st.session_state.pop(VIEW_PAGE, None)
    if st.session_state.get("_analysis_current_task_id") == task_id:
        st.session_state.pop("_analysis_current_task_id", None)
        # Only the currently bound run may unlock this browser's running overlay.
        if cancelled and not task_completed:
            from src.job_guard import finish_detection_job
            finish_detection_job()
    pending = st.session_state.get(PENDING_RESTORE)
    if isinstance(pending, dict) and pending.get("task_id") == task_id:
        st.session_state.pop(PENDING_RESTORE, None)
    notice = st.session_state.get(_RESTORE_FEEDBACK)
    if isinstance(notice, dict) and notice.get("task_id") == task_id:
        st.session_state.pop(_RESTORE_FEEDBACK, None)
    return True



def _stop_saved_task(store, task_id, *, page_key=None):
    """Pause a task while retaining its inputs and successful responses."""
    from src.run_cancellation import cancel_task
    task_id = str(task_id)
    cancelled = False
    error_key = f"_analysis_stop_error:{task_id}"
    try:
        store = _task_action_store(store, page_key)
        cancelled = cancel_task(store.owner_id, task_id)
        task = store.request_stop(task_id)
        if not task.get("stop_requested") and task.get("status") != "complete":
            raise AnalysisCheckpointError("The task did not acknowledge Stop. Retry Stop.")
    except Exception as exc:
        st.session_state[error_key] = str(classify_checkpoint_error(exc))
        return False
    st.session_state.pop(error_key, None)
    if cancelled and st.session_state.get("_analysis_current_task_id") == task_id:
        from src.job_guard import finish_detection_job
        finish_detection_job()
    return True


def render_analysis_recovery(page_key: str):
    # Tasks belong to the submitting account/session, independently of whichever
    # page and inputs the user is editing now.
    from src.background_jobs import list_background_jobs

    local_jobs = list_background_jobs()
    store = None
    tasks = []
    has_more = False
    limit_key = "_analysis_task_list_limit"
    limit = int(st.session_state.get(limit_key, 50))
    try:
        store = get_cloud_store_for_current_user()
        if store is not None:
            tasks = store.list_tasks(
                limit=limit, summary_only=True, exclude_killed=True,
                statuses=["creating", "queued", "running", "incomplete", "complete"],
            )
            has_more = len(tasks) >= limit
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

    tasks = [task for task in tasks if task.get("dynamic_items")
             and page_supports_analysis_recovery(task.get("page_key"))
             and not task.get("kill_requested") and task.get("status") not in {"stopping", "killed"}]
    saved_ids = {str(task["id"]) for task in tasks}
    local_results = {}
    for job in local_jobs:  # The registry returns newest submissions first.
        if (job.get("task_id") and job.get("source_type") == "page_job" and job.get("has_result")
                and job.get("status") in {"completed", "failed"}):
            local_results.setdefault(str(job["task_id"]), job)
    local_jobs = [job for job in local_jobs
                  if job.get("page_key") and str(job.get("task_id") or "") not in saved_ids]
    if store is None and not local_jobs:
        return
    active_local = any(job.get("status") in {"queued", "running", "stopping"} for job in local_jobs)
    st.html(HISTORY_STYLES)
    with st.container(key="analysis_history_list", gap=None):
        with st.expander(PROCESS_MANAGER_TITLE,
                         expanded=active_local or any(t.get("status") != "complete" for t in tasks)):
            with st.container(horizontal=True, vertical_alignment="center", gap="small",
                              key="api_history_toolbar"):
                st.caption("Tasks keep their submitted settings. You can change inputs, switch pages, or submit another task.")
                if st.button("Refresh", key="_analysis_refresh", width="content"):
                    st.rerun()
            if not tasks and not local_jobs:
                st.caption("No submitted tasks yet.")
            for job in local_jobs:
                _render_local_analysis_card(job)
            for task in tasks:
                _render_saved_analysis_card(task, store, task.get("page_key") or page_key,
                                            local_result=local_results.get(str(task["id"])))
            if has_more and limit < 1000 and st.button("Show more", key="_analysis_show_older"):
                st.session_state[limit_key] = min(1000, limit + 50)
                st.rerun()


def _render_local_analysis_card(job):
    from src.background_jobs import force_stop_background_job
    from src.page_jobs import view_page_job

    key = str(job["key"])
    status = str(job.get("status") or "queued")
    active = status in {"queued", "running", "stopping"}
    stopping = status == "stopping"
    display_status = {"queued": "Queued", "running": "Running", "stopping": "Stopping",
                      "completed": "Complete", "failed": "Failed", "stopped": "Paused"}.get(status, status.capitalize())
    color = {"Running": "blue", "Complete": "green", "Stopping": "orange",
             "Paused": "orange", "Failed": "red"}.get(display_status, "gray")
    with st.container(border=True, key=f"history_task_local_{key}", gap="small"):
        title, actions = st.columns([5, 3], gap="small", vertical_alignment="center")
        with title:
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(f"**{job.get('label') or 'Analysis'}**", width="content")
                st.badge(display_status, color=color)
        with actions:
            with st.container(horizontal=True, vertical_alignment="center", gap="small",
                              key=f"history_actions_local_{key}", horizontal_alignment="right"):
                if active:
                    st.button(
                        "Stopping..." if stopping else "Stop", key=f"_analysis_local_stop:{key}",
                        disabled=stopping, width="content",
                        on_click=force_stop_background_job, args=(key,),
                        help="Stop this task without affecting other tasks.",
                    )
                if (status in {"completed", "failed"} and job.get("source_type") == "page_job"
                        and job.get("has_result")):
                    # Navigation/result state must change in a callback before
                    # Streamlit recreates the page and sidebar widgets.
                    st.button(
                        "View", key=f"_analysis_local_view:{key}", width="content", type="primary",
                        on_click=view_page_job, args=(key,),
                        help="Open the saved output of this task.",
                    )
        st.caption(f"Page: {job.get('page_key')}")
        current, total = job.get("current", 0), job.get("total", 1)
        st.caption(f"Progress: {current} / {total}")
        if job.get("message"):
            st.caption(str(job["message"]))
        if active:
            st.caption("Running independently. Inputs and other tasks remain available.")
        if job.get("error"):
            st.warning(str(job["error"]))


def _render_saved_analysis_card(task, store, page_key, *, local_result=None):
    label = (task.get("settings") or {}).get("label", "Analysis")
    status = str(task.get("status") or "incomplete")
    active = _lease_active(task)
    stop_requested = bool(task.get("stop_requested"))
    stopping = stop_requested and active and status != "complete"
    display_status = "Stopping" if stopping else "Paused" if stop_requested and status != "complete" else {
        "creating": "Preparing", "queued": "Queued", "running": "Running" if active else "Paused",
        "incomplete": "Paused", "complete": "Complete",
    }.get(status, status.capitalize())
    status_color = {"Running": "blue", "Complete": "green", "Stopping": "orange",
                    "Paused": "orange"}.get(display_status, "gray")
    display = _task_display(task)
    apis = [api for api in display.get("apis") or [] if isinstance(api, dict)]
    with st.container(border=True, key=f"history_task_{task['id']}", gap="small"):
        title, actions = st.columns([5, 3], gap="small", vertical_alignment="center")
        with title:
            with st.container(horizontal=True, vertical_alignment="center", gap="small",
                              key=f"history_header_{task['id']}"):
                st.markdown(f"**{label}**", width="content")
                st.badge(display_status, color=status_color)
                if active and status != "complete":
                    from src.job_guard import render_force_stop_button
                    render_force_stop_button(
                        key=f"_analysis_stop:{task['id']}", label="Stopping…" if stopping else "Stop",
                        disabled=stopping, width="content",
                        on_click=lambda task_id=str(task['id']), task_page=page_key:
                            _stop_saved_task(store, task_id, page_key=task_page),
                        help="Pause this task and keep saved progress for Resume.",
                    )
        with actions:
            with st.container(horizontal=True, vertical_alignment="center", gap="small",
                              key=f"history_actions_{task['id']}", horizontal_alignment="right"):
                st.button(
                    "View" if status == "complete" else "Resume",
                    key=f"_analysis_resume:{task['id']}", disabled=active and status != "complete", width="content", type="primary",
                    on_click=_queue_analysis_restore, args=(str(task["id"]),),
                    help=("Wait for this active task to stop before resuming."
                          if active and status != "complete" else
                          "View the saved final results without new API calls." if status == "complete" else
                          "Continue this task with its original settings and reuse completed calls."),
                )
                if status != "complete" and local_result is not None:
                    from src.page_jobs import view_page_job
                    local_key = str(local_result["key"])
                    st.button(
                        "View", key=f"_analysis_local_view:{local_key}", width="content",
                        on_click=view_page_job, args=(local_key,),
                        help="View the saved partial results without restarting this task or making API calls.",
                    )
                with st.container(width="content"):
                    from src.job_guard import render_force_stop_button
                    if render_force_stop_button(
                        label="Kill", key=f"_analysis_kill:{task['id']}",
                        disabled=False, width="content",
                        help="Stop this task and permanently delete its inputs, checkpoints and results.",
                    ):
                        confirm_task_deletion(
                            task_id=str(task['id']), owner_id=store.owner_id, label=label,
                            on_confirm=lambda task_id=str(task['id']), complete=status == "complete", task_page=page_key:
                                _kill_saved_task(store, task_id, task_completed=complete, page_key=task_page),
                        )
        st.caption(f"Page: {task.get('page_key') or page_key}")
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            if apis:
                api = apis[0]
                provider, model = api.get("provider") or "Custom API", api.get("model") or "Not recorded"
                extra = f" · +{len(apis) - 1} APIs" if len(apis) > 1 else ""
                st.caption(f"API: {provider} · Model: {model}{extra}", width="content")
            else:
                st.caption("API / Model: not recorded yet", width="content")
            st.caption(format_work_progress(task), width="content")
        st.caption(_step_description(task, display))
        if active and status != "complete":
            st.caption("Finishing current requests…" if stopping else
                       "Running independently. You can change inputs, switch pages, or submit another task.")
        with st.container(key=f"history_details_{task['id']}", gap=None):
            with st.expander("Details", expanded=False):
                render_history_details(task, page_key=page_key)
                if (display.get("current_step") or {}).get("phase") == "failed" and status != "complete":
                    st.caption("The last call failed. Resume retries it and keeps completed progress.")
                if len(apis) > 1:
                    for api in apis:
                        st.caption(f"{api.get('provider') or 'Custom API'} · {api.get('model') or 'Not recorded'}")
                if active and status != "complete":
                    st.caption("After a server restart, Resume becomes available when the lease expires (within 3 minutes).")
        error = st.session_state.get(f"_analysis_kill_error:{task['id']}")
        if error:
            st.warning(error)
        stop_error = st.session_state.get(f"_analysis_stop_error:{task['id']}")
        if stop_error:
            st.warning(stop_error)
        if status != "complete" and local_result and local_result.get("error"):
            st.warning(str(local_result["error"]))
