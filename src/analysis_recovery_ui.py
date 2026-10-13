"""Restore durable analysis inputs before Streamlit creates their widgets."""
from __future__ import annotations

from datetime import datetime, timezone
import re
from urllib.parse import urlsplit
import streamlit as st

from src.analysis_checkpoints import OFFICIAL_GAME_PAGE_KEYS, classify_checkpoint_error
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
    if task.get("kill_requested") or task.get("status") in {"stopping", "killed"}:
        raise AnalysisCheckpointError("This task was killed and cannot be resumed. Start a new analysis.", kind="killed")
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
    index = step.get("index")
    if type(index) is int:
        name = f"Checkpoint {index + 1}: {name}"
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


def _kill_saved_task(store, task_id):
    """Signal only this owner's task, retaining its fenced recovery checkpoints."""
    from src.run_cancellation import cancel_task
    cancel_task(store.owner_id, str(task_id))
    error_key = f"_analysis_kill_error:{task_id}"
    try:
        killed = store.kill_task(str(task_id))
        if killed.get("status") == "complete" and not killed.get("kill_requested"):
            # Completion won the database row lock; keep its result available.
            st.session_state.pop(error_key, None)
            return False
        if not killed.get("kill_requested") and killed.get("status") not in {"stopping", "killed"}:
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
    pending = st.session_state.get(PENDING_RESTORE)
    if isinstance(pending, dict) and pending.get("task_id") == str(task_id):
        st.session_state.pop(PENDING_RESTORE, None)
    return True


def render_analysis_recovery(page_key: str):
    if page_key == "Legal Cases Display":
        return
    try:
        store = get_cloud_store_for_current_user(page_key=page_key)
        if store is None:
            return
        limit_key = f"_analysis_task_list_limit:{page_key}"
        limit = int(st.session_state.get(limit_key, 50))
        tasks = store.list_tasks(
            page_key=page_key, limit=limit, summary_only=True, exclude_killed=True,
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
             and not task.get("kill_requested") and task.get("status") not in {"stopping", "killed"}]
    if not tasks:
        return
    with st.expander("Saved analyses", expanded=any(t.get("status") != "complete" for t in tasks)):
        st.caption("Resume restores the original settings and saved calls. Kill terminates and removes a task from this list; its database record is retained.")
        if st.button("Refresh status", key=f"_analysis_refresh:{page_key}"):
            st.rerun()
        for task in tasks:
            label = (task.get("settings") or {}).get("label", "Analysis")
            completed = int(task.get("completed_items") or 0)
            total = int(task.get("total_items") or 0)
            failed = int(task.get("failed_items") or 0)
            pending = max(0, total - completed - failed)
            status = str(task.get("status") or "incomplete")
            active = _lease_active(task)
            stop_requested = bool(task.get("stop_requested"))
            stopping = stop_requested and active
            display_status = "Stopping" if stopping else "Stopped" if stop_requested and status != "complete" else {
                "creating": "Preparing", "queued": "Queued", "running": "Running" if active else "Paused",
                "incomplete": "Paused", "complete": "Complete",
            }.get(status, status.capitalize())
            progress = f"{completed} saved checkpoints / {pending} pending / {failed} failed"
            display = _task_display(task)
            apis = display.get("apis") or []
            with st.container(border=True):
                columns = st.columns([4, 1, 1], gap="small", vertical_alignment="center")
                with columns[0]:
                    st.markdown(f"**{label}**")
                    st.caption(f"{display_status} / {progress}")
                    for api in apis:
                        if isinstance(api, dict):
                            provider = api.get("provider") or "Custom API"
                            model = api.get("model") or "Not recorded"
                            st.caption(f"API: {provider} / Model: {model}")
                    if not apis:
                        st.caption("API / Model: not recorded yet")
                    st.caption(_step_description(task, display))
                with columns[1]:
                    if st.button(
                        "View results" if status == "complete" else "Resume",
                        key=f"_analysis_resume:{task['id']}", disabled=active, width="stretch",
                        help="Restore the original inputs and reuse saved results.",
                    ):
                        st.session_state[PENDING_RESTORE] = {"task_id": str(task["id"])}
                        st.rerun()
                with columns[2]:
                    from src.job_guard import render_force_stop_button
                    if render_force_stop_button(
                        label="Kill", key=f"_analysis_kill:{task['id']}",
                        on_click=lambda task_id=str(task["id"]): _kill_saved_task(store, task_id),
                        disabled=status == "complete", width="stretch",
                        help="Terminate this task and remove it from Saved analyses. Its run record and saved results remain in the database.",
                    ):
                        st.rerun()
                error = st.session_state.get(f"_analysis_kill_error:{task['id']}")
                if error:
                    st.warning(error)
                with st.expander("Task details", expanded=False):
                    st.caption(f"Task ID: {task['id']}")
                    st.caption(f"Page: {task.get('page_key') or page_key}")
                    for field, title in (("created_at", "Created"), ("started_at", "Started"), ("updated_at", "Last updated"), ("completed_at", "Completed")):
                        value = task.get(field)
                        if value:
                            try:
                                stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                                st.caption(f"{title}: {stamp.astimezone(timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
                            except (TypeError, ValueError):
                                pass
                    step = display.get("current_step") or {}
                    if step.get("operation"):
                        st.caption(f"Operation: {step['operation']}")
                        phase = {"running": "Started", "complete": "Saved", "failed": "Failed; Resume can retry"}.get(step.get("phase"))
                        if phase:
                            st.caption(f"Checkpoint state: {phase}")
                    for api in apis:
                        if isinstance(api, dict) and api.get("endpoint"):
                            st.text(f"{api.get('provider') or 'Custom API'} endpoint: {api['endpoint']}")
                    st.caption("Checkpoints are recorded as the task runs; their count is not the total number of planned steps.")
                    if not task.get("display") and apis:
                        st.caption("This older task shows APIs from recent checkpoints and the latest saved step.")
                if active:
                    st.caption(
                        "Stop requested. Resume is available after current requests finish."
                        if stopping else
                        "This task is active. After a server restart, Resume becomes available within 3 minutes."
                    )
        if has_more and limit < 1000 and st.button("Show more", key=f"_analysis_show_older:{page_key}"):
            st.session_state[limit_key] = min(1000, limit + 50)
            st.rerun()
