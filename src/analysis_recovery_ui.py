"""Restore durable analysis inputs before Streamlit creates their widgets."""
from __future__ import annotations

from datetime import datetime, timezone
import re
from urllib.parse import urlsplit
import streamlit as st

from src.analysis_checkpoints import OFFICIAL_GAME_PAGE_KEYS, classify_checkpoint_error
from src.resumable_analysis import (
    AnalysisCheckpointError, PENDING_RESTORE, RESUME_TASK,
    canonical_page_key, decode, get_cloud_store_for_current_user, is_safe_state_key, key_belongs_to_page, unpack_snapshot,
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
        "run_generation": ("generated_persuasion_mutations",),
        "run_probe_button": ("persuasion_probe_results",),
        "unlearn_rep_submit_run": ("unlearn_last_result",),
        "text_generate_report_narrative": ("text_analysis_results",),
    }.get(source["trigger_key"], ("text_analysis_results", "qa_evaluation_results", "sc_evaluation_results"))
    if source["trigger_key"] == "run_min_k_prob_analysis_button":
        result_keys = tuple(key for key in final if key.startswith("min_k_") and key.endswith(("_batch_results", "_last_result")))
    restore_results = task.get("status") == "complete" and isinstance(final, dict) and any(final.get(key) for key in result_keys)
    if source["trigger_key"] == "text_generate_report_narrative":
        results = decode(final.get("text_analysis_results"))
        restore_results = bool(restore_results and isinstance(results, dict) and results.get("report_narrative"))
    elif source["trigger_key"] == "unlearn_rep_submit_run":
        results = decode(final.get("unlearn_last_result"))
        restore_results = bool(restore_results and isinstance(results, dict)
                               and results.get("__representational_result__") == 1
                               and (results.get("inline_artifacts") or results.get("error")))
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
    if not restore_results:
        st.session_state[RESUME_TASK] = str(task["id"])
    st.session_state.pop("_analysis_resume_background_key", None)
    if source.get("background_key"):
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
        "action": "results" if restore_results else "resume", "status": "restored",
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
    notice = st.session_state.get(_RESTORE_FEEDBACK)
    if not isinstance(notice, dict) or notice.get("page_key") != page_key:
        return None
    slot = st.empty()
    if notice.get("status") in {"blocked", "error"}:
        slot.warning(notice["message"])
    elif notice.get("action") == "results":
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
        message = "The saved task could not finish resuming. Review the error below and retry Resume. Saved progress is preserved."
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
    notice = st.session_state.get(_RESTORE_FEEDBACK)
    if isinstance(notice, dict) and notice.get("task_id") == str(task_id):
        st.session_state.pop(_RESTORE_FEEDBACK, None)
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
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.caption("Resume reuses saved progress. Kill removes the task from this list.")
            if st.button("Refresh status", key=f"_analysis_refresh:{page_key}", width="content"):
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
            status_color = {"Running": "blue", "Complete": "green", "Stopping": "orange",
                            "Paused": "orange", "Stopped": "gray"}.get(display_status, "gray")
            display = _task_display(task)
            apis = [api for api in display.get("apis") or [] if isinstance(api, dict)]
            with st.container(border=True, key=f"analysis_saved_task_{task['id']}", gap="small"):
                with st.container(gap=None):
                    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                        st.markdown(f"**{label}**")
                        st.badge(display_status, color=status_color)
                    if apis:
                        api = apis[0]
                        provider = api.get("provider") or "Custom API"
                        model = api.get("model") or "Not recorded"
                        extra = f" · +{len(apis) - 1} in Task details" if len(apis) > 1 else ""
                        st.caption(f"API: {provider} / Model: {model}{extra}")
                    else:
                        st.caption("API / Model: not recorded yet")
                    st.caption(f"**{completed} saved checkpoints** · {pending} pending · {failed} failed")
                    st.markdown(_step_description(task, display))
                with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                    st.button(
                        "View results" if status == "complete" else "Resume",
                        key=f"_analysis_resume:{task['id']}", disabled=active, width="content",
                        on_click=_queue_analysis_restore, args=(str(task["id"]),),
                        help=("This task is active. After a server restart, Resume becomes available within 3 minutes."
                              if active else "Restore the original inputs and reuse saved results."),
                    )
                    # Keep the force-stop wrapper while avoiding a stretched
                    # column that separates the two task actions.
                    with st.container(width="content"):
                        from src.job_guard import render_force_stop_button
                        if render_force_stop_button(
                            label="Kill", key=f"_analysis_kill:{task['id']}",
                            on_click=lambda task_id=str(task["id"]): _kill_saved_task(store, task_id),
                            disabled=status == "complete", width="content",
                            help="Terminate this task and remove it from Saved analyses. Its run record and saved results remain in the database.",
                        ):
                            st.rerun()
                    if active:
                        st.caption("Finishing current requests…" if stopping else
                                   "Active task · Resume is temporarily unavailable.")
                error = st.session_state.get(f"_analysis_kill_error:{task['id']}")
                if error:
                    st.warning(error)
                with st.expander("Task details", expanded=False):
                    identity, timestamps = st.columns(2, gap="medium")
                    with identity:
                        st.caption(f"Task ID: {task['id']}")
                        st.caption(f"Page: {task.get('page_key') or page_key}")
                        step = display.get("current_step") or {}
                        if step.get("operation"):
                            st.caption(f"Operation: {step['operation']}")
                            phase = {"running": "Started", "complete": "Saved", "failed": "Failed; Resume can retry"}.get(step.get("phase"))
                            if phase:
                                st.caption(f"Checkpoint state: {phase}")
                    with timestamps:
                        for field, title in (("created_at", "Created"), ("started_at", "Started"), ("updated_at", "Last updated"), ("completed_at", "Completed")):
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
                    st.caption("Checkpoints are recorded as the task runs; their count is not the total number of planned steps.")
                    st.caption("Kill retains the task record and saved results in the database.")
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
