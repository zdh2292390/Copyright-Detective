"""Submit isolated API tasks without binding their lifetime to page widgets."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from uuid import uuid4

import streamlit as st

from src.background_jobs import get_background_job, render_background_job_status, submit_background_job
from src.resumable_analysis import canonical_page_key, is_safe_state_key, register_ui_run

_SELECTIONS = "_page_job_selections"
_DELIVERED = "_page_job_delivered"
_PENDING_VIEW = "_page_job_pending_view"
_BASELINES = "_page_job_baselines"
_SECRET_FIELD_PARTS = (
    "api_key", "apikey", "access_token", "refresh_token", "password", "secret",
    "authorization", "service_role", "agent_key", "fernet", "credential",
)
_EXTRA_RESULT_KEYS = frozenset({
    "text_pdf_report", "text_pdf_report_fingerprint", "text_pdf_report_error",
    "qa_pdf_report", "qa_pdf_report_bytes", "qa_sleek_pdf_report",
    "sc_pdf_report", "sc_pdf_report_bytes", "sc_pdf_report_fingerprint", "sc_pdf_report_error",
    "jailbreak_pdf_report_bytes", "jailbreak_boxplot_png_bytes", "jailbreak_histogram_png_bytes",
    "jailbreak_distribution_legend_note", "_text_memorization_api_failure", "last_prompt",
    "results_prompt_selector", "stage2_results",
})


def _validate_output_fields(result):
    if not isinstance(result, dict):
        raise ValueError("A page task must return result fields.")
    for key in result:
        if (not isinstance(key, str) or any(part in key.lower() for part in _SECRET_FIELD_PARTS)
                or not (is_safe_state_key(key) or key in _EXTRA_RESULT_KEYS)
                or key.startswith("sidebar_") or key in {"main_navigation", "detection_navigation", "game_navigation"}):
            raise ValueError("A background task returned an invalid result field.")


def _capture_error_secrets():
    values = set()
    visited = set()
    def collect(mapping, *, secret_container=False):
        if id(mapping) in visited:
            return
        visited.add(id(mapping))
        for key, value in mapping.items():
            secret_field = isinstance(key, str) and any(part in key.lower() for part in _SECRET_FIELD_PARTS)
            if isinstance(value, dict) and (secret_container or secret_field):
                collect(value, secret_container=True)
            elif isinstance(value, str) and value and secret_field:
                values.add(value)
    collect({key: st.session_state[key] for key in list(st.session_state)})
    return tuple(sorted(values, key=len, reverse=True))


def _redact_error(message, secrets):
    text = str(message)
    for secret in secrets:
        text = text.replace(secret, "[redacted]")
    return text


class PageJobResult(dict):
    def __init__(self, outputs=None, *, complete=True, error=None):
        super().__init__(outputs or {})
        self.complete = bool(complete)
        self.error = error


class _TaskState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def freeze_task_state():
    """Copy page inputs on the UI thread; never copy credentials or UI handles."""
    frozen = {}
    for key in list(st.session_state):
        if not isinstance(key, str) or not (is_safe_state_key(key) or key == "_analysis_resume_task"):
            continue
        try:
            frozen[key] = deepcopy(st.session_state[key])
        except (TypeError, ValueError):
            continue
    return frozen


class TaskUI:
    """UI-free progress adapter for an extracted business execution block.

    Each runner owns this state. No Streamlit call or live browser state is
    accessed on the worker, including when an existing block reports an error.
    """
    def __init__(self, state, report):
        self.session_state = _TaskState(deepcopy(state))
        self.errors = []
        self._report = report

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def progress(self, value=0, text=None, **kwargs):
        fraction = float(value) / 100 if isinstance(value, int) and value > 1 else float(value)
        self._report(round(max(0.0, min(1.0, fraction)) * 1000), 1000, str(text or "Running"))
        return self

    def error(self, message, **kwargs):
        self.errors.append(str(message))
        return self

    def columns(self, spec, **kwargs):
        return [self] * (spec if isinstance(spec, int) else len(spec))

    def stop(self):
        raise ValueError(self.errors[-1] if self.errors else "The submitted task could not continue.")

    def result(self, outputs, *, complete=None):
        return PageJobResult(outputs, complete=not self.errors if complete is None else complete,
                             error="; ".join(self.errors) or None)

    def __getattr__(self, name):
        if name in {"empty", "spinner", "status", "container", "expander", "text", "write", "markdown",
                    "caption", "info", "warning", "success", "code", "metric", "dataframe", "table",
                    "divider", "update", "rerun", "html", "plotly_chart", "pyplot"}:
            return lambda *args, **kwargs: self
        raise AttributeError(name)


def _page_key(page_key=None):
    from src.resumable_analysis import _CURRENT, PageRun
    scope = _CURRENT.get()
    if page_key is None and isinstance(scope, PageRun):
        page_key = scope.page_key
    return canonical_page_key(page_key or st.session_state.get("main_navigation"))


def _value_stamp(value):
    from src.resumable_analysis import encode
    try:
        serialized = json.dumps(encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(serialized.encode("utf-8")).hexdigest()
    except (TypeError, ValueError):
        # Unsupported mutable outputs require an explicit View action.
        return None


def _capture_baseline():
    return {key: _value_stamp(st.session_state[key]) for key in list(st.session_state)
            if isinstance(key, str) and (is_safe_state_key(key) or key == "_text_memorization_api_failure")}


def submit_page_job(trigger_key, label, runner):
    """Freeze task identity; each click creates an independent process record."""
    page = _page_key()
    if not page:
        raise ValueError("Select an analysis page before submitting a task.")
    register_ui_run(label, trigger_key)
    baseline = _capture_baseline()
    error_secrets = _capture_error_secrets()
    key = "analysis:" + uuid4().hex
    connection = {
        "base_url": str(st.session_state.get("sidebar_local_vllm_base_url") or "http://localhost:8000/v1"),
        "api_key": str(st.session_state.get("sidebar_local_vllm_api_key") or ""),
    }
    def execute(report):
        from src.api_task_context import api_task_scope
        from src.run_cancellation import AnalysisCancelled
        try:
            with api_task_scope(connection):
                result = runner(report)
                _validate_output_fields(result)
                if isinstance(result, PageJobResult) and isinstance(result.error, str):
                    result.error = _redact_error(result.error, error_secrets)
                return result
        except AnalysisCancelled:
            raise
        except BaseException as exc:
            detail = str(exc).strip() or type(exc).__name__
            raise RuntimeError(_redact_error(detail, error_secrets)) from None
    accepted = submit_background_job(key, label, execute,
        metadata={"page_key": page, "source_type": "page_job", "trigger_key": trigger_key})
    if accepted:
        selections = dict(st.session_state.get(_SELECTIONS) or {})
        selections[page] = key
        st.session_state[_SELECTIONS] = selections
        baselines = dict(st.session_state.get(_BASELINES) or {})
        baselines = {identity: value for identity, value in baselines.items() if identity in selections.values()}
        baselines[key] = baseline
        st.session_state[_BASELINES] = baselines
    return accepted


def detach_page_jobs(page_key=None):
    """Clear this page's result selection without stopping submitted work."""
    selections = dict(st.session_state.get(_SELECTIONS) or {})
    previous = selections.pop(_page_key(page_key), None)
    st.session_state[_SELECTIONS] = selections
    baselines = dict(st.session_state.get(_BASELINES) or {})
    baselines.pop(previous, None)
    st.session_state[_BASELINES] = baselines


def _apply_outputs(record, *, protect_draft=False):
    result = record.get("result")
    if not isinstance(result, dict):
        return False
    # Validate every key before changing any UI state, including extra binary
    # report fields that are intentionally excluded from cloud snapshots.
    _validate_output_fields(result)
    if protect_draft:
        baseline = (st.session_state.get(_BASELINES) or {}).get(record["key"])
        if baseline is None:
            return False
        empty_stamp = _value_stamp(None)
        for key in result:
            if any(marker in key for marker in ("pdf_report", "fingerprint", "jailbreak_")):
                continue  # Derived report caches may be populated while a task runs.
            expected = baseline.get(key, empty_stamp)
            current = _value_stamp(st.session_state.get(key))
            if expected is None or current is None or expected != current:
                return False
    copied = deepcopy(result)
    for key, value in copied.items():
        st.session_state[key] = value
    return True


def view_page_job(key):
    record = get_background_job(key, include_result=False)
    if (not record or record.get("source_type") != "page_job"
            or record.get("status") not in {"completed", "failed"} or not record.get("has_result")):
        return False
    st.session_state[_PENDING_VIEW] = key
    return True


def apply_pending_page_job_view():
    key = st.session_state.pop(_PENDING_VIEW, None)
    if not key:
        return
    record = get_background_job(key)
    if (not record or record.get("source_type") != "page_job"
            or record.get("status") not in {"completed", "failed"} or not isinstance(record.get("result"), dict)):
        return
    page = canonical_page_key(record.get("page_key"))
    if not page:
        return
    _apply_outputs(record)
    # Selecting a result must also discard the old restore's unconsumed Run
    # trigger; leaving it armed could start a new paid task after View.
    from src.analysis_recovery_ui import discard_pending_analysis_restore, _restore_execution_mode
    discard_pending_analysis_restore()
    st.session_state.pop("_analysis_current_task_id", None)
    selections = dict(st.session_state.get(_SELECTIONS) or {})
    selections[page] = key
    st.session_state[_SELECTIONS] = selections
    _restore_execution_mode(page, record.get("trigger_key"), owner_id=record.get("owner_id"))
    if record.get("trigger_key") == "run_min_k_prob_analysis_button":
        modes = {"min_k_user_input_": "User Input", "min_k_upload_": "Upload Document",
                 "min_k_predefined_": "Predefined Examples"}
        result = record.get("result") or {}
        for prefix, mode in modes.items():
            if any(isinstance(name, str) and name.startswith(prefix) for name in result):
                st.session_state["min_k_input_mode"] = mode
                break
    delivered = dict(st.session_state.get(_DELIVERED) or {})
    delivered[key] = record.get("finished_at")
    st.session_state[_DELIVERED] = delivered


def render_page_jobs(page_key=None):
    """Publish only the selected task's outputs before page widgets render."""
    page = _page_key(page_key)
    key = (st.session_state.get(_SELECTIONS) or {}).get(page)
    if not key:
        return
    summary = get_background_job(key, include_result=False)
    if not summary:
        detach_page_jobs(page)
        return
    if summary.get("status") in {"completed", "failed"}:
        delivered = dict(st.session_state.get(_DELIVERED) or {})
        stamp = summary.get("finished_at")
        if delivered.get(key) != stamp:
            # Once delivered, ordinary parameter edits need only status metadata;
            # avoid copying a potentially large result on every Streamlit rerun.
            record = get_background_job(key) if summary.get("has_result") else summary
            if not record:
                detach_page_jobs(page)
                return
            applied = _apply_outputs(record, protect_draft=True)
            delivered[key] = stamp
            st.session_state[_DELIVERED] = delivered
            baselines = dict(st.session_state.get(_BASELINES) or {})
            baselines.pop(key, None)
            st.session_state[_BASELINES] = baselines
            if not applied and summary.get("has_result"):
                st.info("Your current work has changed. Open this task's saved results in Process Manager.")
    render_background_job_status(key)
