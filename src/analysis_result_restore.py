"""Read-only final-result restoration; never submit a new analysis or score."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from src.analysis_checkpoints import AnalysisCheckpointError, OFFICIAL_GAME_PAGE_KEYS
from src.resumable_analysis import (
    CallJournal, canonical_page_key, decode, encode, is_safe_state_key,
    key_belongs_to_page, unpack_snapshot,
)

SCALING_RESULT_KEY = "copyright_game2:saved_scaling_result"
_RESULT_KEYS = {
    "run_snippet_analysis_button": "text_analysis_results",
    "generate_qa_button": "qa_generated_qa_pairs",
    "run_knowledge_eval_button": "qa_evaluation_results",
    "run_sleek_eval_button": "qa_sleek_results",
    "run_sleek_button": "sleek_evaluation_results",
    "sc_generate_mcq_button": "sc_generated_mcqs",
    "sc_run_eval_button": "sc_evaluation_results",
    "qa_run_knowmem_eval": "qa_knowmem_saved_entries",
    "run_generation": "generated_persuasion_mutations",
    "run_probe_button": "persuasion_probe_results",
    "text_generate_report_narrative": "text_analysis_results",
}


def encode_background_result(result: Any, task: dict) -> dict | None:
    """Serialize only this known ordinary game's report, with immutable binding."""
    if canonical_page_key(task.get("page_key")) != "Game 2: The Cross-Model Scaling Quest":
        return None
    from src.game_continuation import ScalingBatch
    if not isinstance(result, ScalingBatch):
        raise AnalysisCheckpointError("The completed scaling result has an invalid type.")
    return encode({"version": 1, "type": "scaling_batch", "task_id": str(task["id"]),
                   "owner_id": str(task["owner_id"]), "batch": asdict(result)})


def decode_scaling_result(value: Any, *, task_id: str, owner_id: str):
    """Use fixed dataclasses only; checkpoint data cannot import executable types."""
    from src.game_continuation import (
        ContinuationAttempt, ScalingBatch, OPENAI_PROVIDER, KIMI_PROVIDER,
        DIRECT_PROBE_METHODS, MAX_RUNS_PER_PROVIDER,
    )
    import math
    try:
        value = decode(value)
        if not isinstance(value, dict) or value.get("version") != 1 or value.get("type") != "scaling_batch":
            raise ValueError
        if value.get("task_id") != str(task_id) or value.get("owner_id") != str(owner_id):
            raise ValueError
        payload = value["batch"]
        if not isinstance(payload, dict) or payload.get("provider") not in {OPENAI_PROVIDER, KIMI_PROVIDER}:
            raise ValueError
        if payload.get("prompt_method") not in DIRECT_PROBE_METHODS or payload.get("prompt_mode") != "Direct Probing":
            raise ValueError
        if not all(isinstance(payload.get(k), str) and payload[k].strip() for k in ("model", "prompt")):
            raise ValueError
        count = payload["requested_runs"]
        if type(count) is not int or not 1 <= count <= MAX_RUNS_PER_PROVIDER:
            raise ValueError
        for key in ("temperature", "top_p"):
            if type(payload[key]) not in (int, float) or not math.isfinite(payload[key]):
                raise ValueError
        if not 0 <= payload["temperature"] <= 2 or not 0 < payload["top_p"] <= 1:
            raise ValueError
        attempts = payload["attempts"]
        if not isinstance(attempts, (list, tuple)) or not 1 <= len(attempts) <= count:
            raise ValueError
        seen = set()
        rows = []
        for item in attempts:
            if not isinstance(item, dict) or type(item.get("run")) is not int or not 1 <= item["run"] <= count or item["run"] in seen:
                raise ValueError
            seen.add(item["run"])
            if not isinstance(item.get("response"), str) or not item["response"].strip():
                raise ValueError
            metrics = item.get("metrics")
            if not isinstance(metrics, dict) or "rouge_l" not in metrics or any(
                not isinstance(k, str) or type(v) not in (int, float) or not math.isfinite(v) for k, v in metrics.items()
            ):
                raise ValueError
            rows.append(ContinuationAttempt(item["run"], item["response"], dict(metrics)))
        errors = payload.get("errors", [])
        if not isinstance(errors, (list, tuple)) or not all(isinstance(error, str) for error in errors):
            raise ValueError
        return ScalingBatch(payload["provider"], payload["model"], payload["prompt_method"],
                            payload["prompt_mode"], payload["prompt"], payload["temperature"],
                            payload["top_p"], count, tuple(rows), tuple(errors))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AnalysisCheckpointError("The saved scaling result is invalid or belongs to another task/account.") from exc


def _rebuild_scaling_result(task: dict, store) -> dict:
    """Older batches lacked a final report; reuse exact saved calls and local scores."""
    from src.game_continuation import (
        ContinuationAttempt, ScalingBatch, DIRECT_PROBE_METHODS, GROUND_TRUTH, _response_text, _similarity_metrics,
    )
    if store is None or str(getattr(store, "owner_id", "")) != str(task.get("owner_id") or ""):
        raise AnalysisCheckpointError("Sign in to the saved task's account before viewing results.", kind="auth")
    source = task.get("source") or {}
    initial = decode(unpack_snapshot(source.get("initial_session") or {}))
    journal = CallJournal(store, task, heartbeat=False)
    try:
        items = journal.items
        if not items or any(row.get("status") != "complete" for row in items):
            raise AnalysisCheckpointError("The saved scaling task has unfinished calls. No API call was started.")
        request = items[0]["payload"].get("request") or {}
        provider = request.get("provider")
        slug = {"OpenAI": "openai", "Kimi": "kimi"}.get(provider)
        if not slug:
            raise AnalysisCheckpointError("The saved scaling provider is invalid.")
        method = initial.get("copyright_game2:prompt_method") or initial.get("copyright_game2:continuation_strategy")
        # Match the stored prompt, so historical template changes do not alter a report.
        if method not in DIRECT_PROBE_METHODS:
            from src.game_continuation import build_challenge_prompt
            method = next((name for name in DIRECT_PROBE_METHODS if build_challenge_prompt(name) == request.get("prompt")), None)
        if method is None:
            raise AnalysisCheckpointError("The saved scaling prompt cannot be identified. No API call was started.")
        runs = initial.get(f"copyright_game2:{slug}_runs", len(items))
        if type(runs) is not int or runs != len(items):
            raise AnalysisCheckpointError("The saved scaling task does not contain all planned responses.")
        attempts = []
        for index, row in enumerate(items, 1):
            payload = row["payload"]
            if payload.get("operation") != "llm.completion" or payload.get("request") != request:
                raise AnalysisCheckpointError("The saved scaling requests do not belong to one batch.")
            result = journal.call("llm.completion", request,
                                  lambda: (_ for _ in ()).throw(AssertionError("Read-only view cannot call APIs")))
            response = _response_text(result)
            attempts.append(ContinuationAttempt(index, response, _similarity_metrics(GROUND_TRUTH, response)))
        result = ScalingBatch(provider, request["model_name"], method, "Direct Probing",
                              request["prompt"], request["temperature"], request["top_p"], runs,
                              tuple(attempts))
        return encode_background_result(result, task)
    except AnalysisCheckpointError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, RuntimeError) as exc:
        raise AnalysisCheckpointError("The saved scaling report could not be reconstructed from its original responses. No API call was started.") from exc
    finally:
        journal.finish(False)


def completed_result_snapshot(task: dict, *, store=None) -> dict | None:
    """Return a safe encoded page snapshot, or None for strictly cached UI replay.

    The caller has fetched the task through its verified owner-scoped store.
    This helper performs reads/local report reconstruction only, including after archival.
    """
    if task.get("status") != "complete":
        return None
    page = canonical_page_key(task.get("page_key"))
    source = task.get("source") or {}
    if task.get("page_key") in OFFICIAL_GAME_PAGE_KEYS:
        if task.get("server_managed") is not True:
            raise AnalysisCheckpointError("This official game's saved task has no trusted server origin.")
        # Official reports live in competition history. Never replay a write-producing worker.
        return unpack_snapshot(source.get("initial_session") or {})
    if page == "Game 2: The Cross-Model Scaling Quest":
        value = (task.get("metadata") or {}).get("background_result")
        if value is None:
            value = _rebuild_scaling_result(task, store)
        decode_scaling_result(value, task_id=str(task["id"]), owner_id=str(task["owner_id"]))
        return {SCALING_RESULT_KEY: deepcopy(value)}
    trigger = source.get("trigger_key")
    result_key = _RESULT_KEYS.get(trigger)
    if result_key is None:
        return None
    final = unpack_snapshot((task.get("metadata") or {}).get("final_session") or {})
    if result_key not in final:
        return None
    result = decode(final[result_key])
    if not isinstance(result, (dict, list)):
        return None
    if trigger == "text_generate_report_narrative" and not (
        isinstance(result, dict) and isinstance(result.get("report_narrative"), str) and result["report_narrative"].strip()
    ):
        return None
    # Keep the final output restricted to its original page; auth and action state never restores.
    return {key: deepcopy(value) for key, value in final.items()
            if isinstance(key, str) and is_safe_state_key(key) and key_belongs_to_page(key, page)}
