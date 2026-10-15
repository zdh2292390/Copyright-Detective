"""Classify completed coordinators that still contain failed business work."""
import re


def is_systemic_api_failure(message):
    """Stop a batch on configuration/auth failures; transient items may continue."""
    if not isinstance(message, str):
        return False
    return bool(re.search(
        r"(?<![0-9])(?:401|403|404)(?![0-9])|invalid[ _-]*(?:api[ _-]*key|credentials?)|"
        r"unauthori[sz]ed|authentication[ _-]*(?:failed|error)|permission denied|"
        r"model[^\n]{0,80}(?:not[ _-]*found|does[ _-]*not[ _-]*exist|has been retired|no longer available)|"
        r"not a valid model(?: id| name)|"
        r"insufficient_quota|billing[ _-]*(?:limit|quota)", message, re.IGNORECASE,
    ))


def task_outcome_errors(outputs):
    errors = []
    # A coordinator returning normally is not sufficient evidence of success:
    # provider adapters can return structured errors for individual requests.
    if "qa_evaluation_results" in outputs and "qa_generated_qa_pairs" not in outputs:
        batches = outputs.get("qa_evaluation_results")
        if isinstance(batches, list):
            rows = [row for batch in batches if isinstance(batch, list) for row in batch]
            failed = sum(not isinstance(row, dict) or bool(row.get("error")) for row in rows)
            judge_failed = sum(isinstance(row, dict) and bool(row.get("llm_judge_error")) for row in rows)
            if not rows:
                errors.append("No knowledge evaluation responses were produced.")
            if failed:
                errors.append(f"{failed} knowledge evaluation response(s) failed. Partial results are preserved.")
            if judge_failed:
                errors.append(f"{judge_failed} judge evaluation(s) failed. Answer scores are preserved.")
    for key in ("qa_sleek_results", "sleek_evaluation_results"):
        value = outputs.get(key)
        if isinstance(value, dict):
            failed = value.get("failed_evaluations", 0)
            if isinstance(failed, int) and not isinstance(failed, bool) and failed > 0:
                errors.append(f"{failed} SLEEK evaluation(s) failed. Partial results are preserved.")
            elif value.get("successful_evaluations") == 0:
                errors.append("No SLEEK responses were successfully evaluated.")
    summary = outputs.get("persuasion_last_run_summary")
    if isinstance(summary, dict):
        failed = summary.get("failed", 0)
        if isinstance(failed, int) and not isinstance(failed, bool) and failed > 0:
            errors.append(f"{failed} planned persuasion response(s) failed. Partial results are preserved.")
    mutations = outputs.get("generated_persuasion_mutations")
    if isinstance(mutations, dict):
        judge_failed = sum(
            bool((row.get("judge_meta") or {}).get("primary_error")
                 or (row.get("judge_meta") or {}).get("secondary_error"))
            for rows in mutations.values() if isinstance(rows, list)
            for row in rows if isinstance(row, dict)
        )
        if judge_failed:
            errors.append(f"{judge_failed} persuasion judgement(s) failed. Scored responses are preserved.")
    return errors
