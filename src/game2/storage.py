"""Supabase persistence scoped to Copyright Challenge 3."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence


from src.game.storage import (
    GameIdentityError,
    GameStageOneRequired,
    GameStorageError,
    GameSubmissionLocked,
    VerifiedParticipant,
    _admin_client,
    _format_database_error as _base_format_database_error,
    _hash_text,
    _jeju_day_bounds,
    _now,
    _row,
    _rows,
    _safe_profile_text,
    _upsert_profile,
    backend_configured,
    format_jeju_leaderboard_window,
    format_finished_at_jeju,
    get_jeju_today,
    get_shared_api_key,
    missing_configuration,
    preload_background_job_secrets,
    shared_api_key_configured,
    verify_participant,
)

from .constants import (
    BENCHMARK_VERSION,
    BOOK_KEYS,
    COMPETITION_SLUG,
    GAME_MODEL,
    GAME_PROVIDER,
    GENERATION_LIMIT_EXCLUSIVE,
    LEADERBOARD_VIEW,
    LEGACY_LEADERBOARD_VIEW,
    MAX_SCORED_ANSWERS,
    MAX_TEMPERATURE,
    PROBE_MODES,
    QUESTIONS_PER_BOOK,
    max_runs_for_mode,
)
from .engine import GameConfig, KnowledgeStageOneResult, StageTwoResult


_KNOWLEDGE_BOOK_KEYS = frozenset(BOOK_KEYS)


def _format_database_error(exc: Exception) -> str:
    """Translate legacy database wording after the game moved to Challenge 3."""
    return (
        _base_format_database_error(exc)
        .replace("Challenge 2", "Challenge 3")
        .replace("Game 2", "Game 3")
    )


def _stage_one_aliases(source: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not source:
        return None
    row = dict(source)
    mean_f1 = float(
        row.get("direct_avg_rouge_l", row.get("direct_rouge_l", 0.0)) or 0.0
    )
    row["mean_f1"] = mean_f1
    row["question_count"] = int(row.get("direct_attempts", 0) or 0)
    row["score_metric"] = "token_f1"
    return row


def _score_aliases(source: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(source)
    row["max_score"] = float(row.get("max_score", row.get("max_rouge_l", 0.0)) or 0.0)
    row["avg_score"] = float(row.get("avg_score", row.get("avg_rouge_l", 0.0)) or 0.0)
    row["peak_f1"] = row["max_score"]
    row["mean_f1"] = row["avg_score"]
    row["score_metric"] = "token_f1"
    return row


def _validate_game2_config(config: GameConfig) -> None:
    config.validate(PROBE_MODES)
    if len(config.selected_book_keys) != 1:
        raise GameStorageError("Game 3 evaluates exactly one selected book per run.")
    if config.book_key not in _KNOWLEDGE_BOOK_KEYS:
        raise GameStorageError("Select one of the five knowledge-memorization books.")
    if config.attempts_per_strategy != QUESTIONS_PER_BOOK:
        raise GameStorageError("Game 3 must evaluate the complete fixed question bank.")
    if not 1 <= config.attempts_per_prompt <= max_runs_for_mode(config.strategy):
        raise GameStorageError("The repetition count is outside the selected mode's limit.")
    if not 1 <= config.scored_generations <= MAX_SCORED_ANSWERS:
        raise GameStorageError("Game 3 must use between 5 and 50 scored answers.")
    if config.scored_generations >= GENERATION_LIMIT_EXCLUSIVE:
        raise GameStorageError("Game 3 exceeds the scored-answer limit.")
    if not 0.0 <= float(config.temperature) <= MAX_TEMPERATURE:
        raise GameStorageError(
            f"Temperature must be between 0 and {MAX_TEMPERATURE:g}."
        )
    if not 0.0 <= float(config.top_p) <= 1.0:
        raise GameStorageError("Top-p must be between 0 and 1.")


def _ensure_single_stage_participant(
    participant: VerifiedParticipant,
    config: GameConfig,
) -> None:
    """Satisfy the legacy participant FK/gate without fabricating a baseline UI."""
    client = _admin_client()
    response = (
        client.table("copyright_game_participants")
        .select("stage_one_completed")
        .eq("competition_slug", COMPETITION_SLUG)
        .eq("user_id", participant.user_id)
        .limit(1)
        .execute()
    )
    existing = _row(response)
    if existing and bool(existing.get("stage_one_completed")):
        return

    marker = {
        "competition_slug": COMPETITION_SLUG,
        "user_id": participant.user_id,
        "direct_model_name": GAME_MODEL,
        "direct_book_key": config.book_key,
        "direct_temperature": float(config.temperature),
        "direct_top_p": float(config.top_p),
        "direct_attempts": QUESTIONS_PER_BOOK,
        "direct_rouge_l": 0.0,
        "direct_avg_rouge_l": 0.0,
        "direct_response_sha256": _hash_text(""),
        "direct_prompt": "Single-stage Game 3 entry marker; no separate baseline.",
        "direct_reference_text": None,
        "direct_completed_at": _now(),
        "stage_one_completed": True,
    }
    (
        client.table("copyright_game_participants")
        .upsert(marker, on_conflict="competition_slug,user_id")
        .execute()
    )

def get_competition() -> Dict[str, Any]:
    try:
        response = (
            _admin_client()
            .table("copyright_game_competitions")
            .select("*")
            .eq("slug", COMPETITION_SLUG)
            .limit(1)
            .execute()
        )
        competition = _row(response)
    except Exception as exc:
        raise GameStorageError(_format_database_error(exc)) from exc
    if not competition:
        raise GameStorageError(
            f"Competition '{COMPETITION_SLUG}' is not present in Supabase. "
            "Run the updated supabase/copyright_game.sql."
        )
    if competition.get("provider") != GAME_PROVIDER:
        raise GameStorageError("The Game 3 database provider does not match OpenAI.")
    if competition.get("model_name") != GAME_MODEL:
        raise GameStorageError("The Game 3 database model does not match gpt-4o-mini.")
    if competition.get("benchmark_version") != BENCHMARK_VERSION:
        raise GameStorageError("The Game 3 database benchmark is out of date.")
    if int(competition.get("max_scored_generations") or 0) != MAX_SCORED_ANSWERS:
        raise GameStorageError("The Game 3 database run limit is out of date.")
    return competition


def get_stage_one(user_id: str) -> Optional[Dict[str, Any]]:
    try:
        response = (
            _admin_client()
            .table("copyright_game_participants")
            .select("*")
            .eq("competition_slug", COMPETITION_SLUG)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        return _stage_one_aliases(_row(response))
    except Exception as exc:
        raise GameStorageError(_format_database_error(exc)) from exc


def list_stage_one_runs(user_id: str) -> List[Dict[str, Any]]:
    """Load the persisted Game 3 baseline and all five detailed answers."""
    try:
        client = _admin_client()
        run_response = (
            client.table("copyright_game_stage_one_runs")
            .select("*")
            .eq("competition_slug", COMPETITION_SLUG)
            .eq("user_id", user_id)
            .order("created_at")
            .execute()
        )
        runs = _rows(run_response)
        run_ids = [str(run.get("id") or "") for run in runs if run.get("id")]
        attempts_by_run: Dict[str, List[Dict[str, Any]]] = {
            run_id: [] for run_id in run_ids
        }
        if run_ids:
            attempt_response = (
                client.table("copyright_game_stage_one_attempts")
                .select(
                    "stage_one_run_id,attempt_number,response_text,"
                    "response_sha256,metrics,rouge_l"
                )
                .in_("stage_one_run_id", run_ids)
                .order("attempt_number")
                .execute()
            )
            for attempt in _rows(attempt_response):
                run_id = str(attempt.get("stage_one_run_id") or "")
                if run_id in attempts_by_run:
                    attempts_by_run[run_id].append(attempt)
        for run in runs:
            run["attempts"] = attempts_by_run.get(str(run.get("id") or ""), [])
        return runs
    except Exception as exc:
        raise GameStorageError(_format_database_error(exc)) from exc


def save_stage_one(
    participant: VerifiedParticipant,
    result: KnowledgeStageOneResult,
) -> Dict[str, Any]:
    """Atomically persist the five-answer baseline and its aggregate score."""
    try:
        if result.book_key not in _KNOWLEDGE_BOOK_KEYS:
            raise GameStorageError("Select one of the five knowledge-memorization books.")
        scores = [float(answer.rouge_l) for answer in result.answers]
        if len(scores) != QUESTIONS_PER_BOOK:
            raise GameStorageError("The baseline must contain all five fixed questions.")
        if any(not 0.0 <= score <= 1.0 for score in scores):
            raise GameStorageError(
                "Every baseline Token F1 score must be between 0 and 1."
            )
        if not 0.0 <= float(result.temperature) <= MAX_TEMPERATURE:
            raise GameStorageError(
                f"Temperature must be between 0 and {MAX_TEMPERATURE:g}."
            )
        if not 0.0 <= float(result.top_p) <= 1.0:
            raise GameStorageError("Top-p must be between 0 and 1.")
        _upsert_profile(participant)
        attempt_payload = [
            {
                "attempt_number": index,
                "response_text": str(answer.response or ""),
                "metrics": {
                    "question": str(answer.question or ""),
                    "ground_truth": str(answer.ground_truth or ""),
                    "token_f1": float(answer.token_f1),
                    "precision": float(answer.precision),
                    "recall": float(answer.recall),
                    "rouge_l": float(answer.rouge_l),
                },
                "rouge_l": float(answer.rouge_l),
            }
            for index, answer in enumerate(result.answers, start=1)
        ]
        response = _admin_client().rpc(
            "save_copyright_game_stage_one_run",
            {
                "p_competition_slug": COMPETITION_SLUG,
                "p_user_id": participant.user_id,
                "p_book_key": result.book_key,
                "p_prompt_text": (
                    "Knowledge Memorization Standard Q/A (five fixed questions)"
                ),
                "p_reference_text": None,
                "p_temperature": float(result.temperature),
                "p_top_p": float(result.top_p),
                "p_attempts": attempt_payload,
            },
        ).execute()
        saved_run = _row(response)
        if not saved_run:
            raise GameStorageError("Supabase did not return the saved Game 3 baseline.")
        saved_run["attempts"] = [
            {
                "stage_one_run_id": saved_run.get("id"),
                "response_sha256": _hash_text(item["response_text"]),
                **item,
            }
            for item in attempt_payload
        ]
        return _stage_one_aliases(saved_run) or saved_run
    except Exception as exc:
        if isinstance(exc, GameStorageError):
            raise
        raise GameStorageError(_format_database_error(exc)) from exc


def reset_participant_game_data(participant: VerifiedParticipant) -> None:
    try:
        (
            _admin_client()
            .table("copyright_game_participants")
            .delete()
            .eq("competition_slug", COMPETITION_SLUG)
            .eq("user_id", participant.user_id)
            .execute()
        )
    except Exception as exc:
        raise GameStorageError(_format_database_error(exc)) from exc


def _get_run(user_id: str, status: str) -> Optional[Dict[str, Any]]:
    try:
        query = (
            _admin_client()
            .table("copyright_game_runs")
            .select("*")
            .eq("competition_slug", COMPETITION_SLUG)
            .eq("user_id", user_id)
            .eq("status", status)
        )
        if status == "completed":
            query = query.order("finished_at", desc=True)
        else:
            query = query.order("started_at", desc=True)
        response = query.limit(1).execute()
        return _row(response)
    except Exception as exc:
        raise GameStorageError(_format_database_error(exc)) from exc


def get_completed_run(user_id: str) -> Optional[Dict[str, Any]]:
    row = _get_run(user_id, "completed")
    return _score_aliases(row) if row else None


def list_completed_runs(user_id: str) -> List[Dict[str, Any]]:
    """Load every completed Game 3 submission and its detailed answers."""
    try:
        client = _admin_client()
        run_response = (
            client.table("copyright_game_runs")
            .select("*")
            .eq("competition_slug", COMPETITION_SLUG)
            .eq("user_id", user_id)
            .eq("status", "completed")
            .order("finished_at")
            .execute()
        )
        runs = _rows(run_response)
        run_ids = [str(run.get("id") or "") for run in runs if run.get("id")]
        attempts_by_run: Dict[str, List[Dict[str, Any]]] = {
            run_id: [] for run_id in run_ids
        }
        if run_ids:
            attempt_response = (
                client.table("copyright_game_attempts")
                .select(
                    "run_id,mutation_attempt,prompt_attempt,book_key,rouge_l,"
                    "mutated_prompt,response_text,mutated_prompt_sha256,"
                    "response_sha256,metrics,trace"
                )
                .in_("run_id", run_ids)
                .order("mutation_attempt")
                .order("prompt_attempt")
                .execute()
            )
            for attempt in _rows(attempt_response):
                run_id = str(attempt.get("run_id") or "")
                if run_id in attempts_by_run:
                    attempts_by_run[run_id].append(attempt)
        detailed_runs: List[Dict[str, Any]] = []
        for source_run in runs:
            run = _score_aliases(source_run)
            run["attempts"] = attempts_by_run.get(str(run.get("id") or ""), [])
            detailed_runs.append(run)
        return detailed_runs
    except Exception as exc:
        raise GameStorageError(_format_database_error(exc)) from exc


def get_active_run(user_id: str) -> Optional[Dict[str, Any]]:
    return _get_run(user_id, "running")


def begin_stage_two(participant: VerifiedParticipant, config: GameConfig) -> str:
    """Atomically reserve this participant's next Game 3 run."""
    _validate_game2_config(config)
    if config.attempts_per_strategy != QUESTIONS_PER_BOOK:
        raise GameStorageError("Game 3 must evaluate the complete fixed question bank.")
    try:
        _upsert_profile(participant)
        _ensure_single_stage_participant(participant, config)
        response = _admin_client().rpc(
            "begin_copyright_game_run",
            {
                "p_competition_slug": COMPETITION_SLUG,
                "p_user_id": participant.user_id,
                "p_shot_mode": config.database_shot_mode,
                "p_strategy": config.strategy,
                "p_attempts_per_strategy": config.attempts_per_strategy,
                "p_attempts_per_prompt": config.attempts_per_prompt,
                "p_temperature": float(config.temperature),
                "p_top_p": float(config.top_p),
                "p_book_key": config.book_key,
                "p_book_keys": list(config.selected_book_keys),
            },
        ).execute()
        created = _row(response)
        run_id = str((created or {}).get("id") or "").strip()
        if not run_id:
            raise GameStorageError("Supabase did not return the reserved Game 3 run.")
        return run_id
    except GameStorageError:
        raise
    except Exception as exc:
        message = str(exc)
        if "23503" in message and "copyright_game_participants" in message:
            raise GameStageOneRequired(_format_database_error(exc)) from exc
        if "23505" in message or "duplicate key" in message.lower():
            raise GameSubmissionLocked(
                "A Game 3 official run is already active for this account."
            ) from exc
        raise GameStorageError(_format_database_error(exc)) from exc


def fail_stage_two(run_id: str, participant: VerifiedParticipant, reason: str) -> None:
    try:
        (
            _admin_client()
            .table("copyright_game_runs")
            .update(
                {
                    "status": "failed",
                    "failure_code": _safe_profile_text(
                        reason, fallback="run_failed", limit=120
                    ),
                    "finished_at": _now(),
                }
            )
            .eq("id", run_id)
            .eq("competition_slug", COMPETITION_SLUG)
            .eq("user_id", participant.user_id)
            .eq("status", "running")
            .execute()
        )
    except Exception:
        pass


def recover_active_run(participant: VerifiedParticipant) -> int:
    """Recover one sufficiently old active run through the guarded database RPC."""
    active = get_active_run(participant.user_id)
    if not active:
        return 0
    run_id = str(active.get("id") or "").strip()
    if not run_id:
        raise GameStorageError("The active Game 3 run has no valid identifier.")
    try:
        response = _admin_client().rpc(
            "recover_copyright_game_run",
            {
                "p_run_id": run_id,
                "p_user_id": participant.user_id,
                "p_competition_slug": COMPETITION_SLUG,
            },
        ).execute()
        return 1 if _row(response) else 0
    except Exception as exc:
        raise GameStorageError(_format_database_error(exc)) from exc


def complete_stage_two(
    run_id: str,
    participant: VerifiedParticipant,
    result: StageTwoResult,
) -> Dict[str, Any]:
    """Atomically store the exact score grid and complete the reserved run."""
    _validate_game2_config(result.config)
    if result.config.attempts_per_strategy != QUESTIONS_PER_BOOK:
        raise GameStorageError("Game 3 must evaluate the complete fixed question bank.")
    scores = result.scores
    if not scores or len(scores) != result.config.scored_generations:
        raise GameStorageError("An incomplete Game 3 run cannot be submitted.")
    if any(not 0.0 <= float(score) <= 1.0 for score in scores):
        raise GameStorageError("Every Game 3 Token F1 score must be between 0 and 1.")
    attempt_rows = [
        {
            "mutation_attempt": generation.mutation_attempt,
            "prompt_attempt": generation.prompt_attempt,
            "book_key": generation.book_key,
            # The shared numeric column keeps its legacy name, but stores Token F1.
            "rouge_l": generation.rouge_l,
            "mutated_prompt": generation.mutated_prompt,
            "response_text": generation.response,
            "mutated_prompt_sha256": _hash_text(generation.mutated_prompt),
            "response_sha256": _hash_text(generation.response),
            "metrics": dict(generation.metrics),
            "trace": dict(generation.trace),
        }
        for generation in result.generations
    ]
    coordinates = {
        (row["mutation_attempt"], row["prompt_attempt"]) for row in attempt_rows
    }
    if len(coordinates) != result.config.scored_generations:
        raise GameStorageError("The official run contains duplicate answer coordinates.")
    try:
        client = _admin_client()
        client.table("copyright_game_attempts").select(
            "book_key,mutated_prompt,response_text,metrics,trace"
        ).limit(1).execute()
        response = client.rpc(
            "complete_copyright_game_run",
            {
                "p_run_id": run_id,
                "p_user_id": participant.user_id,
                "p_competition_slug": COMPETITION_SLUG,
                "p_attempts": attempt_rows,
            },
        ).execute()
        completed = _row(response)
        if not completed:
            raise GameSubmissionLocked("The Game 3 run was no longer active.")
        return _score_aliases(completed)
    except GameStorageError:
        raise
    except Exception as exc:
        message = str(exc).lower()
        if "no longer active" in message or "55000" in message:
            raise GameSubmissionLocked(
                "The Game 3 run is no longer active, so no score was submitted."
            ) from exc
        raise GameStorageError(_format_database_error(exc)) from exc


def _parse_finished_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return datetime.max.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _rank_game3_daily_leaderboard(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank Game 3 by mean Token F1, then peak Token F1, then finish time."""

    ranked_rows = [_score_aliases(row) for row in rows]

    def overall_key(row: Dict[str, Any]) -> tuple[Any, ...]:
        return (
            -float(row["avg_score"]),
            -float(row["max_score"]),
            _parse_finished_at(row.get("finished_at")),
        )

    def assign_rank(field: str, key) -> None:
        ordered = sorted(
            ranked_rows,
            key=lambda row: (key(row), str(row.get("user_id") or "")),
        )
        previous: Any = None
        current_rank = 0
        for position, row in enumerate(ordered, start=1):
            rank_key = key(row)
            if previous is None or rank_key != previous:
                current_rank = position
                previous = rank_key
            row[field] = current_rank

    assign_rank("overall_rank", overall_key)
    assign_rank("average_rank", lambda row: (-float(row["avg_score"]),))
    assign_rank("peak_rank", lambda row: (-float(row["max_score"]),))
    return sorted(
        ranked_rows,
        key=lambda row: (
            int(row.get("overall_rank", 0) or 0),
            overall_key(row),
            str(row.get("user_id") or ""),
        ),
    )

def _leaderboard_view_missing(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "pgrst205" in message
        or "does not exist" in message
        or "could not find" in message
        or "relation" in message and "not found" in message
    )

def list_leaderboard(*, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    start_utc, end_utc, _ = _jeju_day_bounds(now)
    last_error: Optional[Exception] = None
    for view_name in (LEADERBOARD_VIEW, LEGACY_LEADERBOARD_VIEW):
        try:
            response = (
                _admin_client()
                .table(view_name)
                .select("*")
                .eq("competition_slug", COMPETITION_SLUG)
                .gte("finished_at", start_utc.isoformat())
                .lt("finished_at", end_utc.isoformat())
                .execute()
            )
            return _rank_game3_daily_leaderboard(_rows(response))
        except Exception as exc:
            last_error = exc
            if view_name == LEADERBOARD_VIEW and _leaderboard_view_missing(exc):
                continue
            break
    assert last_error is not None
    raise GameStorageError(_format_database_error(last_error)) from last_error


__all__ = [name for name in globals() if not name.startswith("_")]
