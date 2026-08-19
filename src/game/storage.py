"""Authoritative Supabase persistence for the competition.

Normal authenticated users never write official scores directly. The app first
verifies the GitHub-backed Supabase session and then performs narrowly scoped
writes with the server-only service-role client.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from supabase import Client, create_client

from src.supabase_client import (
    auth_enabled,
    get_authenticated_client,
    get_secret,
    preload_secrets,
)

from .constants import (
    BENCHMARK_VERSION,
    BOOK_KEYS,
    COMPETITION_SLUG,
    GAME_MODEL,
    GAME_PROVIDER,
    MAX_STAGE_ONE_ATTEMPTS,
    OTHER_STAGE_ONE_BOOK_KEY,
)
from .engine import GameConfig, StageOneResult, StageTwoResult


class GameStorageError(RuntimeError):
    """Raised when official competition state cannot be read or written."""


class GameStageOneRequired(GameStorageError):
    """Raised when Stage 2 has no persisted Stage 1 participant record."""


class GameIdentityError(GameStorageError):
    """Raised when the browser session cannot be verified with Supabase Auth."""


class GameSubmissionLocked(GameStorageError):
    """Raised when a participant already has an active submission."""


@dataclass(frozen=True)
class VerifiedParticipant:
    user_id: str
    github_login: str
    display_name: str
    avatar_url: str


JEJU_TIME_ZONE_NAME = "Asia/Seoul"
JEJU_TIME_ZONE = ZoneInfo(JEJU_TIME_ZONE_NAME)
LEADERBOARD_REFRESH_LOCAL_TIME = time(hour=12)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jeju_day_bounds(
    now: Optional[datetime] = None,
) -> tuple[datetime, datetime, str]:
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    local_instant = instant.astimezone(JEJU_TIME_ZONE)
    jeju_date = local_instant.date()
    start_local = datetime.combine(
        jeju_date,
        LEADERBOARD_REFRESH_LOCAL_TIME,
        tzinfo=JEJU_TIME_ZONE,
    )
    if local_instant < start_local:
        start_local = start_local - timedelta(days=1)
    end_local = start_local + timedelta(days=1)
    return (
        start_local.astimezone(timezone.utc),
        end_local.astimezone(timezone.utc),
        start_local.date().isoformat(),
    )


def get_jeju_today(now: Optional[datetime] = None) -> str:
    return _jeju_day_bounds(now)[2]


def format_jeju_leaderboard_window(now: Optional[datetime] = None) -> str:
    start_utc, end_utc, _ = _jeju_day_bounds(now)
    start_local = start_utc.astimezone(JEJU_TIME_ZONE)
    end_local = end_utc.astimezone(JEJU_TIME_ZONE)
    return (
        f"{start_local:%Y-%m-%d %H:%M} - "
        f"{end_local:%Y-%m-%d %H:%M} KST"
    )


def _parse_timestamp(value: Any) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        return datetime.max.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return datetime.max.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_finished_at_jeju(value: Any) -> str:
    parsed = _parse_timestamp(value)
    if parsed == datetime.max.replace(tzinfo=timezone.utc):
        return ""
    return parsed.astimezone(JEJU_TIME_ZONE).strftime("%Y-%m-%d %H:%M:%S KST")


def _hash_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _safe_profile_text(value: Any, *, fallback: str, limit: int = 100) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or "")).strip()
    return (text or fallback)[:limit]


def _safe_avatar_url(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("https://"):
        return text[:500]
    return ""


def _as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _verified_github_identity_data(user: Any) -> Dict[str, Any]:
    """Return immutable provider identity data, never editable user metadata."""

    app_metadata = _as_mapping(getattr(user, "app_metadata", None))
    providers = app_metadata.get("providers") or []
    if isinstance(providers, str):
        providers = [providers]
    declared_providers = {str(provider).lower() for provider in providers}
    primary_provider = str(app_metadata.get("provider") or "").lower()
    if primary_provider:
        declared_providers.add(primary_provider)
    if "github" not in declared_providers:
        raise GameIdentityError(
            "The competition requires an account authenticated through GitHub."
        )

    identities = getattr(user, "identities", None) or []
    for identity in identities:
        identity_mapping = _as_mapping(identity)
        provider = str(
            getattr(identity, "provider", None)
            or identity_mapping.get("provider")
            or ""
        ).lower()
        if provider != "github":
            continue
        identity_data = (
            getattr(identity, "identity_data", None)
            or identity_mapping.get("identity_data")
        )
        data = _as_mapping(identity_data)
        if data:
            return data
    raise GameIdentityError(
        "The verified Supabase session does not contain a GitHub identity."
    )


def backend_configured() -> bool:
    return bool(
        auth_enabled()
        and get_secret("SUPABASE_SERVICE_ROLE_KEY")
    )


def shared_api_key_configured() -> bool:
    return bool(get_shared_api_key())


def get_shared_api_key() -> str:
    """Read the shared key without placing it into Streamlit session state."""
    return get_secret("COPYRIGHT_GAME_OPENAI_API_KEY").strip()


def preload_background_job_secrets() -> None:
    """Warm Streamlit-backed secrets before a worker thread starts."""

    preload_secrets([
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "COPYRIGHT_GAME_OPENAI_API_KEY",
    ])


def missing_configuration() -> List[str]:
    missing: List[str] = []
    if not get_secret("SUPABASE_URL"):
        missing.append("SUPABASE_URL")
    if not get_secret("SUPABASE_ANON_KEY"):
        missing.append("SUPABASE_ANON_KEY")
    if not get_secret("SUPABASE_SERVICE_ROLE_KEY"):
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    if not get_shared_api_key():
        missing.append("COPYRIGHT_GAME_OPENAI_API_KEY")
    return missing


def _admin_client() -> Client:
    url = get_secret("SUPABASE_URL").strip()
    service_key = get_secret("SUPABASE_SERVICE_ROLE_KEY").strip()
    if not url or not service_key:
        raise GameStorageError(
            "The competition backend is not configured with Supabase service credentials."
        )
    return create_client(url, service_key)


def _row(response: Any) -> Optional[Dict[str, Any]]:
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data:
        first = data[0]
        return first if isinstance(first, dict) else None
    return None


def _rows(response: Any) -> List[Dict[str, Any]]:
    data = getattr(response, "data", None)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _format_database_error(exc: Exception) -> str:
    message = str(exc)
    if (
        "copyright_game_runs_participant_fk" in message
        or (
            "23503" in message
            and "copyright_game_participants" in message
        )
    ):
        return (
            "Your persisted Stage 1 baseline is missing. Run Stage 1 again before "
            "starting or submitting Stage 2. Saved leaderboard entries are unaffected."
        )
    if any(
        field in message
        for field in (
            "mutated_prompt",
            "response_text",
            "metrics",
            "trace",
            "book_key",
        )
    ) and ("PGRST204" in message or "column" in message.lower()):
        return (
            "Competition history schema is outdated. Re-run "
            "supabase/copyright_game.sql in the Supabase SQL Editor to add "
            "recoverable Stage 2 response history."
        )
    if (
        "copyright_game_stage_one_" in message
        or "save_copyright_game_stage_one_run" in message
    ) and (
        "PGRST202" in message
        or "PGRST204" in message
        or "PGRST205" in message
        or "Could not find" in message
        or "does not exist" in message
    ):
        return (
            "Competition history schema is outdated. Re-run "
            "supabase/copyright_game.sql in the Supabase SQL Editor to add "
            "GitHub-linked Stage 1 run history."
        )
    if "PGRST205" in message or (
        "Could not find the table" in message and "copyright_game" in message
    ):
        return (
            "Competition tables are missing. Run supabase/copyright_game.sql "
            "in the Supabase SQL Editor."
        )
    if (
        "direct_attempts" in message or "direct_avg_rouge_l" in message
    ) and ("PGRST204" in message or "column" in message.lower()):
        return (
            "Competition schema is outdated. Re-run supabase/copyright_game.sql "
            "in the Supabase SQL Editor to add Stage 1 scaling fields."
        )
    return message


def verify_participant() -> VerifiedParticipant:
    """Verify the current JWT with Supabase and derive safe GitHub profile fields."""
    try:
        client = get_authenticated_client()
        if client is None:
            raise GameIdentityError("Sign in with GitHub before entering the challenge.")
        response = client.auth.get_user()
        user = getattr(response, "user", None)
    except GameIdentityError:
        raise
    except Exception as exc:
        raise GameIdentityError(
            "Your GitHub session could not be verified. Please sign in again."
        ) from exc
    if user is None or not getattr(user, "id", None):
        raise GameIdentityError(
            "Your GitHub session could not be verified. Please sign in again."
        )

    user_id = str(user.id)
    identity_data = _verified_github_identity_data(user)
    raw_github_login = str(
        identity_data.get("user_name")
        or identity_data.get("preferred_username")
        or ""
    ).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", raw_github_login):
        raise GameIdentityError(
            "The verified GitHub identity does not contain a valid GitHub login."
        )
    github_login = raw_github_login
    display_name = _safe_profile_text(
        identity_data.get("full_name")
        or identity_data.get("name")
        or github_login,
        fallback=github_login,
        limit=100,
    )
    avatar_url = _safe_avatar_url(
        identity_data.get("avatar_url") or identity_data.get("picture")
    )
    return VerifiedParticipant(
        user_id=user_id,
        github_login=github_login,
        display_name=display_name,
        avatar_url=avatar_url,
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
    except Exception as exc:
        raise GameStorageError(_format_database_error(exc)) from exc
    competition = _row(response)
    if not competition:
        raise GameStorageError(
            f"Competition '{COMPETITION_SLUG}' is not present in Supabase."
        )
    if competition.get("model_name") != GAME_MODEL:
        raise GameStorageError("The database competition model does not match gpt-4o-mini.")
    return competition


def _upsert_profile(participant: VerifiedParticipant) -> None:
    payload = {
        "user_id": participant.user_id,
        "github_login": participant.github_login,
        "display_name": participant.display_name,
        "avatar_url": participant.avatar_url or None,
        "updated_at": _now(),
    }
    _admin_client().table("copyright_game_profiles").upsert(
        payload, on_conflict="user_id"
    ).execute()


def get_stage_one(user_id: str) -> Optional[Dict[str, Any]]:
    """Load the participant's latest Stage 1 summary used for progression."""
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
        row = _row(response)
        if row and not bool(row.get("stage_one_completed", True)):
            return None
        return row
    except Exception as exc:
        raise GameStorageError(_format_database_error(exc)) from exc


def list_stage_one_runs(user_id: str) -> List[Dict[str, Any]]:
    """Load every persisted Stage 1 batch together with its detailed attempts."""
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
    result: StageOneResult,
    *,
    prompt: str,
    reference_text: str,
) -> Dict[str, Any]:
    """Atomically persist one Stage 1 batch and all scored model responses."""
    try:
        if result.book_key not in (*BOOK_KEYS, OTHER_STAGE_ONE_BOOK_KEY):
            raise GameStorageError("Select a supported Stage 1 book option.")
        if not 0.0 <= float(result.temperature) <= 2.0:
            raise GameStorageError("Temperature must be between 0 and 2.")
        if not 0.0 < float(result.top_p) <= 1.0:
            raise GameStorageError("Top-p must be greater than 0 and at most 1.")

        prompt_text = str(prompt or "").strip()
        reference = str(reference_text or "").strip()
        if not prompt_text or len(prompt_text) > 20000:
            raise GameStorageError("The Stage 1 prompt is missing or too long.")
        if not reference or len(reference) > 20000:
            raise GameStorageError("The Stage 1 ground truth is missing or too long.")

        raw_attempts = list(result.attempts)
        if not raw_attempts:
            attempt_payload = [
                {
                    "attempt_number": 1,
                    "response_text": str(result.response or ""),
                    "metrics": dict(result.metrics or {}),
                    "rouge_l": float(result.rouge_l),
                }
            ]
        else:
            attempt_payload = []
            for index, attempt in enumerate(raw_attempts, start=1):
                if int(attempt.attempt) != index:
                    raise GameStorageError(
                        "Stage 1 attempts must be numbered consecutively from 1."
                    )
                attempt_payload.append(
                    {
                        "attempt_number": index,
                        "response_text": str(attempt.response or ""),
                        "metrics": dict(attempt.metrics or {}),
                        "rouge_l": float(attempt.rouge_l),
                    }
                )

        if not 1 <= len(attempt_payload) <= MAX_STAGE_ONE_ATTEMPTS:
            raise GameStorageError(
                f"Stage 1 must contain between 1 and {MAX_STAGE_ONE_ATTEMPTS} attempts."
            )
        if any(not item["response_text"].strip() for item in attempt_payload):
            raise GameStorageError("Every Stage 1 attempt must contain a model response.")
        if any(not 0.0 <= item["rouge_l"] <= 1.0 for item in attempt_payload):
            raise GameStorageError("Every Stage 1 ROUGE-L score must be between 0 and 1.")

        _upsert_profile(participant)
        response = _admin_client().rpc(
            "save_copyright_game_stage_one_run",
            {
                "p_competition_slug": COMPETITION_SLUG,
                "p_user_id": participant.user_id,
                "p_book_key": result.book_key,
                "p_prompt_text": prompt_text,
                "p_reference_text": reference,
                "p_temperature": float(result.temperature),
                "p_top_p": float(result.top_p),
                "p_attempts": attempt_payload,
            },
        ).execute()
        saved_run = _row(response)
        if not saved_run:
            raise GameStorageError("Supabase did not return the saved Stage 1 run.")
        saved_run["attempts"] = [
            {
                "stage_one_run_id": saved_run.get("id"),
                "response_sha256": _hash_text(item["response_text"]),
                **item,
            }
            for item in attempt_payload
        ]
        return saved_run
    except GameStorageError:
        raise
    except Exception as exc:
        raise GameStorageError(_format_database_error(exc)) from exc

def reset_participant_game_data(participant: VerifiedParticipant) -> None:
    """Delete every persisted Game 1 row owned by this verified user."""
    try:
        client = _admin_client()
        stage_one_rows = _rows(
            client.table("copyright_game_stage_one_runs")
            .select("id")
            .eq("competition_slug", COMPETITION_SLUG)
            .eq("user_id", participant.user_id)
            .execute()
        )
        stage_two_rows = _rows(
            client.table("copyright_game_runs")
            .select("id")
            .eq("competition_slug", COMPETITION_SLUG)
            .eq("user_id", participant.user_id)
            .execute()
        )
        stage_one_run_ids = [
            str(row.get("id")) for row in stage_one_rows if row.get("id")
        ]
        stage_two_run_ids = [
            str(row.get("id")) for row in stage_two_rows if row.get("id")
        ]

        # Delete parent runs first. Their ON DELETE CASCADE is the authorized
        # path through the immutable-attempt trigger for completed runs.
        (
            client.table("copyright_game_stage_one_runs")
            .delete()
            .eq("competition_slug", COMPETITION_SLUG)
            .eq("user_id", participant.user_id)
            .execute()
        )
        (
            client.table("copyright_game_runs")
            .delete()
            .eq("competition_slug", COMPETITION_SLUG)
            .eq("user_id", participant.user_id)
            .execute()
        )

        # Compatibility cleanup for deployments created before the cascade
        # foreign keys existed. Once the parent is gone, the attempt guard also
        # permits deletion because no terminal parent remains.
        for offset in range(0, len(stage_one_run_ids), 100):
            (
                client.table("copyright_game_stage_one_attempts")
                .delete()
                .in_(
                    "stage_one_run_id",
                    stage_one_run_ids[offset : offset + 100],
                )
                .execute()
            )
        for offset in range(0, len(stage_two_run_ids), 100):
            (
                client.table("copyright_game_attempts")
                .delete()
                .in_("run_id", stage_two_run_ids[offset : offset + 100])
                .execute()
            )
        (
            client.table("copyright_game_participants")
            .delete()
            .eq("competition_slug", COMPETITION_SLUG)
            .eq("user_id", participant.user_id)
            .execute()
        )

        remaining_tables: List[str] = []
        for table_name, select_column in (
            ("copyright_game_stage_one_runs", "id"),
            ("copyright_game_runs", "id"),
            ("copyright_game_participants", "user_id"),
        ):
            response = (
                client.table(table_name)
                .select(select_column)
                .eq("competition_slug", COMPETITION_SLUG)
                .eq("user_id", participant.user_id)
                .limit(1)
                .execute()
            )
            if _rows(response):
                remaining_tables.append(table_name)
        if remaining_tables:
            raise GameStorageError(
                "Supabase did not delete all Game 1 records from: "
                + ", ".join(remaining_tables)
            )
    except GameStorageError:
        raise
    except Exception as exc:
        raise GameStorageError(_format_database_error(exc)) from exc


def get_completed_run(user_id: str) -> Optional[Dict[str, Any]]:
    try:
        response = (
            _admin_client()
            .table("copyright_game_runs")
            .select("*")
            .eq("competition_slug", COMPETITION_SLUG)
            .eq("user_id", user_id)
            .eq("status", "completed")
            .order("finished_at", desc=True)
            .limit(1)
            .execute()
        )
        return _row(response)
    except Exception as exc:
        raise GameStorageError(_format_database_error(exc)) from exc


def list_completed_runs(user_id: str) -> List[Dict[str, Any]]:
    """Load every completed Stage 2 run with recoverable generation details."""
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
        for run in runs:
            run["attempts"] = attempts_by_run.get(str(run.get("id") or ""), [])
        return runs
    except Exception as exc:
        raise GameStorageError(_format_database_error(exc)) from exc


def get_active_run(user_id: str) -> Optional[Dict[str, Any]]:
    try:
        response = (
            _admin_client()
            .table("copyright_game_runs")
            .select("*")
            .eq("competition_slug", COMPETITION_SLUG)
            .eq("user_id", user_id)
            .eq("status", "running")
            .limit(1)
            .execute()
        )
        return _row(response)
    except Exception as exc:
        raise GameStorageError(_format_database_error(exc)) from exc


def begin_stage_two(
    participant: VerifiedParticipant,
    config: GameConfig,
) -> str:
    """Atomically reserve one active multi-book submission run."""
    config.validate()

    try:
        _upsert_profile(participant)
        response = _admin_client().rpc(
            "begin_copyright_game_run",
            {
                "p_competition_slug": COMPETITION_SLUG,
                "p_user_id": participant.user_id,
                "p_shot_mode": config.database_shot_mode,
                "p_strategy": config.strategy_label,
                "p_attempts_per_strategy": config.total_mutations,
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
            raise GameStorageError("Supabase did not return the reserved official run.")
        return run_id
    except GameStorageError:
        raise
    except Exception as exc:
        message = str(exc)
        if "23503" in message and "copyright_game_participants" in message:
            raise GameStorageError(
                "Challenge 1 participant registration is unavailable. Re-run "
                "supabase/copyright_game.sql, then retry Stage 2."
            ) from exc
        if "23505" in message or "already active" in message.lower():
            raise GameSubmissionLocked(
                "An official run is already active for this account."
            ) from exc
        raise GameStorageError(_format_database_error(exc)) from exc


def fail_stage_two(run_id: str, participant: VerifiedParticipant, reason: str) -> None:
    payload = {
        "status": "failed",
        "failure_code": _safe_profile_text(reason, fallback="run_failed", limit=120),
        "finished_at": _now(),
    }
    try:
        (
            _admin_client()
            .table("copyright_game_runs")
            .update(payload)
            .eq("id", run_id)
            .eq("user_id", participant.user_id)
            .eq("status", "running")
            .execute()
        )
    except Exception:
        # Preserve the original model/storage exception shown by the caller.
        pass


def recover_active_run(participant: VerifiedParticipant) -> int:
    """Recover one sufficiently old active run through the guarded database RPC."""
    active = get_active_run(participant.user_id)
    if not active:
        return 0
    run_id = str(active.get("id") or "").strip()
    if not run_id:
        raise GameStorageError("The active official run has no valid identifier.")
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
    """Atomically store the exact multi-book score grid and complete the run."""
    result.config.validate()
    scores = result.scores
    expected = result.config.scored_generations
    if not scores or len(scores) != expected:
        raise GameStorageError("An incomplete multi-book run cannot be submitted.")
    if any(not 0.0 <= float(score) <= 1.0 for score in scores):
        raise GameStorageError("Every official ROUGE-L score must be between 0 and 1.")
    attempt_rows = [
        {
            "mutation_attempt": generation.mutation_attempt,
            "prompt_attempt": generation.prompt_attempt,
            "book_key": generation.book_key,
            "rouge_l": generation.rouge_l,
            "mutated_prompt": generation.mutated_prompt,
            "response_text": generation.response,
            "mutated_prompt_sha256": _hash_text(generation.mutated_prompt),
            "response_sha256": _hash_text(generation.response),
            "metrics": dict(generation.metrics),
            "trace": {"strategy": generation.strategy},
        }
        for generation in result.generations
    ]
    coordinates = {
        (row["mutation_attempt"], row["prompt_attempt"]) for row in attempt_rows
    }
    if len(coordinates) != expected:
        raise GameStorageError("The official run contains duplicate attempt coordinates.")
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
            raise GameSubmissionLocked("The official run was no longer active.")
        return completed
    except GameStorageError:
        raise
    except Exception as exc:
        message = str(exc).lower()
        if "no longer active" in message or "55000" in message:
            raise GameSubmissionLocked(
                "The official run is no longer active, so no score was submitted."
            ) from exc
        raise GameStorageError(_format_database_error(exc)) from exc


def _rank_daily_leaderboard(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Every fully completed Stage 2 run is a distinct leaderboard entry, even
    # when several entries belong to the same verified participant.
    ranked_rows = [dict(source_row) for source_row in rows]

    def score(row: Dict[str, Any], field: str) -> float:
        return float(row.get(field, 0.0) or 0.0)

    def assign_rank(field: str, key) -> None:
        ordered = sorted(
            ranked_rows,
            key=lambda row: (key(row), str(row.get("user_id") or "")),
        )
        previous = None
        current_rank = 0
        for position, row in enumerate(ordered, start=1):
            rank_key = key(row)
            if previous is None or rank_key != previous:
                current_rank = position
                previous = rank_key
            row[field] = current_rank

    assign_rank(
        "overall_rank",
        lambda row: (
            -score(row, "max_rouge_l"),
            -score(row, "avg_rouge_l"),
            _parse_timestamp(row.get("finished_at")),
        ),
    )
    assign_rank("peak_rank", lambda row: (-score(row, "max_rouge_l"),))
    assign_rank("average_rank", lambda row: (-score(row, "avg_rouge_l"),))
    return sorted(
        ranked_rows,
        key=lambda row: (
            int(row.get("overall_rank", 0) or 0),
            str(row.get("user_id") or ""),
        ),
    )


def list_leaderboard(*, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    start_utc, end_utc, _ = _jeju_day_bounds(now)
    try:
        client = _admin_client()
        response = (
            client
            .table("copyright_challenge1_leaderboard")
            .select("*")
            .eq("competition_slug", COMPETITION_SLUG)
            .gte("finished_at", start_utc.isoformat())
            .lt("finished_at", end_utc.isoformat())
            .execute()
        )
        rows = _rows(response)
        run_ids = [
            str(row.get("run_id") or row.get("id") or "")
            for row in rows
            if row.get("run_id") or row.get("id")
        ]
        best_attempts: Dict[str, Dict[str, Any]] = {}
        if run_ids:
            attempt_response = (
                client.table("copyright_game_attempts")
                .select(
                    "run_id,mutation_attempt,prompt_attempt,book_key,rouge_l,"
                    "mutated_prompt,response_text,metrics"
                )
                .in_("run_id", run_ids)
                .execute()
            )
            for attempt in _rows(attempt_response):
                run_id = str(attempt.get("run_id") or "")
                previous = best_attempts.get(run_id)
                if previous is None or float(
                    attempt.get("rouge_l", 0.0) or 0.0
                ) > float(previous.get("rouge_l", 0.0) or 0.0):
                    best_attempts[run_id] = attempt

        for row in rows:
            run_id = str(row.get("run_id") or row.get("id") or "")
            row["best_attempt"] = best_attempts.get(run_id)
        return _rank_daily_leaderboard(rows)
    except Exception as exc:
        raise GameStorageError(_format_database_error(exc)) from exc

