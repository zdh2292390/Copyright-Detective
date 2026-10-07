"""Private, transactional document checkpoints using the Python standard library.

Recovery tokens are random identifiers, never filesystem paths. Credentials are not
part of the persisted schema. SQLite stores source pairs once and updates changed
chunk rows separately so a long run does not rewrite its accumulated result blob.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any, Iterator
from urllib.parse import urlsplit


class CheckpointError(RuntimeError):
    """A checkpoint is unavailable, corrupt, or incompatible with this run."""


SETTING_KEYS = frozenset({
    "filename", "model", "provider", "chunk_size", "overlap",
    "continuation_method", "temperature", "top_p", "custom_template",
    "extra_prompt_instructions", "base_url",
})
METADATA_KEYS = frozenset({
    "fingerprint", "settings", "total_chunks", "status", "error",
    "current_chunk", "current_attempt", "retry_in_seconds", "retry_attempt",
    "started_at", "updated_at", "completed_at", "owner_id", "revision", "stop_requested",
    "active_chunks", "concurrency_limit", "effective_concurrency",
})
TOKEN_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
MAX_CHUNKS = 500_000
MAX_TEXT_CHARS = 2_000_000
MAX_JSON_CHARS = 4_000_000
SCHEMA_VERSION = 1


def _json(value: Any) -> str:
    try:
        data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        data.encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        raise CheckpointError("Checkpoint contains unsupported or non-finite values.") from exc
    if len(data) > MAX_JSON_CHARS:
        raise CheckpointError("Checkpoint value exceeds the supported size.")
    return data


def _read_json(value: str) -> Any:
    if not isinstance(value, str) or len(value) > MAX_JSON_CHARS:
        raise CheckpointError("Checkpoint contains an invalid or oversized JSON value.")
    def reject_constant(_):
        raise ValueError("Non-finite JSON value")
    try:
        return json.loads(value, parse_constant=reject_constant)
    except (TypeError, ValueError, RecursionError) as exc:
        raise CheckpointError("Checkpoint contains corrupt JSON data.") from exc


def _hash(value: str) -> str:
    if not isinstance(value, str) or len(value) > MAX_JSON_CHARS:
        raise CheckpointError("Checkpoint contains an invalid integrity value.")
    try:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
    except UnicodeError as exc:
        raise CheckpointError("Checkpoint contains invalid text encoding.") from exc


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (ValueError, OverflowError):
        return False


def _token(token: str) -> str:
    if not isinstance(token, str) or TOKEN_PATTERN.fullmatch(token) is None:
        raise CheckpointError("Invalid document recovery token; expected 64 lowercase hexadecimal characters.")
    return token


def _integer(value: Any, label: str, minimum=0, maximum=MAX_CHUNKS) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise CheckpointError(f"Checkpoint has an invalid {label}.")
    return value


def _text(value: Any, label: str, *, nullable=False, limit=MAX_TEXT_CHARS) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or len(value) > limit:
        raise CheckpointError(f"Checkpoint has an invalid {label}.")
    return value


def _settings(settings: Any) -> dict[str, Any]:
    if not isinstance(settings, Mapping):
        raise CheckpointError("Checkpoint settings must be a mapping.")
    clean = {key: value for key, value in settings.items() if key in SETTING_KEYS}
    for key, value in clean.items():
        if key in {"chunk_size", "overlap"}:
            _integer(value, key, minimum=1 if key == "chunk_size" else 0)
        elif key in {"temperature", "top_p"}:
            if not _finite_number(value):
                raise CheckpointError(f"Checkpoint has an invalid {key}.")
        else:
            _text(value, key, nullable=key in {"custom_template", "extra_prompt_instructions", "base_url"})
    base_url = clean.get("base_url")
    if base_url:
        try:
            parsed = urlsplit(base_url)
            if parsed.username is not None or parsed.password is not None:
                raise CheckpointError("Checkpoint base URL must not contain embedded credentials.")
        except ValueError as exc:
            raise CheckpointError("Checkpoint has an invalid base URL.") from exc
    _json(clean)
    return clean


def _metadata(state: Any) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise CheckpointError("Checkpoint state must be a mapping.")
    clean = {key: value for key, value in state.items() if key in METADATA_KEYS}
    if not isinstance(clean.get("fingerprint"), str) or TOKEN_PATTERN.fullmatch(clean["fingerprint"]) is None:
        raise CheckpointError("Checkpoint has an invalid document fingerprint.")
    clean["settings"] = _settings(clean.get("settings"))
    total = _integer(clean.get("total_chunks"), "chunk count")
    if not isinstance(clean.get("status"), str) or clean["status"] not in {"running", "incomplete", "complete"}:
        raise CheckpointError("Checkpoint has an invalid analysis status.")
    clean["error"] = _text(clean.get("error"), "error", nullable=True, limit=100_000)
    if "stop_requested" in clean and not isinstance(clean["stop_requested"], bool):
        raise CheckpointError("Checkpoint has an invalid stop request.")
    if "active_chunks" in clean:
        active = clean["active_chunks"]
        if not isinstance(active, list) or len(active) > total:
            raise CheckpointError("Checkpoint has an invalid active chunk list.")
        for index in active:
            _integer(index, "active chunk", minimum=1, maximum=total)
        if len(set(active)) != len(active):
            raise CheckpointError("Checkpoint has duplicate active chunks.")
        if clean["status"] == "complete" and active:
            raise CheckpointError("A complete checkpoint cannot contain active chunks.")
    for key in ("concurrency_limit", "effective_concurrency"):
        if key in clean:
            _integer(clean[key], key, minimum=1, maximum=8)
    for key in ("started_at", "updated_at", "completed_at", "owner_id"):
        if key in clean:
            _text(clean[key], key, nullable=True, limit=1024)
    for key in ("current_attempt", "retry_attempt", "revision"):
        if key in clean and clean[key] is not None:
            _integer(clean[key], key, maximum=2**63 - 1)
    if "current_chunk" in clean and clean["current_chunk"] is not None:
        _integer(clean["current_chunk"], "current chunk", maximum=total)
    if "retry_in_seconds" in clean and clean["retry_in_seconds"] is not None:
        delay = clean["retry_in_seconds"]
        if not _finite_number(delay) or delay < 0:
            raise CheckpointError("Checkpoint has an invalid retry delay.")
    _json(clean)
    return clean


def _pairs(pairs: Any, total: int) -> list[tuple[str, str]]:
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence) or len(pairs) != total:
        raise CheckpointError("Checkpoint source pairs do not match the planned chunk count.")
    clean = []
    for pair in pairs:
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise CheckpointError("Checkpoint contains an invalid source pair.")
        clean.append((_text(pair[0], "source prefix"), _text(pair[1], "source target")))
    return clean


def _pair_hash(pairs: Sequence[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for pair in pairs:
        # Length-prefixed JSON values make the manifest independent of boundaries.
        payload = _json(pair).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _indexed(values: Any, total: int, label: str) -> dict[int, Any]:
    if not isinstance(values, Mapping):
        raise CheckpointError(f"Checkpoint {label} must be a mapping.")
    clean = {}
    for index, value in values.items():
        if isinstance(index, str) and len(index) <= len(str(MAX_CHUNKS)) and index.isdecimal() and str(int(index)) == index:
            index = int(index)
        _integer(index, f"{label} index", maximum=total - 1)
        if index in clean:
            raise CheckpointError(f"Checkpoint contains duplicate {label} indices.")
        clean[index] = value
    return clean


def _chunk_rows(state: Mapping[str, Any], total: int, *, check_complete=True) -> tuple[dict[int, tuple], dict[int, tuple[str, str]]]:
    results = _indexed(state.get("results", {}), total, "results")
    failures = _indexed(state.get("failures", {}), total, "failures")
    attempts = _indexed(state.get("attempts", {}), total, "attempts")
    if set(results) & set(failures):
        raise CheckpointError("A checkpoint chunk cannot be both successful and failed.")
    rows = {}
    contexts = {}
    for index in results.keys() | failures.keys() | attempts.keys():
        result_json = None
        if index in results:
            result = results[index]
            if isinstance(result, (str, bytes)) or not isinstance(result, Sequence) or len(result) != 4:
                raise CheckpointError("Checkpoint contains an invalid chunk result.")
            upper, lower, generated, metrics = result
            contexts[index] = (_text(upper, "result prefix"), _text(lower, "result target"))
            _text(generated, "generated continuation")
            if not generated.strip():
                raise CheckpointError("Checkpoint contains an empty successful continuation.")
            if not isinstance(metrics, Mapping) or not metrics or any(not isinstance(key, str) for key in metrics):
                raise CheckpointError("Checkpoint contains invalid similarity metrics.")
            if any(not _finite_number(value) for value in metrics.values()):
                raise CheckpointError("Checkpoint contains non-finite or non-numeric similarity metrics.")
            result_json = _json([generated, dict(metrics)])
        failure = _text(failures.get(index), "chunk failure", nullable=True, limit=100_000)
        attempt_count = _integer(attempts.get(index, 0), "attempt count", maximum=2**63 - 1)
        checksum = _hash(_json([result_json, failure, attempt_count]))
        rows[index] = (result_json, failure, attempt_count, checksum)
    if check_complete and state.get("status") == "complete" and (len(results) != total or failures):
        raise CheckpointError("A complete checkpoint must contain a successful result for every chunk.")
    return rows, contexts


class DocumentCheckpointStore:
    """Local durable checkpoints, isolated by unguessable recovery token."""

    def __init__(self, root: Path | None = None):
        default = Path(__file__).resolve().parents[1] / ".cache" / "document-analysis"
        configured = os.environ.get("COPYRIGHT_DETECTIVE_DOCUMENT_CHECKPOINT_DIR")
        self.root = Path(root) if root is not None else Path(configured) if configured else default
        self.db_path = self.root / "checkpoints.sqlite3"

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = None
        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            connection = sqlite3.connect(self.db_path, timeout=2)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 2000")
            self._initialize(connection)
            try:
                os.chmod(self.root, 0o700)
                os.chmod(self.db_path, 0o600)
            except OSError:
                pass  # Windows may not support POSIX permission bits.
            yield connection
        except (OSError, sqlite3.Error) as exc:
            raise CheckpointError("Document checkpoint storage is unavailable or corrupt.") from exc
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version == SCHEMA_VERSION:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"jobs", "source_pairs", "chunk_state"}.issubset(tables):
                raise CheckpointError("Document checkpoint database has an incomplete schema.")
            return
        if version != 0:
            raise CheckpointError("Document checkpoint database schema is incompatible.")
        # Initialization is serialized so separate threads/processes can start together.
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == SCHEMA_VERSION:
                return
            tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if tables:
                raise CheckpointError("Document checkpoint database has an unrecognized schema.")
            connection.execute("CREATE TABLE jobs (token TEXT PRIMARY KEY, metadata TEXT NOT NULL, metadata_hash TEXT NOT NULL, pairs_hash TEXT NOT NULL)")
            connection.execute("CREATE TABLE source_pairs (token TEXT NOT NULL REFERENCES jobs(token) ON DELETE CASCADE, chunk_index INTEGER NOT NULL, upper_text TEXT NOT NULL, lower_text TEXT NOT NULL, PRIMARY KEY (token, chunk_index))")
            connection.execute("CREATE TABLE chunk_state (token TEXT NOT NULL, chunk_index INTEGER NOT NULL, result_json TEXT, failure TEXT, attempts INTEGER NOT NULL, checksum TEXT NOT NULL, PRIMARY KEY (token, chunk_index), FOREIGN KEY (token, chunk_index) REFERENCES source_pairs(token, chunk_index) ON DELETE CASCADE)")
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @staticmethod
    def _stored_metadata(row: sqlite3.Row) -> dict[str, Any]:
        if _hash(row["metadata"]) != row["metadata_hash"]:
            raise CheckpointError("Document checkpoint metadata failed its integrity check.")
        value = _read_json(row["metadata"])
        clean = _metadata(value)
        if clean != value:
            raise CheckpointError("Document checkpoint contains unexpected metadata fields.")
        return clean

    def create(self, state: Mapping[str, Any], chunk_pairs: Sequence[tuple[str, str]]) -> str:
        metadata = _metadata(state)
        total = metadata["total_chunks"]
        pairs = _pairs(chunk_pairs, total)
        rows, contexts = _chunk_rows(state, total)
        if any(pairs[index] != context for index, context in contexts.items()):
            raise CheckpointError("Checkpoint results do not match the document source pairs.")
        token = secrets.token_hex(32)
        metadata_json = _json(metadata)
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("INSERT INTO jobs VALUES (?, ?, ?, ?)", (token, metadata_json, _hash(metadata_json), _pair_hash(pairs)))
            connection.executemany("INSERT INTO source_pairs VALUES (?, ?, ?, ?)", ((token, index, upper, lower) for index, (upper, lower) in enumerate(pairs)))
            connection.executemany("INSERT INTO chunk_state VALUES (?, ?, ?, ?, ?, ?)", ((token, index, *row) for index, row in rows.items()))
        return token

    def save(self, token: str, state: Mapping[str, Any]) -> None:
        token = _token(token)
        metadata = _metadata(state)
        rows, contexts = _chunk_rows(state, metadata["total_chunks"])
        metadata_json = _json(metadata)
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute("SELECT * FROM jobs WHERE token = ?", (token,)).fetchone()
            if job is None:
                raise CheckpointError("Document recovery checkpoint was not found.")
            stored = self._stored_metadata(job)
            self._check_identity(stored, metadata)
            existing = {}
            for row in connection.execute("SELECT * FROM chunk_state WHERE token = ?", (token,)):
                expected = _hash(_json([row["result_json"], row["failure"], row["attempts"]]))
                if expected != row["checksum"]:
                    raise CheckpointError("Document checkpoint chunk data failed its integrity check.")
                existing[row["chunk_index"]] = (row["result_json"], row["failure"], row["attempts"], row["checksum"])
            changed = {index: row for index, row in rows.items() if existing.get(index) != row}
            for index in changed.keys() & contexts.keys():
                source = connection.execute("SELECT upper_text, lower_text FROM source_pairs WHERE token = ? AND chunk_index = ?", (token, index)).fetchone()
                if source is None or tuple(source) != contexts[index]:
                    raise CheckpointError("Checkpoint result does not match its source chunk.")
            if job["metadata"] != metadata_json:
                connection.execute("UPDATE jobs SET metadata = ?, metadata_hash = ? WHERE token = ?", (metadata_json, _hash(metadata_json), token))
            connection.executemany("INSERT INTO chunk_state VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(token, chunk_index) DO UPDATE SET result_json=excluded.result_json, failure=excluded.failure, attempts=excluded.attempts, checksum=excluded.checksum", ((token, index, *row) for index, row in changed.items()))
            connection.executemany("DELETE FROM chunk_state WHERE token = ? AND chunk_index = ?", ((token, index) for index in existing.keys() - rows.keys()))

    @staticmethod
    def _check_identity(stored: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
        if any(stored[key] != metadata[key] for key in ("fingerprint", "settings", "total_chunks")):
            raise CheckpointError("Checkpoint document fingerprint or generation settings do not match.")
        if stored.get("owner_id") != metadata.get("owner_id"):
            raise CheckpointError("Checkpoint owner cannot be changed.")

    def save_progress(self, token: str, state: Mapping[str, Any], chunk_index: int | None = None) -> None:
        """Checkpoint one trusted worker update without scanning prior successes.

        create/load/save validate the complete state at run boundaries. During a
        run the worker changes only its current chunk, so this method validates
        and writes that row together with metadata in one transaction. A final
        complete status still requires all persisted chunks to have succeeded.
        """
        token = _token(token)
        metadata = _metadata(state)
        row = None
        context = None
        if chunk_index is not None:
            _integer(chunk_index, "progress chunk index", maximum=metadata["total_chunks"] - 1)
            partial = {"status": metadata["status"]}
            for key in ("results", "failures", "attempts"):
                values = state.get(key, {})
                if not isinstance(values, Mapping):
                    raise CheckpointError(f"Checkpoint {key} must be a mapping.")
                # JSON-restored states may contain numeric string indices.
                if chunk_index in values and str(chunk_index) in values:
                    raise CheckpointError(f"Checkpoint contains duplicate {key} indices.")
                if chunk_index in values:
                    partial[key] = {chunk_index: values[chunk_index]}
                elif str(chunk_index) in values:
                    partial[key] = {chunk_index: values[str(chunk_index)]}
                else:
                    partial[key] = {}
            rows, contexts = _chunk_rows(partial, metadata["total_chunks"], check_complete=False)
            row = rows.get(chunk_index)
            context = contexts.get(chunk_index)
        metadata_json = _json(metadata)
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute("SELECT * FROM jobs WHERE token = ?", (token,)).fetchone()
            if job is None:
                raise CheckpointError("Document recovery checkpoint was not found.")
            self._check_identity(self._stored_metadata(job), metadata)
            if chunk_index is not None:
                previous = connection.execute("SELECT * FROM chunk_state WHERE token = ? AND chunk_index = ?", (token, chunk_index)).fetchone()
                if previous is not None:
                    expected = _hash(_json([previous["result_json"], previous["failure"], previous["attempts"]]))
                    if expected != previous["checksum"]:
                        raise CheckpointError("Document checkpoint chunk data failed its integrity check.")
                if context is not None:
                    source = connection.execute("SELECT upper_text, lower_text FROM source_pairs WHERE token = ? AND chunk_index = ?", (token, chunk_index)).fetchone()
                    if source is None or tuple(source) != context:
                        raise CheckpointError("Checkpoint result does not match its source chunk.")
                previous_row = None if previous is None else (previous["result_json"], previous["failure"], previous["attempts"], previous["checksum"])
                if row != previous_row:
                    if row is None:
                        connection.execute("DELETE FROM chunk_state WHERE token = ? AND chunk_index = ?", (token, chunk_index))
                    else:
                        connection.execute("INSERT INTO chunk_state VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(token, chunk_index) DO UPDATE SET result_json=excluded.result_json, failure=excluded.failure, attempts=excluded.attempts, checksum=excluded.checksum", (token, chunk_index, *row))
            if metadata["status"] == "complete":
                counts = connection.execute("SELECT COUNT(CASE WHEN result_json IS NOT NULL AND failure IS NULL THEN 1 END), COUNT(CASE WHEN failure IS NOT NULL THEN 1 END) FROM chunk_state WHERE token = ?", (token,)).fetchone()
                if counts[0] != metadata["total_chunks"] or counts[1]:
                    raise CheckpointError("A complete checkpoint must contain a successful result for every chunk.")
            if job["metadata"] != metadata_json:
                connection.execute("UPDATE jobs SET metadata = ?, metadata_hash = ? WHERE token = ?", (metadata_json, _hash(metadata_json), token))

    def load(self, token: str) -> tuple[dict[str, Any], list[tuple[str, str]]] | None:
        token = _token(token)
        if not self.db_path.exists():
            return None
        with self._connection() as connection, connection:
            connection.execute("BEGIN")
            job = connection.execute("SELECT * FROM jobs WHERE token = ?", (token,)).fetchone()
            if job is None:
                return None
            state = self._stored_metadata(job)
            source_rows = connection.execute("SELECT chunk_index, upper_text, lower_text FROM source_pairs WHERE token = ? ORDER BY chunk_index", (token,)).fetchall()
            if [row["chunk_index"] for row in source_rows] != list(range(state["total_chunks"])):
                raise CheckpointError("Checkpoint has missing or invalid document chunks.")
            pairs = _pairs([(row["upper_text"], row["lower_text"]) for row in source_rows], state["total_chunks"])
            if _pair_hash(pairs) != job["pairs_hash"]:
                raise CheckpointError("Document checkpoint source chunks failed their integrity check.")
            state.update(results={}, failures={}, attempts={})
            for row in connection.execute("SELECT * FROM chunk_state WHERE token = ? ORDER BY chunk_index", (token,)):
                index = _integer(row["chunk_index"], "stored chunk index", maximum=state["total_chunks"] - 1)
                expected = _hash(_json([row["result_json"], row["failure"], row["attempts"]]))
                if expected != row["checksum"]:
                    raise CheckpointError("Document checkpoint chunk data failed its integrity check.")
                if row["result_json"] is not None:
                    result = _read_json(row["result_json"])
                    if not isinstance(result, list) or len(result) != 2:
                        raise CheckpointError("Checkpoint contains a corrupt generated result.")
                    state["results"][index] = (*pairs[index], result[0], result[1])
                if row["failure"] is not None:
                    state["failures"][index] = row["failure"]
                if row["attempts"]:
                    state["attempts"][index] = row["attempts"]
            _chunk_rows(state, state["total_chunks"])
            return state, pairs

    def delete(self, token: str) -> None:
        token = _token(token)
        if not self.db_path.exists():
            return
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM jobs WHERE token = ?", (token,))
