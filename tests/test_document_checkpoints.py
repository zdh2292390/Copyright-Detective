"""Durable checkpoint regressions without provider or Streamlit dependencies."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.document_analysis import analysis_fingerprint, new_analysis_state
from src.document_checkpoints import CheckpointError, DocumentCheckpointStore, MAX_TEXT_CHARS


def fixture(count=3):
    settings = {"filename": "source.txt", "model": "gemini-3.5-flash", "provider": "Google Gemini", "chunk_size": 200, "overlap": 50, "continuation_method": "Normal Continuation", "temperature": 0.7, "top_p": 0.9, "custom_template": None, "extra_prompt_instructions": "Continue the source", "base_url": None}
    state = new_analysis_state(analysis_fingerprint("source contents", settings), settings, count)
    return state, [(f"prefix-{index}", f"target-{index}") for index in range(count)]


def succeed(state, pairs, index):
    state["results"][index] = (*pairs[index], f"generated-{index}", {"rouge_l": 0.0})
    state["attempts"][index] = state["attempts"].get(index, 0) + 1
    state["failures"].pop(index, None)


class DocumentCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="document-checkpoints-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "private"
        self.store = DocumentCheckpointStore(self.root)

    def connect(self):
        return sqlite3.connect(self.store.db_path)

    def test_store_is_lazy_and_environment_can_choose_private_directory(self):
        self.assertFalse(self.root.exists())
        self.assertIsNone(self.store.load("a" * 64))
        self.assertFalse(self.root.exists())
        with patch.dict(os.environ, {"COPYRIGHT_DETECTIVE_DOCUMENT_CHECKPOINT_DIR": str(self.root)}):
            self.assertEqual(DocumentCheckpointStore().root, self.root)

    def test_persistence_new_store_preserves_int_indices_tuple_pairs_and_running(self):
        state, pairs = fixture()
        succeed(state, pairs, 0)
        state.update(status="running", owner_id="user-1", current_chunk=2, current_attempt=3, retry_attempt=2, retry_in_seconds=1.5, stop_requested=False, started_at="2026-10-07T10:00:00+00:00", updated_at="2026-10-07T10:01:00+00:00", revision=4)
        state["failures"][1] = "Temporary timeout"
        state["attempts"][1] = 3
        token = self.store.create(state, pairs)
        self.assertRegex(token, r"^[0-9a-f]{64}$")
        loaded, restored_pairs = DocumentCheckpointStore(self.root).load(token)
        self.assertEqual(loaded, state)
        self.assertEqual(restored_pairs, pairs)
        self.assertIsInstance(next(iter(loaded["results"])), int)
        self.assertIsInstance(loaded["results"][0], tuple)
        self.assertEqual(loaded["status"], "running")
        if os.name != "nt":
            self.assertEqual(self.root.stat().st_mode & 0o777, 0o700)
            self.assertEqual(self.store.db_path.stat().st_mode & 0o777, 0o600)

    def test_parallel_metadata_round_trips_without_changing_generation_settings(self):
        state, pairs = fixture(4)
        original_settings = deepcopy(state["settings"])
        original_fingerprint = state["fingerprint"]
        state.update(status="running", active_chunks=[4, 2], concurrency_limit=8, effective_concurrency=2)
        token = self.store.create(state, pairs)
        self.assertEqual(DocumentCheckpointStore(self.root).load(token)[0], state)
        state.update(active_chunks=[4], concurrency_limit=4, effective_concurrency=1)
        self.store.save_progress(token, state)
        self.assertEqual(self.store.load(token)[0], state)
        state.update(active_chunks=[], concurrency_limit=1, effective_concurrency=1)
        self.store.save(token, state)
        loaded, _ = self.store.load(token)
        self.assertEqual(loaded, state)
        self.assertEqual(loaded["settings"], original_settings)
        self.assertEqual(loaded["fingerprint"], original_fingerprint)
        self.assertFalse({"active_chunks", "concurrency_limit", "effective_concurrency"} & loaded["settings"].keys())

    def test_old_metadata_does_not_require_parallel_defaults_or_schema_migration(self):
        state, pairs = fixture()
        token = self.store.create(state, pairs)
        loaded, _ = DocumentCheckpointStore(self.root).load(token)
        self.assertEqual(loaded, state)
        for key in ("active_chunks", "concurrency_limit", "effective_concurrency"):
            self.assertNotIn(key, loaded)
        with self.connect() as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)

    def test_invalid_parallel_metadata_is_rejected_before_creating_storage(self):
        state, pairs = fixture()
        invalid_values = {
            "active_chunks": (None, {}, (), "1", [0], [-1], [4], [True], [1.0], ["1"], [1, 1], [1, 2, 3, 3], [[]]),
            "concurrency_limit": (None, 0, -1, 9, True, 1.0, "2"),
            "effective_concurrency": (None, 0, -1, 9, False, 1.0, "2"),
        }
        for key, values in invalid_values.items():
            for value in values:
                invalid = deepcopy(state)
                invalid[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(CheckpointError):
                    self.store.create(invalid, pairs)
                self.assertFalse(self.root.exists())
        empty, empty_pairs = fixture(0)
        empty.update(active_chunks=[], concurrency_limit=1, effective_concurrency=1)
        token = self.store.create(empty, empty_pairs)
        self.assertEqual(self.store.load(token)[0], empty)

    def test_invalid_parallel_updates_preserve_persisted_results_and_metadata(self):
        state, pairs = fixture(2)
        for index in range(2):
            succeed(state, pairs, index)
        state.update(status="running", active_chunks=[], concurrency_limit=2, effective_concurrency=2)
        token = self.store.create(state, pairs)
        for save in (self.store.save, lambda token, value: self.store.save_progress(token, value, 1)):
            for update in ({"active_chunks": [2, 2]}, {"concurrency_limit": 9}, {"effective_concurrency": 0}, {"status": "complete", "active_chunks": [2]}):
                invalid = deepcopy(state)
                invalid.update(update)
                invalid["results"][1][3]["rouge_l"] = 0.8
                with self.subTest(update=update), self.assertRaises(CheckpointError):
                    save(token, invalid)
                self.assertEqual(self.store.load(token)[0], state)
        state["status"] = "complete"
        self.store.save_progress(token, state, 1)
        self.assertEqual(self.store.load(token)[0], state)

    def test_corrupt_parallel_metadata_with_matching_hash_is_rejected(self):
        state, pairs = fixture()
        token = self.store.create(state, pairs)
        for field, value in (("active_chunks", [1, 1]), ("concurrency_limit", 9), ("effective_concurrency", True)):
            with self.connect() as connection:
                metadata = json.loads(connection.execute("SELECT metadata FROM jobs WHERE token = ?", (token,)).fetchone()[0])
                metadata.update({"active_chunks": [], "concurrency_limit": 1, "effective_concurrency": 1})
                metadata[field] = value
                data = json.dumps(metadata)
                digest = hashlib.sha256(data.encode()).hexdigest()
                connection.execute("UPDATE jobs SET metadata = ?, metadata_hash = ? WHERE token = ?", (data, digest, token))
            with self.subTest(field=field), self.assertRaises(CheckpointError):
                self.store.load(token)

    def test_out_of_order_progress_preserves_successes_and_clears_active_chunks(self):
        state, pairs = fixture(3)
        state.update(status="running", active_chunks=[1, 2, 3], concurrency_limit=3, effective_concurrency=3)
        token = self.store.create(state, pairs)
        for index in (2, 0, 1):
            succeed(state, pairs, index)
            state["current_chunk"] = index + 1
            state["active_chunks"].remove(index + 1)
            state["status"] = "complete" if not state["active_chunks"] else "running"
            self.store.save_progress(token, state, index)
            self.assertEqual(self.store.load(token)[0], state)
        self.assertEqual(set(self.store.load(token)[0]["results"]), {0, 1, 2})

    def test_no_api_keys_or_arbitrary_state_fields_are_persisted(self):
        state, pairs = fixture()
        secret = "private-api-key-do-not-persist"
        state["api_key"] = secret
        state["provider_client"] = {"credential": secret}
        state["settings"]["api_key"] = secret
        state["settings"]["access_token"] = secret
        token = self.store.create(state, pairs)
        loaded, _ = self.store.load(token)
        self.assertNotIn("api_key", loaded)
        self.assertNotIn("provider_client", loaded)
        self.assertNotIn("api_key", loaded["settings"])
        self.assertNotIn(secret.encode(), self.store.db_path.read_bytes())
        state["api_key"] = "another-key"
        self.store.save(token, state)
        self.assertNotIn(b"another-key", self.store.db_path.read_bytes())

    def test_json_numeric_indices_are_restored_as_integers(self):
        state, pairs = fixture()
        succeed(state, pairs, 0)
        state = json.loads(json.dumps(state))
        token = self.store.create(state, pairs)
        loaded, _ = self.store.load(token)
        self.assertEqual(set(loaded["results"]), {0})
        self.assertEqual(loaded["attempts"], {0: 1})

    def test_save_changes_only_modified_rows_and_removes_resolved_failures(self):
        state, pairs = fixture()
        succeed(state, pairs, 0)
        token = self.store.create(state, pairs)
        with self.connect() as connection:
            connection.execute("CREATE TABLE audit (chunk_index INTEGER)")
            connection.execute("CREATE TRIGGER audit_insert AFTER INSERT ON chunk_state BEGIN INSERT INTO audit VALUES (NEW.chunk_index); END")
            connection.execute("CREATE TRIGGER audit_update AFTER UPDATE ON chunk_state BEGIN INSERT INTO audit VALUES (NEW.chunk_index); END")
        self.store.save(token, state)
        with self.connect() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM audit").fetchone()[0], 0)
        state["failures"][1] = "Empty candidate"
        state["attempts"][1] = 1
        self.store.save(token, state)
        succeed(state, pairs, 1)
        self.store.save(token, state)
        state["results"][0][3]["rouge_l"] = 0.9
        self.store.save(token, state)
        with self.connect() as connection:
            self.assertEqual(connection.execute("SELECT chunk_index FROM audit").fetchall(), [(1,), (1,), (0,)])
        loaded, _ = self.store.load(token)
        self.assertEqual(loaded["failures"], {})
        self.assertEqual(loaded["results"][0][3]["rouge_l"], 0.9)
        self.assertEqual(loaded["attempts"][1], 2)

    def test_transaction_failure_rolls_back_metadata_and_all_chunk_changes(self):
        state, pairs = fixture()
        succeed(state, pairs, 0)
        token = self.store.create(state, pairs)
        previous = deepcopy(state)
        with self.connect() as connection:
            connection.execute("CREATE TRIGGER reject_second BEFORE INSERT ON chunk_state WHEN NEW.chunk_index = 1 BEGIN SELECT RAISE(ABORT, 'forced test failure'); END")
        state["status"] = "running"
        state["results"][0][3]["rouge_l"] = 0.7
        succeed(state, pairs, 1)
        with self.assertRaises(CheckpointError):
            self.store.save(token, state)
        self.assertEqual(self.store.load(token)[0], previous)

    def test_fingerprint_settings_and_source_cannot_be_rebound(self):
        state, pairs = fixture()
        token = self.store.create(state, pairs)
        for field, value in (("fingerprint", "b" * 64), ("settings", {"model": "different"}), ("total_chunks", 4)):
            changed = deepcopy(state)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(CheckpointError):
                self.store.save(token, changed)
        succeed(state, pairs, 0)
        state["results"][0] = ("different source", *state["results"][0][1:])
        with self.assertRaises(CheckpointError):
            self.store.save(token, state)

    def test_invalid_tokens_do_not_create_paths_and_delete_cascades(self):
        for token in ("../outside", "A" * 64, "a" * 63, None, "a" * 65):
            for operation in (self.store.load, self.store.delete):
                with self.subTest(token=token), self.assertRaises(CheckpointError):
                    operation(token)
        self.assertFalse(self.root.exists())
        state, pairs = fixture()
        succeed(state, pairs, 0)
        token = self.store.create(state, pairs)
        self.store.delete(token)
        self.assertIsNone(self.store.load(token))
        self.store.delete(token)
        with self.connect() as connection:
            for table in ("jobs", "source_pairs", "chunk_state"):
                self.assertEqual(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)

    def test_corrupt_database_and_unknown_schema_raise_clear_error(self):
        self.root.mkdir()
        self.store.db_path.write_bytes(b"this is not a SQLite database")
        with self.assertRaises(CheckpointError):
            self.store.load("a" * 64)
        self.store.db_path.unlink()
        state, pairs = fixture()
        token = self.store.create(state, pairs)
        with self.connect() as connection:
            connection.execute("PRAGMA user_version = 999")
        with self.assertRaises(CheckpointError):
            self.store.load(token)

    def test_corrupt_metadata_sources_and_results_are_not_silently_reset(self):
        for mutation in ("metadata", "source", "result"):
            with self.subTest(mutation=mutation):
                state, pairs = fixture()
                succeed(state, pairs, 0)
                token = self.store.create(state, pairs)
                with self.connect() as connection:
                    if mutation == "metadata":
                        connection.execute("UPDATE jobs SET metadata = '{}' WHERE token = ?", (token,))
                    elif mutation == "source":
                        connection.execute("UPDATE source_pairs SET upper_text = 'wrong source' WHERE token = ? AND chunk_index = 0", (token,))
                    else:
                        connection.execute("UPDATE chunk_state SET result_json = 'not JSON' WHERE token = ? AND chunk_index = 0", (token,))
                with self.assertRaises(CheckpointError):
                    self.store.load(token)
                if mutation in {"metadata", "result"}:
                    with self.assertRaises(CheckpointError):
                        self.store.save(token, state)
                self.store.delete(token)

    def test_invalid_schema_values_with_matching_checksums_are_rejected(self):
        state, pairs = fixture()
        token = self.store.create(state, pairs)
        with self.connect() as connection:
            metadata = connection.execute("SELECT metadata FROM jobs WHERE token = ?", (token,)).fetchone()[0]
            bad = json.loads(metadata)
            bad["total_chunks"] = -1
            data = json.dumps(bad)
            digest = hashlib.sha256(data.encode()).hexdigest()
            connection.execute("UPDATE jobs SET metadata = ?, metadata_hash = ? WHERE token = ?", (data, digest, token))
        with self.assertRaises(CheckpointError):
            self.store.load(token)

    def test_nonfinite_metrics_and_credential_urls_are_rejected_before_write(self):
        state, pairs = fixture()
        succeed(state, pairs, 0)
        for value in (math.nan, math.inf, "api-key-string"):
            state["results"][0][3]["rouge_l"] = value
            with self.subTest(value=value), self.assertRaises(CheckpointError):
                self.store.create(state, pairs)
        state, pairs = fixture()
        state["settings"]["base_url"] = "https://username:password@example.com/v1"
        with self.assertRaises(CheckpointError):
            self.store.create(state, pairs)
        self.assertFalse(self.root.exists())

    def test_invalid_indices_and_complete_missing_results_are_rejected(self):
        state, pairs = fixture()
        for index in (-1, 3, True):
            state["attempts"] = {index: 1}
            with self.subTest(index=index), self.assertRaises(CheckpointError):
                self.store.create(state, pairs)
        state["attempts"] = {}
        state["status"] = "complete"
        with self.assertRaises(CheckpointError):
            self.store.create(state, pairs)

    def test_size_limits_reject_oversized_source_and_json(self):
        state, pairs = fixture()
        oversized = [("x" * (MAX_TEXT_CHARS + 1), pairs[0][1]), *pairs[1:]]
        with self.assertRaises(CheckpointError):
            self.store.create(state, oversized)
        with patch("src.document_checkpoints.MAX_JSON_CHARS", 20):
            with self.assertRaises(CheckpointError):
                self.store.create(state, pairs)

    def test_malformed_types_and_encodings_raise_checkpoint_errors(self):
        state, pairs = fixture()
        for key, value in (("status", []), ("temperature", 10**1000)):
            invalid = deepcopy(state)
            if key == "temperature":
                invalid["settings"][key] = value
            else:
                invalid[key] = value
            with self.subTest(key=key), self.assertRaises(CheckpointError):
                self.store.create(invalid, pairs)
        invalid = deepcopy(state)
        invalid["attempts"] = {"9" * 5000: 1}
        with self.assertRaises(CheckpointError):
            self.store.create(invalid, pairs)
        invalid = deepcopy(state)
        succeed(invalid, pairs, 0)
        invalid["results"][0][3]["rouge_l"] = 10**1000
        with self.assertRaises(CheckpointError):
            self.store.create(invalid, pairs)
        invalid["results"][0][3]["rouge_l"] = 0.0
        invalid["error"] = "\\ud800".encode().decode("unicode_escape")
        with self.assertRaises(CheckpointError):
            self.store.create(invalid, pairs)
        token = self.store.create(state, pairs)
        with self.connect() as connection:
            connection.execute("UPDATE jobs SET metadata = ? WHERE token = ?", (b"not valid text", token))
        with self.assertRaises(CheckpointError):
            self.store.load(token)

    def test_save_progress_reads_and_writes_only_current_chunk(self):
        statements = []
        class TracedStore(DocumentCheckpointStore):
            @contextmanager
            def _connection(self):
                with super()._connection() as connection:
                    connection.set_trace_callback(statements.append)
                    yield connection
        store = TracedStore(self.root)
        state, pairs = fixture(1974)
        for index in range(104):
            succeed(state, pairs, index)
        token = store.create(state, pairs)
        with self.connect() as connection:
            connection.execute("CREATE TABLE audit (chunk_index INTEGER)")
            connection.execute("CREATE TRIGGER audit_insert AFTER INSERT ON chunk_state BEGIN INSERT INTO audit VALUES (NEW.chunk_index); END")
            connection.execute("CREATE TRIGGER audit_update AFTER UPDATE ON chunk_state BEGIN INSERT INTO audit VALUES (NEW.chunk_index); END")
        statements.clear()
        state["status"] = "running"
        store.save_progress(token, state, None)
        self.assertFalse(any("chunk_state" in sql for sql in statements))
        statements.clear()
        succeed(state, pairs, 104)
        state["current_chunk"] = 105
        store.save_progress(token, state, 104)
        chunk_reads = [sql for sql in statements if sql.startswith("SELECT") and "chunk_state" in sql]
        self.assertEqual(len(chunk_reads), 1)
        self.assertIn("chunk_index = 104", chunk_reads[0])
        store.save_progress(token, state, 104)
        with self.connect() as connection:
            self.assertEqual(connection.execute("SELECT chunk_index FROM audit").fetchall(), [(104,)])
        loaded, _ = store.load(token)
        self.assertEqual(len(loaded["results"]), 105)
        self.assertEqual(loaded, state)

    def test_save_progress_complete_status_checks_persisted_counts_and_rolls_back(self):
        state, pairs = fixture(2)
        token = self.store.create(state, pairs)
        succeed(state, pairs, 0)
        state["status"] = "complete"
        with self.assertRaises(CheckpointError):
            self.store.save_progress(token, state, 0)
        loaded, _ = self.store.load(token)
        self.assertEqual(loaded["results"], {})
        self.assertEqual(loaded["status"], "incomplete")
        state["status"] = "running"
        self.store.save_progress(token, state, 0)
        succeed(state, pairs, 1)
        state["status"] = "complete"
        self.store.save_progress(token, state, 1)
        self.assertEqual(self.store.load(token)[0], state)

    def test_save_progress_preserves_owner_and_rejects_corrupt_current_row(self):
        state, pairs = fixture()
        state["owner_id"] = "user-1"
        token = self.store.create(state, pairs)
        changed = deepcopy(state)
        changed["owner_id"] = "user-2"
        for save in (self.store.save, lambda token, value: self.store.save_progress(token, value, None)):
            with self.assertRaises(CheckpointError):
                save(token, changed)
        succeed(state, pairs, 0)
        self.store.save_progress(token, state, 0)
        with self.connect() as connection:
            connection.execute("UPDATE chunk_state SET attempts = 999 WHERE token = ? AND chunk_index = 0", (token,))
        with self.assertRaises(CheckpointError):
            self.store.save_progress(token, state, 0)

    def test_concurrent_separate_jobs_and_readers_do_not_share_connections(self):
        def job(number):
            store = DocumentCheckpointStore(self.root)
            state, pairs = fixture(8)
            state["owner_id"] = f"user-{number}"
            token = store.create(state, pairs)
            for index in range(len(pairs)):
                succeed(state, pairs, index)
                state["current_chunk"] = index + 1
                state["status"] = "complete" if index + 1 == len(pairs) else "running"
                store.save(token, state)
                restored, restored_pairs = DocumentCheckpointStore(self.root).load(token)
                self.assertEqual(restored, state)
                self.assertEqual(restored_pairs, pairs)
            return token
        with ThreadPoolExecutor(max_workers=6) as executor:
            tokens = list(executor.map(job, range(6)))
        self.assertEqual(len(set(tokens)), 6)
        for token in tokens:
            loaded, _ = self.store.load(token)
            self.assertEqual(loaded["status"], "complete")
            self.assertEqual(len(loaded["results"]), 8)


if __name__ == "__main__":
    unittest.main()
