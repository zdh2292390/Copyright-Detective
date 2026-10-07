"""Background job durability, cancellation and request controls without API calls."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.document_analysis import analysis_fingerprint, new_analysis_state
from src.document_checkpoints import CheckpointError, DocumentCheckpointStore
from src.document_jobs import DocumentAnalysisJobs, _analyze_chunk


def fixture(count=3):
    settings = {
        "filename": "source.txt", "model": "gemini-3.5-flash", "provider": "Google Gemini",
        "chunk_size": 200, "overlap": 50, "continuation_method": "Normal Continuation",
        "temperature": 0.7, "top_p": 0.9, "custom_template": None,
        "extra_prompt_instructions": "Continue the source", "base_url": None,
    }
    state = new_analysis_state(analysis_fingerprint("source contents", settings), settings, count)
    return state, [(f"prefix-{index}", f"target-{index}") for index in range(count)]


class FaultingStore:
    """Fail progress writes after a successful call while keeping durable reads usable."""

    def __init__(self, store):
        self.store = store
        self.fail_after_result = True

    def __getattr__(self, name):
        return getattr(self.store, name)

    def _check(self, args, kwargs):
        state = args[1] if len(args) > 1 else kwargs["state"]
        if self.fail_after_result and state.get("results"):
            raise CheckpointError("simulated checkpoint disk full")

    def save(self, *args, **kwargs):
        self._check(args, kwargs)
        return self.store.save(*args, **kwargs)

    def save_progress(self, *args, **kwargs):
        self._check(args, kwargs)
        return self.store.save_progress(*args, **kwargs)


class FailAfterCompleteStore:
    """Fail any redundant progress write after the complete transaction succeeds."""

    def __init__(self, store):
        self.store = store
        self.complete_writes = 0
        self.writes_after_complete = 0

    def __getattr__(self, name):
        return getattr(self.store, name)

    def save_progress(self, token, state, chunk_index=None):
        if self.complete_writes:
            self.writes_after_complete += 1
            raise CheckpointError("storage unavailable after successful completion")
        self.store.save_progress(token, state, chunk_index)
        if state["status"] == "complete":
            self.complete_writes += 1


class DocumentJobsTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="document-jobs-")
        self.root = Path(self.temp.name)
        self.store = DocumentCheckpointStore(self.root)
        self.managers = []
        self.release_events = []

    def tearDown(self):
        for event in self.release_events:
            event.set()
        for manager in reversed(self.managers):
            manager.close()
        self.temp.cleanup()

    @staticmethod
    def success(settings, api_key, upper, lower):
        return f"generated for {upper}", {"rouge_l": 0.0}

    def manager(self, *, store=None, analyze_chunk=None, **kwargs):
        manager = DocumentAnalysisJobs(
            store=self.store if store is None else store,
            analyze_chunk=self.success if analyze_chunk is None else analyze_chunk,
            **kwargs,
        )
        self.managers.append(manager)
        return manager

    def wait_until(self, predicate, *, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("Background job did not reach the expected state before the test deadline.")

    def finish(self, manager, token):
        self.wait_until(lambda: not manager.is_running(token))
        return manager.get(token)

    def gate(self):
        entered = Event()
        release = Event()
        self.release_events.append(release)
        calls = []

        def analyze(settings, api_key, upper, lower):
            calls.append(upper)
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test gate timed out")
            return self.success(settings, api_key, upper, lower)

        return entered, release, calls, analyze

    def test_new_manager_recovers_104_saved_results_and_processes_only_missing_chunks(self):
        state, pairs = fixture(110)
        for index in range(104):
            state["results"][index] = (*pairs[index], f"saved-{index}", {"rouge_l": 0.0})
            state["attempts"][index] = 1
        state["status"] = "running"
        state["retry_in_seconds"] = 30.0
        state["stop_requested"] = True
        first = self.manager()
        token = first.create(state, pairs)
        first.close()
        self.managers.remove(first)
        calls = []

        def analyze(settings, api_key, upper, lower):
            calls.append(upper)
            return self.success(settings, api_key, upper, lower)

        recovered = self.manager(store=DocumentCheckpointStore(self.root), analyze_chunk=analyze)
        restored = recovered.get(token)
        self.assertEqual(restored["status"], "incomplete")
        self.assertEqual(len(restored["results"]), 104)
        self.assertEqual(restored["retry_in_seconds"], 0.0)
        self.assertFalse(restored["stop_requested"])
        self.assertIn("server stopped", restored["error"])
        self.assertTrue(recovered.submit(token, "fresh-runtime-key"))
        final = self.finish(recovered, token)
        self.assertEqual(calls, [f"prefix-{index}" for index in range(104, 110)])
        self.assertEqual(final["status"], "complete")
        self.assertEqual(len(final["results"]), 110)
        self.assertTrue(all(final["attempts"][index] == 1 for index in range(104)))
        fresh = self.manager(store=DocumentCheckpointStore(self.root))
        self.assertEqual(fresh.get(token)["status"], "complete")
        self.assertFalse(fresh.submit(token, "another-key"))

    def test_repeated_reads_keep_background_work_active_and_duplicate_submit_is_rejected(self):
        entered, release, calls, analyze = self.gate()
        manager = self.manager(analyze_chunk=analyze)
        state, pairs = fixture(1)
        token = manager.create(state, pairs)
        self.assertTrue(manager.submit(token, "test-key"))
        self.assertTrue(entered.wait(2))
        for _ in range(10):
            # These independent snapshots model Streamlit script reruns/polls.
            snapshot = manager.get(token)
            self.assertEqual(snapshot["status"], "running")
            self.assertTrue(manager.is_running(token))
            snapshot["settings"]["model"] = "mutated client snapshot"
        self.assertFalse(manager.submit(token, "test-key"))
        self.assertEqual(manager.get(token)["settings"]["model"], "gemini-3.5-flash")
        self.assertEqual(calls, ["prefix-0"])
        with self.assertRaises(CheckpointError):
            manager.delete(token)
        release.set()
        self.assertEqual(self.finish(manager, token)["status"], "complete")
        self.assertFalse(manager.submit(token, "test-key"))

    def test_stop_during_inflight_call_preserves_its_result_and_leaves_remaining_uncalled(self):
        entered, release, calls, analyze = self.gate()
        manager = self.manager(analyze_chunk=analyze)
        state, pairs = fixture(3)
        token = manager.create(state, pairs)
        manager.submit(token, "test-key")
        self.assertTrue(entered.wait(2))
        self.assertTrue(manager.stop(token))
        self.assertTrue(manager.get(token)["stop_requested"])
        self.assertTrue(manager.is_running(token))
        release.set()
        final = self.finish(manager, token)
        self.assertEqual(calls, ["prefix-0"])
        self.assertEqual(set(final["results"]), {0})
        self.assertEqual(final["status"], "incomplete")
        self.assertIn("stopped by request", final["error"])
        self.assertFalse(manager.stop(token))

    def test_stop_interrupts_provider_backoff_without_waiting_60_seconds(self):
        calls = []

        def analyze(settings, api_key, upper, lower):
            calls.append(upper)
            return "Error: 429 Retry-After: 60", None

        manager = self.manager(analyze_chunk=analyze)
        state, pairs = fixture(1)
        token = manager.create(state, pairs)
        manager.submit(token, "test-key")
        self.wait_until(lambda: manager.get(token)["retry_in_seconds"] == 60.0)
        started = time.monotonic()
        self.assertTrue(manager.stop(token))
        final = self.finish(manager, token)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(calls, ["prefix-0"])
        self.assertEqual(final["attempts"], {0: 1})
        self.assertEqual(final["status"], "incomplete")
        self.assertEqual(final["retry_in_seconds"], 0.0)
        self.assertIn("stopped by request", final["error"])

    def test_owner_mismatch_cannot_read_resume_stop_or_delete_an_active_job(self):
        entered, release, _, analyze = self.gate()
        manager = self.manager(analyze_chunk=analyze)
        state, pairs = fixture(1)
        token = manager.create(state, pairs, owner_id="owner-account")
        manager.submit(token, "test-key", owner_id="owner-account")
        self.assertTrue(entered.wait(2))
        for action in (
            lambda: manager.get(token, owner_id="different-account"),
            lambda: manager.get(token),
            lambda: manager.submit(token, "another-key", owner_id="different-account"),
            lambda: manager.stop(token, owner_id="different-account"),
            lambda: manager.delete(token, owner_id="different-account"),
        ):
            with self.assertRaises(CheckpointError):
                action()
        self.assertTrue(manager.is_running(token))
        release.set()
        self.wait_until(lambda: not manager.is_running(token))
        for action in (
            lambda: manager.get(token, owner_id="different-account"),
            lambda: manager.submit(token, "another-key", owner_id="different-account"),
            lambda: manager.delete(token, owner_id="different-account"),
        ):
            with self.assertRaises(CheckpointError):
                action()
        self.assertEqual(manager.get(token, owner_id="owner-account")["status"], "complete")
        manager.delete(token, owner_id="owner-account")
        self.assertIsNone(manager.get(token, owner_id="owner-account"))

    def test_storage_failure_stops_calls_retains_last_result_and_resumes_after_repair(self):
        store = FaultingStore(self.store)
        calls = []

        def analyze(settings, api_key, upper, lower):
            calls.append(upper)
            return self.success(settings, api_key, upper, lower)

        manager = self.manager(store=store, analyze_chunk=analyze)
        state, pairs = fixture(3)
        token = manager.create(state, pairs)
        self.assertTrue(manager.submit(token, "test-key"))
        paused = self.finish(manager, token)
        self.assertEqual(calls, ["prefix-0"])
        self.assertEqual(paused["status"], "incomplete")
        self.assertEqual(set(paused["results"]), {0})
        self.assertIn("disk full", paused["error"])
        self.assertFalse(manager.is_running(token))
        self.assertEqual(self.store.load(token)[0]["results"], {})
        store.fail_after_result = False
        self.assertTrue(manager.submit(token, "test-key"))
        final = self.finish(manager, token)
        self.assertEqual(calls, ["prefix-0", "prefix-1", "prefix-2"])
        self.assertEqual(final["status"], "complete")
        self.assertEqual(set(final["results"]), {0, 1, 2})
        self.assertEqual(final["attempts"][0], 1)

    def test_queue_capacity_is_bounded_and_reopens_after_a_job_finishes(self):
        entered, release, calls, analyze = self.gate()
        manager = self.manager(analyze_chunk=analyze, max_workers=1, max_pending=1)
        state, pairs = fixture(1)
        first = manager.create(state, pairs)
        second = manager.create(state, pairs)
        self.assertTrue(manager.submit(first, "test-key"))
        self.assertTrue(entered.wait(2))
        with self.assertRaisesRegex(CheckpointError, "capacity"):
            manager.submit(second, "test-key")
        self.assertEqual(manager.get(second)["status"], "incomplete")
        self.assertEqual(calls, ["prefix-0"])
        release.set()
        self.assertEqual(self.finish(manager, first)["status"], "complete")
        self.assertTrue(manager.submit(second, "test-key"))
        self.assertEqual(self.finish(manager, second)["status"], "complete")
        self.assertEqual(calls, ["prefix-0", "prefix-0"])

    def test_pending_queue_is_bounded_and_queued_cancellation_makes_no_calls(self):
        entered, release, calls, analyze = self.gate()
        manager = self.manager(analyze_chunk=analyze, max_workers=1, max_pending=2)
        state, pairs = fixture(1)
        first = manager.create(state, pairs)
        second_state, second_pairs = fixture(1)
        second_pairs = [("queued-prefix", "queued-target")]
        second = manager.create(second_state, second_pairs)
        third = manager.create(state, pairs)
        self.assertTrue(manager.submit(first, "test-key"))
        self.assertTrue(entered.wait(2))
        self.assertTrue(manager.submit(second, "test-key"))
        self.assertTrue(manager.is_running(second))
        self.assertIsNone(manager.get(second)["current_attempt"])
        with self.assertRaisesRegex(CheckpointError, "capacity"):
            manager.submit(third, "test-key")
        self.assertTrue(manager.stop(second))
        release.set()
        self.assertEqual(self.finish(manager, first)["status"], "complete")
        queued = self.finish(manager, second)
        self.assertEqual(calls, ["prefix-0"])
        self.assertEqual(queued["results"], {})
        self.assertEqual(queued["status"], "incomplete")
        self.assertIn("stopped by request", queued["error"])

    def test_missing_api_key_is_rejected_before_scheduling_or_persisting_running_state(self):
        analyze = Mock(side_effect=AssertionError("API call must not be scheduled"))
        manager = self.manager(analyze_chunk=analyze)
        state, pairs = fixture(1)
        token = manager.create(state, pairs)
        for key in (None, "", "   "):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "API key"):
                manager.submit(token, key)
        analyze.assert_not_called()
        self.assertFalse(manager.is_running(token))
        self.assertEqual(manager.get(token)["status"], "incomplete")

    def test_keyless_local_vllm_runs_successfully_with_an_empty_runtime_key(self):
        calls = []

        def analyze(settings, api_key, upper, lower):
            calls.append((settings["provider"], settings["base_url"], api_key))
            return self.success(settings, api_key, upper, lower)

        manager = self.manager(analyze_chunk=analyze)
        for key in (None, "", "   "):
            with self.subTest(key=key):
                state, pairs = fixture(1)
                state["settings"]["provider"] = "Local vLLM"
                state["settings"]["model"] = "local-model"
                state["settings"]["base_url"] = "http://127.0.0.1:8000/v1"
                token = manager.create(state, pairs)
                self.assertTrue(manager.submit(token, key))
                final = self.finish(manager, token)
                self.assertEqual(final["status"], "complete")
                self.assertEqual(len(final["results"]), 1)
        self.assertEqual(calls, [("Local vLLM", "http://127.0.0.1:8000/v1", "")] * 3)

    def test_successful_complete_checkpoint_is_written_once_and_not_overwritten(self):
        store = FailAfterCompleteStore(self.store)
        manager = self.manager(store=store)
        state, pairs = fixture(2)
        token = manager.create(state, pairs)
        self.assertTrue(manager.submit(token, "test-key"))
        final = self.finish(manager, token)
        self.assertEqual(store.complete_writes, 1)
        self.assertEqual(store.writes_after_complete, 0)
        self.assertEqual(final["status"], "complete")
        self.assertEqual(len(final["results"]), 2)
        self.assertIsNone(final["error"])
        self.assertTrue(final["completed_at"])
        durable = self.store.load(token)[0]
        self.assertEqual(durable["status"], "complete")
        self.assertEqual(durable["completed_at"], final["completed_at"])

    def test_missing_required_stored_settings_are_rejected_before_reads_or_calls(self):
        analyze = Mock(side_effect=AssertionError("Invalid restored settings must not make API calls."))
        manager = self.manager(analyze_chunk=analyze)
        for missing in (
            "filename", "model", "provider", "chunk_size", "continuation_method", "temperature", "top_p",
        ):
            with self.subTest(missing=missing):
                state, pairs = fixture(1)
                del state["settings"][missing]
                # Persistence accepts older metadata; the job boundary must validate it.
                token = self.store.create(state, pairs)
                with self.assertRaisesRegex(CheckpointError, "missing generation settings"):
                    manager.get(token)
                with self.assertRaisesRegex(CheckpointError, "missing generation settings"):
                    manager.submit(token, "test-key")
                self.assertFalse(manager.is_running(token))
        analyze.assert_not_called()

    def test_api_keys_are_not_persisted_and_provider_errors_are_redacted(self):
        secret = "runtime-secret-api-key-12345"
        calls = []

        def analyze(settings, api_key, upper, lower):
            calls.append(api_key)
            raise RuntimeError(f"401 invalid API key {api_key}")

        manager = self.manager(analyze_chunk=analyze)
        state, pairs = fixture(1)
        state["api_key"] = "state-secret-key"
        state["settings"]["api_key"] = "settings-secret-key"
        token = manager.create(state, pairs)
        manager.submit(token, "  " + secret + "  ")
        final = self.finish(manager, token)
        self.assertEqual(calls, [secret])
        self.assertIn("[redacted]", final["error"])
        self.assertIn("[redacted]", final["failures"][0])
        self.assertNotIn("api_key", final)
        self.assertNotIn("api_key", final["settings"])
        persisted = self.store.db_path.read_bytes()
        for credential in (secret, "state-secret-key", "settings-secret-key"):
            self.assertNotIn(credential.encode(), persisted)
            self.assertNotIn(credential, repr(final))

    def test_worker_passes_captured_endpoint_and_request_controls_without_session_state(self):
        state, _ = fixture(1)
        state["settings"]["base_url"] = "https://local.example/v1"
        comparison = Mock(return_value=("normal generated", {"rouge_l": 0.0}))
        persuasion = Mock(return_value=("persuaded generated", {"rouge_l": 0.0}))
        modules = {
            "src.direct_recall.comparison": SimpleNamespace(compare_texts=comparison),
            "src.adversarial_persuasion_detection": SimpleNamespace(run_persuasion_probe=persuasion),
        }
        with patch.dict(sys.modules, modules):
            before = deepcopy(state["settings"])
            self.assertEqual(
                _analyze_chunk(state["settings"], "captured-key", "prefix", "one two three")[0],
                "normal generated",
            )
            self.assertEqual(state["settings"], before)
            state["settings"]["continuation_method"] = "Persuasion Continuation"
            self.assertEqual(
                _analyze_chunk(state["settings"], "captured-key", "prefix", "one two three")[0],
                "persuaded generated",
            )
        for provider in (comparison, persuasion):
            options = provider.call_args.kwargs
            self.assertEqual(options["base_url"], "https://local.example/v1")
            self.assertEqual(options["request_timeout"], 120)
            self.assertEqual(options["request_max_retries"], 0)
            self.assertEqual(options["target_word_count"], 3)
            self.assertEqual(options["chunk_size"], 3)
            self.assertEqual(options["temperature"], 0.7)
            self.assertEqual(options["top_p"], 0.9)
        self.assertEqual(comparison.call_args.args, ("prefix", "one two three", "captured-key"))
        self.assertEqual(persuasion.call_args.args[0], "captured-key")
        self.assertEqual(persuasion.call_args.args[-2:], ("prefix", "one two three"))

    def test_endpoint_credentials_and_query_secrets_are_rejected_before_creation(self):
        manager = self.manager()
        for endpoint in (
            "https://user:password@local.example/v1",
            "https://local.example/v1?api_key=secret",
            "https://local.example/v1#secret",
        ):
            with self.subTest(endpoint=endpoint):
                state, pairs = fixture(1)
                state["settings"]["base_url"] = endpoint
                with self.assertRaises(ValueError):
                    manager.create(state, pairs)
        self.assertFalse(self.store.db_path.exists())


if __name__ == "__main__":
    unittest.main()


