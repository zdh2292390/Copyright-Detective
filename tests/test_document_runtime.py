"""Regressions for recovery after document storage failures and deletion."""
import ast
from copy import deepcopy
from concurrent.futures import Future
from threading import Event, Lock, Thread
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from unittest.mock import patch

from src.document_analysis import new_analysis_state
from src.document_checkpoints import CheckpointError, DocumentCheckpointStore
from src.document_jobs import DocumentAnalysisJobs, MAX_ERROR_CHARS, _redact_error
from src.run_cancellation import CancellationToken

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "a" * 64
PAIRS = [("upper A", "lower A"), ("upper B", "lower B")]


def checkpoint(indices=(), *, status="incomplete", owner="owner"):
    settings = dict(filename="document.txt", model="gpt-4o-mini", provider="OpenAI",
                    chunk_size=50, continuation_method="Normal Continuation", temperature=0.7, top_p=0.9)
    state = new_analysis_state("b" * 64, settings, 2)
    state.update(status=status, owner_id=owner)
    for index in indices:
        state["results"][index] = (*PAIRS[index], "generated " + str(index), {"score": 0.5})
        state["attempts"][index] = 1
    return state


class RecoveredStore:
    remote = True

    def __init__(self, state):
        self.state = state
        self.error = None
        self.loads = 0
        self.active = False

    def load(self, token):
        self.loads += 1
        if self.error:
            raise self.error
        return (deepcopy(self.state), list(PAIRS)) if self.state is not None else None

    def is_active(self, token):
        return self.active

    def list_for_owner(self, owner_id, **kwargs):
        return [{"token": TOKEN, "status": self.state["status"],
                 "completed_chunks": len(self.state["results"])}]


class DocumentRecovery(unittest.TestCase):
    def create_jobs(self, durable, retained=None):
        store = RecoveredStore(durable)
        jobs = DocumentAnalysisJobs(store)
        self.addCleanup(jobs._chunk_executor.shutdown, wait=True, cancel_futures=True)
        self.addCleanup(jobs._executor.shutdown, wait=True, cancel_futures=True)
        if retained is not None:
            jobs._jobs[TOKEN] = dict(state=deepcopy(retained), store=store, finished=True, storage_failed=True)
        return jobs, store

    def test_finished_fallback_refreshes_after_another_worker_completes(self):
        jobs, store = self.create_jobs(checkpoint((0, 1), status="complete"), checkpoint((0,)))
        result = jobs.get(TOKEN, owner_id="owner")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(set(result["results"]), {0, 1})
        self.assertNotIn(TOKEN, jobs._jobs, "The recovered task must release its capacity reservation")
        self.assertEqual(store.loads, 1)

    def test_process_listing_releases_recovered_capacity_without_opening_each_report(self):
        jobs, store = self.create_jobs(checkpoint((0, 1), status="complete"), checkpoint((0,)))
        listing = jobs.list_saved(owner_id="owner")
        self.assertEqual(listing[0]["status"], "complete")
        self.assertNotIn(TOKEN, jobs._jobs)
        self.assertEqual(store.loads, 0)

    def test_listing_does_not_drop_unsaved_partial_or_active_tasks(self):
        jobs, store = self.create_jobs(checkpoint((0,)), checkpoint((0, 1)))
        jobs.list_saved(owner_id="owner")
        self.assertIn(TOKEN, jobs._jobs)
        store.state = checkpoint((0, 1), status="complete")
        jobs._jobs[TOKEN]["finished"] = False
        jobs.list_saved(owner_id="owner")
        self.assertIn(TOKEN, jobs._jobs)

    def test_deleted_durable_task_is_not_resurrected_by_retained_fallback(self):
        jobs, store = self.create_jobs(None, checkpoint((0,)))
        self.assertIsNone(jobs.get(TOKEN, owner_id="owner"))
        self.assertNotIn(TOKEN, jobs._jobs)
        self.assertEqual(store.loads, 1)

    def test_storage_still_unavailable_keeps_unsaved_successes(self):
        jobs, store = self.create_jobs(checkpoint(), checkpoint((0,)))
        store.error = CheckpointError("temporarily unavailable")
        result = jobs.get(TOKEN, owner_id="owner")
        self.assertEqual(set(result["results"]), {0})
        self.assertTrue(jobs._jobs[TOKEN]["storage_failed"])
        result["results"].clear()
        self.assertEqual(set(jobs._jobs[TOKEN]["state"]["results"]), {0})

    def test_missing_successes_are_merged_without_hiding_new_worker_progress(self):
        durable = checkpoint((1,), status="running")
        durable["failures"][0] = "old temporary failure"
        retained = checkpoint((0,))
        retained["attempts"][0] = 3
        jobs, store = self.create_jobs(durable, retained)
        store.active = True
        result = jobs.get(TOKEN, owner_id="owner")
        self.assertEqual(result["status"], "running")
        self.assertEqual(set(result["results"]), {0, 1})
        self.assertNotIn(0, result["failures"])
        self.assertEqual(result["attempts"][0], 3)
        self.assertIn(TOKEN, jobs._jobs, "Unsaved successes remain available for explicit Resume")
        store.state = checkpoint((0, 1), status="complete")
        store.active = False
        self.assertEqual(jobs.get(TOKEN, owner_id="owner")["status"], "complete")
        self.assertNotIn(TOKEN, jobs._jobs)

    def test_durable_response_takes_precedence_over_retained_response(self):
        durable = checkpoint((0, 1), status="complete")
        retained = checkpoint((0,))
        retained["results"][0] = (*PAIRS[0], "old fallback", {"score": 0.1})
        jobs, _ = self.create_jobs(durable, retained)
        self.assertEqual(jobs.get(TOKEN, owner_id="owner")["results"][0][2], "generated 0")

    def test_previous_account_cannot_probe_retained_task_storage(self):
        jobs, store = self.create_jobs(checkpoint((0, 1), status="complete"), checkpoint((0,)))
        with self.assertRaises(CheckpointError):
            jobs.get(TOKEN, owner_id="other")
        self.assertEqual(store.loads, 0)


class LimitedErrorStore(DocumentCheckpointStore):
    @staticmethod
    def check_errors(state):
        if any(len(message) > MAX_ERROR_CHARS for message in state["failures"].values()):
            raise CheckpointError("Cloud item error exceeds 16000 characters")

    def save(self, token, state):
        self.check_errors(state)
        return super().save(token, state)

    def save_progress(self, token, state, chunk_index=None):
        self.check_errors(state)
        return super().save_progress(token, state, chunk_index)


class DocumentErrorLimits(unittest.TestCase):
    def create_jobs(self, analyze_chunk):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = LimitedErrorStore(Path(temporary.name))
        jobs = DocumentAnalysisJobs(store, analyze_chunk=analyze_chunk)
        self.addCleanup(jobs.close)
        return jobs, store

    def wait_finished(self, jobs, token):
        deadline = monotonic() + 3
        while jobs.is_active(token, owner_id="owner") and monotonic() < deadline:
            sleep(0.005)
        self.assertFalse(jobs.is_active(token, owner_id="owner"))

    def test_redaction_happens_before_error_is_bounded(self):
        message = _redact_error("secret " + "x" * 20000 + "secret", "secret")
        self.assertNotIn("secret", message)
        self.assertLessEqual(len(message), MAX_ERROR_CHARS)
        self.assertIn("[redacted]", message)
        self.assertTrue(message.endswith("[truncated]"))

    def test_large_provider_error_is_saved_as_chunk_failure(self):
        jobs, store = self.create_jobs(lambda *args: "Error: model not found " + "x" * 20000)
        token = jobs.create(checkpoint(), PAIRS, owner_id="owner", use_cloud=False)
        self.assertTrue(jobs.submit(token, "credential", owner_id="owner"))
        self.wait_finished(jobs, token)
        state, _ = store.load(token)
        self.assertEqual(state["status"], "incomplete")
        self.assertIn(0, state["failures"])
        self.assertLessEqual(len(state["failures"][0]), MAX_ERROR_CHARS)
        self.assertNotIn(token, jobs._jobs, "A provider error must not turn into a storage outage")

    def test_imported_failures_are_bounded_without_mutating_caller(self):
        jobs, store = self.create_jobs(lambda *args: None)
        imported = checkpoint()
        imported["failures"][0] = "Error: " + "x" * 20000
        token = jobs.create(imported, PAIRS, owner_id="owner", use_cloud=False)
        saved, _ = store.load(token)
        self.assertLessEqual(len(saved["failures"][0]), MAX_ERROR_CHARS)
        self.assertGreater(len(imported["failures"][0]), MAX_ERROR_CHARS)

    def test_legacy_failures_are_bounded_before_resume_saves(self):
        jobs, store = self.create_jobs(lambda *args: "Error: model not found")
        legacy = checkpoint()
        legacy["failures"][1] = "Error: " + "x" * 20000
        token = store.create(legacy, PAIRS)
        self.assertTrue(jobs.submit(token, "credential", owner_id="owner"))
        self.wait_finished(jobs, token)
        saved, _ = store.load(token)
        self.assertTrue(all(len(message) <= MAX_ERROR_CHARS for message in saved["failures"].values()))


class DocumentPageRecovery(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / "src/pages/document_memorization_detection.py").read_text())
        names = {"_restore_document_job", "_detach_document_job", "_forget_document_job"}
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.state = {"user_id": "owner", "pdf_analysis_job_token": TOKEN,
                      "pdf_analysis_state": {"old": True}, "pdf_analysis_results": ["old"],
                      "pdf_report_bytes": b"old", "_pdf_document_jobs": {TOKEN: "owner"}}
        self.params = {"document_analysis": TOKEN}
        self.errors = []
        self.result = None
        self.error = None
        def read(*args, **kwargs):
            if self.error:
                raise self.error
            return self.result
        self.ns = dict(st=SimpleNamespace(session_state=self.state, query_params=self.params, error=self.errors.append),
                       PDF_JOB_TOKEN_KEY="pdf_analysis_job_token", PDF_JOB_QUERY_KEY="document_analysis",
                       PDF_SESSION_JOBS_KEY="_pdf_document_jobs", DOCUMENT_JOBS=SimpleNamespace(get=read),
                       CheckpointError=CheckpointError, _document_job_hidden=lambda state: False,
                       _remember_document_job=lambda *args: None, _sync_document_job=lambda state: None)
        exec(compile(ast.Module(body=selected, type_ignores=[]), "document-page-recovery", "exec"), self.ns)
        cloud = ModuleType("src.supabase_document_checkpoints")
        cloud.is_cloud_document_token = lambda token: False
        binding = patch.dict(sys.modules, {cloud.__name__: cloud})
        binding.start()
        self.addCleanup(binding.stop)

    def test_missing_task_clears_stale_recovery_binding_and_report(self):
        self.assertIsNone(self.ns["_restore_document_job"]())
        self.assertNotIn("document_analysis", self.params)
        self.assertNotIn("pdf_analysis_job_token", self.state)
        self.assertNotIn("pdf_analysis_results", self.state)
        self.assertNotIn("pdf_report_bytes", self.state)
        self.assertEqual(self.state["_pdf_document_jobs"], {})
        self.assertEqual(self.errors, [])
        self.assertIsNone(self.ns["_restore_document_job"]())
        self.assertEqual(self.errors, [])

    def test_temporary_storage_failure_keeps_recovery_link_but_clears_old_report(self):
        self.error = CheckpointError("temporarily unavailable")
        self.assertIsNone(self.ns["_restore_document_job"]())
        self.assertEqual(self.params["document_analysis"], TOKEN)
        self.assertEqual(self.state["pdf_analysis_job_token"], TOKEN)
        self.assertNotIn("pdf_analysis_results", self.state)
        self.assertEqual(len(self.errors), 1)


OTHER_TOKEN = "c" * 64


class ControlledStore:
    """Each gate blocks exactly one token's I/O, never the store's data lock."""
    def __init__(self, *, remote=False):
        self.remote = remote
        self.states = {token: checkpoint() for token in (TOKEN, OTHER_TOKEN)}
        self.active = set()
        self.gates = {}
        self.calls = []
        self.lock = Lock()

    def block(self, operation, token=TOKEN):
        entered, released = Event(), Event()
        self.gates[operation, token] = entered, released
        return entered, released

    def call(self, operation, token):
        with self.lock:
            self.calls.append((operation, token))
        gate = self.gates.get((operation, token))
        if gate:
            gate[0].set()
            if not gate[1].wait(4):
                raise AssertionError("Unreleased storage test gate: " + operation)

    def load(self, token):
        self.call("load", token)
        with self.lock:
            state = self.states.get(token)
            return (deepcopy(state), list(PAIRS)) if state is not None else None

    def save(self, token, state):
        self.call("save", token)
        return self.persist(token, state)

    def persist(self, token, state):
        with self.lock:
            if token not in self.states:
                raise CheckpointError("Task deleted")
            self.states[token] = deepcopy(state)
            return deepcopy(state)

    def save_progress(self, token, state, chunk_index=None):
        self.call("save_progress", token)
        return self.persist(token, state)

    def claim(self, token):
        self.call("claim", token)
        with self.lock:
            if token in self.active:
                return False
            self.active.add(token)
            self.states[token]["status"] = "running"
        return True

    def release(self, token):
        self.call("release", token)
        with self.lock:
            self.active.discard(token)

    def heartbeat(self, token):
        with self.lock:
            return deepcopy(self.states[token])

    def is_active(self, token):
        self.call("is_active", token)
        with self.lock:
            return token in self.active

    def request_stop(self, token):
        self.call("request_stop", token)
        with self.lock:
            if token in self.states:
                self.states[token]["stop_requested"] = True
        return True

    def request_kill(self, token, *, active=False):
        self.call("request_kill", token)
        with self.lock:
            self.states.pop(token, None)
            self.active.discard(token)
        return {"deleted": True}

    def delete(self, token):
        self.call("delete", token)
        with self.lock:
            self.states.pop(token, None)


class DocumentTaskIOIsolation(unittest.TestCase):
    def create_jobs(self, *, remote=False, max_pending=16, run_workers=False):
        store = ControlledStore(remote=remote)
        jobs = DocumentAnalysisJobs(store, max_pending=max_pending,
                                    analyze_chunk=lambda *args: ("generated", {"score": 0.5}))
        self.addCleanup(jobs.close)
        if not run_workers:
            hold_workers = patch.object(jobs._executor, "submit", return_value=Future())
            self.submit_mock = hold_workers.start()
            self.addCleanup(hold_workers.stop)
        return jobs, store

    def active_job(self, jobs, store, token):
        record = dict(state=checkpoint(status="running"), store=store,
                      stop=Event(), cancellation=CancellationToken(), kill=Event())
        with jobs._lock:
            jobs._jobs[token] = record
        return record

    def launch(self, operation, store):
        outcome = Future()
        def run():
            try:
                outcome.set_result(operation())
            except BaseException as exc:
                outcome.set_exception(exc)
        thread = Thread(target=run, daemon=True)
        thread.start()
        def cleanup():
            for _, released in store.gates.values():
                released.set()
            thread.join(3)
            self.assertFalse(thread.is_alive(), "Task operation did not finish after its gate was released")
        self.addCleanup(cleanup)
        return outcome

    def assert_other_task_can_stop(self, jobs, store, blocked_operation):
        other = self.active_job(jobs, store, OTHER_TOKEN)
        result = self.launch(lambda: jobs.force_stop(OTHER_TOKEN, owner_id="owner"), store)
        self.assertTrue(result.result(timeout=0.75), blocked_operation + " held the registry lock")
        self.assertTrue(other["stop"].is_set())
        self.assertTrue(other["cancellation"].cancelled)

    def test_large_result_copy_does_not_hold_registry_lock(self):
        import src.document_jobs as module
        jobs, store = self.create_jobs()
        current = self.active_job(jobs, store, TOKEN)
        current["state"]["settings"]["filename"] = "large-report"
        entered, released = Event(), Event()
        def copy_result(value):
            if isinstance(value, dict) and value.get("settings", {}).get("filename") == "large-report":
                entered.set()
                if not released.wait(3):
                    raise AssertionError("Result copy gate was not released")
            return deepcopy(value)
        with patch.object(module, "deepcopy", side_effect=copy_result):
            copying = self.launch(lambda: jobs.get(TOKEN, owner_id="owner"), store)
            self.assertTrue(entered.wait(1))
            try:
                self.assert_other_task_can_stop(jobs, store, "large result copy")
            finally:
                released.set()
            self.assertEqual(copying.result(timeout=1)["settings"]["filename"], "large-report")

    def test_slow_checkpoint_load_does_not_block_other_task_stop(self):
        jobs, store = self.create_jobs()
        entered, released = store.block("load")
        operation = self.launch(lambda: jobs.get(TOKEN, owner_id="owner"), store)
        self.assertTrue(entered.wait(1))
        self.assert_other_task_can_stop(jobs, store, "load")
        released.set()
        self.assertEqual(operation.result(timeout=1)["status"], "incomplete")

    def test_slow_submit_save_and_claim_do_not_block_other_task_stop(self):
        for stage in ("save", "claim"):
            with self.subTest(stage=stage):
                jobs, store = self.create_jobs(remote=True)
                entered, released = store.block(stage)
                operation = self.launch(lambda: jobs.submit(TOKEN, "key", owner_id="owner"), store)
                self.assertTrue(entered.wait(1))
                self.assert_other_task_can_stop(jobs, store, stage)
                released.set()
                self.assertTrue(operation.result(timeout=1))
                self.assertFalse(jobs._submitting)

    def test_restoration_save_does_not_block_other_task_stop(self):
        jobs, store = self.create_jobs()
        store.states[TOKEN]["status"] = "running"
        entered, released = store.block("save")
        operation = self.launch(lambda: jobs.get(TOKEN, owner_id="owner"), store)
        self.assertTrue(entered.wait(1))
        self.assert_other_task_can_stop(jobs, store, "restore save")
        released.set()
        self.assertEqual(operation.result(timeout=1)["status"], "incomplete")

    def test_slow_stop_rpc_does_not_block_other_task_stop(self):
        jobs, store = self.create_jobs(remote=True)
        current = self.active_job(jobs, store, TOKEN)
        entered, released = store.block("request_stop")
        operation = self.launch(lambda: jobs.force_stop(TOKEN, owner_id="owner"), store)
        self.assertTrue(entered.wait(1))
        self.assertTrue(current["cancellation"].cancelled, "Local cancellation must precede the slow RPC")
        self.assert_other_task_can_stop(jobs, store, "stop RPC")
        released.set()
        self.assertTrue(operation.result(timeout=1))

    def test_slow_kill_and_delete_do_not_block_other_task_stop(self):
        for stage in ("request_kill", "delete"):
            with self.subTest(stage=stage):
                jobs, store = self.create_jobs()
                entered, released = store.block(stage)
                command = jobs.force_kill if stage == "request_kill" else jobs.delete
                operation = self.launch(lambda: command(TOKEN, owner_id="owner"), store)
                self.assertTrue(entered.wait(1))
                self.assert_other_task_can_stop(jobs, store, stage)
                released.set()
                operation.result(timeout=1)
                self.assertNotIn(TOKEN, store.states)
                self.assertNotIn(TOKEN, jobs._lifecycle_locks)

    def test_slow_finalize_release_does_not_block_other_task_stop(self):
        jobs, store = self.create_jobs(remote=True, run_workers=True)
        entered, released = store.block("release")
        self.addCleanup(released.set)
        self.assertTrue(jobs.submit(TOKEN, "key", owner_id="owner"))
        self.assertTrue(entered.wait(2))
        self.assert_other_task_can_stop(jobs, store, "finalize release")
        released.set()
        deadline = monotonic() + 2
        while jobs.is_running(TOKEN) and monotonic() < deadline:
            sleep(0.005)
        self.assertFalse(jobs.is_running(TOKEN))

    def test_waiting_for_same_token_does_not_block_other_tokens_or_duplicate_submit(self):
        jobs, store = self.create_jobs(remote=True)
        entered, released = store.block("save")
        first = self.launch(lambda: jobs.submit(TOKEN, "key", owner_id="owner"), store)
        self.assertTrue(entered.wait(1))
        second = self.launch(lambda: jobs.submit(TOKEN, "key", owner_id="owner"), store)
        deadline = monotonic() + 1
        while monotonic() < deadline:
            with jobs._lock:
                if jobs._lifecycle_locks[TOKEN]["users"] == 2:
                    break
            sleep(0.005)
        self.assertFalse(second.done())
        self.assert_other_task_can_stop(jobs, store, "same-token waiter")
        released.set()
        self.assertTrue(first.result(timeout=1))
        self.assertFalse(second.result(timeout=1))
        self.assertEqual(store.calls.count(("claim", TOKEN)), 1)
        self.assertEqual(self.submit_mock.call_count, 1)
        self.assertNotIn(TOKEN, jobs._lifecycle_locks)

    def test_capacity_reservation_includes_submissions_waiting_for_storage(self):
        jobs, store = self.create_jobs(max_pending=1)
        entered, released = store.block("load")
        first = self.launch(lambda: jobs.submit(TOKEN, "key", owner_id="owner"), store)
        self.assertTrue(entered.wait(1))
        second = self.launch(lambda: jobs.submit(OTHER_TOKEN, "key", owner_id="owner"), store)
        with self.assertRaises(CheckpointError):
            second.result(timeout=0.75)
        self.assertNotIn(("load", OTHER_TOKEN), store.calls)
        released.set()
        self.assertTrue(first.result(timeout=1))
        self.assertFalse(jobs._submitting)

    def test_failed_submission_releases_reservation_and_lifecycle_lock(self):
        jobs, store = self.create_jobs(max_pending=1)
        store.states.pop(TOKEN)
        with self.assertRaises(CheckpointError):
            jobs.submit(TOKEN, "key", owner_id="owner")
        self.assertFalse(jobs._submitting)
        self.assertFalse(jobs._lifecycle_locks)
        self.assertTrue(jobs.submit(OTHER_TOKEN, "key", owner_id="owner"))

    def test_shutdown_fences_submission_that_was_waiting_for_storage(self):
        jobs, store = self.create_jobs()
        entered, released = store.block("load")
        submitting = self.launch(lambda: jobs.submit(TOKEN, "key", owner_id="owner"), store)
        self.assertTrue(entered.wait(1))
        jobs.close()
        released.set()
        with self.assertRaises(CheckpointError):
            submitting.result(timeout=1)
        self.assertFalse(jobs._submitting)
        self.assertFalse(jobs._jobs)
        self.assertFalse(jobs._lifecycle_locks)
        self.assertEqual(self.submit_mock.call_count, 0)
        self.assertEqual(store.states[TOKEN]["status"], "incomplete")
        with self.assertRaises(CheckpointError):
            jobs.submit(OTHER_TOKEN, "key", owner_id="owner")

    def test_owner_check_and_worker_identity_fence_are_preserved(self):
        jobs, store = self.create_jobs()
        original = self.active_job(jobs, store, TOKEN)
        with self.assertRaises(CheckpointError):
            jobs.force_stop(TOKEN, owner_id="other")
        self.assertFalse(original["cancellation"].cancelled)
        replacement = self.active_job(jobs, store, TOKEN)
        jobs._discard_job(TOKEN, original)
        with jobs._lock:
            self.assertIsNone(jobs._worker_job(TOKEN, original["cancellation"]))
            self.assertIs(jobs._worker_job(TOKEN, replacement["cancellation"]), replacement)
        self.assertIs(jobs._jobs[TOKEN], replacement)


CLOUD_TOKEN = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
OTHER_CLOUD_TOKEN = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"


class CloudDocumentCacheIsolation(unittest.TestCase):
    def adapter(self, store):
        from src.supabase_document_checkpoints import SupabaseDocumentCheckpointStore
        return SupabaseDocumentCheckpointStore(store)

    def launch(self, operation, release):
        outcome = Future()
        def run():
            try:
                outcome.set_result(operation())
            except BaseException as exc:
                outcome.set_exception(exc)
        thread = Thread(target=run, daemon=True)
        thread.start()
        def cleanup():
            release.set()
            thread.join(3)
            self.assertFalse(thread.is_alive())
        self.addCleanup(cleanup)
        return outcome

    def test_slow_cloud_kill_does_not_hold_other_documents_cache_lock(self):
        entered, release = Event(), Event()
        def kill(token):
            entered.set()
            if not release.wait(3):
                raise AssertionError("Cloud Kill gate was not released")
            return {"deleted": True}
        adapter = self.adapter(SimpleNamespace(owner_id="owner", kill_task=kill, get_task=lambda token: None))
        adapter._leases[CLOUD_TOKEN] = "lease"
        adapter._sources[CLOUD_TOKEN] = [("old", "source")]
        adapter._sources[OTHER_CLOUD_TOKEN] = [("other", "source")]
        deleting = self.launch(lambda: adapter.request_kill(CLOUD_TOKEN), release)
        self.assertTrue(entered.wait(1))
        reading = self.launch(lambda: adapter.load(OTHER_CLOUD_TOKEN), release)
        self.assertIsNone(reading.result(timeout=0.75), "Kill held the shared owner adapter lock")
        release.set()
        self.assertEqual(deleting.result(timeout=1), {"deleted": True})
        self.assertNotIn(CLOUD_TOKEN, adapter._leases)
        self.assertNotIn(CLOUD_TOKEN, adapter._sources)
        self.assertIn(OTHER_CLOUD_TOKEN, adapter._sources)
        self.assertEqual(adapter._cache_generation, 1)

    def test_load_started_before_kill_cannot_refill_deleted_task_cache(self):
        from src.document_checkpoints import _metadata
        from src.supabase_document_checkpoints import DOCUMENT_PAGE_KEY
        entered, release = Event(), Event()
        metadata = _metadata(checkpoint())
        task = dict(id=CLOUD_TOKEN, page_key=DOCUMENT_PAGE_KEY, owner_id="owner", status="incomplete",
                    metadata=metadata, settings=metadata["settings"], total_items=2,
                    fingerprint=metadata["fingerprint"], revision=1)
        deleted = Event()
        first = [True]
        def get_task(token):
            if first[0]:
                first[0] = False
                captured = deepcopy(task)
                entered.set()
                if not release.wait(3):
                    raise AssertionError("Cloud load gate was not released")
                return captured
            return None if deleted.is_set() else deepcopy(task)
        def kill(token):
            deleted.set()
            return {"deleted": True}
        adapter = self.adapter(SimpleNamespace(owner_id="owner", get_task=get_task, kill_task=kill))
        reading = self.launch(lambda: adapter.load(CLOUD_TOKEN), release)
        self.assertTrue(entered.wait(1))
        adapter.request_kill(CLOUD_TOKEN)
        release.set()
        self.assertIsNone(reading.result(timeout=1))
        self.assertNotIn(CLOUD_TOKEN, adapter._sources)
        self.assertNotIn(CLOUD_TOKEN, adapter._states)

    def test_late_write_acknowledgement_cannot_republish_killed_cache(self):
        from src.document_checkpoints import _metadata
        entered, release = Event(), Event()
        def save(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise AssertionError("Cloud write gate was not released")
            return {"revision": 2}
        store = SimpleNamespace(owner_id="owner", save_items=save, kill_task=lambda token: {"deleted": True})
        adapter = self.adapter(store)
        adapter._leases[CLOUD_TOKEN] = "lease"
        adapter._persisted[CLOUD_TOKEN] = {}
        writing = self.launch(lambda: adapter._write(CLOUD_TOKEN, _metadata(checkpoint()), {}), release)
        self.assertTrue(entered.wait(1))
        adapter.request_kill(CLOUD_TOKEN)
        release.set()
        with self.assertRaises(CheckpointError):
            writing.result(timeout=1)
        self.assertNotIn(CLOUD_TOKEN, adapter._persisted)
        self.assertNotIn(CLOUD_TOKEN, adapter._leases)

    def test_failed_cloud_kill_keeps_cache_and_lease_available_for_retry(self):
        def failed(token):
            raise CheckpointError("storage unavailable")
        adapter = self.adapter(SimpleNamespace(owner_id="owner", kill_task=failed))
        adapter._leases[CLOUD_TOKEN] = "lease"
        adapter._sources[CLOUD_TOKEN] = [("old", "source")]
        with self.assertRaises(CheckpointError):
            adapter.request_kill(CLOUD_TOKEN)
        self.assertEqual(adapter._cache_generation, 0)
        self.assertIn(CLOUD_TOKEN, adapter._leases)
        self.assertIn(CLOUD_TOKEN, adapter._sources)


if __name__ == "__main__":
    unittest.main()
