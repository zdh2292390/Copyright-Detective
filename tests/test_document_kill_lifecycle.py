"""Terminal document Kill retains checkpoints while removing resumability."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock
import tempfile
import time
import unittest
from uuid import uuid4

from src.analysis_checkpoints import AnalysisLeaseError
from src.document_analysis import new_analysis_state
from src.document_checkpoints import CheckpointError, DocumentCheckpointStore
from src.document_jobs import DocumentAnalysisJobs
from src.run_cancellation import cancellable_call
from src.supabase_document_checkpoints import SupabaseDocumentCheckpointStore

OWNER = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


def manifest(total=2, *, saved_first=False):
    pairs = [(f"prefix {index}", f"target {index}") for index in range(total)]
    settings = {
        "filename": "source.txt", "provider": "Local vLLM", "model": "test-model",
        "chunk_size": 200, "continuation_method": "Normal Continuation",
        "temperature": 0.7, "top_p": 0.9,
    }
    state = new_analysis_state("a" * 64, settings, total)
    state["owner_id"] = OWNER
    if saved_first:
        state["results"][0] = (*pairs[0], "already saved", {"ROUGE-L": 0.5})
        state["attempts"][0] = 1
    return state, pairs


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class LifecycleBackend:
    """Owner-bound RPC model: task columns win over stale worker metadata."""
    owner_id = OWNER

    def __init__(self):
        self.tasks = {}
        self.lock = RLock()
        self.kill_calls = []

    def get_task(self, token):
        with self.lock:
            task = self.tasks.get(token)
            return deepcopy({k: v for k, v in task.items() if k != "items"}) if task else None

    def create_task(self, page_key, settings, source, inputs, *, fingerprint, metadata):
        token = str(uuid4())
        inputs = list(inputs)
        self.tasks[token] = {
            "id": token, "owner_id": OWNER, "page_key": page_key,
            "settings": deepcopy(settings), "source": deepcopy(source), "fingerprint": fingerprint,
            "metadata": deepcopy(metadata), "total_items": len(inputs), "status": "queued",
            "revision": 0, "completed_items": 0, "failed_items": 0,
            "stop_requested": False, "kill_requested": False,
            "kill_requested_at": None, "killed_at": None,
            "lease_token": None, "lease_expires_at": None,
            "items": [{"item_index": i, "input": value, "status": "pending", "result": None,
                       "error": None, "attempts": 0} for i, value in enumerate(inputs)],
        }
        return self.get_task(token)

    def load_items(self, token, *, offset=0, limit=1000):
        with self.lock:
            return deepcopy(self.tasks[token]["items"][offset:offset + limit])

    def list_tasks(self, *, page_key, statuses, limit):
        return [self.get_task(token) for token, task in self.tasks.items()
                if task["page_key"] == page_key and task["status"] in statuses][:limit]

    def claim(self, token, *, ttl_seconds):
        with self.lock:
            task = self.tasks[token]
            if task["lease_token"] or task["kill_requested"]:
                raise AnalysisLeaseError("Task cannot be claimed.")
            task.update(lease_token=str(uuid4()), lease_expires_at="2099-01-01T00:00:00Z", status="running")
            task["revision"] += 1
            return self.get_task(token)

    def save_items(self, token, updates, *, metadata, status, lease_token):
        with self.lock:
            task = self.tasks[token]
            assert task["lease_token"] == lease_token
            for update in updates:
                task["items"][update["index"]].update({key: deepcopy(update[key]) for key in
                                                       ("status", "result", "error", "attempts")})
            task["metadata"] = deepcopy(metadata)
            task["status"] = "stopping" if task["kill_requested"] else status
            task["completed_items"] = sum(item["status"] == "complete" for item in task["items"])
            task["failed_items"] = sum(item["status"] == "failed" for item in task["items"])
            task["revision"] += 1
            return self.get_task(token)

    def heartbeat(self, token, lease_token, *, ttl_seconds):
        assert self.tasks[token]["lease_token"] == lease_token
        return self.get_task(token)

    def release(self, token, lease_token, *, status):
        assert status in {"incomplete", "complete", "killed"}, "Unsupported release lifecycle status"
        with self.lock:
            task = self.tasks[token]
            assert task["lease_token"] == lease_token
            task.update(lease_token=None, lease_expires_at=None,
                        status="killed" if task["kill_requested"] else status)
            if task["kill_requested"]:
                task["killed_at"] = datetime.now(timezone.utc).isoformat()
            task["revision"] += 1
            return self.get_task(token)

    def request_stop(self, token):
        self.tasks[token]["stop_requested"] = True
        return self.get_task(token)

    def kill_task(self, token):
        with self.lock:
            task = self.tasks[token]
            self.kill_calls.append(token)
            if task["status"] in {"complete", "killed"}:
                return self.get_task(token)
            task["metadata"]["stop_requested"] = True
            task.update(kill_requested=True, stop_requested=True,
                        kill_requested_at=task["kill_requested_at"] or datetime.now(timezone.utc).isoformat(),
                        status="stopping" if task["lease_token"] else "killed")
            if not task["lease_token"]:
                task["killed_at"] = datetime.now(timezone.utc).isoformat()
            task["revision"] += 1
            return self.get_task(token)


class DocumentKillLifecycleTests(unittest.TestCase):
    def test_paused_local_kill_retains_source_and_results_and_blocks_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentCheckpointStore(Path(directory))
            state, pairs = manifest(saved_first=True)
            jobs = DocumentAnalysisJobs(store, analyze_chunk=lambda *args: self.fail("Killed task made API call"))
            try:
                token = jobs.create(state, pairs, owner_id=OWNER, use_cloud=False)
                with self.assertRaises(CheckpointError):
                    jobs.force_kill(token, owner_id="another-owner")
                self.assertFalse(store.load(token)[0].get("kill_requested", False))
                self.assertTrue(jobs.force_kill(token, owner_id=OWNER))
                saved, source = store.load(token)
                self.assertEqual(saved["status"], "killed")
                self.assertTrue(saved["kill_requested"])
                self.assertTrue(saved["killed_at"])
                self.assertEqual(source, pairs)
                self.assertEqual(saved["results"], state["results"])
                self.assertFalse(jobs.submit(token, "", owner_id=OWNER))
                reopened = DocumentAnalysisJobs(store)
                try:
                    self.assertEqual(reopened.get(token, owner_id=OWNER)["status"], "killed")
                finally:
                    reopened.close()
            finally:
                jobs.close()

    def test_complete_task_is_unchanged_when_completion_wins_kill_race(self):
        for remote in (False, True):
            with self.subTest(remote=remote), tempfile.TemporaryDirectory() as directory:
                store = DocumentCheckpointStore(Path(directory))
                jobs = DocumentAnalysisJobs(store)
                if remote:
                    backend = LifecycleBackend()
                    jobs.configure_cloud_store(backend)
                state, pairs = manifest(1, saved_first=True)
                state["status"] = "complete"
                try:
                    token = jobs.create(state, pairs, owner_id=OWNER, use_cloud=remote)
                    self.assertFalse(jobs.force_kill(token, owner_id=OWNER))
                    saved = jobs.get(token, owner_id=OWNER)
                    self.assertEqual(saved["status"], "complete")
                    self.assertFalse(saved.get("kill_requested", False))
                    self.assertEqual(saved["results"], state["results"])
                    self.assertFalse(jobs.submit(token, "", owner_id=OWNER))
                finally:
                    jobs.close()

    def test_stale_sqlite_worker_snapshot_cannot_revive_killed_task(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentCheckpointStore(Path(directory))
            state, pairs = manifest()
            token = store.create(state, pairs)
            store.request_kill(token)
            state.update(status="running", stop_requested=False)
            store.save_progress(token, state)
            self.assertEqual(store.load(token)[0]["status"], "killed")
            store.save(token, state)
            self.assertTrue(store.load(token)[0]["kill_requested"])

    def test_running_local_kill_stays_stopping_until_accepted_request_drains(self):
        self._active_kill(remote=False)

    def test_running_cloud_kill_retains_lease_then_finalizes_columns_as_killed(self):
        self._active_kill(remote=True)

    def _active_kill(self, *, remote):
        entered, release = Event(), Event()
        calls = []
        def analyze(settings, key, upper, lower):
            calls.append(upper)
            def provider():
                entered.set()
                release.wait(5)
                return "late result", {"ROUGE-L": 0.5}
            return cancellable_call(provider)
        with tempfile.TemporaryDirectory() as directory:
            local = DocumentCheckpointStore(Path(directory))
            jobs = DocumentAnalysisJobs(local, analyze_chunk=analyze)
            backend = LifecycleBackend() if remote else None
            if remote:
                jobs.configure_cloud_store(backend)
            state, pairs = manifest()
            try:
                token = jobs.create(state, pairs, owner_id=OWNER, use_cloud=remote)
                self.assertTrue(jobs.submit(token, "", owner_id=OWNER))
                self.assertTrue(entered.wait(5))
                self.assertTrue(jobs.force_kill(token, owner_id=OWNER))
                self.assertTrue(wait_for(lambda: jobs._jobs.get(token, {}).get("cleanup_pending")))
                self.assertTrue(jobs.is_active(token, owner_id=OWNER))
                self.assertEqual(jobs.get(token, owner_id=OWNER)["status"], "stopping")
                self.assertFalse(jobs.submit(token, "", owner_id=OWNER))
                if remote:
                    self.assertEqual(backend.tasks[token]["status"], "stopping")
                    self.assertTrue(backend.tasks[token]["lease_token"])
                    self.assertEqual(jobs.list_saved(owner_id=OWNER), [])
                release.set()
                self.assertTrue(wait_for(lambda: not jobs.is_active(token, owner_id=OWNER)))
                saved = jobs.get(token, owner_id=OWNER)
                self.assertEqual(saved["status"], "killed")
                self.assertEqual(saved["results"], {})
                self.assertEqual(calls, [pairs[0][0]])
                self.assertFalse(jobs.submit(token, "", owner_id=OWNER))
                if remote:
                    self.assertTrue(backend.tasks[token]["killed_at"])
                    self.assertIsNone(backend.tasks[token]["lease_token"])
            finally:
                release.set()
                jobs.close()

    def test_cloud_inactive_kill_projects_columns_without_adding_kill_metadata(self):
        backend = LifecycleBackend()
        adapter = SupabaseDocumentCheckpointStore(backend)
        state, pairs = manifest()
        token = adapter.create(state, pairs)
        adapter.load(token)  # Exercise an already populated cache.
        metadata = deepcopy(backend.tasks[token]["metadata"])
        self.assertTrue(adapter.request_kill(token))
        metadata["stop_requested"] = True
        self.assertEqual(backend.tasks[token]["metadata"], metadata)
        self.assertNotIn("kill_requested", backend.tasks[token]["metadata"])
        for reader in (adapter, SupabaseDocumentCheckpointStore(backend)):
            restored, source = reader.load(token)
            self.assertEqual(restored["status"], "killed")
            self.assertTrue(restored["kill_requested"])
            self.assertEqual(source, pairs)
            self.assertEqual(reader.list_for_owner(), [])
        self.assertEqual(backend.kill_calls, [token])

    def test_release_maps_stopping_metadata_to_terminal_killed(self):
        backend = LifecycleBackend()
        adapter = SupabaseDocumentCheckpointStore(backend)
        state, pairs = manifest()
        token = adapter.create(state, pairs)
        self.assertTrue(adapter.claim(token))
        adapter.request_kill(token)
        self.assertEqual(adapter._identities[token]["status"], "stopping")
        adapter.release(token)
        self.assertEqual(backend.tasks[token]["status"], "killed")
        self.assertIsNone(backend.tasks[token]["lease_token"])

    def test_release_finalizes_kill_observed_only_in_save_response(self):
        backend = LifecycleBackend()
        adapter = SupabaseDocumentCheckpointStore(backend)
        state, pairs = manifest()
        token = adapter.create(state, pairs)
        self.assertTrue(adapter.claim(token))
        # Another session kills the task after this worker prepared its state.
        backend.kill_task(token)
        state["status"] = "running"
        response = adapter.save_progress(token, state)
        self.assertEqual(response["status"], "stopping")
        self.assertFalse(adapter._identities[token].get("kill_requested", False))
        adapter.release(token)
        self.assertEqual(backend.tasks[token]["status"], "killed")
        self.assertIsNone(backend.tasks[token]["lease_token"])
        self.assertEqual(adapter.load(token)[0]["status"], "killed")

    def test_cloud_cache_projects_lifecycle_even_without_revision_change(self):
        backend = LifecycleBackend()
        adapter = SupabaseDocumentCheckpointStore(backend)
        state, pairs = manifest()
        token = adapter.create(state, pairs)
        adapter.load(token)
        backend.tasks[token].update(status="killed", kill_requested=True, stop_requested=True,
                                    killed_at="2026-10-13T15:00:00Z")
        restored, _ = adapter.load(token)
        self.assertEqual(restored["status"], "killed")
        self.assertTrue(restored["kill_requested"])
        self.assertEqual(restored["killed_at"], "2026-10-13T15:00:00Z")

    def test_resumable_force_stop_does_not_set_terminal_kill(self):
        entered, release = Event(), Event()
        calls = []
        def analyze(settings, key, upper, lower):
            calls.append(upper)
            if len(calls) == 1:
                cancellable_call(lambda: (entered.set(), release.wait(5)))
            return "generated", {"ROUGE-L": 0.5}
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentCheckpointStore(Path(directory))
            jobs = DocumentAnalysisJobs(store, analyze_chunk=analyze)
            state, pairs = manifest(1)
            try:
                token = jobs.create(state, pairs, owner_id=OWNER, use_cloud=False)
                jobs.submit(token, "", owner_id=OWNER)
                self.assertTrue(entered.wait(5))
                jobs.force_stop(token, owner_id=OWNER)
                release.set()
                self.assertTrue(wait_for(lambda: not jobs.is_active(token, owner_id=OWNER)))
                self.assertFalse(store.load(token)[0].get("kill_requested", False))
                self.assertTrue(jobs.submit(token, "", owner_id=OWNER))
                self.assertTrue(wait_for(lambda: not jobs.is_active(token, owner_id=OWNER)))
                self.assertEqual(store.load(token)[0]["status"], "complete")
            finally:
                release.set()
                jobs.close()


if __name__ == "__main__":
    unittest.main()
