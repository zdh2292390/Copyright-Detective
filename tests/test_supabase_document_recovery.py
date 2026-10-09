"""Document cloud recovery and fenced worker regressions; no external calls."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, RLock
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4
import unittest

from src.analysis_checkpoints import AnalysisCheckpointError, AnalysisLeaseError
from src.document_analysis import analysis_fingerprint, new_analysis_state
from src.document_checkpoints import CheckpointError, DocumentCheckpointStore
from src.document_jobs import DocumentAnalysisJobs
from src.supabase_document_checkpoints import SupabaseDocumentCheckpointStore

OWNER = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"


def fixture(count=110, completed=104, owner=OWNER):
    settings = {
        "filename": "original.txt", "provider": "Google Gemini", "model": "gemini-3.5-flash",
        "chunk_size": 200, "overlap": 50, "continuation_method": "Normal Continuation",
        "temperature": 0.7, "top_p": 0.9, "base_url": None,
    }
    state = new_analysis_state(analysis_fingerprint("original document", settings), settings, count)
    state["owner_id"] = owner
    pairs = [(str(index), "original target") for index in range(count)]
    for index in range(completed):
        state["results"][index] = (*pairs[index], f"saved continuation {index}", {"rouge_l": 0.1})
        state["attempts"][index] = 1
    return state, pairs


class MemoryBackend:
    """Shared durable records; mutations model the RPC transaction contract."""
    def __init__(self):
        self.lock = RLock()
        self.tasks = {}
        self.items = {}
        self.writes = []
        self.claims = 0
        self.heartbeats = 0
        self.fail_next_success = False
        self.fail_all_writes = False
        self.read_hook = None


class MemoryCloudStore:
    def __init__(self, backend, owner_id=OWNER):
        self.backend = backend
        self.owner_id = owner_id
        self.item_reads = 0

    def _owned(self, task_id):
        task = self.backend.tasks.get(task_id)
        return task if task and task["owner_id"] == self.owner_id else None

    @staticmethod
    def _now():
        return datetime.now(timezone.utc)

    def _lease(self, task_id, lease_token):
        task = self._owned(task_id)
        if not task or task["lease_token"] != lease_token or datetime.fromisoformat(task["lease_expires_at"]) <= self._now():
            raise AnalysisLeaseError("The worker lease expired or is held by another worker.")
        return task

    def create_task(self, page_key, settings, source, work_items, *, fingerprint, metadata, **kwargs):
        task_id = str(uuid4())
        with self.backend.lock:
            task = {
                "id": task_id, "owner_id": self.owner_id, "page_key": page_key,
                "settings": deepcopy(settings), "source": deepcopy(source), "metadata": deepcopy(metadata),
                "fingerprint": fingerprint, "status": "queued", "total_items": len(work_items),
                "completed_items": 0, "failed_items": 0, "stop_requested": False,
                "lease_token": None, "lease_expires_at": None, "lease_generation": 0,
                "updated_at": self._now().isoformat(), "revision": 1,
            }
            self.backend.tasks[task_id] = task
            self.backend.items[task_id] = [{"item_index": index, "input": deepcopy(value), "status": "pending", "result": None, "error": None, "attempts": 0} for index, value in enumerate(work_items)]
            return deepcopy(task)

    def get_task(self, task_id):
        with self.backend.lock:
            return deepcopy(self._owned(task_id))

    def list_tasks(self, *, page_key=None, statuses=None, offset=0, limit=50):
        with self.backend.lock:
            tasks = [task for task in self.backend.tasks.values() if task["owner_id"] == self.owner_id and (page_key is None or task["page_key"] == page_key) and (statuses is None or task["status"] in statuses)]
            return deepcopy(tasks[offset:offset + limit])

    def load_items(self, task_id, *, offset=0, limit=200):
        with self.backend.lock:
            self.item_reads += 1
            if not self._owned(task_id):
                return []
            result = deepcopy(self.backend.items[task_id][offset:offset + limit])
            if self.backend.read_hook:
                hook = self.backend.read_hook
                self.backend.read_hook = None
                hook(task_id)
            return result

    def claim(self, task_id, *, ttl_seconds=180):
        with self.backend.lock:
            task = self._owned(task_id)
            if not task:
                raise AnalysisCheckpointError("Task not found")
            if task["status"] == "complete" or task["lease_token"] and datetime.fromisoformat(task["lease_expires_at"]) > self._now():
                raise AnalysisLeaseError("Task is busy or already complete")
            task.update(status="running", stop_requested=False, lease_token=str(uuid4()), lease_expires_at=(self._now() + timedelta(seconds=ttl_seconds)).isoformat())
            task["lease_generation"] += 1
            task["revision"] += 1
            self.backend.claims += 1
            return deepcopy(task)

    def heartbeat(self, task_id, lease_token, *, ttl_seconds=180):
        with self.backend.lock:
            task = self._lease(task_id, lease_token)
            self.backend.heartbeats += 1
            task["lease_expires_at"] = (self._now() + timedelta(seconds=ttl_seconds)).isoformat()
            task["revision"] += 1
            return deepcopy(task)

    def save_items(self, task_id, updates, *, metadata=None, status=None, lease_token):
        with self.backend.lock:
            old = self._lease(task_id, lease_token)
            if self.backend.fail_all_writes or self.backend.fail_next_success and any(update["status"] == "complete" for update in updates):
                self.backend.fail_next_success = False
                raise AnalysisCheckpointError("Simulated cloud checkpoint outage")
            task = deepcopy(old)
            items = deepcopy(self.backend.items[task_id])
            for update in updates:
                index = update["index"]
                if items[index]["status"] == "complete" and update["status"] != "complete":
                    raise AnalysisCheckpointError("Saved successful results cannot be cleared")
                items[index].update({key: deepcopy(value) for key, value in update.items() if key != "index"})
            task["completed_items"] = sum(item["status"] == "complete" for item in items)
            task["failed_items"] = sum(item["status"] == "failed" for item in items)
            if status == "complete" and (task["completed_items"] != task["total_items"] or task["failed_items"]):
                raise AnalysisCheckpointError("All chunks must succeed before completion")
            if metadata is not None:
                task["metadata"] = deepcopy(metadata)
            if status is not None:
                task["status"] = status
            task["revision"] += 1
            self.backend.tasks[task_id] = task
            self.backend.items[task_id] = items
            self.backend.writes.append(deepcopy(updates))
            return deepcopy(task)

    def release(self, task_id, lease_token, *, status="incomplete", metadata=None):
        with self.backend.lock:
            task = self._lease(task_id, lease_token)
            if status == "complete" and task["completed_items"] != task["total_items"]:
                raise AnalysisCheckpointError("Incomplete task")
            task.update(status=status, lease_token=None, lease_expires_at=None)
            task["revision"] += 1
            return deepcopy(task)

    def request_stop(self, task_id):
        with self.backend.lock:
            task = self._owned(task_id)
            task["stop_requested"] = True
            task["revision"] += 1
            return deepcopy(task)

    def delete(self, task_id):
        with self.backend.lock:
            task = self._owned(task_id)
            if task and task["lease_token"] and datetime.fromisoformat(task["lease_expires_at"]) > self._now():
                raise AnalysisLeaseError("Stop active task first")
            self.backend.tasks.pop(task_id, None)
            self.backend.items.pop(task_id, None)


class SupabaseDocumentRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.backend = MemoryBackend()
        self.cloud = MemoryCloudStore(self.backend)
        self.temp = TemporaryDirectory()
        self.managers = []
        self.events = []

    def tearDown(self):
        for event in self.events:
            event.set()
        for manager in reversed(self.managers):
            manager.close()
        self.temp.cleanup()

    def manager(self, analyze=None, *, owner=OWNER, **kwargs):
        manager = DocumentAnalysisJobs(DocumentCheckpointStore(Path(self.temp.name) / str(len(self.managers))), analyze_chunk=analyze or (lambda settings, key, upper, lower: ("generated", {"rouge_l": 0.2})), **kwargs)
        manager.configure_cloud_store(MemoryCloudStore(self.backend, owner))
        self.managers.append(manager)
        return manager

    def finish(self, manager, token):
        deadline = time.monotonic() + 10
        while manager.is_running(token) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(manager.is_running(token))
        return manager.get(token, owner_id=OWNER)

    def test_fresh_process_restores_104_results_and_resumes_without_source_upload(self):
        state, pairs = fixture()
        state["api_key"] = "runtime-secret"
        state["settings"]["api_key"] = "runtime-secret"
        token = SupabaseDocumentCheckpointStore(self.cloud).create(state, pairs)
        self.assertNotIn("runtime-secret", json.dumps(self.backend.tasks))
        calls = []
        def analyze(settings, key, upper, lower):
            calls.append((int(upper), settings["model"], key))
            return "new continuation", {"rouge_l": 0.2}
        fresh = self.manager(analyze)
        restored = fresh.get(token, owner_id=OWNER)
        self.assertEqual(len(restored["results"]), 104)
        self.assertEqual(restored["settings"]["chunk_size"], 200)
        self.assertTrue(fresh.submit(token, "new-runtime-key", owner_id=OWNER))
        complete = self.finish(fresh, token)
        self.assertEqual(complete["status"], "complete")
        self.assertEqual([call[0] for call in calls], list(range(104, 110)))
        self.assertTrue(all(call[1:] == ("gemini-3.5-flash", "new-runtime-key") for call in calls))
        self.assertEqual(list(complete["results"]), list(range(110)))
        self.assertNotIn("new-runtime-key", json.dumps(self.backend.tasks) + json.dumps(self.backend.items))

    def test_peer_worker_is_not_interrupted_or_duplicated_and_can_be_stopped(self):
        state, pairs = fixture()
        token = SupabaseDocumentCheckpointStore(self.cloud).create(state, pairs)
        entered, release = Event(), Event()
        self.events.append(release)
        calls = []
        def analyze(settings, key, upper, lower):
            calls.append(int(upper)); entered.set(); release.wait(5)
            return "generated", {"rouge_l": 0.2}
        first, peer = self.manager(analyze), self.manager()
        self.assertTrue(first.submit(token, "key", owner_id=OWNER))
        self.assertTrue(entered.wait(2))
        self.assertEqual(peer.get(token, owner_id=OWNER)["status"], "running")
        self.assertTrue(peer.is_active(token, owner_id=OWNER))
        self.assertFalse(peer.submit(token, "key", owner_id=OWNER))
        self.assertTrue(peer.stop(token, owner_id=OWNER))
        release.set()
        finished = self.finish(first, token)
        self.assertEqual(finished["status"], "incomplete")
        self.assertEqual(len(finished["results"]), 105)
        self.assertEqual(calls, [104])

    def test_heartbeat_renews_lease_and_honors_stop_while_provider_call_is_blocked(self):
        state, pairs = fixture(2, 0)
        token = SupabaseDocumentCheckpointStore(self.cloud).create(state, pairs)
        entered, release = Event(), Event()
        self.events.append(release)
        def analyze(settings, key, upper, lower):
            entered.set(); release.wait(5)
            return "generated", {"rouge_l": 0.2}
        manager, peer = self.manager(analyze), self.manager()
        with patch("src.document_jobs.LEASE_HEARTBEAT_SECONDS", 0.01):
            self.assertTrue(manager.submit(token, "key", owner_id=OWNER))
            self.assertTrue(entered.wait(2))
            initial = self.backend.tasks[token]["lease_expires_at"]
            self.assertTrue(peer.stop(token, owner_id=OWNER))
            deadline = time.monotonic() + 2
            while not manager._jobs[token]["stop"].is_set() and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(manager._jobs[token]["stop"].is_set())
            self.assertGreater(self.backend.heartbeats, 0)
            self.assertGreater(self.backend.tasks[token]["lease_expires_at"], initial)
            release.set()
            self.assertEqual(len(self.finish(manager, token)["results"]), 1)

    def test_unchanged_revision_reuses_snapshot_without_downloading_source_again(self):
        state, pairs = fixture()
        store = SupabaseDocumentCheckpointStore(self.cloud)
        token = store.create(state, pairs)
        first, _ = store.load(token)
        reads = self.cloud.item_reads
        first["results"].clear()
        second, saved_pairs = store.load(token)
        self.assertEqual(len(second["results"]), 104)
        self.assertEqual(saved_pairs, pairs)
        self.assertEqual(self.cloud.item_reads, reads)

    def test_expired_lease_fences_stale_writer(self):
        state, pairs = fixture(2, 0)
        first = SupabaseDocumentCheckpointStore(self.cloud)
        token = first.create(state, pairs)
        self.assertTrue(first.claim(token)); first.load(token)
        self.backend.tasks[token]["lease_expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        second = SupabaseDocumentCheckpointStore(MemoryCloudStore(self.backend))
        self.assertTrue(second.claim(token)); second.load(token)
        state["status"] = "running"
        state["results"][0] = (*pairs[0], "stale result", {"rouge_l": 0.1})
        with self.assertRaises(CheckpointError):
            first.save_progress(token, state, 0)
        self.assertIsNone(self.backend.items[token][0]["result"])
        second.release(token)

    def test_progress_writes_only_changed_chunk_and_complete_guard_rolls_back(self):
        state, pairs = fixture()
        store = SupabaseDocumentCheckpointStore(self.cloud)
        token = store.create(state, pairs)
        self.assertTrue(store.claim(token)); state, pairs = store.load(token)
        reads = self.cloud.item_reads
        state["status"] = "running"
        state["results"][104] = (*pairs[104], "generated", {"rouge_l": 0.2})
        store.save_progress(token, state, 104)
        self.assertEqual([row["index"] for row in self.backend.writes[-1]], [104])
        self.assertEqual(self.cloud.item_reads, reads)
        store.save_progress(token, state)
        self.assertEqual(self.backend.writes[-1], [])
        state["status"] = "complete"
        before = deepcopy(self.backend.tasks[token])
        with self.assertRaises(CheckpointError):
            store.save_progress(token, state, 104)
        self.assertEqual(self.backend.tasks[token], before)
        store.release(token)

    def test_storage_fault_drains_parallel_successes_in_one_atomic_recovery_write(self):
        state, pairs = fixture(4, 0)
        token = SupabaseDocumentCheckpointStore(self.cloud).create(state, pairs)
        release, all_entered = Event(), Event()
        self.events.append(release)
        calls, lock = [], RLock()
        def analyze(settings, key, upper, lower):
            with lock:
                calls.append(int(upper))
                if len(calls) == 4:
                    all_entered.set()
            release.wait(5)
            return "generated", {"rouge_l": 0.2}
        manager = self.manager(analyze, chunk_concurrency=4, parallel_threshold=1)
        self.backend.fail_next_success = True
        self.assertTrue(manager.submit(token, "key", owner_id=OWNER))
        self.assertTrue(all_entered.wait(2)); release.set()
        finished = self.finish(manager, token)
        self.assertEqual(finished["status"], "incomplete")
        self.assertEqual(len(finished["results"]), 4)
        self.assertEqual({row["index"] for row in self.backend.writes[-1]}, set(range(4)))
        self.assertEqual(self.backend.tasks[token]["completed_items"], 4)

    def test_owner_scoping_and_old_local_token_survive_cloud_configuration(self):
        manager = self.manager()
        state, pairs = fixture()
        local_token = manager.create(state, pairs, owner_id=OWNER, use_cloud=False)
        self.assertEqual(len(local_token), 64)
        self.assertEqual(len(manager.get(local_token, owner_id=OWNER)["results"]), 104)
        token = manager.create(state, pairs, owner_id=OWNER)
        other = self.manager(owner=OTHER)
        self.assertEqual(other.list_saved(owner_id=OTHER), [])
        self.assertIsNone(other.get(token, owner_id=OTHER))
        with self.assertRaises(CheckpointError):
            other.get(token)
        self.assertEqual(len(manager.list_saved(owner_id=OWNER)), 1)

    def test_manifest_corruption_and_revision_race_are_distinguished(self):
        state, pairs = fixture(2, 0)
        store = SupabaseDocumentCheckpointStore(self.cloud)
        token = store.create(state, pairs)
        self.backend.read_hook = lambda task_id: self.backend.tasks[task_id].update(revision=self.backend.tasks[task_id]["revision"] + 1)
        self.assertEqual(store.load(token)[1], pairs)
        self.assertEqual(self.cloud.item_reads, 2)
        self.backend.items[token][0]["input"]["upper_text"] = "corrupt source"
        self.backend.tasks[token]["revision"] += 1
        with self.assertRaisesRegex(CheckpointError, "integrity"):
            store.load(token)

    def test_missing_current_auth_cannot_use_cached_owner_store(self):
        import src.pages.document_memorization_detection as page
        manager = self.manager()
        state, pairs = fixture(2, 0)
        token = manager.create(state, pairs, owner_id=OWNER)
        ui = SimpleNamespace(session_state={"user_id": OWNER, "pdf_analysis_state": state, "pdf_analysis_results": ["private"], "_pdf_verified_cloud_owner": OWNER}, query_params={"document_analysis": token}, error=Mock())
        with patch.object(page, "DOCUMENT_JOBS", manager), patch.object(page, "st", ui), patch("src.resumable_analysis.get_cloud_store_for_current_user", return_value=None):
            self.assertFalse(page._prepare_document_cloud())
            self.assertIsNone(page._restore_document_job())
        self.assertNotIn("pdf_analysis_results", ui.session_state)
        self.assertNotIn("_pdf_verified_cloud_owner", ui.session_state)

    def test_authentication_failure_clears_previous_cloud_verification(self):
        import src.pages.document_memorization_detection as page
        from src.resumable_analysis import AnalysisCheckpointError as CloudSetupError
        ui = SimpleNamespace(session_state={"_pdf_verified_cloud_owner": OWNER})
        with patch.object(page, "st", ui), patch("src.resumable_analysis.get_cloud_store_for_current_user", side_effect=CloudSetupError("Sign in again")):
            with self.assertRaisesRegex(CheckpointError, "Sign in again"):
                page._prepare_document_cloud()
        self.assertNotIn("_pdf_verified_cloud_owner", ui.session_state)

    def test_cloud_setup_failure_does_not_create_a_local_task(self):
        import src.pages.document_memorization_detection as page
        manager = self.manager()
        ui = SimpleNamespace(error=Mock())
        with patch.object(page, "DOCUMENT_JOBS", manager), patch.object(page, "st", ui), patch.object(page, "_prepare_document_cloud", side_effect=AnalysisCheckpointError("Apply cloud schema")):
            page.render_pdf_analysis_page("key", "gemini-3.5-flash", "Google Gemini")
        self.assertTrue(ui.error.called)
        self.assertFalse(manager.store.db_path.exists())

    def test_streamlit_account_list_opens_and_resumes_without_upload(self):
        from streamlit.testing.v1 import AppTest
        import src.resumable_analysis as recovery
        state, pairs = fixture()
        token = SupabaseDocumentCheckpointStore(self.cloud).create(state, pairs)
        app_source = """
import tempfile
import streamlit as st
import src.pages.document_memorization_detection as page
from src.document_jobs import DocumentAnalysisJobs
from src.document_checkpoints import DocumentCheckpointStore
st.session_state['user_id'] = '11111111-1111-4111-8111-111111111111'
if 'mock_manager' not in st.session_state:
    st.session_state['mock_temp'] = tempfile.TemporaryDirectory()
    st.session_state['mock_calls'] = []
    # Workers avoid Streamlit state; capture the plain list now.
    calls = st.session_state['mock_calls']
    def analyze(settings, key, upper, lower):
        calls.append(int(upper))
        return 'generated', {'rouge_l': 0.2}
    st.session_state['mock_manager'] = DocumentAnalysisJobs(DocumentCheckpointStore(st.session_state['mock_temp'].name), analyze_chunk=analyze)
page.DOCUMENT_JOBS = st.session_state['mock_manager']
page._list_example_documents = lambda: []
page.render_prompt_preview = lambda prompt: None
def render(results, document, model, **kwargs):
    st.session_state['mock_rendered'] = (len(results), document.name, model)
    st.text('Restored results: ' + str(len(results)))
page.render_pdf_results_section = render
page.render_pdf_analysis_page('new-key', 'different-current-model', 'Google Gemini')
"""
        with patch.object(recovery, "get_cloud_store_for_current_user", return_value=self.cloud):
            app = AppTest.from_string(app_source, default_timeout=30).run()
            manager = app.session_state["mock_manager"]
            self.managers.append(manager)
            self.addCleanup(app.session_state["mock_temp"].cleanup)
            self.assertEqual(len(app.exception), 0)
            app.selectbox(key="pdf_cloud_recovery_selection").select(token).run()
            app.button(key="open_saved_pdf_analysis").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["mock_rendered"], (104, "original.txt", "gemini-3.5-flash"))
            self.assertFalse(app.button(key="resume_saved_pdf_analysis").disabled)
            app.button(key="resume_saved_pdf_analysis").click().run()
            self.finish(manager, token)
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["mock_calls"], list(range(104, 110)))
            self.assertEqual(app.session_state["mock_rendered"][0], 110)


if __name__ == "__main__":
    unittest.main()
