"""Fresh-session replay, fencing failures and secret-free recovery snapshots."""
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
from uuid import uuid4

from src.resumable_analysis import (
    AnalysisCheckpointError, AnalysisResumeMismatch, CallJournal, PageRun,
    checkpoint_call, decode, encode, get_cloud_store_for_current_user,
    journal_scope, snapshot_session,
)


class MemoryStore:
    def __init__(self):
        self.owner_id = str(uuid4())
        self.tasks = {}
        self.items = {}
        self.fail_append = False
        self.fail_save = False

    def create_task(self, *, page_key="test", settings=None, source=None, work_items=None, dynamic_items=True):
        task_id = str(uuid4())
        self.tasks[task_id] = {"id": task_id, "owner_id": self.owner_id, "page_key": page_key,
                               "settings": settings or {}, "source": source or {}, "metadata": {},
                               "status": "queued", "total_items": 0, "completed_items": 0, "dynamic_items": dynamic_items}
        self.items[task_id] = []
        return deepcopy(self.tasks[task_id])

    def get_task(self, task_id):
        return deepcopy(self.tasks.get(task_id))

    def list_tasks(self, *, page_key=None, limit=50, summary_only=False):
        return [deepcopy(task) for task in self.tasks.values() if page_key is None or task["page_key"] == page_key][:limit]

    def load_items(self, task_id, *, offset=0, limit=200):
        return deepcopy(self.items[task_id][offset:offset + limit])

    def claim(self, task_id, *, ttl_seconds=180):
        task = self.tasks[task_id]
        if task.get("lease_token"):
            raise RuntimeError("lease busy")
        task.update(status="running", lease_token=str(uuid4()))
        return deepcopy(task)

    def heartbeat(self, task_id, lease_token, *, ttl_seconds=180):
        task = self.tasks[task_id]
        if task.get("lease_token") != lease_token:
            raise RuntimeError("fenced")
        return deepcopy(task)

    def append_item(self, task_id, index, payload, *, lease_token):
        self.heartbeat(task_id, lease_token)
        if self.fail_append:
            raise RuntimeError("database unavailable")
        assert index == len(self.items[task_id])
        self.items[task_id].append({"item_index": index, "input": deepcopy(payload), "status": "pending", "attempts": 0})
        self.tasks[task_id]["total_items"] += 1
        return self.get_task(task_id)

    def save_item(self, task_id, index, *, result=None, status="complete", attempts=1, error=None, lease_token=None):
        self.heartbeat(task_id, lease_token)
        if self.fail_save:
            raise RuntimeError("database unavailable")
        self.items[task_id][index].update(status=status, result=deepcopy(result), attempts=attempts, error=error)
        self.tasks[task_id]["completed_items"] = sum(row["status"] == "complete" for row in self.items[task_id])
        return self.get_task(task_id)

    def release(self, task_id, lease_token, *, status="incomplete", metadata=None):
        self.heartbeat(task_id, lease_token)
        task = self.tasks[task_id]
        if status == "complete":
            assert all(item["status"] == "complete" for item in self.items[task_id])
        task.update(status=status, lease_token=None)
        task["metadata"].update(metadata or {})
        return self.get_task(task_id)


class JournalTests(TestCase):
    def setUp(self):
        self.store = MemoryStore()
        self.task = self.store.create_task()
        self.journal = CallJournal(self.store, self.task, heartbeat=False)

    def tearDown(self):
        if not self.journal.closed:
            self.journal.finish(False)

    def test_fresh_engine_replays_success_then_retries_only_unfinished(self):
        first = Mock(return_value=("answer 1", [{"logprob": -0.4}]))
        self.journal.call("answer", {"prompt": "one", "model": "fixed"}, first)
        with self.assertRaises(RuntimeError):
            self.journal.call("answer", {"prompt": "two", "model": "fixed"}, Mock(side_effect=RuntimeError("server stopped")))
        self.journal.finish(False)
        fresh = CallJournal(self.store, self.store.get_task(self.task["id"]), heartbeat=False)
        replay = Mock(side_effect=AssertionError("saved call must not execute"))
        self.assertEqual(fresh.call("answer", {"prompt": "one", "model": "fixed"}, replay), ("answer 1", [{"logprob": -0.4}]))
        self.assertEqual(fresh.call("answer", {"prompt": "two", "model": "fixed"}, lambda: "answer 2"), "answer 2")
        fresh.finish(True)
        self.assertEqual(self.store.get_task(self.task["id"])["status"], "complete")
        replay.assert_not_called()
        first.assert_called_once()

    def test_request_change_never_reuses_or_calls_new_model(self):
        self.journal.call("answer", {"model": "original"}, lambda: "saved")
        self.journal.finish(False)
        fresh = CallJournal(self.store, self.store.get_task(self.task["id"]), heartbeat=False)
        invoke = Mock()
        with self.assertRaises(AnalysisResumeMismatch):
            fresh.call("answer", {"model": "changed"}, invoke)
        with self.assertRaises(AnalysisCheckpointError):
            fresh.call("answer", {"model": "original"}, invoke)
        invoke.assert_not_called()
        fresh.finish(False)

    def test_database_failure_before_request_blocks_api(self):
        self.store.fail_append = True
        invoke = Mock()
        with self.assertRaises(AnalysisCheckpointError):
            self.journal.call("answer", {}, invoke)
        invoke.assert_not_called()
        self.store.fail_append = False
        with self.assertRaises(AnalysisCheckpointError):
            self.journal.call("answer", {}, invoke)
        invoke.assert_not_called()

    def test_database_failure_after_response_latches_paid_calls(self):
        self.store.fail_save = True
        first = Mock(return_value="generated")
        with self.assertRaises(AnalysisCheckpointError):
            self.journal.call("answer", {}, first)
        more = Mock()
        with self.assertRaises(AnalysisCheckpointError):
            self.journal.call("answer", {}, more)
        first.assert_called_once()
        more.assert_not_called()

    def test_error_response_keeps_task_incomplete_and_can_be_retried(self):
        self.assertEqual(self.journal.call("answer", {}, lambda: "Error: unavailable"), "Error: unavailable")
        self.journal.finish(True)
        self.assertEqual(self.store.get_task(self.task["id"])["status"], "incomplete")
        fresh = CallJournal(self.store, self.store.get_task(self.task["id"]), heartbeat=False)
        self.assertEqual(fresh.call("answer", {}, lambda: "recovered"), "recovered")
        fresh.finish(True)

    def test_nested_wrappers_save_one_outer_result(self):
        with journal_scope(self.journal):
            value = checkpoint_call("outer", {}, lambda: checkpoint_call("inner", {}, lambda: {"choice": "A"}))
        self.assertEqual(value, {"choice": "A"})
        self.assertEqual(len(self.store.items[self.task["id"]]), 1)

    def test_completed_result_rebuild_is_read_only(self):
        self.journal.call("answer", {}, lambda: "saved")
        self.journal.finish(True)
        fresh = CallJournal(self.store, self.store.get_task(self.task["id"]), heartbeat=False)
        invoke = Mock()
        self.assertEqual(fresh.call("answer", {}, invoke), "saved")
        with self.assertRaises(AnalysisResumeMismatch):
            fresh.call("answer", {}, invoke)
        invoke.assert_not_called()
        fresh.finish(False)

    def test_lease_failure_does_not_start_next_request(self):
        self.store.tasks[self.task["id"]]["lease_token"] = "a newer worker"
        invoke = Mock()
        with self.assertRaises(AnalysisCheckpointError):
            self.journal.call("answer", {}, invoke)
        invoke.assert_not_called()
        self.journal.closed = True  # Test deliberately fenced this writer.

    def test_source_snapshot_excludes_credentials_uploads_and_buttons(self):
        snapshot = snapshot_session({
            "sidebar_openai_api_key": "key", "access_token": "session",
            "qa_input_text": "source", "qa_num_eval_runs": 3,
            "qa_upload_widget": object(), "qa_run_button": False,
            "qa_enable_llm_judge": True, "sc_upload_source_text": "decoded",
            "min_k_upload_document_uploaded_text": "decoded min k",
            "qa_bad_metadata": {"api_key": "secret"},
        })
        self.assertEqual(set(snapshot), {"qa_input_text", "qa_num_eval_runs", "qa_enable_llm_judge", "sc_upload_source_text", "min_k_upload_document_uploaded_text"})
        self.assertNotIn("secret", str(snapshot))
        self.assertEqual(decode(encode(("text", None))), ("text", None))

    def test_no_context_preserves_legacy_behavior(self):
        invoke = Mock(return_value="legacy")
        self.assertEqual(checkpoint_call("answer", {"api_key": "only ephemeral"}, invoke), "legacy")
        invoke.assert_called_once()

    def test_create_failure_cannot_silently_fall_back(self):
        scope = PageRun("test", {"qa_input_text": "source"})
        scope.trigger = "run"
        broken = Mock()
        broken.create_task.side_effect = RuntimeError("schema missing")
        with patch("src.resumable_analysis.get_cloud_store_for_current_user", return_value=broken):
            with self.assertRaises(AnalysisCheckpointError):
                scope.ensure_journal()
            broken.create_task.side_effect = None
            with self.assertRaises(AnalysisCheckpointError):
                scope.ensure_journal()
        self.assertEqual(broken.create_task.call_count, 1)


class IdentityTests(TestCase):
    def test_cloud_owner_is_verified_by_server_not_stale_session_state(self):
        state = {"access_token": "ephemeral", "user_id": "stale"}
        authenticated = SimpleNamespace(auth=SimpleNamespace(get_user=lambda: SimpleNamespace(user=SimpleNamespace(id="actual"))))
        with patch("streamlit.session_state", state), patch("src.supabase_client.get_secret", side_effect=lambda name, default='': {"SUPABASE_URL": "https://db.test", "SUPABASE_SERVICE_ROLE_KEY": "server-only"}.get(name, default)), patch("src.supabase_client.get_authenticated_client", return_value=authenticated):
            with self.assertRaises(AnalysisCheckpointError):
                get_cloud_store_for_current_user()

    def test_signed_out_session_never_constructs_service_store(self):
        with patch("streamlit.session_state", {"user_id": "stale"}), patch("src.supabase_client.get_secret", return_value="configured"), patch("src.supabase_client.get_authenticated_client") as verify:
            self.assertIsNone(get_cloud_store_for_current_user())
            verify.assert_not_called()


class BackgroundCheckpointFailureTests(TestCase):
    def test_executor_failure_releases_ui_even_if_cloud_release_fails(self):
        from src import background_jobs
        key = f"test-submit-fail:{uuid4()}"
        journal = Mock()
        journal.finish.side_effect = AnalysisCheckpointError("database unavailable")
        with patch("src.resumable_analysis.prepare_background_journal", return_value=journal), patch.object(background_jobs._EXECUTOR, "submit", side_effect=RuntimeError("executor shutdown")):
            self.assertTrue(background_jobs.submit_background_job(key, "test", lambda report: None))
        self.assertEqual(background_jobs.get_background_job(key)["status"], "failed")
        background_jobs.forget_background_job(key)

    def test_cancelled_submission_releases_ui_when_cloud_release_fails(self):
        from concurrent.futures import Future
        from src import background_jobs
        key = f"test-cancel:{uuid4()}"
        journal = Mock()
        journal.finish.side_effect = AnalysisCheckpointError("database unavailable")
        future = Future()
        future.cancel()
        with patch("src.resumable_analysis.prepare_background_journal", return_value=journal), patch.object(background_jobs._EXECUTOR, "submit", return_value=future):
            self.assertTrue(background_jobs.submit_background_job(key, "test", lambda report: None))
        self.assertEqual(background_jobs.get_background_job(key)["status"], "failed")
        background_jobs.forget_background_job(key)

    def test_background_prepare_storage_failure_latches_page(self):
        from src.resumable_analysis import page_analysis_scope, prepare_background_journal, register_ui_run
        store = Mock()
        store.create_task.side_effect = RuntimeError("missing schema")
        with patch("src.resumable_analysis.get_cloud_store_for_current_user", return_value=store):
            with page_analysis_scope("test", {}) as scope:
                register_ui_run("test", "run")
                with self.assertRaises(AnalysisCheckpointError):
                    prepare_background_journal("worker", "test")
                store.create_task.side_effect = None
                with self.assertRaises(AnalysisCheckpointError):
                    prepare_background_journal("worker", "test")
                self.assertTrue(scope.broken)
        self.assertEqual(store.create_task.call_count, 1)


class RemoteTaskResumeTests(TestCase):
    def test_accepted_task_resumes_polling_without_resubmitting(self):
        from src.unlearning_detection import remote_execution as remote
        from test_unlearning_robustness import response, payload
        store = MemoryStore()
        task = store.create_task()
        journal = CallJournal(store, task, heartbeat=False)
        with patch.object(remote, "read_analysis_code_files", return_value={"analysis.py": "unchanged code"}), patch.object(remote.requests, "post", return_value=response(202, {"task_id": "accepted-task"})) as post, patch.object(remote.requests, "get", side_effect=RuntimeError("process stopped")):
            with journal_scope(journal), self.assertRaises(RuntimeError):
                remote.execute_analysis_remotely("https://agent.test", "cka", "reference", "updated", ["query"], api_key="ephemeral")
            journal.finish(False)
            post.assert_called_once()
        fresh = CallJournal(store, store.get_task(task["id"]), heartbeat=False)
        with patch.object(remote, "read_analysis_code_files", return_value={"analysis.py": "unchanged code"}), patch.object(remote.requests, "post") as post, patch.object(remote.requests, "get", return_value=response(200, {"status": "completed", "result": payload()})) as poll:
            with journal_scope(fresh):
                result = remote.execute_analysis_remotely("https://agent.test", "cka", "reference", "updated", ["query"], api_key="ephemeral")
            fresh.finish(True)
            post.assert_not_called()
            poll.assert_called_once()
            self.assertIn("accepted-task", poll.call_args.args[0])
            self.assertIsNotNone(result)
        rebuilt = CallJournal(store, store.get_task(task["id"]), heartbeat=False)
        with patch.object(remote, "read_analysis_code_files", return_value={"analysis.py": "unchanged code"}), patch.object(remote.requests, "post") as post, patch.object(remote.requests, "get") as poll:
            with journal_scope(rebuilt):
                remote.execute_analysis_remotely("https://agent.test", "cka", "reference", "updated", ["query"], api_key="ephemeral")
            rebuilt.finish(True)
            post.assert_not_called()
            poll.assert_not_called()


class AuthoritativeCompletionTests(TestCase):
    def test_verified_saved_result_finishes_without_replaying_all_responses(self):
        from src.resumable_analysis import acknowledge_task_completion
        store = MemoryStore()
        task = store.create_task()
        original = CallJournal(store, task, heartbeat=False)
        original.call("reserve", {}, lambda: "same run")
        original.call("answer", {}, lambda: "saved")
        original.finish(False)
        fresh = CallJournal(store, store.get_task(task["id"]), heartbeat=False)
        with journal_scope(fresh):
            self.assertEqual(checkpoint_call("reserve", {}, Mock()), "same run")
            acknowledge_task_completion()
        fresh.finish(True)
        self.assertEqual(store.get_task(task["id"])["status"], "complete")

    def test_authoritative_completion_cannot_hide_unfinished_calls(self):
        from src.resumable_analysis import acknowledge_task_completion
        store = MemoryStore()
        task = store.create_task()
        journal = CallJournal(store, task, heartbeat=False)
        with self.assertRaises(RuntimeError):
            journal.call("answer", {}, Mock(side_effect=RuntimeError("stopped")))
        journal.finish(False)
        fresh = CallJournal(store, store.get_task(task["id"]), heartbeat=False)
        with journal_scope(fresh), self.assertRaises(AnalysisCheckpointError):
            acknowledge_task_completion()
        fresh.finish(False)
        self.assertEqual(store.get_task(task["id"])["status"], "incomplete")


class SnapshotSizeTests(TestCase):
    def test_large_input_snapshot_compresses_and_restores_exactly(self):
        from src.resumable_analysis import pack_snapshot, unpack_snapshot
        from src.analysis_checkpoints import _json
        snapshot = {"min_k_predefined_batch_data": [{"text": "original book text " * 200, "label": i % 2} for i in range(5000)]}
        packed = pack_snapshot(snapshot)
        self.assertIn("__analysis_snapshot_gzip__", packed)
        _json({"initial_session": packed}, "source")
        self.assertEqual(unpack_snapshot(packed), snapshot)

    def test_large_aggregate_omitted_but_calls_and_small_inputs_preserved(self):
        saved = snapshot_session({"qa_input_text": "source", "qa_evaluation_results": ["answer" * 5000] * 100}, "Knowledge Memorization Detection", max_bytes=4096)
        self.assertEqual(saved, {"qa_input_text": "source"})

    def test_compression_does_not_hide_credential_urls(self):
        from src.resumable_analysis import pack_snapshot
        from src.analysis_checkpoints import AnalysisCheckpointError as StoreError
        with self.assertRaises(StoreError):
            pack_snapshot({"sidebar_local_vllm_base_url": "https://user:password@api.test", "qa_input_text": "text " * 200000})

    def test_corrupt_snapshot_rejected_before_restoring_widgets(self):
        from src.resumable_analysis import unpack_snapshot
        with self.assertRaises(AnalysisCheckpointError):
            unpack_snapshot({"__analysis_snapshot_gzip__": "not base64"})
