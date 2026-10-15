"""Real journal/task runtime regressions with an in-memory store and fake UI."""
import importlib
import json
import sys
import types
import unittest
from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
from threading import Event, Thread
from unittest.mock import Mock, patch


def setUpModule():
    global st, journal, jobs, pages, recovery, patches
    patches = ExitStack()
    st = types.ModuleType("streamlit")
    st.session_state = {}
    st.query_params = {}
    st.fragment = st.dialog = lambda *args, **kwargs: lambda function: function
    st.info = st.warning = st.error = Mock()
    scripts = types.ModuleType("streamlit.runtime.scriptrunner")
    scripts.get_script_run_ctx = lambda **kwargs: None
    replacements = {"streamlit": st, "streamlit.runtime": types.ModuleType("streamlit.runtime"),
                    "streamlit.runtime.scriptrunner": scripts}
    try:
        import requests
    except ImportError:
        replacements["requests"] = types.ModuleType("requests")
    patches.enter_context(patch.dict(sys.modules, replacements))
    journal = importlib.import_module("src.resumable_analysis")
    jobs = importlib.import_module("src.background_jobs")
    pages = importlib.import_module("src.page_jobs")
    recovery = importlib.import_module("src.analysis_recovery_ui")
    for module in (jobs, pages, recovery):
        patches.enter_context(patch.object(module, "st", st))
    patches.enter_context(patch.object(pages, "render_background_job_status"))


def tearDownModule():
    patches.close()


PAGE = "Knowledge Memorization Detection"
TRIGGER = "generate_qa_button"


def saved_row(prompt, result, *, index=0, status="complete"):
    request = {"operation": "llm.completion", "request": {"prompt": prompt}}
    request["fingerprint"] = sha256(json.dumps(
        request, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode()).hexdigest()
    return {"index": index, "payload": request, "status": status, "result": result}


class Store:
    owner_id = "owner"
    def __init__(self, *, status="complete", rows=(), page=PAGE, trigger=TRIGGER):
        self.task = {"id": "saved-task", "owner_id": self.owner_id, "page_key": page,
                     "status": status, "metadata": {}, "source": {"trigger_key": trigger,
                     "initial_session": {"qa_input_text": "submitted"}, "journal_version": 1}}
        self.tasks = {self.task["id"]: self.task}
        self.rows = {self.task["id"]: list(rows)}
        self.writes = []
        self.on_heartbeat = None
        self.on_release = None
    def get_task(self, identity):
        return deepcopy(self.tasks.get(identity))
    def create_task(self, **kwargs):
        self.writes.append(("create", kwargs))
        task = {"id": "new-task", "owner_id": self.owner_id, "page_key": kwargs["page_key"],
                "source": kwargs["source"], "metadata": {}, "status": "incomplete"}
        self.tasks[task["id"]] = task
        self.rows[task["id"]] = []
        return deepcopy(task)
    def claim(self, identity, **kwargs):
        self.writes.append(("claim", identity))
        self.tasks[identity].update(status="running", lease_token="lease")
        return deepcopy(self.tasks[identity])
    def load_items(self, identity, *, offset=0, limit=200):
        return deepcopy(self.rows[identity][offset:offset + limit])
    def heartbeat(self, identity, token, **kwargs):
        if self.on_heartbeat:
            return self.on_heartbeat()
        return deepcopy(self.tasks[identity])
    def release(self, identity, token, *, status, metadata):
        self.writes.append(("release", status))
        self.tasks[identity].update(status=status, metadata=deepcopy(metadata))
        if self.on_release:
            self.on_release()
        return deepcopy(self.tasks[identity])
    def append_item(self, *args, **kwargs):
        raise AssertionError("Cached replay cannot append a work item")
    def save_item(self, *args, **kwargs):
        raise AssertionError("Cached replay cannot save a work item")
    def save_progress(self, *args, **kwargs):
        raise AssertionError("Cached replay cannot update progress")


class TaskRecoveryRuntime(unittest.TestCase):
    def setUp(self):
        st.session_state = {"main_navigation": PAGE, "user_id": "owner", "access_token": "test-token"}
        st.query_params = {}
        with jobs._LOCK:
            jobs._JOBS.clear()
        self.store = Store()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(journal, "get_cloud_store_for_current_user", side_effect=lambda **kwargs: self.store))
        self.stack.enter_context(patch.object(recovery, "get_cloud_store_for_current_user", side_effect=lambda **kwargs: self.store))
        self.live_calls = Mock(side_effect=AssertionError("A cached View invoked a live API"))
    def submit(self, runner, *, trigger=TRIGGER, page=PAGE):
        st.session_state["main_navigation"] = page
        with journal.page_analysis_scope(page, st.session_state):
            self.assertTrue(pages.submit_page_job(trigger, "Saved analysis", runner))
        key = st.session_state[pages._SELECTIONS][page]
        with jobs._LOCK:
            future = jobs._JOBS[key].get("_future")
        if future is not None:
            future.result(timeout=3)
        return jobs.get_background_job(key)
    def select_legacy_view(self, store):
        self.store = store
        st.session_state[journal.PENDING_RESTORE] = {"task_id": "saved-task"}
        recovery.apply_pending_analysis_restore()
        self.assertEqual(st.session_state[journal.VIEW_TASK], "saved-task")
        self.assertTrue(st.session_state.pop("detection_run_" + TRIGGER))
    def replay(self, prompt):
        return journal.checkpoint_call("llm.completion", {"prompt": prompt}, self.live_calls)

    def test_legacy_completed_view_rebuilds_real_cached_calls_without_cloud_writes(self):
        self.select_legacy_view(Store(rows=[saved_row("original", "cached answer")]))
        record = self.submit(lambda report: {"qa_generated_qa_pairs": [self.replay("original")]})
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["result"]["qa_generated_qa_pairs"], ["cached answer"])
        self.assertEqual(self.store.writes, [])
        self.live_calls.assert_not_called()

    def test_cached_only_mode_cannot_fall_back_to_an_untracked_guest_worker(self):
        self.select_legacy_view(Store(rows=[saved_row("original", "cached")]))
        with patch.object(journal, "get_cloud_store_for_current_user", return_value=None):
            with journal.page_analysis_scope(PAGE, st.session_state):
                with self.assertRaises(journal.AnalysisCheckpointError):
                    pages.submit_page_job(TRIGGER, "Read only", self.live_calls)
        self.live_calls.assert_not_called()
        self.assertEqual(self.store.writes, [])

    def test_cached_only_mode_rejects_excluded_and_official_page_workers(self):
        for page in ("Unlearning Detection", "Game 1: The Hidden Passage Hunt"):
            with self.subTest(page=page):
                st.session_state.update({journal.VIEW_TASK: "saved-task", journal.VIEW_PAGE: page,
                                         journal.RESUME_TASK: "saved-task", "main_navigation": page})
                with journal.page_analysis_scope(page, st.session_state):
                    with self.assertRaises(journal.AnalysisCheckpointError):
                        pages.submit_page_job(TRIGGER, "Read only", self.live_calls)
        self.live_calls.assert_not_called()
        self.assertEqual(self.store.writes, [])

    def test_cached_view_missing_response_fails_without_invoking_provider(self):
        self.select_legacy_view(Store(rows=[]))
        record = self.submit(lambda report: {"qa_generated_qa_pairs": [self.replay("missing")]})
        self.assertEqual(record["status"], "failed")
        self.assertIsNone(record["result"])
        self.assertEqual(self.store.writes, [])
        self.live_calls.assert_not_called()

    def test_cached_view_with_failed_saved_response_cannot_retry_api(self):
        self.select_legacy_view(Store(rows=[saved_row("original", None, status="failed")]))
        record = self.submit(lambda report: {"qa_generated_qa_pairs": [self.replay("original")]})
        self.assertEqual(record["status"], "failed")
        self.assertEqual(self.store.writes, [])
        self.live_calls.assert_not_called()

    def test_cached_view_must_replay_every_saved_call_before_publishing_complete(self):
        self.select_legacy_view(Store(rows=[saved_row("one", "first"), saved_row("two", "second", index=1)]))
        record = self.submit(lambda report: {"qa_generated_qa_pairs": [self.replay("one")]})
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["result"]["qa_generated_qa_pairs"], ["first"])
        self.assertEqual(self.store.writes, [])
        self.live_calls.assert_not_called()

    def test_partial_cached_view_is_failed_with_saved_partial_outputs(self):
        self.select_legacy_view(Store(rows=[saved_row("original", "cached")]))
        record = self.submit(lambda report: pages.PageJobResult(
            {"qa_generated_qa_pairs": [self.replay("original")]}, complete=False, error="Report step failed"))
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["result"]["qa_generated_qa_pairs"], ["cached"])
        self.assertEqual(self.store.writes, [])
        self.live_calls.assert_not_called()

    def test_final_snapshot_failure_does_not_discard_successful_local_results(self):
        self.store = Store(status="incomplete")
        with patch.object(journal, "pack_snapshot", side_effect=journal.AnalysisCheckpointError("Snapshot too large")):
            # Existing resume inputs avoid snapshot packing during task creation.
            st.session_state[journal.RESUME_TASK] = "saved-task"
            record = self.submit(lambda report: {"qa_generated_qa_pairs": ["available locally"]})
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["result"]["qa_generated_qa_pairs"], ["available locally"])
        self.assertEqual(self.store.tasks["saved-task"]["status"], "complete")

    def test_incomplete_replay_is_not_reported_locally_complete(self):
        self.store = Store(status="incomplete", rows=[saved_row("one", "first"), saved_row("two", "second", index=1)])
        st.session_state[journal.RESUME_TASK] = "saved-task"
        record = self.submit(lambda report: {"qa_generated_qa_pairs": [self.replay("one")]})
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["result"]["qa_generated_qa_pairs"], ["first"])
        self.assertEqual(self.store.tasks["saved-task"]["status"], "incomplete")

    def test_new_background_result_metadata_wins_over_older_saved_report(self):
        store = Store(status="incomplete")
        store.task["metadata"] = {"background_result": {"runs": ["old"]}, "retained": "unchanged"}
        call_journal = journal.CallJournal(store, store.get_task("saved-task"), heartbeat=False)
        call_journal.metadata["background_result"] = {"runs": ["old", "new"]}
        self.assertTrue(call_journal.finish(True))
        metadata = store.tasks["saved-task"]["metadata"]
        self.assertEqual(metadata["background_result"], {"runs": ["old", "new"]})
        self.assertEqual(metadata["retained"], "unchanged")

    def test_checkpoint_release_failure_retains_calculated_local_outputs(self):
        self.store = Store(status="incomplete")
        st.session_state[journal.RESUME_TASK] = "saved-task"
        self.store.on_release = Mock(side_effect=RuntimeError("Cloud unavailable"))
        record = self.submit(lambda report: {"qa_generated_qa_pairs": ["already calculated"]})
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["result"]["qa_generated_qa_pairs"], ["already calculated"])
        self.assertTrue(record["has_result"])

    def test_stop_during_checkpoint_release_failure_never_publishes_results(self):
        self.store = Store(status="incomplete")
        st.session_state[journal.RESUME_TASK] = "saved-task"
        releasing, fail_release = Event(), Event()
        self.addCleanup(fail_release.set)
        def release_error():
            releasing.set()
            self.assertTrue(fail_release.wait(2))
            raise RuntimeError("Cloud unavailable")
        self.store.on_release = release_error
        with journal.page_analysis_scope(PAGE, st.session_state):
            self.assertTrue(pages.submit_page_job(TRIGGER, "Stoppable", lambda report: {"qa_generated_qa_pairs": ["private result"]}))
        key = st.session_state[pages._SELECTIONS][PAGE]
        self.assertTrue(releasing.wait(2))
        self.assertTrue(jobs.force_stop_background_job(key))
        fail_release.set()
        with jobs._LOCK:
            future = jobs._JOBS[key].get("_future")
        if future is not None:
            future.result(timeout=3)
        record = jobs.get_background_job(key)
        self.assertEqual(record["status"], "stopped")
        self.assertIsNone(record["result"])

    def test_failed_coordinator_keeps_key_fenced_until_accepted_calls_drain(self):
        from src.run_cancellation import current_cancellation_token
        for stop in (False, True):
            with self.subTest(stop=stop):
                accepted, release_call, terminal = Event(), Event(), Event()
                self.addCleanup(release_call.set)
                key = "fixed-failure-key-" + str(stop)
                def runner(report):
                    token = current_cancellation_token()
                    token._begin_call()
                    def accepted_request():
                        accepted.set()
                        release_call.wait(2)
                        token._end_call()
                    Thread(target=accepted_request, daemon=True).start()
                    raise RuntimeError("Another parallel step failed")
                original_clear = jobs._clear_finished_handles_locked
                def signal_terminal(record):
                    original_clear(record)
                    if record.get("key") == key and record.get("status") in {"completed", "failed", "stopped"}:
                        terminal.set()
                with patch.object(jobs, "_clear_finished_handles_locked", side_effect=signal_terminal):
                    self.assertTrue(jobs.submit_background_job(key, "Pending failure", runner))
                    self.assertTrue(accepted.wait(2))
                    with jobs._LOCK:
                        future = jobs._JOBS[key].get("_future")
                    if future is not None:
                        future.result(timeout=3)
                    self.assertEqual(jobs.get_background_job(key)["status"], "stopping")
                    self.assertFalse(jobs.submit_background_job(key, "Cannot replace", lambda report: {}))
                    if stop:
                        self.assertTrue(jobs.force_stop_background_job(key))
                    release_call.set()
                    self.assertTrue(terminal.wait(2))
                record = jobs.get_background_job(key)
                self.assertEqual(record["status"], "stopped" if stop else "failed")
                self.assertIsNone(record["result"])
                self.assertTrue(jobs.submit_background_job(key, "Can replace now", lambda report: {}))
                with jobs._LOCK:
                    future = jobs._JOBS[key].get("_future")
                if future is not None:
                    future.result(timeout=3)
                self.assertEqual(jobs.get_background_job(key)["status"], "completed")

    def test_blocked_resume_on_other_page_does_not_hijack_new_task(self):
        self.store = Store(status="incomplete", page="Content Recall Detection", trigger="run_snippet_analysis_button")
        st.session_state[journal.RESUME_TASK] = "saved-task"
        record = self.submit(lambda report: {"qa_generated_qa_pairs": ["new output"]})
        self.assertEqual(record["task_id"], "new-task")
        self.assertEqual(record["status"], "completed")
        self.assertNotIn(("claim", "saved-task"), self.store.writes)

    def test_partial_result_keeps_actual_progress_instead_of_forcing_total(self):
        for complete in (False, True):
            with self.subTest(complete=complete):
                def runner(report):
                    report(1, 5, "One of five steps finished")
                    return pages.PageJobResult({"min_k_user_input_results": ["saved"]},
                                               complete=complete, error=None if complete else "Remaining steps failed")
                record = self.submit(runner, trigger="run_min_k_prob_analysis_button", page="Unlearning Detection")
                self.assertEqual(record["status"], "completed" if complete else "failed")
                self.assertEqual(record["total"], 5)
                self.assertEqual(record["current"], 5 if complete else 1)

    def test_success_waits_for_accepted_calls_before_journal_finalization(self):
        from src.run_cancellation import CancellationToken, current_cancellation_token
        for stop in (False, True):
            with self.subTest(stop=stop):
                self.store = Store(status="incomplete")
                st.session_state[journal.RESUME_TASK] = "saved-task"
                accepted, release_call, waiting, finishing = Event(), Event(), Event(), Event()
                self.addCleanup(release_call.set)
                holder = []
                def runner(report):
                    token = current_cancellation_token()
                    holder.append(token)
                    token._begin_call()
                    def request_cleanup():
                        accepted.set()
                        release_call.wait(2)
                        token._end_call()
                    Thread(target=request_cleanup, daemon=True).start()
                    return {"qa_generated_qa_pairs": ["finished output"]}
                original_wait = CancellationToken.wait_for_idle
                original_finish = journal.CallJournal.finish
                def observe_wait(token, timeout=None):
                    if holder and token is holder[0] and token.pending_count:
                        waiting.set()
                    return original_wait(token, timeout)
                def observe_finish(call_journal, *args, **kwargs):
                    finishing.set()
                    return original_finish(call_journal, *args, **kwargs)
                with patch.object(CancellationToken, "wait_for_idle", new=observe_wait):
                    with patch.object(journal.CallJournal, "finish", new=observe_finish):
                        with journal.page_analysis_scope(PAGE, st.session_state):
                            self.assertTrue(pages.submit_page_job(TRIGGER, "Drain success", runner))
                        key = st.session_state[pages._SELECTIONS][PAGE]
                        try:
                            self.assertTrue(accepted.wait(2))
                            self.assertTrue(waiting.wait(2))
                            self.assertFalse(finishing.is_set(), "Finalized while accepted requests were pending")
                            self.assertEqual(jobs.get_background_job(key)["status"], "running")
                            if stop:
                                self.assertTrue(jobs.force_stop_background_job(key))
                        finally:
                            release_call.set()
                        with jobs._LOCK:
                            future = jobs._JOBS[key].get("_future")
                        if future is not None:
                            future.result(timeout=3)
                record = jobs.get_background_job(key)
                self.assertEqual(record["status"], "stopped" if stop else "completed")
                self.assertEqual(self.store.tasks["saved-task"]["status"], "incomplete" if stop else "complete")
                if stop:
                    self.assertIsNone(record["result"])

    def test_stale_account_callbacks_cannot_stop_or_delete_previous_owner_task(self):
        from src.run_cancellation import CancellationToken, bind_task_cancellation
        self.store.request_stop = Mock(return_value={"stop_requested": True})
        self.store.kill_task = Mock(return_value={"deleted": True})
        token = CancellationToken()
        unbind = bind_task_cancellation("owner", "saved-task", token)
        self.addCleanup(unbind)
        st.session_state["user_id"] = "different-account"
        for page in (PAGE, "Game 1: The Hidden Passage Hunt"):
            with self.subTest(page=page):
                self.assertFalse(recovery._stop_saved_task(self.store, "saved-task", page_key=page))
                self.assertFalse(recovery._kill_saved_task(self.store, "saved-task", page_key=page))
        self.assertFalse(token.cancelled)
        self.store.request_stop.assert_not_called()
        self.store.kill_task.assert_not_called()

    def test_delivered_result_is_not_copied_again_on_parameter_reruns(self):
        record = self.submit(lambda report: {"qa_generated_qa_pairs": ["saved"]})
        pages.render_page_jobs(PAGE)
        self.assertEqual(st.session_state["qa_generated_qa_pairs"], ["saved"])
        with patch.object(pages, "get_background_job", wraps=jobs.get_background_job) as get:
            pages.render_page_jobs(PAGE)
            pages.render_page_jobs(PAGE)
        self.assertEqual(get.call_count, 2)
        for call in get.call_args_list:
            self.assertEqual(call.args, (record["key"],))
            self.assertIs(call.kwargs.get("include_result"), False)

    def test_cloud_card_keeps_view_for_latest_local_partial_result_after_dedup(self):
        self.store = Store(status="incomplete")
        self.store.release = Mock(side_effect=RuntimeError("Checkpoint unavailable"))
        st.session_state[journal.RESUME_TASK] = "saved-task"
        older = self.submit(lambda report: {"qa_generated_qa_pairs": ["older output"]})
        self.store.task["status"] = "incomplete"  # A later recovery after lease expiry.
        st.session_state[journal.RESUME_TASK] = "saved-task"
        newest = self.submit(lambda report: {"qa_generated_qa_pairs": ["latest partial output"]})
        self.assertNotEqual(older["key"], newest["key"])
        self.assertEqual(newest["status"], "failed")
        self.store.list_tasks = Mock(return_value=[{**self.store.task, "dynamic_items": True}])
        buttons, containers = [], []
        class Context:
            def __enter__(self): return self
            def __exit__(self, *args): return False
        def container(*args, **kwargs):
            containers.append(kwargs.get("key"))
            return Context()
        def button(label, **kwargs):
            buttons.append((label, kwargs))
            return False
        from src import job_guard
        with ExitStack() as ui:
            ui.enter_context(patch.object(job_guard, "st", st))
            for name in ("html", "caption", "markdown", "badge"):
                ui.enter_context(patch.object(st, name, lambda *args, **kwargs: None, create=True))
            ui.enter_context(patch.object(st, "container", container, create=True))
            ui.enter_context(patch.object(st, "expander", lambda *args, **kwargs: Context(), create=True))
            ui.enter_context(patch.object(st, "columns", lambda *args, **kwargs: [Context(), Context()], create=True))
            ui.enter_context(patch.object(st, "button", button, create=True))
            recovery.render_analysis_recovery(PAGE)
        self.assertEqual(containers.count("history_task_saved-task"), 1)
        self.assertFalse(any(isinstance(key, str) and key.startswith("history_task_local_") for key in containers))
        views = [options for label, options in buttons if label == "View"]
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0]["args"], (newest["key"],))
        self.assertTrue(any(label == "Resume" for label, options in buttons))
        views[0]["on_click"](*views[0]["args"])
        pages.apply_pending_page_job_view()
        self.assertEqual(st.session_state["qa_generated_qa_pairs"], ["latest partial output"])

    def test_delayed_heartbeat_cannot_cancel_normally_released_journal(self):
        from src.run_cancellation import CancellationToken, cancellation_scope
        for response in (RuntimeError("Released lease"), {"stop_requested": True}):
            with self.subTest(response=type(response).__name__):
                store = Store(status="incomplete")
                entered, released = Event(), Event()
                self.addCleanup(released.set)
                def heartbeat():
                    entered.set()
                    self.assertTrue(released.wait(2))
                    if isinstance(response, Exception):
                        raise response
                    return response
                store.on_heartbeat = heartbeat
                store.on_release = released.set
                token = CancellationToken()
                with cancellation_scope(token):
                    call_journal = journal.CallJournal(store, store.get_task("saved-task"), heartbeat=False)
                with patch.object(journal, "HEARTBEAT_SECONDS", 0):
                    heartbeat_thread = Thread(target=call_journal._heartbeat, daemon=True)
                    heartbeat_thread.start()
                    self.assertTrue(entered.wait(2))
                    self.assertTrue(call_journal.finish(True))
                    heartbeat_thread.join(2)
                self.assertFalse(heartbeat_thread.is_alive())
                self.assertFalse(token.cancelled)
                self.assertFalse(call_journal.broken)
                self.assertTrue(call_journal.completed)

    def test_discard_restore_only_removes_its_own_unconsumed_trigger(self):
        st.session_state.update({journal.RESUME_TASK: "old", journal.VIEW_TASK: "old",
            "_analysis_restore_feedback": {"task_id": "old", "trigger_key": "old_run"},
            "detection_run_old_run": True, "detection_run_user_new_run": True,
            "detection_job_running": True})
        recovery.discard_pending_analysis_restore()
        self.assertNotIn("detection_run_old_run", st.session_state)
        self.assertNotIn(journal.RESUME_TASK, st.session_state)
        self.assertTrue(st.session_state["detection_run_user_new_run"])
        self.assertTrue(st.session_state["detection_job_running"])

    def test_local_view_discards_stale_restore_and_selects_min_k_input_mode(self):
        record = self.submit(lambda report: {"min_k_upload_response": "saved"},
                             trigger="run_min_k_prob_analysis_button", page="Unlearning Detection")
        st.session_state.update({"min_k_input_mode": "User Input", journal.RESUME_TASK: "old",
            "_analysis_restore_feedback": {"task_id": "old", "trigger_key": "old_run"},
            "detection_run_old_run": True, "detection_run_user_new_run": True})
        self.assertTrue(pages.view_page_job(record["key"]))
        pages.apply_pending_page_job_view()
        self.assertEqual(st.session_state["min_k_input_mode"], "Upload Document")
        self.assertNotIn("detection_run_old_run", st.session_state)
        self.assertTrue(st.session_state["detection_run_user_new_run"])
        self.assertNotIn(journal.RESUME_TASK, st.session_state)


if __name__ == "__main__":
    unittest.main()
