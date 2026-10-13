"""Behavioral terminal Kill checks, distinct from resumable Stop.

The SQL integration suite verifies durable transitions; these exercise native
Python entry points so killed tasks cannot restore inputs, replay, or buy calls.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from postgrest import SyncPostgrestClient
from streamlit.testing.v1 import AppTest

from src import analysis_checkpoints, analysis_recovery_ui, resumable_analysis
from src.analysis_checkpoints import AnalysisCheckpointError, SupabaseAnalysisCheckpointStore
from src.run_cancellation import AnalysisCancelled, CancellationToken, cancellation_scope


OWNER = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
TASK = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
LEASE = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
PAGE = "Text Snippet Analysis"


def task(**changes):
    value = {"id": TASK, "owner_id": OWNER, "page_key": PAGE, "status": "incomplete",
             "source": {"initial_session": {"text_input": "saved private input"},
                        "trigger_key": "run_snippet_analysis_button"},
             "metadata": {}, "stop_requested": False, "kill_requested": False}
    value.update(changes)
    return value


def journal_store(row=None):
    store = Mock(owner_id=OWNER)
    store.claim.return_value = task(status="running", lease_token=LEASE)
    store.load_items.return_value = []
    store.heartbeat.return_value = task(status="running", lease_token=LEASE)
    store.save_progress.return_value = task(status="running", lease_token=LEASE)
    store.release.return_value = task(status="incomplete", lease_token=None)
    if row is not None:
        store.claim.return_value = deepcopy(row)
    return store


class KillRpcTests(unittest.TestCase):
    def test_terminal_kill_uses_owner_scoped_rpc_and_retries_same_identity_after_lost_response(self):
        requests = []
        killed = task(status="killed", kill_requested=True, killed_at="2026-10-13T12:00:00Z")

        def respond(request):
            requests.append(request)
            if len(requests) == 1:
                raise httpx.ReadTimeout("private provider detail", request=request)
            return httpx.Response(200, json=killed)

        with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
            client = SyncPostgrestClient("https://checkpoint.invalid/rest/v1", http_client=http_client)
            store = SupabaseAnalysisCheckpointStore(SimpleNamespace(rpc=client.rpc), OWNER)
            with patch.object(analysis_checkpoints.time, "sleep"):
                result = store.kill_task(TASK)
        self.assertEqual(result, killed)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].url.path, "/rest/v1/rpc/analysis_kill_task")
        self.assertEqual(requests[0].read(), requests[1].read())
        self.assertEqual(json.loads(requests[0].read()), {"p_owner_id": OWNER, "p_task_id": TASK})
        self.assertEqual([request.method for request in requests], ["POST", "POST"])

    def test_missing_kill_rpc_returns_actionable_schema_error_without_retry_or_raw_details(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(404, json={"code": "PGRST202", "message": "private schema details",
                                            "details": None, "hint": None})

        with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
            client = SyncPostgrestClient("https://checkpoint.invalid/rest/v1", http_client=http_client)
            store = SupabaseAnalysisCheckpointStore(SimpleNamespace(rpc=client.rpc), OWNER)
            with self.assertRaises(AnalysisCheckpointError) as raised:
                store.kill_task(TASK)
        self.assertEqual(raised.exception.kind, "schema")
        self.assertEqual(raised.exception.code, "PGRST202")
        self.assertNotIn("private schema details", str(raised.exception))
        self.assertEqual(len(requests), 1)

    def test_full_task_retrieval_reconciles_crashed_kills_and_preserves_terminal_record(self):
        requests = []
        killed = task(status="killed", kill_requested=True, lease_token=None,
                      kill_requested_at="2026-10-13T12:00:00Z", killed_at="2026-10-13T12:03:00Z",
                      started_at="2026-10-13T11:30:00Z", completed_at=None)

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json=1 if "/rpc/" in request.url.path else [killed])

        with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
            client = SyncPostgrestClient("https://checkpoint.invalid/rest/v1", http_client=http_client)
            store = SupabaseAnalysisCheckpointStore(SimpleNamespace(rpc=client.rpc, table=client.from_), OWNER)
            result = store.get_task(TASK)
        self.assertEqual(result, killed)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].url.path, "/rest/v1/rpc/analysis_reconcile_killed_tasks")
        self.assertEqual(json.loads(requests[0].read()), {"p_owner_id": OWNER})
        self.assertEqual(requests[1].url.params["id"], "eq." + TASK)
        self.assertEqual(requests[1].url.params["owner_id"], "eq." + OWNER)
        self.assertNotIn("status", requests[1].url.params, "Direct retrieval must still expose durable terminal history")
        for column in ("kill_requested", "kill_requested_at", "killed_at", "started_at", "completed_at"):
            self.assertIn(column, requests[1].url.params["select"].split(","))


class KilledTaskRestoreTests(unittest.TestCase):
    def test_pending_restore_rejects_terminal_task_without_replacing_inputs_or_arming_a_run(self):
        for row in (task(status="stopping"), task(status="killed"), task(kill_requested=True)):
            with self.subTest(status=row["status"], kill_requested=row["kill_requested"]):
                store = Mock(owner_id=OWNER)
                store.get_task.return_value = row
                with patch.object(analysis_recovery_ui, "get_cloud_store_for_current_user", return_value=store):
                    app = AppTest.from_string(
                        "import streamlit as st\n"
                        "from src.analysis_recovery_ui import apply_pending_analysis_restore\n"
                        "from src.analysis_checkpoints import AnalysisCheckpointError\n"
                        "from src.resumable_analysis import PENDING_RESTORE\n"
                        "st.session_state['text_input'] = 'current input'\n"
                        f"st.session_state[PENDING_RESTORE] = {{'task_id': {TASK!r}}}\n"
                        "try:\n"
                        "    apply_pending_analysis_restore()\n"
                        "except AnalysisCheckpointError as exc:\n"
                        "    st.warning(str(exc))\n"
                    ).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.warning), 1)
                self.assertEqual(app.session_state["text_input"], "current input")
                self.assertNotIn(resumable_analysis.RESUME_TASK, app.session_state.filtered_state)
                self.assertNotIn("detection_run_run_snippet_analysis_button", app.session_state.filtered_state)
                store.get_task.assert_called_once_with(TASK)


class KilledJournalTests(unittest.TestCase):
    def test_terminal_journal_is_rejected_before_claim_or_loading_saved_responses(self):
        for row in (task(status="stopping"), task(status="killed"), task(kill_requested=True),
                    task(status="complete", kill_requested=True)):
            with self.subTest(status=row["status"], kill_requested=row["kill_requested"]):
                store = journal_store()
                with self.assertRaises(AnalysisCheckpointError):
                    resumable_analysis.CallJournal(store, row, heartbeat=False)
                store.claim.assert_not_called()
                store.load_items.assert_not_called()

    def test_kill_during_claim_blocks_journal_replay_before_any_provider_call(self):
        claimed = task(status="stopping", kill_requested=True, lease_token=LEASE)
        store = journal_store(claimed)
        with self.assertRaises(AnalysisCheckpointError):
            resumable_analysis.CallJournal(store, task(), heartbeat=False)
        store.claim.assert_called_once()
        store.load_items.assert_not_called()

    def test_remote_kill_before_paid_call_cancels_without_starting_provider_or_saving_result(self):
        for operation in ("heartbeat", "save_progress"):
            with self.subTest(operation=operation):
                store = journal_store()
                token = CancellationToken()
                invoke = Mock(return_value={"answer": "must not be called"})
                with cancellation_scope(token):
                    journal = resumable_analysis.CallJournal(store, task(), heartbeat=False)
                    getattr(store, operation).return_value = task(status="stopping", kill_requested=True,
                                                                 stop_requested=True, lease_token=LEASE)
                    try:
                        with self.assertRaises(AnalysisCancelled):
                            journal.call("llm.completion", {"model": "fixture"}, invoke)
                        self.assertTrue(token.cancelled)
                        invoke.assert_not_called()
                        store.save_item.assert_not_called()
                    finally:
                        journal.finish(False)

    def test_resumable_stop_still_replays_saved_response_without_paid_calls(self):
        store = journal_store()
        request = {"operation": "fixture", "request": {"question": "old input"}}
        request["fingerprint"] = sha256(json.dumps(request, sort_keys=True, ensure_ascii=False,
                                                  separators=(",", ":")).encode()).hexdigest()
        store.load_items.return_value = [{"item_index": 0, "input": request, "status": "complete",
                                         "result": {"answer": ["saved response"]}}]
        invoke = Mock()
        journal = resumable_analysis.CallJournal(store, task(stop_requested=True), heartbeat=False)
        try:
            result = journal.call("fixture", {"question": "old input"}, invoke)
            self.assertEqual(result, {"answer": ["saved response"]})
            invoke.assert_not_called()
            store.append_item.assert_not_called()
            store.heartbeat.assert_not_called()
            store.save_item.assert_not_called()
        finally:
            journal.finish(True)
        self.assertEqual(store.release.call_args.kwargs["status"], "complete")


if __name__ == "__main__":
    unittest.main()
