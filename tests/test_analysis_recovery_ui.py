"""Behavioral checks for saved-analysis details and owner-scoped Kill."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from postgrest import SyncPostgrestClient
from streamlit.testing.v1 import AppTest

from src import analysis_recovery_ui
from src.analysis_checkpoints import SupabaseAnalysisCheckpointStore
from src.run_cancellation import CancellationToken, bind_task_cancellation


OWNER = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
OTHER_OWNER = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
TASK_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
OTHER_TASK_ID = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
PAGE = "Text Snippet Analysis"


def task(**overrides):
    value = {
        "id": TASK_ID,
        "owner_id": OWNER,
        "page_key": PAGE,
        "settings": {"label": "Saved test analysis"},
        "dynamic_items": True,
        "status": "incomplete",
        "completed_items": 4,
        "total_items": 6,
        "failed_items": 1,
        "stop_requested": False,
        "lease_expires_at": None,
        "created_at": "2026-10-01T12:30:00Z",
        "updated_at": "2026-10-01T13:30:00Z",
    }
    value.update(overrides)
    return value


def display(scope=None, *, phase="running", index=7):
    return {
        "apis": [{"provider": "OpenAI", "model": "saved-model", "endpoint": "https://api.openai.com/v1"}],
        "current_step": {"operation": "llm.completion", "scope": scope or [], "phase": phase, "index": index},
    }


def lease(delta):
    return (datetime.now(timezone.utc) + delta).isoformat()


class MemoryStore:
    """Simulate owner-bound checkpoint lifecycle without any provider or network."""

    owner_id = OWNER

    def __init__(self):
        self.tasks = []
        self.stop_calls = []
        self.failure = None

    def list_tasks(self, **kwargs):
        return deepcopy(self.tasks)

    def request_stop(self, task_id):
        self.stop_calls.append(task_id)
        if self.failure is not None:
            raise self.failure
        row = next(row for row in self.tasks if row["id"] == task_id)
        if row["status"] != "complete":
            row["stop_requested"] = True
        return deepcopy(row)


class SavedAnalysisUITests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore()
        self.patcher = patch.object(analysis_recovery_ui, "get_cloud_store_for_current_user", return_value=self.store)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def open_app(self):
        app = AppTest.from_string(
            "from src.analysis_recovery_ui import render_analysis_recovery\n"
            f"render_analysis_recovery({PAGE!r})"
        ).run()
        self.assertEqual(len(app.exception), 0)
        return app

    @staticmethod
    def text(app):
        return "\n".join(element.value for kind in (app.caption, app.markdown, app.text, app.warning) for element in kind)

    def bind_token(self, owner, task_id):
        token = CancellationToken()
        self.addCleanup(bind_task_cancellation(owner, task_id, token))
        return token

    def test_running_card_shows_saved_api_stage_and_checkpoint_counts(self):
        self.store.tasks = [task(status="running", lease_expires_at=lease(timedelta(minutes=5)),
                                 display=display(["knowledge_qa", 0, 2]))]
        app = self.open_app()
        text = self.text(app)
        self.assertIn("API: OpenAI / Model: saved-model", text)
        self.assertIn("Current / Checkpoint 8: Evaluate knowledge Q&A / run 1, question 3", text)
        self.assertIn("4 saved checkpoints / 1 pending / 1 failed", text)
        self.assertIn("Task ID: " + TASK_ID, text)
        self.assertIn("Operation: llm.completion", text)
        self.assertIn("Task details", [expander.label for expander in app.expander])
        self.assertTrue(app.button(key=f"_analysis_resume:{TASK_ID}").disabled)
        self.assertFalse(app.button(key=f"_analysis_kill:{TASK_ID}").disabled)
        self.assertEqual(len(app.get("progress")), 0)
        self.assertNotIn("%", text)
        self.assertIn("not the total number of planned steps", text)

    def test_legacy_card_recovers_api_and_step_without_exposing_url_credentials(self):
        self.store.tasks = [task(latest_items=[{
            "item_index": 3, "operation": "llm.completion", "status": "complete",
            "updated_at": "2026-10-01T13:30:00Z", "scope": {"path": ["sleek_qa", 2, 0]},
            "model_name": "legacy-model", "base_url": "https://username:password@api.openai.com/v1?api_key=SECRET#fragment",
            "prompt": "PRIVATE PROMPT", "result": "PRIVATE RESULT",
        }])]
        app = self.open_app()
        text = self.text(app)
        self.assertIn("API: OpenAI / Model: legacy-model", text)
        self.assertIn("Paused at / Checkpoint 4: Evaluate SLEEK Q&A / question 3, run 1", text)
        self.assertIn("OpenAI endpoint: https://api.openai.com/v1", text)
        for private in ("username", "password", "SECRET", "fragment", "PRIVATE PROMPT", "PRIVATE RESULT"):
            self.assertNotIn(private, text)

    def test_recent_prompt_preparation_keeps_previous_api_and_latest_step(self):
        self.store.tasks = [task(latest_items=[
            {"item_index": 4, "operation": "prompt.full", "status": "complete", "updated_at": "2026-10-01T14:00:00Z"},
            {"item_index": 3, "operation": "llm.completion", "provider": "Anthropic", "model": "previous-model",
             "status": "complete", "updated_at": "2026-10-01T13:30:00Z"},
        ])]
        text = self.text(self.open_app())
        self.assertIn("API: Anthropic / Model: previous-model", text)
        self.assertIn("Paused at / Checkpoint 5: Prepare sampled prompt", text)

    def test_live_metadata_wins_over_older_checkpoint(self):
        self.store.tasks = [task(display=display(["text_report_narrative"], index=2), latest_items=[{
            "item_index": 100, "operation": "llm.completion", "provider": "Legacy provider", "model": "old-model",
            "status": "complete", "updated_at": "2026-10-01T13:30:00Z",
        }])]
        text = self.text(self.open_app())
        self.assertIn("Checkpoint 3: Generate report explanation", text)
        self.assertIn("Model: saved-model", text)
        self.assertNotIn("old-model", text)

    def test_expired_running_task_is_paused_and_can_restore_original_task(self):
        self.store.tasks = [task(status="running", lease_expires_at=lease(timedelta(minutes=-5)),
                                 display=display(["game_direct", 0], phase="complete"))]
        app = self.open_app()
        text = self.text(app)
        self.assertIn("Paused / 4 saved checkpoints", text)
        self.assertIn("Paused at / Checkpoint 8: Game direct recall / item 1", text)
        self.assertFalse(app.button(key=f"_analysis_resume:{TASK_ID}").disabled)
        self.assertFalse(app.button(key=f"_analysis_kill:{TASK_ID}").disabled)
        app.button(key=f"_analysis_resume:{TASK_ID}").click().run()
        self.assertEqual(app.session_state[analysis_recovery_ui.PENDING_RESTORE], {"task_id": TASK_ID})
        self.assertEqual(self.store.stop_calls, [])
        self.assertEqual(len(app.exception), 0)

    def test_completed_and_stopping_tasks_keep_correct_controls(self):
        self.store.tasks = [
            task(status="complete", display=display(["text_report_narrative"], phase="complete")),
            task(id=OTHER_TASK_ID, status="running", stop_requested=True,
                 lease_expires_at=lease(timedelta(minutes=5)), display=display(["knowledge_qa", 0, 0])),
        ]
        app = self.open_app()
        completed = app.button(key=f"_analysis_resume:{TASK_ID}")
        self.assertEqual(completed.label, "View results")
        self.assertFalse(completed.disabled)
        self.assertTrue(app.button(key=f"_analysis_kill:{TASK_ID}").disabled)
        self.assertTrue(app.button(key=f"_analysis_resume:{OTHER_TASK_ID}").disabled)
        stopping = app.button(key=f"_analysis_kill:{OTHER_TASK_ID}")
        self.assertTrue(stopping.disabled)
        self.assertEqual(stopping.label, "Stopping...")
        self.assertIn("Resume is available after current requests finish", self.text(app))

    def test_kill_stops_only_selected_owned_task_and_persists_cloud_request(self):
        self.store.tasks = [task(), task(id=OTHER_TASK_ID, settings={"label": "Other saved task"})]
        selected = self.bind_token(OWNER, TASK_ID)
        different_owner = self.bind_token(OTHER_OWNER, TASK_ID)
        different_task = self.bind_token(OWNER, OTHER_TASK_ID)
        app = self.open_app()
        app.button(key=f"_analysis_kill:{TASK_ID}").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(selected.cancelled)
        self.assertFalse(different_owner.cancelled)
        self.assertFalse(different_task.cancelled)
        self.assertEqual(self.store.stop_calls, [TASK_ID])
        self.assertIn("Stopped / 4 saved checkpoints", self.text(app))
        self.assertTrue(app.button(key=f"_analysis_kill:{TASK_ID}").disabled)
        self.assertFalse(app.button(key=f"_analysis_resume:{TASK_ID}").disabled)
        self.assertFalse(app.button(key=f"_analysis_kill:{OTHER_TASK_ID}").disabled)
        self.assertEqual(self.store.tasks[0]["completed_items"], 4)

    def test_active_kill_blocks_resume_until_requests_release_lease(self):
        self.store.tasks = [task(status="running", lease_expires_at=lease(timedelta(minutes=5)),
                                 display=display(["knowledge_qa", 0, 0]))]
        local = self.bind_token(OWNER, TASK_ID)
        app = self.open_app()
        app.button(key=f"_analysis_kill:{TASK_ID}").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(local.cancelled)
        self.assertEqual(self.store.stop_calls, [TASK_ID])
        self.assertTrue(app.button(key=f"_analysis_resume:{TASK_ID}").disabled)
        self.assertTrue(app.button(key=f"_analysis_kill:{TASK_ID}").disabled)
        self.assertIn("Stopping at / Checkpoint 8", self.text(app))
        self.store.tasks[0].update(lease_expires_at=None, status="incomplete")
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertFalse(app.button(key=f"_analysis_resume:{TASK_ID}").disabled)
        self.assertIn("Stopped at / Checkpoint 8", self.text(app))
        self.assertEqual(self.store.tasks[0]["completed_items"], 4)

    def test_cloud_kill_failure_is_actionable_and_retry_clears_error(self):
        self.store.tasks = [task()]
        local = self.bind_token(OWNER, TASK_ID)
        self.store.failure = ConnectionError("PRIVATE CONNECTION DETAIL")
        app = self.open_app()
        app.button(key=f"_analysis_kill:{TASK_ID}").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(local.cancelled)
        text = self.text(app)
        self.assertIn("Retry Kill to stop this task across sessions", text)
        self.assertIn("Saved progress is preserved", text)
        self.assertNotIn("PRIVATE CONNECTION DETAIL", text)
        self.assertFalse(app.button(key=f"_analysis_kill:{TASK_ID}").disabled)
        self.store.failure = None
        app.button(key=f"_analysis_kill:{TASK_ID}").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(self.store.stop_calls, [TASK_ID, TASK_ID])
        self.assertEqual(len(app.warning), 0)
        self.assertTrue(app.button(key=f"_analysis_kill:{TASK_ID}").disabled)

    def test_missing_checkpoint_still_renders_actionable_card(self):
        self.store.tasks = [task(total_items=0, completed_items=0, failed_items=0)]
        app = self.open_app()
        self.assertIn("API / Model: not recorded yet", self.text(app))
        self.assertIn("No checkpoint recorded yet", self.text(app))
        self.assertFalse(app.button(key=f"_analysis_resume:{TASK_ID}").disabled)
        self.assertFalse(app.button(key=f"_analysis_kill:{TASK_ID}").disabled)


class CheckpointSummaryQueryTests(unittest.TestCase):
    def test_sdk_summary_query_is_owner_scoped_and_bounds_compact_nested_items(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json=[], headers={"Content-Type": "application/json"})

        with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
            client = SyncPostgrestClient("https://checkpoint.invalid/rest/v1", http_client=http_client)
            store = SupabaseAnalysisCheckpointStore(SimpleNamespace(table=client.from_), OWNER)
            self.assertEqual(store.list_tasks(page_key=PAGE, limit=10, summary_only=True), [])

        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.url.path, "/rest/v1/analysis_tasks")
        query = request.url.params
        self.assertEqual(query["owner_id"], "eq." + OWNER)
        self.assertEqual(query["page_key"], "eq." + PAGE)
        self.assertEqual(query["order"], "updated_at.desc,id.asc")
        self.assertEqual(query["latest_items.order"], "updated_at.desc,item_index.desc")
        self.assertEqual(query["latest_items.offset"], "0")
        self.assertEqual(query["latest_items.limit"], "5")
        self.assertEqual(query["limit"], "10")
        selection = query["select"]
        self.assertIn("display:metadata->display", selection)
        self.assertIn("latest_items:analysis_items(", selection)
        self.assertIn("provider:input->request->>provider", selection)
        self.assertIn("model:input->request->>model", selection)
        self.assertIn("operation:input->>operation", selection)
        for excluded in ("source", "initial_session", "final_session", "prompt", "result", "api_key", "access_token"):
            self.assertNotIn(excluded, selection)


if __name__ == "__main__":
    unittest.main()
