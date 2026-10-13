"""Behavioral checks for document saved-task summaries and owner-scoped Kill."""
from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.pages import document_memorization_detection as document_page


OWNER = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
TOKEN = "b" * 64


def task(**overrides):
    value = {
        "status": "incomplete",
        "owner_id": OWNER,
        "settings": {
            "filename": "saved-document.txt", "provider": "OpenAI", "model": "saved-model",
            "chunk_size": 200, "continuation_method": "Normal Continuation",
            "temperature": 0.7, "top_p": 0.9,
            "base_url": "https://username:password@example.com/v1?api_key=SECRET#PRIVATE",
            "custom_template": "PRIVATE PROMPT",
        },
        "results": {0: ("input", "target", "result", {}), 1: ("input", "target", "result", {})},
        "failures": {2: "Temporary error"}, "attempts": {2: 2},
        "total_chunks": 10, "current_chunk": 3, "current_attempt": 2,
        "retry_in_seconds": 0, "retry_attempt": None,
        "active_chunks": [], "concurrency_limit": 3, "effective_concurrency": 3,
        "stop_requested": False,
        "started_at": "2026-10-01T12:30:00Z", "updated_at": "2026-10-01T13:30:00Z",
    }
    value.update(overrides)
    return value


class MemoryDocumentJobs:
    def __init__(self, state, *, active=False):
        self.state = state
        self.active = active
        self.stop_calls = []
        self.submit_calls = []
        self.failure = None

    def get(self, token, *, owner_id):
        return deepcopy(self.state)

    def is_active(self, token, *, owner_id):
        return self.active

    def cloud_enabled_for(self, owner_id):
        return owner_id == OWNER

    def list_saved(self, *, owner_id):
        return [{
            "token": TOKEN, "settings": deepcopy(self.state["settings"]),
            "status": self.state["status"], "completed_chunks": len(self.state["results"]),
            "failed_chunks": len(self.state["failures"]), "total_chunks": self.state["total_chunks"],
            "updated_at": self.state["updated_at"], "active": self.active,
        }]

    def force_stop(self, token, *, owner_id):
        self.stop_calls.append((token, owner_id))
        if self.failure:
            raise self.failure
        if self.active:
            self.state["stop_requested"] = True
        return self.active

    def submit(self, token, api_key, *, owner_id):
        self.submit_calls.append((token, api_key, owner_id))
        self.active = True
        self.state["status"] = "running"
        self.state["stop_requested"] = False
        return True


class DocumentSavedAnalysisUITests(unittest.TestCase):
    def open_app(self, state=None, *, active=False, recovery=False, show_resume=True):
        self.jobs = MemoryDocumentJobs(state or task(), active=active)
        patchers = [
            patch.object(document_page, "DOCUMENT_JOBS", self.jobs),
            patch.object(document_page, "_render_saved_pdf_results"),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        script = (
            "import streamlit as st\n"
            "from src.pages.document_memorization_detection import _render_document_job, _render_document_recovery_list\n"
            f"st.session_state['user_id'] = {OWNER!r}\n"
            f"st.session_state['_pdf_verified_cloud_owner'] = {OWNER!r}\n"
        )
        if recovery:
            script += "_render_document_recovery_list(disabled=True)\n"
        else:
            script += f"_render_document_job({TOKEN!r}, 'current-secret-key', 'OpenAI', show_resume_button={show_resume!r})\n"
        app = AppTest.from_string(script).run()
        self.assertEqual(len(app.exception), 0)
        return app

    @staticmethod
    def text(app):
        return "\n".join(element.value for kind in (app.caption, app.markdown, app.info, app.error) for element in kind)

    def test_paused_summary_uses_saved_api_and_chunk_stage_without_secrets(self):
        app = self.open_app()
        text = self.text(app)
        self.assertIn("API: OpenAI / Model: saved-model", text)
        self.assertIn("Stage: Paused / last attempted chunk 3 / 10", text)
        self.assertIn("2/10 chunks saved / 1 failed / 8 remaining", text)
        self.assertIn("Started: 2026-10-01 12:30:00 UTC", text)
        self.assertIn("Updated: 2026-10-01 13:30:00 UTC", text)
        self.assertIn("Task ID: " + TOKEN, text)
        self.assertIn("API endpoint: https://example.com/v1", text)
        for secret in ("username", "password", "SECRET", "PRIVATE", "current-secret-key"):
            self.assertNotIn(secret, text)
        self.assertFalse(app.button(key="resume_saved_pdf_analysis").disabled)
        self.assertEqual(app.button(key="kill_saved_pdf_analysis").label, "Kill")

    def test_paused_kill_calls_owner_scoped_stop_and_keeps_progress(self):
        app = self.open_app()
        app.button(key="kill_saved_pdf_analysis").click().run()
        self.assertEqual(self.jobs.stop_calls, [(TOKEN, OWNER)])
        self.assertEqual(len(self.jobs.state["results"]), 2)
        self.assertIn("No active requests to kill. Saved progress is kept.", self.text(app))
        self.assertFalse(app.button(key="resume_saved_pdf_analysis").disabled)

    def test_active_kill_waits_for_existing_requests_before_resume(self):
        app = self.open_app(task(status="running", active_chunks=[3, 4]), active=True)
        self.assertIn("Stage: Analyzing chunks 3, 4 / 10", self.text(app))
        self.assertEqual(app.button(key="stop_pdf_analysis").label, "Kill")
        self.assertNotIn("resume_saved_pdf_analysis", [button.key for button in app.button])
        app.button(key="stop_pdf_analysis").click().run()
        self.assertEqual(self.jobs.stop_calls, [(TOKEN, OWNER)])
        self.assertTrue(self.jobs.active)
        self.assertTrue(app.button(key="stop_pdf_analysis").disabled)
        self.assertEqual(app.button(key="stop_pdf_analysis").label, "Killing...")
        self.assertNotIn("resume_saved_pdf_analysis", [button.key for button in app.button])
        self.jobs.active = False
        self.jobs.state.update(status="incomplete", active_chunks=[])
        app.run()
        self.assertFalse(app.button(key="resume_saved_pdf_analysis").disabled)
        self.assertEqual(len(self.jobs.state["results"]), 2)

    def test_retry_stage_shows_attempt_and_delay(self):
        app = self.open_app(task(status="running", active_chunks=[3], retry_in_seconds=4.5, retry_attempt=2), active=True)
        self.assertIn("Stage: Retrying API call (attempt 2) in 4.5 seconds", self.text(app))

    def test_selected_recovery_can_kill_while_open_is_disabled(self):
        app = self.open_app(task(status="running", active_chunks=[3]), active=True, recovery=True)
        self.assertTrue(app.button(key="kill_selected_saved_pdf_analysis").disabled)
        app.selectbox(key="pdf_cloud_recovery_selection").select(TOKEN).run()
        self.assertTrue(app.button(key="open_saved_pdf_analysis").disabled)
        self.assertFalse(app.button(key="kill_selected_saved_pdf_analysis").disabled)
        self.assertIn("API: OpenAI / Model: saved-model", self.text(app))
        app.button(key="kill_selected_saved_pdf_analysis").click().run()
        self.assertEqual(self.jobs.stop_calls, [(TOKEN, OWNER)])
        self.assertTrue(app.button(key="kill_selected_saved_pdf_analysis").disabled)

    def test_kill_remains_available_when_resume_button_is_elsewhere(self):
        app = self.open_app(show_resume=False)
        self.assertNotIn("resume_saved_pdf_analysis", [button.key for button in app.button])
        self.assertFalse(app.button(key="kill_saved_pdf_analysis").disabled)

    def test_chunk_overflow_and_unstarted_stage_have_plain_text(self):
        self.assertEqual(document_page._document_task_stage(task(current_chunk=None)), "Paused / ready to start chunk analysis")
        stage = document_page._document_task_stage(task(active_chunks=list(range(1, 11))), active=True)
        self.assertEqual(stage, "Analyzing chunks 1, 2, 3, 4, 5, 6, 7, 8, ... / 10")
        self.assertNotIn("?", stage)


if __name__ == "__main__":
    unittest.main()
