"""Streamlit regressions for persistent background analysis (no API calls)."""

import time
import unittest
from streamlit.testing.v1 import AppTest


APP = '''
import tempfile
import threading
import streamlit as st
import src.pages.document_memorization_detection as page
from src.document_jobs import DocumentAnalysisJobs
from src.document_checkpoints import DocumentCheckpointStore

if "mock_control" not in st.session_state:
    control = {"calls": [], "fail": True, "block": False, "release": threading.Event(), "entered": threading.Event()}
    def compare(settings, key, upper, lower):
        control["calls"].append(int(upper))
        if control["block"]:
            control["entered"].set()
            control["release"].wait(10)
        if int(upper) == 104 and control["fail"]:
            return "Error calling API: 403 PERMISSION_DENIED", None
        return "generated continuation", {"rouge_l": 0.1}
    st.session_state["mock_control"] = control
    st.session_state["mock_directory"] = tempfile.TemporaryDirectory()
    st.session_state["mock_service"] = DocumentAnalysisJobs(
        DocumentCheckpointStore(st.session_state["mock_directory"].name),
        analyze_chunk=compare,
    )

page.DOCUMENT_JOBS = st.session_state["mock_service"]
class Document:
    name = "source.txt"
    type = "text/plain"
    def getvalue(self):
        return b"source contents"

page._list_example_documents = lambda: [("Test document", None)]
page._resolve_active_document = lambda *args: Document()
page.extract_text_from_document = lambda document: "source contents"
page.split_text_into_chunks = lambda text, chunk_size, overlap=50: [(str(i), "target") for i in range(st.session_state.get("mock_total", 1974))]
page.render_prompt_preview = lambda prompt: None

def render_results(results, document, model, **kwargs):
    st.session_state["mock_rendered"] = {
        "count": len(results), "filename": document.name, "model": model,
        "chunk_size": kwargs["chunk_size"],
        "status": kwargs["analysis_progress"]["status"],
        "total": kwargs["analysis_progress"]["total_chunks"],
    }
    st.text("Result count: " + str(len(results)))

page.render_pdf_results_section = render_results
page.render_pdf_analysis_page(st.session_state.get("mock_api_key", "test-key"), st.session_state.get("mock_model", "gemini-3.5-flash"), st.session_state.get("mock_provider", "Google Gemini"))
'''


class DocumentAnalysisPageTests(unittest.TestCase):
    def setUp(self):
        self.apps = []

    def tearDown(self):
        for app in self.apps:
            app.session_state["mock_control"]["release"].set()
            app.session_state["mock_service"].close()
            app.session_state["mock_directory"].cleanup()

    def make_app(self):
        app = AppTest.from_string(APP, default_timeout=30).run()
        self.apps.append(app)
        self.assertEqual(len(app.exception), 0)
        return app

    def wait_finished(self, app):
        service = app.session_state["mock_service"]
        token = app.session_state["pdf_analysis_job_token"]
        # Durable writes for 1,974 chunks vary with filesystem load. Verify
        # eventual completion and call counts rather than a 30-second speed target.
        deadline = time.monotonic() + 90
        while service.is_running(token) and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(service.is_running(token), "Background worker did not finish")
        app.run()
        self.assertEqual(len(app.exception), 0)

    def run_button(self, app):
        return app.button(key="analyze_pdf_button")

    def test_partial_run_resume_and_snapshot_metadata(self):
        app = self.make_app()
        self.run_button(app).click().run()
        self.wait_finished(app)
        state = app.session_state["pdf_analysis_state"]
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(len(state["results"]), 104)
        self.assertEqual(state["total_chunks"], 1974)
        self.assertEqual(app.session_state["mock_control"]["calls"].count(104), 1)
        self.assertEqual(app.session_state["mock_rendered"]["model"], "gemini-3.5-flash")
        self.assertIn("Resume", self.run_button(app).label)
        app.session_state["mock_control"]["fail"] = False
        self.run_button(app).click().run()
        self.wait_finished(app)
        self.assertEqual(app.session_state["pdf_analysis_state"]["status"], "complete")
        self.assertEqual(app.session_state["mock_rendered"]["count"], 1974)
        calls = app.session_state["mock_control"]["calls"]
        self.assertEqual(calls.count(0), 1)
        self.assertEqual(calls.count(103), 1)
        self.assertEqual(calls.count(104), 2)
        app.session_state["mock_model"] = "different-model"
        app.number_input(key="pdf_chunk_size_input").set_value(300).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["mock_rendered"]["model"], "gemini-3.5-flash")
        self.assertEqual(app.session_state["mock_rendered"]["chunk_size"], 200)

    def test_resume_requires_matching_settings_and_clear_removes_report(self):
        app = self.make_app()
        self.run_button(app).click().run()
        self.wait_finished(app)
        token = app.session_state["pdf_analysis_job_token"]
        app.session_state["mock_model"] = "different-model"
        app.run()
        self.assertNotIn("Resume", self.run_button(app).label)
        app.session_state["mock_model"] = "gemini-3.5-flash"
        app.run()
        self.assertIn("Resume", self.run_button(app).label)
        app.session_state["pdf_report_bytes"] = b"stale report"
        app.session_state["pdf_report_fingerprint"] = "stale fingerprint"
        app.button(key="clear_pdf_cache").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertNotIn("pdf_analysis_state", app.session_state)
        self.assertNotIn("pdf_report_bytes", app.session_state)
        self.assertNotIn("pdf_report_fingerprint", app.session_state)
        self.assertIsNone(app.session_state["mock_service"].store.load(token))
        self.assertNotIn("document_analysis", app.query_params)

    def test_rerun_keeps_background_task_and_stop_preserves_results(self):
        app = self.make_app()
        control = app.session_state["mock_control"]
        control["block"] = True
        self.run_button(app).click().run()
        self.assertTrue(control["entered"].wait(2))
        token = app.session_state["pdf_analysis_job_token"]
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.session_state["mock_service"].is_running(token))
        self.assertTrue(self.run_button(app).disabled)
        self.assertTrue(app.button(key="clear_pdf_cache").disabled)
        app.button(key="stop_pdf_analysis").click().run()
        control["release"].set()
        self.wait_finished(app)
        state = app.session_state["pdf_analysis_state"]
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(len(state["results"]), 1)
        self.assertEqual(control["calls"], [0])
        self.assertIn("stopped", state["error"].lower())

    def test_keyless_local_endpoint_can_start_and_complete(self):
        app = self.make_app()
        app.session_state["mock_api_key"] = None
        app.session_state["mock_provider"] = "Local vLLM"
        app.session_state["mock_total"] = 2
        app.session_state["mock_control"]["fail"] = False
        app.run()
        self.run_button(app).click().run()
        self.wait_finished(app)
        self.assertEqual(app.session_state["pdf_analysis_state"]["status"], "complete")
        self.assertEqual(app.session_state["mock_control"]["calls"], [0, 1])
        self.assertEqual(app.session_state["pdf_analysis_state"]["settings"]["base_url"], "http://localhost:8000/v1")

    def test_recovery_link_restores_without_current_document_settings(self):
        app = self.make_app()
        self.run_button(app).click().run()
        self.wait_finished(app)
        token = app.session_state["pdf_analysis_job_token"]
        # A reconnect has no session checkpoint; the URL token restores the disk state.
        del app.session_state["pdf_analysis_job_token"]
        del app.session_state["pdf_analysis_state"]
        del app.session_state["pdf_analysis_results"]
        app.session_state["mock_model"] = "different-model"
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["pdf_analysis_job_token"], token)
        self.assertEqual(app.session_state["mock_rendered"]["count"], 104)
        self.assertFalse(app.button(key="resume_saved_pdf_analysis").disabled)


if __name__ == "__main__":
    unittest.main()
