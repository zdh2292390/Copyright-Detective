"""UI race and account-change regressions without model calls."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.document_analysis import new_analysis_state
from src.document_checkpoints import CheckpointError
import src.pages.document_memorization_detection as page


class DocumentPageEdgeTests(unittest.TestCase):
    def settings(self):
        return {"filename": "test.txt", "model": "gemini-3.5-flash", "provider": "Google Gemini",
                "continuation_method": "Normal Continuation", "chunk_size": 200,
                "temperature": 0.7, "top_p": 0.9}

    def test_worker_finishing_between_reads_renders_final_results(self):
        running = new_analysis_state("a" * 64, self.settings(), 1)
        running["status"] = "running"
        complete = new_analysis_state("a" * 64, self.settings(), 1)
        complete["status"] = "complete"
        complete["results"][0] = ("prefix", "target", "generation", {"rouge_l": 0.1})
        service = SimpleNamespace(get=Mock(side_effect=[running, complete]), is_running=Mock(return_value=False))
        ui = SimpleNamespace(session_state={}, markdown=Mock(), caption=Mock())
        with patch.object(page, "DOCUMENT_JOBS", service), patch.object(page, "st", ui), patch.object(page, "_render_saved_pdf_results") as render:
            page._render_document_job("token", "test-key", "Google Gemini")
        render.assert_called_once_with(complete)
        self.assertEqual(ui.session_state["pdf_analysis_state"]["status"], "complete")

    def test_clear_after_account_switch_handles_owner_error(self):
        ui = SimpleNamespace(session_state={"pdf_analysis_job_token": "token", "user_id": "new-account"}, error=Mock(), warning=Mock())
        service = SimpleNamespace(is_running=Mock(return_value=True), stop=Mock(side_effect=CheckpointError("Wrong account")), delete=Mock())
        with patch.object(page, "DOCUMENT_JOBS", service), patch.object(page, "st", ui):
            page._clear_pdf_cache()
        self.assertIn("Wrong account", ui.error.call_args.args[0])
        service.delete.assert_not_called()

    def test_owner_failure_does_not_show_cached_previous_account_document(self):
        ui = SimpleNamespace(session_state={"pdf_analysis_state": {"owner_id": "previous"}, "pdf_analysis_results": ["private"], "pdf_report_bytes": b"private"}, query_params={"document_analysis": "token"}, error=Mock())
        service = SimpleNamespace(get=Mock(side_effect=CheckpointError("Wrong account")))
        with patch.object(page, "DOCUMENT_JOBS", service), patch.object(page, "st", ui):
            self.assertIsNone(page._restore_document_job())
        self.assertNotIn("pdf_analysis_results", ui.session_state)
        self.assertNotIn("pdf_analysis_state", ui.session_state)
        self.assertNotIn("pdf_report_bytes", ui.session_state)


if __name__ == "__main__":
    unittest.main()
