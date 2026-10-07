"""Document report coverage and cache regressions; no model calls are made."""

import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PyPDF2 import PdfReader
import src.pdf_preview as report


def rows(count):
    return [("prefix context", "target text", "generated text", {"rouge_l": 0.1}) for _ in range(count)]


def pdf_text(results, progress=None):
    data = report.generate_document_memorization_pdf_report(
        results, "gemini-3.5-flash", "Normal Continuation", 0.7, 0.9, 200,
        "source.txt", analysis_progress=progress,
    )
    reader = PdfReader(io.BytesIO(data))
    return " ".join(" ".join(page.extract_text() or "" for page in reader.pages).split())


class DocumentReportTests(unittest.TestCase):
    def test_partial_104_of_1974_is_not_document_wide(self):
        results = rows(104)
        progress = {"total_chunks": 1974, "status": "incomplete", "failures": {104: "429 quota exceeded"}, "error": "429 quota exceeded"}
        scope = report._document_analysis_scope(results, progress)
        self.assertFalse(scope["complete"])
        self.assertEqual(scope["pending"], 1869)
        self.assertEqual(scope["failed"], 1)
        text = pdf_text(results, progress)
        self.assertIn("INCOMPLETE ANALYSIS: 104 of 1974 planned chunks were analyzed", text)
        self.assertIn("5.3%", text)
        self.assertIn("429 quota exceeded", text)
        self.assertNotIn("Document-wide analysis", text)
        self.assertNotIn("across the entire document", text)

    def test_complete_requires_every_chunk_to_succeed(self):
        progress = {"total_chunks": 1974, "status": "complete", "failures": {}}
        scope = report._document_analysis_scope(rows(1974), progress)
        self.assertTrue(scope["complete"])
        text = pdf_text(rows(1974), progress)
        self.assertIn("100.0%", text)
        self.assertNotIn("INCOMPLETE ANALYSIS", text)
        self.assertNotIn("COMPLETION UNVERIFIED", text)

    def test_failed_final_chunk_still_marks_report_incomplete(self):
        progress = {"total_chunks": 104, "status": "complete", "failures": {103: "empty content"}}
        scope = report._document_analysis_scope(rows(103), progress)
        self.assertFalse(scope["complete"])
        self.assertEqual(scope["pending"], 0)
        self.assertIn("1 failed and 0 remain pending", pdf_text(rows(103), progress))

    def test_legacy_results_have_unverified_coverage(self):
        scope = report._document_analysis_scope(rows(104))
        self.assertIsNone(scope["total"])
        self.assertFalse(scope["complete"])
        text = pdf_text(rows(104))
        self.assertIn("COMPLETION UNVERIFIED", text)
        self.assertNotIn("Document-wide analysis", text)

    def test_zero_success_report_retains_failure_reason(self):
        progress = {"total_chunks": 1, "status": "incomplete", "failures": {0: "403 forbidden"}, "error": "403 forbidden"}
        text = pdf_text([], progress)
        self.assertIn("0 of 1 planned chunks", text)
        self.assertIn("No data available", text)
        self.assertIn("No analyzed segments", text)
        self.assertNotIn("Top 0 Highest Risk Segments", text)
        self.assertIn("403 forbidden", text)
        self.assertIn("0.0%", text)

    def test_cache_invalidates_for_result_scope_and_setting_changes(self):
        fake_st = SimpleNamespace(session_state={})
        results = rows(1)
        progress = {"total_chunks": 2, "status": "incomplete", "failures": {}}
        document = SimpleNamespace(name="source.txt")
        with patch.object(report, "st", fake_st), patch.object(report, "generate_document_memorization_pdf_report", return_value=b"pdf") as generate, patch.object(report, "render_pdf_preview_with_blob"):
            def render(temperature=0.7, chunk_size=200):
                report._render_document_report_preview(results, document, "gemini-3.5-flash", "Normal Continuation", temperature, 0.9, chunk_size, progress)
            render()
            render()
            self.assertEqual(generate.call_count, 1)
            progress["error"] = "interrupted"
            render()
            self.assertEqual(generate.call_count, 2)
            results.append(rows(1)[0])
            render()
            self.assertEqual(generate.call_count, 3)
            render(temperature=0.2)
            self.assertEqual(generate.call_count, 4)
            render(temperature=0.2, chunk_size=300)
            self.assertEqual(generate.call_count, 5)
            results[0][3]["rouge_l"] = 0.9
            render(temperature=0.2, chunk_size=300)
            self.assertEqual(generate.call_count, 6)

    def test_zero_success_ui_displays_coverage_without_ranking(self):
        fake_st = SimpleNamespace(session_state={}, warning=Mock(), info=Mock(), caption=Mock())
        progress = {"total_chunks": 1, "status": "incomplete", "failures": {0: "empty content"}, "error": "empty content"}
        with patch.object(report, "st", fake_st), patch.object(report, "_render_document_report_preview") as preview:
            report.render_pdf_results_section([], None, "gemini-3.5-flash", default_score_type="ROUGE-L", default_top_k=5, continuation_method="Normal Continuation", temperature=0.7, top_p=0.9, analysis_progress=progress, chunk_size=200)
        self.assertIn("0/1 chunks analyzed", fake_st.warning.call_args.args[0])
        self.assertIn("empty content", fake_st.caption.call_args.args[0])
        preview.assert_called_once()


if __name__ == "__main__":
    unittest.main()
