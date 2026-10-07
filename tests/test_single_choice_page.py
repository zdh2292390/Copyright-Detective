
"""Single-choice screen regressions with real batch scoring and mocked providers."""

import unittest
from streamlit.testing.v1 import AppTest


APP = r'''
import streamlit as st
from unittest.mock import patch
import src.pages.single_choice_detection as page
import src.direct_recall.single_choice as sc

if not st.session_state.get("mock_initialized"):
    st.session_state["mock_initialized"] = True
    st.session_state["sc_generated_mcqs"] = [
        {"question": "First question?", "options": [{"label": "A", "text": "correct"}, {"label": "B", "text": "wrong"}], "correct_option": "A"},
        {"question": "Second question?", "options": [{"label": "A", "text": "correct"}, {"label": "B", "text": "wrong"}], "correct_option": "A"},
    ]

def complete(*args, **kwargs):
    index = st.session_state.get("mock_completion_index", 0)
    st.session_state["mock_completion_index"] = index + 1
    if st.session_state.get("mock_all_failed", False) or index % 2 == 0:
        return "Error calling API: TimeoutError"
    return "A"

def report(*args, **kwargs):
    calls = st.session_state.get("mock_report_calls", [])
    calls.append({"model": args[1], "data": args[0]})
    st.session_state["mock_report_calls"] = calls
    if st.session_state.get("mock_report_fails", False):
        raise OSError("report storage unavailable")
    return b"mock pdf"

with patch.object(page, "generate_single_choice_question_pdf_report", side_effect=report), \
     patch.object(page, "render_pdf_preview_with_blob", return_value=None), \
     patch.object(sc, "get_llm_completion", side_effect=complete):
    page.render_single_choice_detection_page("test-key", st.session_state.get("mock_model", "original-model"), "Google Gemini")
'''


class SingleChoicePageTests(unittest.TestCase):
    def make_app(self):
        app = AppTest.from_string(APP, default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def evaluate(self, app):
        app.button(key="sc_run_eval_button").click().run()
        self.assertEqual(len(app.exception), 0)

    def test_failed_row_keeps_question_alignment_and_only_valid_answer_scores(self):
        app = self.make_app()
        self.evaluate(app)
        rows = app.session_state["sc_evaluation_results"][0]
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[0]["is_correct"])
        self.assertEqual(rows[1]["question"], "Second question?")
        self.assertTrue(rows[1]["is_correct"])
        metrics = app.session_state["mock_report_calls"][-1]["data"]["metrics"]
        self.assertEqual(metrics["overall_accuracy"], 1)
        self.assertEqual(metrics["failed_attempts"], 1)

    def test_all_failed_has_no_low_memorization_assessment(self):
        app = self.make_app()
        app.session_state["mock_all_failed"] = True
        self.evaluate(app)
        self.assertTrue(any("No successful model responses" in item.value for item in app.warning))
        self.assertFalse(any("Low memorization signal" in item.value for item in app.success))
        self.assertIsNone(app.session_state["mock_report_calls"][-1]["data"]["metrics"]["overall_accuracy"])

    def test_pdf_exception_keeps_evaluation_results_visible(self):
        app = self.make_app()
        app.session_state["mock_report_fails"] = True
        self.evaluate(app)
        self.assertEqual(len(app.session_state["sc_evaluation_results"][0]), 2)
        self.assertTrue(any("PDF report is unavailable" in item.value for item in app.warning))

    def test_model_switch_report_uses_completed_snapshot(self):
        app = self.make_app()
        self.evaluate(app)
        app.session_state["mock_model"] = "changed-model"
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["mock_report_calls"][-1]["model"], "original-model")
        self.assertEqual(len(app.session_state["sc_evaluation_results"][0]), 2)


if __name__ == "__main__":
    unittest.main()
