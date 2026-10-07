"""QA screen regressions using real batch scoring and mocked outbound providers."""

import unittest
from streamlit.testing.v1 import AppTest


APP = '''
import streamlit as st
from unittest.mock import patch
import src.ui as ui
import src.direct_recall.knowledge_qa as qa

pairs = [
    {"question": "First capital?", "answer": "London"},
    {"question": "Second capital?", "answer": "Paris"},
]
def complete(*args, **kwargs):
    index = st.session_state.get("mock_completion_index", 0)
    st.session_state["mock_completion_index"] = index + 1
    if st.session_state.get("mock_all_failed", False) or index % 2 == 0:
        return "Error calling API: 429 unavailable"
    return "Paris"

def report(*args, **kwargs):
    calls = st.session_state.get("mock_report_calls", [])
    calls.append({"model": args[3], "pairs": args[2], "metrics": args[1]})
    st.session_state["mock_report_calls"] = calls
    if st.session_state.get("mock_report_fails", False):
        raise OSError("report storage unavailable")
    return b"mock pdf"

def diff(truth, answer, **kwargs):
    calls = st.session_state.get("mock_diff_calls", [])
    calls.append({"truth": truth, "answer": answer})
    st.session_state["mock_diff_calls"] = calls
    st.text("Scored answer: " + truth + " -> " + answer)

with patch.object(ui, "list_knowledge_book_titles", return_value=["Test book"]), \
     patch.object(ui, "get_knowledge_question_bank_by_title", return_value=pairs), \
     patch.object(ui, "generate_open_ended_question_pdf_report", side_effect=report), \
     patch.object(ui, "render_pdf_preview_with_blob", return_value=None), \
     patch.object(ui, "render_direct_recall_diff", side_effect=diff), \
     patch.object(qa, "get_llm_completion", side_effect=complete):
    ui.render_qa_based_detection("test-key", st.session_state.get("mock_model", "original-model"), "OpenAI")
'''


class KnowledgeQAPageTests(unittest.TestCase):
    def make_app(self):
        app = AppTest.from_string(APP, default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def evaluate(self, app):
        app.button(key="run_knowledge_eval_button").click().run()
        self.assertEqual(len(app.exception), 0)

    def test_failed_answers_remain_aligned_and_success_is_scored_against_its_own_question(self):
        app = self.make_app()
        self.evaluate(app)
        rows = app.session_state["qa_evaluation_results"][0]
        self.assertEqual(len(rows), 2)
        self.assertIn("error", rows[0])
        self.assertEqual(rows[1]["question"], "Second capital?")
        self.assertEqual(rows[1]["f1"], 1)
        self.assertEqual(app.session_state["mock_diff_calls"], [{"truth": "Paris", "answer": "Paris"}])
        self.assertTrue(any("1 successful; 1 failed" in warning.value for warning in app.warning))

    def test_all_failed_has_no_low_risk_assessment(self):
        app = self.make_app()
        app.session_state["mock_all_failed"] = True
        self.evaluate(app)
        self.assertTrue(any("No memorization assessment" in error.value for error in app.error))
        self.assertFalse(any("Low Risk" in expander.label for expander in app.expander))
        self.assertEqual(len(app.session_state["qa_evaluation_results"][0]), 2)

    def test_report_regeneration_after_model_switch_uses_saved_snapshot(self):
        app = self.make_app()
        self.evaluate(app)
        app.session_state["mock_model"] = "changed-model"
        app.session_state["qa_pdf_report_bytes"] = None
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["mock_report_calls"][-1]["model"], "original-model")
        self.assertEqual(app.session_state["qa_evaluation_metadata"]["model"], "original-model")
        self.assertEqual(len(app.session_state["qa_evaluation_results"][0]), 2)

    def test_pdf_exception_keeps_results_visible(self):
        app = self.make_app()
        app.session_state["mock_report_fails"] = True
        self.evaluate(app)
        self.assertEqual(len(app.session_state["qa_evaluation_results"][0]), 2)
        self.assertTrue(any("PDF report could not be generated" in warning.value for warning in app.warning))
        self.assertEqual(app.session_state["mock_diff_calls"], [{"truth": "Paris", "answer": "Paris"}])


if __name__ == "__main__":
    unittest.main()
