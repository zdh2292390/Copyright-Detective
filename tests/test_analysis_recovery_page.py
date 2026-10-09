from unittest import TestCase
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from test_resumable_analysis import MemoryStore


ENTRY = """
import streamlit as st
from src.analysis_recovery_ui import apply_pending_analysis_restore, render_analysis_recovery
from src.resumable_analysis import page_analysis_scope
from src.job_guard import render_run_button, detection_job
from src.direct_recall.comparison import get_llm_completion

apply_pending_analysis_restore()
with page_analysis_scope("Recovery test", st.session_state):
    render_analysis_recovery("Recovery test")
    st.text_input("Source", key="qa_input_text")
    st.selectbox("Model", ["original", "changed"], key="sidebar_openai_model_selectbox")
    st.number_input("Runs", min_value=1, max_value=5, value=2, key="qa_num_eval_runs")
    if render_run_button("Question evaluation", "qa_run_resume_test", "Run"):
        try:
            with detection_job("Question evaluation"):
                outputs = []
                for i in range(int(st.session_state["qa_num_eval_runs"])):
                    outputs.append(get_llm_completion(
                        f'{st.session_state["qa_input_text"]}:{i}', "ephemeral-key",
                        st.session_state["sidebar_openai_model_selectbox"],
                    ))
                st.session_state["qa_evaluation_results"] = outputs
        except RuntimeError as exc:
            st.error(str(exc))
    if st.session_state.get("qa_evaluation_results"):
        st.write(st.session_state["qa_evaluation_results"])
"""


class RecoveryPageTests(TestCase):
    def test_new_browser_restores_frozen_inputs_and_skips_saved_requests(self):
        store = MemoryStore()
        requests = []
        failed_once = [False]

        def complete(prompt, api_key, model_name, *args, **kwargs):
            requests.append((prompt, model_name))
            if prompt.endswith(":1") and not failed_once[0]:
                failed_once[0] = True
                raise RuntimeError("Simulated server interruption")
            return f"saved:{prompt}"

        with patch("src.resumable_analysis.get_cloud_store_for_current_user", return_value=store), patch("src.analysis_recovery_ui.get_cloud_store_for_current_user", return_value=store), patch("src.direct_recall.comparison._get_llm_completion_uncheckpointed", side_effect=complete):
            first = AppTest.from_string(ENTRY, default_timeout=20).run()
            first.text_input(key="qa_input_text").set_value("original document").run()
            first.button(key="qa_run_resume_test").click().run()
            self.assertFalse(first.exception)
            task_id = next(iter(store.tasks))
            self.assertEqual(store.tasks[task_id]["status"], "incomplete")
            self.assertEqual(store.tasks[task_id]["completed_items"], 1)

            fresh = AppTest.from_string(ENTRY, default_timeout=20).run()
            fresh.text_input(key="qa_input_text").set_value("other document").run()
            fresh.selectbox(key="sidebar_openai_model_selectbox").set_value("changed").run()
            fresh.button(key=f"_analysis_resume:{task_id}").click().run()
            self.assertFalse(fresh.exception)
            self.assertEqual(fresh.text_input(key="qa_input_text").value, "original document")
            self.assertEqual(fresh.selectbox(key="sidebar_openai_model_selectbox").value, "original")
            self.assertEqual(fresh.session_state["qa_evaluation_results"], ["saved:original document:0", "saved:original document:1"])
            self.assertEqual(requests, [("original document:0", "original"), ("original document:1", "original"), ("original document:1", "original")])
            self.assertEqual(store.tasks[task_id]["status"], "complete")

            fresh.run()
            fresh.button(key=f"_analysis_resume:{task_id}").click().run()
            self.assertFalse(fresh.exception)
            self.assertEqual(len(requests), 3, "completed result rebuild must make no paid calls")

    def test_viewing_a_page_does_not_create_or_run_tasks(self):
        store = MemoryStore()
        with patch("src.resumable_analysis.get_cloud_store_for_current_user", return_value=store), patch("src.analysis_recovery_ui.get_cloud_store_for_current_user", return_value=store), patch("src.direct_recall.comparison._get_llm_completion_uncheckpointed") as completion:
            app = AppTest.from_string(ENTRY, default_timeout=20).run()
            self.assertFalse(app.exception)
            self.assertEqual(store.tasks, {})
            completion.assert_not_called()
