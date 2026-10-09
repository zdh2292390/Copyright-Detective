"""Actual MIN-K controls and canonical samples survive cloud recovery."""
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.analysis_checkpoints import _json
from src.pages.unlearning_detection import load_bookmia_dataset
from test_resumable_analysis import MemoryStore


ENTRY = """
import streamlit as st
import streamlit.elements.lib.policies as policies
policies._shown_default_value_warning = False
from src.analysis_recovery_ui import apply_pending_analysis_restore, render_analysis_recovery
from src.pages.unlearning_detection import render_min_k_prob_page
from src.resumable_analysis import page_analysis_scope

apply_pending_analysis_restore()
with page_analysis_scope("Unlearning Detection", st.session_state):
    render_analysis_recovery("Unlearning Detection")
    render_min_k_prob_page("ephemeral-key", st.session_state.get("sidebar_openai_model_selectbox", "gpt-4o-mini"), "OpenAI")
"""


class SizeCheckedMemoryStore(MemoryStore):
    """Use the real cloud boundary validation with deterministic in-memory RPCs."""

    def create_task(self, **kwargs):
        kwargs["source"] = _json(kwargs.get("source", {}), "source")
        return super().create_task(**kwargs)


def seed_bookmia(app, selected, full_data):
    settings = {
        "min_k_input_mode": "Predefined Examples",
        "min_k_model_path": "gpt2",
        "min_k_predefined_dataset_type": "BookMIA",
        "min_k_predefined_dataset_type_select": "BookMIA",
        "min_k_predefined_dataset_name": "BookMIA",
        "min_k_predefined_dataset_length": 512,
        "min_k_predefined_sample_count": len(selected),
        "min_k_predefined_percentage": 17,
        "min_k_predefined_max_tokens": 150,
        "min_k_predefined_batch_data": deepcopy(selected),
        "min_k_full_data_bookmia": full_data,
        "sidebar_openai_model_selectbox": "gpt-4o-mini",
    }
    for key, value in settings.items():
        app.session_state[key] = value
    return app


class ActualMinKRecoveryTests(TestCase):
    def assert_clean(self, app):
        self.assertFalse(app.exception)
        self.assertEqual([item.value for item in app.warning if "also had its value set via the Session State API" in item.value], [])

    def test_actual_bookmia_preview_cache_does_not_exceed_cloud_source_limit(self):
        full_data = load_bookmia_dataset()
        self.assertGreater(len(full_data), 1000)
        selected = [full_data[0], next(item for item in full_data if item["label"] != full_data[0]["label"])]
        store = SizeCheckedMemoryStore()
        with patch("src.resumable_analysis.get_cloud_store_for_current_user", return_value=store), patch("src.analysis_recovery_ui.get_cloud_store_for_current_user", return_value=store), patch("src.pages.unlearning_detection._get_completion_logprobs_uncached", return_value=([-0.2, -0.8], None)) as completion, patch("src.pages.unlearning_detection._display_evaluation_results"):
            app = seed_bookmia(AppTest.from_string(ENTRY, default_timeout=30), selected, full_data).run()
            self.assert_clean(app)
            app.button(key="run_min_k_prob_analysis_button").click().run()
            self.assert_clean(app)
            self.assertEqual(len(store.tasks), 1, "A full dataset preview must not block creation for two selected samples")
            task = next(iter(store.tasks.values()))
            self.assertEqual(task["status"], "complete")
            snapshot = task["source"]["initial_session"]
            self.assertNotIn("min_k_full_data_bookmia", snapshot)
            self.assertEqual(snapshot["min_k_predefined_batch_data"], selected)
            self.assertEqual(snapshot["min_k_predefined_percentage"], 17)
            self.assertEqual(snapshot["min_k_predefined_max_tokens"], 150)
            self.assertGreater(completion.call_count, 0)

    def test_fresh_browser_restores_original_selected_samples_and_parameters(self):
        selected = [{"text": "Original nonmember passage", "label": 0}, {"text": "Original member passage", "label": 1}]
        store = SizeCheckedMemoryStore()
        requests = []
        failed_once = [False]

        def complete(prompt, api_key, model_name, provider, temperature, top_p, *args, **kwargs):
            requests.append((prompt, model_name, kwargs["max_output_tokens"], temperature, top_p))
            if prompt == selected[1]["text"] and not failed_once[0]:
                failed_once[0] = True
                raise RuntimeError("Simulated interrupted request")
            return "answer", [{"token": "answer", "logprob": -0.4}]

        with patch("src.resumable_analysis.get_cloud_store_for_current_user", return_value=store), patch("src.analysis_recovery_ui.get_cloud_store_for_current_user", return_value=store), patch("src.pages.unlearning_detection.load_bookmia_dataset", return_value=selected), patch("src.pages.unlearning_detection._get_completion_logprobs_uncached", return_value=([], "Completion endpoint unsupported")), patch("src.direct_recall.comparison._get_llm_completion_uncheckpointed", side_effect=complete), patch("src.pages.unlearning_detection._display_evaluation_results"):
            first = seed_bookmia(AppTest.from_string(ENTRY, default_timeout=30), selected, selected).run()
            first.button(key="run_min_k_prob_analysis_button").click().run()
            self.assert_clean(first)
            task_id = next(iter(store.tasks))
            self.assertEqual(store.tasks[task_id]["status"], "incomplete")
            self.assertEqual(first.session_state["min_k_predefined_batch_progress"]["completed"], 1)
            snapshot = store.tasks[task_id]["source"]["initial_session"]
            self.assertNotIn("run_min_k_prob_analysis_button", snapshot)
            self.assertNotIn("min_k_load_selected", snapshot)

            fresh = AppTest.from_string(ENTRY, default_timeout=30)
            fresh.session_state["min_k_input_mode"] = "User Input"
            fresh.session_state["min_k_model_path"] = "gpt2-xl"
            fresh.session_state["min_k_user_input_prompt"] = "Other document"
            fresh.session_state["min_k_user_input_percentage"] = 3
            fresh.session_state["min_k_user_input_max_tokens"] = 50
            fresh.session_state["sidebar_openai_model_selectbox"] = "gpt-4o"
            fresh.session_state["qa_input_text"] = "Other page state must stay"
            fresh.run()
            self.assert_clean(fresh)
            fresh.button(key=f"_analysis_resume:{task_id}").click().run()
            self.assert_clean(fresh)
            self.assertEqual(fresh.radio(key="min_k_input_mode").value, "Predefined Examples")
            self.assertEqual(fresh.selectbox(key="min_k_predefined_dataset_type_select").value, "BookMIA")
            self.assertEqual(fresh.text_input(key="min_k_model_path_input").value, "gpt2")
            self.assertEqual(fresh.number_input(key="min_k_predefined_examples_percentage_input").value, 17)
            self.assertEqual(fresh.number_input(key="min_k_predefined_examples_max_tokens_input").value, 150)
            self.assertEqual(fresh.session_state["min_k_predefined_batch_data"], selected)
            self.assertEqual(fresh.session_state["qa_input_text"], "Other page state must stay")
            self.assertEqual(requests, [(selected[0]["text"], "gpt-4o-mini", 150, 1.0, 1.0), (selected[1]["text"], "gpt-4o-mini", 150, 1.0, 1.0), (selected[1]["text"], "gpt-4o-mini", 150, 1.0, 1.0)])
            self.assertEqual(store.tasks[task_id]["status"], "complete")
            self.assertEqual(fresh.session_state["min_k_predefined_batch_progress"]["completed"], 2)

            fresh.run()
            fresh.button(key=f"_analysis_resume:{task_id}").click().run()
            self.assert_clean(fresh)
            self.assertEqual(len(requests), 3, "Saved results rebuild must make no new model requests")
