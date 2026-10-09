"""Keyed widgets retain fresh defaults and restored inputs without warnings."""
from unittest import TestCase
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.model_catalog import MODEL_CONFIG, model_replacement


PREFIX = """
import streamlit as st
import streamlit.elements.lib.policies as policies
policies._shown_default_value_warning = False
"""
QA = """
from src.ui import render_qa_based_detection
render_qa_based_detection("", "gpt-4o-mini", "OpenAI")
"""
SC = """
from src.pages.single_choice_detection import render_single_choice_detection_page
render_single_choice_detection_page("", "gpt-4o-mini", "OpenAI")
"""
MIN_K = """
from src.pages.unlearning_detection import render_min_k_prob_page
render_min_k_prob_page("", "gpt-4o-mini", "OpenAI")
"""
SIDEBAR = """
from src.sidebar_utils import render_model_selectbox
from src.model_catalog import MODEL_CONFIG
render_model_selectbox("OpenAI", MODEL_CONFIG["OpenAI"])
"""
DOC = """
from src.pages.document_memorization_detection import render_pdf_analysis_page
render_pdf_analysis_page("", "gpt-4o-mini", "OpenAI")
"""


class WidgetStateDefaultsTests(TestCase):
    def make_app(self, body, state=None):
        app = AppTest.from_string(PREFIX + body, default_timeout=40)
        for key, value in (state or {}).items():
            app.session_state[key] = value
        return app.run()

    def assert_clean(self, app):
        self.assertFalse(app.exception)
        duplicate = [item.value for item in app.warning if "also had its value set via the Session State API" in item.value]
        self.assertEqual(duplicate, [])

    def test_qa_first_input_render_keeps_generation_defaults_and_user_changes(self):
        app = self.make_app(QA, {"qa_source_mode": "Input Text"})
        self.assert_clean(app)
        self.assertEqual(app.number_input(key="num_qa_pairs").value, 5)
        self.assertEqual(app.slider(key="qa_gen_temperature").value, .7)
        self.assertEqual(app.slider(key="qa_gen_top_p").value, .9)
        app.slider(key="qa_gen_temperature").set_value(.4).run()
        app.number_input(key="num_qa_pairs").set_value(8).run()
        self.assert_clean(app)
        self.assertEqual(app.slider(key="qa_gen_temperature").value, .4)
        self.assertEqual(app.number_input(key="num_qa_pairs").value, 8)

    def test_qa_restored_widget_values_take_priority_over_canonical_defaults(self):
        app = self.make_app(QA, {
            "qa_source_mode": "Input Text",
            "qa_generated_qa_pairs": [{"question": "A question", "answer": "An answer"}],
            "qa_num_qa_pairs": 2, "num_qa_pairs": 9,
            "qa_gen_temperature": .35, "qa_gen_top_p": .65,
            "qa_num_eval_runs": 1, "num_eval_runs": 7,
            "qa_eval_temperature": .7, "eval_temperature": .25,
            "qa_eval_top_p": .9, "eval_top_p": .55,
            "qa_enable_llm_judge": True, "enable_llm_judge": True,
        })
        self.assert_clean(app)
        self.assertEqual(app.number_input(key="num_qa_pairs").value, 9)
        self.assertEqual(app.number_input(key="num_eval_runs").value, 7)
        self.assertEqual(app.slider(key="qa_gen_temperature").value, .35)
        self.assertEqual(app.slider(key="qa_gen_top_p").value, .65)
        self.assertEqual(app.slider(key="eval_temperature").value, .25)
        self.assertEqual(app.slider(key="eval_top_p").value, .55)
        self.assertTrue(app.checkbox(key="enable_llm_judge").value)
        app.run()
        self.assert_clean(app)
        self.assertEqual(app.number_input(key="num_eval_runs").value, 7)

    def test_qa_sleek_restored_mode_and_sampling_are_preserved(self):
        mode = "Step-by-step Leaking and Extraction"
        app = self.make_app(QA, {
            "qa_source_mode": "Input Text",
            "qa_generated_qa_pairs": [{"question": "A question", "answer": "An answer"}],
            "qa_evaluation_mode": mode, "qa_evaluation_mode_radio": mode,
            "sleek_num_eval_runs": 6, "sleek_eval_temperature": .3,
            "sleek_eval_top_p": .6,
        })
        self.assert_clean(app)
        self.assertEqual(app.radio(key="qa_evaluation_mode_radio").value, mode)
        self.assertEqual(app.number_input(key="sleek_num_eval_runs").value, 6)
        self.assertEqual(app.slider(key="sleek_eval_temperature").value, .3)
        self.assertEqual(app.slider(key="sleek_eval_top_p").value, .6)

    def test_single_choice_distractors_keep_fresh_default_and_restored_count(self):
        with patch("src.pages.single_choice_detection.render_temperature_top_p", return_value=(.7, .9)):
            fresh = self.make_app(SC, {"sc_source_mode": "Input Text"})
            self.assert_clean(fresh)
            self.assertEqual(fresh.number_input(key="sc_num_distractors").value, 3)
            restored = self.make_app(SC, {"sc_source_mode": "Input Text", "sc_num_distractors": 4})
            self.assert_clean(restored)
            self.assertEqual(restored.number_input(key="sc_num_distractors").value, 4)

    def test_min_k_restored_text_and_numeric_values_take_priority(self):
        app = self.make_app(MIN_K, {
            "min_k_input_mode": "User Input",
            "min_k_model_path": "gpt2", "min_k_model_path_input": "gpt2-xl",
            "min_k_deploy_agent_url": "https://original.test", "min_k_deploy_agent_url_input": "https://restored.test",
            "min_k_user_input_prompt": "Canonical source", "min_k_user_input_prompt_input": "Restored original source",
            "min_k_user_input_percentage": 17, "min_k_user_input_percentage_input": 31,
            "min_k_user_input_max_tokens": 150, "min_k_user_input_max_tokens_input": 250,
        })
        self.assert_clean(app)
        self.assertEqual(app.text_input(key="min_k_model_path_input").value, "gpt2-xl")
        self.assertEqual(app.text_input(key="min_k_deploy_agent_url_input").value, "https://restored.test")
        self.assertEqual(app.text_area(key="min_k_user_input_prompt_input").value, "Restored original source")
        self.assertEqual(app.number_input(key="min_k_user_input_percentage_input").value, 31)
        self.assertEqual(app.number_input(key="min_k_user_input_max_tokens_input").value, 250)

    def test_min_k_restored_wikimia_length_keeps_original_selected_samples(self):
        selected = [{"text": "Original selected nonmember", "label": 0}, {"text": "Original selected member", "label": 1}]
        state = {"min_k_input_mode": "Predefined Examples", "min_k_model_path": "gpt2",
                 "min_k_predefined_dataset_type": "WikiMIA", "min_k_predefined_dataset_type_select": "WikiMIA",
                 "min_k_predefined_wikimia_length": 64, "min_k_predefined_wikimia_length_select": 64,
                 "min_k_predefined_dataset_length": 64, "min_k_predefined_batch_data": selected}
        with patch("src.pages.unlearning_detection.load_wikimia_dataset", return_value=selected):
            app = self.make_app(MIN_K, state)
            self.assert_clean(app)
            self.assertEqual(app.selectbox(key="min_k_predefined_wikimia_length_select").value, 64)
            self.assertEqual(app.session_state["min_k_predefined_batch_data"], selected)

    def test_document_restored_custom_prompt_keeps_template_and_chunk_size(self):
        state = {"pdf_source_mode": "Upload Document", "pdf_chunk_size": 200,
                 "pdf_chunk_size_input": 350, "pdf_continuation_method": "Custom Prompt",
                 "pdf_custom_prompt_text": "Canonical {input_text}",
                 "pdf_custom_prompt": "Restored original {input_text}"}
        with patch("src.pages.document_memorization_detection._prepare_document_cloud", return_value=False), patch("src.pages.document_memorization_detection._list_example_documents", return_value=[]), patch("src.pages.document_memorization_detection._render_document_recovery_list"), patch("src.pages.document_memorization_detection.render_temperature_top_p", return_value=(.7, .9)):
            app = self.make_app(DOC, state)
            self.assert_clean(app)
            self.assertEqual(app.number_input(key="pdf_chunk_size_input").value, 350)
            self.assertEqual(app.text_area(key="pdf_custom_prompt").value, "Restored original {input_text}")

    def test_sidebar_fresh_recovered_and_retired_model_selection(self):
        config = MODEL_CONFIG["OpenAI"]
        key = config["key"]
        fresh = self.make_app(SIDEBAR)
        self.assert_clean(fresh)
        self.assertEqual(fresh.selectbox(key=key).value, config["models"][config["default_index"]])
        restored = self.make_app(SIDEBAR, {key: "gpt-5.4"})
        self.assert_clean(restored)
        self.assertEqual(restored.selectbox(key=key).value, "gpt-5.4")
        retired = self.make_app(SIDEBAR, {key: "chatgpt-4o-latest"})
        self.assert_clean(retired)
        self.assertEqual(retired.selectbox(key=key).value, model_replacement("OpenAI", "chatgpt-4o-latest"))
