"""SLEEK QA screen coverage and saved report metadata regressions."""
import unittest
from streamlit.testing.v1 import AppTest

APP = '''
import streamlit as st
from unittest.mock import patch
import src.ui as ui
import src.direct_recall.sleek_attack as sleek
st.session_state.setdefault('qa_evaluation_mode_radio', 'Step-by-step Leaking and Extraction')
pairs = [{'question':'Question?', 'answer':'reference answer'}]
def complete(*args, **kwargs):
    index = st.session_state.get('mock_call_index', 0)
    st.session_state['mock_call_index'] = index + 1
    if st.session_state.get('mock_all_failed') or index == 0:
        return 'Error: 503 temporarily unavailable'
    if index == 1:
        return '[{"question":"sub?", "category":"Direct"}]'
    return '{"final_answer":"reference answer", "sub_question_answers":[]}'
def report(*args, **kwargs):
    calls = st.session_state.get('mock_report_models', [])
    calls.append(args[1])
    st.session_state['mock_report_models'] = calls
    if st.session_state.get('mock_report_failed'):
        raise OSError('PDF unavailable')
    return b'mock pdf'
with patch.object(ui, 'list_knowledge_book_titles', return_value=['Test book']), \
     patch.object(ui, 'get_knowledge_question_bank_by_title', return_value=pairs), \
     patch.object(ui, 'generate_sleek_attack_pdf_report', side_effect=report), \
     patch.object(ui, 'render_direct_recall_diff'), \
     patch.object(ui, 'render_pdf_preview_with_blob'), \
     patch.object(sleek, 'get_llm_completion', side_effect=complete):
    ui.render_qa_based_detection('key', st.session_state.get('mock_model', 'original-model'), 'OpenAI')
'''

class SleekQAPageTests(unittest.TestCase):
    def make_app(self):
        app = AppTest.from_string(APP, default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        return app
    def run_analysis(self, app):
        app.button(key='run_sleek_eval_button').click().run()
        self.assertEqual(len(app.exception), 0)
    def test_partial_failure_is_visible_and_success_score_is_preserved(self):
        app = self.make_app()
        app.number_input(key='sleek_num_eval_runs').set_value(2).run()
        self.run_analysis(app)
        result = app.session_state['qa_sleek_results']
        self.assertEqual(result['successful_evaluations'], 1)
        self.assertEqual(result['failed_evaluations'], 1)
        self.assertEqual(result['avg_rouge_score'], 1.)
        self.assertTrue(any('1 successful; 1 failed' in item.value for item in app.warning))
        self.assertTrue(any('503' in item.value for item in app.error))
    def test_all_failed_has_no_low_leakage_assessment(self):
        app = self.make_app()
        app.session_state['mock_all_failed'] = True
        self.run_analysis(app)
        self.assertTrue(any('No leakage assessment' in item.value for item in app.error))
        self.assertFalse(any('Low Knowledge Leakage' in item.value for item in app.success))
    def test_pdf_failure_and_model_switch_preserve_results_and_original_model(self):
        app = self.make_app()
        app.number_input(key='sleek_num_eval_runs').set_value(2).run()
        app.session_state['mock_report_failed'] = True
        self.run_analysis(app)
        self.assertEqual(app.session_state['qa_sleek_results']['successful_evaluations'], 1)
        self.assertEqual(app.session_state['mock_call_index'], 3)
        app.session_state['mock_report_failed'] = False
        app.session_state['mock_model'] = 'changed-model'
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state['mock_report_models'][-1], 'original-model')
        self.assertEqual(app.session_state['mock_call_index'], 3)

if __name__ == '__main__':
    unittest.main()
