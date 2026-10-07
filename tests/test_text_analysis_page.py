"""Actual text-page UI regressions with fake providers and report generation."""

import unittest
from streamlit.testing.v1 import AppTest


APP = '''
import streamlit as st
import src.pages.text_memorization_detection as page
st.session_state.setdefault('text_custom_input_text1', 'prefix contents')
st.session_state.setdefault('text_custom_input_text2', 'reference target')
page.render_run_button = lambda label, key, title: st.button(title, key=key)
page.render_prompt_preview = lambda *args, **kwargs: None
page.render_direct_recall_diff = lambda *args, **kwargs: None
page.render_pdf_preview_with_blob = lambda *args, **kwargs: None
page.build_text_memorization_plots = lambda *args, **kwargs: None
page._run_blackbox_analysis_auto = lambda *args, **kwargs: None

def compare(*args, **kwargs):
    calls = st.session_state.setdefault('mock_calls', [])
    calls.append({'args': args, 'kwargs': kwargs})
    count = len(calls)
    if count == st.session_state.get('mock_fail_at'):
        mode = st.session_state.get('mock_failure_mode', 'tuple')
        if mode == 'raise':
            raise TimeoutError('request timed out')
        if mode == 'invalid':
            return object()
        return 'Error calling API: 429 RESOURCE_EXHAUSTED', None
    return 'generated continuation with longer text', {'rouge_l': 0.63, 'jaccard_index': 0.22}

page.compare_texts = compare
page.run_persuasion_probe = compare

def report(results, prompt_type, model, api_key, provider, **kwargs):
    reports = st.session_state.setdefault('mock_reports', [])
    reports.append({'generated': results.get('generated_text'), 'model': model, 'provider': provider, 'prompt_type': prompt_type})
    if st.session_state.get('mock_report_fail'):
        raise RuntimeError('PDF backend unavailable')
    return b'%PDF-fake-report'

page.generate_text_memorization_pdf_report = report
page.render_text_analysis_page(st.session_state.get('mock_api_key', 'test-key'), st.session_state.get('mock_model', 'original-model'), st.session_state.get('mock_provider', 'Google Gemini'))
'''


class TextAnalysisPageTests(unittest.TestCase):
    def make_app(self):
        app = AppTest.from_string(APP, default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def run_analysis(self, app):
        app.button(key='run_snippet_analysis_button').click().run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_partial_tuple_failure_preserves_success_and_unlocks_controls(self):
        app = self.make_app()
        app.number_input(key='text_inference_runs_input').set_value(3).run()
        app.session_state['mock_fail_at'] = 2
        self.run_analysis(app)
        result = app.session_state['text_analysis_results']
        self.assertEqual(result['generated_texts'], ['generated contin'])
        self.assertEqual(result['similarity_scores'], [{'rouge_l': 0.63, 'jaccard_index': 0.22}])
        self.assertEqual(result['analysis_progress']['completed_runs'], 1)
        self.assertEqual(result['analysis_progress']['requested_runs'], 3)
        self.assertEqual(result['analysis_progress']['status'], 'incomplete')
        self.assertIn('Run 2/3 failed', result['analysis_progress']['error'])
        self.assertEqual(len(app.session_state['mock_calls']), 2)
        self.assertFalse(app.session_state['detection_job_running'])
        self.assertTrue(any('Partial analysis: 1/3' in warning.value for warning in app.warning))
        self.assertEqual(app.session_state['mock_reports'][-1]['model'], 'original-model')

    def test_raised_and_invalid_provider_failures_keep_previous_single_results(self):
        for mode in ('raise', 'invalid'):
            with self.subTest(mode=mode):
                app = self.make_app()
                self.run_analysis(app)
                previous = app.session_state['text_analysis_results']
                previous_pdf = app.session_state['text_pdf_report']
                app.session_state['mock_fail_at'] = 2
                app.session_state['mock_failure_mode'] = mode
                self.run_analysis(app)
                self.assertEqual(app.session_state['text_analysis_results'], previous)
                self.assertEqual(app.session_state['text_pdf_report'], previous_pdf)
                self.assertEqual(len(app.session_state['mock_calls']), 2)
                self.assertFalse(app.session_state['detection_job_running'])
                self.assertTrue(any('previous analysis results remain' in error.value for error in app.error))

    def test_report_cache_tracks_new_success_and_uses_captured_model_after_sidebar_changes(self):
        app = self.make_app()
        self.run_analysis(app)
        first_fingerprint = app.session_state['text_pdf_report_fingerprint']
        self.assertEqual(len(app.session_state['mock_reports']), 1)
        app.session_state['mock_model'] = 'changed-model'
        app.run()
        self.assertEqual(len(app.session_state['mock_reports']), 1)
        del app.session_state['text_pdf_report']
        app.run()
        self.assertEqual(app.session_state['mock_reports'][-1]['model'], 'original-model')
        self.run_analysis(app)
        self.assertEqual(app.session_state['mock_reports'][-1]['model'], 'changed-model')
        self.assertNotEqual(app.session_state['text_pdf_report_fingerprint'], first_fingerprint)
        self.assertEqual(app.session_state['text_analysis_results']['metrics_map']['rouge_l'], 0.63)

    def test_pdf_failure_retains_analysis_and_retry_does_not_repeat_inference(self):
        app = self.make_app()
        app.session_state['mock_report_fail'] = True
        self.run_analysis(app)
        self.assertEqual(app.session_state['text_analysis_results']['analysis_progress']['status'], 'complete')
        self.assertEqual(len(app.session_state['mock_calls']), 1)
        self.assertEqual(len(app.session_state['mock_reports']), 1)
        app.run()
        self.assertEqual(len(app.session_state['mock_reports']), 1)
        app.session_state['mock_report_fail'] = False
        app.button(key='text_retry_pdf_report').click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.session_state['mock_calls']), 1)
        self.assertEqual(len(app.session_state['mock_reports']), 2)
        self.assertTrue(app.session_state['text_pdf_report'])

    def test_whitespace_input_is_rejected_and_keyless_local_provider_still_runs(self):
        app = self.make_app()
        app.text_area(key='text_input_text1_widget_Next-Passage Prediction').set_value('   ').run()
        self.run_analysis(app)
        self.assertNotIn('mock_calls', app.session_state)
        self.assertFalse(app.session_state['detection_job_running'])
        app.text_area(key='text_input_text1_widget_Next-Passage Prediction').set_value('prefix contents').run()
        app.session_state['mock_provider'] = 'Local vLLM'
        app.session_state['mock_api_key'] = ''
        self.run_analysis(app)
        self.assertEqual(len(app.session_state['mock_calls']), 1)
        self.assertEqual(app.session_state['text_analysis_results']['user_inputs']['provider'], 'Local vLLM')

    def test_corrupt_cached_parameters_are_recovered_before_widgets_render(self):
        app = AppTest.from_string(APP, default_timeout=30)
        app.session_state['text_inference_runs'] = 'bad cached runs'
        app.session_state['text_temperature'] = float('nan')
        app.session_state['text_top_p'] = 'bad cached probability'
        app.session_state['text_prompt_type_index'] = 'bad cached selection'
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.number_input(key='text_inference_runs_input').value, 1)
        self.assertEqual(app.session_state['text_temperature'], 0.7)
        self.assertEqual(app.session_state['text_top_p'], 0.9)

    def test_clear_cache_resets_widget_values_and_analysis_report(self):
        app = self.make_app()
        app.text_area(key='text_input_text1_widget_Next-Passage Prediction').set_value('changed custom prefix').run()
        app.number_input(key='text_inference_runs_input').set_value(2).run()
        self.run_analysis(app)
        app.button(key='clear_text_memorization_cache').click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIsNone(app.session_state['text_analysis_results'])
        self.assertNotIn('text_pdf_report', app.session_state)
        self.assertEqual(app.number_input(key='text_inference_runs_input').value, 1)
        self.assertEqual(app.text_area(key='text_input_text1_widget_Next-Passage Prediction').value, 'prefix contents')

    def test_user_defined_evaluation_does_not_inherit_few_shot_mode(self):
        app = self.make_app()
        app.selectbox(key='prompt_mode_selector').set_value('Few-Shot').run()
        app.selectbox(key='text_prompt_type_selectbox').set_value('User-Defined Evaluation').run()
        app.text_area(key='custom_user_prompt').set_value('Respond to this complete custom instruction').run()
        app.text_area(key='text_ground_truth_user_defined').set_value('reference target').run()
        self.run_analysis(app)
        request = app.session_state['mock_calls'][0]['kwargs']
        self.assertEqual(request['prompt_type'], 'User-Defined Evaluation')
        self.assertEqual(request['mode'], 'Zero-Shot')
        self.assertEqual(request['temperature'], 0.7)
        self.assertEqual(request['top_p'], 0.9)


if __name__ == '__main__':
    unittest.main()
