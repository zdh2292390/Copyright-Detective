"""Standalone persuasion probe UI failure regressions without outbound calls."""

import unittest
from streamlit.testing.v1 import AppTest


APP = '''
import streamlit as st
from unittest.mock import patch
import src.ui as ui
st.session_state.setdefault('probe_input_text', 'prefix contents')
st.session_state.setdefault('probe_ground_truth', 'reference target')

def probe(*args, **kwargs):
    calls = st.session_state.setdefault('mock_probe_calls', [])
    calls.append({'args': args, 'kwargs': kwargs})
    mode = st.session_state.get('mock_failure_mode', 'valid')
    if mode == 'tuple_error':
        return 'Error: 503 temporarily unavailable', None
    if mode == 'raised_error':
        raise TimeoutError()
    return 'generated continuation', {'rouge_l': 0.63, 'jaccard_index': 0.22, 'levenshtein': 9.0}

def diff(truth, generated, **kwargs):
    st.session_state['mock_probe_diff'] = {'truth': truth, 'generated': generated, 'metrics': kwargs['metrics']}

with patch.object(ui, 'run_persuasion_probe', side_effect=probe), \
     patch.object(ui, 'render_direct_recall_diff', side_effect=diff), \
     patch.object(ui, 'render_prompt_preview', return_value=None):
    ui.render_jailbreak_persuasion_probe_section('test-key', 'test-model', 'OpenAI')
'''


class PersuasionProbePageTests(unittest.TestCase):
    def execute(self, mode):
        app = AppTest.from_string(APP, default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        app.session_state['mock_failure_mode'] = mode
        app.button(key='run_probe_button').click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.session_state['mock_probe_calls']), 1)
        self.assertFalse(app.session_state['detection_job_running'])
        return app

    def test_error_tuple_has_no_completed_claim_or_scored_diff_and_unlocks(self):
        app = self.execute('tuple_error')
        self.assertTrue(any('Probe failed' in error.value and '503' in error.value for error in app.error))
        self.assertFalse(any('Probe completed' in success.value for success in app.success))
        self.assertNotIn('mock_probe_diff', app.session_state)

    def test_raised_empty_timeout_has_no_completed_claim_or_scored_diff_and_unlocks(self):
        app = self.execute('raised_error')
        self.assertTrue(any('Probe failed: TimeoutError' in error.value for error in app.error))
        self.assertFalse(any('Probe completed' in success.value for success in app.success))
        self.assertNotIn('mock_probe_diff', app.session_state)

    def test_valid_metrics_and_request_parameters_remain_unchanged(self):
        app = self.execute('valid')
        self.assertTrue(any('Probe completed' in success.value for success in app.success))
        self.assertEqual(app.session_state['mock_probe_diff'], {
            'truth': 'reference target', 'generated': 'generated continuation',
            'metrics': {'rouge_l': 0.63, 'jaccard_index': 0.22, 'levenshtein': 9.0},
        })
        request = app.session_state['mock_probe_calls'][0]
        self.assertEqual(request['args'], ('test-key', 'test-model', 'OpenAI', 'Role-Playing: The Author', 'prefix contents', 'reference target'))
        self.assertEqual(request['kwargs'], {'chunk_size': 2})


if __name__ == '__main__':
    unittest.main()
