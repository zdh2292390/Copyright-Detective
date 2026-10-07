"""Knowmem UI preserves successful questions through provider failures."""

import json
import unittest
from streamlit.testing.v1 import AppTest


APP = '''
import streamlit as st
from unittest.mock import patch
import src.ui as ui
import src.direct_recall.comparison as comparison
st.session_state.setdefault('qa_eval_examples', [
    {'question': 'capital France?', 'answer': 'Paris'},
    {'question': 'capital Spain?', 'answer': 'Madrid'},
    {'question': 'capital Italy?', 'answer': 'Rome'},
])
st.session_state.setdefault('qa_icl_examples', [])
st.session_state.setdefault('qa_prompt_mode', 'Zero-Shot')
st.session_state.setdefault('qa_knowmem_max_new_tokens', 64)

def completion(*args, **kwargs):
    calls = st.session_state.setdefault('mock_knowmem_calls', [])
    calls.append({'args': args, 'kwargs': kwargs})
    index = len(calls)
    mode = st.session_state.get('mock_failure_mode', 'timeout')
    if mode == 'all_failed':
        if index == 1:
            raise TimeoutError()
        return None if index == 2 else 'Error: 503 temporarily unavailable'
    if index == 2:
        if mode == 'timeout':
            raise TimeoutError()
        if mode == 'blank':
            return '   '
        return None
    return 'Paris\\nQuestion: ignored follow-up' if index == 1 else 'Rome\\n\\nignored follow-up'

with patch.object(comparison, 'get_llm_completion', side_effect=completion):
    if st.button('Evaluate', key='evaluate_knowmem'):
        ui.run_knowmem_evaluation('test-key', 'test-model', 'OpenAI')
'''


class KnowmemPageTests(unittest.TestCase):
    def execute(self, mode):
        app = AppTest.from_string(APP, default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        app.session_state['mock_failure_mode'] = mode
        app.button(key='evaluate_knowmem').click().run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_failed_middle_question_keeps_both_successes_and_original_prompt_and_scores(self):
        for mode in ('timeout', 'none', 'blank'):
            with self.subTest(mode=mode):
                app = self.execute(mode)
                calls = app.session_state['mock_knowmem_calls']
                self.assertEqual(len(calls), 3)
                self.assertEqual(calls[0]['args'], ('Question: capital France?\nAnswer: ', 'test-key', 'test-model', 'OpenAI'))
                self.assertEqual(calls[0]['kwargs'], {
                    'temperature': 0.7, 'top_p': 0.9, 'max_output_tokens': 64,
                    'stop_sequences': ['\n\n', '\nQuestion', 'Question:'],
                })
                self.assertTrue(any('Partial evaluation: 2/3' in warning.value for warning in app.warning))
                self.assertFalse(any('Knowmem evaluation completed' in success.value for success in app.success))
                self.assertEqual(len(app.json), 2)
                for result in app.json:
                    self.assertEqual(json.loads(result.value), {'f1': 1.0, 'precision': 1.0, 'recall': 1.0})
                self.assertEqual(len(app.expander), 2)
                self.assertIn('capital France?', app.expander[0].label)
                self.assertIn('capital Italy?', app.expander[1].label)
                self.assertFalse(any('ignored follow-up' in str(element.value) for element in app.markdown))

    def test_all_failed_questions_show_no_score_or_completion_claim(self):
        app = self.execute('all_failed')
        self.assertEqual(len(app.session_state['mock_knowmem_calls']), 3)
        self.assertTrue(any('No answers were successfully evaluated. No score is available.' in error.value for error in app.error))
        self.assertEqual(len(app.json), 0)
        self.assertEqual(len(app.expander), 0)
        self.assertFalse(any('Knowmem evaluation completed' in success.value for success in app.success))
        self.assertFalse(any('Summary Metrics' in element.value for element in app.markdown))


if __name__ == '__main__':
    unittest.main()
