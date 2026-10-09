"""Recovered upload text renders and regenerates without an UploadedFile widget."""
import unittest
from streamlit.testing.v1 import AppTest

APP = '''
import streamlit as st
from unittest.mock import patch
import src.pages.single_choice_detection as page
import src.direct_recall.single_choice as sc

mcqs = [{'question': 'Recovered question?', 'options': [{'label': 'A', 'text': 'correct'},
    {'label': 'B', 'text': 'wrong'}], 'correct_option': 'A'}]

def generate(text, *args, **kwargs):
    st.session_state.setdefault('generation_calls', []).append({'text': text, 'args': args, 'kwargs': {key: value for key, value in kwargs.items() if key != 'progress_callback'}})
    return mcqs

with patch.object(page, 'generate_single_choice_questions_from_fragments', side_effect=generate), \
     patch.object(page, 'generate_single_choice_question_pdf_report', return_value=b'pdf'), \
     patch.object(page, 'render_pdf_preview_with_blob', return_value=None), \
     patch.object(sc, 'get_llm_completion', return_value='A'):
    page.render_single_choice_detection_page('key', 'mock-model', 'Google Gemini')
'''

class SingleChoiceSourceResumeTests(unittest.TestCase):
    def make_app(self):
        app = AppTest.from_string(APP, default_timeout=60)
        app.session_state['sc_source_mode'] = 'Upload Document'
        app.session_state['sc_upload_source_text'] = 'Recovered source text containing enough material.'
        app.session_state['sc_upload_source_name'] = 'saved.txt'
        app.session_state['sc_upload_source_identity'] = 'saved-identity'
        app.session_state['sc_source_identity'] = 'saved-identity'
        return app

    def test_empty_upload_widget_preserves_recovered_questions_and_results(self):
        app = self.make_app()
        app.session_state['sc_generated_mcqs'] = [{'question': 'Recovered question?', 'options': [{'label': 'A', 'text': 'correct'}, {'label': 'B', 'text': 'wrong'}], 'correct_option': 'A'}]
        rows = [[{'question': 'Recovered question?', 'options': [{'label': 'A', 'text': 'correct'}, {'label': 'B', 'text': 'wrong'}], 'correct_option': 'A', 'llm_choice': 'A', 'is_correct': True, 'error': None, 'question_index': 0, 'raw_response': 'A', 'option_probabilities': None, 'logit_mode': 'text'}]]
        app.session_state['sc_evaluation_results'] = rows
        app.session_state['sc_evaluation_metadata'] = {'model': 'original-model', 'provider': 'Google Gemini', 'source_mode': 'Upload Document'}
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state['sc_evaluation_results'], rows)
        self.assertEqual(app.session_state['sc_source_identity'], 'saved-identity')
        self.assertTrue(any('Recovered source: saved.txt' in item.value for item in app.caption))
        self.assertNotIn('generation_calls', app.session_state)

    def test_regeneration_uses_saved_text_and_existing_generation_parameters(self):
        app = self.make_app().run()
        self.assertEqual(len(app.exception), 0)
        app.button(key='sc_generate_mcq_button').click().run()
        self.assertEqual(len(app.exception), 0)
        call = app.session_state['generation_calls'][0]
        self.assertEqual(call['text'], 'Recovered source text containing enough material.')
        self.assertEqual(call['args'], ('key', 'mock-model', 'Google Gemini'))
        self.assertEqual(call['kwargs'], {'num_questions': 5, 'num_distractors': 3, 'temperature': .7, 'top_p': .9})
        self.assertEqual(len(app.session_state['sc_generated_mcqs']), 1)

    def test_changing_source_clears_recovered_upload_snapshot(self):
        app = self.make_app().run()
        app.radio(key='sc_source_mode').set_value('Input Text').run()
        self.assertEqual(len(app.exception), 0)
        self.assertNotIn('sc_upload_source_text', app.session_state)
        self.assertEqual(app.session_state['sc_generated_mcqs'], [])

    def test_uploaded_source_is_saved_before_api_scope_using_original_3500_word_cap(self):
        app = AppTest.from_string(APP, default_timeout=60)
        text = ' '.join('word' + str(index) for index in range(3501))
        app.session_state['sc_source_mode'] = 'Upload Document'
        app.session_state['sc_cached_upload'] = {'data': text.encode(), 'name': 'input.txt', 'mime_type': 'text/plain'}
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.session_state['sc_upload_source_text'].split()), 3500)
        self.assertEqual(app.session_state['sc_upload_source_name'], 'input.txt')
        self.assertEqual(app.session_state['sc_upload_source_identity'], app.session_state['sc_source_identity'])
        self.assertNotIn('generation_calls', app.session_state)

if __name__ == '__main__':
    unittest.main()
