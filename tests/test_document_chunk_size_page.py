"""Actual document page regressions for the 50-word chunk boundary."""

import time
import unittest
from streamlit.testing.v1 import AppTest


APP = '''
import tempfile
import streamlit as st
from unittest.mock import patch
import src.pages.document_memorization_detection as page
from src.direct_recall.pdf_utils import extract_text_from_document, split_text_into_chunks
from src.document_jobs import DocumentAnalysisJobs
from src.document_checkpoints import DocumentCheckpointStore

if 'mock_document_control' not in st.session_state:
    st.session_state['mock_document_control'] = {'calls': []}
    st.session_state['mock_document_directory'] = tempfile.TemporaryDirectory()
control = st.session_state['mock_document_control']

def compare(settings, key, upper, lower):
    control['calls'].append({'settings': dict(settings), 'upper': upper, 'lower': lower})
    return lower, {'rouge_l': 0.5}

if 'mock_document_service' not in st.session_state:
    st.session_state['mock_document_service'] = DocumentAnalysisJobs(
        DocumentCheckpointStore(st.session_state['mock_document_directory'].name),
        analyze_chunk=compare,
    )

class Document:
    name = 'boundary.txt'
    type = 'text/plain'
    def getvalue(self):
        return ' '.join('word' + str(index) for index in range(501)).encode('utf-8')


def render_results(results, document, model, **kwargs):
    state = kwargs['analysis_progress']
    st.session_state['mock_boundary_report'] = {
        'count': len(results), 'status': state['status'],
        'total': state['total_chunks'], 'model': model,
        'chunk_size': kwargs['chunk_size'], 'settings': dict(state['settings']),
    }
    st.text('Saved result count: ' + str(len(results)))

with patch.object(page, 'DOCUMENT_JOBS', st.session_state['mock_document_service']), \
     patch.object(page, '_list_example_documents', return_value=[('Boundary document', None)]), \
     patch.object(page, '_resolve_active_document', return_value=Document()), \
     patch.object(page, 'extract_text_from_document', extract_text_from_document), \
     patch.object(page, 'split_text_into_chunks', split_text_into_chunks), \
     patch.object(page, 'render_prompt_preview', return_value=None), \
     patch.object(page, 'render_pdf_results_section', side_effect=render_results):
    page.render_pdf_analysis_page('test-key', 'gemini-3.5-flash', 'Google Gemini')
'''


CACHED_UNSET = object()


class DocumentChunkSizePageTests(unittest.TestCase):
    def setUp(self):
        self.apps = []

    def tearDown(self):
        for app in self.apps:
            app.session_state['mock_document_service'].close()
            app.session_state['mock_document_directory'].cleanup()

    def make_app(self, cached_size=CACHED_UNSET):
        app = AppTest.from_string(APP, default_timeout=60)
        if cached_size is not CACHED_UNSET:
            app.session_state['pdf_chunk_size'] = cached_size
            app.session_state['pdf_chunk_size_input'] = cached_size
        self.apps.append(app)
        app.run()
        self.assertEqual(len(app.exception), 0)
        return app

    def assert_preview(self, app, count, size, overlap):
        previews = [info.value for info in app.info if 'will be processed' in info.value]
        self.assertEqual(len(previews), 1)
        self.assertIn('**' + str(count) + '**', previews[0])
        self.assertIn('chunk size ' + str(size) + ' words', previews[0])
        self.assertIn(str(overlap) + '-word overlap', previews[0])
        self.assertFalse(any('unexpected problem' in error.value.lower() for error in app.error))

    def run_to_completion(self, app, count, size, overlap):
        app.button(key='analyze_pdf_button').click().run()
        self.assertEqual(len(app.exception), 0)
        service = app.session_state['mock_document_service']
        token = app.session_state['pdf_analysis_job_token']
        deadline = time.monotonic() + 30
        while service.is_running(token) and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(service.is_running(token), 'Boundary document analysis did not finish')
        app.run()
        self.assertEqual(len(app.exception), 0)
        state = app.session_state['pdf_analysis_state']
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(state['total_chunks'], count)
        self.assertEqual(len(state['results']), count)
        self.assertEqual(state['settings']['chunk_size'], size)
        self.assertEqual(state['settings']['overlap'], overlap)
        calls = app.session_state['mock_document_control']['calls']
        self.assertEqual(len(calls), count)
        self.assertEqual(len({call['upper'] for call in calls}), count)
        for index, call in enumerate(calls):
            self.assertEqual(call['settings']['chunk_size'], size)
            self.assertEqual(call['settings']['overlap'], overlap)
            self.assertEqual(call['upper'].split()[0], 'word' + str(index * (size - overlap)))
            self.assertEqual(call['lower'].split()[0], 'word' + str((index + 1) * (size - overlap)))
            self.assertGreater(len(call['upper'].split()), 0)
            self.assertGreater(len(call['lower'].split()), 0)
        report = app.session_state['mock_boundary_report']
        self.assertEqual(report['count'], count)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['total'], count)
        self.assertEqual(report['chunk_size'], size)
        self.assertEqual(report['settings']['overlap'], overlap)
        self.assert_preview(app, count, size, overlap)
        return state

    def test_50_word_selection_has_matching_preview_run_and_complete_saved_report(self):
        app = self.make_app()
        app.number_input(key='pdf_chunk_size_input').set_value(50).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.number_input(key='pdf_chunk_size_input').value, 50)
        self.assert_preview(app, 20, 50, 25)
        self.run_to_completion(app, 20, 50, 25)

    def test_legacy_cached_50_word_widget_runs_and_reloads_complete_checkpoint(self):
        app = self.make_app(cached_size=50)
        self.assertEqual(app.number_input(key='pdf_chunk_size_input').value, 50)
        self.assertEqual(app.session_state['pdf_chunk_size'], 50)
        self.assert_preview(app, 20, 50, 25)
        completed = self.run_to_completion(app, 20, 50, 25)
        token = app.session_state['pdf_analysis_job_token']
        app.session_state['mock_document_service'].close()
        del app.session_state['mock_document_service']
        for key in ('pdf_analysis_job_token', 'pdf_analysis_state', 'pdf_analysis_results'):
            del app.session_state[key]
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state['pdf_analysis_job_token'], token)
        self.assertEqual(app.session_state['pdf_analysis_state']['results'], completed['results'])
        self.assertEqual(app.session_state['mock_boundary_report']['count'], 20)
        self.assertEqual(app.session_state['mock_boundary_report']['settings']['overlap'], 25)
        self.assertEqual(len(app.session_state['mock_document_control']['calls']), 20)
        self.assertEqual(app.number_input(key='pdf_chunk_size_input').value, 50)
        self.assert_preview(app, 20, 50, 25)

    def test_default_200_word_chunk_count_and_50_word_overlap_are_unchanged(self):
        app = self.make_app()
        self.assertEqual(app.number_input(key='pdf_chunk_size_input').value, 200)
        self.assert_preview(app, 3, 200, 50)
        self.run_to_completion(app, 3, 200, 50)


    def test_invalid_cached_widget_values_are_repaired_before_render(self):
        cases = [(0, 50), (5000, 2000), ('50', 200), (None, 200),
                 (50.0, 200), (float('nan'), 200), (True, 200)]
        for cached, expected in cases:
            with self.subTest(cached=cached):
                app = self.make_app(cached_size=cached)
                self.assertEqual(app.number_input(key='pdf_chunk_size_input').value, expected)
                self.assertEqual(app.session_state['pdf_chunk_size'], expected)
                self.assertEqual(len(app.error), 0)
                self.assertEqual(app.session_state['mock_document_control']['calls'], [])

    def test_clear_cache_resets_cached_50_word_widget_to_default_200(self):
        app = self.make_app(cached_size=50)
        self.assert_preview(app, 20, 50, 25)
        app.button(key='clear_pdf_cache').click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.number_input(key='pdf_chunk_size_input').value, 200)
        self.assertEqual(app.session_state['pdf_chunk_size'], 200)
        self.assert_preview(app, 3, 200, 50)
        self.assertEqual(app.session_state['mock_document_control']['calls'], [])


if __name__ == '__main__':
    unittest.main()
