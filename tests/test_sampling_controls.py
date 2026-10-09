"""Shared sampling caches recover invalid values without changing valid controls."""
import unittest
from unittest.mock import patch
import streamlit.elements.lib.policies as policies
from streamlit.testing.v1 import AppTest
from src.pages.sampling_controls import _sampling_value

APP = '''
import streamlit as st
from src.pages.sampling_controls import render_temperature_top_p
values = render_temperature_top_p(temp_session_key='test_temp', top_p_session_key='test_top_p')
st.text(str(values))
'''

class SamplingControlsTests(unittest.TestCase):
    def run_without_default_warning(self, app, changed_slider=None):
        # This policy warns only once per process. Reset it on every render so
        # another test's warning cannot hide a regression in these controls.
        with patch.object(policies, '_shown_default_value_warning', False), \
             self.assertNoLogs('streamlit.elements.lib.policies', level='WARNING'):
            (changed_slider or app).run()
            self.assertFalse(policies._shown_default_value_warning)
        self.assertEqual(len(app.exception), 0)
        self.assertFalse(any('also had its value set via the Session State API' in item.value for item in app.warning))

    def test_first_render_and_user_changes_use_state_without_default_warning(self):
        app = AppTest.from_string(APP)
        self.run_without_default_warning(app)
        self.assertEqual(app.slider(key='test_temp_slider').value, .7)
        self.assertEqual(app.slider(key='test_top_p_slider').value, .9)
        self.run_without_default_warning(app, app.slider(key='test_temp_slider').set_value(.63))
        self.run_without_default_warning(app, app.slider(key='test_top_p_slider').set_value(.84))
        self.assertEqual(app.session_state['test_temp'], .63)
        self.assertEqual(app.session_state['test_top_p'], .84)

    def test_recovered_widget_precedence_fragment_and_next_user_change_have_no_default_warning(self):
        fragment_app = APP.replace('render_temperature_top_p', 'render_fragmented_temperature_top_p')
        app = AppTest.from_string(fragment_app)
        app.session_state['test_temp'] = .42
        app.session_state['test_top_p'] = .81
        app.session_state['test_temp_slider'] = .66
        app.session_state['test_top_p_slider'] = .77
        self.run_without_default_warning(app)
        self.assertEqual(app.session_state['test_temp'], .66)
        self.assertEqual(app.session_state['test_top_p'], .77)
        self.run_without_default_warning(app, app.slider(key='test_temp_slider').set_value(.58))
        self.assertEqual(app.session_state['test_temp'], .58)
        self.assertEqual(app.slider(key='test_top_p_slider').value, .77)

    def test_invalid_cache_values_recover_to_configured_default(self):
        for value in (None, object(), True, float('nan'), float('inf'), -1, 5, 'bad'):
            with self.subTest(value=type(value).__name__):
                self.assertEqual(_sampling_value(value, .7, (0, 1.2)), .7)
        self.assertEqual(_sampling_value(.42, .7, (0, 1.2)), .42)
        self.assertEqual(_sampling_value('0.42', .7, (0, 1.2)), .42)

    def test_corrupt_session_and_widget_values_recover_before_rendering(self):
        app = AppTest.from_string(APP)
        app.session_state['test_temp'] = float('nan')
        app.session_state['test_top_p'] = 'bad'
        app.session_state['test_temp_slider'] = float('inf')
        app.session_state['test_top_p_slider'] = True
        self.run_without_default_warning(app)
        self.assertEqual(app.slider(key='test_temp_slider').value, .7)
        self.assertEqual(app.slider(key='test_top_p_slider').value, .9)

    def test_valid_values_and_slider_changes_are_preserved(self):
        app = AppTest.from_string(APP)
        app.session_state['test_temp'] = .42
        app.session_state['test_top_p'] = .81
        self.run_without_default_warning(app)
        self.assertEqual(app.slider(key='test_temp_slider').value, .42)
        self.assertEqual(app.slider(key='test_top_p_slider').value, .81)
        self.run_without_default_warning(app, app.slider(key='test_temp_slider').set_value(.63))
        self.assertEqual(app.session_state['test_temp'], .63)
        self.assertEqual(app.session_state['test_top_p'], .81)

if __name__ == '__main__':
    unittest.main()
