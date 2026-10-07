"""Shared sampling caches recover invalid values without changing valid controls."""
import unittest
from streamlit.testing.v1 import AppTest
from src.pages.sampling_controls import _sampling_value

APP = '''
import streamlit as st
from src.pages.sampling_controls import render_temperature_top_p
values = render_temperature_top_p(temp_session_key='test_temp', top_p_session_key='test_top_p')
st.text(str(values))
'''

class SamplingControlsTests(unittest.TestCase):
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
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.slider(key='test_temp_slider').value, .7)
        self.assertEqual(app.slider(key='test_top_p_slider').value, .9)

    def test_valid_values_and_slider_changes_are_preserved(self):
        app = AppTest.from_string(APP)
        app.session_state['test_temp'] = .42
        app.session_state['test_top_p'] = .81
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.slider(key='test_temp_slider').value, .42)
        self.assertEqual(app.slider(key='test_top_p_slider').value, .81)
        app.slider(key='test_temp_slider').set_value(.63).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state['test_temp'], .63)
        self.assertEqual(app.session_state['test_top_p'], .81)

if __name__ == '__main__':
    unittest.main()
