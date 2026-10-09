"""Real Streamlit selection recovers retired models while preserving valid caches."""
import unittest
from streamlit.testing.v1 import AppTest
from src.model_catalog import DEFAULT_MODELS, MODEL_CONFIG

APP = '''
import streamlit as st
from src.model_catalog import MODEL_CONFIG
from src.sidebar_utils import render_model_selectbox
provider = st.session_state.get('test_provider', 'Kimi')
st.session_state['selected_model'] = render_model_selectbox(provider, MODEL_CONFIG[provider])
st.session_state.setdefault('saved_analysis', {'model': 'moonshot-v1-32k', 'completed': 104})
'''

class SidebarModelTests(unittest.TestCase):
    def app(self, provider, cached=...):
        app = AppTest.from_string(APP, default_timeout=60)
        app.session_state['test_provider'] = provider
        if cached is not ...:
            app.session_state[MODEL_CONFIG[provider]['key']] = cached
        app.run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_all_current_defaults_are_valid_and_kimi_default_is_general_purpose(self):
        for provider, config in MODEL_CONFIG.items():
            with self.subTest(provider=provider):
                app = self.app(provider)
                self.assertEqual(app.session_state['selected_model'], DEFAULT_MODELS[provider])
                self.assertEqual(config['models'][config['default_index']], DEFAULT_MODELS[provider])
        self.assertEqual(set(MODEL_CONFIG['Kimi']['models']), {
            'kimi-k3', 'kimi-k2.6', 'kimi-k2.7-code', 'kimi-k2.7-code-highspeed',
        })
        self.assertEqual(DEFAULT_MODELS['Kimi'], 'kimi-k2.6')

    def test_retired_kimi_cached_models_recover_with_explicit_notice(self):
        models = ('kimi-k2.5', 'moonshot-v1-auto', 'moonshot-v1-8k', 'moonshot-v1-32k',
            'moonshot-v1-128k', 'moonshot-v1-8k-vision-preview',
            'moonshot-v1-32k-vision-preview', 'moonshot-v1-128k-vision-preview')
        for model in models:
            with self.subTest(model=model):
                app = self.app('Kimi', model)
                self.assertEqual(app.session_state['selected_model'], 'kimi-k2.6')
                self.assertTrue(any(model in item.value and 'new analyses' in item.value for item in app.info))
                self.assertEqual(app.session_state['saved_analysis'], {'model': 'moonshot-v1-32k', 'completed': 104})
                app.run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(app.selectbox(key=MODEL_CONFIG['Kimi']['key']).value, 'kimi-k2.6')

    def test_valid_cached_choice_and_user_changes_survive_reruns(self):
        app = self.app('Kimi', 'kimi-k2.7-code-highspeed')
        self.assertEqual(app.session_state['selected_model'], 'kimi-k2.7-code-highspeed')
        self.assertEqual(len(app.info), 0)
        app.selectbox(key=MODEL_CONFIG['Kimi']['key']).set_value('kimi-k3').run()
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state['selected_model'], 'kimi-k3')
        self.assertEqual(len(app.info), 0)

    def test_none_and_invalid_cached_values_do_not_leave_empty_or_broken_selection(self):
        for cached in (None, 99, [], {}, ''):
            with self.subTest(cached=repr(cached)):
                app = self.app('Kimi', cached)
                self.assertEqual(app.session_state['selected_model'], 'kimi-k2.6')
                self.assertEqual(app.selectbox(key=MODEL_CONFIG['Kimi']['key']).value, 'kimi-k2.6')

    def test_provider_replacements_preserve_free_tier_and_model_type(self):
        cases = (
            ('OpenRouter', 'openai/gpt-oss-20b:free', 'google/gemma-4-26b-a4b-it:free'),
            ('Google Gemini', 'gemini-3-pro-preview', 'gemini-3.1-pro-preview'),
            ('Google Gemini', 'gemini-3.1-flash-lite-preview', 'gemini-3.5-flash-lite'),
            ('Google Gemini', 'gemini-1.5-flash', 'gemini-3.8-flash'),
        )
        for provider, old, expected in cases:
            with self.subTest(provider=provider, old=old):
                app = self.app(provider, old)
                self.assertEqual(app.session_state['selected_model'], expected)
                self.assertTrue(any(old in item.value and expected in item.value for item in app.info))

if __name__ == '__main__':
    unittest.main()
