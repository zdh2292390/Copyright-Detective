"""Kimi fixed sampling and locked challenge identity without inference calls."""
import unittest
from unittest.mock import Mock, patch
from streamlit.testing.v1 import AppTest
from src import game_continuation as scaling
from src.kimi_utils import (
    kimi_requires_fixed_sampling,
    kimi_request_extra_body,
    normalize_kimi_sampling_params,
)

GAME_APP = '''
import streamlit as st
from unittest.mock import patch
from src import game_continuation as scaling
from src.pages import copyright_game2 as page
page._render_provider_settings(scaling.KIMI_PROVIDER, disabled=False)
old = scaling.ScalingBatch(
    provider='Kimi', model='moonshot-v1-32k', prompt_method=scaling.DIRECT_PROBE_METHODS[0],
    prompt_mode='Direct Probing', prompt='historic prompt', temperature=.7, top_p=.9,
    requested_runs=1, attempts=(scaling.ContinuationAttempt(1, 'historic answer', {'rouge_l': .25}),),
)
with patch.object(page, 'render_direct_recall_diff', return_value=None):
    page._render_batch(old, 'historic run', show_distribution=False)
st.session_state['historic_batch'] = old
'''

class KimiSamplingTests(unittest.TestCase):
    def test_current_k2_6_uses_explicit_non_thinking_fixed_sampling(self):
        self.assertEqual(normalize_kimi_sampling_params('kimi-k2.6', .2, .3), (.6, .95))
        self.assertEqual(kimi_request_extra_body(' KIMI-K2.6 '), {'thinking': {'type': 'disabled'}})
        self.assertEqual(normalize_kimi_sampling_params('kimi-k2.6', .2, .3, thinking_enabled=True), (1., .95))

    def test_k3_and_code_keep_thinking_and_fixed_sampling_without_k2_body(self):
        for model in ('kimi-k3', 'kimi-k2.7-code', 'kimi-k2.7-code-highspeed'):
            with self.subTest(model=model):
                self.assertTrue(kimi_requires_fixed_sampling(model))
                self.assertEqual(normalize_kimi_sampling_params(model, .2, .3), (1., .95))
                self.assertEqual(kimi_request_extra_body(model), {})

    def test_other_providers_and_historical_moonshot_sampling_are_unchanged(self):
        for model in ('gpt-4o-mini', 'moonshot-v1-32k', '', None):
            with self.subTest(model=model):
                self.assertFalse(kimi_requires_fixed_sampling(model))
                self.assertEqual(normalize_kimi_sampling_params(model, .7, .9), (.7, .9))
                self.assertEqual(kimi_request_extra_body(model), {})

    def test_request_body_is_not_shared_mutable_state(self):
        first = kimi_request_extra_body('kimi-k2.6')
        first['thinking']['type'] = 'enabled'
        self.assertEqual(kimi_request_extra_body('kimi-k2.6'), {'thinking': {'type': 'disabled'}})


class KimiChallengeTests(unittest.TestCase):
    def run_batch(self, provider, model, completion):
        return scaling.run_provider_scaling('key', provider=provider, model=model, runs=2,
            temperature=.7, top_p=.9, prompt_method=scaling.DIRECT_PROBE_METHODS[0],
            prompt_mode='Direct Probing', completion_fn=completion,
            metrics_fn=lambda reference, answer: {'rouge_l': .25})

    def test_new_kimi_batches_capture_actual_model_sampling_and_existing_cap(self):
        completion = Mock(return_value='generated passage')
        result = self.run_batch('Kimi', scaling.KIMI_MODEL, completion)
        self.assertEqual(scaling.KIMI_MODEL, 'kimi-k2.6')
        self.assertEqual((result.model, result.temperature, result.top_p), ('kimi-k2.6', .6, .95))
        self.assertEqual((result.completed_runs, result.avg_rouge_l), (2, .25))
        for call in completion.call_args_list:
            self.assertEqual(call.args[0], scaling.build_challenge_prompt(scaling.DIRECT_PROBE_METHODS[0]))
            self.assertEqual(call.args[2], 'kimi-k2.6')
            self.assertEqual(call.kwargs, {'provider': 'Kimi', 'temperature': .6, 'top_p': .95, 'max_output_tokens': 700})

    def test_openai_challenge_parameters_and_scores_do_not_change(self):
        completion = Mock(return_value='generated passage')
        result = self.run_batch('OpenAI', scaling.OPENAI_MODEL, completion)
        self.assertEqual((result.temperature, result.top_p, result.avg_rouge_l), (.7, .9, .25))
        self.assertEqual(completion.call_args.kwargs['temperature'], .7)
        self.assertEqual(completion.call_args.kwargs['top_p'], .9)

    def test_removed_locked_model_cannot_generate_or_mix_with_new_batch(self):
        completion = Mock()
        with self.assertRaisesRegex(scaling.ContinuationValidationError, 'kimi-k2.6'):
            self.run_batch('Kimi', 'moonshot-v1-32k', completion)
        completion.assert_not_called()

    def test_current_game_heading_and_sampling_explanation_preserve_historical_identity(self):
        app = AppTest.from_string(GAME_APP, default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        headings = ' '.join(item.value for item in app.markdown)
        self.assertIn('Model: <code>kimi-k2.6</code>', headings)
        self.assertIn('non-thinking mode with temperature 0.60', headings)
        self.assertTrue(any('moonshot-v1-32k' in item.value for item in app.caption))
        old = app.session_state['historic_batch']
        self.assertEqual((old.model, old.temperature, old.top_p), ('moonshot-v1-32k', .7, .9))
        self.assertTrue(any(item.label == 'Temperature / top-p' and item.value == '0.70 / 0.90' for item in app.metric))

if __name__ == '__main__':
    unittest.main()
