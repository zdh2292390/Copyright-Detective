"""Current OpenAI aliases preserve valid controls and one-token request budgets."""

from copy import deepcopy
import unittest
from unittest.mock import Mock

from src.openai_utils import (
    apply_openai_request_compat,
    apply_openai_short_answer_compat,
    openai_model_rejects_sampling_params,
    unsupported_openai_sampling_param,
)


def request(model, **overrides):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Select A or B."}],
        "temperature": 0.0, "top_p": 0.9, "max_tokens": 1,
        "logprobs": True, "top_logprobs": 4, "seed": 17,
        **overrides,
    }


class CurrentOpenAIModelTests(unittest.TestCase):
    def test_default_none_models_preserve_sampling_and_real_probability_requests(self):
        for model in (
            "gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano",
            "gpt-5.1-2025-11-13", "gpt-5.2-2025-12-11", "gpt-5.4-2026-03-05",
            "gpt-5.4-mini-2026-03-17", "gpt-5.4-nano-2026-03-17",
        ):
            with self.subTest(model=model):
                original = request(model)
                before = deepcopy(original)
                adjusted = apply_openai_request_compat(original)
                self.assertEqual(original, before)
                self.assertEqual(adjusted["temperature"], 0.0)
                self.assertEqual(adjusted["top_p"], 0.9)
                self.assertTrue(adjusted["logprobs"])
                self.assertEqual(adjusted["top_logprobs"], 4)
                self.assertNotIn("reasoning_effort", adjusted)
                self.assertEqual(adjusted["max_completion_tokens"], 1)
                self.assertNotIn("max_tokens", adjusted)
                self.assertFalse(openai_model_rejects_sampling_params(model))

    def test_reasoning_defaults_remove_incompatible_controls_without_changing_effort_or_budget(self):
        for model in (
            "gpt-5.5", "gpt-5.5-2026-04-23", "gpt-6-astra", "gpt-6.1-sol",
            "gpt-6-sol", "gpt-6-luna", "gpt-5", "gpt-5-mini", "gpt-5-nano", "o3", "o4-mini",
        ):
            with self.subTest(model=model):
                original = request(model, logprobs=False, top_logprobs=None)
                adjusted = apply_openai_request_compat(original)
                for key in ("temperature", "top_p", "logprobs", "top_logprobs"):
                    self.assertNotIn(key, adjusted)
                    self.assertIn(key, original)
                self.assertNotIn("reasoning_effort", adjusted)
                self.assertEqual(adjusted["max_completion_tokens"], 1)
                self.assertEqual(adjusted["seed"], 17)
                self.assertTrue(openai_model_rejects_sampling_params(model))

    def test_explicit_reasoning_uses_model_effort_and_removes_sampling_and_probabilities(self):
        for model in ("gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.4-mini", "gpt-5.5", "gpt-6-sol", "gpt-6-luna", "gpt-6-astra", "gpt-6.1-sol"):
            with self.subTest(model=model):
                adjusted = apply_openai_request_compat(request(model, reasoning_effort="high", logprobs=False, top_logprobs=None))
                self.assertEqual(adjusted["reasoning_effort"], "high")
                self.assertEqual(adjusted["max_completion_tokens"], 1)
                self.assertFalse({"temperature", "top_p", "logprobs", "top_logprobs"} & adjusted.keys())

    def test_explicit_none_preserves_sampling_for_supported_current_models(self):
        for model in ("gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5", "gpt-6-sol", "gpt-6-luna"):
            with self.subTest(model=model):
                adjusted = apply_openai_request_compat(request(model, reasoning_effort="none"))
                self.assertEqual(adjusted["reasoning_effort"], "none")
                self.assertEqual(adjusted["temperature"], 0.0)
                self.assertTrue(adjusted["logprobs"])
                self.assertFalse(openai_model_rejects_sampling_params(model, "none"))

    def test_unsupported_none_fails_before_an_outbound_call(self):
        for model in ("gpt-6-astra", "gpt-6.1-sol", "gpt-5", "gpt-5-mini", "o3", "gpt-5.4-pro", "gpt-5.1-codex"):
            with self.subTest(model=model):
                outbound = Mock()
                with self.assertRaisesRegex(ValueError, "does not support"):
                    outbound(**apply_openai_request_compat(request(model, reasoning_effort="none")))
                outbound.assert_not_called()

    def test_non_reasoning_and_other_providers_are_not_reclassified(self):
        for model in (
            "gpt-4o", "gpt-4o-mini", "gpt-5-chat-latest", "gpt-5.1-chat-latest", "gpt-5.3-chat-latest",
            "openai/gpt-oss-20b", "google/gemma-4-26b-a4b-it:free", "moonshotai/kimi-k2.6", "qwen/qwen3-235b-a22b-thinking-2507", "unknown-model",
        ):
            with self.subTest(model=model):
                original = request(model)
                self.assertEqual(apply_openai_request_compat(original), original)
                self.assertEqual(apply_openai_short_answer_compat(original), original)
                self.assertFalse(openai_model_rejects_sampling_params(model))

    def test_openrouter_openai_prefix_obeys_actual_model_contract_only(self):
        original = request("openai/gpt-5.4")
        adjusted = apply_openai_request_compat(original)
        self.assertEqual(adjusted["model"], original["model"])
        self.assertTrue(adjusted["logprobs"])
        self.assertEqual(adjusted["max_completion_tokens"], 1)
        adjusted = apply_openai_request_compat(request("openai/gpt-6-astra", logprobs=False, top_logprobs=None))
        self.assertNotIn("temperature", adjusted)
        self.assertNotIn("logprobs", adjusted)
        self.assertEqual(apply_openai_short_answer_compat(request("openai/gpt-6-luna"))["reasoning_effort"], "none")

    def test_explicit_completion_budget_wins_and_no_duplicate_limits_are_sent(self):
        original = request("gpt-6-sol", max_completion_tokens=7, logprobs=False, top_logprobs=None)
        adjusted = apply_openai_request_compat(original)
        self.assertEqual(adjusted["max_completion_tokens"], 7)
        self.assertNotIn("max_tokens", adjusted)
        self.assertEqual(original["max_tokens"], 1)
        adjusted = apply_openai_request_compat({"model": "gpt-5.4", "max_completion_tokens": 1})
        self.assertEqual(adjusted, {"model": "gpt-5.4", "max_completion_tokens": 1})
        self.assertNotIn("max_completion_tokens", apply_openai_request_compat({"model": "gpt-5.5"}))

    def test_short_answer_none_is_explicit_and_preserves_one_token_and_probabilities(self):
        for model in ("gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5", "gpt-5.5-2026-04-23", "gpt-6-sol", "gpt-6-luna"):
            with self.subTest(model=model):
                original = request(model)
                adjusted = apply_openai_short_answer_compat(original)
                self.assertEqual(adjusted["reasoning_effort"], "none")
                self.assertEqual(adjusted["max_completion_tokens"], 1)
                self.assertEqual(adjusted["temperature"], 0.0)
                self.assertTrue(adjusted["logprobs"])
                self.assertEqual(adjusted["top_logprobs"], 4)
                self.assertNotIn("reasoning_effort", original)

    def test_short_answer_for_mandatory_reasoning_models_is_clear_error_without_calls(self):
        for model in ("gpt-6-astra", "gpt-6.1-sol", "gpt-5", "gpt-5-mini", "gpt-5-nano", "o1", "o3", "o4-mini", "gpt-5.4-pro"):
            with self.subTest(model=model):
                outbound = Mock()
                with self.assertRaisesRegex(ValueError, "one-token.*gpt-4o-mini.*gpt-6-luna"):
                    outbound(**apply_openai_short_answer_compat(request(model)))
                outbound.assert_not_called()

    def test_short_answer_does_not_override_an_explicit_reasoning_choice(self):
        for effort in ("low", "medium", "high", "xhigh"):
            with self.subTest(effort=effort), self.assertRaisesRegex(ValueError, "one-token budget"):
                apply_openai_short_answer_compat(request("gpt-5.5", reasoning_effort=effort))
        self.assertEqual(apply_openai_request_compat(request("gpt-5.5", logprobs=False, top_logprobs=None))["max_completion_tokens"], 1)
        self.assertNotIn("reasoning_effort", apply_openai_request_compat(request("gpt-5.5", logprobs=False, top_logprobs=None)))

    def test_requested_probabilities_fail_before_inference_in_unsupported_reasoning_mode(self):
        for model in ("gpt-5.5", "gpt-6-sol", "gpt-6-luna", "gpt-6-astra", "gpt-6.1-sol", "gpt-5", "o3"):
            for flags in ({"logprobs": True, "top_logprobs": None}, {"logprobs": False, "top_logprobs": 4}):
                with self.subTest(model=model, flags=flags):
                    outbound = Mock()
                    with self.assertRaisesRegex(ValueError, "[Ll]ogprobs.*(none|gpt-4o-mini)"):
                        outbound(**apply_openai_request_compat(request(model, **flags)))
                    outbound.assert_not_called()
        outbound = Mock()
        with self.assertRaisesRegex(ValueError, "require reasoning_effort='none'"):
            outbound(**apply_openai_request_compat(request("gpt-5.4", reasoning_effort="high")))
        outbound.assert_not_called()

    def test_short_answer_does_not_need_stop_sequences_and_does_not_expand_budget(self):
        for model in ("gpt-5.4", "gpt-5.5", "gpt-6-sol", "gpt-6-luna"):
            with self.subTest(model=model):
                original = request(model, stop=["\n"])
                adjusted = apply_openai_short_answer_compat(original)
                self.assertNotIn("stop", adjusted)
                self.assertEqual(original["stop"], ["\n"])
                self.assertEqual(adjusted["max_completion_tokens"], 1)
                self.assertTrue(adjusted["logprobs"])
        original = request("gpt-4o-mini", stop=["\n"])
        self.assertEqual(apply_openai_short_answer_compat(original), original)
        original = request("gpt-5.4", stop=["END"])
        self.assertEqual(apply_openai_request_compat(original)["stop"], ["END"])

    def test_unsupported_sampling_error_detector_keeps_existing_behavior(self):
        self.assertEqual(unsupported_openai_sampling_param(ValueError("Unsupported parameter: temperature")), "temperature")
        self.assertEqual(unsupported_openai_sampling_param(ValueError("top_p is not supported")), "top_p")
        self.assertIsNone(unsupported_openai_sampling_param(ValueError("429 too many requests")))
        self.assertIsNone(unsupported_openai_sampling_param(ValueError("missing logprobs")))


if __name__ == "__main__":
    unittest.main()
