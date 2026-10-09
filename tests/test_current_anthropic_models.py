"""Current Claude requests preserve text budgets, response identity and scoring."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import anthropic
import httpx
from google.genai import types

from src.anthropic_utils import (
    anthropic_short_answer_error, apply_anthropic_request_compat,
    create_anthropic_message, extract_anthropic_response_text,
)
from src.direct_recall import comparison
from src.direct_recall import single_choice as sc
from src.model_catalog import DEFAULT_MODELS


QUESTION = {
    "question": "Which passage?",
    "options": [{"label": label, "text": label + " text"} for label in sc.OPTION_LABELS],
    "correct_option": "A",
}


def response(text="A", *, thinking=False, stop_reason="end_turn"):
    blocks = []
    if thinking:
        blocks.append(SimpleNamespace(type="thinking", thinking="Choose D", text="D"))
    if text is not None:
        blocks.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        content=blocks, stop_reason=stop_reason, text=text,
        choices=[SimpleNamespace(
            text=text, logprobs=None,
            message=SimpleNamespace(content=text, refusal=None), finish_reason="stop",
        )],
    )


class FakeClient:
    def __init__(self, reply=None, error=None):
        self.create = Mock(return_value=reply, side_effect=error)
        self.messages = SimpleNamespace(create=self.create)
        self.models = SimpleNamespace(generate_content=self.create)
        self.completions = SimpleNamespace(create=self.create)
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


class CurrentAnthropicTests(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ("start_llm_progress", (None, None, None)),
            ("update_llm_progress", None), ("complete_llm_progress", None),
        ):
            patcher = patch.object(comparison, name, return_value=value)
            self.addCleanup(patcher.stop)
            setattr(self, name, patcher.start())

    def _completion(self, model, reply=None, **kwargs):
        client = FakeClient(reply or response("continuation"))
        with patch.object(comparison.anthropic, "Anthropic", return_value=client):
            result = comparison.get_llm_completion(
                "original prompt", "key", model, "Anthropic", **kwargs,
            )
        return result, client

    def test_current_models_omit_sampling_without_altering_budget_or_prompt(self):
        for model in (
            "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5-5",
            "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1",
        ):
            with self.subTest(model=model):
                result, client = self._completion(
                    model, max_output_tokens=77, stop_sequences=["END"],
                    temperature=.2, top_p=.4,
                )
                self.assertEqual(result, "continuation")
                request = client.create.call_args.kwargs
                self.assertEqual(request["model"], model)
                self.assertEqual(request["max_tokens"], 77)
                self.assertEqual(request["messages"], [{"role": "user", "content": "original prompt"}])
                self.assertEqual(request["stop_sequences"], ["END"])
                self.assertNotIn("temperature", request)
                self.assertNotIn("top_p", request)
                self.assertTrue(client.closed)

    def test_sonnet_and_haiku_preserve_text_only_mode(self):
        expected = {
            "claude-sonnet-5-5": {"type": "between_tools"},
            "claude-haiku-5-5": {"type": "disabled"},
        }
        for model, thinking in expected.items():
            with self.subTest(model=model):
                _, client = self._completion(model, max_output_tokens=1)
                self.assertEqual(client.create.call_args.kwargs["thinking"], thinking)
                self.assertEqual(client.create.call_args.kwargs["max_tokens"], 1)

    def test_default_claude_uses_catalog_model_in_text_mode(self):
        _, client = self._completion(None)
        request = client.create.call_args.kwargs
        self.assertEqual(request["model"], DEFAULT_MODELS["Anthropic"])
        self.assertEqual(request["thinking"], {"type": "between_tools"})
        self.assertEqual(request["max_tokens"], 1000)

    def test_active_older_models_keep_sampling_and_no_added_thinking(self):
        for model in ("claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"):
            with self.subTest(model=model):
                _, client = self._completion(model, temperature=.2, top_p=.4, max_output_tokens=88)
                request = client.create.call_args.kwargs
                self.assertEqual(request["temperature"], .2)
                if model.startswith("claude-haiku-4-5"):
                    self.assertNotIn("top_p", request)
                else:
                    self.assertEqual(request["top_p"], .4)
                self.assertEqual(request["max_tokens"], 88)
                self.assertNotIn("thinking", request)

    def test_unknown_future_names_do_not_inherit_assumed_capabilities(self):
        for model in ("claude-sonnet-5-6", "claude-opus-5-6", "claude-haiku-5-6", "claude-fable-5-2"):
            with self.subTest(model=model):
                request = {"model": model, "max_tokens": 1, "temperature": 0, "top_p": .8}
                self.assertEqual(apply_anthropic_request_compat(request), request)
                self.assertIsNone(anthropic_short_answer_error(model))

    def test_compatibility_does_not_mutate_callers_or_override_explicit_thinking(self):
        request = {
            "model": "claude-sonnet-5-5", "max_tokens": 5, "temperature": .3,
            "thinking": {"type": "adaptive"},
        }
        result = apply_anthropic_request_compat(request)
        self.assertEqual(request["temperature"], .3)
        self.assertEqual(result["thinking"], {"type": "adaptive"})
        self.assertEqual(result["max_tokens"], 5)

    def test_answer_text_blocks_ignore_thinking_position_and_content(self):
        self.assertEqual(extract_anthropic_response_text(response(" A ", thinking=True)), "A")
        self.assertEqual(extract_anthropic_response_text({
            "content": [{"type": "thinking", "text": "D"}, {"type": "text", "text": "A"},
                        {"type": "text", "text": "B"}], "stop_reason": "end_turn",
        }), "AB")

    def test_thinking_only_is_failed_and_never_scored(self):
        client = FakeClient(response(None, thinking=True, stop_reason="max_tokens"))
        with patch.object(comparison.anthropic, "Anthropic", return_value=client):
            result = comparison.compare_texts(
                "prefix", "answer", "key", "claude-opus-5-5", "Anthropic",
                return_logprobs=True,
            )
        self.assertTrue(result[0].startswith("Error:"))
        self.assertIn("finish_reason=max_tokens", result[0])
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])
        self.assertFalse(self.complete_llm_progress.call_args.kwargs["success"])
        self.assertTrue(client.closed)

    def test_refusal_cannot_become_valid_letter(self):
        client = FakeClient(response("A", stop_reason="refusal"))
        with patch.object(sc, "Anthropic", return_value=client), patch.object(sc, "get_llm_completion") as fallback:
            result = sc.evaluate_single_choice_question(QUESTION, "key", "claude-haiku-5-5", "Anthropic")
        self.assertIn("refused", result["error"])
        self.assertEqual(result["choice"], "?")
        fallback.assert_not_called()

    def test_single_choice_uses_same_prompt_budget_and_text_mode_without_fake_logprobs(self):
        for model, thinking in (
            ("claude-sonnet-5-5", {"type": "between_tools"}),
            ("claude-haiku-5-5", {"type": "disabled"}),
        ):
            with self.subTest(model=model):
                client = FakeClient(response("A", thinking=True))
                with patch.object(sc, "Anthropic", return_value=client):
                    result = sc.evaluate_single_choice_question(QUESTION, "key", model, "Anthropic")
                self.assertEqual(result["choice"], "A")
                self.assertIsNone(result["option_probabilities"])
                self.assertEqual(result["logit_mode"], "text")
                request = client.create.call_args.kwargs
                self.assertEqual(request["max_tokens"], 1)
                self.assertEqual(request["messages"][0]["content"], sc._build_sc_prompt_body(QUESTION))
                self.assertEqual(request["thinking"], thinking)
                self.assertNotIn("temperature", request)
                self.assertTrue(client.closed)

    def test_always_thinking_single_choice_fails_before_any_request(self):
        for provider, models in (
            ("Anthropic", ("claude-opus-5-5", "claude-fable-5-1")),
            ("Kimi", ("kimi-k3", "kimi-k2.7-code", "kimi-k2.7-code-highspeed")),
        ):
            for model in models:
                with self.subTest(model=model), patch.object(sc, "Anthropic") as claude, patch.object(sc, "OpenAI") as openai, patch.object(sc, "get_llm_completion") as fallback:
                    result = sc.evaluate_single_choice_question(QUESTION, "key", model, provider)
                    self.assertEqual(result["choice"], "?")
                    self.assertIn("one-token", result["error"])
                    claude.assert_not_called()
                    openai.assert_not_called()
                    fallback.assert_not_called()

    def test_claude_api_error_does_not_duplicate_request_via_same_endpoint(self):
        client = FakeClient(error=RuntimeError("400 invalid request"))
        with patch.object(sc, "Anthropic", return_value=client), patch.object(sc, "get_llm_completion") as fallback:
            result = sc.evaluate_single_choice_question(QUESTION, "key", "claude-sonnet-5-5", "Anthropic")
        self.assertIn("400", result["error"])
        client.create.assert_called_once()
        fallback.assert_not_called()
        self.assertTrue(client.closed)

    def test_sdk_without_sampling_keywords_keeps_older_model_values_on_wire(self):
        seen = {}
        def create(*, model, messages, max_tokens, extra_body=None):
            seen.update(extra_body or {})
            return response("A")
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        result = create_anthropic_message(client, {
            "model": "claude-sonnet-4-6", "messages": [], "max_tokens": 1,
            "temperature": .2, "top_p": .4, "extra_body": {"metadata": {"user_id": "example"}},
        })
        self.assertEqual(result.content[0].text, "A")
        self.assertEqual(seen["temperature"], .2)
        self.assertEqual(seen["top_p"], .4)
        self.assertEqual(seen["metadata"], {"user_id": "example"})

    def test_haiku_legacy_top_p_only_and_temperature_only_remain_valid(self):
        for controls in ({"top_p": .4}, {"temperature": .2}):
            with self.subTest(controls=controls):
                request = {"model": "claude-haiku-4-5-20251001", "max_tokens": 1, **controls}
                self.assertEqual(apply_anthropic_request_compat(request), request)

    def test_explicit_server_dual_sampling_400_retries_same_model_once(self):
        client = FakeClient()
        client.create.side_effect = [
            RuntimeError("400 invalid_request_error: temperature and top_p cannot both be specified"),
            response("A"),
        ]
        request = {
            "model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "same prompt"}],
            "max_tokens": 77, "temperature": .2, "top_p": .4, "stop_sequences": ["END"],
        }
        result = create_anthropic_message(client, request)
        self.assertEqual(extract_anthropic_response_text(result), "A")
        self.assertEqual(client.create.call_count, 2)
        first, second = [call.kwargs for call in client.create.call_args_list]
        self.assertEqual(first, request)
        self.assertEqual(second, {key: value for key, value in request.items() if key != "top_p"})
        self.assertEqual(request["top_p"], .4)

    def test_dual_sampling_retry_handles_removed_sdk_keywords(self):
        requests = []
        def create(*, model, messages, max_tokens, extra_body=None):
            requests.append({"model": model, "messages": messages, "max_tokens": max_tokens, "extra_body": dict(extra_body or {})})
            if len(requests) == 1:
                raise RuntimeError("400: specify only one of temperature or top_p")
            return response("A")
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        create_anthropic_message(client, {
            "model": "claude-opus-4-6", "messages": [], "max_tokens": 77,
            "temperature": .2, "top_p": .4,
        })
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["extra_body"], {"temperature": .2, "top_p": .4})
        self.assertEqual(requests[1]["extra_body"], {"temperature": .2})
        self.assertEqual(requests[0]["model"], requests[1]["model"])
        self.assertEqual(requests[0]["max_tokens"], requests[1]["max_tokens"])

    def test_other_errors_never_trigger_sampling_compatibility_retry(self):
        for message in (
            "400: invalid temperature value",
            "401: authentication failed",
            "429: temperature and top_p cannot both be specified",
            "503: temperature and top_p cannot both be specified",
            "temperature and top_p cannot both be specified",
        ):
            with self.subTest(message=message):
                client = FakeClient(error=RuntimeError(message))
                with self.assertRaises(RuntimeError):
                    create_anthropic_message(client, {
                        "model": "claude-sonnet-4-6", "messages": [], "max_tokens": 77,
                        "temperature": .2, "top_p": .4,
                    })
                client.create.assert_called_once()

    def test_installed_sdk_serializes_new_thinking_modes_without_network(self):
        for model, mode in (("claude-sonnet-5-5", "between_tools"), ("claude-haiku-5-5", "disabled")):
            with self.subTest(model=model):
                bodies = []
                def handler(request):
                    bodies.append(json.loads(request.content))
                    return httpx.Response(200, json={
                        "id": "msg_test", "type": "message", "role": "assistant", "model": model,
                        "content": [{"type": "text", "text": "A"}], "stop_reason": "end_turn",
                        "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 1},
                    })
                with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
                    with anthropic.Anthropic(api_key="test-key", http_client=transport, max_retries=0) as client:
                        result = create_anthropic_message(client, {
                            "model": model, "messages": [{"role": "user", "content": "original"}],
                            "max_tokens": 1, "temperature": 0, "top_p": .9,
                        })
                self.assertEqual(extract_anthropic_response_text(result), "A")
                self.assertEqual(len(bodies), 1)
                self.assertEqual(bodies[0]["thinking"], {"type": mode})
                self.assertEqual(bodies[0]["max_tokens"], 1)
                self.assertNotIn("temperature", bodies[0])
                self.assertNotIn("top_p", bodies[0])

    def test_retired_gemini_never_silently_switches_or_calls_api(self):
        for model in ("gemini-2.0-flash", "gemini-3-pro-preview"):
            with self.subTest(model=model), patch.object(comparison.genai, "Client") as outbound:
                result = comparison.get_llm_completion("prompt", "key", model, "Google Gemini")
                self.assertTrue(result.startswith("Error:"))
                self.assertIn(model, result)
                outbound.assert_not_called()
                with self.assertRaises(ValueError):
                    comparison._normalize_gemini_model(model)
        self.assertEqual(comparison._normalize_gemini_model(None), DEFAULT_MODELS["Google Gemini"])

    def test_current_gemini_keeps_supported_sampling_and_requested_model(self):
        client = FakeClient(response("answer"))
        with patch.object(comparison.genai, "Client", return_value=client):
            result = comparison.get_llm_completion(
                "prompt", "key", "gemini-3.8-flash", "Google Gemini",
                temperature=.2, top_p=.4, max_output_tokens=77,
            )
        self.assertEqual(result, "answer")
        request = client.create.call_args.kwargs
        config = types.GenerateContentConfig(**request["config"])
        self.assertEqual(request["model"], "gemini-3.8-flash")
        self.assertEqual(config.temperature, .2)
        self.assertEqual(config.top_p, .4)
        self.assertEqual(config.max_output_tokens, 77)

    def test_openrouter_429_keeps_model_and_uses_one_api_call(self):
        client = FakeClient(error=RuntimeError("429 rate limit"))
        model = "google/gemma-4-31b-it:free"
        with patch.object(comparison.openai, "OpenAI", return_value=client):
            result = comparison.get_llm_completion("prompt", "key", model, "OpenRouter", request_max_retries=0)
        self.assertIn("429", result)
        client.create.assert_called_once()
        self.assertEqual(client.create.call_args.kwargs["model"], model)
        self.assertTrue(client.closed)

    def test_single_choice_openrouter_429_does_not_switch_or_fallback(self):
        client = FakeClient(error=RuntimeError("429 rate limit"))
        model = "google/gemma-4-31b-it:free"
        with patch.object(sc, "OpenAI", return_value=client), patch.object(sc, "get_llm_completion") as fallback:
            result = sc.evaluate_single_choice_question(QUESTION, "key", model, "OpenRouter")
        self.assertIn("429", result["error"])
        self.assertEqual(client.create.call_args.kwargs["model"], model)
        client.create.assert_called_once()
        fallback.assert_not_called()

    def test_kimi_non_thinking_single_choice_preserves_one_token_cap(self):
        client = FakeClient(response("A"))
        with patch.object(comparison.openai, "OpenAI", return_value=client), patch.object(sc, "OpenAI") as legacy:
            result = sc.evaluate_single_choice_question(QUESTION, "key", "kimi-k2.6", "Kimi")
        self.assertEqual(result["choice"], "A")
        request = client.create.call_args.kwargs
        self.assertEqual(request["max_tokens"], 1)
        self.assertEqual(request["temperature"], .6)
        self.assertEqual(request["top_p"], .95)
        self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})
        legacy.assert_not_called()

    def test_kimi_k3_general_completion_preserves_budget_without_disabling_thinking(self):
        client = FakeClient(response("answer"))
        with patch.object(comparison.openai, "OpenAI", return_value=client):
            result = comparison.get_llm_completion("prompt", "key", "kimi-k3", "Kimi", max_output_tokens=77)
        self.assertEqual(result, "answer")
        request = client.create.call_args.kwargs
        self.assertEqual(request["max_completion_tokens"], 77)
        self.assertNotIn("max_tokens", request)
        self.assertNotIn("extra_body", request)

    def test_openai_short_answer_chat_reserves_budget_for_text_without_legacy_api(self):
        client = FakeClient(response("A"))
        with patch.object(comparison.openai, "OpenAI", return_value=client), patch.object(sc, "OpenAI") as legacy:
            result = sc.evaluate_single_choice_question(QUESTION, "key", "gpt-6-luna", "OpenAI")
        self.assertEqual(result["choice"], "A")
        request = client.create.call_args.kwargs
        self.assertEqual(request["max_completion_tokens"], 1)
        self.assertEqual(request["reasoning_effort"], "none")
        self.assertEqual(request["temperature"], .7)
        self.assertEqual(request["top_p"], .9)
        legacy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
