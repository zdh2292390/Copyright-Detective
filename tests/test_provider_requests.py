"""Provider request controls remain bounded without hidden SDK retry amplification."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.genai import types

from src.direct_recall import comparison
from src.document_analysis import new_analysis_state, run_chunk_analysis
from src.adversarial_persuasion_detection import jailbreak_probe


class _FakeClient:
    def __init__(self, error=None):
        response = SimpleNamespace(
            text="continuation",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="continuation", refusal=None),
                    finish_reason="stop",
                )
            ],
            content=[SimpleNamespace(text="continuation")],
        )
        create = Mock(return_value=response, side_effect=error)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self.messages = SimpleNamespace(create=create)
        self.models = SimpleNamespace(generate_content=create)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


class ProviderRequestTests(unittest.TestCase):
    PROVIDERS = ("OpenAI", "OpenRouter", "Anthropic", "Google Gemini", "Kimi", "Local vLLM")

    def _execute(self, provider, *, error=None, **kwargs):
        client = _FakeClient(error=error)
        options = dict(
            prompt="Continue",
            api_key="test-key",
            model_name="test-model",
            provider=provider,
            temperature=0.7,
            top_p=0.9,
            base_url="http://127.0.0.1:8000/v1" if provider == "Local vLLM" else None,
            max_output_tokens=None,
            stop_sequences=None,
            return_logprobs=False,
            request_timeout=None,
            request_max_retries=None,
            label_placeholder=None,
            bar_placeholder=None,
            progress_bar=None,
        )
        options.update(kwargs)
        if provider == "Google Gemini":
            constructor = patch.object(comparison.genai, "Client", return_value=client)
        elif provider == "Anthropic":
            constructor = patch.object(comparison.anthropic, "Anthropic", return_value=client)
        else:
            constructor = patch.object(comparison.openai, "OpenAI", return_value=client)
        with constructor as factory:
            with patch.object(comparison, "complete_llm_progress"):
                result = comparison._execute_llm_completion(**options)
        return result, factory.call_args.kwargs, client

    def test_explicit_timeout_and_zero_sdk_retries_for_every_provider(self):
        for provider in self.PROVIDERS:
            with self.subTest(provider=provider):
                result, kwargs, client = self._execute(
                    provider, request_timeout=120, request_max_retries=0
                )
                self.assertEqual(result, "continuation")
                if provider == "Google Gemini":
                    options = types.HttpOptions(**kwargs["http_options"])
                    self.assertEqual(options.timeout, 120000)
                    self.assertEqual(options.retry_options.attempts, 1)
                else:
                    self.assertEqual(kwargs["timeout"], 120)
                    self.assertEqual(kwargs["max_retries"], 0)
                self.assertTrue(client.closed)

    def test_clients_close_after_api_errors_for_every_provider(self):
        for provider in self.PROVIDERS:
            with self.subTest(provider=provider):
                result, _, client = self._execute(
                    provider,
                    error=RuntimeError("503 unavailable"),
                    request_timeout=120,
                    request_max_retries=0,
                )
                self.assertTrue(result.startswith("Error calling API:"))
                self.assertIn("503 unavailable", result)
                self.assertTrue(client.closed)

    def test_unspecified_controls_preserve_sdk_retry_defaults(self):
        for provider in self.PROVIDERS:
            with self.subTest(provider=provider):
                _, kwargs, _ = self._execute(provider)
                if provider == "Google Gemini":
                    self.assertNotIn("retry_options", kwargs["http_options"])
                else:
                    self.assertNotIn("max_retries", kwargs)
                    self.assertEqual(kwargs["timeout"], 120)

    def test_default_timeout_is_bounded_for_every_provider(self):
        for provider in self.PROVIDERS:
            with self.subTest(provider=provider):
                _, kwargs, _ = self._execute(provider)
                if provider == "Google Gemini":
                    self.assertEqual(kwargs["http_options"]["timeout"], 120000)
                else:
                    self.assertEqual(kwargs["timeout"], 120)

    def test_invalid_controls_never_call_provider(self):
        for controls in ({"request_timeout": float("nan")}, {"request_timeout": float("inf")}, {"request_timeout": -1}, {"request_max_retries": .5}, {"request_max_retries": -1}, {"request_max_retries": True}):
            with self.subTest(controls=controls), patch.object(comparison, "_execute_llm_completion") as outbound:
                result = comparison.get_llm_completion("prompt", "key", "model", **controls)
                self.assertTrue(result.startswith("Error:"))
                outbound.assert_not_called()

    def test_gemini_additional_retries_map_to_total_attempts(self):
        _, kwargs, _ = self._execute(
            "Google Gemini", request_timeout=12.5, request_max_retries=2
        )
        options = types.HttpOptions(**kwargs["http_options"])
        self.assertEqual(options.timeout, 12500)
        self.assertEqual(options.retry_options.attempts, 3)

    def test_local_endpoint_and_key_are_captured_without_worker_session_state(self):
        forbidden_state = SimpleNamespace(get=Mock(side_effect=AssertionError("worker session read")))
        with patch.object(comparison.st, "session_state", forbidden_state):
            result, kwargs, _ = self._execute(
                "Local vLLM", base_url="https://local.example/v1", api_key="captured-key"
            )
        self.assertEqual(result, "continuation")
        self.assertEqual(kwargs["base_url"], "https://local.example/v1")
        self.assertEqual(kwargs["api_key"], "captured-key")
        forbidden_state.get.assert_not_called()

    def test_keyless_local_completion_uses_nonsecret_placeholder(self):
        client = _FakeClient()
        forbidden_state = SimpleNamespace(get=Mock(side_effect=AssertionError("worker session read")))
        with patch.object(comparison.openai, "OpenAI", return_value=client) as factory:
            with patch.object(comparison.st, "session_state", forbidden_state):
                with patch.dict("os.environ", {"OPENAI_API_KEY": "remote-env-key"}):
                    result = comparison.get_llm_completion(
                        "Continue", "", "local-model", "Local vLLM",
                        base_url="http://127.0.0.1:8000/v1",
                        request_timeout=120, request_max_retries=0,
                    )
        self.assertEqual(result, "continuation")
        self.assertEqual(factory.call_args.kwargs["api_key"], "local-vllm")
        self.assertEqual(factory.call_args.kwargs["base_url"], "http://127.0.0.1:8000/v1")
        self.assertTrue(client.closed)
        forbidden_state.get.assert_not_called()

    def test_default_remote_completion_requires_key_before_outbound_work(self):
        with patch.object(comparison, "_execute_llm_completion") as outbound:
            result = comparison.get_llm_completion("Continue", "", "test-model")
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("API key is missing", result)
        outbound.assert_not_called()

    def test_concurrency_timeout_is_error_before_outbound_call_or_scoring(self):
        with patch.object(
            comparison, "limit_api_concurrency",
            side_effect=comparison.ApiConcurrencyTimeout("Too many concurrent API requests"),
        ) as limiter:
            with patch.object(comparison, "_execute_llm_completion") as outbound:
                with patch.object(comparison, "calculate_similarity_metrics") as scoring:
                    generated, metrics = comparison.compare_texts(
                        "prefix", "continuation", "test-key", "test-model",
                        request_timeout=120, request_max_retries=0,
                    )
        self.assertTrue(generated.startswith("Error:"))
        self.assertIn("Too many concurrent API requests", generated)
        self.assertIsNone(metrics)
        limiter.assert_called_once_with(timeout=120)
        outbound.assert_not_called()
        scoring.assert_not_called()

    def test_empty_network_exceptions_remain_retryable_through_provider_and_comparison(self):
        for exception_type in (TimeoutError, ConnectionError):
            with self.subTest(exception_type=exception_type):
                clients = [_FakeClient(error=exception_type()), _FakeClient(error=exception_type()), _FakeClient()]
                pairs = [("prefix", "continuation")]
                state = new_analysis_state("test-fingerprint", {"chunk_size": 200}, 1)
                with patch.object(comparison.openai, "OpenAI", side_effect=clients) as factory:
                    with patch.object(comparison, "calculate_similarity_metrics", return_value={"rouge_l": 0.5}):
                        run_chunk_analysis(
                            pairs,
                            lambda upper, lower: comparison.compare_texts(
                                upper, lower, "test-key", "test-model",
                                request_timeout=120, request_max_retries=0,
                            ),
                            state, sleep=lambda delay: None, rng=lambda: 0.0,
                        )
                self.assertEqual(factory.call_count, 3)
                self.assertEqual(state["status"], "complete")
                self.assertEqual(state["attempts"][0], 3)
                self.assertEqual(state["failures"], {})
                self.assertEqual(len(state["results"]), 1)
                self.assertTrue(all(client.closed for client in clients))

    def test_normal_continuation_forwards_request_controls_with_and_without_logprobs(self):
        for logprobs in (False, True):
            with self.subTest(logprobs=logprobs):
                response = ("continuation", []) if logprobs else "continuation"
                with patch.object(comparison, "get_llm_completion", return_value=response) as completion:
                    with patch.object(comparison, "calculate_similarity_metrics", return_value={}):
                        comparison.compare_texts(
                            "prefix",
                            "continuation",
                            "test-key",
                            "test-model",
                            return_logprobs=logprobs,
                            request_timeout=120,
                            request_max_retries=0,
                            base_url="https://local.example/v1",
                        )
                self.assertEqual(completion.call_args.kwargs["request_timeout"], 120)
                self.assertEqual(completion.call_args.kwargs["request_max_retries"], 0)
                self.assertEqual(completion.call_args.kwargs["base_url"], "https://local.example/v1")

    def test_persuasion_continuation_forwards_request_controls(self):
        with patch.object(jailbreak_probe, "get_llm_completion", return_value="continuation") as completion:
            with patch.object(jailbreak_probe, "calculate_similarity_metrics", return_value={}):
                jailbreak_probe.run_persuasion_probe(
                    "test-key",
                    "test-model",
                    "Local vLLM",
                    "Creative Writing Exercise",
                    "prefix",
                    "continuation",
                    request_timeout=120,
                    request_max_retries=0,
                    base_url="https://local.example/v1",
                )
        self.assertEqual(completion.call_args.kwargs["request_timeout"], 120)
        self.assertEqual(completion.call_args.kwargs["request_max_retries"], 0)
        self.assertEqual(completion.call_args.kwargs["base_url"], "https://local.example/v1")


if __name__ == "__main__":
    unittest.main()