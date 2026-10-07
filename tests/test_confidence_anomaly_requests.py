"""Confidence request resource handling and input validation without API calls."""

from contextlib import contextmanager
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.direct_recall import confidence_anomaly as confidence


class RerunSignal(BaseException):
    pass


class FakeClient:
    def __init__(self, responses):
        self.closed = False
        self.create = Mock(side_effect=responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


def response(text="generated passage", records=None):
    if records is None:
        records = [SimpleNamespace(token="generated", logprob=-0.1)]
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=text, refusal=None),
        finish_reason="stop", logprobs=SimpleNamespace(content=records),
    )])


class ConfidenceRequestTests(unittest.TestCase):
    def call(self, responses, **kwargs):
        client = FakeClient(responses)
        slot = {"active": False, "released": False}

        @contextmanager
        def concurrency(*, timeout):
            self.assertEqual(timeout, 120.0)
            slot["active"] = True
            try:
                yield
            finally:
                slot["active"] = False
                slot["released"] = True

        original = client.create.side_effect
        items = iter(original)

        def create(**request):
            self.assertTrue(slot["active"], "Every outbound request must hold an API slot.")
            item = next(items)
            if isinstance(item, BaseException):
                raise item
            return item

        client.create.side_effect = create
        arguments = dict(prompt="Continue exactly this prompt", api_key="secret-key", model_name="test-model")
        arguments.update(kwargs)
        with patch.object(confidence.openai, "OpenAI", return_value=client) as factory:
            with patch.object(confidence, "limit_api_concurrency", concurrency):
                result = confidence.get_completion_with_logprobs_openai(**arguments)
        return result, factory.call_args.kwargs, client, slot

    def test_success_has_bounded_timeout_no_sdk_retries_and_closed_slot_client(self):
        (generated, tokens, error), options, client, slot = self.call([response()], temperature=0.8, top_p=0.4, max_tokens=123)
        self.assertEqual(generated, "generated passage")
        self.assertIsNone(error)
        self.assertEqual(tokens[0].logprob, -0.1)
        self.assertEqual(tokens[0].linear_prob, math.exp(-0.1))
        self.assertEqual(options["timeout"], 120.0)
        self.assertEqual(options["max_retries"], 0)
        self.assertTrue(client.closed)
        self.assertTrue(slot["released"])
        request = client.create.call_args.kwargs
        self.assertEqual(request["messages"], [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Continue exactly this prompt"}])
        self.assertEqual((request["temperature"], request["top_p"], request["max_tokens"]), (0.8, 0.4, 123))

    def test_api_error_closes_resources_and_redacts_key_with_empty_error_class_preserved(self):
        for exc, expected in ((RuntimeError("503 unavailable secret-key"), "[redacted]"), (TimeoutError(), "TimeoutError")):
            with self.subTest(exception=type(exc)):
                (_, tokens, error), _, client, slot = self.call([exc])
                self.assertEqual(tokens, [])
                self.assertIn(expected, error)
                self.assertNotIn("secret-key", error)
                self.assertTrue(client.closed)
                self.assertTrue(slot["released"])

    def test_empty_text_missing_choices_and_missing_logprobs_return_explicit_errors(self):
        for generated_response, expected in (
            (response(text=""), "empty content"),
            (SimpleNamespace(choices=[]), "no response choices"),
            (response(records=[]), "did not return token logprobs"),
        ):
            with self.subTest(expected=expected):
                (_, tokens, error), _, client, slot = self.call([generated_response])
                self.assertEqual(tokens, [])
                self.assertIn(expected, error)
                self.assertTrue(client.closed)
                self.assertTrue(slot["released"])

    def test_invalid_token_logprobs_are_rejected_without_creating_false_zero_scores(self):
        for token, logprob in (("text", float("nan")), ("text", float("inf")), ("text", -float("inf")), ("text", 0.01), ("text", None), ("text", True), (None, -0.1)):
            with self.subTest(token=token, logprob=logprob):
                (_, tokens, error), _, client, _ = self.call([response(records=[SimpleNamespace(token=token, logprob=logprob)])])
                self.assertEqual(tokens, [])
                self.assertTrue(error)
                self.assertTrue(client.closed)

    def test_gemma_429_fallback_keeps_original_prompt_and_only_expected_second_request(self):
        (generated, _, error), options, client, slot = self.call(
            [RuntimeError("429 rate limited"), response()],
            model_name="google/gemma-4-31b-it:free", base_url="https://openrouter.ai/api/v1",
            extra_headers={"X-Title": "Copyright Detective"},
        )
        self.assertEqual(generated, "generated passage")
        self.assertIsNone(error)
        self.assertEqual(options["base_url"], "https://openrouter.ai/api/v1")
        requests = [call.kwargs for call in client.create.call_args_list]
        self.assertEqual([request["model"] for request in requests], ["google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free"])
        for request in requests:
            self.assertEqual(request["messages"], [{"role": "user", "content": "You are a helpful assistant.\n\nContinue exactly this prompt"}])
            self.assertEqual(request["extra_headers"], {"X-Title": "Copyright Detective"})
        self.assertTrue(client.closed)
        self.assertTrue(slot["released"])

    def test_gemma_fallback_failure_still_closes_resources(self):
        (_, _, error), _, client, slot = self.call(
            [RuntimeError("429 rate limited"), RuntimeError("503 unavailable")],
            model_name="google/gemma-4-31b-it:free", base_url="https://openrouter.ai/api/v1",
        )
        self.assertIn("503 unavailable", error)
        self.assertEqual(client.create.call_count, 2)
        self.assertTrue(client.closed)
        self.assertTrue(slot["released"])

    def test_invalid_inputs_make_no_client_or_outbound_request(self):
        for changed in ({"api_key": " "}, {"prompt": " "}, {"model_name": ""}, {"temperature": float("nan")}, {"top_p": 1.1}, {"max_tokens": 0}, {"max_tokens": True}):
            with self.subTest(changed=changed):
                options = dict(prompt="prompt", api_key="key", model_name="model")
                options.update(changed)
                with patch.object(confidence.openai, "OpenAI") as factory:
                    _, tokens, error = confidence.get_completion_with_logprobs_openai(**options)
                factory.assert_not_called()
                self.assertEqual(tokens, [])
                self.assertTrue(error)

    def test_genuine_generated_error_word_is_not_treated_as_provider_failure(self):
        (generated, tokens, error), _, _, _ = self.call([response(text="Errors were common in the village.")])
        self.assertEqual(generated, "Errors were common in the village.")
        self.assertIsNone(error)
        self.assertEqual(len(tokens), 1)

    def test_api_slot_denial_does_not_construct_a_client(self):
        @contextmanager
        def denied(**kwargs):
            raise RuntimeError("Too many concurrent API requests")
            yield

        with patch.object(confidence.openai, "OpenAI") as factory:
            with patch.object(confidence, "limit_api_concurrency", denied):
                _, tokens, error = confidence.get_completion_with_logprobs_openai("prompt", "key", "model")
        factory.assert_not_called()
        self.assertEqual(tokens, [])
        self.assertIn("Too many concurrent API requests", error)

    def test_rerun_signal_propagates_and_still_closes_client_and_api_slot(self):
        client = FakeClient([RerunSignal()])
        released = []

        @contextmanager
        def slot(**kwargs):
            try:
                yield
            finally:
                released.append(True)

        with patch.object(confidence.openai, "OpenAI", return_value=client):
            with patch.object(confidence, "limit_api_concurrency", slot):
                with self.assertRaises(RerunSignal):
                    confidence.get_completion_with_logprobs_openai("prompt", "key", "model")
        self.assertTrue(client.closed)
        self.assertEqual(released, [True])

    def test_extreme_finite_logprob_perplexity_overflow_is_positive_infinity(self):
        token = confidence.TokenLogprob("rare", -9999.0, 0.0)
        self.assertEqual(confidence._calculate_perplexity([token]), float("inf"))
        self.assertEqual(confidence._calculate_perplexity([confidence.TokenLogprob("rare", -0.1, math.exp(-0.1))]), math.exp(0.1))

    def test_invalid_preexisting_data_is_explicitly_unavailable(self):
        for data, text in (
            ([], "generated"), ([{}], "generated"),
            ([{"token": "text", "logprob": float("nan"), "linear_prob": 0.9}], "generated"),
            ([{"token": "text", "logprob": 0.1, "linear_prob": 0.9}], "generated"),
            ([{"token": "text", "logprob": -0.1, "linear_prob": float("nan")}], "generated"),
            ([{"token": "text", "logprob": -0.1, "linear_prob": 0.9}], ""),
        ):
            with self.subTest(data=data, text=text):
                result = confidence.analyze_logprobs_for_confidence(data, text)
                self.assertFalse(result.analysis_available)
                self.assertTrue(result.error_message)
                self.assertEqual(result.tokens, [])

    def test_valid_preexisting_probability_values_are_preserved(self):
        data = [{"token": token, "logprob": logprob, "linear_prob": math.exp(logprob)} for token, logprob in (("rare", -0.01), (" bright", -0.02), (" constellation", -0.03), (" the", -1.0))]
        result = confidence.analyze_logprobs_for_confidence(data, "rare bright constellation the")
        self.assertTrue(result.analysis_available)
        self.assertEqual(result.confidence_timeline, [record["linear_prob"] for record in data])
        self.assertEqual([token.logprob for token in result.tokens], [record["logprob"] for record in data])
        self.assertAlmostEqual(result.perplexity, math.exp(sum(-record["logprob"] for record in data) / len(data)))
        self.assertEqual(result.high_confidence_ratio, 0.75)
        # Golden values independently verified against the original detector.
        self.assertAlmostEqual(result.memorization_score, 0.6675000000000001)
        self.assertEqual(result.spike_coverage, 0.75)
        self.assertEqual(result.longest_spike_length, 3)
        self.assertAlmostEqual(result.avg_entropy, 0.1538263773348205)


if __name__ == "__main__":
    unittest.main()
