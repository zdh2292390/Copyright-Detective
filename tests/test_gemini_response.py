"""Gemini responses with HTTP success may still contain no generated text."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.genai import types

from src.direct_recall import comparison


class _FakeGeminiClient:
    def __init__(self, response=None, error=None):
        self.closed = False
        self.models = SimpleNamespace(
            generate_content=Mock(return_value=response, side_effect=error)
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


class GeminiResponseTests(unittest.TestCase):
    def _execute(self, client, **kwargs):
        options = dict(
            prompt="Continue the passage",
            api_key="test-key",
            model_name="gemini-3.5-flash",
            provider="Google Gemini",
            temperature=0.7,
            top_p=0.9,
            base_url=None,
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
        with patch.object(comparison.genai, "Client", return_value=client) as factory:
            with patch.object(comparison, "complete_llm_progress") as progress:
                result = comparison._execute_llm_completion(**options)
        return result, factory, progress

    def test_text_is_preserved_and_trimmed(self):
        response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(parts=[types.Part(text="  continuation  ")]),
                    finish_reason=types.FinishReason.STOP,
                )
            ]
        )
        self.assertEqual(comparison._extract_gemini_response_text(response), "continuation")

    def test_recitation_without_content_keeps_finish_reason(self):
        response = types.GenerateContentResponse(
            candidates=[types.Candidate(finish_reason=types.FinishReason.RECITATION)]
        )
        text = comparison._extract_gemini_response_text(response)
        self.assertTrue(text.startswith("Error:"))
        self.assertIn("finish_reason=RECITATION", text)
        self.assertNotIn("NoneType", text)

    def test_blocked_prompt_keeps_block_reason(self):
        response = types.GenerateContentResponse(
            prompt_feedback=types.GenerateContentResponsePromptFeedback(
                block_reason=types.BlockedReason.SAFETY
            )
        )
        self.assertIn("block_reason=SAFETY", comparison._extract_gemini_response_text(response))

    def test_empty_output_at_token_limit_keeps_reason(self):
        response = types.GenerateContentResponse(
            candidates=[types.Candidate(finish_reason=types.FinishReason.MAX_TOKENS)]
        )
        self.assertIn("finish_reason=MAX_TOKENS", comparison._extract_gemini_response_text(response))

    def test_no_candidates_and_whitespace_are_errors(self):
        for response in [types.GenerateContentResponse(), SimpleNamespace(text=" \n ")]:
            with self.subTest(response=response):
                text = comparison._extract_gemini_response_text(response)
                self.assertTrue(text.startswith("Error:"))
                self.assertIn("no text candidate returned", text)

    def test_timeout_is_converted_to_milliseconds_and_client_closed(self):
        client = _FakeGeminiClient(SimpleNamespace(text="continuation"))
        result, factory, _ = self._execute(client, request_timeout=12.5)
        self.assertEqual(result, "continuation")
        factory.assert_called_once_with(api_key="test-key", http_options={"timeout": 12500})
        self.assertTrue(client.closed)

    def test_default_timeout_is_bounded(self):
        client = _FakeGeminiClient(SimpleNamespace(text="continuation"))
        _, factory, _ = self._execute(client)
        factory.assert_called_once_with(api_key="test-key", http_options={"timeout": 120000})

    def test_api_exception_also_closes_client(self):
        client = _FakeGeminiClient(error=RuntimeError("429 RESOURCE_EXHAUSTED"))
        result, _, progress = self._execute(client)
        self.assertIn("429 RESOURCE_EXHAUSTED", result)
        self.assertTrue(result.startswith("Error calling API:"))
        self.assertTrue(client.closed)
        self.assertFalse(progress.call_args.kwargs["success"])

    def test_empty_content_is_reported_as_failure(self):
        client = _FakeGeminiClient(
            types.GenerateContentResponse(
                candidates=[types.Candidate(finish_reason=types.FinishReason.RECITATION)]
            )
        )
        result, _, progress = self._execute(client, return_logprobs=True)
        self.assertEqual(result[1], None)
        self.assertIn("finish_reason=RECITATION", result[0])
        self.assertTrue(client.closed)
        self.assertFalse(progress.call_args.kwargs["success"])


if __name__ == "__main__":
    unittest.main()
