"""Catalog migration and saved-model identity regressions; no inference requests."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from src.document_analysis import analysis_fingerprint, new_analysis_state
from src.document_checkpoints import DocumentCheckpointStore
from src.document_jobs import DocumentAnalysisJobs
from src.model_catalog import MODEL_CONFIG, model_replacement, model_unavailability_error


def fixture(model="kimi-k2.5", provider="Kimi", count=3):
    settings = {
        "filename": "source.txt", "model": model, "provider": provider,
        "chunk_size": 200, "overlap": 50, "continuation_method": "Normal Continuation",
        "temperature": 0.7, "top_p": 0.9,
    }
    pairs = [(f"prefix-{i}", f"target-{i}") for i in range(count)]
    state = new_analysis_state(analysis_fingerprint("original document", settings), settings, count)
    state["results"][0] = (*pairs[0], "saved continuation", {"rouge_l": 0.0})
    state["attempts"][0] = 1
    state["status"] = "incomplete"
    return state, pairs


class ModelCatalogTests(unittest.TestCase):
    def test_defaults_are_selectable_and_not_known_unavailable(self):
        for provider, config in MODEL_CONFIG.items():
            with self.subTest(provider=provider):
                models = config["models"]
                self.assertTrue(models)
                self.assertEqual(len(models), len(set(models)))
                self.assertIn(models[config["default_index"]], models)
                for model in models:
                    self.assertIsNone(model_unavailability_error(provider, model))

    def test_retirement_is_provider_scoped(self):
        self.assertIn("shut down", model_unavailability_error("Kimi", "kimi-k2.5"))
        self.assertIsNone(model_unavailability_error("OpenRouter", "moonshotai/kimi-k2.5"))
        self.assertIsNone(model_unavailability_error("Local vLLM", "kimi-k2.5"))
        self.assertIsNone(model_unavailability_error("OpenAI", "gpt-4o-mini"))
        self.assertIsNone(model_unavailability_error("OpenAI", "gpt-5.1"))
        self.assertIsNone(model_unavailability_error("Anthropic", "claude-sonnet-4-5-20250929"))

    def test_confirmed_retired_openai_and_claude_models_fail_with_replacements(self):
        for provider, model, replacement in (
            ("OpenAI", "gpt-5.1-chat-latest", "gpt-4o-mini"),
            ("Anthropic", "claude-opus-4-1-20250805", "claude-opus-5-5"),
            ("Anthropic", "claude-3-5-haiku-20241022", "claude-haiku-5-5"),
        ):
            with self.subTest(provider=provider, model=model):
                self.assertIsNotNone(model_unavailability_error(provider, model))
                self.assertEqual(model_replacement(provider, model), replacement)
                self.assertIn(replacement, MODEL_CONFIG[provider]["models"])
                self.assertIsNone(model_unavailability_error("Local vLLM", model))

    def test_removed_free_route_never_suggests_paid_route(self):
        model = "openai/gpt-oss-20b:free"
        self.assertIn("no longer listed", model_unavailability_error("OpenRouter", model))
        replacement = model_replacement("OpenRouter", model)
        self.assertTrue(replacement.endswith(":free"))
        self.assertIsNone(model_unavailability_error("OpenRouter", "openai/gpt-oss-20b"))

    def test_gemini_resource_prefix_is_recognized_without_rewriting_valid_id(self):
        self.assertIn("shut down", model_unavailability_error("Google Gemini", "models/gemini-2.0-flash"))
        self.assertEqual(model_replacement("Google Gemini", "gemini-3-pro-preview"), "gemini-3.1-pro-preview")
        self.assertIsNone(model_unavailability_error("Google Gemini", "gemini-2.5-pro"))
        self.assertIsNone(model_unavailability_error("Google Gemini", "models/gemini-3.8-flash"))


class RetiredDocumentModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="retired-document-model-")
        self.store = DocumentCheckpointStore(Path(self.temp.name))
        self.analyze = Mock(side_effect=AssertionError("must not send inference"))
        self.manager = DocumentAnalysisJobs(store=self.store, analyze_chunk=self.analyze)

    def tearDown(self):
        self.manager.close()
        self.temp.cleanup()

    def test_new_run_with_retired_model_is_rejected_before_checkpoint_creation(self):
        state, pairs = fixture()
        with self.assertRaisesRegex(ValueError, "start a new analysis"):
            self.manager.create(state, pairs)
        self.analyze.assert_not_called()

    def test_partial_retired_run_is_readable_but_cannot_resume_or_change_identity(self):
        state, pairs = fixture(count=110)
        for index in range(104):
            state["results"][index] = (*pairs[index], f"saved-{index}", {"rouge_l": 0.0})
            state["attempts"][index] = 1
        original = deepcopy(state)
        token = self.store.create(state, pairs)
        restored = self.manager.get(token)
        self.assertEqual(len(restored["results"]), 104)
        with self.assertRaisesRegex(ValueError, "cannot resume with a different model"):
            self.manager.submit(token, "fake-key")
        final = self.manager.get(token)
        self.assertEqual(final, original)
        self.assertFalse(self.manager.is_running(token))
        self.analyze.assert_not_called()

    def test_completed_historical_run_remains_readable(self):
        state, pairs = fixture(count=1)
        state["status"] = "complete"
        token = self.store.create(state, pairs)
        self.assertEqual(self.manager.get(token)["results"], state["results"])
        self.assertFalse(self.manager.submit(token, "fake-key"))
        self.analyze.assert_not_called()


if __name__ == "__main__":
    unittest.main()
