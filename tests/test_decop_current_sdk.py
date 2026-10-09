"""DECOP keeps its active baseline working with sampling-free SDK signatures."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import importlib.util
import unittest

import pandas as pd
from src.direct_recall import decop_analysis as decop


class NewSdkMessages:
    def __init__(self, content=None):
        self.request = None
        self.content = content if content is not None else [
            {"type": "thinking", "text": "D", "thinking": "hidden reasoning"},
            {"type": "text", "text": "A"},
        ]

    def create(self, *, model, max_tokens, messages, extra_body=None):
        self.request = {
            "model": model, "max_tokens": max_tokens,
            "messages": messages, "extra_body": extra_body,
        }
        return {"content": self.content, "stop_reason": "end_turn"}


class DeCopCurrentSdkTests(unittest.TestCase):
    def test_new_sdk_signature_keeps_baseline_model_prompt_and_temperature(self):
        messages = NewSdkMessages()
        row = pd.Series({f"Example_{label}": label + " passage" for label in "ABCD"})
        answer = decop.query_llm_claude(row, "document", "author", "BookTection", SimpleNamespace(messages=messages))
        self.assertEqual(answer, "A")
        self.assertEqual(messages.request["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(messages.request["max_tokens"], 1)
        self.assertEqual(messages.request["extra_body"], {"temperature": 0})
        self.assertIn('"document" book by author', messages.request["messages"][0]["content"])

    def test_thinking_only_response_is_not_scored_as_a_choice(self):
        messages = NewSdkMessages(content=[{"type": "thinking", "text": "A"}])
        row = pd.Series({f"Example_{label}": label for label in "ABCD"})
        with self.assertRaisesRegex(ValueError, "no valid single-choice"):
            decop.query_llm_claude(row, "document", "author", "BookTection", SimpleNamespace(messages=messages))

    def test_standalone_decop_script_uses_same_sdk_adapter(self):
        path = Path(decop.__file__).parent / "decop" / "2_decop_blackbox.py"
        spec = importlib.util.spec_from_file_location("standalone_decop_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        messages = NewSdkMessages()
        module.anthropic = SimpleNamespace(messages=messages)
        result = module.Query_LLM("BookTection", "Claude", ["A", "B", "C", "D"], "document", "author")
        self.assertEqual(result, "A")
        self.assertEqual(messages.request["model"], decop.DECOP_ANTHROPIC_MODEL)
        self.assertEqual(messages.request["max_tokens"], 1)
        self.assertEqual(messages.request["extra_body"], {"temperature": 0})


if __name__ == "__main__":
    unittest.main()
