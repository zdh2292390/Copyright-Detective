import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import torch

from src.direct_recall import knowledge_qa as qa
from src.direct_recall import single_choice as sc
from src.direct_recall import decop_analysis as decop
from src.pages import single_choice_detection as page


QUESTION = {
    "question": "Which passage?",
    "options": [{"label": label, "text": label + " text"} for label in sc.OPTION_LABELS],
    "correct_option": "A",
}


class FakeClient:
    def __init__(self, response=None, error=None):
        self.create = Mock(return_value=response, side_effect=error)
        self.completions = SimpleNamespace(create=self.create)
        self.chat = SimpleNamespace(completions=self.completions)
        self.messages = SimpleNamespace(create=self.create)
        self.closed = False

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class KnowledgeQATests(unittest.TestCase):
    def test_failed_question_keeps_positions_and_successful_score(self):
        completion = Mock(side_effect=["Error: 429 busy", "Answer: Paris"])
        results = qa.run_knowledge_qa_evaluation(
            [{"question": "First?", "answer": "London"}, {"question": "Second?", "answer": "Paris"}],
            "key", "model", "OpenAI", completion_fn=completion,
        )
        self.assertEqual(len(results[0]), 2)
        self.assertEqual(results[0][1]["qa_index"], 1)
        self.assertIn("error", results[0][0])
        self.assertNotIn("f1", results[0][0])
        metrics = qa.calculate_aggregate_metrics(results)
        self.assertEqual(metrics["avg_f1"], 1)
        self.assertEqual(metrics["total_evaluations"], 1)
        self.assertEqual(metrics["failed_evaluations"], 1)

    def test_first_empty_run_does_not_hide_later_success(self):
        result = qa.evaluate_qa_comparison("Q?", "Paris", "Paris")
        metrics = qa.calculate_aggregate_metrics([[], [result]])
        self.assertEqual(metrics["total_evaluations"], 1)
        self.assertEqual(metrics["avg_f1"], 1)

    def test_nontext_and_raised_answers_do_not_score(self):
        for response in (None, {}, "", TimeoutError()):
            with self.subTest(response=response):
                completion = Mock(side_effect=response) if isinstance(response, Exception) else Mock(return_value=response)
                results = qa.run_knowledge_qa_evaluation(
                    [{"question": "Q?", "answer": "Paris"}], "key", "model", "OpenAI",
                    completion_fn=completion,
                )
                self.assertIn("error", results[0][0])
                self.assertNotIn("f1", results[0][0])
                self.assertEqual(qa.calculate_aggregate_metrics(results)["total_evaluations"], 0)

    def test_generator_rejects_null_and_blank_pairs_without_changing_valid_pairs(self):
        response = json.dumps([
            {"question": None, "answer": "A"},
            {"question": "Q", "answer": " "},
            {"question": " Valid? ", "answer": " Paris "},
        ])
        with patch.object(qa, "get_llm_completion", return_value=response):
            result = qa.generate_qa_pairs_from_text("source", "key", "model", "OpenAI", num_pairs=3)
        self.assertEqual(result, [{"question": "Valid?", "answer": "Paris"}])

    def test_failed_judge_does_not_replace_token_score_or_invent_judge_score(self):
        with patch.object(qa, "llm_judge_evaluate", return_value={"score": None, "error": "judge unavailable"}):
            results = qa.run_knowledge_qa_evaluation(
                [{"question": "Q?", "answer": "Paris"}], "key", "model", "OpenAI",
                completion_fn=Mock(return_value="Paris"), llm_judge_fn=Mock(),
            )
        self.assertEqual(results[0][0]["f1"], 1)
        self.assertNotIn("llm_judge_score", results[0][0])
        self.assertIn("llm_judge_error", results[0][0])

    def test_direct_comparison_rejects_provider_error(self):
        with self.assertRaises(ValueError):
            qa.evaluate_qa_comparison("Q?", "API", "Error calling API: timeout")


class SingleChoiceTests(unittest.TestCase):
    def test_error_message_cannot_be_parsed_as_option(self):
        with patch.object(sc, "get_llm_completion", return_value="Error calling API: 503"):
            result = sc.evaluate_single_choice_question(QUESTION, "key", "model", "Google Gemini")
        self.assertEqual(result["choice"], "?")
        self.assertIn("error", result)

    def test_word_letter_does_not_become_answer(self):
        self.assertEqual(sc._extract_option_from_text("declined"), "")
        self.assertEqual(sc._extract_option_from_text("Answer: C"), "C")

    def test_sdk_completions_logprob_dictionary_and_nonfinite_values(self):
        probabilities = sc._parse_openai_top_logprobs([{" A": math.log(0.75), "B": math.log(0.25)}])
        self.assertAlmostEqual(probabilities["A"], 0.75)
        self.assertIsNone(sc._parse_openai_top_logprobs([{"A": float("nan")}]))
        self.assertIsNone(sc._normalize_option_probabilities({"A": float("inf")}))

    def test_specialized_timeout_closes_client_and_does_not_fallback_to_more_calls(self):
        client = FakeClient(error=TimeoutError())
        with patch.object(sc, "OpenAI", return_value=client):
            with patch.object(sc, "get_llm_completion") as fallback:
                result = sc.evaluate_single_choice_question(QUESTION, "key", "model", "OpenAI")
        self.assertIn("TimeoutError", result["error"])
        self.assertTrue(client.closed)
        fallback.assert_not_called()

    def test_batch_retains_failed_row_and_excludes_it_from_accuracy(self):
        with patch.object(sc, "evaluate_single_choice_question", side_effect=[
            TimeoutError(), {"choice": "A", "raw_response": "A"},
        ]):
            results = sc.run_single_choice_evaluation([QUESTION, QUESTION], "key", "model", "OpenAI")
        self.assertEqual(len(results[0]), 2)
        self.assertIsNone(results[0][0]["is_correct"])
        metrics = sc.summarize_single_choice_results(results)
        self.assertEqual(metrics["overall_accuracy"], 1)
        self.assertEqual(metrics["total_attempts"], 2)
        self.assertEqual(metrics["successful_attempts"], 1)
        self.assertEqual(metrics["failed_attempts"], 1)

    def test_option_absent_from_question_is_failure_instead_of_wrong_answer(self):
        with patch.object(sc, "get_llm_completion", return_value="C"):
            result = sc.evaluate_single_choice_question(
                {**QUESTION, "options": QUESTION["options"][:2]}, "key", "model", "Google Gemini"
            )
        self.assertIn("absent", result["error"])

    def test_invalid_input_rows_are_aligned_and_make_no_requests(self):
        invalid = [None, {**QUESTION, "options": [{"label": "A", "text": None}]}]
        with patch.object(sc, "get_llm_completion") as outbound:
            result = sc.run_single_choice_evaluation(invalid, "key", "model", "Google Gemini")
        self.assertEqual(len(result[0]), 2)
        self.assertTrue(all(row["error"] for row in result[0]))
        outbound.assert_not_called()

    def test_all_failed_attempts_have_no_accuracy(self):
        with patch.object(sc, "evaluate_single_choice_question", side_effect=TimeoutError()):
            result = sc.run_single_choice_evaluation([QUESTION], "key", "model", "OpenAI")
        self.assertIsNone(sc.summarize_single_choice_results(result)["overall_accuracy"])

    def test_normal_single_choice_score_and_count_are_unchanged(self):
        with patch.object(sc, "evaluate_single_choice_question", side_effect=[
            {"choice": "A"}, {"choice": "B"}, {"choice": "A"}, {"choice": "A"},
        ]):
            result = sc.run_single_choice_evaluation([QUESTION, QUESTION], "key", "model", "OpenAI", num_runs=2)
        self.assertEqual([len(run) for run in result], [2, 2])
        self.assertEqual(sc.summarize_single_choice_results(result)["overall_accuracy"], 0.75)

    def test_changing_source_clears_questions_and_previous_evaluation(self):
        state = {"sc_source_identity": "old", "sc_generated_mcqs": [QUESTION],
                 "sc_evaluation_results": [[{}]], "sc_document_text": "old source"}
        with patch.object(page.st, "session_state", state):
            page._sync_source_identity("new")
        self.assertEqual(state["sc_generated_mcqs"], [])
        self.assertIsNone(state["sc_evaluation_results"])


class DeCopTests(unittest.TestCase):
    def _row(self):
        return pd.Series({"Example_A": "A", "Example_B": "B", "Example_C": "C", "Example_D": "D"})

    def test_missing_logprobs_cannot_fabricate_uniform_A(self):
        client = FakeClient(SimpleNamespace(choices=[SimpleNamespace(logprobs=None)]))
        with self.assertRaises(ValueError):
            decop.query_llm_chatgpt(self._row(), "doc", "author", "BookTection", client)

    def test_nonfinite_logprobs_are_rejected(self):
        response = SimpleNamespace(choices=[
            SimpleNamespace(logprobs=SimpleNamespace(content=[
                SimpleNamespace(top_logprobs=[SimpleNamespace(token="A", logprob=float("nan"))])
            ]))
        ])
        with self.assertRaises(ValueError):
            decop.query_llm_chatgpt(self._row(), "doc", "author", "BookTection", FakeClient(response))

    def test_normal_probability_formula_is_unchanged(self):
        values = [-0.1, -1.0, -2.0, -3.0]
        response = SimpleNamespace(choices=[
            SimpleNamespace(logprobs=SimpleNamespace(content=[
                SimpleNamespace(top_logprobs=[SimpleNamespace(token=label, logprob=value)
                    for label, value in zip(sc.OPTION_LABELS, values)])
            ]))
        ])
        result = decop.query_llm_chatgpt(self._row(), "doc", "author", "BookTection", FakeClient(response))
        torch.testing.assert_close(result, torch.softmax(torch.tensor(values), dim=0))

    def test_overlapping_dataset_evaluation_returns_busy_without_requests(self):
        with patch.object(decop, "OpenAI") as outbound:
            decop._DATASET_EVALUATION_LOCK.acquire()
            try:
                success, message, output_dir = decop.run_dataset_evaluation("arXivTection", "ChatGPT", "key")
            finally:
                decop._DATASET_EVALUATION_LOCK.release()
        self.assertFalse(success)
        self.assertIn("Another dataset evaluation", message)
        self.assertIsNone(output_dir)
        outbound.assert_not_called()

    def test_dataset_error_returns_failure_and_closes_client(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            pd.DataFrame([{"ID": "doc", "Example_A": "A", "Example_B": "B",
                "Example_C": "C", "Example_D": "D", "Answer": "A"}]).to_csv(data_dir / "arXivTection.csv", index=False)
            client = FakeClient(error=TimeoutError())
            with patch.object(decop, "DATA_DIR", data_dir), patch.object(decop, "OpenAI", return_value=client):
                success, message, _ = decop.run_dataset_evaluation("arXivTection", "ChatGPT", "key")
            self.assertFalse(success)
            self.assertIn("TimeoutError", message)
            self.assertTrue(client.closed)
            self.assertEqual(list(data_dir.rglob("*.xlsx")), [])


if __name__ == "__main__":
    unittest.main()
