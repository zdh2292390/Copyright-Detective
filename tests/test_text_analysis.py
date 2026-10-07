"""Text inference validation and partial-success behavior without provider calls."""

import unittest

from src.text_analysis import (
    TextAnalysisError, run_text_inferences, safe_numeric_setting,
    safe_selection_index, text_report_fingerprint, unpack_text_result,
    validate_text_parameters,
)


class RerunSignal(BaseException):
    pass


class TextAnalysisTests(unittest.TestCase):
    def test_two_and_three_item_results_preserve_exact_metric_values_and_logprobs(self):
        metrics = {"rouge_l": 0.63, "jaccard_index": 0.22, "levenshtein": 17.0}
        logprobs = [{"token": "text", "logprob": -0.1}]
        self.assertEqual(unpack_text_result(("generated", metrics)), ("generated", metrics, None))
        self.assertEqual(unpack_text_result(("generated", metrics, logprobs)), ("generated", metrics, logprobs))
        self.assertEqual(unpack_text_result(("Errors were common in the village.", metrics))[1], metrics)

    def test_invalid_and_error_results_are_never_counted_as_success(self):
        for result in (
            "Error: 429 resource exhausted", "unexpected bare text", None, ("only one",),
            ("Error calling API: 503 unavailable", None), ("", {"rouge_l": 0.0}),
            ("generated", None), ("generated", {}), ("generated", {"rouge_l": float("nan")}),
            ("generated", {"rouge_l": float("inf")}), ("generated", {"rouge_l": "0.2"}),
            ("generated", {"rouge_l": True}), ("generated", {"rouge_l": 10 ** 1000}),
        ):
            with self.subTest(result=result), self.assertRaises(TextAnalysisError):
                unpack_text_result(result)

    def test_multi_run_failure_retains_successes_and_stops_additional_calls(self):
        calls = []
        successes = []

        def analyze(index):
            calls.append(index)
            if index == 2:
                return "Error: timeout", None
            return f"generated-{index}", {"rouge_l": index / 10}

        with self.assertRaisesRegex(TextAnalysisError, "Run 3/5 failed"):
            run_text_inferences(5, analyze, lambda *result: successes.append(result))
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual([result[1] for result in successes], ["generated-0", "generated-1"])
        self.assertEqual([result[2]["rouge_l"] for result in successes], [0.0, 0.1])

    def test_raised_exception_is_reported_and_base_exception_keeps_prior_checkpoint(self):
        successes = []
        with self.assertRaisesRegex(TextAnalysisError, "TimeoutError"):
            run_text_inferences(1, lambda index: (_ for _ in ()).throw(TimeoutError()), lambda *args: successes.append(args))
        self.assertEqual(successes, [])

        def interrupted(index):
            if index:
                raise RerunSignal()
            return "generated", {"rouge_l": 0.3}

        with self.assertRaises(RerunSignal):
            run_text_inferences(3, interrupted, lambda *args: successes.append(args))
        self.assertEqual(len(successes), 1)

    def test_parameter_bounds_keep_existing_zero_top_p_and_temperature_range(self):
        validate_text_parameters(1, 0.0, 0.0)
        validate_text_parameters(1000, 1.2, 1.0)
        for runs, temperature, top_p in (
            (0, 0.7, 0.9), (1001, 0.7, 0.9), (True, 0.7, 0.9), (2.0, 0.7, 0.9),
            (1, float("nan"), 0.9), (1, 1.3, 0.9), (1, 0.7, -0.1), (1, 0.7, float("inf")),
        ):
            with self.subTest(parameters=(runs, temperature, top_p)), self.assertRaises(TextAnalysisError):
                validate_text_parameters(runs, temperature, top_p)

    def test_invalid_widget_cache_recovers_to_defaults_without_changing_valid_values(self):
        self.assertEqual(safe_numeric_setting(float("nan"), 0.7, 0, 1.2), 0.7)
        self.assertEqual(safe_numeric_setting("broken", 1, 1, 1000, integer=True), 1)
        self.assertEqual(safe_numeric_setting(3, 1, 1, 1000, integer=True), 3)
        self.assertEqual(safe_numeric_setting(0.0, 0.9, 0, 1), 0.0)
        self.assertEqual(safe_selection_index(-9, 3), 0)
        self.assertEqual(safe_selection_index(9, 3), 2)
        self.assertEqual(safe_selection_index("bad", 3), 0)

    def test_report_fingerprint_changes_with_results_and_captured_model(self):
        first = {"generated_text": "alpha", "user_inputs": {"model": "one"}}
        self.assertEqual(text_report_fingerprint(first), text_report_fingerprint(dict(reversed(list(first.items())))))
        self.assertNotEqual(text_report_fingerprint(first), text_report_fingerprint({**first, "generated_text": "gamma"}))
        self.assertNotEqual(text_report_fingerprint(first), text_report_fingerprint({**first, "user_inputs": {"model": "two"}}))


if __name__ == "__main__":
    unittest.main()
