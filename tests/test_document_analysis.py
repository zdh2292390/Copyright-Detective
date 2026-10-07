"""Regression tests for complete, resumable document analysis without API calls."""

import copy
import unittest
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import patch

from src.document_analysis import (
    analysis_fingerprint,
    analysis_results,
    new_analysis_state,
    run_chunk_analysis,
    validate_analysis_state,
)


class RerunSignal(BaseException):
    """Match Streamlit rerun/stop signals, which are not ordinary exceptions."""


class DocumentAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.settings = {
            "chunk_size": 200,
            "chunk_overlap": 50,
            "model": "Gemini 3.5 Flash",
            "continuation_method": "Normal Continuation",
        }

    @staticmethod
    def pairs(count):
        return [(f"prefix-{index}", f"reference-{index}") for index in range(count)]

    def state(self, count):
        return new_analysis_state(
            analysis_fingerprint("document contents", self.settings),
            self.settings,
            count,
        )

    @staticmethod
    def success(upper, lower):
        # A zero score still represents a valid, completed comparison.
        return f"generated for {upper}", {"rouge_l": 0.0}

    def test_1974_chunks_preserve_104_successes_and_resume_missing_only(self):
        pairs = self.pairs(1974)
        state = self.state(len(pairs))
        calls = []
        sleeps = []
        checkpoints = []

        def initial_analysis(upper, lower):
            index = int(upper.rsplit("-", 1)[1])
            calls.append(index)
            if index == 104:
                # Normal Continuation's error contract; no 503 is involved.
                return "Error: Request timed out.", None
            return self.success(upper, lower)

        returned = run_chunk_analysis(
            pairs,
            initial_analysis,
            state,
            sleep=sleeps.append, rng=lambda: 0.0,
            on_update=lambda update: checkpoints.append(copy.deepcopy(update)),
        )

        self.assertIs(returned, state)
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(state["total_chunks"], 1974)
        self.assertEqual(len(state["results"]), 104)
        self.assertEqual(set(state["results"]), set(range(104)))
        self.assertEqual(calls, list(range(104)) + [104, 104, 104])
        self.assertEqual(sleeps, [1, 2])
        self.assertEqual(state["attempts"][104], 3)
        self.assertIn("Chunk 105", state["error"])
        self.assertEqual(checkpoints[-1]["status"], "incomplete")
        self.assertEqual(len(checkpoints[-1]["results"]), 104)

        resumed_calls = []

        def resumed_analysis(upper, lower):
            resumed_calls.append(int(upper.rsplit("-", 1)[1]))
            return self.success(upper, lower)

        run_chunk_analysis(pairs, resumed_analysis, state, sleep=sleeps.append, rng=lambda: 0.0)

        self.assertEqual(resumed_calls, list(range(104, 1974)))
        self.assertEqual(state["status"], "complete")
        self.assertEqual(len(analysis_results(state)), 1974)
        self.assertEqual(state["failures"], {})
        self.assertIsNone(state["error"])
        self.assertTrue(all(state["attempts"][index] == 1 for index in range(104)))
        self.assertEqual(state["attempts"][104], 4)
        self.assertEqual(
            [result[0] for result in analysis_results(state)],
            [upper for upper, _ in pairs],
        )

    def test_transient_returned_error_and_exception_recover_with_bounded_retries(self):
        pairs = self.pairs(2)
        state = self.state(len(pairs))
        calls = Counter()
        sleeps = []

        def analyze(upper, lower):
            calls[upper] += 1
            if upper == "prefix-0":
                if calls[upper] == 1:
                    return "Error: 429 RESOURCE_EXHAUSTED", None
                if calls[upper] == 2:
                    raise TimeoutError("request timed out")
            return self.success(upper, lower)

        run_chunk_analysis(pairs, analyze, state, sleep=sleeps.append, rng=lambda: 0.0)

        self.assertEqual(state["status"], "complete")
        self.assertEqual(calls, Counter({"prefix-0": 3, "prefix-1": 1}))
        self.assertEqual(sleeps, [1, 2])
        self.assertEqual(state["failures"], {})
        self.assertIsNone(state["error"])

    def test_bare_transient_error_is_not_counted_as_success(self):
        state = self.state(1)
        calls = []
        sleeps = []

        def analyze(upper, lower):
            calls.append(upper)
            return "Error: 502 Bad Gateway"

        run_chunk_analysis(self.pairs(1), analyze, state, sleep=sleeps.append, rng=lambda: 0.0)

        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [1, 2])
        self.assertEqual(state["results"], {})
        self.assertEqual(state["status"], "incomplete")
        self.assertIn(0, state["failures"])

    def test_isolated_blocked_and_empty_chunks_do_not_prevent_remaining_analysis(self):
        pairs = self.pairs(5)
        state = self.state(len(pairs))
        calls = []
        sleeps = []

        def analyze(upper, lower):
            calls.append(upper)
            if upper == "prefix-1":
                return "Error: Response blocked (finish_reason=SAFETY)", None
            if upper == "prefix-3":
                return "", {"rouge_l": 0.0}
            return self.success(upper, lower)

        run_chunk_analysis(pairs, analyze, state, sleep=sleeps.append, rng=lambda: 0.0)

        self.assertEqual(calls, [upper for upper, _ in pairs])
        self.assertEqual(sleeps, [])
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(set(state["results"]), {0, 2, 4})
        self.assertEqual(set(state["failures"]), {1, 3})

        resumed_calls = []

        def resume(upper, lower):
            resumed_calls.append(upper)
            return self.success(upper, lower)

        run_chunk_analysis(pairs, resume, state, sleep=sleeps.append, rng=lambda: 0.0)

        self.assertEqual(resumed_calls, ["prefix-1", "prefix-3"])
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["failures"], {})
        self.assertIsNone(state["error"])
        self.assertEqual(
            [result[0] for result in analysis_results(state)],
            [upper for upper, _ in pairs],
        )

    def test_gemini_empty_candidate_is_isolated_and_remaining_chunks_continue(self):
        pairs = self.pairs(3)
        state = self.state(len(pairs))
        calls = []

        def analyze(upper, lower):
            calls.append(upper)
            if upper == "prefix-1":
                return (
                    "Error: Gemini returned empty content (no text candidate returned).",
                    None,
                )
            return self.success(upper, lower)

        run_chunk_analysis(pairs, analyze, state)

        self.assertEqual(calls, [upper for upper, _ in pairs])
        self.assertEqual(set(state["results"]), {0, 2})
        self.assertEqual(set(state["failures"]), {1})
        self.assertEqual(state["status"], "incomplete")

    def test_auth_and_config_errors_with_block_markers_pause_without_retry(self):
        for message in (
            "Error: 403 PERMISSION_DENIED: service unavailable (block_reason=restricted API key)",
            "Error: 404 NOT_FOUND: model unavailable (finish_reason=OTHER)",
        ):
            with self.subTest(message=message):
                pairs = self.pairs(3)
                state = self.state(len(pairs))
                calls = []
                sleeps = []

                def analyze(upper, lower):
                    calls.append(upper)
                    if upper == "prefix-1":
                        return message, None
                    return self.success(upper, lower)

                run_chunk_analysis(pairs, analyze, state, sleep=sleeps.append, rng=lambda: 0.0)

                self.assertEqual(calls, ["prefix-0", "prefix-1"])
                self.assertEqual(sleeps, [])
                self.assertEqual(set(state["results"]), {0})
                self.assertEqual(state["status"], "incomplete")
                self.assertEqual(state["attempts"][1], 1)

    def test_authentication_error_pauses_without_retry_and_can_resume(self):
        pairs = self.pairs(3)
        state = self.state(len(pairs))
        calls = []
        sleeps = []

        def analyze(upper, lower):
            calls.append(upper)
            if upper == "prefix-1":
                return "Error: 401 Unauthorized: invalid API key", None
            return self.success(upper, lower)

        run_chunk_analysis(pairs, analyze, state, sleep=sleeps.append, rng=lambda: 0.0)

        self.assertEqual(calls, ["prefix-0", "prefix-1"])
        self.assertEqual(sleeps, [])
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(state["attempts"][1], 1)
        self.assertEqual(set(state["results"]), {0})
        self.assertIn("401", state["error"])

        resumed_calls = []

        def resume(upper, lower):
            resumed_calls.append(upper)
            return self.success(upper, lower)

        run_chunk_analysis(pairs, resume, state, sleep=sleeps.append, rng=lambda: 0.0)
        self.assertEqual(resumed_calls, ["prefix-1", "prefix-2"])
        self.assertEqual(state["status"], "complete")

    def test_rerun_signal_keeps_checkpoints_and_resumes_without_duplicate_successes(self):
        pairs = self.pairs(4)
        state = self.state(len(pairs))
        calls = []
        checkpoints = []

        def analyze(upper, lower):
            calls.append(upper)
            if upper == "prefix-2":
                raise RerunSignal()
            return self.success(upper, lower)

        with self.assertRaises(RerunSignal):
            run_chunk_analysis(
                pairs,
                analyze,
                state,
                on_update=lambda update: checkpoints.append(copy.deepcopy(update)),
                sleep=lambda delay: self.fail("An interruption must not be retried."),
            )

        self.assertEqual(calls, ["prefix-0", "prefix-1", "prefix-2"])
        self.assertEqual(set(state["results"]), {0, 1})
        self.assertEqual(set(checkpoints[-1]["results"]), {0, 1})
        self.assertNotEqual(state["status"], "complete")

        resumed_calls = []

        def resume(upper, lower):
            resumed_calls.append(upper)
            return self.success(upper, lower)

        run_chunk_analysis(pairs, resume, state)

        self.assertEqual(resumed_calls, ["prefix-2", "prefix-3"])
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["attempts"][2], 2)

    def test_fingerprint_matches_contents_and_all_generation_settings(self):
        text = "alpha beta"
        original = analysis_fingerprint(text, self.settings)
        reordered = dict(reversed(list(self.settings.items())))
        self.assertEqual(analysis_fingerprint(text, reordered), original)
        self.assertEqual(len(text), len("gamma beta"))
        self.assertNotEqual(analysis_fingerprint("gamma beta", self.settings), original)
        self.assertNotEqual(analysis_fingerprint(text + " more", self.settings), original)

        for key, value in (
            ("chunk_size", 300),
            ("chunk_overlap", 0),
            ("model", "another model"),
            ("continuation_method", "another continuation method"),
        ):
            with self.subTest(setting=key):
                changed = dict(self.settings, **{key: value})
                self.assertNotEqual(analysis_fingerprint(text, changed), original)

    def test_valid_generated_text_starting_with_error_is_analyzed(self):
        state = self.state(1)
        generated = "Errors were common in the village."

        run_chunk_analysis(
            self.pairs(1),
            lambda upper, lower: (generated, {"rouge_l": 0.0}),
            state,
        )

        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["results"][0][2], generated)
        self.assertEqual(state["results"][0][3], {"rouge_l": 0.0})
        self.assertEqual(state["failures"], {})
        self.assertIsNone(state["error"])

    def test_missing_or_invalid_metrics_do_not_count_as_analyzed_chunks(self):
        for metrics in (None, {}, [], "invalid metrics"):
            with self.subTest(metrics=metrics):
                pairs = self.pairs(3)
                state = self.state(len(pairs))
                calls = []

                def analyze(upper, lower):
                    calls.append(upper)
                    if upper == "prefix-1":
                        return "generated text", metrics
                    return self.success(upper, lower)

                run_chunk_analysis(pairs, analyze, state)

                self.assertEqual(calls, ["prefix-0", "prefix-1"])
                self.assertEqual(set(state["results"]), {0})
                self.assertEqual(state["status"], "incomplete")
                self.assertIn("similarity metrics", state["error"])

    def test_checkpoint_rejects_wrong_indexes_pairs_and_invalid_metrics_before_calls(self):
        pairs = self.pairs(1)
        valid = ("prefix-0", "reference-0", "generated", {"rouge_l": 0.0})
        corrupt_results = (
            {-1: valid},
            {1: valid},
            {"0": valid},
            {False: valid},
            {0: ("wrong prefix", "reference-0", "generated", {"rouge_l": 0.0})},
            {0: ("prefix-0", "wrong reference", "generated", {"rouge_l": 0.0})},
            {0: ("prefix-0", "reference-0", "generated")},
            {0: ("prefix-0", "reference-0", "", {"rouge_l": 0.0})},
            {0: ("prefix-0", "reference-0", "generated", {"rouge_l": float("nan")})},
            {0: ("prefix-0", "reference-0", "generated", {"rouge_l": float("inf")})},
            {0: ("prefix-0", "reference-0", "generated", {"rouge_l": "0.5"})},
            {0: ("prefix-0", "reference-0", "generated", {"rouge_l": True})},
        )
        for results in corrupt_results:
            with self.subTest(results=results):
                state = self.state(1)
                state["results"] = results
                state["status"] = "complete"
                original = copy.deepcopy(state)
                with self.assertRaises(ValueError):
                    run_chunk_analysis(
                        pairs,
                        lambda upper, lower: self.fail("Corrupt checkpoints must not make calls."),
                        state,
                    )
                # Validation happens before a complete checkpoint is trusted or mutated.
                self.assertEqual(state["status"], original["status"])
                self.assertEqual(state["attempts"], original["attempts"])

    def test_checkpoint_metadata_corruption_is_rejected(self):
        for field, invalid in (
            ("total_chunks", True),
            ("results", []),
            ("failures", {0: None}),
            ("attempts", {0: -1}),
            ("attempts", {0: True}),
            ("attempts", {"0": 1}),
        ):
            with self.subTest(field=field, value=invalid):
                state = self.state(1)
                state[field] = invalid
                with self.assertRaises(ValueError):
                    validate_analysis_state(self.pairs(1), state)

    def test_nonfinite_and_invalid_generated_metrics_are_not_saved_as_success(self):
        for metrics in (
            {"rouge_l": float("nan")},
            {"rouge_l": float("inf")},
            {"rouge_l": -float("inf")},
            {"rouge_l": "0.5"},
            {"rouge_l": None},
            {"rouge_l": True},
            {"rouge_l": 1j},
            {"rouge_l": 10 ** 1000},
            {0: 0.5},
        ):
            with self.subTest(metrics=metrics):
                state = self.state(1)
                run_chunk_analysis(
                    self.pairs(1), lambda upper, lower: ("generated", metrics), state,
                )
                self.assertEqual(state["status"], "incomplete")
                self.assertEqual(state["results"], {})
                self.assertIn("similarity metrics", state["error"])

    def test_retry_hints_are_honored_bounded_and_published_for_the_ui(self):
        for message, delay in (
            ("Error: 429 Retry-After: 25", 25.0),
            ("Error: 503 {'Retry-After': '35.5'}", 35.5),
            ('Error: 429 {"retryDelay": "12.25s"}', 12.25),
            ('Error: 429 {"retryDelay": {"seconds": "19"}}', 19.0),
            ("Error: 429 retry_delay { seconds: 13 }", 13.0),
            ("Error: 429 Please retry in 4.5s.", 4.5),
            ("Error: 503 Retry-After: 90", 60.0),
        ):
            with self.subTest(message=message):
                state = self.state(1)
                snapshots = []
                calls = []
                sleeps = []

                def analyze(upper, lower):
                    calls.append(upper)
                    if len(calls) == 1:
                        return message, None
                    self.assertEqual(state["retry_in_seconds"], 0.0)
                    self.assertIsNone(state["retry_attempt"])
                    return self.success(upper, lower)

                run_chunk_analysis(
                    self.pairs(1), analyze, state,
                    sleep=sleeps.append, rng=lambda: 0.0,
                    on_update=lambda update: snapshots.append(copy.deepcopy(update)),
                )

                self.assertEqual(sleeps, [delay])
                retry_snapshot = next(update for update in snapshots if update["retry_in_seconds"])
                self.assertEqual(retry_snapshot["retry_in_seconds"], delay)
                self.assertEqual(retry_snapshot["retry_attempt"], 2)
                self.assertEqual(retry_snapshot["current_attempt"], 1)
                self.assertEqual(state["status"], "complete")
                self.assertEqual(state["retry_in_seconds"], 0.0)
                self.assertIsNone(state["retry_attempt"])

    def test_retry_after_http_date_is_supported(self):
        now = datetime(2026, 10, 7, 12, 0, 0, tzinfo=timezone.utc)
        retry_date = format_datetime(now + timedelta(seconds=30), usegmt=True)
        state = self.state(1)
        sleeps = []
        calls = []

        def analyze(upper, lower):
            calls.append(upper)
            if len(calls) == 1:
                return f"Error: 503 Retry-After: {retry_date}", None
            return self.success(upper, lower)

        with patch("src.document_analysis.time.time", return_value=now.timestamp()):
            run_chunk_analysis(
                self.pairs(1), analyze, state,
                sleep=sleeps.append, rng=lambda: 0.0,
            )
        self.assertEqual(sleeps, [30.0])
        self.assertEqual(state["status"], "complete")

    def test_exponential_retry_jitter_is_deterministic_and_capped_at_60_seconds(self):
        state = self.state(1)
        sleeps = []
        run_chunk_analysis(
            self.pairs(1), lambda upper, lower: ("Error: request timed out", None), state,
            max_attempts=9, sleep=sleeps.append, rng=lambda: 1.0,
        )
        self.assertEqual(sleeps, [1.25, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 60.0])
        self.assertEqual(state["attempts"][0], 9)
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(state["retry_in_seconds"], 0.0)
        self.assertIsNone(state["retry_attempt"])

    def test_empty_timeout_and_connection_exception_messages_are_retried(self):
        for exception in (TimeoutError, ConnectionError, ConnectionResetError, ConnectionAbortedError):
            with self.subTest(exception=exception):
                state = self.state(1)
                calls = []
                sleeps = []

                def analyze(upper, lower):
                    calls.append(upper)
                    if len(calls) == 1:
                        raise exception()
                    return self.success(upper, lower)

                run_chunk_analysis(
                    self.pairs(1), analyze, state,
                    sleep=sleeps.append, rng=lambda: 0.0,
                )
                self.assertEqual(len(calls), 2)
                self.assertEqual(sleeps, [1.0])
                self.assertEqual(state["status"], "complete")

    def test_active_chunk_and_attempt_are_checkpointed_before_the_call(self):
        state = self.state(1)
        snapshots = []

        def analyze(upper, lower):
            latest = snapshots[-1]
            self.assertEqual(latest["current_chunk"], 1)
            self.assertEqual(latest["current_attempt"], 1)
            self.assertEqual(latest["attempts"], {0: 1})
            raise RerunSignal()

        with self.assertRaises(RerunSignal):
            run_chunk_analysis(
                self.pairs(1), analyze, state,
                on_update=lambda update: snapshots.append(copy.deepcopy(update)),
            )
        self.assertEqual(snapshots[-1]["attempts"], {0: 1})
        self.assertEqual(state["results"], {})

    def test_cancellation_before_first_chunk_makes_no_calls(self):
        state = self.state(2)
        run_chunk_analysis(
            self.pairs(2), lambda upper, lower: self.fail("Cancelled runs must not make calls."),
            state, should_stop=lambda: True,
        )
        self.assertEqual(state["status"], "incomplete")
        self.assertIn("stopped by request", state["error"])
        self.assertEqual(state["attempts"], {})

    def test_cancellation_between_chunks_preserves_successes_for_resume(self):
        pairs = self.pairs(3)
        state = self.state(3)
        calls = []

        def analyze(upper, lower):
            calls.append(upper)
            return self.success(upper, lower)

        run_chunk_analysis(pairs, analyze, state, should_stop=lambda: bool(state["results"]))
        self.assertEqual(calls, ["prefix-0"])
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(set(state["results"]), {0})
        resumed_calls = []

        def resume(upper, lower):
            resumed_calls.append(upper)
            return self.success(upper, lower)

        run_chunk_analysis(pairs, resume, state)
        self.assertEqual(resumed_calls, ["prefix-1", "prefix-2"])
        self.assertEqual(state["status"], "complete")

    def test_cancellation_during_backoff_is_responsive_and_stops_before_retry(self):
        state = self.state(1)
        stopped = False
        calls = []
        sleeps = []

        def analyze(upper, lower):
            calls.append(upper)
            return "Error: 429 Retry-After: 60", None

        def sleep(delay):
            nonlocal stopped
            sleeps.append(delay)
            stopped = True

        run_chunk_analysis(
            self.pairs(1), analyze, state, sleep=sleep, rng=lambda: 0.0,
            should_stop=lambda: stopped,
        )
        self.assertEqual(calls, ["prefix-0"])
        self.assertEqual(sleeps, [1.0])
        self.assertEqual(state["attempts"], {0: 1})
        self.assertEqual(state["status"], "incomplete")
        self.assertIn("stopped by request", state["error"])
        self.assertEqual(state["retry_in_seconds"], 0.0)
        self.assertIsNone(state["retry_attempt"])

    def test_mismatched_chunk_count_and_invalid_attempt_limit_reject_before_calls(self):
        analyze = lambda upper, lower: self.fail("Invalid runs must not make API calls.")
        with self.assertRaisesRegex(ValueError, "chunk count"):
            run_chunk_analysis(self.pairs(2), analyze, self.state(1))
        with self.assertRaisesRegex(ValueError, "at least one"):
            run_chunk_analysis(self.pairs(1), analyze, self.state(1), max_attempts=0)


if __name__ == "__main__":
    unittest.main()



