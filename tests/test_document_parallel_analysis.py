
"""Parallel document coordinator regressions without provider calls."""

import copy
import threading
import time
import unittest
from collections import Counter
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from unittest.mock import Mock, patch

from src.document_analysis import analysis_results, new_analysis_state, run_chunk_analysis


def pairs(count):
    return [(f"prefix-{index}", f"reference-{index}") for index in range(count)]


def success(upper, lower):
    return f"generated for {upper}", {"rouge_l": 0.0}


def index_of(upper):
    return int(upper.rsplit("-", 1)[1])


class TrackingExecutor(Executor):
    def __init__(self, workers=3):
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.lock = threading.Lock()
        self.outstanding = 0
        self.maximum_outstanding = 0
        self.submitted = 0
        self.shutdown_calls = 0

    def submit(self, fn, /, *args, **kwargs):
        with self.lock:
            self.outstanding += 1
            self.maximum_outstanding = max(self.maximum_outstanding, self.outstanding)
            self.submitted += 1
        future = self.pool.submit(fn, *args, **kwargs)
        def finished(_):
            with self.lock:
                self.outstanding -= 1
        future.add_done_callback(finished)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.shutdown_calls += 1
        self.pool.shutdown(wait=wait, cancel_futures=cancel_futures)


class ManualExecutor(Executor):
    def __init__(self):
        self.tasks = []
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.shutdown_calls = 0

    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        with self.lock:
            self.tasks.append((future, fn, args, kwargs))
            if len(self.tasks) == 3:
                self.ready.set()
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.shutdown_calls += 1


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.lock = threading.Lock()

    def monotonic(self):
        with self.lock:
            return self.now

    def sleep(self, delay):
        with self.lock:
            self.sleeps.append(delay)
            self.now += delay


class ParallelDocumentAnalysisTests(unittest.TestCase):
    @staticmethod
    def state(count):
        return new_analysis_state("test-fingerprint", {"chunk_size": 200}, count)

    def test_concurrent_calls_are_bounded_and_only_coordinator_publishes(self):
        state = self.state(12)
        lock = threading.Lock()
        first_window = threading.Barrier(3, timeout=3)
        active = 0
        peak = 0
        workers = set()
        publisher = threading.get_ident()
        updates = []

        def analyze(upper, lower):
            nonlocal active, peak
            index = index_of(upper)
            with lock:
                active += 1
                peak = max(peak, active)
                workers.add(threading.get_ident())
            if index < 3:
                first_window.wait()
                time.sleep((2 - index) * 0.015)
            answer = success(upper, lower)
            with lock:
                active -= 1
            return answer

        def checkpoint(updated):
            self.assertEqual(threading.get_ident(), publisher)
            updates.append(copy.deepcopy(updated))

        returned = run_chunk_analysis(pairs(12), analyze, state, max_concurrency=3, on_update=checkpoint)
        self.assertIs(returned, state)
        self.assertEqual(peak, 3)
        self.assertNotIn(publisher, workers)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["active_chunks"], [])
        self.assertEqual(state["concurrency_limit"], 3)
        self.assertEqual([row[0] for row in analysis_results(state)], [upper for upper, _ in pairs(12)])
        prior = set()
        for update in updates:
            current = set(update["results"])
            changed = current - prior
            self.assertLessEqual(len(changed), 1)
            if changed:
                self.assertEqual(changed, {update["current_chunk"] - 1})
            prior = current
            self.assertLessEqual(len(update["active_chunks"]), 3)

    def test_1974_chunk_resume_skips_104_completed_and_does_not_fill_executor_queue(self):
        state = self.state(1974)
        for index, (upper, lower) in enumerate(pairs(104)):
            generated, metrics = success(upper, lower)
            state["results"][index] = (upper, lower, generated, metrics)
        calls = []
        lock = threading.Lock()
        executor = TrackingExecutor()

        def analyze(upper, lower):
            with lock:
                calls.append(index_of(upper))
            return success(upper, lower)

        try:
            run_chunk_analysis(pairs(1974), analyze, state, max_concurrency=3, executor=executor)
            self.assertEqual(executor.shutdown_calls, 0)
            self.assertLessEqual(executor.maximum_outstanding, 3)
            self.assertEqual(executor.submitted, 1870)
            self.assertEqual(sorted(calls), list(range(104, 1974)))
            self.assertEqual(state["status"], "complete")
            self.assertEqual(len(analysis_results(state)), 1974)
            self.assertEqual(executor.submit(lambda: "still usable").result(timeout=2), "still usable")
        finally:
            executor.shutdown()

    def test_stop_retains_successes_from_running_window_and_starts_no_more(self):
        state = self.state(20)
        stop = threading.Event()
        started = threading.Barrier(3, timeout=3)
        calls = []
        lock = threading.Lock()

        def analyze(upper, lower):
            with lock:
                calls.append(index_of(upper))
            started.wait()
            stop.set()
            return success(upper, lower)

        run_chunk_analysis(pairs(20), analyze, state, max_concurrency=3, should_stop=stop.is_set)
        self.assertEqual(sorted(calls), [0, 1, 2])
        self.assertEqual(set(state["results"]), {0, 1, 2})
        self.assertEqual(state["status"], "incomplete")
        self.assertIn("stopped by request", state["error"])
        self.assertEqual(state["active_chunks"], [])

    def test_authentication_failure_drains_other_running_success_and_pauses(self):
        state = self.state(20)
        started = threading.Barrier(2, timeout=3)
        release = threading.Event()
        calls = []
        lock = threading.Lock()

        def analyze(upper, lower):
            index = index_of(upper)
            with lock:
                calls.append(index)
            started.wait()
            if index == 0:
                return "Error: 401 invalid API key", None
            if not release.wait(3):
                raise AssertionError("coordinator did not publish the permanent error")
            return success(upper, lower)

        def checkpoint(updated):
            if 0 in updated["failures"]:
                release.set()

        run_chunk_analysis(pairs(20), analyze, state, max_concurrency=2, on_update=checkpoint)
        self.assertEqual(sorted(calls), [0, 1])
        self.assertEqual(set(state["results"]), {1})
        self.assertIn(0, state["failures"])
        self.assertIn("401", state["error"])
        self.assertEqual(state["status"], "incomplete")

    def test_checkpoint_failure_drains_successes_then_reraises_without_new_calls(self):
        state = self.state(20)
        started = threading.Barrier(2, timeout=3)
        release = threading.Event()
        calls = []
        checkpoints_after_failure = []

        def analyze(upper, lower):
            index = index_of(upper)
            calls.append(index)
            started.wait()
            if index == 1 and not release.wait(3):
                raise AssertionError("failed checkpoint did not stop dispatch")
            return success(upper, lower)

        def checkpoint(updated):
            if release.is_set():
                checkpoints_after_failure.append(copy.deepcopy(updated))
            if 0 in updated["results"]:
                release.set()
                raise OSError("checkpoint disk full")

        with self.assertRaisesRegex(OSError, "disk full"):
            run_chunk_analysis(pairs(20), analyze, state, max_concurrency=2, on_update=checkpoint)
        self.assertEqual(sorted(calls), [0, 1])
        self.assertEqual(set(state["results"]), {0, 1})
        self.assertEqual(checkpoints_after_failure, [])
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(state["active_chunks"], [])

    def test_shared_executor_pending_calls_are_cancelled_after_internal_halt(self):
        state = self.state(20)
        executor = ManualExecutor()
        outbound = Mock(side_effect=lambda upper, lower: ("Error: 401 unauthorized", None))
        done = threading.Event()
        failures = []

        def run():
            try:
                run_chunk_analysis(pairs(20), outbound, state, max_concurrency=3, executor=executor)
            except BaseException as exc:
                failures.append(exc)
            finally:
                done.set()

        coordinator = threading.Thread(target=run)
        coordinator.start()
        try:
            self.assertTrue(executor.ready.wait(3))
            future, fn, args, kwargs = executor.tasks[0]
            self.assertTrue(future.set_running_or_notify_cancel())
            future.set_result(fn(*args, **kwargs))
            self.assertTrue(done.wait(3))
            self.assertEqual(failures, [])
            self.assertEqual(outbound.call_count, 1)
            self.assertEqual(executor.shutdown_calls, 0)
            self.assertTrue(all(task[0].cancelled() for task in executor.tasks[1:]))
            # Even an executor that begins an already queued callable late must
            # hit the internal guard before reaching the provider.
            for _, queued_fn, queued_args, queued_kwargs in executor.tasks[1:]:
                queued_fn(*queued_args, **queued_kwargs)
            self.assertEqual(outbound.call_count, 1)
            self.assertEqual(state["attempts"], {0: 1})
            self.assertEqual(state["active_chunks"], [])
        finally:
            for future, _, _, _ in executor.tasks:
                future.cancel()
            coordinator.join(timeout=3)
        self.assertFalse(coordinator.is_alive())

    def test_external_stop_guard_covers_queued_shared_executor_calls(self):
        state = self.state(20)
        executor = ManualExecutor()
        stop = threading.Event()
        outbound = Mock(side_effect=success)
        done = threading.Event()

        def run():
            try:
                run_chunk_analysis(pairs(20), outbound, state, max_concurrency=3,
                                   executor=executor, should_stop=stop.is_set)
            finally:
                done.set()

        coordinator = threading.Thread(target=run)
        coordinator.start()
        try:
            self.assertTrue(executor.ready.wait(3))
            stop.set()
            self.assertTrue(done.wait(3))
            for _, fn, args, kwargs in executor.tasks:
                fn(*args, **kwargs)
            outbound.assert_not_called()
            self.assertEqual(state["attempts"], {})
            self.assertEqual(state["status"], "incomplete")
        finally:
            stop.set()
            coordinator.join(timeout=3)
        self.assertFalse(coordinator.is_alive())

    def test_rate_limit_publishes_reduction_and_shared_sixty_second_cooldown_before_new_calls(self):
        state = self.state(8)
        clock = FakeClock()
        start = threading.Barrier(3, timeout=3)
        released = threading.Event()
        calls = Counter()
        timestamps = []
        updates = []
        lock = threading.Lock()

        def analyze(upper, lower):
            index = index_of(upper)
            with lock:
                calls[index] += 1
                attempt = calls[index]
                timestamps.append((index, attempt, clock.monotonic()))
            if index < 3 and attempt == 1:
                start.wait()
                if index == 0:
                    return "Error: 429 RESOURCE_EXHAUSTED Retry-After: 999", None
                if not released.wait(3):
                    raise AssertionError("rate limit was not published")
            return success(upper, lower)

        def checkpoint(updated):
            if 0 in updated["failures"] and not released.is_set():
                updates.append(copy.deepcopy(updated))
                released.set()

        with patch("src.document_analysis.time.monotonic", side_effect=clock.monotonic):
            run_chunk_analysis(pairs(8), analyze, state, max_concurrency=3,
                               sleep=clock.sleep, rng=lambda: 0.0, on_update=checkpoint)
        self.assertEqual(updates[0]["effective_concurrency"], 1)
        self.assertEqual(updates[0]["retry_in_seconds"], 60)
        self.assertEqual(clock.sleeps, [60])
        self.assertTrue(all(when >= 60 for index, attempt, when in timestamps if index >= 3 or attempt > 1))
        self.assertEqual(calls[0], 2)
        self.assertEqual(state["effective_concurrency"], 1)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["failures"], {})

    def test_rate_limit_cancels_queued_window_and_reschedules_without_spending_attempts(self):
        state = self.state(4)
        executor = ManualExecutor()
        clock = FakeClock()
        stop = threading.Event()
        sleeping = threading.Event()
        allow_cooldown = threading.Event()
        done = threading.Event()
        calls = Counter()
        timestamps = []
        updates = []
        failures = []

        def analyze(upper, lower):
            index = index_of(upper)
            calls[index] += 1
            timestamps.append((index, calls[index], clock.monotonic()))
            if index == 0 and calls[index] == 1:
                return "Error: 429 Retry-After: 5", None
            return success(upper, lower)

        def sleep(delay):
            sleeping.set()
            if not allow_cooldown.wait(3):
                raise AssertionError("test did not release provider cooldown")
            clock.sleep(delay)

        def run():
            try:
                run_chunk_analysis(
                    pairs(4), analyze, state, max_concurrency=3, executor=executor,
                    sleep=sleep, rng=lambda: 0.0, should_stop=stop.is_set,
                    on_update=lambda updated: updates.append(copy.deepcopy(updated)),
                )
            except BaseException as exc:
                failures.append(exc)
            finally:
                done.set()

        def task(number):
            deadline = time.perf_counter() + 3
            while time.perf_counter() < deadline:
                with executor.lock:
                    if len(executor.tasks) > number:
                        return executor.tasks[number]
                threading.Event().wait(0.002)
            self.fail(f"coordinator did not schedule task {number}")

        coordinator = threading.Thread(target=run)
        with patch("src.document_analysis.time.monotonic", side_effect=clock.monotonic):
            coordinator.start()
            try:
                self.assertTrue(executor.ready.wait(3))
                future, fn, args, kwargs = task(0)
                self.assertTrue(future.set_running_or_notify_cancel())
                future.set_result(fn(*args, **kwargs))
                self.assertTrue(sleeping.wait(3))
                self.assertTrue(task(1)[0].cancelled())
                self.assertTrue(task(2)[0].cancelled())
                self.assertEqual(len(executor.tasks), 3)
                self.assertEqual(calls, {0: 1})
                self.assertEqual(state["attempts"], {0: 1})
                rate_update = next(update for update in updates if 0 in update["failures"])
                self.assertEqual(rate_update["current_chunk"], 1)
                self.assertEqual(rate_update["effective_concurrency"], 1)
                self.assertEqual(rate_update["retry_in_seconds"], 5)
                allow_cooldown.set()
                for number in range(3, 7):
                    future, fn, args, kwargs = task(number)
                    self.assertTrue(future.set_running_or_notify_cancel())
                    future.set_result(fn(*args, **kwargs))
                self.assertTrue(done.wait(3))
                self.assertEqual(failures, [])
                self.assertEqual(calls, {0: 2, 1: 1, 2: 1, 3: 1})
                self.assertEqual(state["attempts"], dict(calls))
                self.assertEqual(state["status"], "complete")
                self.assertEqual(set(state["results"]), {0, 1, 2, 3})
                self.assertTrue(all(when >= 5 for _, attempt, when in timestamps[1:]))
                self.assertEqual(sum(clock.sleeps), 5)
                self.assertEqual(executor.shutdown_calls, 0)
            finally:
                stop.set()
                allow_cooldown.set()
                coordinator.join(timeout=3)
        self.assertFalse(coordinator.is_alive())

    def test_exhausted_rate_limit_retries_pause_and_never_analyze_rest_of_document(self):
        state = self.state(10)
        clock = FakeClock()
        calls = Counter()

        def analyze(upper, lower):
            index = index_of(upper)
            calls[index] += 1
            if index == 0:
                return "Error: 429 quota unavailable", None
            return success(upper, lower)

        with patch("src.document_analysis.time.monotonic", side_effect=clock.monotonic):
            run_chunk_analysis(pairs(10), analyze, state, max_concurrency=2, sleep=clock.sleep, rng=lambda: 0.0)
        self.assertEqual(calls[0], 3)
        self.assertLessEqual(set(calls), {0, 1})
        self.assertEqual(clock.sleeps, [1, 2])
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(state["retry_in_seconds"], 0)
        self.assertEqual(state["active_chunks"], [])

    def test_empty_timeout_exception_retries_and_isolated_empty_chunk_keeps_alignment(self):
        state = self.state(6)
        clock = FakeClock()
        calls = Counter()

        def analyze(upper, lower):
            index = index_of(upper)
            calls[index] += 1
            if index == 0 and calls[index] < 3:
                raise TimeoutError()
            if index == 3:
                return "", {"rouge_l": 0.0}
            return success(upper, lower)

        with patch("src.document_analysis.time.monotonic", side_effect=clock.monotonic):
            run_chunk_analysis(pairs(6), analyze, state, max_concurrency=2, sleep=clock.sleep, rng=lambda: 0.0)
        self.assertEqual(calls[0], 3)
        self.assertEqual(set(state["results"]), {0, 1, 2, 4, 5})
        self.assertEqual(set(state["failures"]), {3})
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual([row[0] for row in analysis_results(state)],
                         ["prefix-0", "prefix-1", "prefix-2", "prefix-4", "prefix-5"])

    def test_metrics_validation_exception_is_recorded_and_running_success_is_retained(self):
        class BrokenMetrics(dict):
            def items(self):
                raise ValueError("provider metric mapping is malformed")

        state = self.state(10)
        start = threading.Barrier(2, timeout=3)
        release = threading.Event()
        calls = []

        def analyze(upper, lower):
            index = index_of(upper)
            calls.append(index)
            start.wait()
            if index == 0:
                return "generated", BrokenMetrics(rouge_l=0.0)
            if not release.wait(3):
                raise AssertionError("metric error was not published")
            return success(upper, lower)

        def checkpoint(updated):
            if 0 in updated["failures"]:
                release.set()

        run_chunk_analysis(pairs(10), analyze, state, max_concurrency=2, on_update=checkpoint)
        self.assertEqual(sorted(calls), [0, 1])
        self.assertIn("ValueError", state["failures"][0])
        self.assertEqual(set(state["results"]), {1})
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(state["active_chunks"], [])

    def test_standalone_default_stays_serial_and_exposes_inflight_metadata(self):
        state = self.state(3)
        calls = []
        publisher = threading.get_ident()

        def analyze(upper, lower):
            calls.append(index_of(upper))
            self.assertEqual(threading.get_ident(), publisher)
            self.assertEqual(state["active_chunks"], [index_of(upper) + 1])
            self.assertEqual(state["effective_concurrency"], 1)
            return success(upper, lower)

        run_chunk_analysis(pairs(3), analyze, state)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(state["active_chunks"], [])
        self.assertEqual(state["status"], "complete")

    def test_stop_during_initial_publication_prevents_all_calls(self):
        state = self.state(5)
        stop = threading.Event()
        outbound = Mock(side_effect=success)
        run_chunk_analysis(pairs(5), outbound, state, max_concurrency=3,
                           on_update=lambda _: stop.set(), should_stop=stop.is_set)
        outbound.assert_not_called()
        self.assertEqual(state["active_chunks"], [])
        self.assertEqual(state["status"], "incomplete")

    def test_invalid_concurrency_fails_before_provider_calls(self):
        for value in (False, 0, -1, 9, 2.0, "3", None):
            with self.subTest(value=value):
                outbound = Mock()
                with self.assertRaisesRegex(ValueError, "max_concurrency"):
                    run_chunk_analysis(pairs(2), outbound, self.state(2), max_concurrency=value)
                outbound.assert_not_called()


if __name__ == "__main__":
    unittest.main()
