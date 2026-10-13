"""Cancellation cleans response handoffs and rolls back startup resources."""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore, Event
import unittest
from unittest.mock import Mock, patch

from src import api_concurrency as concurrency
from src import run_cancellation as cancellation


class Response:
    def __init__(self, fail_close=False):
        self.close_calls = 0
        self.fail_close = fail_close

    def close(self):
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("response close failed")


class CancellationInternalRobustnessTests(unittest.TestCase):
    def assert_startup_rollback(self, target):
        token = cancellation.CancellationToken()
        semaphore = BoundedSemaphore(1)
        with patch.object(concurrency, "_API_SEMAPHORE", semaphore), cancellation.cancellation_scope(token):
            with concurrency.limit_api_concurrency():
                with patch.object(cancellation, target, side_effect=RuntimeError("setup failed")):
                    with self.assertRaisesRegex(RuntimeError, "setup failed"):
                        cancellation.cancellable_call(lambda: "not invoked")
                self.assertEqual(token.pending_count, 0)
                self.assertTrue(token.wait_for_idle(timeout=0))
            self.assertTrue(semaphore.acquire(blocking=False))
            self.assertFalse(semaphore.acquire(blocking=False))
            semaphore.release()

    def test_future_construction_failure_rolls_back_pending_count_and_api_slot(self):
        self.assert_startup_rollback("Future")

    def test_context_copy_failure_rolls_back_pending_count_and_api_slot(self):
        self.assert_startup_rollback("copy_context")

    def test_thread_construction_failure_rolls_back_pending_count_and_api_slot(self):
        self.assert_startup_rollback("Thread")

    def test_closer_lock_failure_does_not_increment_pending_count(self):
        token = cancellation.CancellationToken()
        with cancellation.cancellation_scope(token), patch.object(cancellation, "Lock", side_effect=RuntimeError("lock failed")):
            with self.assertRaisesRegex(RuntimeError, "lock failed"):
                cancellation.cancellable_call(lambda: "not invoked")
        self.assertEqual(token.pending_count, 0)

    def test_worker_context_binding_failure_returns_error_and_releases_resources(self):
        token = cancellation.CancellationToken()
        semaphore = BoundedSemaphore(1)
        binding = Mock()
        binding.get.return_value = False
        binding.set.side_effect = RuntimeError("worker binding failed")
        with patch.object(concurrency, "_API_SEMAPHORE", semaphore), cancellation.cancellation_scope(token):
            with concurrency.limit_api_concurrency(), patch.object(cancellation, "_IN_CALL", binding):
                with self.assertRaisesRegex(RuntimeError, "worker binding failed"):
                    cancellation.cancellable_call(lambda: "not invoked")
                self.assertTrue(token.wait_for_idle(timeout=1))
                self.assertEqual(token.pending_count, 0)
            self.assertTrue(semaphore.acquire(blocking=False))
            self.assertFalse(semaphore.acquire(blocking=False))
            semaphore.release()
        binding.reset.assert_not_called()

    def test_response_is_closed_if_cancellation_lands_during_future_publication(self):
        token = cancellation.CancellationToken()
        response = Response()

        class PublishingFuture(Future):
            def set_result(self, value):
                token.cancel()
                return super().set_result(value)

        with cancellation.cancellation_scope(token), patch.object(cancellation, "Future", PublishingFuture):
            with self.assertRaises(cancellation.AnalysisCancelled):
                cancellation.cancellable_call(lambda: response)
        self.assertTrue(token.wait_for_idle(timeout=1))
        self.assertEqual(response.close_calls, 1)

    def test_response_is_closed_if_cancellation_lands_after_caller_receives_future_value(self):
        token = cancellation.CancellationToken()
        response = Response()

        class ReceivingFuture(Future):
            def result(self, timeout=None):
                value = super().result(timeout)
                token.cancel()
                return value

        with cancellation.cancellation_scope(token), patch.object(cancellation, "Future", ReceivingFuture):
            with self.assertRaises(cancellation.AnalysisCancelled):
                cancellation.cancellable_call(lambda: response)
        self.assertTrue(token.wait_for_idle(timeout=1))
        self.assertEqual(response.close_calls, 1)

    def test_cancelled_pending_response_closes_once_when_request_returns(self):
        token = cancellation.CancellationToken()
        response = Response()
        entered, release = Event(), Event()

        def request():
            entered.set()
            release.wait(3)
            return response

        def caller():
            with cancellation.cancellation_scope(token):
                return cancellation.cancellable_call(request)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(caller)
            try:
                self.assertTrue(entered.wait(1))
                token.cancel()
                with self.assertRaises(cancellation.AnalysisCancelled):
                    future.result(timeout=1)
                self.assertGreater(token.pending_count, 0)
            finally:
                release.set()
        self.assertTrue(token.wait_for_idle(timeout=1))
        self.assertEqual(response.close_calls, 1)

    def test_normal_handoff_keeps_response_open_for_original_caller(self):
        token = cancellation.CancellationToken()
        response = Response()
        with cancellation.cancellation_scope(token):
            self.assertIs(cancellation.cancellable_call(lambda: response), response)
        token.cancel()
        self.assertTrue(token.wait_for_idle(timeout=1))
        self.assertEqual(response.close_calls, 0)

    def test_failed_close_preserves_stop_signal_and_releases_pending_count(self):
        token = cancellation.CancellationToken()
        response = Response(fail_close=True)

        class PublishingFuture(Future):
            def set_result(self, value):
                token.cancel()
                return super().set_result(value)

        with cancellation.cancellation_scope(token), patch.object(cancellation, "Future", PublishingFuture):
            with self.assertRaises(cancellation.AnalysisCancelled):
                cancellation.cancellable_call(lambda: response)
        self.assertTrue(token.wait_for_idle(timeout=1))
        self.assertEqual(response.close_calls, 1)

    def test_provider_timeout_is_not_mistaken_for_waiter_poll_timeout(self):
        token = cancellation.CancellationToken()

        def request():
            raise TimeoutError("provider timed out")

        with cancellation.cancellation_scope(token):
            with self.assertRaisesRegex(TimeoutError, "provider timed out"):
                cancellation.cancellable_call(request)
        self.assertTrue(token.wait_for_idle(timeout=1))

    def test_provider_exception_preserves_type_and_does_not_leak_pending_count(self):
        token = cancellation.CancellationToken()

        def request():
            raise ValueError("provider rejected request")

        with cancellation.cancellation_scope(token):
            with self.assertRaisesRegex(ValueError, "provider rejected request"):
                cancellation.cancellable_call(request)
        self.assertTrue(token.wait_for_idle(timeout=1))


if __name__ == "__main__":
    unittest.main()
