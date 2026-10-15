"""Deterministic cancellation/resource lifetime regressions; no provider SDKs."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from threading import BoundedSemaphore, Event
import unittest
from unittest.mock import Mock, patch

from src import api_concurrency as concurrency
from src import run_cancellation as cancellation


class CancellationRuntime(unittest.TestCase):
    def call(self, token, invoke):
        with cancellation.cancellation_scope(token):
            return cancellation.cancellable_call(invoke)

    def gated_cleanup(self, invoke, expected_error=None):
        token = cancellation.CancellationToken()
        entered, release = Event(), Event()
        def cleanup():
            entered.set()
            release.wait(2)
        with patch.object(concurrency, 'retain_api_slot', return_value=cleanup):
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(self.call, token, invoke)
                try:
                    self.assertTrue(entered.wait(1))
                    self.assertEqual(token.pending_count, 1)
                    with self.assertRaises(FutureTimeout):
                        pending.result(timeout=0.05)
                finally:
                    release.set()
                if expected_error:
                    with self.assertRaises(expected_error):
                        pending.result(timeout=2)
                else:
                    self.assertEqual(pending.result(timeout=2), 'response')
        self.assertEqual(token.pending_count, 0)

    def test_success_is_not_published_before_request_cleanup(self):
        self.gated_cleanup(lambda: 'response')

    def test_failure_is_not_published_before_request_cleanup(self):
        def fail():
            raise ValueError('provider failed')
        self.gated_cleanup(fail, ValueError)

    def test_cleanup_failure_reaches_waiter_and_does_not_leak_pending_count(self):
        token = cancellation.CancellationToken()
        response = Mock()
        def fail_cleanup():
            raise RuntimeError('cleanup failed')
        with patch.object(concurrency, 'retain_api_slot', return_value=fail_cleanup):
            with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                self.call(token, lambda: response)
        self.assertEqual(token.pending_count, 0)
        response.close.assert_called_once()

    def assert_cleanup_error_is_published(self, response):
        token = cancellation.CancellationToken()
        def fail_cleanup():
            raise RuntimeError('cleanup failed')
        with patch.object(concurrency, 'retain_api_slot', return_value=fail_cleanup):
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(self.call, token, lambda: response)
                try:
                    with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                        pending.result(timeout=1)
                finally:
                    # Keep a failed regression bounded if the result is lost.
                    token.cancel()
        self.assertEqual(token.pending_count, 0)

    def test_response_close_lookup_cannot_strand_cleanup_failure(self):
        class InvalidResponse:
            @property
            def close(self):
                raise ValueError('close lookup failed')
        self.assert_cleanup_error_is_published(InvalidResponse())

    def test_response_close_base_exception_cannot_strand_cleanup_failure(self):
        response = Mock()
        response.close.side_effect = cancellation.AnalysisCancelled('close cancelled')
        self.assert_cleanup_error_is_published(response)
        response.close.assert_called_once()

    def test_thread_start_failure_rolls_back_retained_resources(self):
        token = cancellation.CancellationToken()
        release = Mock()
        with patch.object(concurrency, 'retain_api_slot', return_value=release):
            with patch.object(cancellation, 'Thread', side_effect=RuntimeError('cannot start')):
                with self.assertRaisesRegex(RuntimeError, 'cannot start'):
                    self.call(token, lambda: 'never called')
        self.assertEqual(token.pending_count, 0)
        release.assert_called_once()

    def test_cancelled_waiter_closes_late_response_once(self):
        token = cancellation.CancellationToken()
        started, release, closed = Event(), Event(), Event()
        response = Mock()
        response.close.side_effect = closed.set
        def invoke():
            started.set()
            release.wait(2)
            return response
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self.call, token, invoke)
            try:
                self.assertTrue(started.wait(1))
                token.cancel()
                with self.assertRaises(cancellation.AnalysisCancelled):
                    pending.result(timeout=1)
                self.assertEqual(token.pending_count, 1)
            finally:
                release.set()
            self.assertTrue(token.wait_for_idle(timeout=2))
            self.assertTrue(closed.wait(1))
        response.close.assert_called_once()

    def test_cancelled_request_holds_concurrency_permit_until_network_exits(self):
        token = cancellation.CancellationToken()
        started, release = Event(), Event()
        semaphore = BoundedSemaphore(1)
        def invoke():
            started.set()
            release.wait(2)
            return 'late response'
        def run():
            with cancellation.cancellation_scope(token):
                with concurrency.limit_api_concurrency(timeout=1):
                    return cancellation.cancellable_call(invoke)
        with patch.object(concurrency, '_API_SEMAPHORE', semaphore):
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(run)
                try:
                    self.assertTrue(started.wait(1))
                    token.cancel()
                    with self.assertRaises(cancellation.AnalysisCancelled):
                        pending.result(timeout=1)
                    self.assertFalse(semaphore.acquire(blocking=False))
                finally:
                    release.set()
                self.assertTrue(token.wait_for_idle(timeout=2))
            self.assertTrue(semaphore.acquire(blocking=False))
            semaphore.release()

    def test_nested_calls_share_one_pending_request(self):
        token = cancellation.CancellationToken()
        counts = []
        def invoke():
            counts.append(token.pending_count)
            return cancellation.cancellable_call(lambda: 'response')
        self.assertEqual(self.call(token, invoke), 'response')
        self.assertEqual(counts, [1])
        self.assertEqual(token.pending_count, 0)

    def test_provider_timeout_is_not_confused_with_wait_polling_timeout(self):
        token = cancellation.CancellationToken()
        def invoke():
            raise TimeoutError('provider timed out')
        with self.assertRaisesRegex(TimeoutError, 'provider timed out'):
            self.call(token, invoke)
        self.assertEqual(token.pending_count, 0)


if __name__ == '__main__':
    unittest.main()
