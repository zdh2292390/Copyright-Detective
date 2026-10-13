"""Protect terminal job memory, snapshots, and cancellation cleanup."""

from concurrent.futures import Future, ThreadPoolExecutor
import gc
from threading import Event
import time
import unittest
from unittest.mock import Mock, patch
import weakref

from src import background_jobs as jobs
from src.run_cancellation import cancellable_call


class FakeJournal:
    def __init__(self):
        self.payload = bytearray(1024 * 1024)
        self.finished = None

    def finish(self, success):
        self.finished = success


class ImmediateExecutor:
    def submit(self, invoke):
        future = Future()
        invoke()
        future.set_result(None)
        return future


class BackgroundInternalRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.gates = []
        self.journal_refs = []

        def prepare(key, label):
            journal = FakeJournal()
            self.journal_refs.append(weakref.ref(journal))
            return journal

        self.patches = [
            patch.object(jobs, "_JOBS", {}),
            patch.object(jobs, "_NEXT_PRUNE_AT", 0.0),
            patch.object(jobs, "_EXECUTOR", self.executor),
            patch.object(jobs, "_origin", return_value=("owner", "browser")),
            patch("src.resumable_analysis.prepare_background_journal", side_effect=prepare),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for gate in self.gates:
            gate.set()
        self.executor.shutdown(wait=True)
        for item in reversed(self.patches):
            item.stop()

    def gate(self):
        gate = Event()
        self.gates.append(gate)
        return gate

    def wait(self, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if predicate():
                return
            Event().wait(0.01)
        self.fail("Background work did not reach the expected state.")

    def wait_terminal(self, key, status):
        self.wait(lambda: jobs.get_background_job(key)["status"] == status
                  and jobs._JOBS[key]["_future"] is None)

    def test_completed_journal_is_collected_while_result_remains_available(self):
        jobs.submit_background_job("completed", "Original label", lambda report: {"responses": ["saved"]})
        self.wait_terminal("completed", "completed")
        self.executor.shutdown(wait=True)
        gc.collect()
        self.assertIsNone(self.journal_refs[0]())
        self.assertIsNone(jobs._JOBS["completed"]["_cancel_token"])
        state = jobs.get_background_job("completed")
        self.assertEqual(state["label"], "Original label")
        self.assertEqual(state["result"], {"responses": ["saved"]})
        state["result"]["responses"].append("caller mutation")
        self.assertEqual(jobs.get_background_job("completed")["result"], {"responses": ["saved"]})
        self.assertFalse(any(key.startswith("_") for key in state))

    def test_immediate_completion_does_not_retain_future_registered_after_finish(self):
        with patch.object(jobs, "_EXECUTOR", ImmediateExecutor()):
            jobs.submit_background_job("instant", "Instant", lambda report: "saved")
        gc.collect()
        self.assertIsNone(jobs._JOBS["instant"]["_future"])
        self.assertIsNone(self.journal_refs[0]())
        self.assertEqual(jobs.get_background_job("instant")["result"], "saved")

    def test_failed_job_releases_journal_and_private_handles(self):
        def fail(report):
            raise RuntimeError("expected failure")

        jobs.submit_background_job("failed", "Failed", fail)
        self.wait_terminal("failed", "failed")
        self.executor.shutdown(wait=True)
        gc.collect()
        self.assertIsNone(self.journal_refs[0]())
        self.assertEqual(jobs.get_background_job("failed")["error"], "expected failure")

    def test_queued_stop_cancels_without_runner_or_journal_retention(self):
        entered, release = Event(), self.gate()

        def blocker():
            entered.set()
            release.wait(5)

        self.executor.submit(blocker)
        self.assertTrue(entered.wait(1))
        runner = Mock()
        jobs.submit_background_job("queued", "Queued", runner)
        self.assertTrue(jobs.force_stop_background_job("queued"))
        self.wait_terminal("queued", "stopped")
        runner.assert_not_called()
        gc.collect()
        self.assertIsNone(self.journal_refs[0]())
        self.assertFalse(jobs.background_job_running("queued"))

    def test_cancelled_queue_releases_runner_input_closure_before_queue_drains(self):
        entered, release = Event(), self.gate()

        def blocker():
            entered.set()
            release.wait(5)

        class Runner:
            def __init__(self):
                self.input_payload = bytearray(1024 * 1024)

            def __call__(self, report):
                raise AssertionError("cancelled runner must not execute")

        self.executor.submit(blocker)
        self.assertTrue(entered.wait(1))
        runner = Runner()
        reference = weakref.ref(runner)
        jobs.submit_background_job("runner-input", "Original label", runner)
        del runner
        self.assertTrue(jobs.force_stop_background_job("runner-input"))
        self.wait_terminal("runner-input", "stopped")
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(release.is_set())

    def test_stopping_preserves_journal_and_lock_until_accepted_request_drains(self):
        entered, release = Event(), self.gate()

        def request():
            entered.set()
            release.wait(5)
            return "late result"

        jobs.submit_background_job("stopping", "Stopping", lambda report: cancellable_call(request))
        self.assertTrue(entered.wait(1))
        self.assertTrue(jobs.force_stop_background_job("stopping"))
        self.wait(lambda: jobs._JOBS["stopping"]["_future"].done())
        gc.collect()
        self.assertIsNotNone(self.journal_refs[0]())
        self.assertIsNotNone(jobs._JOBS["stopping"]["_cancel_token"])
        self.assertFalse(jobs.forget_background_job("stopping"))
        self.assertFalse(jobs.submit_background_job("stopping", "Duplicate", Mock()))
        release.set()
        self.wait_terminal("stopping", "stopped")
        self.executor.shutdown(wait=True)
        gc.collect()
        self.assertIsNone(self.journal_refs[0]())
        self.assertIsNone(jobs._JOBS["stopping"]["_cancel_token"])

    def test_cleanup_thread_failure_still_retains_lock_until_request_drains(self):
        entered, release, attempted = Event(), self.gate(), Event()

        def request():
            entered.set()
            release.wait(5)
            return "late result"

        def fail_start():
            attempted.set()
            raise RuntimeError("no thread resources")

        cleanup = Mock()
        cleanup.start.side_effect = fail_start
        with patch.object(jobs, "Thread", return_value=cleanup):
            jobs.submit_background_job("fallback", "Fallback", lambda report: cancellable_call(request))
            self.assertTrue(entered.wait(1))
            self.assertTrue(jobs.force_stop_background_job("fallback"))
            self.assertTrue(attempted.wait(1))
            self.assertTrue(jobs.background_job_running("fallback"))
            self.assertFalse(jobs._JOBS["fallback"]["_future"].done())
            self.assertIsNotNone(self.journal_refs[0]())
            release.set()
            self.wait_terminal("fallback", "stopped")
        self.assertIsNone(jobs._JOBS["fallback"]["_cancel_token"])

    @staticmethod
    def record(status, finished, accessed=None):
        return {"status": status, "label": "Original", "result": "saved",
                "_finished_monotonic": finished,
                "_last_accessed_monotonic": finished if accessed is None else accessed}

    def test_pruning_removes_only_abandoned_terminal_records(self):
        now = jobs._TERMINAL_RETENTION_SECONDS * 2
        jobs._JOBS.update({
            "old-completed": self.record("completed", 0),
            "old-stopped": self.record("stopped", 0),
            "old-failed": self.record("failed", 0),
            "fresh-undelivered": self.record("completed", now - 1),
            "recently-viewed": self.record("completed", 0, now - 1),
            "queued": self.record("queued", 0),
            "running": self.record("running", 0),
            "stopping": self.record("stopping", 0),
            "unknown-age": {"status": "completed", "result": "legacy result"},
        })
        with patch.object(jobs, "monotonic", return_value=now), jobs._LOCK:
            jobs._prune_finished_jobs_locked()
        self.assertEqual(set(jobs._JOBS), {"fresh-undelivered", "recently-viewed", "queued",
                                          "running", "stopping", "unknown-age"})

    def test_direct_result_read_preserves_current_browser_evidence(self):
        now = jobs._TERMINAL_RETENTION_SECONDS * 2
        jobs._JOBS.update({"current": self.record("completed", 0),
                           "abandoned": self.record("completed", 0)})
        with patch.object(jobs, "monotonic", return_value=now):
            state = jobs.get_background_job("current")
        self.assertEqual(state["result"], "saved")
        self.assertIn("current", jobs._JOBS)
        self.assertNotIn("abandoned", jobs._JOBS)

    def test_prune_scans_are_throttled_until_next_interval(self):
        now = jobs._TERMINAL_RETENTION_SECONDS * 2
        with patch.object(jobs, "monotonic", return_value=now), jobs._LOCK:
            jobs._prune_finished_jobs_locked()
        jobs._JOBS["old"] = self.record("completed", 0)
        with patch.object(jobs, "monotonic", return_value=now + 1), jobs._LOCK:
            jobs._prune_finished_jobs_locked()
        self.assertIn("old", jobs._JOBS)
        with patch.object(jobs, "monotonic", return_value=now + jobs._PRUNE_INTERVAL_SECONDS), jobs._LOCK:
            jobs._prune_finished_jobs_locked()
        self.assertNotIn("old", jobs._JOBS)


if __name__ == "__main__":
    unittest.main()
