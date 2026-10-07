"""Real SQLite document job concurrency, cancellation, and recovery regressions."""

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Condition, Event
import time
import unittest

from src.document_analysis import analysis_fingerprint, analysis_results, new_analysis_state
from src.document_checkpoints import CheckpointError, DocumentCheckpointStore
from src.document_jobs import DocumentAnalysisJobs


def fixture(count=8, label="source.txt"):
    settings = {
        "filename": label, "model": "gemini-3.5-flash", "provider": "Google Gemini",
        "chunk_size": 200, "overlap": 50, "continuation_method": "Normal Continuation",
        "temperature": 0.7, "top_p": 0.9, "custom_template": None,
        "extra_prompt_instructions": "Continue the source", "base_url": None,
    }
    state = new_analysis_state(analysis_fingerprint(label, settings), settings, count)
    return state, [(f"prefix-{index}", f"target-{index}") for index in range(count)]


class RequestGate:
    """Hold actual worker calls independently, without timer-based provider fakes."""

    def __init__(self):
        self.condition = Condition()
        self.calls = []
        self.active = 0
        self.peak = 0
        self.releases = {}
        self.allow_all = False
        self.response = None

    @staticmethod
    def success(call):
        return f"generated-{call['index']}", {"rouge_l": call["index"] / 100.0}

    def analyze(self, settings, api_key, upper, lower):
        index = int(upper.rsplit("-", 1)[1])
        key = (settings["filename"], index)
        with self.condition:
            release = self.releases.setdefault(key, Event())
            if self.allow_all:
                release.set()
            call = {"filename": settings["filename"], "index": index, "api_key": api_key}
            self.calls.append(call)
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.condition.notify_all()
        try:
            if not release.wait(10):
                raise RuntimeError("test request gate timed out")
            response = self.response
            return response(call) if response else self.success(call)
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()

    def wait_for_calls(self, count, timeout=5):
        with self.condition:
            return self.condition.wait_for(lambda: len(self.calls) >= count, timeout)

    def release(self, index, label="source.txt"):
        with self.condition:
            self.releases.setdefault((label, index), Event()).set()

    def release_all(self):
        with self.condition:
            self.allow_all = True
            for event in self.releases.values():
                event.set()

    def snapshot(self):
        with self.condition:
            return [dict(call) for call in self.calls]


class FaultingProgressStore:
    """Keep SQLite readable while rejecting all writes containing a success."""

    def __init__(self, store):
        self.store = store
        self.broken = True
        self.failed = Event()

    def __getattr__(self, name):
        return getattr(self.store, name)

    def check(self, state):
        if self.broken and state["results"]:
            self.failed.set()
            raise CheckpointError("simulated checkpoint disk full")

    def save(self, token, state):
        self.check(state)
        return self.store.save(token, state)

    def save_progress(self, token, state, chunk_index=None):
        self.check(state)
        return self.store.save_progress(token, state, chunk_index)


class DocumentParallelJobsTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="document-parallel-jobs-")
        self.root = Path(self.temp.name)
        self.store = DocumentCheckpointStore(self.root)
        self.managers = []
        self.gates = []

    def tearDown(self):
        for gate in self.gates:
            gate.release_all()
        for manager in reversed(self.managers):
            manager.close()
        self.temp.cleanup()

    def gate(self):
        gate = RequestGate()
        self.gates.append(gate)
        return gate

    def manager(self, *, store=None, analyze_chunk=None, **kwargs):
        options = {"chunk_concurrency": 3, "parallel_threshold": 1, "max_chunk_workers": 8}
        options.update(kwargs)
        manager = DocumentAnalysisJobs(
            self.store if store is None else store,
            analyze_chunk=analyze_chunk or (lambda settings, key, upper, lower: (
                "generated-" + upper.rsplit("-", 1)[1], {"rouge_l": 0.0}
            )),
            **options,
        )
        self.managers.append(manager)
        return manager

    def wait_until(self, predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("The parallel job did not reach the expected state.")

    def finish(self, manager, token, *, owner_id=None):
        self.wait_until(lambda: not manager.is_running(token))
        return manager.get(token, owner_id=owner_id)

    def assert_ordered_results(self, state, count):
        results = analysis_results(state)
        self.assertEqual(len(results), count)
        self.assertEqual([result[0] for result in results], [f"prefix-{index}" for index in range(count)])
        self.assertEqual([result[2] for result in results], [f"generated-{index}" for index in range(count)])

    def test_three_requests_are_concurrent_and_out_of_order_results_are_durable_and_sorted(self):
        gate = self.gate()
        manager = self.manager(analyze_chunk=gate.analyze)
        state, pairs = fixture(6)
        token = manager.create(state, pairs)
        self.assertTrue(manager.submit(token, "runtime-key"))
        self.assertTrue(gate.wait_for_calls(3), "Serial execution cannot fill three gated calls.")
        self.assertEqual({call["index"] for call in gate.snapshot()}, {0, 1, 2})
        active = manager.get(token)
        self.assertEqual(set(active["active_chunks"]), {1, 2, 3})
        self.assertEqual(active["concurrency_limit"], 3)
        self.assertEqual(active["effective_concurrency"], 3)
        self.assertFalse(manager.submit(token, "runtime-key"))

        gate.release(2)
        self.wait_until(lambda: 2 in manager.get(token)["results"])
        self.assertNotIn(0, manager.get(token)["results"])
        self.assertIn(2, self.store.load(token)[0]["results"])
        gate.release(1)
        self.wait_until(lambda: 1 in manager.get(token)["results"])
        gate.release_all()
        final = self.finish(manager, token)
        self.assertEqual(final["status"], "complete")
        self.assertEqual(gate.peak, 3)
        self.assertEqual(len(gate.snapshot()), 6)
        self.assertEqual(final["active_chunks"], [])
        self.assert_ordered_results(final, 6)
        durable = self.store.load(token)[0]
        self.assertEqual(durable["results"], final["results"])
        self.assert_ordered_results(durable, 6)
        self.assertTrue(all(count == 1 for count in final["attempts"].values()))

    def test_shared_chunk_pool_bounds_multiple_jobs_and_cancelled_queue_makes_no_calls(self):
        gate = self.gate()
        manager = self.manager(analyze_chunk=gate.analyze, max_chunk_workers=4)
        first_state, first_pairs = fixture(6, label="first.txt")
        second_state, second_pairs = fixture(6, label="second.txt")
        first = manager.create(first_state, first_pairs)
        second = manager.create(second_state, second_pairs)
        self.assertTrue(manager.submit(first, "runtime-key"))
        self.assertTrue(manager.submit(second, "runtime-key"))
        self.assertTrue(gate.wait_for_calls(4))
        initial = gate.snapshot()
        self.assertEqual(len(initial), 4)
        self.assertEqual(gate.peak, 4)
        self.assertLessEqual(max(Counter(call["filename"] for call in initial).values()), 3)
        self.assertTrue(manager.stop(first))
        self.assertTrue(manager.stop(second))
        gate.release_all()
        first_final = self.finish(manager, first)
        second_final = self.finish(manager, second)
        self.assertEqual(len(gate.snapshot()), 4)
        self.assertEqual(gate.peak, 4)
        self.assertEqual(first_final["status"], "incomplete")
        self.assertEqual(second_final["status"], "incomplete")
        self.assertEqual(len(first_final["results"]) + len(second_final["results"]), 4)
        self.assertEqual(first_final["active_chunks"], [])
        self.assertEqual(second_final["active_chunks"], [])

    def test_small_serial_job_uses_shared_pool_and_cancelled_waiting_call_never_starts(self):
        gate = self.gate()
        manager = self.manager(
            analyze_chunk=gate.analyze, parallel_threshold=8, max_chunk_workers=3
        )
        large_state, large_pairs = fixture(8, label="large.txt")
        small_state, small_pairs = fixture(1, label="small.txt")
        large = manager.create(large_state, large_pairs)
        small = manager.create(small_state, small_pairs)
        self.assertTrue(manager.submit(large, "runtime-key"))
        self.assertTrue(gate.wait_for_calls(3))
        self.assertTrue(manager.submit(small, "runtime-key"))
        self.wait_until(lambda: manager.get(small)["current_attempt"] == 1)
        self.assertEqual(manager.get(small)["effective_concurrency"], 1)
        self.assertFalse(gate.wait_for_calls(4, timeout=0.2),
                         "A serial document bypassed the shared chunk pool.")
        self.assertTrue(manager.stop(small))
        self.assertTrue(manager.stop(large))
        gate.release_all()
        small_final = self.finish(manager, small)
        large_final = self.finish(manager, large)
        self.assertEqual(len(gate.snapshot()), 3)
        self.assertEqual(gate.peak, 3)
        self.assertEqual(small_final["status"], "incomplete")
        self.assertEqual(small_final["results"], {})
        self.assertEqual(small_final["attempts"], {})
        self.assertEqual(large_final["status"], "incomplete")
        self.assertEqual(len(large_final["results"]), 3)

    def test_stop_drains_three_inflight_successes_and_resume_does_not_repeat_them(self):
        gate = self.gate()
        manager = self.manager(analyze_chunk=gate.analyze)
        state, pairs = fixture(8)
        token = manager.create(state, pairs)
        manager.submit(token, "runtime-key")
        self.assertTrue(gate.wait_for_calls(3))
        self.assertTrue(manager.stop(token))
        self.assertTrue(manager.is_running(token))
        gate.release_all()
        paused = self.finish(manager, token)
        self.assertEqual(paused["status"], "incomplete")
        self.assertIn("stopped by request", paused["error"].lower())
        self.assertEqual(set(paused["results"]), {0, 1, 2})
        self.assertEqual(len(gate.snapshot()), 3)
        self.assertEqual(self.store.load(token)[0]["results"], paused["results"])
        self.assertTrue(manager.submit(token, "replacement-key"))
        final = self.finish(manager, token)
        self.assertEqual(final["status"], "complete")
        self.assertEqual(Counter(call["index"] for call in gate.snapshot()), Counter(range(8)))
        self.assert_ordered_results(final, 8)
        self.assertTrue(all(final["attempts"][index] == 1 for index in range(8)))

    def test_storage_failure_keeps_other_inflight_successes_in_memory_and_resumes_only_missing(self):
        gate = self.gate()
        faulty = FaultingProgressStore(self.store)
        manager = self.manager(store=faulty, analyze_chunk=gate.analyze)
        state, pairs = fixture(7)
        token = manager.create(state, pairs)
        manager.submit(token, "runtime-key")
        self.assertTrue(gate.wait_for_calls(3))
        gate.release(2)
        self.assertTrue(faulty.failed.wait(5))
        self.assertTrue(manager.is_running(token), "Other in-flight calls must be drained.")
        self.assertEqual(len(gate.snapshot()), 3)
        gate.release_all()
        retained = self.finish(manager, token)
        self.assertEqual(retained["status"], "incomplete")
        self.assertIn("disk full", retained["error"])
        self.assertEqual(set(retained["results"]), {0, 1, 2})
        self.assertEqual(len(gate.snapshot()), 3)
        self.assertEqual(self.store.load(token)[0]["results"], {})
        faulty.broken = False
        self.assertTrue(manager.submit(token, "replacement-key"))
        final = self.finish(manager, token)
        self.assertEqual(final["status"], "complete")
        self.assertEqual(Counter(call["index"] for call in gate.snapshot()), Counter(range(7)))
        self.assert_ordered_results(final, 7)
        self.assertEqual(self.store.load(token)[0]["results"], final["results"])

    def test_fresh_manager_recovers_noncontiguous_results_and_processes_only_missing(self):
        state, pairs = fixture(6)
        state["status"] = "running"
        for index in (2, 0):
            state["results"][index] = (*pairs[index], f"generated-{index}", {"rouge_l": 0.0})
            state["attempts"][index] = 1
        state["active_chunks"] = [2, 4]
        state["concurrency_limit"] = 3
        state["effective_concurrency"] = 3
        token = self.store.create(state, pairs)
        gate = self.gate()
        manager = self.manager(store=DocumentCheckpointStore(self.root), analyze_chunk=gate.analyze)
        restored = manager.get(token)
        self.assertEqual(restored["status"], "incomplete")
        self.assertEqual(set(restored["results"]), {0, 2})
        self.assertEqual(restored["active_chunks"], [])
        self.assertTrue(manager.submit(token, "fresh-key"))
        self.assertTrue(gate.wait_for_calls(3))
        self.assertEqual({call["index"] for call in gate.snapshot()}, {1, 3, 4})
        gate.release_all()
        final = self.finish(manager, token)
        self.assertEqual(final["status"], "complete")
        self.assertEqual(Counter(call["index"] for call in gate.snapshot()), Counter((1, 3, 4, 5)))
        self.assert_ordered_results(final, 6)
        self.assertEqual(final["attempts"][0], 1)
        self.assertEqual(final["attempts"][2], 1)

    def test_parallel_auth_failure_drains_successes_and_redacts_runtime_key_everywhere(self):
        gate = self.gate()
        secret = "parallel-runtime-secret-391738"
        gate.response = lambda call: (
            ("Error: 403 PERMISSION_DENIED " + call["api_key"], None)
            if call["index"] == 0 else gate.success(call)
        )
        manager = self.manager(analyze_chunk=gate.analyze)
        state, pairs = fixture(6)
        token = manager.create(state, pairs, owner_id="owner")
        manager.submit(token, secret, owner_id="owner")
        self.assertTrue(gate.wait_for_calls(3))
        with self.assertRaises(CheckpointError):
            manager.get(token, owner_id="other")
        gate.release(0)
        self.wait_until(lambda: 0 in manager.get(token, owner_id="owner")["failures"])
        gate.release_all()
        paused = self.finish(manager, token, owner_id="owner")
        self.assertEqual(paused["status"], "incomplete")
        self.assertEqual(set(paused["results"]), {1, 2})
        self.assertEqual(len(gate.snapshot()), 3)
        self.assertIn("[redacted]", paused["failures"][0])
        self.assertNotIn(secret, repr(paused))
        self.assertNotIn(secret.encode(), self.store.db_path.read_bytes())
        self.assertEqual(self.store.load(token)[0]["results"], paused["results"])
        gate.response = None
        self.assertTrue(manager.submit(token, "replacement-key", owner_id="owner"))
        final = self.finish(manager, token, owner_id="owner")
        self.assertEqual(final["status"], "complete")
        counts = Counter(call["index"] for call in gate.snapshot())
        self.assertEqual(counts, Counter({0: 2, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}))
        self.assert_ordered_results(final, 6)

    def test_small_documents_keep_serial_execution_with_default_threshold(self):
        gate = self.gate()
        manager = self.manager(analyze_chunk=gate.analyze, parallel_threshold=8)
        state, pairs = fixture(3)
        token = manager.create(state, pairs)
        manager.submit(token, "runtime-key")
        self.assertTrue(gate.wait_for_calls(1))
        active = manager.get(token)
        self.assertEqual(active["effective_concurrency"], 1)
        self.assertEqual(active["active_chunks"], [1])
        self.assertEqual(len(gate.snapshot()), 1)
        gate.release_all()
        final = self.finish(manager, token)
        self.assertEqual(final["status"], "complete")
        self.assertEqual(gate.peak, 1)
        self.assert_ordered_results(final, 3)


if __name__ == "__main__":
    unittest.main()
