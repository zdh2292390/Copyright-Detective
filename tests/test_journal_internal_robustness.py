"""Lease startup, replay isolation and bounded snapshot allocations."""
import base64
from copy import deepcopy
import gzip
import hashlib
import json
from threading import Event
import tracemalloc
import unittest
from unittest.mock import Mock, patch

from src import resumable_analysis as recovery
from src.analysis_checkpoints import AnalysisCheckpointError
from src.run_cancellation import AnalysisCancelled, CancellationToken, cancellation_scope


class JournalStore:
    owner_id = 'owner-test'
    def __init__(self):
        self.task = {'id': 'task-test', 'page_key': 'Content Recall Detection', 'source': {}, 'metadata': {}, 'status': 'queued'}
        self.items = []
        self.renewed = Event()
        self.released = []
    def claim(self, task_id, **kwargs):
        self.task.update(status='running', lease_token='lease-test')
        return deepcopy(self.task)
    def heartbeat(self, task_id, lease_token, **kwargs):
        self.renewed.set()
        return deepcopy(self.task)
    def load_items(self, task_id, offset=0, limit=200):
        return deepcopy(self.items[offset:offset + limit])
    def release(self, task_id, lease_token, **kwargs):
        self.released.append((task_id, lease_token, kwargs))
        self.task.update(status=kwargs['status'], lease_token=None)
        return deepcopy(self.task)


class JournalStartupTests(unittest.TestCase):
    def test_restoration_renews_lease_while_loading_before_any_api(self):
        store = JournalStore()
        def slow_load(*args, **kwargs):
            self.assertTrue(store.renewed.wait(1), 'Lease heartbeat must already be running')
            return []
        store.load_items = slow_load
        with patch.object(recovery, 'HEARTBEAT_SECONDS', 0.01):
            journal = recovery.CallJournal(store, deepcopy(store.task))
            try:
                self.assertTrue(store.renewed.is_set())
            finally:
                journal.finish(False)
                journal.thread.join(1)
        self.assertEqual(store.released[0][2]['status'], 'incomplete')
        self.assertFalse(journal.thread.is_alive())

    def test_thread_start_failure_releases_claim_and_never_leaves_active_token(self):
        store, token = JournalStore(), CancellationToken()
        with cancellation_scope(token), patch.object(recovery, 'Thread', side_effect=RuntimeError('cannot start thread')):
            with self.assertRaises(AnalysisCheckpointError):
                recovery.CallJournal(store, deepcopy(store.task))
        self.assertEqual(len(store.released), 1)
        self.assertEqual(store.released[0][2]['status'], 'incomplete')
        self.assertIsNone(store.task['lease_token'])

    def test_cancel_during_final_load_cleans_up_own_lease(self):
        store, token = JournalStore(), CancellationToken()
        def load(*args, **kwargs):
            token.cancel()
            return []
        store.load_items = load
        with cancellation_scope(token):
            with self.assertRaises(AnalysisCancelled):
                recovery.CallJournal(store, deepcopy(store.task), heartbeat=False)
        self.assertEqual(len(store.released), 1)
        self.assertEqual(store.released[0][2]['status'], 'incomplete')

    def test_loader_base_exception_releases_lease_without_swallowing_signal(self):
        store = JournalStore()
        store.load_items = Mock(side_effect=SystemExit('script stopped'))
        with self.assertRaises(SystemExit):
            recovery.CallJournal(store, deepcopy(store.task), heartbeat=False)
        self.assertEqual(len(store.released), 1)

    def test_heartbeat_failure_cancels_nested_provider_work(self):
        store, token = JournalStore(), CancellationToken()
        with cancellation_scope(token):
            journal = recovery.CallJournal(store, deepcopy(store.task), heartbeat=False)
        try:
            store.heartbeat = Mock(side_effect=ConnectionError('lost lease connection'))
            with patch.object(recovery, 'HEARTBEAT_SECONDS', 0):
                journal._heartbeat()
            self.assertTrue(journal.broken)
            self.assertTrue(token.cancelled)
            with cancellation_scope(token), self.assertRaises(AnalysisCancelled):
                recovery.checkpoint_call('nested', {}, Mock())
        finally:
            journal.finish(False)

    def test_cached_replay_detaches_nested_results_without_double_copy(self):
        store = JournalStore()
        request = {'operation': 'test', 'request': {'x': 1}}
        request['fingerprint'] = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        stored = {'items': [{'values': [1, 2]}], 'pair': {'__analysis_tuple__': ['a', {'list': [3]}]}}
        store.items = [{'item_index': 0, 'input': request, 'status': 'complete', 'result': deepcopy(stored)}]
        store.task['status'] = 'complete'
        journal = recovery.CallJournal(store, deepcopy(store.task), heartbeat=False)
        invoke = Mock()
        try:
            result = journal.call('test', {'x': 1}, invoke)
            result['items'][0]['values'].append(99)
            result['pair'][1]['list'].append(99)
            self.assertEqual(journal.items[0]['result'], stored)
            invoke.assert_not_called()
        finally:
            journal.finish(True)


class SnapshotAllocationTests(unittest.TestCase):
    def test_small_snapshot_keeps_original_representation(self):
        snapshot = {'text': 'hello', 'flags': [1, True, None]}
        self.assertIs(recovery.pack_snapshot(snapshot), snapshot)

    def test_streamed_unicode_snapshot_round_trips_and_is_deterministic(self):
        snapshot = {'text': '测试🙂 with quotes " and newlines\n' * 20000,
                    'nested': [{'__analysis_tuple__': [1, {'v': 2.5}]}]}
        packed = recovery.pack_snapshot(snapshot)
        self.assertIn('__analysis_snapshot_gzip__', packed)
        self.assertEqual(recovery.unpack_snapshot(packed), snapshot)
        self.assertEqual(recovery.pack_snapshot(snapshot), packed)

    def test_legacy_gzip_snapshot_remains_readable(self):
        snapshot = {'text': 'a' * 600000}
        old = {'__analysis_snapshot_gzip__': base64.b64encode(gzip.compress(json.dumps(snapshot, ensure_ascii=False, separators=(',', ':')).encode(), mtime=0)).decode()}
        self.assertEqual(recovery.unpack_snapshot(old), snapshot)

    def test_nested_credentials_still_rejected_before_compression(self):
        with self.assertRaisesRegex(AnalysisCheckpointError, 'Credentials'):
            recovery.pack_snapshot({'rows': [{'API_key': 'private'}]})

    def test_raw_size_guard_stops_stream_without_full_allocation(self):
        # Feed encoder pieces rather than allocating a 64 MiB test document.
        piece = 'x' * 65536
        with patch.object(recovery.json.JSONEncoder, 'iterencode', return_value=iter([piece] * 1025)):
            with self.assertRaisesRegex(AnalysisCheckpointError, 'too large'):
                recovery.pack_snapshot({})

    def test_large_structured_snapshot_has_lower_peak_python_allocation(self):
        snapshot = {'rows': [{'text': 'sample text ' * 24, 'score': i / 1000} for i in range(8000)]}
        def old_pack():
            raw = json.dumps(snapshot, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
            return base64.b64encode(gzip.compress(raw, mtime=0)).decode()
        def peak(call):
            tracemalloc.start()
            try:
                call()
                return tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
        previous = peak(old_pack)
        current = peak(lambda: recovery.pack_snapshot(snapshot))
        self.assertLess(current, previous / 2)
        print(f'Snapshot peak allocations: previous={previous} bytes, streamed={current} bytes')


if __name__ == '__main__':
    unittest.main()
