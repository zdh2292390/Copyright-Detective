"""Backend regressions for large document recovery; no UI/API services needed."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from threading import Event
import time
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from src.document_analysis import new_analysis_state
from src.document_checkpoints import CheckpointError, DocumentCheckpointStore
from src.document_jobs import DocumentAnalysisJobs
from src.run_cancellation import cancellable_call
from src.supabase_document_checkpoints import SupabaseDocumentCheckpointStore


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def manifest(total, *, complete=False, generated='generated'):
    settings = {
        'filename': 'source.txt', 'provider': 'Local vLLM', 'model': 'test-model',
        'chunk_size': 200, 'continuation_method': 'Normal Continuation',
        'temperature': 0.7, 'top_p': 0.9,
    }
    pairs = [(f'prefix {index}', f'target {index}') for index in range(total)]
    state = new_analysis_state('a' * 64, settings, total)
    state['owner_id'] = 'owner-a'
    if complete:
        state['status'] = 'complete'
        state['results'] = {
            index: (*pair, generated, {'ROUGE-L': 0.5}) for index, pair in enumerate(pairs)
        }
        state['attempts'] = {index: 1 for index in range(total)}
    return state, pairs


class CloudBackend:
    """Model owner-bound lifecycle constraints and the actual RPC limits."""
    owner_id = 'owner-a'

    def __init__(self, tasks=None):
        self.tasks = {} if tasks is None else tasks
        self.calls = []
        self.fail_save_call = None
        self.save_attempts = 0
        self.release_calls = 0
        self.heartbeat_calls = 0
        self.latest_id = None
        self.manifest_inputs_lazy = None

    def create_task(self, page_key, settings, source, inputs, *, fingerprint, metadata):
        self.manifest_inputs_lazy = not isinstance(inputs, (list, tuple))
        inputs = list(inputs)  # The fake database retains its rows.
        token = str(uuid4())
        self.latest_id = token
        task = {
            'id': token, 'owner_id': self.owner_id, 'page_key': page_key,
            'settings': deepcopy(settings), 'source': deepcopy(source),
            'fingerprint': fingerprint, 'metadata': deepcopy(metadata),
            'total_items': len(inputs), 'status': 'queued', 'revision': 0,
            'completed_items': 0, 'failed_items': 0, 'stop_requested': False,
            'lease_token': None, 'lease_expires_at': None,
            'items': [
                {'item_index': index, 'input': deepcopy(value), 'status': 'pending',
                 'result': None, 'error': None, 'attempts': 0}
                for index, value in enumerate(inputs)
            ],
        }
        self.tasks[token] = task
        return self.get_task(token)

    def get_task(self, token):
        task = self.tasks.get(token)
        if task is None:
            return None
        return deepcopy({key: value for key, value in task.items() if key != 'items'})

    def load_items(self, token, *, offset=0, limit=1000):
        return deepcopy(self.tasks[token]['items'][offset:offset + limit])

    def claim(self, token, *, ttl_seconds):
        from src.analysis_checkpoints import AnalysisLeaseError
        task = self.tasks[token]
        if task['lease_token'] is not None:
            raise AnalysisLeaseError('Another worker owns this task.')
        task.update(lease_token=str(uuid4()), status='running', lease_expires_at='2099-01-01T00:00:00+00:00')
        task['revision'] += 1
        return self.get_task(token)

    def save_items(self, token, updates, *, metadata, status, lease_token):
        self.save_attempts += 1
        if self.save_attempts == self.fail_save_call:
            raise RuntimeError('Database write failed.')
        assert len(updates) <= 1000
        assert len(json.dumps(updates, ensure_ascii=False).encode('utf-8')) <= 16 * 1024 * 1024
        task = self.tasks[token]
        assert task['lease_token'] == lease_token
        candidate = deepcopy(task['items'])
        for update in updates:
            index = update['index']
            old = candidate[index]
            if old['status'] == 'complete':
                assert update['status'] == 'complete' and update['result'] == old['result']
            old.update({key: deepcopy(update[key]) for key in ('status', 'result', 'error', 'attempts')})
        if status == 'complete':
            assert all(item['status'] == 'complete' for item in candidate), 'Completed before all batches committed.'
        task['items'] = candidate
        task['metadata'] = deepcopy(metadata)
        task['status'] = status
        task['completed_items'] = sum(item['status'] == 'complete' for item in candidate)
        task['failed_items'] = sum(item['status'] == 'failed' for item in candidate)
        task['revision'] += 1
        self.calls.append({'indices': [update['index'] for update in updates], 'status': status,
                           'bytes': len(json.dumps(updates, ensure_ascii=False).encode('utf-8'))})
        return self.get_task(token)

    def heartbeat(self, token, lease_token, *, ttl_seconds):
        self.heartbeat_calls += 1
        assert self.tasks[token]['lease_token'] == lease_token
        return {'stop_requested': self.tasks[token]['stop_requested']}

    def release(self, token, lease_token, *, status):
        self.release_calls += 1
        task = self.tasks[token]
        assert task['lease_token'] == lease_token
        if task['status'] == 'complete':
            assert status == 'complete', 'An old snapshot downgraded a completed task.'
        if status == 'complete':
            assert task['completed_items'] == task['total_items']
        task.update(lease_token=None, lease_expires_at=None, status=status)
        task['revision'] += 1


class CloudDocumentRobustnessTests(unittest.TestCase):
    def test_import_1501_successes_batches_and_completes_only_after_all_rows(self):
        backend = CloudBackend()
        adapter = SupabaseDocumentCheckpointStore(backend)
        state, pairs = manifest(1501, complete=True)
        token = adapter.create(state, pairs)
        self.assertTrue(backend.manifest_inputs_lazy)
        self.assertEqual([len(call['indices']) for call in backend.calls], [1000, 501])
        self.assertEqual([call['status'] for call in backend.calls], ['incomplete', 'complete'])
        restored, source = adapter.load(token)
        self.assertEqual(restored['results'], state['results'])
        self.assertEqual(source, pairs)
        self.assertNotIn('results', adapter._identities[token])
        self.assertNotIn('attempts', adapter._identities[token])
        self.assertIsNone(backend.tasks[token]['lease_token'])

    def test_failed_second_batch_keeps_acknowledged_rows_and_retry_skips_them(self):
        backend = CloudBackend()
        backend.fail_save_call = 2
        adapter = SupabaseDocumentCheckpointStore(backend)
        state, pairs = manifest(1501, complete=True)
        with self.assertRaises(CheckpointError):
            adapter.create(state, pairs)
        token = backend.latest_id
        self.assertEqual(backend.tasks[token]['status'], 'incomplete')
        self.assertEqual(backend.tasks[token]['completed_items'], 1000)
        self.assertEqual(len(adapter._persisted[token]), 1000)
        backend.fail_save_call = None
        self.assertTrue(adapter.claim(token))
        adapter.save(token, state)
        adapter.release(token)
        self.assertEqual(backend.calls[-1]['indices'], list(range(1000, 1501)))
        self.assertEqual(backend.tasks[token]['status'], 'complete')
        self.assertEqual(adapter.load(token)[0]['results'], state['results'])

    def test_stale_load_cannot_replace_newer_acknowledged_results_or_completion(self):
        backend = CloudBackend()
        adapter = SupabaseDocumentCheckpointStore(backend)
        initial, pairs = manifest(2)
        token = adapter.create(initial, pairs)
        self.assertTrue(adapter.claim(token))
        completed, _ = manifest(2, complete=True)
        original_get = backend.get_task
        reads = 0
        wrote = False
        def interleaved_get(task_id):
            nonlocal reads, wrote
            snapshot = original_get(task_id)
            reads += 1
            if reads == 2 and not wrote:
                wrote = True
                # Commit after the loader captured its final stable server
                # snapshot, but before it can fill the local caches.
                adapter.save(token, completed)
            return snapshot
        with patch.object(backend, 'get_task', side_effect=interleaved_get):
            restored, _ = adapter.load(token)
        self.assertTrue(wrote)
        self.assertEqual(restored['status'], 'complete')
        self.assertEqual(restored['results'], completed['results'])
        self.assertEqual(set(adapter._persisted[token]), {0, 1})
        self.assertEqual(adapter._identities[token]['status'], 'complete')
        adapter.release(token)
        self.assertEqual(backend.tasks[token]['status'], 'complete')
        self.assertIsNone(backend.tasks[token]['lease_token'])

    def test_large_unicode_results_use_byte_bounded_batches(self):
        backend = CloudBackend()
        adapter = SupabaseDocumentCheckpointStore(backend)
        state, pairs = manifest(90, complete=True, generated='文' * 30000)
        token = adapter.create(state, pairs)
        self.assertGreater(len(backend.calls), 1)
        self.assertTrue(all(call['bytes'] <= 4 * 1024 * 1024 for call in backend.calls))
        self.assertEqual(backend.tasks[token]['completed_items'], 90)
        self.assertEqual(backend.calls[-1]['status'], 'complete')
        self.assertTrue(all(call['status'] == 'incomplete' for call in backend.calls[:-1]))

    def test_inactive_cache_is_bounded_but_active_worker_entries_are_pinned(self):
        backend = CloudBackend()
        adapter = SupabaseDocumentCheckpointStore(backend, max_cached_tasks=2)
        state, pairs = manifest(3)
        active = adapter.create(state, pairs)
        self.assertTrue(adapter.claim(active))
        tokens = [adapter.create(state, pairs) for _ in range(4)]
        self.assertIn(active, adapter._sources)
        self.assertIn(active, adapter._leases)
        self.assertLessEqual(len(adapter._cache_order), 2)
        evicted = tokens[0]
        self.assertNotIn(evicted, adapter._sources)
        restored, loaded_pairs = adapter.load(evicted)
        self.assertEqual(loaded_pairs, pairs)
        self.assertEqual(restored['total_chunks'], 3)
        self.assertIn(active, adapter._sources)
        self.assertLessEqual(len(adapter._cache_order), 2)
        adapter.release(active)

    def test_oversized_inactive_cache_is_evicted_without_losing_database_results(self):
        backend = CloudBackend()
        adapter = SupabaseDocumentCheckpointStore(backend, max_cache_bytes=1024)
        state, pairs = manifest(2, complete=True, generated='stored result ' * 100)
        token = adapter.create(state, pairs)
        self.assertNotIn(token, adapter._cache_order)
        restored, loaded_pairs = adapter.load(token)
        self.assertEqual(restored['results'], state['results'])
        self.assertEqual(loaded_pairs, pairs)
        self.assertNotIn(token, adapter._cache_order)
        self.assertEqual(adapter._identities, {})

    def test_same_owner_client_refresh_preserves_existing_worker_lease(self):
        backend = CloudBackend()
        jobs = DocumentAnalysisJobs()
        try:
            jobs.configure_cloud_store(backend)
            adapter = jobs._cloud_stores['owner-a']
            state, pairs = manifest(2)
            token = adapter.create(state, pairs)
            self.assertTrue(adapter.claim(token))
            lease = adapter._leases[token]
            refreshed = CloudBackend(tasks=backend.tasks)
            jobs.configure_cloud_store(refreshed)
            self.assertIs(jobs._cloud_stores['owner-a'], adapter)
            self.assertIs(adapter.store, refreshed)
            self.assertEqual(adapter._leases[token], lease)
            adapter.heartbeat(token)
            adapter.release(token)
            self.assertEqual(backend.heartbeat_calls, 0)
            self.assertEqual(refreshed.heartbeat_calls, 1)
            self.assertEqual(refreshed.release_calls, 1)
            other = CloudBackend()
            other.owner_id = 'owner-b'
            with self.assertRaises(CheckpointError):
                adapter.bind_store(other)
        finally:
            jobs.close()


class LeasedLocalStore(DocumentCheckpointStore):
    remote = True

    def __init__(self, root):
        super().__init__(root)
        self.claimed = False
        self.release_calls = 0

    def claim(self, token):
        self.claimed = True
        return True

    def release(self, token):
        self.release_calls += 1
        self.claimed = False

    def is_active(self, token):
        return self.claimed

    def heartbeat(self, token):
        return {'stop_requested': False}


class DocumentWorkerRobustnessTests(unittest.TestCase):
    def test_heartbeat_thread_start_failure_cleans_task_without_sending_api(self):
        from src import document_jobs as module
        for failure in ('constructor', 'start'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                store = LeasedLocalStore(Path(directory))
                analyze = Mock(return_value=('generated', {'ROUGE-L': 0.5}))
                jobs = DocumentAnalysisJobs(store, analyze_chunk=analyze)
                try:
                    state, pairs = manifest(2)
                    token = jobs.create(state, pairs, owner_id='owner-a', use_cloud=False)
                    broken_thread = Mock()
                    broken_thread.start.side_effect = RuntimeError('Could not start heartbeat thread.')
                    thread_patch = (
                        patch.object(module, 'Thread', side_effect=RuntimeError('Could not construct heartbeat thread.'))
                        if failure == 'constructor' else patch.object(module, 'Thread', return_value=broken_thread)
                    )
                    with thread_patch:
                        self.assertTrue(jobs.submit(token, '', owner_id='owner-a'))
                        self.assertTrue(wait_for(lambda: not jobs.is_running(token)))
                    self.assertEqual(store.release_calls, 1)
                    self.assertFalse(store.claimed)
                    self.assertEqual(jobs.get(token, owner_id='owner-a')['status'], 'incomplete')
                    self.assertIn('RuntimeError', jobs.get(token, owner_id='owner-a')['error'])
                    analyze.assert_not_called()
                    broken_thread.join.assert_not_called()
                finally:
                    jobs.close()

    def test_unsaved_retained_results_block_new_jobs_but_can_resume_themselves(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentCheckpointStore(Path(directory))
            calls = []
            def analyze(settings, key, upper, lower):
                calls.append(upper)
                return 'generated', {'ROUGE-L': 0.5}
            jobs = DocumentAnalysisJobs(store, max_pending=1, analyze_chunk=analyze)
            try:
                state, pairs = manifest(2)
                token = jobs.create(state, pairs, owner_id='owner-a', use_cloud=False)
                retained = deepcopy(state)
                retained['results'][0] = (*pairs[0], 'already paid', {'ROUGE-L': 0.5})
                jobs._jobs[token] = {'state': retained, 'finished': True, 'storage_failed': True}
                new_token = jobs.create(state, pairs, owner_id='owner-a', use_cloud=False)
                with self.assertRaises(CheckpointError):
                    jobs.submit(new_token, '', owner_id='owner-a')
                self.assertTrue(jobs.submit(token, '', owner_id='owner-a'))
                self.assertTrue(wait_for(lambda: not jobs.is_running(token)))
                restored, _ = store.load(token)
                self.assertEqual(restored['status'], 'complete')
                self.assertEqual(restored['results'][0][2], 'already paid')
                self.assertEqual(calls, [pairs[1][0]])
            finally:
                jobs.close()

    def test_initial_snapshot_redacts_saved_failure_before_incremental_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            state, pairs = manifest(2)
            state['failures'][1] = 'Error: response blocked while using secret-key'
            entered = Event()
            release = Event()
            def analyze(*args):
                def request():
                    entered.set()
                    release.wait(5)
                cancellable_call(request)
                return 'generated', {'ROUGE-L': 0.5}
            jobs = DocumentAnalysisJobs(DocumentCheckpointStore(Path(directory)), analyze_chunk=analyze)
            try:
                token = jobs.create(state, pairs, owner_id='owner-a', use_cloud=False)
                self.assertTrue(jobs.submit(token, 'secret-key', owner_id='owner-a'))
                self.assertTrue(entered.wait(5))
                failure = jobs.get(token, owner_id='owner-a')['failures'][1]
                self.assertNotIn('secret-key', failure)
                self.assertIn('[redacted]', failure)
                jobs.force_stop(token, owner_id='owner-a')
                release.set()
                self.assertTrue(wait_for(lambda: not jobs.is_running(token)))
            finally:
                release.set()
                jobs.close()

    def test_parallel_snapshots_keep_all_successes_and_defend_against_ui_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentCheckpointStore(Path(directory))
            state, pairs = manifest(41)
            entered = Event()
            release = Event()
            def analyze(settings, key, upper, lower):
                if upper == pairs[-1][0]:
                    def request():
                        entered.set()
                        release.wait(5)
                    cancellable_call(request)
                return 'generated', {'ROUGE-L': 0.5}
            jobs = DocumentAnalysisJobs(store, analyze_chunk=analyze, chunk_concurrency=3)
            try:
                token = jobs.create(state, pairs, owner_id='owner-a', use_cloud=False)
                self.assertTrue(jobs.submit(token, '', owner_id='owner-a'))
                self.assertTrue(entered.wait(5))
                self.assertTrue(wait_for(lambda: len(jobs.get(token, owner_id='owner-a')['results']) == 40))
                snapshot = jobs.get(token, owner_id='owner-a')
                snapshot['results'][0][3]['ROUGE-L'] = 999
                snapshot['results'].clear()
                untouched = jobs.get(token, owner_id='owner-a')
                self.assertEqual(len(untouched['results']), 40)
                self.assertEqual(untouched['results'][0][3]['ROUGE-L'], 0.5)
                release.set()
                self.assertTrue(wait_for(lambda: not jobs.is_running(token)))
                restored, _ = store.load(token)
                self.assertEqual(restored['status'], 'complete')
                self.assertEqual(set(restored['results']), set(range(41)))
            finally:
                release.set()
                jobs.close()


if __name__ == '__main__':
    unittest.main()
