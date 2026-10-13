"""Regression checks for bounded checkpoint retries and streamed manifests."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import tempfile
import tracemalloc
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from src import analysis_checkpoints as checkpoints
from src.analysis_checkpoints import AnalysisCheckpointError, SupabaseAnalysisCheckpointStore

OWNER = '00000000-0000-0000-0000-000000000001'
TASK = '00000000-0000-0000-0000-000000000002'
LEASE = '00000000-0000-0000-0000-000000000003'
PAGE = 'Content Recall Detection'


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def legacy_fingerprint(page, settings, source, items, dynamic=False):
    """The public identity algorithm used before manifest spooling."""
    digest = hashlib.sha256()
    for value in (page, settings, source, dynamic):
        digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8'))
        digest.update(b'\0')
    for item in items:
        digest.update(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()


class BackendFailure(Exception):
    def __init__(self, code=None, message='fixture failure', status=None):
        self.code = code
        self.message = message
        self.status_code = status
        super().__init__(message)


class RpcClient:
    """Preserves call arguments independently of any mutable request builder."""
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.executions = 0

    def rpc(self, name, arguments):
        self.calls.append((name, deepcopy(arguments)))
        return self

    def execute(self):
        self.executions += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(data=outcome)


class ManifestStore(SupabaseAnalysisCheckpointStore):
    def __init__(self, *, status='creating', batch_size=200, fail_method=None):
        super().__init__(None, OWNER, batch_size=batch_size)
        self.status = status
        self.fail_method = fail_method
        self.calls = []
        self.inputs = {}
        self.reads = []

    def _rpc(self, method, **params):
        self.calls.append((method, deepcopy(params)))
        if method == self.fail_method:
            raise AnalysisCheckpointError('fixture write failure')
        if method == 'put_items':
            for item in params['p_items']:
                self.inputs[item['index']] = deepcopy(item['input'])
        return {'id': params['p_task_id'], 'status': self.status if method == 'create_task' else 'queued'}

    def load_items(self, identity, *, offset=0, limit=200):
        self.reads.append((identity, offset, limit))
        return [{'item_index': index, 'input': deepcopy(self.inputs[index])}
                for index in range(offset, offset + limit) if index in self.inputs]


class DiscardingStore(SupabaseAnalysisCheckpointStore):
    """Do not retain upload data when measuring the production streaming path."""
    def __init__(self, *, batch_size=200):
        super().__init__(None, OWNER, batch_size=batch_size)
        self.created = None
        self.uploaded = 0
        self.largest_batch = 0

    def _rpc(self, method, **params):
        if method == 'create_task':
            self.created = {key: value for key, value in params.items() if key != 'p_items'}
        if method == 'put_items':
            self.uploaded += len(params['p_items'])
            self.largest_batch = max(self.largest_batch, len(compact(params['p_items'])))
        return {'id': params['p_task_id'], 'status': 'creating' if method == 'create_task' else 'queued'}


class CheckpointRetryTests(TestCase):
    def test_idempotent_rpc_network_retries_keep_identical_owner_task_lease_arguments(self):
        methods = {'create_task', 'put_items', 'finalize_task', 'claim', 'heartbeat', 'save_items', 'request_stop', 'delete'}
        for method in methods:
            with self.subTest(method=method):
                client = RpcClient([TimeoutError('fixture timeout'), {'id': TASK, 'status': 'running'}])
                store = SupabaseAnalysisCheckpointStore(client, OWNER)
                with patch.object(checkpoints.time, 'sleep') as pause:
                    result = store._rpc(method, p_task_id=TASK, p_lease_token=LEASE, p_updates=[{'index': 4}])
                self.assertEqual(result['id'], TASK)
                self.assertEqual(client.calls[0], client.calls[1])
                self.assertEqual(client.calls[0][1]['p_owner_id'], OWNER)
                self.assertEqual(client.calls[0][1]['p_task_id'], TASK)
                self.assertEqual(client.calls[0][1]['p_lease_token'], LEASE)
                pause.assert_called_once_with(0.25)

    def test_network_retries_stop_after_two_attempts_even_if_third_would_succeed(self):
        for failure in (TimeoutError('fixture timeout'), BackendFailure(code='PGRST003'), BackendFailure(status=503)):
            with self.subTest(failure_type=type(failure).__name__):
                client = RpcClient([failure, failure, {'id': TASK}])
                with patch.object(checkpoints.time, 'sleep') as pause:
                    with self.assertRaises(AnalysisCheckpointError) as raised:
                        SupabaseAnalysisCheckpointStore(client, OWNER)._rpc('claim', p_task_id=TASK, p_lease_token=LEASE)
                self.assertEqual(raised.exception.kind, 'network')
                self.assertEqual(client.executions, 2)
                self.assertEqual(len(client.outcomes), 1)
                pause.assert_called_once_with(0.25)

    def test_append_release_and_unknown_rpc_never_retry_a_lost_response(self):
        for method in ('append_item', 'release', 'future_unknown_method'):
            with self.subTest(method=method):
                client = RpcClient([TimeoutError('fixture lost response'), {'id': TASK}])
                with patch.object(checkpoints.time, 'sleep') as pause:
                    with self.assertRaises(AnalysisCheckpointError) as raised:
                        SupabaseAnalysisCheckpointStore(client, OWNER)._rpc(method, p_task_id=TASK, p_lease_token=LEASE)
                self.assertEqual(raised.exception.kind, 'network')
                self.assertEqual(client.executions, 1)
                pause.assert_not_called()

    def test_schema_auth_permission_lease_unknown_and_invalid_rows_do_not_retry(self):
        cases = [(BackendFailure(code='42703'), 'schema'), (BackendFailure(status=401), 'auth'),
                 (BackendFailure(message='Invalid API key'), 'api_key'),
                 (BackendFailure(code='42501'), 'permission'),
                 (BackendFailure(code='P0001', message='analysis_lease: replaced'), 'lease'),
                 (RuntimeError('unknown failure'), 'checkpoint'), ([], 'checkpoint'),
                 (AnalysisCheckpointError('local validation rejected'), 'checkpoint')]
        for failure, kind in cases:
            with self.subTest(kind=kind, failure_type=type(failure).__name__):
                client = RpcClient([failure, {'id': TASK}])
                with patch.object(checkpoints.time, 'sleep') as pause:
                    with self.assertRaises(AnalysisCheckpointError) as raised:
                        SupabaseAnalysisCheckpointStore(client, OWNER)._rpc('save_items', p_task_id=TASK)
                self.assertEqual(raised.exception.kind, kind)
                self.assertEqual(client.executions, 1)
                pause.assert_not_called()

    def test_read_retries_same_query_once_and_returns_successful_rows(self):
        query = Mock()
        query.execute.side_effect = [ConnectionError('fixture disconnect'), SimpleNamespace(data=[{'id': TASK}])]
        with patch.object(checkpoints.time, 'sleep') as pause:
            rows = SupabaseAnalysisCheckpointStore(None, OWNER)._query(query)
        self.assertEqual(rows, [{'id': TASK}])
        self.assertEqual(query.execute.call_count, 2)
        pause.assert_called_once_with(0.25)

    def test_read_retry_is_bounded_and_non_network_errors_or_invalid_rows_are_not_retried(self):
        cases = [(TimeoutError('fixture timeout'), 'network', 2), (BackendFailure(code='PGRST205'), 'schema', 1),
                 (BackendFailure(status=403), 'permission', 1), (BackendFailure(status=401), 'auth', 1),
                 (RuntimeError('unknown failure'), 'checkpoint', 1), (SimpleNamespace(data='invalid rows'), 'checkpoint', 1)]
        for failure, kind, count in cases:
            with self.subTest(kind=kind, count=count):
                query = Mock()
                query.execute.side_effect = [failure, failure, SimpleNamespace(data=[{'id': TASK}])]
                with patch.object(checkpoints.time, 'sleep') as pause:
                    with self.assertRaises(AnalysisCheckpointError) as raised:
                        SupabaseAnalysisCheckpointStore(None, OWNER)._query(query)
                self.assertEqual(raised.exception.kind, kind)
                self.assertEqual(query.execute.call_count, count)
                self.assertEqual(pause.call_count, count - 1)


class ManifestStreamingTests(TestCase):
    def test_single_pass_1501_generator_validates_all_inputs_before_first_write_and_keeps_digest(self):
        store = ManifestStore()
        consumed = []
        settings = {'b': 2, 'a': {'temperature': 0.7}}
        source = {'text': '中文 source\nwith newline', 'nested': {'z': 9, 'a': 1}}
        def payload(index):
            return {'source': f'片段 {index}\nsecond line', 'prefix': ['α', index], 'nested': {'z': index, 'a': False}}
        def inputs():
            for index in range(1501):
                self.assertEqual(store.calls, [], 'No database mutation may precede complete manifest validation')
                consumed.append(index)
                yield payload(index)
        task = store.create_task(PAGE, settings, source, inputs(), task_id=TASK)
        self.assertEqual(consumed, list(range(1501)))
        create = store.calls[0][1]
        self.assertEqual(create['p_total_items'], 1501)
        self.assertEqual(create['p_fingerprint'], legacy_fingerprint(PAGE, settings, source, (payload(i) for i in range(1501))))
        uploaded = [item for method, params in store.calls if method == 'put_items' for item in params['p_items']]
        self.assertEqual([item['index'] for item in uploaded], list(range(1501)))
        self.assertEqual([item['input'] for item in uploaded], [payload(i) for i in range(1501)])
        self.assertEqual([len(params['p_items']) for method, params in store.calls if method == 'put_items'], [200] * 7 + [101])
        self.assertEqual(store.calls[-1][0], 'finalize_task')
        self.assertEqual(task['status'], 'queued')

    def test_empty_dynamic_task_creates_and_resumes_without_work_item_uploads_or_reads(self):
        for status in ('creating', 'queued'):
            with self.subTest(status=status):
                store = ManifestStore(status=status)
                result = store.create_task(PAGE, {'model': 'test-model'}, {}, iter(()), task_id=TASK, dynamic_items=True)
                self.assertEqual(store.calls[0][1]['p_total_items'], 0)
                self.assertEqual(store.calls[0][1]['p_fingerprint'], legacy_fingerprint(PAGE, {'model': 'test-model'}, {}, (), dynamic=True))
                self.assertEqual([method for method, _ in store.calls], ['create_task', 'finalize_task'] if status == 'creating' else ['create_task'])
                self.assertEqual(store.reads, [])
                self.assertEqual(result['status'], 'queued')

    def test_late_invalid_payload_dynamic_input_bad_fingerprint_or_generator_failure_write_nothing(self):
        def late_secret():
            for index in range(1001):
                yield {'text': f'valid {index}'}
            yield {'nested': {'api_key': 'must not persist'}}
        def broken_generator():
            yield {'text': 'valid first'}
            raise RuntimeError('fixture generator interrupted')
        cases = [(late_secret, {}, AnalysisCheckpointError),
                 (lambda: iter([{'text': 'nonempty'}]), {'dynamic_items': True}, AnalysisCheckpointError),
                 (lambda: iter([{'text': 'valid'}]), {'fingerprint': 'not-valid'}, AnalysisCheckpointError),
                 (broken_generator, {}, RuntimeError)]
        for factory, kwargs, error_type in cases:
            with self.subTest(kwargs=kwargs, error_type=error_type.__name__):
                store = ManifestStore()
                with self.assertRaises(error_type):
                    store.create_task(PAGE, {}, {}, factory(), task_id=TASK, **kwargs)
                self.assertEqual(store.calls, [])

    def test_unicode_upload_batches_obey_serialized_array_byte_budget_including_delimiters(self):
        store = ManifestStore(batch_size=1000)
        # Each row fits individually; two row byte sums fit but JSON commas/
        # brackets cross the budget. Byte counting must use UTF-8, not characters.
        one = {'index': 0, 'input': {'text': '汉' * 10}}
        budget = len(compact(one)) * 2
        with patch.object(checkpoints, '_MAX_BATCH_BYTES', budget):
            store.create_task(PAGE, {}, {}, ({'text': '汉' * 10} for _ in range(7)), task_id=TASK)
        batches = [params['p_items'] for method, params in store.calls if method == 'put_items']
        self.assertEqual([item['index'] for batch in batches for item in batch], list(range(7)))
        for batch in batches:
            self.assertLessEqual(len(compact(batch)), budget)

    def test_single_item_that_only_fits_before_row_array_wrapping_is_rejected_before_writes(self):
        store = ManifestStore()
        payload = {'text': '汉' * 29}
        self.assertLessEqual(len(compact(payload)), 100)
        self.assertGreater(len(compact([{'index': 0, 'input': payload}])), 100)
        with patch.object(checkpoints, '_MAX_BATCH_BYTES', 100):
            with self.assertRaises(AnalysisCheckpointError):
                store.create_task(PAGE, {}, {}, [payload], task_id=TASK)
        self.assertEqual(store.calls, [])

    def test_resume_compares_every_immutable_input_even_with_explicit_unchanged_fingerprint(self):
        store = ManifestStore(status='queued', batch_size=4)
        original = [{'text': f'input {index}'} for index in range(11)]
        store.inputs = dict(enumerate(original))
        store.create_task(PAGE, {}, {}, iter(original), task_id=TASK, fingerprint='a' * 64)
        self.assertEqual(store.reads, [(TASK, 0, 4), (TASK, 4, 4), (TASK, 8, 3)])
        self.assertEqual([method for method, _ in store.calls], ['create_task'])
        changed = deepcopy(original)
        changed[-1] = {'text': 'changed late input'}
        with self.assertRaisesRegex(AnalysisCheckpointError, 'inputs cannot be changed'):
            store.create_task(PAGE, {}, {}, iter(changed), task_id=TASK, fingerprint='a' * 64)
        self.assertEqual([method for method, _ in store.calls], ['create_task', 'create_task'])

    def test_resume_rejects_missing_or_reordered_items_not_just_equal_counts(self):
        for mode in ('missing', 'reordered'):
            with self.subTest(mode=mode):
                store = ManifestStore(status='running', batch_size=3)
                original = [{'text': f'input {index}'} for index in range(4)]
                store.inputs = dict(enumerate(original))
                if mode == 'missing':
                    store.inputs.pop(3)
                else:
                    store.inputs[2], store.inputs[3] = store.inputs[3], store.inputs[2]
                with self.assertRaisesRegex(AnalysisCheckpointError, 'inputs cannot be changed'):
                    store.create_task(PAGE, {}, {}, iter(original), task_id=TASK, fingerprint='a' * 64)
                self.assertEqual([method for method, _ in store.calls], ['create_task'])

    def test_spooled_manifest_closes_on_validation_upload_and_resume_failures(self):
        original_factory = tempfile.SpooledTemporaryFile
        handles = []
        def tracked(*args, **kwargs):
            handle = original_factory(*args, **kwargs)
            handles.append(handle)
            return handle
        cases = [(ManifestStore(), [{'secret_key': 'no'}], {}),
                 (ManifestStore(fail_method='put_items'), [{'text': 'valid'}], {}),
                 (ManifestStore(status='queued'), [{'text': 'missing stored'}], {'fingerprint': 'a' * 64})]
        with patch.object(checkpoints.tempfile, 'SpooledTemporaryFile', side_effect=tracked):
            for store, items, kwargs in cases:
                with self.subTest(status=store.status, fail=store.fail_method):
                    with self.assertRaises(AnalysisCheckpointError):
                        store.create_task(PAGE, {}, {}, iter(items), task_id=TASK, **kwargs)
                    self.assertTrue(handles[-1].closed)
        self.assertEqual(len(handles), len(cases))

    def test_large_manifest_rolls_to_disk_and_does_not_retain_the_full_input_list(self):
        original_factory = tempfile.SpooledTemporaryFile
        handles = []
        def tracked(*args, **kwargs):
            handle = original_factory(*args, **kwargs)
            handles.append(handle)
            return handle
        store = DiscardingStore()
        total = 5000
        text = 'x' * 4096
        tracemalloc.start()
        try:
            with patch.object(checkpoints.tempfile, 'SpooledTemporaryFile', side_effect=tracked):
                store.create_task(PAGE, {}, {}, ({'text': text + str(index)} for index in range(total)), task_id=TASK)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertTrue(handles[0]._rolled, 'The large manifest must spill rather than grow its RAM buffer')
        self.assertTrue(handles[0].closed)
        self.assertEqual(store.uploaded, total)
        self.assertEqual(store.created['p_total_items'], total)
        self.assertLessEqual(store.largest_batch, checkpoints._MAX_BATCH_BYTES)
        self.assertLess(peak, total * len(text) // 2, 'Streaming must not retain a second full document corpus')
        self.peak_bytes = peak
