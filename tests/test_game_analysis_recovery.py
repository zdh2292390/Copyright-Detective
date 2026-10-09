"""Official game writes keep legacy behavior and replay new durable task identities."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from streamlit.testing.v1 import AppTest
from src.game import storage
from src.game.engine import GameConfig, StageOneResult, StageTwoResult, ScoredGeneration
from src.pages import copyright_game as page
from src import resumable_analysis as journal

TASK = '11111111-1111-4111-8111-111111111111'
OWNER = '22222222-2222-4222-8222-222222222222'
PLAYER = storage.VerifiedParticipant(OWNER, 'tester', 'Tester', '')
CONFIG = GameConfig('Zero-Shot', 'Baseline', 1, 1)

class FakeQuery:
    def __init__(self, service, table):
        self.service, self.table_name, self.filters = service, table, {}
    def select(self, *args): return self
    def eq(self, key, value): self.filters[key] = value; return self
    def limit(self, count): return self
    def execute(self):
        rows = list(self.service.links.values()) if self.table_name == 'analysis_game_task_links' else []
        return SimpleNamespace(data=[copy.deepcopy(row) for row in rows if all(row.get(key) == value for key, value in self.filters.items())])

class AtomicService:
    """Model a committed RPC followed by a lost network response, with owner checks."""
    def __init__(self):
        self.links, self.rows, self.calls = {}, {}, []
        self.creations = 0
        self.lose_response = None
    def table(self, name): return FakeQuery(self, name)
    def rpc(self, name, params):
        self.calls.append((name, copy.deepcopy(params)))
        return SimpleNamespace(execute=lambda: self.execute(name, params))
    def execute(self, name, params):
        is_baseline = 'stage_one' in name
        is_complete = name.startswith('complete_')
        stage = 'stage_one' if is_baseline else 'stage_two'
        task = params.get('p_task_id')
        key = (task, stage)
        requested = {key: value for key, value in params.items() if key not in {'p_task_id', 'p_expected_run_id'}}
        fingerprint = json.dumps(requested, sort_keys=True)
        if is_complete:
            linked = self.links.get(key)
            if task and (not linked or linked['user_id'] != params['p_user_id'] or linked['run_id'] != params['p_run_id']):
                raise RuntimeError('42501 saved reservation does not belong to task')
            row = self.rows[params['p_run_id']]
            if row['status'] == 'completed':
                if linked['completion'] != fingerprint:
                    raise RuntimeError('23514 completed scores differ')
            elif row['status'] == 'running':
                row['status'] = 'completed'
                if task: linked['completion'] = fingerprint
            else:
                raise RuntimeError('55000 failed official run')
        elif task and key in self.links:
            linked = self.links[key]
            if linked['user_id'] != params['p_user_id'] or linked['request'] != fingerprint:
                raise RuntimeError('42501 owner or configuration differs')
            if params.get('p_expected_run_id') and params['p_expected_run_id'] != linked['run_id']:
                raise RuntimeError('42501 reserved ID differs')
            row = self.rows[linked['run_id']]
            if row.get('status') == 'failed':
                raise RuntimeError('55000 saved official run is failed and cannot resume')
        else:
            if params.get('p_expected_run_id'):
                raise RuntimeError('P0002 saved reservation missing')
            self.creations += 1
            run_id = str(uuid4())
            row = {'id': run_id, 'user_id': params['p_user_id'], 'competition_slug': params['p_competition_slug']}
            if not is_baseline: row['status'] = 'running'
            self.rows[run_id] = row
            if task:
                self.links[key] = {'task_id': task, 'stage': stage, 'run_id': run_id, 'user_id': params['p_user_id'], 'competition_slug': params['p_competition_slug'], 'request': fingerprint}
        if self.lose_response == name:
            self.lose_response = None
            raise TimeoutError('response lost after transaction commit')
        return SimpleNamespace(data=copy.deepcopy(row))

class OfficialCheckpointStorageTests(unittest.TestCase):
    def setUp(self):
        self.service = AtomicService()
        self.client = patch.object(storage, '_admin_client', return_value=self.service)
        self.profile = patch.object(storage, '_upsert_profile')
        self.client.start(); self.profile.start()
        self.addCleanup(self.client.stop); self.addCleanup(self.profile.stop)
        self.baseline = StageOneResult('answer', {'rouge_l': .5}, .7, .9)
        self.result = StageTwoResult(CONFIG, [ScoredGeneration(1, 1, 'mutated prompt', 'answer', {'rouge_l': .5}, strategy='Baseline')])

    def save(self, participant=PLAYER, prompt='prompt'):
        return storage.save_stage_one(participant, self.baseline, prompt=prompt, reference_text='reference')

    def test_no_journal_keeps_original_rpcs_and_separate_baseline_runs(self):
        with patch.object(storage, '_checkpoint_task_id', return_value=None):
            first, second = self.save(), self.save()
            run_id = storage.begin_stage_two(PLAYER, CONFIG)
            storage.complete_stage_two(run_id, PLAYER, self.result)
        self.assertNotEqual(first['id'], second['id'])
        self.assertEqual([call[0] for call in self.service.calls], ['save_copyright_game_stage_one_run'] * 2 + ['begin_copyright_game_run', 'complete_copyright_game_run'])
        self.assertTrue(all('p_task_id' not in params for _, params in self.service.calls))

    def test_baseline_rpc_commit_then_response_loss_reuses_the_same_saved_run(self):
        self.service.lose_response = 'save_copyright_game_stage_one_checkpoint'
        with patch.object(storage, '_checkpoint_task_id', return_value=TASK):
            with self.assertRaises(storage.GameStorageError): self.save()
            restored = self.save()
            again = self.save()
        self.assertEqual(restored['id'], again['id'])
        self.assertEqual(self.service.creations, 1)
        self.assertEqual(restored['attempts'][0]['rouge_l'], .5)
        self.assertTrue(all(name == 'save_copyright_game_stage_one_checkpoint' and params['p_task_id'] == TASK for name, params in self.service.calls))

    def test_existing_task_rejects_changed_owner_or_payload_without_new_record(self):
        with patch.object(storage, '_checkpoint_task_id', return_value=TASK):
            self.save()
            other = storage.VerifiedParticipant(str(uuid4()), 'other', 'Other', '')
            with self.assertRaises(storage.GameStorageError): self.save(other)
            with self.assertRaises(storage.GameStorageError): self.save(prompt='changed prompt')
        self.assertEqual(self.service.creations, 1)

    def test_reservation_commit_then_response_loss_reuses_same_run_and_checks_expected_id(self):
        self.service.lose_response = 'begin_copyright_game_run_checkpoint'
        with patch.object(storage, '_checkpoint_task_id', return_value=TASK):
            with self.assertRaises(storage.GameStorageError): storage.begin_stage_two(PLAYER, CONFIG)
            run_id = storage.begin_stage_two(PLAYER, CONFIG)
            row = storage.get_stage_two_checkpoint_run(PLAYER, CONFIG, run_id)
            with self.assertRaises(storage.GameStorageError): storage.get_stage_two_checkpoint_run(PLAYER, CONFIG, str(uuid4()))
        self.assertEqual(row['id'], run_id)
        self.assertEqual(row['status'], 'running')
        self.assertEqual(self.service.creations, 1)
        self.assertEqual(storage.get_stage_two_checkpoint_task(run_id, PLAYER), TASK)
        other = storage.VerifiedParticipant(str(uuid4()), 'other', 'Other', '')
        self.assertIsNone(storage.get_stage_two_checkpoint_task(run_id, other))

    def test_final_rpc_commit_then_response_loss_replays_without_new_leaderboard_record(self):
        with patch.object(storage, '_checkpoint_task_id', return_value=TASK):
            run_id = storage.begin_stage_two(PLAYER, CONFIG)
            self.service.lose_response = 'complete_copyright_game_run_checkpoint'
            with self.assertRaises(storage.GameStorageError): storage.complete_stage_two(run_id, PLAYER, self.result)
            restored = storage.complete_stage_two(run_id, PLAYER, self.result)
            row = storage.get_stage_two_checkpoint_run(PLAYER, CONFIG, run_id)
        self.assertEqual(restored['status'], 'completed')
        self.assertEqual(row['id'], run_id)
        self.assertEqual(self.service.creations, 1)

    def test_failed_official_run_is_not_reopened_or_replaced(self):
        with patch.object(storage, '_checkpoint_task_id', return_value=TASK):
            run_id = storage.begin_stage_two(PLAYER, CONFIG)
            self.service.rows[run_id]['status'] = 'failed'
            with self.assertRaisesRegex(storage.GameStorageError, 'failed'):
                storage.get_stage_two_checkpoint_run(PLAYER, CONFIG, run_id)
        self.assertEqual(self.service.creations, 1)
        self.assertEqual(self.service.rows[run_id]['status'], 'failed')

    def test_missing_migration_is_explicit_and_does_not_fallback_to_legacy_insert(self):
        client = Mock()
        client.rpc.return_value.execute.side_effect = RuntimeError('PGRST202 could not find the function')
        with patch.object(storage, '_admin_client', return_value=client), patch.object(storage, '_checkpoint_task_id', return_value=TASK), self.assertRaisesRegex(storage.GameStorageError, 'analysis_game_recovery.sql'):
            self.save()
        self.assertEqual(client.rpc.call_count, 1)
        self.assertEqual(client.rpc.call_args.args[0], 'save_copyright_game_stage_one_checkpoint')

class OfficialCheckpointRunnerTests(unittest.TestCase):
    def execute(self, task_id, status, invoke):
        with patch.object(journal, 'current_task_id', return_value=task_id), \
             patch.object(page, '_reserve_checkpoint_stage_two', return_value='saved-run'), \
             patch.object(page, 'get_stage_two_checkpoint_run', return_value={'id': 'saved-run', 'status': status}), \
             patch.object(page, 'complete_stage_two') as complete, \
             patch.object(page, '_mark_run_failed_safely') as fail:
            try:
                page._execute_checkpoint_stage_two(PLAYER, CONFIG, invoke, Mock())
            except Exception as exc:
                return complete, fail, exc
            return complete, fail, None

    def test_completed_official_entry_is_read_only_and_skips_model_and_save(self):
        invoke = Mock()
        complete, fail, error = self.execute(TASK, 'completed', invoke)
        self.assertIsNone(error)
        invoke.assert_not_called(); complete.assert_not_called(); fail.assert_not_called()

    def test_official_completed_recovery_closes_saved_journal_without_replaying_models(self):
        store = Mock()
        task = {'id': TASK, 'status': 'incomplete', 'lease_token': str(uuid4())}
        store.claim.return_value = task
        store.heartbeat.return_value = task
        store.load_items.return_value = []
        store.save_item.return_value = task
        original = journal.CallJournal(store, task, heartbeat=False)
        with journal.journal_scope(original), patch.object(page, 'begin_stage_two', return_value='saved-run'):
            page._reserve_checkpoint_stage_two(PLAYER, CONFIG)
            journal.checkpoint_call('llm.completion', {'prompt': 'frozen prompt'}, lambda: 'saved answer')
        original.finish(False)
        store.load_items.return_value = copy.deepcopy(original.items)
        resumed = journal.CallJournal(store, task, heartbeat=False)
        invoke, complete = Mock(), Mock()
        with journal.journal_scope(resumed), \
             patch.object(page, 'get_stage_two_checkpoint_run', return_value={'id': 'saved-run', 'status': 'completed'}), \
             patch.object(page, 'complete_stage_two', complete):
            page._execute_checkpoint_stage_two(PLAYER, CONFIG, invoke, Mock())
        self.assertEqual(resumed.index, 1)
        self.assertEqual(len(resumed.items), 2)
        resumed.finish(True)
        self.assertEqual(store.release.call_args.kwargs['status'], 'complete')
        invoke.assert_not_called(); complete.assert_not_called()

    def test_official_completed_does_not_close_a_journal_with_missing_api_result(self):
        store = Mock()
        task = {'id': TASK, 'status': 'incomplete', 'lease_token': str(uuid4())}
        store.claim.return_value = task
        store.heartbeat.return_value = task
        store.load_items.return_value = []
        store.save_item.return_value = task
        original = journal.CallJournal(store, task, heartbeat=False)
        with journal.journal_scope(original), patch.object(page, 'begin_stage_two', return_value='saved-run'):
            page._reserve_checkpoint_stage_two(PLAYER, CONFIG)
            journal.checkpoint_call('llm.completion', {'prompt': 'frozen prompt'}, lambda: 'Error: failed')
        original.finish(False)
        store.load_items.return_value = copy.deepcopy(original.items)
        resumed = journal.CallJournal(store, task, heartbeat=False)
        invoke = Mock()
        with journal.journal_scope(resumed), \
             patch.object(page, 'get_stage_two_checkpoint_run', return_value={'id': 'saved-run', 'status': 'completed'}), \
             self.assertRaises(journal.AnalysisCheckpointError):
            page._execute_checkpoint_stage_two(PLAYER, CONFIG, invoke, Mock())
        resumed.finish(False)
        self.assertEqual(store.release.call_args.kwargs['status'], 'incomplete')
        invoke.assert_not_called()

    def test_failed_official_entry_rejects_resume_before_paid_call(self):
        invoke = Mock()
        complete, fail, error = self.execute(TASK, 'failed', invoke)
        self.assertIsInstance(error, storage.GameStorageError)
        invoke.assert_not_called(); complete.assert_not_called(); fail.assert_not_called()

    def test_temporary_failure_keeps_durable_reservation_but_legacy_run_still_fails(self):
        for task_id in (TASK, None):
            with self.subTest(task_id=task_id):
                invoke = Mock(side_effect=TimeoutError())
                complete, fail, error = self.execute(task_id, 'running', invoke)
                self.assertIsInstance(error, TimeoutError)
                self.assertEqual(invoke.call_count, 1)
                complete.assert_not_called()
                self.assertEqual(fail.call_count, 0 if task_id else 1)

    def test_reservation_journal_payload_freezes_full_config_and_owner(self):
        captured = []
        def checkpoint(operation, payload, invoke, **kwargs):
            captured.append((operation, payload, kwargs))
            return invoke()
        with patch.object(journal, 'checkpoint_call', side_effect=checkpoint), patch.object(page, 'begin_stage_two', return_value='saved-run'):
            self.assertEqual(page._reserve_checkpoint_stage_two(PLAYER, CONFIG), 'saved-run')
        operation, payload, kwargs = captured[0]
        self.assertEqual(operation, 'game.stage_two.reserve')
        self.assertEqual(payload['owner_id'], OWNER)
        self.assertEqual(payload['config']['attempts_per_prompt'], 1)
        self.assertTrue(kwargs['is_success']('saved-run'))
        self.assertFalse(kwargs['is_success'](''))

APP = '''
import streamlit as st
from unittest.mock import patch
from src.pages import copyright_game as page
from src.game.storage import VerifiedParticipant
participant = VerifiedParticipant('22222222-2222-4222-8222-222222222222', 'tester', 'Tester', '')

def mark(*args):
    st.session_state['marked_failed'] = True
with patch.object(page, 'backend_configured', return_value=True), \
     patch.object(page, 'is_logged_in', return_value=True), \
     patch.object(page, 'verify_participant', return_value=participant), \
     patch.object(page, 'get_stage_one', return_value=None), \
     patch.object(page, 'list_stage_one_runs', return_value=[]), \
     patch.object(page, 'get_completed_run', return_value=None), \
     patch.object(page, 'list_completed_runs', return_value=[]), \
     patch.object(page, 'get_active_run', return_value={'id': 'saved-run', 'status': 'running'}), \
     patch.object(page, '_render_stage_navigation', return_value=('stage_two', 'test_view')), \
     patch.object(page, 'get_stage_two_checkpoint_task', return_value='11111111-1111-4111-8111-111111111111'), \
     patch.object(page, 'background_job_running', return_value=False), \
     patch.object(page, '_mark_run_failed_safely', side_effect=mark), \
     patch.object(page, 'list_game_strategies', return_value=['Baseline']), \
     patch.object(page, 'shared_api_key_configured', return_value=False), \
     patch.object(page, 'render_background_job_status', return_value=None):
    page._render_play_tab({'is_open': True}, st.empty())
'''

class OfficialCheckpointPageTests(unittest.TestCase):
    def test_linked_running_reservation_survives_plain_reload_and_other_task_restore(self):
        for resume in (None, 'another-task'):
            with self.subTest(resume=resume):
                app = AppTest.from_string(APP, default_timeout=60)
                if resume: app.session_state[journal.RESUME_TASK] = resume
                app.run()
                self.assertEqual(len(app.exception), 0)
                self.assertNotIn('marked_failed', app.session_state)
                self.assertTrue(any('Restore and continue' in item.value for item in app.info))
                self.assertFalse(any(item.key and item.key.endswith('run_stage_two') for item in app.button))

    def test_correct_armed_resume_reaches_original_stage_two_run_action(self):
        app = AppTest.from_string(APP, default_timeout=60)
        app.session_state[journal.RESUME_TASK] = TASK
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertNotIn('marked_failed', app.session_state)
        self.assertTrue(any(item.key and item.key.endswith('run_stage_two') for item in app.button))

if __name__ == '__main__':
    unittest.main()
