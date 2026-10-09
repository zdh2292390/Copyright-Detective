"""Optional real PostgreSQL tests for atomic official-game recovery.

GAME_RECOVERY_TEST_PSQL/SOCKET/DB must point to an isolated local test database
whose name starts with analysis_game_test. No model or Supabase API is called.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
from threading import Barrier
import unittest
from uuid import uuid4


def literal(value):
    if value is None:
        return 'null'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (dict, list)):
        return "'" + json.dumps(value).replace("'", "''") + "'::jsonb"
    return "'" + str(value).replace("'", "''") + "'"


class PgClient:
    def __init__(self, role='service_role'):
        self.role = role
    def run(self, sql, privileged=False):
        prefix = '' if privileged else 'set role ' + self.role + '; '
        command = [os.environ['GAME_RECOVERY_TEST_PSQL'], '-X', '-q', '-A', '-t', '-v', 'ON_ERROR_STOP=1',
            '-h', os.environ['GAME_RECOVERY_TEST_SOCKET'], '-p', os.environ.get('GAME_RECOVERY_TEST_PORT', '55439'),
            '-U', 'postgres', '-d', os.environ['GAME_RECOVERY_TEST_DB']]
        result = subprocess.run(command, input=prefix + sql, text=True, capture_output=True, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr)
        output = result.stdout.strip()
        return json.loads(output) if output else None
    def rpc(self, name, params):
        arguments = []
        for key, value in params.items():
            encoded = 'array[' + ','.join(map(literal, value)) + ']::text[]' if key == 'p_book_keys' else literal(value)
            arguments.append(key + ' => ' + encoded)
        return self.run('select public.' + name + '(' + ','.join(arguments) + ');')


@unittest.skipUnless(all(os.environ.get(key) for key in ('GAME_RECOVERY_TEST_PSQL', 'GAME_RECOVERY_TEST_SOCKET', 'GAME_RECOVERY_TEST_DB')),
    'isolated official-game PostgreSQL database is not configured')
class GameRecoveryPostgresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ['GAME_RECOVERY_TEST_DB'].startswith('analysis_game_test'):
            raise RuntimeError('Only an isolated analysis_game_test database is allowed.')
        cls.admin = PgClient()
        cls.admin.run("""
            create schema if not exists auth;
            create table if not exists auth.users(id uuid primary key);
            do $$ begin
                if not exists(select 1 from pg_roles where rolname='anon') then create role anon nologin; end if;
                if not exists(select 1 from pg_roles where rolname='authenticated') then create role authenticated nologin; end if;
                if not exists(select 1 from pg_roles where rolname='service_role') then create role service_role nologin bypassrls; end if;
            end $$;
            grant usage on schema public,auth to anon,authenticated,service_role;
        """, privileged=True)
        migrations = Path(__file__).resolve().parents[1] / 'supabase'
        cls.admin.run((migrations / 'copyright_game.sql').read_text(), privileged=True)
        recovery = (migrations / 'analysis_game_recovery.sql').read_text()
        cls.admin.run(recovery, privileged=True)
        cls.admin.run(recovery, privileged=True)

    def setUp(self):
        self.owner, self.other, self.task = map(str, (uuid4(), uuid4(), uuid4()))
        self.client = PgClient()
        self.admin.run(f"insert into auth.users values({literal(self.owner)}),({literal(self.other)}); "
            f"insert into public.copyright_game_profiles(user_id,github_login,display_name) values"
            f"({literal(self.owner)},'owner','Owner'),({literal(self.other)},'other','Other'); "
            "update public.copyright_game_competitions set is_open=true where slug='hp-first-100-gpt-4o-mini-v1';", privileged=True)
        self.base = {'p_task_id': self.task, 'p_competition_slug': 'hp-first-100-gpt-4o-mini-v1', 'p_user_id': self.owner}
        self.config = {**self.base, 'p_shot_mode': 'zero_shot', 'p_strategy': 'Baseline', 'p_attempts_per_strategy': 1,
            'p_attempts_per_prompt': 1, 'p_temperature': .7, 'p_top_p': .9, 'p_book_key': 'harry_potter', 'p_book_keys': ['harry_potter']}

    def reserve(self, params=None):
        return self.client.rpc('begin_copyright_game_run_checkpoint', params or self.config)

    def attempts(self, score=.5):
        return [{'mutation_attempt': 1, 'prompt_attempt': 1, 'book_key': 'harry_potter', 'rouge_l': score,
            'mutated_prompt': 'prompt', 'response_text': 'answer', 'metrics': {'rouge_l': score}, 'trace': {},
            'mutated_prompt_sha256': hashlib.sha256(b'prompt').hexdigest(), 'response_sha256': hashlib.sha256(b'answer').hexdigest()}]

    def complete(self, run_id, attempts=None):
        return self.client.rpc('complete_copyright_game_run_checkpoint', {**self.base, 'p_run_id': run_id, 'p_attempts': self.attempts() if attempts is None else attempts})

    def test_baseline_commit_replay_is_atomic_and_changed_owner_or_input_is_denied(self):
        args = {**self.base, 'p_book_key': 'harry_potter', 'p_prompt_text': 'prompt', 'p_reference_text': 'reference',
            'p_temperature': .7, 'p_top_p': .9,
            'p_attempts': [{'attempt_number': 1, 'response_text': 'answer', 'metrics': {'rouge_l': .5}, 'rouge_l': .5}]}
        first = self.client.rpc('save_copyright_game_stage_one_checkpoint', args)
        replay = self.client.rpc('save_copyright_game_stage_one_checkpoint', args)
        self.assertEqual(first['id'], replay['id'])
        self.assertEqual(self.client.run('select count(*) from public.copyright_game_stage_one_runs where user_id=' + literal(self.owner)), 1)
        for changed in ({'p_user_id': self.other}, {'p_prompt_text': 'changed'}):
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                self.client.rpc('save_copyright_game_stage_one_checkpoint', {**args, **changed})

    def test_concurrent_reservation_retry_returns_one_run_and_expected_id_is_checked(self):
        barrier = Barrier(2)
        def reserve():
            barrier.wait(timeout=5)
            return PgClient().rpc('begin_copyright_game_run_checkpoint', self.config)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first, replay = list(pool.map(lambda _: reserve(), range(2)))
        self.assertEqual(first['id'], replay['id'])
        self.assertEqual(self.reserve({**self.config, 'p_expected_run_id': first['id']})['id'], first['id'])
        self.assertEqual(self.client.run('select count(*) from public.copyright_game_runs where user_id=' + literal(self.owner)), 1)
        for changed in ({'p_user_id': self.other}, {'p_temperature': .8}, {'p_expected_run_id': str(uuid4())}):
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                self.reserve({**self.config, **changed})

    def test_complete_committed_then_replayed_does_not_duplicate_or_change_scores(self):
        run_id = self.reserve()['id']
        first, replay = self.complete(run_id), self.complete(run_id)
        self.assertEqual(first, replay)
        self.assertEqual(replay['status'], 'completed')
        self.assertEqual(replay['max_rouge_l'], .5)
        self.assertEqual(self.client.run('select count(*) from public.copyright_game_attempts where run_id=' + literal(run_id)), 1)
        with self.assertRaises(RuntimeError):
            self.complete(run_id, self.attempts(.9))
        self.assertEqual(self.reserve()['status'], 'completed')

    def test_failed_official_run_is_not_reopened(self):
        run_id = self.reserve()['id']
        self.client.run("update public.copyright_game_runs set status='failed',failure_code='background_run_error' where id=" + literal(run_id) + '::uuid;')
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            self.reserve()
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            self.complete(run_id)
        self.assertEqual(self.client.run('select to_jsonb(status) from public.copyright_game_runs where id=' + literal(run_id) + '::uuid'), 'failed')

    def test_closed_competition_blocks_running_resume_but_keeps_completed_read_only(self):
        run_id = self.reserve()['id']
        self.admin.run("update public.copyright_game_competitions set is_open=false where slug='hp-first-100-gpt-4o-mini-v1';", privileged=True)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            self.reserve()
        self.admin.run("update public.copyright_game_competitions set is_open=true where slug='hp-first-100-gpt-4o-mini-v1';", privileged=True)
        self.complete(run_id)
        self.admin.run("update public.copyright_game_competitions set is_open=false where slug='hp-first-100-gpt-4o-mini-v1';", privileged=True)
        self.assertEqual(self.reserve()['status'], 'completed')

    def test_unlinked_expected_run_cannot_create_a_new_reservation(self):
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            self.reserve({**self.config, 'p_expected_run_id': str(uuid4())})
        self.assertEqual(self.client.run('select count(*) from public.copyright_game_runs where user_id=' + literal(self.owner)), 0)

    def test_browser_roles_cannot_read_links_or_call_recovery_rpcs(self):
        self.reserve()
        for role in ('anon', 'authenticated'):
            with self.subTest(role=role):
                attacker = PgClient(role)
                with self.assertRaisesRegex(RuntimeError, 'permission denied'):
                    attacker.run('select count(*) from public.analysis_game_task_links;')
                with self.assertRaisesRegex(RuntimeError, 'permission denied'):
                    attacker.rpc('begin_copyright_game_run_checkpoint', self.config)


if __name__ == '__main__':
    unittest.main()
