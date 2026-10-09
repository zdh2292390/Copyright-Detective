"""Store boundary tests plus optional real PostgreSQL RPC/RLS regressions.

For the database tests, point ANALYSIS_CHECKPOINT_TEST_PSQL at a psql binary and
ANALYSIS_CHECKPOINT_TEST_SOCKET at an isolated local PostgreSQL UNIX socket.
The optional suite creates auth fixtures and analysis tables in that test database;
it must never point at a production database. Default tests do not use a network.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from threading import Event
from types import SimpleNamespace
import unittest
from uuid import uuid4

from src.analysis_checkpoints import (
    AnalysisCheckpointError, AnalysisLeaseError, SupabaseAnalysisCheckpointStore,
)


class DatabaseError(Exception):
    def __init__(self, message, code="P0001"):
        super().__init__(message)
        self.message, self.code = message, code


class RecordingQuery:
    def __init__(self, client, table):
        self.client, self.table_name, self.operations = client, table, []
    def __getattr__(self, name):
        def chain(*args, **kwargs):
            self.operations.append((name, args, kwargs))
            return self
        return chain
    def execute(self):
        self.client.queries.append((self.table_name, self.operations))
        return SimpleNamespace(data=[])


class RecordingClient:
    def __init__(self):
        self.calls, self.queries, self.error = [], [], None
    def table(self, name):
        return RecordingQuery(self, name)
    def rpc(self, name, params):
        self.calls.append((name, params))
        def execute():
            if self.error:
                raise self.error
            status = "creating" if name == "analysis_create_task" else "queued"
            return SimpleNamespace(data={"id": params.get("p_task_id"), "status": status})
        return SimpleNamespace(execute=execute)


class StoreBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.client = RecordingClient()
        self.owner, self.task, self.lease = map(str, (uuid4(), uuid4(), uuid4()))
        self.store = SupabaseAnalysisCheckpointStore(self.client, self.owner, batch_size=2)

    def test_creation_batches_are_owner_scoped_and_detached(self):
        settings, items = {"model": "same-model"}, [{"prompt": str(i)} for i in range(5)]
        result = self.store.create_task("document", settings, {"filename": "a.pdf"}, items, task_id=self.task)
        settings["model"] = "changed"
        items[0]["prompt"] = "changed"
        self.assertEqual(result["status"], "queued")
        self.assertEqual([name for name, _ in self.client.calls], [
            "analysis_create_task", "analysis_put_items", "analysis_put_items", "analysis_put_items", "analysis_finalize_task"
        ])
        self.assertTrue(all(params["p_owner_id"] == self.owner for _, params in self.client.calls))
        self.assertEqual(self.client.calls[0][1]["p_settings"]["model"], "same-model")
        self.assertEqual(self.client.calls[1][1]["p_items"][0]["input"]["prompt"], "0")
        self.assertEqual(len(self.client.calls[0][1]["p_fingerprint"]), 64)

    def test_credentials_and_nonfinite_values_fail_before_outbound(self):
        for settings in [
            {"nested": {"api_key": "secret"}}, {"Gemini_API_Key": "secret"}, {"api_keys": {"provider": "secret"}}, {"headers": {"Authorization": "Bearer secret"}},
            {"base_url": "http://user:secret@localhost/v1"},
            {"base_url": "https://host/v1?api_key=secret"}, {"temperature": float("nan")},
        ]:
            with self.subTest(settings=settings):
                with self.assertRaises(AnalysisCheckpointError):
                    self.store.create_task("qa", settings, {}, [])
        self.assertFalse(self.client.calls)

    def test_document_text_is_opaque_even_when_it_quotes_a_url(self):
        payload = {"upper_text": "https://example.test?api_key=public-example", "lower_text": "reference"}
        self.store.create_task("document", {}, {"text": payload["upper_text"]}, [payload])
        self.assertEqual(self.client.calls[1][1]["p_items"][0]["input"], payload)

    def test_all_reads_filter_owner_before_execute_and_use_bounded_paging(self):
        self.assertIsNone(self.store.get_task(self.task))
        self.store.load_items(self.task, offset=200, limit=200)
        self.store.list_tasks(page_key="document", statuses=["incomplete"], offset=50, limit=50)
        for _, operations in self.client.queries:
            self.assertIn(("eq", ("owner_id", self.owner), {}), operations)
        self.assertIn(("range", (200, 399), {}), self.client.queries[1][1])
        self.assertIn(("order", ("item_index",), {}), self.client.queries[1][1])
        with self.assertRaises(AnalysisCheckpointError):
            self.store.load_items(self.task, limit=1001)

    def test_summary_projection_omits_large_inputs_and_results(self):
        self.store.list_tasks(page_key="Content Recall Detection", summary_only=True)
        columns = self.client.queries[0][1][0][1][0].split(",")
        self.assertNotIn("source", columns)
        self.assertNotIn("metadata", columns)
        self.assertNotIn("lease_token", columns)
        self.assertIn("dynamic_items", columns)
        self.assertIn("settings", columns)

    def test_save_item_status_does_not_complete_dynamic_task(self):
        self.store.save_item(self.task, 0, result={"response": "A"}, status="complete", attempts=1, lease_token=self.lease)
        name, params = self.client.calls[-1]
        self.assertEqual(name, "analysis_save_items")
        self.assertEqual(params["p_updates"][0]["status"], "complete")
        self.assertIsNone(params["p_status"])
        self.assertEqual(params["p_lease_token"], self.lease)

    def test_invalid_batch_is_not_partially_sent(self):
        for updates in [
            [{"index": 0, "status": "complete", "result": "ok"}, {"index": 0, "status": "pending"}],
            [{"index": 0, "status": "complete", "result": None}],
            [{"index": True, "status": "pending"}],
            [{"index": 0, "status": "failed", "error": ""}],
            [{"index": 0, "status": "complete", "result": {"access_token": "secret"}}],
        ]:
            with self.subTest(updates=updates):
                with self.assertRaises(AnalysisCheckpointError):
                    self.store.save_items(self.task, updates, lease_token=self.lease)
        self.assertFalse(self.client.calls)

    def test_backend_error_does_not_echo_raw_details_or_credentials(self):
        self.client.error = DatabaseError("authorization Bearer secret-key-details")
        with self.assertRaises(AnalysisCheckpointError) as caught:
            self.store.claim(self.task)
        self.assertNotIn("secret-key", str(caught.exception))
        self.client.error = DatabaseError("analysis_lease: stale")
        with self.assertRaises(AnalysisLeaseError):
            self.store.claim(self.task)
        self.client.error = DatabaseError("missing function", "PGRST202")
        with self.assertRaisesRegex(AnalysisCheckpointError, "analysis_checkpoints.sql"):
            self.store.claim(self.task)

    def test_dynamic_initial_items_and_invalid_lease_fail_before_outbound(self):
        with self.assertRaises(AnalysisCheckpointError):
            self.store.create_task("qa", {}, {}, ["one"], dynamic_items=True)
        for ttl in [59, 901, True, 1.5]:
            with self.assertRaises(AnalysisCheckpointError):
                self.store.claim(self.task, ttl_seconds=ttl)
        self.assertFalse(self.client.calls)


def _literal(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (dict, list)):
        return "'" + json.dumps(value, allow_nan=False).replace("'", "''") + "'::jsonb"
    return "'" + str(value).replace("'", "''") + "'"


class PostgresClient:
    def __init__(self, owner, role="service_role"):
        self.owner, self.role = owner, role
    def run(self, sql, privileged=False):
        prefix = "" if privileged else (
            f"set role {self.role}; set request.jwt.claim.role = {_literal(self.role)}; "
            f"set request.jwt.claim.sub = {_literal(self.owner)}; "
        )
        command = [os.environ["ANALYSIS_CHECKPOINT_TEST_PSQL"], "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1",
                   "-h", os.environ["ANALYSIS_CHECKPOINT_TEST_SOCKET"],
                   "-p", os.environ.get("ANALYSIS_CHECKPOINT_TEST_PORT", "55439"), "-U", "postgres", "-d", "postgres"]
        result = subprocess.run(command, input=prefix + sql, text=True, capture_output=True, timeout=30)
        if result.returncode:
            raise DatabaseError(result.stderr)
        output = result.stdout.strip()
        return json.loads(output) if output else None
    def rpc(self, name, params):
        args = ",".join(f"{key} => {_literal(value)}" for key, value in params.items())
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=self.run(f"select public.{name}({args});")))
    def table(self, name):
        return PostgresQuery(self, name)


class PostgresQuery:
    def __init__(self, client, table):
        self.client, self.table_name = client, table
        self.columns, self.filters, self.orders, self.count, self.offset = "*", [], [], None, 0
    def select(self, columns):
        self.columns = columns
        return self
    def eq(self, key, value):
        self.filters.append(f"{key} = {_literal(value)}")
        return self
    def in_(self, key, values):
        self.filters.append(f"{key} in ({','.join(map(_literal, values))})")
        return self
    def order(self, key, desc=False):
        self.orders.append(key + (" desc" if desc else " asc"))
        return self
    def limit(self, count):
        self.count = count
        return self
    def range(self, first, last):
        self.offset, self.count = first, last - first + 1
        return self
    def execute(self):
        sql = f"select {self.columns} from public.{self.table_name}"
        if self.filters:
            sql += " where " + " and ".join(self.filters)
        if self.orders:
            sql += " order by " + ",".join(self.orders)
        if self.count is not None:
            sql += f" limit {self.count} offset {self.offset}"
        return SimpleNamespace(data=self.client.run(f"select coalesce(jsonb_agg(rows), '[]'::jsonb) from ({sql}) rows;"))


@unittest.skipUnless(os.environ.get("ANALYSIS_CHECKPOINT_TEST_PSQL") and os.environ.get("ANALYSIS_CHECKPOINT_TEST_SOCKET"),
                     "isolated PostgreSQL test socket is not configured")
class PostgresCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.admin = PostgresClient(str(uuid4()))
        cls.admin.run("""
            do $$ begin
                if not exists(select 1 from pg_roles where rolname='anon') then create role anon nologin; end if;
                if not exists(select 1 from pg_roles where rolname='authenticated') then create role authenticated nologin; end if;
                if not exists(select 1 from pg_roles where rolname='service_role') then create role service_role nologin bypassrls; end if;
            end $$;
            create schema if not exists auth;
            create table if not exists auth.users(id uuid primary key);
            create or replace function auth.uid() returns uuid language sql stable as
                $$ select nullif(current_setting('request.jwt.claim.sub', true), '')::uuid $$;
            create or replace function auth.role() returns text language sql stable as
                $$ select current_setting('request.jwt.claim.role', true) $$;
            grant usage on schema public,auth to anon,authenticated,service_role;
            grant execute on function auth.uid(),auth.role() to anon,authenticated,service_role;
        """, privileged=True)
        migration = (Path(__file__).resolve().parents[1] / "supabase" / "analysis_checkpoints.sql").read_text()
        cls.admin.run(migration, privileged=True)
        cls.admin.run(migration, privileged=True)  # migration rerun is safe

    def setUp(self):
        self.owner, self.other, self.task_id = map(str, (uuid4(), uuid4(), uuid4()))
        self.admin.run(f"insert into auth.users values({_literal(self.owner)}),({_literal(self.other)});", privileged=True)
        self.client = PostgresClient(self.owner)
        self.store = SupabaseAnalysisCheckpointStore(self.client, self.owner, batch_size=200)
        self.other_store = SupabaseAnalysisCheckpointStore(PostgresClient(self.other), self.other)

    def create(self, count=3, dynamic=False):
        return self.store.create_task("document" if not dynamic else "journal", {"model": "original-model"},
                                      {"filename": "quoted's document"},
                                      [{"upper_text": str(i), "lower_text": "lower"} for i in range(count)],
                                      task_id=self.task_id, dynamic_items=dynamic)

    def claim(self):
        return self.store.claim(self.task_id)["lease_token"]

    def expire(self):
        self.admin.run(f"update public.analysis_tasks set lease_expires_at=now()-interval '1 second' "
                       f"where id={_literal(self.task_id)};", privileged=True)

    def test_large_creation_paging_owner_scope_and_durable_reads(self):
        task = self.create(510)
        self.assertEqual((task["status"], task["total_items"]), ("queued", 510))
        rows = self.store.load_items(self.task_id, offset=200, limit=200)
        self.assertEqual((rows[0]["item_index"], rows[-1]["item_index"]), (200, 399))
        self.assertIsNone(self.other_store.get_task(self.task_id))
        self.assertEqual(self.other_store.load_items(self.task_id), [])
        lease = self.claim()
        self.store.save_item(self.task_id, 3, result={"response": "saved"}, attempts=1, lease_token=lease)
        self.store.release(self.task_id, lease)
        new_store = SupabaseAnalysisCheckpointStore(PostgresClient(self.owner), self.owner)
        self.assertEqual(new_store.load_items(self.task_id)[3]["result"], {"response": "saved"})
        self.assertEqual(new_store.get_task(self.task_id)["completed_items"], 1)

    def test_atomic_batch_rollback_and_complete_guard(self):
        self.create(2)
        lease = self.claim()
        with self.assertRaises(AnalysisCheckpointError):
            self.store.save_items(self.task_id, [
                {"index": 0, "status": "complete", "result": "first", "attempts": 1},
                {"index": 4, "status": "complete", "result": "invalid", "attempts": 1},
            ], lease_token=lease)
        self.assertEqual(self.store.get_task(self.task_id)["completed_items"], 0)
        self.assertEqual(self.store.load_items(self.task_id)[0]["status"], "pending")
        with self.assertRaises(AnalysisCheckpointError):
            self.store.release(self.task_id, lease, status="complete")
        task = self.store.save_items(self.task_id, [
            {"index": 1, "status": "complete", "result": "second", "attempts": 1},
            {"index": 0, "status": "complete", "result": "first", "attempts": 1},
        ], status="complete", lease_token=lease)
        self.assertEqual(task["completed_items"], 2)
        final = self.store.release(self.task_id, lease, status="complete")
        self.assertIsNone(final["lease_token"])
        self.assertEqual([row["result"] for row in self.store.load_items(self.task_id)], ["first", "second"])

    def test_lease_fencing_busy_expiry_and_immutable_success(self):
        self.create(2)
        first = self.store.claim(self.task_id)
        lease = first["lease_token"]
        self.assertEqual(self.store.claim(self.task_id, lease_token=lease)["lease_generation"], first["lease_generation"])
        with self.assertRaises(AnalysisLeaseError):
            self.store.claim(self.task_id)
        self.store.save_item(self.task_id, 0, result="saved", attempts=1, lease_token=lease)
        self.expire()
        second = self.store.claim(self.task_id)
        self.assertGreater(second["lease_generation"], first["lease_generation"])
        with self.assertRaises(AnalysisLeaseError):
            self.store.heartbeat(self.task_id, lease)
        with self.assertRaises(AnalysisLeaseError):
            self.store.save_item(self.task_id, 1, result="stale", lease_token=lease)
        with self.assertRaises(AnalysisCheckpointError):
            self.store.save_item(self.task_id, 0, result="replaced", lease_token=second["lease_token"])
        self.store.save_item(self.task_id, 0, result="saved", lease_token=second["lease_token"])
        self.assertEqual(self.store.load_items(self.task_id)[0]["attempts"], 1)

    def test_stop_survives_saves_heartbeat_and_idempotent_claim(self):
        self.create(1)
        lease = self.claim()
        self.store.request_stop(self.task_id)
        task = self.store.save_progress(self.task_id, metadata={"stop_requested": False, "note": "drain"}, lease_token=lease)
        self.assertTrue(task["stop_requested"])
        self.assertTrue(task["metadata"]["stop_requested"])
        self.assertTrue(self.store.heartbeat(self.task_id, lease)["stop_requested"])
        self.assertTrue(self.store.claim(self.task_id, lease_token=lease)["stop_requested"])
        self.store.release(self.task_id, lease)
        self.assertFalse(self.store.claim(self.task_id)["stop_requested"])

    def test_dynamic_append_contiguous_resume_and_final_completion(self):
        self.create(0, dynamic=True)
        lease = self.claim()
        with self.assertRaises(AnalysisCheckpointError):
            self.store.append_item(self.task_id, 1, {"request": "gap"}, lease_token=lease)
        self.assertEqual(self.store.append_item(self.task_id, 0, {"request": "first"}, lease_token=lease)["total_items"], 1)
        self.store.save_item(self.task_id, 0, result={"response": "first"}, lease_token=lease)
        self.assertEqual(self.store.get_task(self.task_id)["status"], "running")
        self.store.request_stop(self.task_id)
        with self.assertRaises(AnalysisCheckpointError):
            self.store.append_item(self.task_id, 1, {"request": "after-stop"}, lease_token=lease)
        self.store.release(self.task_id, lease)
        lease = self.claim()
        self.store.append_item(self.task_id, 1, {"request": "second"}, lease_token=lease)
        self.store.save_item(self.task_id, 1, result={"response": "second"}, lease_token=lease)
        self.assertEqual(self.store.release(self.task_id, lease, status="complete")["completed_items"], 2)

    def test_identity_is_immutable_and_partial_upload_can_resume(self):
        fingerprint = "a" * 64
        self.store._rpc("create_task", p_task_id=self.task_id, p_page_key="document", p_settings={"model": "locked"},
                        p_source={}, p_total_items=2, p_fingerprint=fingerprint, p_metadata={}, p_dynamic_items=False)
        self.store._rpc("put_items", p_task_id=self.task_id, p_items=[{"index": 0, "input": "first"}])
        task = self.store.create_task("document", {"model": "locked"}, {}, ["first", "second"],
                                      task_id=self.task_id, fingerprint=fingerprint)
        self.assertEqual(task["status"], "queued")
        with self.assertRaises(AnalysisCheckpointError):
            self.store.create_task("document", {"model": "changed"}, {}, ["first", "second"],
                                   task_id=self.task_id, fingerprint=fingerprint)
        self.assertEqual(self.store.get_task(self.task_id)["settings"]["model"], "locked")

    def test_rls_and_rpc_permissions_block_cross_owner_and_direct_writes(self):
        self.create(1)
        attacker = PostgresClient(self.other, role="authenticated")
        self.assertEqual(attacker.run("select coalesce(jsonb_agg(t),'[]'::jsonb) from public.analysis_tasks t;"), [])
        spoofed = SupabaseAnalysisCheckpointStore(attacker, self.owner)
        with self.assertRaises(AnalysisCheckpointError):
            spoofed.request_stop(self.task_id)
        with self.assertRaises(AnalysisCheckpointError):
            self.other_store.request_stop(self.task_id)
        with self.assertRaises(DatabaseError):
            self.client.run("update public.analysis_tasks set settings='{}'::jsonb;")
        with self.assertRaises(DatabaseError):
            self.client.run(f"select public.analysis_locked_task({_literal(self.owner)},{_literal(self.task_id)},null,false);")
        with self.assertRaises(DatabaseError):
            PostgresClient(self.owner, role="anon").run(
                f"select public.analysis_request_stop({_literal(self.owner)},{_literal(self.task_id)});"
            )
        self.assertFalse(self.store.get_task(self.task_id)["stop_requested"])

    def test_page_display_name_and_summary_projection(self):
        self.store.create_task("Content Recall Detection", {"label": "run"}, {"initial_session": {"text_source": "body"}},
                               [], task_id=self.task_id, dynamic_items=True)
        summary = self.store.list_tasks(page_key="Content Recall Detection", summary_only=True)[0]
        self.assertEqual(summary["page_key"], "Content Recall Detection")
        self.assertNotIn("source", summary)
        self.assertNotIn("metadata", summary)
        self.assertIn("initial_session", self.store.get_task(self.task_id)["source"])

    def test_racing_claims_only_grant_one_worker(self):
        self.create(1)
        gate = Event()
        def claim():
            gate.wait(2)
            try:
                return self.store.claim(self.task_id)
            except AnalysisLeaseError:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(claim), pool.submit(claim)]
            gate.set()
            results = [future.result(10) for future in futures]
        self.assertEqual(sum(result is not None for result in results), 1)
        winner = next(result for result in results if result)
        self.assertEqual(self.store.get_task(self.task_id)["lease_token"], winner["lease_token"])

    def test_existing_task_rejects_changed_inputs_even_with_caller_fingerprint(self):
        self.store.create_task("qa", {}, {}, ["original"], task_id=self.task_id, fingerprint="c" * 64)
        with self.assertRaisesRegex(AnalysisCheckpointError, "work inputs"):
            self.store.create_task("qa", {}, {}, ["changed"], task_id=self.task_id, fingerprint="c" * 64)
        self.assertEqual(self.store.load_items(self.task_id)[0]["input"], "original")

    def test_sql_secret_guard_and_opaque_document_text(self):
        for settings in [{"nested": {"apiKey": "private"}}, {"Gemini_API_Key": "private"}, {"base_url": "https://user:pass@host"}]:
            with self.assertRaises(AnalysisCheckpointError):
                self.store._rpc("create_task", p_task_id=self.task_id, p_page_key="qa", p_settings=settings,
                                p_source={}, p_total_items=0, p_fingerprint="b" * 64, p_metadata={}, p_dynamic_items=True)
        task = self.store.create_task("document", {}, {"text": "https://example.test?api_key=example"},
                                      [{"upper_text": "https://example.test?api_key=example"}], task_id=self.task_id)
        self.assertEqual(task["total_items"], 1)

    def test_delete_requires_lease_release_and_cascades_items(self):
        self.create(1)
        lease = self.claim()
        with self.assertRaises(AnalysisLeaseError):
            self.store.delete(self.task_id)
        self.store.release(self.task_id, lease)
        self.store.delete(self.task_id)
        self.assertIsNone(self.store.get_task(self.task_id))
        self.assertEqual(self.store.load_items(self.task_id), [])
        self.store.delete(self.task_id)

    def test_failure_counts_retry_and_pending_attempt_rollback(self):
        self.create(2)
        lease = self.claim()
        self.store.save_item(self.task_id, 0, error="temporary", attempts=1, lease_token=lease)
        self.assertEqual(self.store.get_task(self.task_id)["failed_items"], 1)
        self.store.save_item(self.task_id, 0, result="recovered", attempts=2, lease_token=lease)
        self.assertEqual(self.store.get_task(self.task_id)["failed_items"], 0)
        self.store.save_item(self.task_id, 1, attempts=1, lease_token=lease)
        self.store.save_item(self.task_id, 1, attempts=0, lease_token=lease)
        self.assertEqual(self.store.load_items(self.task_id)[1]["attempts"], 0)


if __name__ == "__main__":
    unittest.main()