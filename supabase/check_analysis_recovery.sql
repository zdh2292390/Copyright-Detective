-- Read-only diagnostics. Run in the same project used by SUPABASE_URL.
-- Only PostgreSQL catalogs are read; no task content or credentials are selected.
-- Functional checks below should be true. If a column/RPC/FK is missing or wrong,
-- rerun analysis_checkpoints.sql (or the incremental analysis_task_lifecycle.sql),
-- then apply analysis_saved_details.sql if the recent-item index is missing.

-- Both tables need RLS and SELECT for browser reads and service-role workers.
-- The policy expression is included for review: it should scope auth.uid() to owner_id.
with expected(table_name, policy_name) as (values
    ('analysis_tasks', 'analysis_tasks_owner_read'),
    ('analysis_items', 'analysis_items_owner_read')
), roles as (
    select (select oid from pg_roles where rolname = 'authenticated') as authenticated_oid,
           (select oid from pg_roles where rolname = 'service_role') as service_role_oid
)
select expected.table_name,
       coalesce(c.relkind in ('r', 'p'), false) as table_exists,
       coalesce(c.relrowsecurity, false) as rls_enabled,
       coalesce(has_table_privilege(roles.authenticated_oid, c.oid, 'SELECT'), false) as authenticated_can_read,
       coalesce(has_table_privilege(roles.service_role_oid, c.oid, 'SELECT'), false) as service_role_can_read,
       coalesce(p.polcmd = 'r' and p.polpermissive
           and roles.authenticated_oid = any(p.polroles), false) as owner_read_policy_present,
       pg_get_expr(p.polqual, p.polrelid) as owner_read_policy_expression
from expected
cross join roles
left join pg_class c on c.oid = to_regclass('public.' || expected.table_name)
left join pg_policy p on p.polrelid = c.oid and p.polname = expected.policy_name
order by expected.table_name;

-- Every column selected by _TASK_COLUMNS, _ITEM_COLUMNS and the compact
-- Saved Analysis summary must exist with the expected type and nullability.
-- Display details live in metadata->display; latest item API/step details use input.
with expected(table_name, column_name, expected_type, expected_not_null) as (values
    ('analysis_tasks', 'id', 'uuid', true),
    ('analysis_tasks', 'owner_id', 'uuid', true),
    ('analysis_tasks', 'page_key', 'text', true),
    ('analysis_tasks', 'fingerprint', 'text', true),
    ('analysis_tasks', 'settings', 'jsonb', true),
    ('analysis_tasks', 'source', 'jsonb', true),
    ('analysis_tasks', 'metadata', 'jsonb', true),
    ('analysis_tasks', 'dynamic_items', 'boolean', true),
    ('analysis_tasks', 'server_managed', 'boolean', true),
    ('analysis_tasks', 'total_items', 'integer', true),
    ('analysis_tasks', 'status', 'text', true),
    ('analysis_tasks', 'completed_items', 'integer', true),
    ('analysis_tasks', 'failed_items', 'integer', true),
    ('analysis_tasks', 'stop_requested', 'boolean', true),
    ('analysis_tasks', 'kill_requested', 'boolean', true),
    ('analysis_tasks', 'kill_requested_at', 'timestamp with time zone', false),
    ('analysis_tasks', 'killed_at', 'timestamp with time zone', false),
    ('analysis_tasks', 'started_at', 'timestamp with time zone', false),
    ('analysis_tasks', 'completed_at', 'timestamp with time zone', false),
    ('analysis_tasks', 'lease_token', 'uuid', false),
    ('analysis_tasks', 'lease_generation', 'bigint', true),
    ('analysis_tasks', 'lease_expires_at', 'timestamp with time zone', false),
    ('analysis_tasks', 'revision', 'bigint', true),
    ('analysis_tasks', 'created_at', 'timestamp with time zone', true),
    ('analysis_tasks', 'updated_at', 'timestamp with time zone', true),
    ('analysis_items', 'task_id', 'uuid', true),
    ('analysis_items', 'owner_id', 'uuid', true),
    ('analysis_items', 'item_index', 'integer', true),
    ('analysis_items', 'input', 'jsonb', true),
    ('analysis_items', 'status', 'text', true),
    ('analysis_items', 'result', 'jsonb', false),
    ('analysis_items', 'error', 'text', false),
    ('analysis_items', 'attempts', 'integer', true),
    ('analysis_items', 'updated_at', 'timestamp with time zone', true)
)
select expected.table_name, expected.column_name, expected.expected_type,
       a.attnum is not null as column_exists,
       coalesce(a.atttypid = to_regtype(expected.expected_type), false) as type_matches,
       coalesce(a.attnotnull = expected.expected_not_null, false) as nullability_matches
from expected
left join pg_class c on c.oid = to_regclass('public.' || expected.table_name)
left join pg_attribute a on a.attrelid = c.oid and a.attname = expected.column_name
    and a.attnum > 0 and not a.attisdropped
order by expected.table_name, expected.column_name;

-- PostgREST needs this owner-scoped relationship for latest_items:analysis_items(...).
-- Match columns rather than a generated constraint name. A validated, cascading
-- composite FK also prevents cross-owner items and preserves task-delete behavior.
with matching_fk as (
    select con.convalidated, con.confdeltype
    from pg_constraint con
    where con.contype = 'f'
      and con.conrelid = to_regclass('public.analysis_items')
      and con.confrelid = to_regclass('public.analysis_tasks')
      and array(
          select a.attname::text
          from unnest(con.conkey) with ordinality as key_column(attnum, position)
          join pg_attribute a on a.attrelid = con.conrelid and a.attnum = key_column.attnum
          order by key_column.position
      ) = array['task_id', 'owner_id']::text[]
      and array(
          select a.attname::text
          from unnest(con.confkey) with ordinality as key_column(attnum, position)
          join pg_attribute a on a.attrelid = con.confrelid and a.attnum = key_column.attnum
          order by key_column.position
      ) = array['id', 'owner_id']::text[]
)
select count(*) > 0 as owner_task_foreign_key_exists,
       coalesce(bool_or(convalidated and confdeltype = 'c'), false) as validated_cascading_relationship_ready
from matching_fk;

-- Performance check, not a functional prerequisite: the recent-item index lets
-- each Saved Analysis summary retrieve five latest items without sorting a task's
-- whole journal. A missing index does not change results or Kill/Resume behavior.
-- analysis_saved_details.sql adds this index; the base migration also includes it.
select i.indexrelid is not null as latest_items_index_exists,
       coalesce(i.indisvalid and i.indisready and am.amname = 'btree'
           and i.indrelid = to_regclass('public.analysis_items')
           and i.indnkeyatts = 4 and i.indnatts = 4 and i.indpred is null
           and array(
               select a.attname::text
               from unnest(i.indkey) with ordinality as key_column(attnum, position)
               join pg_attribute a on a.attrelid = i.indrelid and a.attnum = key_column.attnum
               order by key_column.position
           ) = array['task_id', 'owner_id', 'updated_at', 'item_index']::text[]
           and (i.indoption[0] & 1) = 0 and (i.indoption[1] & 1) = 0
           and (i.indoption[2] & 1) = 1 and (i.indoption[3] & 1) = 1,
           false) as latest_items_index_ready
from (select to_regclass('public.analysis_items_task_updated_idx') as oid) expected
left join pg_index i on i.indexrelid = expected.oid
left join pg_class c on c.oid = i.indexrelid
left join pg_am am on am.oid = c.relam;

-- The status constraint must allow terminal Kill while retaining every existing
-- resumable lifecycle state. This reads only the constraint, never task contents.
with expected as (
    select array['complete','creating','incomplete','killed','queued','running','stopping']::text[] as statuses
), actual as (
    select c.oid, c.convalidated,
           array(select distinct m[1]
                 from regexp_matches(pg_get_expr(c.conbin, c.conrelid), '''([^'']+)''::text', 'g') m
                 order by m[1]) as statuses
    from pg_constraint c
    where c.conrelid = to_regclass('public.analysis_tasks')
      and c.conname = 'analysis_tasks_status_check' and c.contype = 'c'
)
select actual.oid is not null as lifecycle_status_constraint_exists,
       coalesce(actual.convalidated and actual.statuses = expected.statuses, false) as lifecycle_status_constraint_ready
from expected left join actual on true;

-- Terminal Kill flags, timestamps and leases must agree. Review the catalog
-- expression below as well as these key predicates; no task rows are inspected.
with expected as (
    select to_regclass('public.analysis_tasks') as task_table
), actual as (
    select c.oid, c.convalidated, pg_get_expr(c.conbin, c.conrelid) as expression
    from pg_constraint c
    where c.conrelid = to_regclass('public.analysis_tasks')
      and c.conname = 'analysis_tasks_kill_lifecycle_check' and c.contype = 'c'
)
select actual.oid is not null as kill_lifecycle_constraint_exists,
       coalesce(actual.convalidated
           and actual.expression like '%NOT kill_requested%'
           and actual.expression like '%kill_requested AND stop_requested%'
           and actual.expression like '%kill_requested_at IS NULL%'
           and actual.expression like '%kill_requested_at IS NOT NULL%'
           and actual.expression like '%killed_at IS NULL%'
           and actual.expression like '%killed_at IS NOT NULL%'
           and actual.expression like '%lease_token IS NULL%'
           and actual.expression like '%lease_token IS NOT NULL%'
           and actual.expression like '%lease_expires_at IS NULL%'
           and actual.expression like '%lease_expires_at IS NOT NULL%'
           and actual.expression like '%''stopping''::text%'
           and actual.expression like '%''killed''::text%', false) as kill_lifecycle_constraint_ready,
       actual.expression as kill_lifecycle_constraint_expression
from expected left join actual on true;

-- Cleanup never scans unrelated tasks or revokes an active worker's lease.
select i.indexrelid is not null as kill_cleanup_index_exists,
       coalesce(i.indisvalid and i.indisready and am.amname = 'btree'
           and i.indrelid = to_regclass('public.analysis_tasks')
           and i.indnkeyatts = 2 and i.indnatts = 2
           and array(
               select a.attname::text
               from unnest(i.indkey) with ordinality as key_column(attnum, position)
               join pg_attribute a on a.attrelid = i.indrelid and a.attnum = key_column.attnum
               order by key_column.position
           ) = array['owner_id', 'lease_expires_at']::text[]
           and pg_get_expr(i.indpred, i.indrelid) = '(kill_requested AND (status = ''stopping''::text))',
           false) as kill_cleanup_index_ready
from (select to_regclass('public.analysis_tasks_kill_cleanup_idx') as oid) expected
left join pg_index i on i.indexrelid = expected.oid
left join pg_class c on c.oid = i.indexrelid
left join pg_am am on am.oid = c.relam;

-- All twelve lifecycle RPCs must have these exact argument types and names, because
-- the Python store invokes PostgREST with named p_* JSON arguments. Matching a
-- function name alone can hide a stale overload, missing argument, or wrong type.
-- Stop uses analysis_request_stop; terminal Kill uses analysis_kill_task.
-- Reconciliation returns an integer count; all other lifecycle RPCs return jsonb.
with expected(name, signature, parameter_names) as (values
    ('analysis_create_task',
     'public.analysis_create_task(uuid,uuid,text,jsonb,jsonb,integer,text,jsonb,boolean)',
     array['p_owner_id','p_task_id','p_page_key','p_settings','p_source','p_total_items','p_fingerprint','p_metadata','p_dynamic_items']::text[]),
    ('analysis_put_items', 'public.analysis_put_items(uuid,uuid,jsonb)',
     array['p_owner_id','p_task_id','p_items']::text[]),
    ('analysis_finalize_task', 'public.analysis_finalize_task(uuid,uuid)',
     array['p_owner_id','p_task_id']::text[]),
    ('analysis_claim', 'public.analysis_claim(uuid,uuid,uuid,integer)',
     array['p_owner_id','p_task_id','p_lease_token','p_ttl_seconds']::text[]),
    ('analysis_heartbeat', 'public.analysis_heartbeat(uuid,uuid,uuid,integer)',
     array['p_owner_id','p_task_id','p_lease_token','p_ttl_seconds']::text[]),
    ('analysis_append_item', 'public.analysis_append_item(uuid,uuid,integer,jsonb,uuid)',
     array['p_owner_id','p_task_id','p_index','p_input','p_lease_token']::text[]),
    ('analysis_save_items', 'public.analysis_save_items(uuid,uuid,jsonb,jsonb,text,uuid)',
     array['p_owner_id','p_task_id','p_updates','p_metadata','p_status','p_lease_token']::text[]),
    ('analysis_release', 'public.analysis_release(uuid,uuid,uuid,text,jsonb)',
     array['p_owner_id','p_task_id','p_lease_token','p_status','p_metadata']::text[]),
    ('analysis_request_stop', 'public.analysis_request_stop(uuid,uuid)',
     array['p_owner_id','p_task_id']::text[]),
    ('analysis_kill_task', 'public.analysis_kill_task(uuid,uuid)',
     array['p_owner_id','p_task_id']::text[]),
    ('analysis_reconcile_killed_tasks', 'public.analysis_reconcile_killed_tasks(uuid)',
     array['p_owner_id']::text[]),
    ('analysis_delete', 'public.analysis_delete(uuid,uuid)',
     array['p_owner_id','p_task_id']::text[])
), roles as (
    select (select oid from pg_roles where rolname = 'authenticated') as authenticated_oid,
           (select oid from pg_roles where rolname = 'service_role') as service_role_oid
)
select expected.name as rpc, expected.signature as expected_signature,
       p.oid is not null as exact_signature_exists,
       coalesce(p.proargnames = expected.parameter_names and p.proargmodes is null
           and p.pronargs = cardinality(expected.parameter_names), false) as parameter_names_match,
       coalesce(p.prokind = 'f' and p.prorettype = to_regtype(
           case when expected.name = 'analysis_reconcile_killed_tasks' then 'integer' else 'jsonb' end
       ) and not p.proretset, false) as returns_expected_type,
       coalesce(has_function_privilege(roles.authenticated_oid, p.oid, 'EXECUTE'), false) as authenticated_can_execute,
       coalesce(has_function_privilege(roles.service_role_oid, p.oid, 'EXECUTE'), false) as service_role_can_execute,
       coalesce(p.prosecdef, false) as security_definer
from expected
cross join roles
left join pg_proc p on p.oid = to_regprocedure(expected.signature)
order by expected.name;
