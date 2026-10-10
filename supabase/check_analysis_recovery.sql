-- Read-only diagnostics. Run in the same project used by SUPABASE_URL.
-- Both table rows should show true in all three boolean columns.
with expected(table_name) as (values ('analysis_tasks'), ('analysis_items'))
select expected.table_name,
       c.oid is not null as table_exists,
       coalesce(c.relrowsecurity, false) as rls_enabled,
       coalesce(has_table_privilege('authenticated', c.oid, 'SELECT'), false) as authenticated_can_read
from expected
left join pg_class c on c.oid = to_regclass('public.' || expected.table_name)
order by expected.table_name;

-- This must be true, including on installations upgraded from older versions.
select exists (
    select 1 from information_schema.columns
    where table_schema = 'public' and table_name = 'analysis_tasks'
      and column_name = 'server_managed' and data_type = 'boolean'
) as server_managed_column_exists;

-- All ten RPC rows should show true for exists/grant/security-definer checks.
with expected(name) as (values
    ('analysis_create_task'), ('analysis_put_items'), ('analysis_finalize_task'),
    ('analysis_claim'), ('analysis_heartbeat'), ('analysis_append_item'),
    ('analysis_save_items'), ('analysis_release'), ('analysis_request_stop'), ('analysis_delete')
)
select expected.name as rpc,
       count(p.oid) > 0 as rpc_exists,
       coalesce(bool_or(has_function_privilege('authenticated', p.oid, 'EXECUTE')), false) as authenticated_can_execute,
       coalesce(bool_or(p.prosecdef), false) as security_definer
from expected
left join pg_namespace n on n.nspname = 'public'
left join pg_proc p on p.pronamespace = n.oid and p.proname = expected.name
 group by expected.name
 order by expected.name;
