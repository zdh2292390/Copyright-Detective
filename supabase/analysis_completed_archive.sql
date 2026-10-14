-- Archive completed analyses while retaining their final results for View.
-- Run after analysis_checkpoints.sql / analysis_task_lifecycle.sql in the same
-- Supabase project as the app. Safe to run again; no stored results are rewritten.
begin;

-- Archive completed analyses without changing results or terminal Kill flags.
-- Historical records stay visible until their owner explicitly archives them.
alter table public.analysis_tasks add column if not exists archived_at timestamptz;
alter table public.analysis_tasks drop constraint if exists analysis_tasks_archive_lifecycle_check;
alter table public.analysis_tasks add constraint analysis_tasks_archive_lifecycle_check
    check (archived_at is null or (
        status = 'complete' and not kill_requested
        and lease_token is null and lease_expires_at is null
    ));
create index if not exists analysis_tasks_owner_archive_updated_idx
    on public.analysis_tasks(owner_id, updated_at desc, id)
    where archived_at is not null;

create or replace function public.analysis_release(
    p_owner_id uuid, p_task_id uuid, p_lease_token uuid, p_status text, p_metadata jsonb
)
returns jsonb language plpgsql security definer set search_path = '' as $$
declare task public.analysis_tasks%rowtype;
begin
    -- Completion can win the lock immediately before the owner archives it.
    -- This stale worker may close its run, but must not rewrite the saved result.
    task := public.analysis_locked_task(p_owner_id, p_task_id, null, false);
    if task.archived_at is not null and task.status = 'complete'
        and p_status in ('incomplete', 'complete') then
        return to_jsonb(task);
    end if;
    task := public.analysis_locked_task(p_owner_id, p_task_id, p_lease_token);
    if p_status is null or p_status not in ('incomplete', 'complete', 'killed')
        or (p_status = 'killed' and not task.kill_requested) then
        raise exception 'Invalid released analysis status.' using errcode = '22023';
    end if;
    task := jsonb_populate_record(null::public.analysis_tasks,
        public.analysis_save_items(p_owner_id, p_task_id, '[]'::jsonb, p_metadata, p_status, p_lease_token));
    update public.analysis_tasks set lease_token = null, lease_expires_at = null,
        status = case when task.kill_requested then 'killed' else task.status end,
        killed_at = case when task.kill_requested then coalesce(task.killed_at, clock_timestamp())
            else task.killed_at end,
        revision = revision + 1, updated_at = clock_timestamp()
        where id = p_task_id and owner_id = p_owner_id returning * into task;
    return to_jsonb(task);
end;
$$;

create or replace function public.analysis_kill_task(p_owner_id uuid, p_task_id uuid)
returns jsonb language plpgsql security definer set search_path = '' as $$
declare
    task public.analysis_tasks%rowtype;
    active_lease boolean;
    transition_time timestamptz := clock_timestamp();
begin
    task := public.analysis_locked_task(p_owner_id, p_task_id, null, false);
    if task.status = 'complete' then
        if task.archived_at is not null then
            return to_jsonb(task);
        end if;
        -- Preserve the final snapshot and the immutable API journal. Clearing
        -- the lease fences any late writer while View remains a read-only action.
        update public.analysis_tasks set archived_at = clock_timestamp(),
            lease_token = null, lease_expires_at = null,
            revision = revision + 1, updated_at = clock_timestamp()
            where id = p_task_id and owner_id = p_owner_id returning * into task;
        return to_jsonb(task);
    end if;
    if task.status = 'killed' then
        return to_jsonb(task);
    end if;
    active_lease := task.lease_token is not null and task.lease_expires_at > transition_time;
    if task.kill_requested and active_lease and task.status = 'stopping' then
        return to_jsonb(task);
    end if;
    update public.analysis_tasks set kill_requested = true, stop_requested = true,
        kill_requested_at = coalesce(task.kill_requested_at, transition_time),
        metadata = jsonb_set(metadata, '{stop_requested}', 'true'::jsonb),
        status = case when active_lease then 'stopping' else 'killed' end,
        killed_at = case when active_lease then task.killed_at else coalesce(task.killed_at, transition_time) end,
        lease_token = case when active_lease then task.lease_token else null end,
        lease_expires_at = case when active_lease then task.lease_expires_at else null end,
        revision = revision + 1, updated_at = transition_time
        where id = p_task_id and owner_id = p_owner_id returning * into task;
    return to_jsonb(task);
end;
$$;

-- No new RPC surface: preserve the existing owner-checked lifecycle grants.
revoke all on function public.analysis_release(uuid,uuid,uuid,text,jsonb)
    from public, anon, authenticated, service_role;
grant execute on function public.analysis_release(uuid,uuid,uuid,text,jsonb)
    to authenticated, service_role;
revoke all on function public.analysis_kill_task(uuid,uuid)
    from public, anon, authenticated, service_role;
grant execute on function public.analysis_kill_task(uuid,uuid)
    to authenticated, service_role;

notify pgrst, 'reload schema';
commit;
