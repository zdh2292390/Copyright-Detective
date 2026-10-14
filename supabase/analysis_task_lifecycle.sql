-- Terminal Kill and durable lifecycle timestamps for existing analysis databases.
-- Run after analysis_checkpoints.sql, then run check_analysis_recovery.sql.
-- No tasks, results, scores or legacy Stop requests are deleted/reinterpreted.
begin;

-- Preserve legacy Stop semantics; never infer a terminal Kill from stop_requested.
-- Historical timestamps remain null when the original transition time is unknown.
alter table public.analysis_tasks add column if not exists kill_requested boolean not null default false;
alter table public.analysis_tasks add column if not exists kill_requested_at timestamptz;
alter table public.analysis_tasks add column if not exists killed_at timestamptz;
alter table public.analysis_tasks add column if not exists started_at timestamptz;
alter table public.analysis_tasks add column if not exists completed_at timestamptz;
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

alter table public.analysis_tasks drop constraint if exists analysis_tasks_status_check;
alter table public.analysis_tasks add constraint analysis_tasks_status_check
    check (status in ('creating', 'queued', 'running', 'incomplete', 'complete', 'stopping', 'killed'));

-- Terminal transitions retain the active lease while draining and clear it only
-- after finalization. Kill timestamps never appear on an ordinary resumable Stop.
alter table public.analysis_tasks drop constraint if exists analysis_tasks_kill_lifecycle_check;
alter table public.analysis_tasks add constraint analysis_tasks_kill_lifecycle_check
    check (
        (not kill_requested and status not in ('stopping', 'killed')
            and kill_requested_at is null and killed_at is null)
        or (kill_requested and stop_requested and kill_requested_at is not null
            and ((status = 'stopping' and lease_token is not null and lease_expires_at is not null and killed_at is null)
                or (status = 'killed' and lease_token is null and lease_expires_at is null and killed_at is not null)))
    );

-- Crash recovery only examines this owner's terminal kills with an expired lease.
create index if not exists analysis_tasks_kill_cleanup_idx
    on public.analysis_tasks(owner_id, lease_expires_at)
    where kill_requested and status = 'stopping';

create or replace function public.analysis_put_items(p_owner_id uuid, p_task_id uuid, p_items jsonb)
returns jsonb language plpgsql security definer set search_path = '' as $$
declare
    task public.analysis_tasks%rowtype;
    entry jsonb;
    next_index integer;
    old_input jsonb;
begin
    task := public.analysis_locked_task(p_owner_id, p_task_id, null, false);
    perform public.analysis_check_json(p_items);
    if task.kill_requested or task.status <> 'creating' or task.dynamic_items or jsonb_typeof(p_items) <> 'array'
        or jsonb_array_length(p_items) > 1000 then
        raise exception 'Analysis task is not accepting initial work items.' using errcode = '22023';
    end if;
    for entry in select value from jsonb_array_elements(p_items) loop
        if jsonb_typeof(entry) <> 'object' or not(entry ? 'input')
            or coalesce(entry->>'index', '') !~ '^(0|[1-9][0-9]*)$' then
            raise exception 'Invalid initial analysis item.' using errcode = '22023';
        end if;
        next_index := (entry->>'index')::integer;
        if next_index < 0 or next_index >= task.total_items then
            raise exception 'Initial analysis item index out of range.' using errcode = '22023';
        end if;
        insert into public.analysis_items(task_id, owner_id, item_index, input)
        values(p_task_id, p_owner_id, next_index, entry->'input')
        on conflict(task_id, item_index) do nothing;
        select input into old_input from public.analysis_items
            where task_id = p_task_id and owner_id = p_owner_id and analysis_items.item_index = next_index;
        if old_input is distinct from entry->'input' then
            raise exception 'Analysis input cannot be changed.' using errcode = '22023';
        end if;
    end loop;
    update public.analysis_tasks set revision = revision + 1, updated_at = clock_timestamp()
        where id = p_task_id and owner_id = p_owner_id returning * into task;
    return to_jsonb(task);
end;
$$;

create or replace function public.analysis_finalize_task(p_owner_id uuid, p_task_id uuid)
returns jsonb language plpgsql security definer set search_path = '' as $$
declare
    task public.analysis_tasks%rowtype;
    item_count integer;
begin
    task := public.analysis_locked_task(p_owner_id, p_task_id, null, false);
    if task.kill_requested or task.status <> 'creating' then
        return to_jsonb(task);
    end if;
    select count(*) into item_count from public.analysis_items
        where task_id = p_task_id and owner_id = p_owner_id;
    if item_count <> task.total_items then
        raise exception 'Analysis work item upload is incomplete.' using errcode = '22023';
    end if;
    update public.analysis_tasks set status = 'queued', revision = revision + 1, updated_at = clock_timestamp()
        where id = p_task_id and owner_id = p_owner_id returning * into task;
    return to_jsonb(task);
end;
$$;

create or replace function public.analysis_claim(
    p_owner_id uuid, p_task_id uuid, p_lease_token uuid, p_ttl_seconds integer
)
returns jsonb language plpgsql security definer set search_path = '' as $$
declare
    task public.analysis_tasks%rowtype;
    new_generation bigint;
begin
    task := public.analysis_locked_task(p_owner_id, p_task_id, null, false);
    if p_lease_token is null or p_ttl_seconds is null or p_ttl_seconds not between 60 and 900 then
        raise exception 'Invalid analysis lease.' using errcode = '22023';
    end if;
    if task.kill_requested or task.status in ('creating', 'complete', 'stopping', 'killed') then
        raise exception 'analysis_lease: task cannot be claimed in this state.' using errcode = 'P0001';
    end if;
    if task.lease_token is not null and task.lease_expires_at > clock_timestamp()
        and task.lease_token is distinct from p_lease_token then
        raise exception 'analysis_lease: another worker owns this task.' using errcode = 'P0001';
    end if;
    -- Same active token supports retrying a claim whose HTTP response was lost.
    new_generation := task.lease_generation + case when task.lease_token = p_lease_token
        and task.lease_expires_at > clock_timestamp() then 0 else 1 end;
    update public.analysis_tasks set status = 'running', lease_token = p_lease_token,
        lease_generation = new_generation, started_at = coalesce(task.started_at, clock_timestamp()),
        lease_expires_at = clock_timestamp() + make_interval(secs => p_ttl_seconds),
        stop_requested = case when new_generation = task.lease_generation then task.stop_requested else false end,
        metadata = case when new_generation = task.lease_generation then task.metadata
            else jsonb_set(task.metadata, '{stop_requested}', 'false'::jsonb) end,
        revision = revision + 1, updated_at = clock_timestamp()
        where id = p_task_id and owner_id = p_owner_id returning * into task;
    return to_jsonb(task);
end;
$$;

create or replace function public.analysis_append_item(
    p_owner_id uuid, p_task_id uuid, p_index integer, p_input jsonb, p_lease_token uuid
)
returns jsonb language plpgsql security definer set search_path = '' as $$
declare task public.analysis_tasks%rowtype;
begin
    task := public.analysis_locked_task(p_owner_id, p_task_id, p_lease_token);
    perform public.analysis_check_json(p_input);
    if not task.dynamic_items or task.status <> 'running' or task.stop_requested or task.kill_requested
        or p_index is null or p_index <> task.total_items or task.total_items >= 500000 then
        raise exception 'Analysis append must be the next item of an active dynamic task.' using errcode = '22023';
    end if;
    insert into public.analysis_items(task_id, owner_id, item_index, input)
        values(p_task_id, p_owner_id, p_index, p_input);
    update public.analysis_tasks set total_items = total_items + 1,
        revision = revision + 1, updated_at = clock_timestamp()
        where id = p_task_id and owner_id = p_owner_id returning * into task;
    return to_jsonb(task);
end;
$$;

create or replace function public.analysis_save_items(
    p_owner_id uuid, p_task_id uuid, p_updates jsonb, p_metadata jsonb,
    p_status text, p_lease_token uuid
)
returns jsonb language plpgsql security definer set search_path = '' as $$
declare
    task public.analysis_tasks%rowtype;
    entry jsonb;
    old_item public.analysis_items%rowtype;
    next_index integer;
    next_status text;
    next_result jsonb;
    next_error text;
    next_attempts integer;
    completed_count integer;
    failed_count integer;
    actual_count integer;
    actual_complete integer;
    seen integer[] := '{}';
    next_metadata jsonb;
    next_stop boolean;
begin
    task := public.analysis_locked_task(p_owner_id, p_task_id, p_lease_token);
    perform public.analysis_check_json(p_updates);
    if jsonb_typeof(p_updates) <> 'array' or jsonb_array_length(p_updates) > 1000
        or (p_status is not null and p_status not in ('running', 'incomplete', 'complete', 'stopping', 'killed'))
        or (p_status in ('stopping', 'killed') and not task.kill_requested)
        or (task.status = 'complete' and p_status is not null and p_status <> 'complete') then
        raise exception 'Invalid analysis progress update.' using errcode = '22023';
    end if;
    if p_metadata is not null then
        perform public.analysis_check_json(p_metadata);
        if jsonb_typeof(p_metadata) <> 'object' or (
            p_metadata ? 'stop_requested' and jsonb_typeof(p_metadata->'stop_requested') <> 'boolean'
        ) then
            raise exception 'Invalid analysis metadata.' using errcode = '22023';
        end if;
    end if;
    completed_count := task.completed_items;
    failed_count := task.failed_items;
    for entry in select value from jsonb_array_elements(p_updates) loop
        if jsonb_typeof(entry) <> 'object' or coalesce(entry->>'index', '') !~ '^(0|[1-9][0-9]*)$'
            or coalesce(entry->>'attempts', '') !~ '^(0|[1-9][0-9]*)$' then
            raise exception 'Invalid analysis item update.' using errcode = '22023';
        end if;
        next_index := (entry->>'index')::integer;
        next_status := entry->>'status';
        next_result := nullif(entry->'result', 'null'::jsonb);
        next_error := entry->>'error';
        next_attempts := (entry->>'attempts')::integer;
        if next_index = any(seen) or next_index >= task.total_items or next_attempts not between 0 and 1000000
            or next_status is null or next_status not in ('pending', 'complete', 'failed')
            or (next_status = 'complete' and (next_result is null or next_error is not null))
            or (next_status = 'failed' and (next_result is not null or next_error is null
                or length(btrim(next_error)) = 0 or length(next_error) > 16000))
            or (next_status = 'pending' and (next_result is not null or next_error is not null)) then
            raise exception 'Invalid analysis item result/status.' using errcode = '22023';
        end if;
        seen := array_append(seen, next_index);
        select * into old_item from public.analysis_items
            where task_id = p_task_id and owner_id = p_owner_id and item_index = next_index for update;
        if not found then
            raise exception 'Analysis item not found.' using errcode = 'P0002';
        end if;
        if old_item.status = 'complete' then
            if next_status <> 'complete' or old_item.result is distinct from next_result then
                raise exception 'Completed analysis result cannot be replaced.' using errcode = '22023';
            end if;
            continue;
        end if;
        completed_count := completed_count + case when next_status = 'complete' then 1 else 0 end;
        failed_count := failed_count + case when next_status = 'failed' then 1 else 0 end
            - case when old_item.status = 'failed' then 1 else 0 end;
        update public.analysis_items set status = next_status, result = next_result,
            error = next_error, attempts = next_attempts, updated_at = clock_timestamp()
            where task_id = p_task_id and owner_id = p_owner_id and item_index = next_index;
    end loop;
    -- A Kill that won the row lock cannot be overwritten by a late completion.
    if p_status = 'complete' and not task.kill_requested then
        select count(*), count(*) filter (where status = 'complete') into actual_count, actual_complete
            from public.analysis_items where task_id = p_task_id and owner_id = p_owner_id;
        if actual_count <> task.total_items or actual_complete <> task.total_items then
            raise exception 'Analysis task still has unfinished items.' using errcode = '22023';
        end if;
    end if;
    next_metadata := coalesce(p_metadata, task.metadata);
    next_stop := task.stop_requested or coalesce((next_metadata->>'stop_requested')::boolean, false);
    if next_stop then
        next_metadata := jsonb_set(next_metadata, '{stop_requested}', 'true'::jsonb);
    end if;
    update public.analysis_tasks set completed_items = completed_count, failed_items = failed_count,
        metadata = next_metadata, stop_requested = next_stop,
        status = case when task.kill_requested then 'stopping' else coalesce(p_status, task.status) end,
        completed_at = case when not task.kill_requested and coalesce(p_status, task.status) = 'complete'
            then coalesce(task.completed_at, clock_timestamp()) else task.completed_at end,
        revision = revision + 1, updated_at = clock_timestamp()
        where id = p_task_id and owner_id = p_owner_id returning * into task;
    return to_jsonb(task);
end;
$$;

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

create or replace function public.analysis_request_stop(p_owner_id uuid, p_task_id uuid)
returns jsonb language plpgsql security definer set search_path = '' as $$
declare task public.analysis_tasks%rowtype;
begin
    task := public.analysis_locked_task(p_owner_id, p_task_id, null, false);
    if task.status = 'complete' or task.kill_requested or task.status in ('stopping', 'killed') then
        return to_jsonb(task);
    end if;
    update public.analysis_tasks set stop_requested = true,
        metadata = jsonb_set(metadata, '{stop_requested}', 'true'::jsonb),
        status = case when task.status = 'creating' then 'creating'
            when task.lease_expires_at > clock_timestamp() then task.status else 'incomplete' end,
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

create or replace function public.analysis_reconcile_killed_tasks(p_owner_id uuid)
returns integer language plpgsql security definer set search_path = '' as $$
declare
    reconciled_count integer;
    transition_time timestamptz := clock_timestamp();
begin
    perform public.analysis_check_owner(p_owner_id);
    update public.analysis_tasks set status = 'killed', killed_at = coalesce(killed_at, transition_time),
        lease_token = null, lease_expires_at = null,
        revision = revision + 1, updated_at = transition_time
        where owner_id = p_owner_id and kill_requested and status = 'stopping'
          and (lease_expires_at is null or lease_expires_at <= transition_time)
          and (not public.analysis_is_official_game_page(page_key)
               or (coalesce(auth.role(), '') = 'service_role' and server_managed));
    get diagnostics reconciled_count = row_count;
    return reconciled_count;
end;
$$;

-- Replacement functions retain old ACLs; new functions have PUBLIC execute by
-- default. Normalize every changed lifecycle RPC to the same restricted grants.
do $$
declare f record;
begin
    for f in select p.oid::regprocedure as signature
        from pg_proc p join pg_namespace n on n.oid = p.pronamespace
        where n.nspname = 'public' and p.proname in (
            'analysis_put_items','analysis_finalize_task','analysis_claim','analysis_append_item',
            'analysis_save_items','analysis_release','analysis_request_stop','analysis_kill_task',
            'analysis_reconcile_killed_tasks'
        ) loop
        execute format('revoke all on function %s from public, anon, authenticated, service_role', f.signature);
        execute format('grant execute on function %s to authenticated, service_role', f.signature);
    end loop;
end;
$$;

notify pgrst, 'reload schema';
commit;
