-- Durable, owner-scoped analysis tasks. Run this migration in Supabase SQL Editor.
-- This migration only adds analysis tables/RPCs; it does not change game scores.
-- API credentials must remain in the worker. Settings, source and inputs are immutable.
begin;

create table if not exists public.analysis_tasks (
    id uuid primary key,
    owner_id uuid not null references auth.users(id) on delete cascade,
    page_key text not null check (page_key ~ '^[A-Za-z0-9_ ./:-]{1,100}$'),
    fingerprint text not null check (fingerprint ~ '^[a-f0-9]{64}$'),
    settings jsonb not null check (jsonb_typeof(settings) = 'object'),
    source jsonb not null,
    metadata jsonb not null default '{}'::jsonb check (jsonb_typeof(metadata) = 'object'),
    dynamic_items boolean not null default false,
    server_managed boolean not null default false,
    total_items integer not null check (total_items between 0 and 500000),
    status text not null default 'creating'
        check (status in ('creating', 'queued', 'running', 'incomplete', 'complete')),
    completed_items integer not null default 0 check (completed_items >= 0),
    failed_items integer not null default 0 check (failed_items >= 0),
    stop_requested boolean not null default false,
    lease_token uuid,
    lease_generation bigint not null default 0,
    lease_expires_at timestamptz,
    revision bigint not null default 0,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (id, owner_id),
    check (completed_items + failed_items <= total_items),
    check ((lease_token is null) = (lease_expires_at is null))
);

-- Existing rows have no authenticated/server provenance. Never infer trusted
-- Game input from ownership, a client-supplied field or an old saved response.
alter table public.analysis_tasks add column if not exists server_managed boolean not null default false;

alter table public.analysis_tasks drop constraint if exists analysis_tasks_page_key_check;
alter table public.analysis_tasks add constraint analysis_tasks_page_key_check
    check (page_key ~ '^[A-Za-z0-9_ ./:-]{1,100}$');

create table if not exists public.analysis_items (
    task_id uuid not null,
    owner_id uuid not null,
    item_index integer not null check (item_index between 0 and 499999),
    input jsonb not null,
    status text not null default 'pending' check (status in ('pending', 'complete', 'failed')),
    result jsonb,
    error text,
    attempts integer not null default 0 check (attempts between 0 and 1000000),
    updated_at timestamptz not null default now(),
    primary key (task_id, item_index),
    foreign key (task_id, owner_id) references public.analysis_tasks(id, owner_id) on delete cascade,
    check (
        (status = 'pending' and result is null and error is null)
        or (status = 'complete' and result is not null and result <> 'null'::jsonb and error is null)
        or (status = 'failed' and result is null and error is not null
            and length(btrim(error)) > 0 and length(error) <= 16000)
    )
);

create index if not exists analysis_tasks_owner_page_updated_idx
    on public.analysis_tasks(owner_id, page_key, updated_at desc, id);
create index if not exists analysis_tasks_owner_updated_idx
    on public.analysis_tasks(owner_id, updated_at desc, id);

alter table public.analysis_tasks enable row level security;
alter table public.analysis_items enable row level security;
drop policy if exists analysis_tasks_owner_read on public.analysis_tasks;
create policy analysis_tasks_owner_read on public.analysis_tasks for select to authenticated
    using ((select auth.uid()) = owner_id);
drop policy if exists analysis_items_owner_read on public.analysis_items;
create policy analysis_items_owner_read on public.analysis_items for select to authenticated
    using ((select auth.uid()) = owner_id);
-- RPCs are the only write path, including for service-role workers. Service-role
-- bypasses RLS, so every RPC and every application read still scopes owner_id.
revoke all on public.analysis_tasks, public.analysis_items from public, anon, authenticated, service_role;
grant select on public.analysis_tasks, public.analysis_items to authenticated, service_role;

create or replace function public.analysis_check_owner(p_owner_id uuid)
returns void language plpgsql security invoker set search_path = '' as $$
begin
    if p_owner_id is null or (
        coalesce(auth.role(), '') <> 'service_role' and auth.uid() is distinct from p_owner_id
    ) then
        raise exception 'Analysis owner authorization failed.' using errcode = '42501';
    end if;
end;
$$;

create or replace function public.analysis_check_json(p_value jsonb)
returns void language plpgsql security invoker set search_path = '' as $$
declare
    forbidden boolean;
begin
    if p_value is null or octet_length(p_value::text) > 16777216 then
        raise exception 'Invalid analysis JSON size.' using errcode = '22023';
    end if;
    with recursive walk(value) as (
        select p_value
        union all
        select children.value from walk
        cross join lateral (
            select e.value from jsonb_each(case when jsonb_typeof(walk.value) = 'object'
                then walk.value else '{}'::jsonb end) e
            union all
            select a.value from jsonb_array_elements(case when jsonb_typeof(walk.value) = 'array'
                then walk.value else '[]'::jsonb end) a
        ) children
    )
    select exists (
        select 1 from walk cross join lateral jsonb_each(
            case when jsonb_typeof(walk.value) = 'object' then walk.value else '{}'::jsonb end
        ) e
        where regexp_replace(lower(e.key), '[^a-z0-9]', '', 'g') = any(array[
            'apikey','accesskey','secretkey','accesskeyid','secretaccesskey','accesstoken',
            'refreshtoken','authorization','password','passwd','clientsecret','servicerolekey',
            'supabaseservicerolekey','anonkey','supabaseanonkey','credentials','credential','bearertoken','apikeys','apitoken'
        ]) or regexp_replace(lower(e.key), '[^a-z0-9]', '', 'g')
            ~ '(apikeys?|apitoken|accesstoken|refreshtoken|secretkey|servicerolekey)$'
        union all
        select 1 from walk cross join lateral jsonb_each(
            case when jsonb_typeof(walk.value) = 'object' then walk.value else '{}'::jsonb end
        ) e where jsonb_typeof(e.value) = 'string' and (
            regexp_replace(lower(e.key), '[^a-z0-9]', '', 'g') ~ 'url$'
            or regexp_replace(lower(e.key), '[^a-z0-9]', '', 'g') in ('endpoint', 'baseuri', 'apibase')
        ) and (
            (e.value #>> '{}') ~* '^[a-z][a-z0-9+.-]*://[^/?#]*@'
            or (e.value #>> '{}') ~* '^[a-z][a-z0-9+.-]*://.*[?&](api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|password|secret[_-]?key)='
        )
    ) into forbidden;
    if forbidden then
        raise exception 'Credentials cannot be stored in analysis checkpoints.' using errcode = '22023';
    end if;
end;
$$;

-- Official score inputs must originate in the trusted server. Authenticated
-- owners may checkpoint ordinary analyses but cannot forge/rewrite Game API
-- responses that a service-role runner would later submit to the leaderboard.
-- Include legacy application aliases as long as they execute official games.
create or replace function public.analysis_is_official_game_page(p_page_key text)
returns boolean language sql immutable set search_path = '' as $$
    select p_page_key = any(array[
        'Game 1: The Hidden Passage Hunt',
        'Game 2: The Hidden Passage Hunt',
        'Copyright Challenge',
        'Copyright Challenge 1'
    ]);
$$;

create or replace function public.analysis_check_page_write(p_page_key text)
returns void language plpgsql security invoker set search_path = '' as $$
begin
    if public.analysis_is_official_game_page(p_page_key)
        and coalesce(auth.role(), '') <> 'service_role' then
        raise exception 'analysis_official_game: official task writes require the trusted server.'
            using errcode = '42501';
    end if;
end;
$$;

-- Internal lock/fencing helper, never callable by anon/authenticated/service_role.
create or replace function public.analysis_locked_task(
    p_owner_id uuid, p_task_id uuid, p_lease_token uuid default null,
    p_require_lease boolean default true
)
returns public.analysis_tasks language plpgsql security definer set search_path = '' as $$
declare
    task public.analysis_tasks%rowtype;
begin
    perform public.analysis_check_owner(p_owner_id);
    select * into task from public.analysis_tasks
        where id = p_task_id and owner_id = p_owner_id for update;
    if not found then
        raise exception 'Analysis task not found.' using errcode = 'P0002';
    end if;
    perform public.analysis_check_page_write(task.page_key);
    if public.analysis_is_official_game_page(task.page_key) and not task.server_managed then
        raise exception 'analysis_untrusted_game: this legacy official task has no verified server origin.'
            using errcode = '42501';
    end if;
    if p_require_lease and (
        p_lease_token is null or task.lease_token is distinct from p_lease_token
        or task.lease_expires_at is null or task.lease_expires_at <= clock_timestamp()
    ) then
        raise exception 'analysis_lease: task lease is absent, expired or replaced.' using errcode = 'P0001';
    end if;
    return task;
end;
$$;

create or replace function public.analysis_create_task(
    p_owner_id uuid, p_task_id uuid, p_page_key text, p_settings jsonb, p_source jsonb,
    p_total_items integer, p_fingerprint text, p_metadata jsonb, p_dynamic_items boolean
)
returns jsonb language plpgsql security definer set search_path = '' as $$
declare
    task public.analysis_tasks%rowtype;
begin
    perform public.analysis_check_owner(p_owner_id);
    perform public.analysis_check_page_write(p_page_key);
    perform public.analysis_check_json(p_settings);
    perform public.analysis_check_json(p_source);
    perform public.analysis_check_json(p_metadata);
    if p_task_id is null or p_dynamic_items is null or p_total_items is null
        or p_total_items not between 0 and 500000 or (p_dynamic_items and p_total_items <> 0)
        or jsonb_typeof(p_settings) <> 'object' or jsonb_typeof(p_metadata) <> 'object'
        or (p_metadata ? 'stop_requested' and jsonb_typeof(p_metadata->'stop_requested') <> 'boolean') then
        raise exception 'Invalid analysis task manifest.' using errcode = '22023';
    end if;
    -- ON CONFLICT followed by a row lock also makes racing creation idempotent.
    insert into public.analysis_tasks(id, owner_id, page_key, fingerprint, settings, source,
        metadata, dynamic_items, server_managed, total_items)
    values(p_task_id, p_owner_id, p_page_key, p_fingerprint, p_settings, p_source,
        p_metadata, p_dynamic_items, coalesce(auth.role(), '') = 'service_role', p_total_items)
    on conflict(id) do nothing;
    task := public.analysis_locked_task(p_owner_id, p_task_id, null, false);
    if task.page_key is distinct from p_page_key or task.fingerprint is distinct from p_fingerprint
        or task.settings is distinct from p_settings or task.source is distinct from p_source
        or task.dynamic_items is distinct from p_dynamic_items
        or (not task.dynamic_items and task.total_items is distinct from p_total_items) then
        raise exception 'Analysis task identity/settings cannot be changed.' using errcode = '22023';
    end if;
    return to_jsonb(task);
end;
$$;

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
    if task.status <> 'creating' or task.dynamic_items or jsonb_typeof(p_items) <> 'array'
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
    if task.status <> 'creating' then
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
    if task.status in ('creating', 'complete') then
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
        lease_generation = new_generation,
        lease_expires_at = clock_timestamp() + make_interval(secs => p_ttl_seconds),
        stop_requested = case when new_generation = task.lease_generation then task.stop_requested else false end,
        metadata = case when new_generation = task.lease_generation then task.metadata
            else jsonb_set(task.metadata, '{stop_requested}', 'false'::jsonb) end,
        revision = revision + 1, updated_at = clock_timestamp()
        where id = p_task_id and owner_id = p_owner_id returning * into task;
    return to_jsonb(task);
end;
$$;

create or replace function public.analysis_heartbeat(
    p_owner_id uuid, p_task_id uuid, p_lease_token uuid, p_ttl_seconds integer
)
returns jsonb language plpgsql security definer set search_path = '' as $$
declare task public.analysis_tasks%rowtype;
begin
    task := public.analysis_locked_task(p_owner_id, p_task_id, p_lease_token);
    if p_ttl_seconds is null or p_ttl_seconds not between 60 and 900 then
        raise exception 'Invalid analysis lease duration.' using errcode = '22023';
    end if;
    update public.analysis_tasks set lease_expires_at = clock_timestamp() + make_interval(secs => p_ttl_seconds),
        updated_at = clock_timestamp()
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
    if not task.dynamic_items or task.status <> 'running' or task.stop_requested
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
        or (p_status is not null and p_status not in ('running', 'incomplete', 'complete'))
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
    if p_status = 'complete' then
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
        metadata = next_metadata, stop_requested = next_stop, status = coalesce(p_status, task.status),
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
    if p_status is null or p_status not in ('incomplete', 'complete') then
        raise exception 'Invalid released analysis status.' using errcode = '22023';
    end if;
    perform public.analysis_save_items(p_owner_id, p_task_id, '[]'::jsonb, p_metadata, p_status, p_lease_token);
    update public.analysis_tasks set lease_token = null, lease_expires_at = null,
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
    if task.status = 'complete' then
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

create or replace function public.analysis_delete(p_owner_id uuid, p_task_id uuid)
returns jsonb language plpgsql security definer set search_path = '' as $$
declare task public.analysis_tasks%rowtype;
begin
    perform public.analysis_check_owner(p_owner_id);
    select * into task from public.analysis_tasks
        where id = p_task_id and owner_id = p_owner_id for update;
    if not found then
        return jsonb_build_object('deleted', true);
    end if;
    perform public.analysis_check_page_write(task.page_key);
    if task.lease_token is not null and task.lease_expires_at > clock_timestamp() then
        raise exception 'analysis_lease: stop the active worker before deleting this task.' using errcode = 'P0001';
    end if;
    delete from public.analysis_tasks where id = p_task_id and owner_id = p_owner_id;
    return jsonb_build_object('deleted', true);
end;
$$;

-- Remove PostgreSQL's default PUBLIC execute privilege. Internal helpers remain
-- private; only these narrow owner/lease-checked lifecycle RPCs are exposed.
do $$
declare f record;
begin
    for f in select p.oid::regprocedure as signature, p.proname
        from pg_proc p join pg_namespace n on n.oid = p.pronamespace
        where n.nspname = 'public' and p.proname in (
            'analysis_check_owner','analysis_check_json','analysis_is_official_game_page','analysis_check_page_write','analysis_locked_task',
            'analysis_create_task','analysis_put_items','analysis_finalize_task',
            'analysis_claim','analysis_heartbeat','analysis_append_item',
            'analysis_save_items','analysis_release','analysis_request_stop','analysis_delete'
        ) loop
        execute format('revoke all on function %s from public, anon, authenticated, service_role', f.signature);
        if f.proname not in ('analysis_check_owner','analysis_check_json','analysis_is_official_game_page','analysis_check_page_write','analysis_locked_task') then
            execute format('grant execute on function %s to authenticated, service_role', f.signature);
        end if;
    end loop;
end;
$$;

commit;