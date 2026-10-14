-- Compatibility upgrade for legacy completed-result archives.
-- Existing archives remain intact; Kill now deletes task journals.
-- Prefer analysis_history_delete.sql for the current History migration.
-- Run after analysis_checkpoints.sql / analysis_task_lifecycle.sql.
begin;

-- Retain legacy archive markers for existing results in History.
-- New Kill actions delete task journals rather than create archive markers.
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
    task := public.analysis_locked_task(p_owner_id, p_task_id, null, false);
    -- Existing archived results remain readable during an upgrade.
    if task.archived_at is not null and task.status = 'complete'
        and p_status in ('incomplete', 'complete') then
        return to_jsonb(task);
    end if;
    task := public.analysis_locked_task(p_owner_id, p_task_id, p_lease_token);
    if p_status is null or p_status not in ('incomplete', 'complete', 'killed')
        or (p_status = 'killed' and not task.kill_requested) then
        raise exception 'Invalid released analysis status.' using errcode = '22023';
    end if;
    -- Finish a Kill requested by an older deployment by deleting its journal.
    if task.kill_requested then
        return public.analysis_kill_task(p_owner_id, p_task_id);
    end if;
    task := jsonb_populate_record(null::public.analysis_tasks,
        public.analysis_save_items(p_owner_id, p_task_id, '[]'::jsonb, p_metadata, p_status, p_lease_token));
    update public.analysis_tasks set lease_token = null, lease_expires_at = null,
        revision = revision + 1, updated_at = clock_timestamp()
        where id = p_task_id and owner_id = p_owner_id returning * into task;
    return to_jsonb(task);
end;
$$;

create or replace function public.analysis_kill_task(p_owner_id uuid, p_task_id uuid)
returns jsonb language plpgsql security definer set search_path = '' as $$
declare task public.analysis_tasks%rowtype;
begin
    perform public.analysis_check_owner(p_owner_id);
    if p_task_id is null then
        raise exception 'Analysis task is required.' using errcode = '22023';
    end if;
    select * into task from public.analysis_tasks
        where id = p_task_id and owner_id = p_owner_id for update;
    if found then
        perform public.analysis_check_page_write(task.page_key);
        -- Deleting an owner's legacy task cannot forge or replay official inputs.
        -- Provenance remains mandatory for claims, writes and game submissions.
        -- Compatibility with game-link installations preceding the cascading FK.
        -- Independent official scores and user settings are never deleted.
        if to_regclass('public.analysis_game_task_links') is not null then
            if public.analysis_is_official_game_page(task.page_key) and task.server_managed
                and to_regclass('public.copyright_game_runs') is not null then
                -- Same guarded running->failed transition as fail_stage_two.
                -- Task lock precedes run locks, matching checkpoint game RPCs.
                -- No participant lock is acquired after locking a run, so the
                -- participant->run recovery RPC cannot form a lock cycle.
                update public.copyright_game_runs as runs
                    set status = 'failed', failure_code = 'analysis_task_deleted',
                        finished_at = clock_timestamp()
                    from public.analysis_game_task_links as links
                    where links.task_id = p_task_id and links.user_id = p_owner_id
                      and links.stage = 'stage_two' and runs.id = links.run_id
                      and runs.user_id = p_owner_id
                      and runs.competition_slug = links.competition_slug and runs.status = 'running';
            end if;
            delete from public.analysis_game_task_links
                where task_id = p_task_id and user_id = p_owner_id;
        end if;
        -- All checkpoint writes lock this row first. Deleting it fences active
        -- workers immediately; late progress/results cannot recreate a task.
        delete from public.analysis_tasks where id = p_task_id and owner_id = p_owner_id;
    end if;
    -- An absent task is an idempotent success after a lost HTTP response.
    return jsonb_build_object('id', p_task_id, 'owner_id', p_owner_id,
        'deleted', true, 'status', 'killed', 'kill_requested', true,
        'stop_requested', true, 'lease_token', null, 'lease_expires_at', null);
end;
$$;

create or replace function public.analysis_reconcile_killed_tasks(p_owner_id uuid)
returns integer language plpgsql security definer set search_path = '' as $$
declare
    task public.analysis_tasks%rowtype;
    reconciled_count integer := 0;
begin
    perform public.analysis_check_owner(p_owner_id);
    -- Only clean up explicit terminal Kills from older versions. An ordinary
    -- Stop remains resumable, and an older active worker retains its lease.
    for task in select * from public.analysis_tasks
        where owner_id = p_owner_id and kill_requested and status in ('stopping', 'killed')
          and (lease_expires_at is null or lease_expires_at <= clock_timestamp())
          and (not public.analysis_is_official_game_page(page_key)
               or coalesce(auth.role(), '') = 'service_role')
        for update skip locked
    loop
        perform public.analysis_kill_task(p_owner_id, task.id);
        reconciled_count := reconciled_count + 1;
    end loop;
    return reconciled_count;
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

revoke all on function public.analysis_reconcile_killed_tasks(uuid) from public, anon, authenticated, service_role;
grant execute on function public.analysis_reconcile_killed_tasks(uuid) to authenticated, service_role;

-- Remove orphan recovery links only; official runs/scores are independent.
-- The composite FK rejects cross-owner links and cascades deleted task links.
do $$
begin
    if to_regclass('public.analysis_game_task_links') is not null then
        delete from public.analysis_game_task_links as links
            where not exists (select 1 from public.analysis_tasks as tasks
                where tasks.id = links.task_id and tasks.owner_id = links.user_id);
        alter table public.analysis_game_task_links
            drop constraint if exists analysis_game_task_links_task_owner_fkey;
        alter table public.analysis_game_task_links
            add constraint analysis_game_task_links_task_owner_fkey
            foreign key (task_id, user_id) references public.analysis_tasks(id, owner_id) on delete cascade;
    end if;
end;
$$;

-- Serialize official recovery submissions with Kill, without installing games.
do $upgrade$
begin
    if to_regprocedure('public.analysis_game_check_trusted_task(uuid,uuid,text)') is not null then
        execute $definition$
create or replace function public.analysis_game_check_trusted_task(
    p_task_id uuid, p_user_id uuid, p_competition_slug text
) returns void language plpgsql security definer set search_path = '' as $$
declare
    task public.analysis_tasks%rowtype;
    allowed_pages text[];
begin
    -- A row lock needs UPDATE rights; keep direct writes revoked and lock under
    -- this narrowly service-role-checked helper rather than grant table UPDATE.
    if coalesce(auth.role(), '') <> 'service_role' then
        raise exception 'analysis_official_game: official recovery requires the trusted server.'
            using errcode = '42501';
    end if;
    if p_competition_slug = 'hp-first-100-gpt-4o-mini-v1' then
        allowed_pages := array[
            'Game 1: The Hidden Passage Hunt', 'Game 2: The Hidden Passage Hunt',
            'Copyright Challenge', 'Copyright Challenge 1'
        ];
    else
        raise exception 'This competition has no supported trusted checkpoint page.' using errcode = '42501';
    end if;
    select * into task from public.analysis_tasks
     where id = p_task_id and owner_id = p_user_id for update;
    if not found or not task.server_managed or task.kill_requested
        or task.status in ('stopping', 'killed') or not (task.page_key = any(allowed_pages)) then
        raise exception 'analysis_untrusted_game: official input must come from this owner''s server-created game task.'
            using errcode = '42501';
    end if;
end;
$$;
$definition$;
        revoke all on function public.analysis_game_check_trusted_task(uuid,uuid,text)
            from public, anon, authenticated, service_role;
        grant execute on function public.analysis_game_check_trusted_task(uuid,uuid,text) to service_role;
    end if;
end;
$upgrade$;


notify pgrst, 'reload schema';
commit;
