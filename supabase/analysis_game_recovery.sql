-- Apply after copyright_game.sql. Only new durable analysis tasks use these RPCs.
-- The task link and existing lifecycle RPC commit in one transaction. A lost
-- response can therefore replay the same record without creating a new score.
begin;

create table if not exists public.analysis_game_task_links (
    task_id uuid not null,
    stage text not null check (stage in ('stage_one', 'stage_two')),
    user_id uuid not null,
    competition_slug text not null,
    run_id uuid not null,
    request_hash text not null,
    completion_hash text,
    created_at timestamptz not null default now(),
    primary key (task_id, stage),
    unique (stage, run_id)
);
alter table public.analysis_game_task_links enable row level security;
revoke all on table public.analysis_game_task_links from public, anon, authenticated;
grant select, insert, update, delete on table public.analysis_game_task_links to service_role;

create or replace function public.save_copyright_game_stage_one_checkpoint(
    p_task_id uuid, p_competition_slug text, p_user_id uuid, p_book_key text,
    p_prompt_text text, p_reference_text text, p_temperature double precision,
    p_top_p double precision, p_attempts jsonb
) returns jsonb language plpgsql security invoker
set search_path = public, pg_temp as $$
declare
    linked public.analysis_game_task_links%rowtype;
    saved jsonb;
    wanted_hash text;
begin
    if p_task_id is null or p_user_id is null then
        raise exception 'Checkpoint task and owner are required.' using errcode = '22023';
    end if;
    wanted_hash := encode(sha256(convert_to(jsonb_build_array(
        p_competition_slug, p_user_id, p_book_key, p_prompt_text,
        p_reference_text, p_temperature, p_top_p, p_attempts
    )::text, 'UTF8')), 'hex');
    perform pg_advisory_xact_lock(hashtextextended('analysis-game:' || p_task_id::text || ':stage_one', 0));
    select * into linked from public.analysis_game_task_links
     where task_id = p_task_id and stage = 'stage_one' for update;
    if found then
        if linked.user_id is distinct from p_user_id
           or linked.competition_slug is distinct from p_competition_slug
           or linked.request_hash is distinct from wanted_hash then
            raise exception 'Checkpoint owner or saved configuration differs.' using errcode = '42501';
        end if;
        select to_jsonb(runs) into saved from public.copyright_game_stage_one_runs as runs
         where runs.id = linked.run_id and runs.user_id = p_user_id
           and runs.competition_slug = p_competition_slug;
        if saved is null then
            raise exception 'The saved baseline was removed; start a new task.' using errcode = 'P0002';
        end if;
        return saved;
    end if;
    saved := public.save_copyright_game_stage_one_run(
        p_competition_slug, p_user_id, p_book_key, p_prompt_text,
        p_reference_text, p_temperature, p_top_p, p_attempts
    );
    if nullif(saved->>'id', '') is null then
        raise exception 'Baseline save returned no run.' using errcode = 'P0002';
    end if;
    insert into public.analysis_game_task_links(task_id, stage, user_id, competition_slug, run_id, request_hash)
    values (p_task_id, 'stage_one', p_user_id, p_competition_slug, (saved->>'id')::uuid, wanted_hash);
    return saved;
end;
$$;

create or replace function public.begin_copyright_game_run_checkpoint(
    p_task_id uuid, p_competition_slug text, p_user_id uuid,
    p_shot_mode text, p_strategy text, p_attempts_per_strategy integer,
    p_attempts_per_prompt integer, p_temperature double precision,
    p_top_p double precision, p_book_key text, p_book_keys text[],
    p_expected_run_id uuid default null
) returns jsonb language plpgsql security invoker
set search_path = public, pg_temp as $$
declare
    linked public.analysis_game_task_links%rowtype;
    reserved public.copyright_game_runs%rowtype;
    saved jsonb;
    wanted_hash text;
begin
    if p_task_id is null or p_user_id is null then
        raise exception 'Checkpoint task and owner are required.' using errcode = '22023';
    end if;
    wanted_hash := encode(sha256(convert_to(jsonb_build_array(
        p_competition_slug, p_user_id, p_shot_mode, btrim(p_strategy),
        p_attempts_per_strategy, p_attempts_per_prompt, p_temperature,
        p_top_p, p_book_key, p_book_keys
    )::text, 'UTF8')), 'hex');
    perform pg_advisory_xact_lock(hashtextextended('analysis-game:' || p_task_id::text || ':stage_two', 0));
    select * into linked from public.analysis_game_task_links
     where task_id = p_task_id and stage = 'stage_two' for update;
    if found then
        if linked.user_id is distinct from p_user_id
           or linked.competition_slug is distinct from p_competition_slug
           or linked.request_hash is distinct from wanted_hash
           or (p_expected_run_id is not null and linked.run_id is distinct from p_expected_run_id) then
            raise exception 'Checkpoint owner or saved configuration differs.' using errcode = '42501';
        end if;
        select * into reserved from public.copyright_game_runs
         where id = linked.run_id and user_id = p_user_id
           and competition_slug = p_competition_slug for update;
        if not found then
            raise exception 'The reserved run was removed; start a new task.' using errcode = 'P0002';
        end if;
        -- Terminal state remains immutable. Explicit participant recovery releases
        -- the official reservation permanently; this task cannot reopen it.
        if reserved.status = 'failed' then
            raise exception 'The saved official run is failed. Start a new task; this run cannot resume.' using errcode = '55000';
        elsif reserved.status = 'completed' then
            return to_jsonb(reserved);
        end if;
        perform 1 from public.copyright_game_competitions
         where slug = p_competition_slug and is_open for share;
        if not found then
            raise exception 'This competition is currently closed.' using errcode = '55000';
        end if;
        if exists (select 1 from public.copyright_game_runs
            where competition_slug = p_competition_slug and user_id = p_user_id
              and status = 'running' and id <> reserved.id) then
            raise exception 'Another official run is active for this account.' using errcode = '55000';
        end if;
        return to_jsonb(reserved);
    end if;
    if p_expected_run_id is not null then
        raise exception 'The saved task reservation is missing; start a new task.' using errcode = 'P0002';
    end if;
    -- Retain the original validation, hourly limit, competition/model checks,
    -- participant locking and authoritative timestamps for every new run.
    saved := public.begin_copyright_game_run(
        p_competition_slug, p_user_id, p_shot_mode, p_strategy,
        p_attempts_per_strategy, p_attempts_per_prompt, p_temperature,
        p_top_p, p_book_key, p_book_keys
    );
    if nullif(saved->>'id', '') is null then
        raise exception 'Reservation returned no run.' using errcode = 'P0002';
    end if;
    insert into public.analysis_game_task_links(task_id, stage, user_id, competition_slug, run_id, request_hash)
    values (p_task_id, 'stage_two', p_user_id, p_competition_slug, (saved->>'id')::uuid, wanted_hash);
    return saved;
end;
$$;

create or replace function public.complete_copyright_game_run_checkpoint(
    p_task_id uuid, p_run_id uuid, p_user_id uuid, p_competition_slug text, p_attempts jsonb
) returns jsonb language plpgsql security invoker
set search_path = public, pg_temp as $$
declare
    linked public.analysis_game_task_links%rowtype;
    reserved public.copyright_game_runs%rowtype;
    saved jsonb;
    wanted_hash text;
begin
    if p_task_id is null or p_user_id is null then
        raise exception 'Checkpoint task and owner are required.' using errcode = '22023';
    end if;
    perform pg_advisory_xact_lock(hashtextextended('analysis-game:' || p_task_id::text || ':stage_two', 0));
    select * into linked from public.analysis_game_task_links
     where task_id = p_task_id and stage = 'stage_two' for update;
    if not found or linked.user_id is distinct from p_user_id
       or linked.competition_slug is distinct from p_competition_slug
       or linked.run_id is distinct from p_run_id then
        raise exception 'The saved reservation does not belong to this task.' using errcode = '42501';
    end if;
    select * into reserved from public.copyright_game_runs
     where id = p_run_id and user_id = p_user_id
       and competition_slug = p_competition_slug for update;
    if not found then
        raise exception 'The reserved run was removed.' using errcode = 'P0002';
    end if;
    if jsonb_typeof(p_attempts) is distinct from 'array' then
        raise exception 'Attempt payload must be an array.' using errcode = '22023';
    end if;
    wanted_hash := encode(sha256(convert_to(p_attempts::text, 'UTF8')), 'hex');
    if reserved.status = 'completed' then
        if linked.completion_hash is distinct from wanted_hash then
            raise exception 'Completed scores differ from this saved task.' using errcode = '23514';
        end if;
        return to_jsonb(reserved);
    end if;
    if reserved.status <> 'running' then
        raise exception 'The official run is failed and cannot resume.' using errcode = '55000';
    end if;
    saved := public.complete_copyright_game_run(p_run_id, p_user_id, p_competition_slug, p_attempts);
    update public.analysis_game_task_links set completion_hash = wanted_hash
     where task_id = p_task_id and stage = 'stage_two';
    return saved;
end;
$$;

revoke all on function public.save_copyright_game_stage_one_checkpoint(
    uuid, text, uuid, text, text, text, double precision, double precision, jsonb
) from public, anon, authenticated;
revoke all on function public.begin_copyright_game_run_checkpoint(
    uuid, text, uuid, text, text, integer, integer, double precision, double precision, text, text[], uuid
) from public, anon, authenticated;
revoke all on function public.complete_copyright_game_run_checkpoint(uuid, uuid, uuid, text, jsonb)
from public, anon, authenticated;
grant execute on function public.save_copyright_game_stage_one_checkpoint(
    uuid, text, uuid, text, text, text, double precision, double precision, jsonb
) to service_role;
grant execute on function public.begin_copyright_game_run_checkpoint(
    uuid, text, uuid, text, text, integer, integer, double precision, double precision, text, text[], uuid
) to service_role;
grant execute on function public.complete_copyright_game_run_checkpoint(uuid, uuid, uuid, text, jsonb)
to service_role;
commit;
