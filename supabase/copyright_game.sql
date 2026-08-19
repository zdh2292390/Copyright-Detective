-- Authoritative schema for the Copyright Detective public challenge.
-- Run this file in Supabase SQL Editor with an owner/admin account.
--
-- Security model:
--   * GitHub users authenticate with the existing Supabase Auth flow.
--   * anon/authenticated roles cannot write or directly read official tables.
--   * the Streamlit server verifies auth.get_user(), then uses its server-only
--     service-role key for narrowly scoped official writes.
--   * Stage 1 prompt/response history is stored in user-scoped, server-only tables.
--     Browser roles cannot read or write those records directly.

create extension if not exists pgcrypto;

create table if not exists public.copyright_game_competitions (
    slug text primary key,
    title text not null,
    is_open boolean not null default true,
    provider text not null default 'OpenAI'
        check (provider = 'OpenAI'),
    model_name text not null default 'gpt-4o-mini'
        check (model_name = 'gpt-4o-mini'),
    benchmark_version text not null default 'three-books-first-100-v1',
    max_scored_generations smallint not null default 99
        check (max_scored_generations between 1 and 1000),
    max_runs_per_hour smallint not null default 3,
    recovery_min_age_minutes smallint not null default 30,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint copyright_game_competitions_hourly_run_limit
        check (max_runs_per_hour between 1 and 3),
    constraint copyright_game_competitions_recovery_age
        check (recovery_min_age_minutes between 30 and 1440)
);

-- Keep this file safely re-runnable on a database that already has the first
-- version of the competition schema.
alter table public.copyright_game_competitions
    add column if not exists max_runs_per_hour smallint not null default 3;
alter table public.copyright_game_competitions
    add column if not exists recovery_min_age_minutes smallint not null default 30;

alter table public.copyright_game_competitions
    drop constraint if exists copyright_game_competitions_max_scored_generations_check;
alter table public.copyright_game_competitions
    add constraint copyright_game_competitions_max_scored_generations_check
    check (max_scored_generations between 1 and 1000);

do $migration$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'copyright_game_competitions_hourly_run_limit'
          and conrelid = 'public.copyright_game_competitions'::regclass
    ) then
        alter table public.copyright_game_competitions
            add constraint copyright_game_competitions_hourly_run_limit
            check (max_runs_per_hour between 1 and 3);
    end if;
    if not exists (
        select 1
        from pg_constraint
        where conname = 'copyright_game_competitions_recovery_age'
          and conrelid = 'public.copyright_game_competitions'::regclass
    ) then
        alter table public.copyright_game_competitions
            add constraint copyright_game_competitions_recovery_age
            check (recovery_min_age_minutes between 30 and 1440);
    end if;
end;
$migration$;

create table if not exists public.copyright_game_profiles (
    user_id uuid primary key references auth.users (id) on delete cascade,
    github_login text not null check (char_length(github_login) between 1 and 80),
    display_name text not null check (char_length(display_name) between 1 and 100),
    avatar_url text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.copyright_game_participants (
    competition_slug text not null
        references public.copyright_game_competitions (slug) on delete cascade,
    user_id uuid not null
        references public.copyright_game_profiles (user_id) on delete cascade,
    direct_model_name text not null
        check (direct_model_name = 'gpt-4o-mini'),
    direct_book_key text not null default 'harry_potter'
        check (direct_book_key in ('harry_potter', 'the_hobbit', 'a_game_of_thrones', 'other_books', 'pride_and_prejudice', 'nineteen_eighty_four', 'the_great_gatsby', 'to_kill_a_mockingbird', 'harry_potter_philosophers_stone')),
    direct_temperature double precision not null default 0.0,
    direct_top_p double precision not null default 1.0,
    direct_attempts smallint not null default 1,
    direct_rouge_l double precision not null
        check (direct_rouge_l between 0 and 1),
    direct_avg_rouge_l double precision not null default 0.0,
    direct_response_sha256 text not null
        check (char_length(direct_response_sha256) = 64),
    direct_prompt text,
    direct_reference_text text,
    direct_completed_at timestamptz not null,
    stage_one_completed boolean not null default true,
    created_at timestamptz not null default now(),
    primary key (competition_slug, user_id),
    constraint copyright_game_participants_direct_temperature_range
        check (direct_temperature between 0 and 2),
    constraint copyright_game_participants_direct_top_p_range
        check (direct_top_p between 0 and 1),
    constraint copyright_game_participants_direct_attempts_range
        check (direct_attempts between 1 and 50),
    constraint copyright_game_participants_direct_avg_rouge_l_range
        check (direct_avg_rouge_l between 0 and 1)
);

-- Stage 1 sampling and book choice became participant-controlled after the
-- initial schema. Defaults preserve already-completed Harry Potter baselines.
alter table public.copyright_game_participants
    add column if not exists direct_book_key text not null default 'harry_potter';
alter table public.copyright_game_participants
    add column if not exists direct_temperature double precision not null default 0.0;
alter table public.copyright_game_participants
    add column if not exists direct_top_p double precision not null default 1.0;
alter table public.copyright_game_participants
    add column if not exists direct_attempts smallint not null default 1;
alter table public.copyright_game_participants
    add column if not exists direct_avg_rouge_l double precision not null default 0.0;
alter table public.copyright_game_participants
    add column if not exists stage_one_completed boolean not null default true;
alter table public.copyright_game_participants
    add column if not exists direct_prompt text;
alter table public.copyright_game_participants
    add column if not exists direct_reference_text text;

-- Existing one-shot baselines have identical max and average scores.
update public.copyright_game_participants
set direct_avg_rouge_l = direct_rouge_l
where direct_attempts = 1
  and direct_avg_rouge_l = 0.0
  and direct_rouge_l <> 0.0;

alter table public.copyright_game_participants
    drop constraint if exists copyright_game_participants_direct_attempts_range;
alter table public.copyright_game_participants
    add constraint copyright_game_participants_direct_attempts_range
    check (direct_attempts between 1 and 50);
alter table public.copyright_game_participants
    drop constraint if exists copyright_game_participants_direct_avg_rouge_l_range;
alter table public.copyright_game_participants
    add constraint copyright_game_participants_direct_avg_rouge_l_range
    check (direct_avg_rouge_l between 0 and 1);

do $migration$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'copyright_game_participants_direct_temperature_range'
          and conrelid = 'public.copyright_game_participants'::regclass
    ) then
        alter table public.copyright_game_participants
            add constraint copyright_game_participants_direct_temperature_range
            check (direct_temperature between 0 and 2);
    end if;
    if not exists (
        select 1
        from pg_constraint
        where conname = 'copyright_game_participants_direct_top_p_range'
          and conrelid = 'public.copyright_game_participants'::regclass
    ) then
        alter table public.copyright_game_participants
            add constraint copyright_game_participants_direct_top_p_range
            check (direct_top_p between 0 and 1);
    end if;
end;
$migration$;

create table if not exists public.copyright_game_stage_one_runs (
    id uuid primary key default gen_random_uuid(),
    competition_slug text not null,
    user_id uuid not null,
    provider text not null check (provider = 'OpenAI'),
    direct_model_name text not null check (direct_model_name = 'gpt-4o-mini'),
    benchmark_version text not null,
    direct_book_key text not null
        check (direct_book_key in ('harry_potter', 'the_hobbit', 'a_game_of_thrones', 'other_books', 'pride_and_prejudice', 'nineteen_eighty_four', 'the_great_gatsby', 'to_kill_a_mockingbird', 'harry_potter_philosophers_stone')),
    direct_prompt text check (
        direct_prompt is null or char_length(direct_prompt) <= 20000
    ),
    direct_reference_text text check (
        direct_reference_text is null or char_length(direct_reference_text) <= 20000
    ),
    direct_temperature double precision not null
        check (direct_temperature between 0 and 2),
    direct_top_p double precision not null
        check (direct_top_p > 0 and direct_top_p <= 1),
    direct_attempts smallint not null
        check (direct_attempts between 1 and 50),
    direct_rouge_l double precision not null
        check (direct_rouge_l between 0 and 1),
    direct_avg_rouge_l double precision not null
        check (direct_avg_rouge_l between 0 and 1),
    is_legacy boolean not null default false,
    created_at timestamptz not null default now(),
    foreign key (competition_slug, user_id)
        references public.copyright_game_participants (competition_slug, user_id)
        on delete cascade
);

alter table public.copyright_game_stage_one_runs
    add column if not exists direct_reference_text text;
alter table public.copyright_game_stage_one_runs
    drop constraint if exists copyright_game_stage_one_runs_direct_book_key_check;
alter table public.copyright_game_stage_one_runs
    add constraint copyright_game_stage_one_runs_direct_book_key_check
    check (direct_book_key in (
        'harry_potter',
        'the_hobbit',
        'a_game_of_thrones',
        'other_books',
        'pride_and_prejudice',
        'nineteen_eighty_four',
        'the_great_gatsby',
        'to_kill_a_mockingbird',
        'harry_potter_philosophers_stone'
    ));
alter table public.copyright_game_stage_one_runs
    drop constraint if exists copyright_game_stage_one_runs_direct_reference_text_check;
alter table public.copyright_game_stage_one_runs
    add constraint copyright_game_stage_one_runs_direct_reference_text_check
    check (
        direct_reference_text is null
        or char_length(direct_reference_text) <= 20000
    );
alter table public.copyright_game_stage_one_runs
    drop constraint if exists copyright_game_stage_one_runs_direct_top_p_check;
alter table public.copyright_game_stage_one_runs
    add constraint copyright_game_stage_one_runs_direct_top_p_check
    check (direct_top_p between 0 and 1);

create index if not exists copyright_game_stage_one_runs_user_history_idx
    on public.copyright_game_stage_one_runs
    (competition_slug, user_id, created_at, id);

create unique index if not exists copyright_game_stage_one_runs_legacy_idx
    on public.copyright_game_stage_one_runs (competition_slug, user_id)
    where is_legacy;

create table if not exists public.copyright_game_stage_one_attempts (
    stage_one_run_id uuid not null
        references public.copyright_game_stage_one_runs (id) on delete cascade,
    attempt_number smallint not null check (attempt_number between 1 and 50),
    response_text text not null,
    response_sha256 text not null check (char_length(response_sha256) = 64),
    metrics jsonb not null default '{}'::jsonb
        check (jsonb_typeof(metrics) = 'object'),
    rouge_l double precision not null check (rouge_l between 0 and 1),
    created_at timestamptz not null default now(),
    primary key (stage_one_run_id, attempt_number)
);

-- Preserve pre-migration summaries. Raw prompts and responses were never stored
-- for these rows, so they remain explicitly marked as legacy history.
insert into public.copyright_game_stage_one_runs (
    competition_slug,
    user_id,
    provider,
    direct_model_name,
    benchmark_version,
    direct_book_key,
    direct_prompt,
    direct_reference_text,
    direct_temperature,
    direct_top_p,
    direct_attempts,
    direct_rouge_l,
    direct_avg_rouge_l,
    is_legacy,
    created_at
)
select
    participants.competition_slug,
    participants.user_id,
    competitions.provider,
    participants.direct_model_name,
    competitions.benchmark_version,
    participants.direct_book_key,
    null,
    participants.direct_reference_text,
    participants.direct_temperature,
    participants.direct_top_p,
    participants.direct_attempts,
    participants.direct_rouge_l,
    participants.direct_avg_rouge_l,
    true,
    participants.direct_completed_at
from public.copyright_game_participants as participants
join public.copyright_game_competitions as competitions
    on competitions.slug = participants.competition_slug
where participants.direct_completed_at is not null
  and participants.stage_one_completed
  and not exists (
      select 1
      from public.copyright_game_stage_one_runs as existing
      where existing.competition_slug = participants.competition_slug
        and existing.user_id = participants.user_id
  );

create table if not exists public.copyright_game_runs (
    id uuid primary key default gen_random_uuid(),
    competition_slug text not null,
    user_id uuid not null,
    status text not null
        check (status in ('running', 'completed', 'failed')),
    provider text not null
        check (provider = 'OpenAI'),
    model_name text not null
        check (model_name = 'gpt-4o-mini'),
    benchmark_version text not null
        check (benchmark_version in ('hp-first-100-v1', 'three-books-first-100-v1', 'three-books-knowledge-qa-v1', 'five-books-knowledge-qa-v2')),
    book_key text not null default 'harry_potter'
        check (book_key in ('harry_potter', 'the_hobbit', 'a_game_of_thrones', 'pride_and_prejudice', 'nineteen_eighty_four', 'the_great_gatsby', 'to_kill_a_mockingbird', 'harry_potter_philosophers_stone')),
    book_keys text[] not null default array['harry_potter']::text[]
        check (
            cardinality(book_keys) between 1 and 5
            and book_keys <@ array[
                'harry_potter',
                'the_hobbit',
                'a_game_of_thrones',
                'pride_and_prejudice',
                'nineteen_eighty_four',
                'the_great_gatsby',
                'to_kill_a_mockingbird',
                'harry_potter_philosophers_stone'
            ]::text[]
            and book_keys[1] = book_key
        ),
    shot_mode text not null
        check (shot_mode in ('zero_shot', 'few_shot')),
    strategy text not null check (char_length(strategy) between 1 and 120),
    attempts_per_strategy smallint not null
        check (attempts_per_strategy between 1 and 500),
    attempts_per_prompt smallint not null
        check (attempts_per_prompt between 1 and 500),
    expected_generations smallint generated always as
        (attempts_per_strategy * attempts_per_prompt) stored,
    successful_generations smallint not null default 0,
    temperature double precision not null
        check (temperature between 0 and 2),
    top_p double precision not null
        check (top_p between 0 and 1),
    max_rouge_l double precision,
    avg_rouge_l double precision,
    failure_code text,
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    created_at timestamptz not null default now(),
    constraint copyright_game_runs_participant_fk
        foreign key (competition_slug, user_id)
        references public.copyright_game_participants (competition_slug, user_id)
        on delete cascade,
    constraint copyright_game_runs_id_user_unique unique (id, user_id),
    -- Keep 1000 here so historical rows remain valid; new Game 1 runs are
    -- capped at 500 by competitions.max_scored_generations + app validation.
    constraint copyright_game_runs_generation_budget
        check (expected_generations > 0 and expected_generations <= 1000),
    constraint copyright_game_runs_success_count
        check (
            successful_generations between 0
            and (attempts_per_strategy * attempts_per_prompt)
        ),
    constraint copyright_game_runs_max_score
        check (max_rouge_l is null or max_rouge_l between 0 and 1),
    constraint copyright_game_runs_avg_score
        check (avg_rouge_l is null or avg_rouge_l between 0 and 1),
    constraint copyright_game_runs_score_order
        check (
            max_rouge_l is null
            or avg_rouge_l is null
            or max_rouge_l >= avg_rouge_l
        ),
    constraint copyright_game_runs_completed_shape
        check (
            status <> 'completed'
            or (
                successful_generations =
                    (attempts_per_strategy * attempts_per_prompt)
                and max_rouge_l is not null
                and avg_rouge_l is not null
                and finished_at is not null
            )
        )
);

-- Preserve old HP runs while allowing new three-book runs.
alter table public.copyright_game_runs
    add column if not exists book_key text not null default 'harry_potter';
alter table public.copyright_game_runs
    add column if not exists book_keys text[];
update public.copyright_game_runs
set book_keys = array[book_key]::text[]
where book_keys is null;
alter table public.copyright_game_runs
    alter column book_keys set default array['harry_potter']::text[];
alter table public.copyright_game_runs
    alter column book_keys set not null;
alter table public.copyright_game_runs
    drop constraint if exists copyright_game_runs_book_keys_check;
alter table public.copyright_game_runs
    add constraint copyright_game_runs_book_keys_check
    check (
        cardinality(book_keys) between 1 and 5
        and book_keys <@ array[
            'harry_potter',
            'the_hobbit',
            'a_game_of_thrones',
            'pride_and_prejudice',
            'nineteen_eighty_four',
            'the_great_gatsby',
            'to_kill_a_mockingbird',
            'harry_potter_philosophers_stone'
        ]::text[]
        and book_keys[1] = book_key
    );
alter table public.copyright_game_runs
    drop constraint if exists copyright_game_runs_attempts_per_strategy_check;
alter table public.copyright_game_runs
    add constraint copyright_game_runs_attempts_per_strategy_check
    check (attempts_per_strategy between 1 and 500);
alter table public.copyright_game_runs
    drop constraint if exists copyright_game_runs_attempts_per_prompt_check;
alter table public.copyright_game_runs
    add constraint copyright_game_runs_attempts_per_prompt_check
    check (attempts_per_prompt between 1 and 500);
alter table public.copyright_game_runs
    drop constraint if exists copyright_game_runs_generation_budget;
alter table public.copyright_game_runs
    add constraint copyright_game_runs_generation_budget
    check (expected_generations > 0 and expected_generations <= 1000);
alter table public.copyright_game_runs
    drop constraint if exists copyright_game_runs_benchmark_version_check;
alter table public.copyright_game_runs
    add constraint copyright_game_runs_benchmark_version_check
    check (benchmark_version in ('hp-first-100-v1', 'three-books-first-100-v1', 'three-books-knowledge-qa-v1', 'five-books-knowledge-qa-v2'));

-- The shared tables retain legacy Challenge 1 rows while accepting the five
-- canonical Knowledge Memorization books. Competition-specific triggers below
-- enforce the narrower book set for each challenge.
alter table public.copyright_game_participants
    drop constraint if exists copyright_game_participants_direct_book_key_check;
alter table public.copyright_game_participants
    add constraint copyright_game_participants_direct_book_key_check
    check (direct_book_key in (
        'harry_potter',
        'the_hobbit',
        'a_game_of_thrones',
        'other_books',
        'pride_and_prejudice',
        'nineteen_eighty_four',
        'the_great_gatsby',
        'to_kill_a_mockingbird',
        'harry_potter_philosophers_stone'
    ));
alter table public.copyright_game_participants
    drop constraint if exists copyright_game_participants_direct_top_p_range;
alter table public.copyright_game_participants
    add constraint copyright_game_participants_direct_top_p_range
    check (direct_top_p between 0 and 1);

alter table public.copyright_game_runs
    drop constraint if exists copyright_game_runs_book_key_check;
alter table public.copyright_game_runs
    add constraint copyright_game_runs_book_key_check
    check (book_key in (
        'harry_potter',
        'the_hobbit',
        'a_game_of_thrones',
        'other_books',
        'pride_and_prejudice',
        'nineteen_eighty_four',
        'the_great_gatsby',
        'to_kill_a_mockingbird',
        'harry_potter_philosophers_stone'
    ));
alter table public.copyright_game_runs
    drop constraint if exists copyright_game_runs_top_p_check;
alter table public.copyright_game_runs
    add constraint copyright_game_runs_top_p_check
    check (top_p between 0 and 1);

-- Validate Stage 1 using the selected competition's own benchmark contract.
create or replace function public.guard_copyright_game_participant()
returns trigger
language plpgsql
set search_path = public, pg_temp
as $$
begin
    if not new.stage_one_completed then
        return new;
    end if;
    if new.competition_slug = 'hp-first-100-gpt-4o-mini-v1' then
        if new.direct_book_key not in (
            'harry_potter', 'the_hobbit', 'a_game_of_thrones', 'other_books'
        ) then
            raise exception 'Challenge 1 uses its three recall books or Other books.'
                using errcode = '23514';
        end if;
        if new.direct_book_key = 'other_books'
           and (
                btrim(coalesce(new.direct_prompt, '')) = ''
                or btrim(coalesce(new.direct_reference_text, '')) = ''
           ) then
            raise exception 'Other books requires a custom prompt and ground truth.'
                using errcode = '23514';
        end if;
    elsif new.competition_slug = 'knowledge-memorization-gpt-4o-mini-v2' then
        if new.direct_book_key not in (
            'pride_and_prejudice',
            'nineteen_eighty_four',
            'the_great_gatsby',
            'to_kill_a_mockingbird',
            'harry_potter_philosophers_stone'
        ) then
            raise exception 'Challenge 3 uses only the five Knowledge Memorization books.'
                using errcode = '23514';
        end if;
        if new.direct_attempts <> 5 then
            raise exception 'Challenge 3 Stage 1 must contain exactly five questions.'
                using errcode = '23514';
        end if;
        if new.direct_temperature < 0 or new.direct_temperature > 1.2
           or new.direct_top_p < 0 or new.direct_top_p > 1 then
            raise exception 'Challenge 3 Stage 1 sampling settings are outside the official range.'
                using errcode = '23514';
        end if;
        if new.direct_avg_rouge_l is distinct from new.direct_rouge_l then
            raise exception 'Challenge 3 Stage 1 must store its mean Token F1 consistently.'
                using errcode = '23514';
        end if;
    end if;
    return new;
end;
$$;

drop trigger if exists guard_copyright_game_participant_write
    on public.copyright_game_participants;
create trigger guard_copyright_game_participant_write
before insert or update on public.copyright_game_participants
for each row execute function public.guard_copyright_game_participant();
-- Challenge 1 participants may submit any number of completed runs. Every
-- competition still serializes the currently running attempt so duplicate
-- concurrent starts are rejected.
drop index if exists public.copyright_game_one_active_or_completed_run;
create unique index if not exists copyright_game_one_running_run
    on public.copyright_game_runs (competition_slug, user_id)
    where status = 'running';

-- Challenge 1 and knowledge memorization rankings allow multiple completed submissions.
-- Other competitions retain the original one-completed-submission rule on this shared table.
create unique index if not exists copyright_game_one_completed_run_outside_challenge_one
    on public.copyright_game_runs (competition_slug, user_id)
    where status = 'completed'
      and competition_slug not in (
          'hp-first-100-gpt-4o-mini-v1',
          'knowledge-memorization-gpt-4o-mini-v2'
      );
create index if not exists copyright_game_completed_score_index
    on public.copyright_game_runs (
        competition_slug,
        max_rouge_l desc,
        avg_rouge_l desc,
        finished_at asc
    )
    where status = 'completed';

create table if not exists public.copyright_game_attempts (
    id bigint generated by default as identity primary key,
    run_id uuid not null,
    user_id uuid not null,
    mutation_attempt smallint not null check (mutation_attempt >= 1),
    prompt_attempt smallint not null check (prompt_attempt >= 1),
    rouge_l double precision not null check (rouge_l between 0 and 1),
    mutated_prompt_sha256 text not null
        check (char_length(mutated_prompt_sha256) = 64),
    response_sha256 text not null
        check (char_length(response_sha256) = 64),
    created_at timestamptz not null default now(),
    constraint copyright_game_attempts_run_user_fk
        foreign key (run_id, user_id)
        references public.copyright_game_runs (id, user_id)
        on delete cascade,
    constraint copyright_game_attempts_one_score
        unique (run_id, mutation_attempt, prompt_attempt)
);

-- Detailed Stage 2 history was added after the original leaderboard-only
-- schema. Columns remain nullable so existing hash-only submissions stay valid.
alter table public.copyright_game_attempts
    add column if not exists book_key text;
alter table public.copyright_game_attempts
    add column if not exists mutated_prompt text;
alter table public.copyright_game_attempts
    add column if not exists response_text text;
alter table public.copyright_game_attempts
    add column if not exists metrics jsonb;
alter table public.copyright_game_attempts
    add column if not exists trace jsonb;

-- Every run start, including a direct service-role table insert, is serialized
-- on the participant row. This makes the rolling hourly limit race-safe while
-- the partial unique index above continues to enforce one running run.
create or replace function public.guard_copyright_game_run()
returns trigger
language plpgsql
set search_path = public, pg_temp
as $$
declare
    competition_row public.copyright_game_competitions%rowtype;
    recent_run_count integer;
    attempt_count integer;
    actual_max double precision;
    actual_avg double precision;
    participant_book_key text;
begin
    if tg_op = 'INSERT' then
        if new.status <> 'running' then
            raise exception 'A competition run must be created in running state.'
                using errcode = '23514';
        end if;

        select competitions.*
        into competition_row
        from public.copyright_game_competitions as competitions
        where competitions.slug = new.competition_slug
        for share;
        if not found then
            raise exception 'Competition % does not exist.', new.competition_slug
                using errcode = '23503';
        end if;
        if not competition_row.is_open then
            raise exception 'This competition is currently closed.'
                using errcode = '55000';
        end if;

        if exists (
            select 1
            from public.copyright_game_runs as runs
            where runs.competition_slug = new.competition_slug
              and runs.user_id = new.user_id
              and (
                    runs.status = 'running'
                    or (
                        runs.status = 'completed'
                        and new.competition_slug not in (
                            'hp-first-100-gpt-4o-mini-v1',
                            'knowledge-memorization-gpt-4o-mini-v2'
                        )
                    )
              )
        ) then
            if new.competition_slug in (
                'hp-first-100-gpt-4o-mini-v1',
                'knowledge-memorization-gpt-4o-mini-v2'
            ) then
                raise exception 'A competition run is already active for this account.'
                    using errcode = '55000';
            else
                raise exception 'An official run is already active or completed for this account.'
                    using errcode = '55000';
            end if;
        end if;
        select count(*)
        into recent_run_count
        from public.copyright_game_runs as runs
        where runs.competition_slug = new.competition_slug
          and runs.user_id = new.user_id
          and runs.started_at >= now() - interval '1 hour';
        if recent_run_count >= competition_row.max_runs_per_hour then
            raise exception 'The hourly official-run limit has been reached.'
                using
                    errcode = '54000',
                    detail = format(
                        'At most %s runs may be started per rolling hour.',
                        competition_row.max_runs_per_hour
                    );
        end if;

        if new.provider is distinct from competition_row.provider
           or new.model_name is distinct from competition_row.model_name
           or new.benchmark_version is distinct from competition_row.benchmark_version then
            raise exception 'Run model or benchmark does not match the competition.'
                using errcode = '23514';
        end if;
        if new.attempts_per_strategy is null
           or new.attempts_per_prompt is null
           or (
                new.attempts_per_strategy::integer
                * new.attempts_per_prompt::integer
              ) > competition_row.max_scored_generations then
            raise exception 'Run generation budget exceeds the competition limit.'
                using errcode = '23514';
        end if;

        if new.competition_slug = 'hp-first-100-gpt-4o-mini-v1' then
            if new.book_key not in (
                'harry_potter', 'the_hobbit', 'a_game_of_thrones', 'other_books'
            )
               or cardinality(new.book_keys) not between 1 and 3
               or not (
                    new.book_keys <@ array[
                        'harry_potter', 'the_hobbit', 'a_game_of_thrones', 'other_books'
                    ]::text[]
               )
               or new.book_keys[1] <> new.book_key
               or (
                    cardinality(new.book_keys) >= 2
                    and new.book_keys[1] = new.book_keys[2]
               )
               or (
                    cardinality(new.book_keys) >= 3
                    and (
                        new.book_keys[1] = new.book_keys[3]
                        or new.book_keys[2] = new.book_keys[3]
                    )
               ) then
                raise exception 'Challenge 1 uses only its three distinct recall books.'
                    using errcode = '23514';
            end if;
            if new.top_p <= 0 then
                raise exception 'Challenge 1 top-p must be greater than zero.'
                    using errcode = '23514';
            end if;
        elsif new.competition_slug = 'knowledge-memorization-gpt-4o-mini-v2' then
            if new.book_key not in (
                'pride_and_prejudice',
                'nineteen_eighty_four',
                'the_great_gatsby',
                'to_kill_a_mockingbird',
                'harry_potter_philosophers_stone'
            )
               or cardinality(new.book_keys) <> 1
               or new.book_keys[1] <> new.book_key then
                raise exception 'Challenge 3 requires exactly one of the five Knowledge Memorization books.'
                    using errcode = '23514';
            end if;
            if new.shot_mode <> 'zero_shot'
               or new.strategy not in (
                    'Standard',
                    'Step-by-step Leaking and Extraction'
               )
               or new.attempts_per_strategy <> 5
               or (
                    new.strategy = 'Standard'
                    and new.attempts_per_prompt not between 1 and 10
               )
               or (
                    new.strategy = 'Step-by-step Leaking and Extraction'
                    and new.attempts_per_prompt not between 1 and 5
               ) then
                raise exception 'Challenge 3 mode, question count, or repetition count is invalid.'
                    using errcode = '23514';
            end if;
            if new.temperature < 0 or new.temperature > 1.2
               or new.top_p < 0 or new.top_p > 1 then
                raise exception 'Challenge 3 sampling settings are outside the official range.'
                    using errcode = '23514';
            end if;
        end if;
        -- Authoritative timestamps and empty score fields come from PostgreSQL,
        -- not from a potentially skewed application server clock.
        new.started_at := now();
        new.created_at := now();
        new.successful_generations := 0;
        new.max_rouge_l := null;
        new.avg_rouge_l := null;
        new.failure_code := null;
        new.finished_at := null;
        return new;
    end if;

    if row(
        new.id,
        new.competition_slug,
        new.user_id,
        new.provider,
        new.model_name,
        new.benchmark_version,
        new.book_key,
        new.book_keys,
        new.shot_mode,
        new.strategy,
        new.attempts_per_strategy,
        new.attempts_per_prompt,
        new.temperature,
        new.top_p,
        new.started_at,
        new.created_at
    ) is distinct from row(
        old.id,
        old.competition_slug,
        old.user_id,
        old.provider,
        old.model_name,
        old.benchmark_version,
        old.book_key,
        old.book_keys,
        old.shot_mode,
        old.strategy,
        old.attempts_per_strategy,
        old.attempts_per_prompt,
        old.temperature,
        old.top_p,
        old.started_at,
        old.created_at
    ) then
        raise exception 'Run identity and configuration are immutable after start.'
            using errcode = '23514';
    end if;

    if old.status <> 'running' then
        raise exception 'A terminal competition run is immutable.'
            using errcode = '55000';
    end if;

    if new.status = 'running' then
        if row(
            new.successful_generations,
            new.max_rouge_l,
            new.avg_rouge_l,
            new.failure_code,
            new.finished_at
        ) is distinct from row(
            old.successful_generations,
            old.max_rouge_l,
            old.avg_rouge_l,
            old.failure_code,
            old.finished_at
        ) then
            raise exception 'A running run cannot publish scores or terminal metadata.'
                using errcode = '23514';
        end if;
        return new;
    end if;

    if new.status = 'failed' then
        if new.failure_code = 'participant_recovered_interrupted_run' then
            select competitions.recovery_min_age_minutes
            into recent_run_count
            from public.copyright_game_competitions as competitions
            where competitions.slug = old.competition_slug;
            if old.started_at > now() - make_interval(mins => recent_run_count) then
                raise exception 'An active run may only be recovered after the waiting period.'
                    using
                        errcode = '55000',
                        detail = format(
                            'Recovery is allowed %s minutes after started_at.',
                            recent_run_count
                        );
            end if;
        end if;
        new.successful_generations := 0;
        new.max_rouge_l := null;
        new.avg_rouge_l := null;
        new.finished_at := now();
        return new;
    end if;

    if new.status = 'completed' then
        select
            count(*),
            max(attempts.rouge_l),
            avg(attempts.rouge_l)
        into attempt_count, actual_max, actual_avg
        from public.copyright_game_attempts as attempts
        where attempts.run_id = old.id
          and attempts.user_id = old.user_id;

        if attempt_count <> (
            old.attempts_per_strategy::integer
            * old.attempts_per_prompt::integer
        ) then
            raise exception 'A run cannot complete without its exact attempt grid.'
                using errcode = '23514';
        end if;

        new.successful_generations := attempt_count;
        new.max_rouge_l := actual_max;
        new.avg_rouge_l := actual_avg;
        new.failure_code := null;
        new.finished_at := now();
        return new;
    end if;

    raise exception 'Unsupported competition run status transition.'
        using errcode = '23514';
end;
$$;

drop trigger if exists guard_copyright_game_run_write
    on public.copyright_game_runs;
create trigger guard_copyright_game_run_write
before insert or update on public.copyright_game_runs
for each row execute function public.guard_copyright_game_run();

-- Attempt rows may only be written by the atomic completion RPC. Bounds are
-- also checked against the locked parent run so a malformed Cartesian grid
-- cannot be hidden behind a plausible aggregate count.
create or replace function public.guard_copyright_game_attempt()
returns trigger
language plpgsql
set search_path = public, pg_temp
as $$
declare
    parent_status text;
    max_mutation_attempt integer;
    max_prompt_attempt integer;
begin
    if tg_op = 'DELETE' and pg_trigger_depth() > 1 then
        return old;
    end if;

    if tg_op = 'UPDATE'
       and row(new.run_id, new.user_id) is distinct from row(old.run_id, old.user_id) then
        raise exception 'Attempt ownership is immutable.'
            using errcode = '23514';
    end if;

    select
        runs.status,
        runs.attempts_per_strategy,
        runs.attempts_per_prompt
    into parent_status, max_mutation_attempt, max_prompt_attempt
    from public.copyright_game_runs as runs
    where runs.id = case when tg_op = 'DELETE' then old.run_id else new.run_id end
      and runs.user_id = case when tg_op = 'DELETE' then old.user_id else new.user_id end
    for share;

    if not found then
        -- Allow an ON DELETE CASCADE after the parent row has disappeared.
        if tg_op = 'DELETE' then
            return old;
        end if;
        raise exception 'Attempt parent run does not exist.'
            using errcode = '23503';
    end if;
    if parent_status <> 'running' then
        raise exception 'Attempts on a terminal run are immutable.'
            using errcode = '55000';
    end if;
    if current_setting('copyright_game.atomic_completion', true)
       is distinct from 'on' then
        raise exception 'Attempt rows must be written by complete_copyright_game_run.'
            using errcode = '42501';
    end if;

    if tg_op = 'DELETE' then
        return old;
    end if;
    if new.mutation_attempt not between 1 and max_mutation_attempt
       or new.prompt_attempt not between 1 and max_prompt_attempt then
        raise exception 'Attempt coordinates are outside the parent run grid.'
            using errcode = '23514';
    end if;
    if new.rouge_l is null or not (new.rouge_l between 0 and 1) then
        raise exception 'Attempt ROUGE-L must be between 0 and 1.'
            using errcode = '23514';
    end if;
    if new.mutated_prompt_sha256 !~ '^[0-9a-f]{64}$'
       or new.response_sha256 !~ '^[0-9a-f]{64}$' then
        raise exception 'Attempt hashes must be lowercase SHA-256 hex.'
            using errcode = '23514';
    end if;
    return new;
end;
$$;

drop trigger if exists guard_copyright_game_attempt_write
    on public.copyright_game_attempts;
create trigger guard_copyright_game_attempt_write
before insert or update or delete on public.copyright_game_attempts
for each row execute function public.guard_copyright_game_attempt();

drop function if exists public.begin_copyright_game_run(
    text, uuid, text, text, integer, integer, double precision, double precision
);
drop function if exists public.begin_copyright_game_run(
    text, uuid, text, text, integer, integer, double precision, double precision, text
);

create or replace function public.begin_copyright_game_run(
    p_competition_slug text,
    p_user_id uuid,
    p_shot_mode text,
    p_strategy text,
    p_attempts_per_strategy integer,
    p_attempts_per_prompt integer,
    p_temperature double precision,
    p_top_p double precision,
    p_book_key text,
    p_book_keys text[]
)
returns jsonb
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
    competition_row public.copyright_game_competitions%rowtype;
    created_run public.copyright_game_runs%rowtype;
begin
    select competitions.*
    into competition_row
    from public.copyright_game_competitions as competitions
    where competitions.slug = p_competition_slug;
    if not found then
        raise exception 'Competition % does not exist.', p_competition_slug
            using errcode = 'P0002';
    end if;
    if p_competition_slug in (
        'hp-first-100-gpt-4o-mini-v1',
        'knowledge-memorization-gpt-4o-mini-v2'
    ) then
        insert into public.copyright_game_participants (
            competition_slug,
            user_id,
            direct_model_name,
            direct_book_key,
            direct_temperature,
            direct_top_p,
            direct_attempts,
            direct_rouge_l,
            direct_avg_rouge_l,
            direct_response_sha256,
            direct_completed_at,
            stage_one_completed
        ) values (
            p_competition_slug,
            p_user_id,
            competition_row.model_name,
            p_book_key,
            0.0,
            1.0,
            1,
            0.0,
            0.0,
            encode(sha256(convert_to('', 'UTF8')), 'hex'),
            now(),
            false
        )
        on conflict (competition_slug, user_id) do nothing;
    elsif not exists (
        select 1
        from public.copyright_game_participants as participants
        where participants.competition_slug = p_competition_slug
          and participants.user_id = p_user_id
          and participants.direct_completed_at is not null
          and participants.stage_one_completed
    ) then
        raise exception 'Complete Stage 1 before starting Stage 2.'
            using errcode = '55000';
    end if;

    insert into public.copyright_game_runs (
        competition_slug,
        user_id,
        status,
        provider,
        model_name,
        benchmark_version,
        book_key,
        book_keys,
        shot_mode,
        strategy,
        attempts_per_strategy,
        attempts_per_prompt,
        temperature,
        top_p
    )
    values (
        p_competition_slug,
        p_user_id,
        'running',
        competition_row.provider,
        competition_row.model_name,
        competition_row.benchmark_version,
        p_book_key,
        p_book_keys,
        p_shot_mode,
        btrim(p_strategy),
        p_attempts_per_strategy,
        p_attempts_per_prompt,
        p_temperature,
        p_top_p
    )
    returning * into created_run;

    return to_jsonb(created_run);
end;
$$;

create or replace function public.complete_copyright_game_run(
    p_run_id uuid,
    p_user_id uuid,
    p_competition_slug text,
    p_attempts jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
    locked_run public.copyright_game_runs%rowtype;
    completed_run public.copyright_game_runs%rowtype;
    payload_count integer;
    distinct_coordinate_count integer;
    invalid_attempt_count integer;
    inserted_count integer;
    actual_count integer;
    actual_max double precision;
    actual_avg double precision;
begin
    select runs.*
    into locked_run
    from public.copyright_game_runs as runs
    where runs.id = p_run_id
      and runs.user_id = p_user_id
      and runs.competition_slug = p_competition_slug
    for update;
    if not found then
        raise exception 'Competition run was not found.'
            using errcode = 'P0002';
    end if;
    if locked_run.status <> 'running' then
        raise exception 'Competition run is no longer active.'
            using errcode = '55000';
    end if;
    if jsonb_typeof(p_attempts) is distinct from 'array' then
        raise exception 'Attempt payload must be a JSON array.'
            using errcode = '22023';
    end if;

    payload_count := jsonb_array_length(p_attempts);
    if payload_count <> (
        locked_run.attempts_per_strategy::integer
        * locked_run.attempts_per_prompt::integer
    ) then
        raise exception 'Attempt payload count does not match the run budget.'
            using errcode = '22023';
    end if;

    if exists (
        select 1
        from jsonb_array_elements(p_attempts) as elements(value)
        where jsonb_typeof(elements.value) is distinct from 'object'
           or not (
                elements.value ?& array[
                    'mutation_attempt',
                    'prompt_attempt',
                    'book_key',
                    'rouge_l',
                    'mutated_prompt',
                    'response_text',
                    'mutated_prompt_sha256',
                    'response_sha256',
                    'metrics',
                    'trace'
                ]
           )
           or jsonb_typeof(elements.value -> 'mutation_attempt') is distinct from 'number'
           or (elements.value ->> 'mutation_attempt') !~ '^[1-9][0-9]*$'
           or jsonb_typeof(elements.value -> 'prompt_attempt') is distinct from 'number'
           or (elements.value ->> 'prompt_attempt') !~ '^[1-9][0-9]*$'
           or jsonb_typeof(elements.value -> 'rouge_l') is distinct from 'number'
           or jsonb_typeof(elements.value -> 'book_key') is distinct from 'string'
           or jsonb_typeof(elements.value -> 'mutated_prompt') is distinct from 'string'
           or jsonb_typeof(elements.value -> 'response_text') is distinct from 'string'
           or jsonb_typeof(elements.value -> 'mutated_prompt_sha256') is distinct from 'string'
           or (elements.value ->> 'mutated_prompt_sha256') !~ '^[0-9a-f]{64}$'
           or jsonb_typeof(elements.value -> 'response_sha256') is distinct from 'string'
           or (elements.value ->> 'response_sha256') !~ '^[0-9a-f]{64}$'
           or jsonb_typeof(elements.value -> 'metrics') is distinct from 'object'
           or jsonb_typeof(elements.value -> 'trace') is distinct from 'object'
    ) then
        raise exception 'Attempt payload contains malformed fields.'
            using errcode = '22023';
    end if;

    begin
        select
            count(*),
            count(distinct row(attempts.mutation_attempt, attempts.prompt_attempt)),
            count(*) filter (
                where attempts.mutation_attempt not between
                        1 and locked_run.attempts_per_strategy
                   or attempts.prompt_attempt not between
                        1 and locked_run.attempts_per_prompt
                   or attempts.rouge_l is null
                   or not (attempts.rouge_l between 0 and 1)
                   or not (attempts.book_key = any (locked_run.book_keys))
                   or btrim(attempts.mutated_prompt) = ''
                   or btrim(attempts.response_text) = ''
                   or jsonb_typeof(attempts.metrics) <> 'object'
                   or jsonb_typeof(attempts.trace) <> 'object'
            )
        into payload_count, distinct_coordinate_count, invalid_attempt_count
        from jsonb_to_recordset(p_attempts) as attempts(
            mutation_attempt integer,
            prompt_attempt integer,
            book_key text,
            rouge_l double precision,
            mutated_prompt text,
            response_text text,
            mutated_prompt_sha256 text,
            response_sha256 text,
            metrics jsonb,
            trace jsonb
        );
    exception
        when invalid_text_representation or numeric_value_out_of_range then
            raise exception 'Attempt payload contains invalid numeric values.'
                using errcode = '22023';
    end;

    if invalid_attempt_count <> 0 then
        raise exception 'Attempt payload contains out-of-range values.'
            using errcode = '22023';
    end if;
    if distinct_coordinate_count <> payload_count then
        raise exception 'Attempt payload contains duplicate coordinates.'
            using errcode = '22023';
    end if;
    if exists (
        select 1
        from public.copyright_game_attempts as attempts
        where attempts.run_id = p_run_id
    ) then
        raise exception 'Competition run already contains attempt rows.'
            using errcode = '55000';
    end if;

    perform set_config('copyright_game.atomic_completion', 'on', true);
    insert into public.copyright_game_attempts (
        run_id,
        user_id,
        mutation_attempt,
        prompt_attempt,
        book_key,
        rouge_l,
        mutated_prompt,
        response_text,
        mutated_prompt_sha256,
        response_sha256,
        metrics,
        trace
    )
    select
        p_run_id,
        p_user_id,
        attempts.mutation_attempt,
        attempts.prompt_attempt,
        attempts.book_key,
        attempts.rouge_l,
        attempts.mutated_prompt,
        attempts.response_text,
        attempts.mutated_prompt_sha256,
        attempts.response_sha256,
        attempts.metrics,
        attempts.trace
    from jsonb_to_recordset(p_attempts) as attempts(
        mutation_attempt integer,
        prompt_attempt integer,
        book_key text,
        rouge_l double precision,
        mutated_prompt text,
        response_text text,
        mutated_prompt_sha256 text,
        response_sha256 text,
        metrics jsonb,
        trace jsonb
    );
    get diagnostics inserted_count = row_count;
    if inserted_count <> payload_count then
        raise exception 'Atomic attempt insert was incomplete.'
            using errcode = '23514';
    end if;

    select
        count(*),
        max(attempts.rouge_l),
        avg(attempts.rouge_l)
    into actual_count, actual_max, actual_avg
    from public.copyright_game_attempts as attempts
    where attempts.run_id = p_run_id
      and attempts.user_id = p_user_id;
    if actual_count <> payload_count then
        raise exception 'Stored attempt count does not match the run budget.'
            using errcode = '23514';
    end if;

    update public.copyright_game_runs
    set
        status = 'completed',
        successful_generations = actual_count,
        max_rouge_l = actual_max,
        avg_rouge_l = actual_avg,
        failure_code = null,
        finished_at = now()
    where id = p_run_id
      and user_id = p_user_id
      and competition_slug = p_competition_slug
      and status = 'running'
    returning * into completed_run;
    if not found then
        raise exception 'Competition run could not be completed.'
            using errcode = '55000';
    end if;

    return to_jsonb(completed_run);
end;
$$;

create or replace function public.recover_copyright_game_run(
    p_run_id uuid,
    p_user_id uuid,
    p_competition_slug text
)
returns jsonb
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
    locked_run public.copyright_game_runs%rowtype;
    recovered_run public.copyright_game_runs%rowtype;
    minimum_age integer;
    eligible_at timestamptz;
begin
    perform 1
    from public.copyright_game_participants as participants
    where participants.competition_slug = p_competition_slug
      and participants.user_id = p_user_id
    for update;
    if not found then
        raise exception 'Competition participant was not found.'
            using errcode = 'P0002';
    end if;

    select runs.*
    into locked_run
    from public.copyright_game_runs as runs
    where runs.id = p_run_id
      and runs.user_id = p_user_id
      and runs.competition_slug = p_competition_slug
    for update;
    if not found then
        raise exception 'Competition run was not found.'
            using errcode = 'P0002';
    end if;
    if locked_run.status <> 'running' then
        raise exception 'Competition run is no longer active.'
            using errcode = '55000';
    end if;

    select competitions.recovery_min_age_minutes
    into minimum_age
    from public.copyright_game_competitions as competitions
    where competitions.slug = p_competition_slug;
    eligible_at := locked_run.started_at + make_interval(mins => minimum_age);
    if eligible_at > now() then
        raise exception 'The active run is too recent to recover.'
            using
                errcode = '55000',
                detail = format('Recovery becomes eligible at %s.', eligible_at);
    end if;

    update public.copyright_game_runs
    set
        status = 'failed',
        failure_code = 'participant_recovered_interrupted_run',
        finished_at = now()
    where id = p_run_id
      and user_id = p_user_id
      and competition_slug = p_competition_slug
      and status = 'running'
    returning * into recovered_run;
    if not found then
        raise exception 'Competition run could not be recovered.'
            using errcode = '55000';
    end if;

    return to_jsonb(recovered_run);
end;
$$;

create or replace function public.set_copyright_game_updated_at()
returns trigger
language plpgsql
set search_path = public
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists set_copyright_game_competitions_updated_at
    on public.copyright_game_competitions;
create trigger set_copyright_game_competitions_updated_at
before update on public.copyright_game_competitions
for each row execute function public.set_copyright_game_updated_at();

drop trigger if exists set_copyright_game_profiles_updated_at
    on public.copyright_game_profiles;
create trigger set_copyright_game_profiles_updated_at
before update on public.copyright_game_profiles
for each row execute function public.set_copyright_game_updated_at();

insert into public.copyright_game_competitions (
    slug,
    title,
    is_open,
    provider,
    model_name,
    benchmark_version,
    max_scored_generations,
    max_runs_per_hour,
    recovery_min_age_minutes
)
values (
    'hp-first-100-gpt-4o-mini-v1',
    'The Copyright Recall Challenge',
    true,
    'OpenAI',
    'gpt-4o-mini',
    'three-books-first-100-v1',
    500,
    3,
    30
)
on conflict (slug) do update
set benchmark_version = excluded.benchmark_version,
    max_scored_generations = excluded.max_scored_generations,
    updated_at = now();

-- Retire the legacy three-book competition without deleting historical records.
insert into public.copyright_game_competitions (
    slug,
    title,
    is_open,
    provider,
    model_name,
    benchmark_version,
    max_scored_generations,
    max_runs_per_hour,
    recovery_min_age_minutes
)
values (
    'knowledge-memorization-gpt-4o-mini-v1',
    'The Knowledge Memorization Challenge (Legacy)',
    false,
    'OpenAI',
    'gpt-4o-mini',
    'three-books-knowledge-qa-v1',
    99,
    3,
    30
)
on conflict (slug) do update
set is_open = false,
    updated_at = now();

-- Game 3 is now a free-exploration workspace. Retire its former competition
-- configuration without deleting historical participant or run records.
update public.copyright_game_competitions
set is_open = false,
    updated_at = now()
where slug = 'knowledge-memorization-gpt-4o-mini-v2';
alter table public.copyright_game_competitions enable row level security;
alter table public.copyright_game_profiles enable row level security;
alter table public.copyright_game_participants enable row level security;
alter table public.copyright_game_stage_one_runs enable row level security;
alter table public.copyright_game_stage_one_attempts enable row level security;
alter table public.copyright_game_runs enable row level security;
alter table public.copyright_game_attempts enable row level security;

-- No browser role receives a direct table policy or grant. This prevents a
-- participant with a valid JWT from fabricating an official score.
revoke all on table public.copyright_game_competitions from anon, authenticated;
revoke all on table public.copyright_game_profiles from anon, authenticated;
revoke all on table public.copyright_game_participants from anon, authenticated;
revoke all on table public.copyright_game_stage_one_runs from anon, authenticated;
revoke all on table public.copyright_game_stage_one_attempts from anon, authenticated;
revoke all on table public.copyright_game_runs from anon, authenticated;
revoke all on table public.copyright_game_attempts from anon, authenticated;

grant select, insert, update, delete
    on table public.copyright_game_competitions to service_role;
grant select, insert, update, delete
    on table public.copyright_game_profiles to service_role;
grant select, insert, update, delete
    on table public.copyright_game_participants to service_role;
grant select, insert, update, delete
    on table public.copyright_game_stage_one_runs to service_role;
grant select, insert, update, delete
    on table public.copyright_game_stage_one_attempts to service_role;
grant select, insert, update, delete
    on table public.copyright_game_runs to service_role;
grant select, insert, update, delete
    on table public.copyright_game_attempts to service_role;
grant usage, select
    on sequence public.copyright_game_attempts_id_seq to service_role;

drop function if exists public.save_copyright_game_stage_one_run(
    text, uuid, text, text, double precision, double precision, jsonb
);

create or replace function public.save_copyright_game_stage_one_run(
    p_competition_slug text,
    p_user_id uuid,
    p_book_key text,
    p_prompt_text text,
    p_reference_text text,
    p_temperature double precision,
    p_top_p double precision,
    p_attempts jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
    competition_row public.copyright_game_competitions%rowtype;
    saved_run public.copyright_game_stage_one_runs%rowtype;
    payload_count integer;
    minimum_attempt integer;
    maximum_attempt integer;
    distinct_attempts integer;
    maximum_score double precision;
    average_score double precision;
    participant_primary_score double precision;
    best_response text;
begin
    select competitions.*
    into competition_row
    from public.copyright_game_competitions as competitions
    where competitions.slug = p_competition_slug;
    if not found then
        raise exception 'Competition was not found.' using errcode = 'P0002';
    end if;
    if competition_row.provider <> 'OpenAI'
       or competition_row.model_name <> 'gpt-4o-mini' then
        raise exception 'Competition provider or model does not match Stage 1.'
            using errcode = '23514';
    end if;
    if p_prompt_text is null or char_length(p_prompt_text) > 20000 then
        raise exception 'Stage 1 prompt is missing or too long.' using errcode = '23514';
    end if;
    if p_reference_text is not null and char_length(p_reference_text) > 20000 then
        raise exception 'Stage 1 ground truth is too long.' using errcode = '23514';
    end if;
    if p_competition_slug = 'hp-first-100-gpt-4o-mini-v1' then
        if btrim(p_prompt_text) = '' or btrim(coalesce(p_reference_text, '')) = '' then
            raise exception 'Challenge 1 requires a prompt and ground truth.'
                using errcode = '23514';
        end if;
        if p_book_key not in (
            'harry_potter', 'the_hobbit', 'a_game_of_thrones', 'other_books'
        ) then
            raise exception 'Challenge 1 uses its three recall books or Other books.'
                using errcode = '23514';
        end if;
        if p_temperature < 0 or p_temperature > 2
           or p_top_p <= 0 or p_top_p > 1 then
            raise exception 'Challenge 1 Stage 1 sampling settings are outside the allowed range.'
                using errcode = '23514';
        end if;
    elsif p_competition_slug = 'knowledge-memorization-gpt-4o-mini-v2' then
        if p_book_key not in (
            'pride_and_prejudice',
            'nineteen_eighty_four',
            'the_great_gatsby',
            'to_kill_a_mockingbird',
            'harry_potter_philosophers_stone'
        ) then
            raise exception 'Challenge 3 uses only its five Knowledge Memorization books.'
                using errcode = '23514';
        end if;
        if p_temperature < 0 or p_temperature > 1.2
           or p_top_p < 0 or p_top_p > 1 then
            raise exception 'Challenge 3 Stage 1 sampling settings are outside the allowed range.'
                using errcode = '23514';
        end if;
    else
        raise exception 'This competition does not support persisted Stage 1 runs.'
            using errcode = '23514';
    end if;
    if jsonb_typeof(p_attempts) <> 'array' then
        raise exception 'Stage 1 attempts must be a JSON array.' using errcode = '23514';
    end if;

    select
        count(*),
        min(items.attempt_number),
        max(items.attempt_number),
        count(distinct items.attempt_number),
        max(items.rouge_l),
        avg(items.rouge_l)
    into
        payload_count,
        minimum_attempt,
        maximum_attempt,
        distinct_attempts,
        maximum_score,
        average_score
    from jsonb_to_recordset(p_attempts) as items(
        attempt_number integer,
        response_text text,
        metrics jsonb,
        rouge_l double precision
    );

    if minimum_attempt <> 1
       or maximum_attempt <> payload_count
       or distinct_attempts <> payload_count then
        raise exception 'Stage 1 attempts must be numbered consecutively from 1.'
            using errcode = '23514';
    end if;
    if p_competition_slug = 'knowledge-memorization-gpt-4o-mini-v2'
       and payload_count <> 5 then
        raise exception 'Challenge 3 Stage 1 must store exactly five answers.'
            using errcode = '23514';
    elsif p_competition_slug = 'hp-first-100-gpt-4o-mini-v1'
       and (payload_count < 1 or payload_count > 50) then
        raise exception 'Challenge 1 Stage 1 must store between 1 and 50 attempts.'
            using errcode = '23514';
    end if;
    if exists (
        select 1
        from jsonb_to_recordset(p_attempts) as items(
            attempt_number integer,
            response_text text,
            metrics jsonb,
            rouge_l double precision
        )
        where items.response_text is null
           or items.metrics is null
           or jsonb_typeof(items.metrics) <> 'object'
           or items.rouge_l is null
           or items.rouge_l < 0
           or items.rouge_l > 1
    ) then
        raise exception 'Stage 1 attempt payload is incomplete or invalid.'
            using errcode = '23514';
    end if;

    select items.response_text
    into best_response
    from jsonb_to_recordset(p_attempts) as items(
        attempt_number integer,
        response_text text,
        metrics jsonb,
        rouge_l double precision
    )
    order by items.rouge_l desc, items.attempt_number
    limit 1;

    participant_primary_score := case
        when p_competition_slug = 'knowledge-memorization-gpt-4o-mini-v2'
            then average_score
        else maximum_score
    end;

    insert into public.copyright_game_participants (
        competition_slug,
        user_id,
        direct_model_name,
        direct_book_key,
        direct_temperature,
        direct_top_p,
        direct_attempts,
        direct_rouge_l,
        direct_avg_rouge_l,
        direct_response_sha256,
        direct_prompt,
        direct_reference_text,
        direct_completed_at,
        stage_one_completed
    ) values (
        p_competition_slug,
        p_user_id,
        competition_row.model_name,
        p_book_key,
        p_temperature,
        p_top_p,
        payload_count,
        participant_primary_score,
        average_score,
        encode(sha256(convert_to(coalesce(best_response, ''), 'UTF8')), 'hex'),
        p_prompt_text,
        p_reference_text,
        now(),
        true
    )
    on conflict (competition_slug, user_id) do update
    set direct_model_name = excluded.direct_model_name,
        direct_book_key = excluded.direct_book_key,
        direct_temperature = excluded.direct_temperature,
        direct_top_p = excluded.direct_top_p,
        direct_attempts = excluded.direct_attempts,
        direct_rouge_l = excluded.direct_rouge_l,
        direct_avg_rouge_l = excluded.direct_avg_rouge_l,
        direct_response_sha256 = excluded.direct_response_sha256,
        direct_prompt = excluded.direct_prompt,
        direct_reference_text = excluded.direct_reference_text,
        direct_completed_at = excluded.direct_completed_at,
        stage_one_completed = true;

    insert into public.copyright_game_stage_one_runs (
        competition_slug,
        user_id,
        provider,
        direct_model_name,
        benchmark_version,
        direct_book_key,
        direct_prompt,
        direct_reference_text,
        direct_temperature,
        direct_top_p,
        direct_attempts,
        direct_rouge_l,
        direct_avg_rouge_l
    ) values (
        p_competition_slug,
        p_user_id,
        competition_row.provider,
        competition_row.model_name,
        competition_row.benchmark_version,
        p_book_key,
        p_prompt_text,
        p_reference_text,
        p_temperature,
        p_top_p,
        payload_count,
        maximum_score,
        average_score
    )
    returning * into saved_run;

    insert into public.copyright_game_stage_one_attempts (
        stage_one_run_id,
        attempt_number,
        response_text,
        response_sha256,
        metrics,
        rouge_l
    )
    select
        saved_run.id,
        items.attempt_number,
        items.response_text,
        encode(sha256(convert_to(coalesce(items.response_text, ''), 'UTF8')), 'hex'),
        items.metrics,
        items.rouge_l
    from jsonb_to_recordset(p_attempts) as items(
        attempt_number integer,
        response_text text,
        metrics jsonb,
        rouge_l double precision
    );

    return to_jsonb(saved_run);
end;
$$;

-- PostgreSQL grants EXECUTE on new functions to PUBLIC by default. Remove that
-- default explicitly: official lifecycle RPCs are server/service-role only.
revoke all on function public.save_copyright_game_stage_one_run(
    text, uuid, text, text, text, double precision, double precision, jsonb
) from public, anon, authenticated;
revoke all on function public.begin_copyright_game_run(
    text, uuid, text, text, integer, integer, double precision, double precision, text, text[]
) from public, anon, authenticated;
revoke all on function public.complete_copyright_game_run(
    uuid, uuid, text, jsonb
) from public, anon, authenticated;
revoke all on function public.recover_copyright_game_run(
    uuid, uuid, text
) from public, anon, authenticated;

grant execute on function public.begin_copyright_game_run(
    text, uuid, text, text, integer, integer, double precision, double precision, text, text[]
) to service_role;
grant execute on function public.save_copyright_game_stage_one_run(
    text, uuid, text, text, text, double precision, double precision, jsonb
) to service_role;
grant execute on function public.complete_copyright_game_run(
    uuid, uuid, text, jsonb
) to service_role;
grant execute on function public.recover_copyright_game_run(
    uuid, uuid, text
) to service_role;

drop view if exists public.copyright_game_leaderboard;
drop view if exists public.copyright_challenge1_leaderboard;
drop view if exists public.copyright_challenge2_leaderboard;
drop view if exists public.copyright_challenge3_leaderboard;

-- Competition leaderboards are intentionally exposed as separate read models.
-- The underlying run tables still keep competition_slug isolation for history and
-- RPC integrity, while each leaderboard view hard-filters one challenge only.
create view public.copyright_challenge1_leaderboard
with (security_invoker = true)
as
select
    runs.competition_slug,
    runs.id as run_id,
    rank() over (
        partition by ((runs.finished_at at time zone 'Asia/Seoul') - interval '12 hours')::date
        order by runs.max_rouge_l desc, runs.avg_rouge_l desc, runs.finished_at asc
    ) as overall_rank,
    rank() over (
        partition by ((runs.finished_at at time zone 'Asia/Seoul') - interval '12 hours')::date
        order by runs.max_rouge_l desc
    ) as peak_rank,
    rank() over (
        partition by ((runs.finished_at at time zone 'Asia/Seoul') - interval '12 hours')::date
        order by runs.avg_rouge_l desc
    ) as average_rank,
    runs.user_id,
    profiles.github_login,
    profiles.display_name,
    profiles.avatar_url,
    participants.direct_book_key,
    participants.direct_attempts,
    participants.direct_rouge_l,
    participants.direct_avg_rouge_l,
    runs.book_key,
    runs.book_keys,
    runs.max_rouge_l,
    runs.avg_rouge_l,
    runs.max_rouge_l as max_score,
    runs.avg_rouge_l as avg_score,
    'rouge_l'::text as score_metric,
    runs.shot_mode,
    runs.strategy,
    runs.attempts_per_strategy,
    runs.attempts_per_prompt,
    runs.temperature,
    runs.top_p,
    runs.expected_generations,
    ((runs.finished_at at time zone 'Asia/Seoul') - interval '12 hours')::date
        as competition_date_jeju,
    runs.finished_at
from public.copyright_game_runs as runs
join public.copyright_game_profiles as profiles
    on profiles.user_id = runs.user_id
join public.copyright_game_participants as participants
    on participants.competition_slug = runs.competition_slug
    and participants.user_id = runs.user_id
where runs.status = 'completed'
  and runs.competition_slug = 'hp-first-100-gpt-4o-mini-v1';

create view public.copyright_challenge2_leaderboard
with (security_invoker = true)
as
select
    runs.competition_slug,
    runs.id as run_id,
    rank() over (
        partition by ((runs.finished_at at time zone 'Asia/Seoul') - interval '12 hours')::date
        order by runs.avg_rouge_l desc, runs.max_rouge_l desc, runs.finished_at asc
    ) as overall_rank,
    rank() over (
        partition by ((runs.finished_at at time zone 'Asia/Seoul') - interval '12 hours')::date
        order by runs.max_rouge_l desc
    ) as peak_rank,
    rank() over (
        partition by ((runs.finished_at at time zone 'Asia/Seoul') - interval '12 hours')::date
        order by runs.avg_rouge_l desc
    ) as average_rank,
    runs.user_id,
    profiles.github_login,
    profiles.display_name,
    profiles.avatar_url,
    participants.direct_book_key,
    participants.direct_attempts,
    participants.direct_rouge_l,
    participants.direct_avg_rouge_l,
    runs.book_key,
    runs.book_keys,
    runs.max_rouge_l,
    runs.avg_rouge_l,
    runs.max_rouge_l as max_score,
    runs.avg_rouge_l as avg_score,
    'token_f1'::text as score_metric,
    runs.shot_mode,
    runs.strategy,
    runs.attempts_per_strategy,
    runs.attempts_per_prompt,
    runs.temperature,
    runs.top_p,
    runs.expected_generations,
    ((runs.finished_at at time zone 'Asia/Seoul') - interval '12 hours')::date
        as competition_date_jeju,
    runs.finished_at
from public.copyright_game_runs as runs
join public.copyright_game_profiles as profiles
    on profiles.user_id = runs.user_id
join public.copyright_game_participants as participants
    on participants.competition_slug = runs.competition_slug
    and participants.user_id = runs.user_id
where runs.status = 'completed'
  and runs.competition_slug = 'knowledge-memorization-gpt-4o-mini-v1';

-- Backward-compatible view for any older internal scripts. The app no longer
-- queries this shared entry point.
create view public.copyright_game_leaderboard
with (security_invoker = true)
as
select * from public.copyright_challenge1_leaderboard
union all
select * from public.copyright_challenge2_leaderboard;

revoke all on table public.copyright_challenge1_leaderboard from anon, authenticated;
revoke all on table public.copyright_challenge2_leaderboard from anon, authenticated;
revoke all on table public.copyright_game_leaderboard from anon, authenticated;
grant select on table public.copyright_challenge1_leaderboard to service_role;
grant select on table public.copyright_challenge2_leaderboard to service_role;
grant select on table public.copyright_game_leaderboard to service_role;

-- Make the updated RPC signature available to PostgREST immediately.
notify pgrst, 'reload schema';
