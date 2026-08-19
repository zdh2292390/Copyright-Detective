-- Run this entire file in Supabase SQL Editor.
-- It is safe to run again after deployments.

begin;

create table if not exists public.user_settings (
    user_id uuid primary key references auth.users (id) on delete cascade,
    encrypted_api_key text not null,
    provider text not null default 'OpenAI',
    model_name text,
    extra_config jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- Keep an older installation aligned with the current application contract.
alter table public.user_settings
    add column if not exists provider text,
    add column if not exists model_name text,
    add column if not exists extra_config jsonb,
    add column if not exists created_at timestamptz,
    add column if not exists updated_at timestamptz;

update public.user_settings
set provider = 'OpenAI'
where provider is null or btrim(provider) = '';

update public.user_settings
set extra_config = '{}'::jsonb
where extra_config is null;

update public.user_settings
set created_at = now()
where created_at is null;

update public.user_settings
set updated_at = now()
where updated_at is null;

alter table public.user_settings
    alter column provider set default 'OpenAI',
    alter column provider set not null,
    alter column extra_config set default '{}'::jsonb,
    alter column extra_config set not null,
    alter column created_at set default now(),
    alter column created_at set not null,
    alter column updated_at set default now(),
    alter column updated_at set not null;

alter table public.user_settings
    drop constraint if exists user_settings_extra_config_object_check;
alter table public.user_settings
    add constraint user_settings_extra_config_object_check
    check (jsonb_typeof(extra_config) = 'object') not valid;

-- Existing legacy plain: rows remain readable so users can migrate them by
-- clicking Keep after FERNET_KEY is configured. New plaintext writes are blocked.
alter table public.user_settings
    drop constraint if exists user_settings_encrypted_api_key_not_plain_check;
alter table public.user_settings
    add constraint user_settings_encrypted_api_key_not_plain_check
    check (encrypted_api_key not like 'plain:%') not valid;

create or replace function public.set_user_settings_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = public
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists set_user_settings_updated_at
    on public.user_settings;
create trigger set_user_settings_updated_at
before update on public.user_settings
for each row execute function public.set_user_settings_updated_at();

alter table public.user_settings enable row level security;

drop policy if exists "Users can view own settings" on public.user_settings;
create policy "Users can view own settings"
    on public.user_settings
    for select
    to authenticated
    using ((select auth.uid()) = user_id);

drop policy if exists "Users can insert own settings" on public.user_settings;
create policy "Users can insert own settings"
    on public.user_settings
    for insert
    to authenticated
    with check ((select auth.uid()) = user_id);

drop policy if exists "Users can update own settings" on public.user_settings;
create policy "Users can update own settings"
    on public.user_settings
    for update
    to authenticated
    using ((select auth.uid()) = user_id)
    with check ((select auth.uid()) = user_id);

drop policy if exists "Users can delete own settings" on public.user_settings;
create policy "Users can delete own settings"
    on public.user_settings
    for delete
    to authenticated
    using ((select auth.uid()) = user_id);

revoke all on table public.user_settings from anon;
revoke all on table public.user_settings from authenticated;
grant select, insert, update, delete on table public.user_settings to authenticated;
grant all on table public.user_settings to service_role;

revoke all on function public.set_user_settings_updated_at() from public;
grant execute on function public.set_user_settings_updated_at() to authenticated;
grant execute on function public.set_user_settings_updated_at() to service_role;

commit;
