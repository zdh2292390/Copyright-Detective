-- Incremental Saved analyses upgrade for an existing checkpoint installation.
-- Run in the same Supabase project as SUPABASE_URL, after analysis_checkpoints.sql.
-- API/model/current-step summaries use existing metadata.display JSONB; no new
-- columns, statuses, backfill, or Kill RPC are needed. Existing tasks fall back
-- to recent analysis_items; the current analysis_request_stop RPC retains leases.
-- If check_analysis_recovery.sql reports missing columns, FK, grants or RPCs,
-- reapply the full analysis_checkpoints.sql before running this optional index upgrade.
-- This script is rerunnable and leaves tasks, checkpoints and official scores intact.
begin;

create index if not exists analysis_items_task_updated_idx
    on public.analysis_items(task_id, owner_id, updated_at desc, item_index desc);

notify pgrst, 'reload schema';
commit;
