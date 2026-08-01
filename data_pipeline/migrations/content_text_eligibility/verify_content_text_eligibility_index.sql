\set ON_ERROR_STOP on

SELECT
    indexrelid::regclass AS index_name,
    indisready AS is_ready,
    indisvalid AS is_valid,
    pg_get_indexdef(indexrelid) AS index_definition,
    pg_get_expr(indpred, indrelid) AS partial_predicate,
    EXISTS (
        SELECT 1
        FROM pipeline_migrations
        WHERE migration_name = 'batched_export_eligibility_v2'
    ) AS migration_recorded
FROM pg_index
WHERE indexrelid = 'public.idx_jobs_export_eligible_flag_recent'::regclass;

SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'jobs'
  AND column_name = 'is_export_eligible';

SELECT
    tgname AS trigger_name,
    tgenabled AS trigger_enabled,
    pg_get_triggerdef(oid) AS trigger_definition
FROM pg_trigger
WHERE tgrelid = 'public.jobs'::regclass
  AND tgname = 'trg_jobs_set_export_eligibility'
  AND NOT tgisinternal;

SELECT count(*) AS eligibility_rows_still_null
FROM public.jobs
WHERE is_export_eligible IS NULL;
