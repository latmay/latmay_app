\set ON_ERROR_STOP on
\set eligibility_batch_size 5000

\echo Adding the stored export-eligibility column and maintenance trigger...

ALTER TABLE public.jobs
ADD COLUMN IF NOT EXISTS is_export_eligible BOOLEAN;

DO $column_validation$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'jobs'
          AND column_name = 'is_export_eligible'
          AND is_generated = 'NEVER'
    ) THEN
        RAISE EXCEPTION
            'jobs.is_export_eligible exists but is not a normal writable Boolean column';
    END IF;
END
$column_validation$;

CREATE OR REPLACE FUNCTION public.calculate_job_export_eligibility(
    job_is_active BOOLEAN,
    job_content_text TEXT,
    job_posted_at_utc TIMESTAMPTZ,
    job_location_parse_status TEXT
)
RETURNS BOOLEAN
LANGUAGE SQL
IMMUTABLE
PARALLEL SAFE
AS $eligibility_function$
    SELECT
        job_is_active = TRUE
        AND NULLIF(btrim(job_content_text), '') IS NOT NULL
        AND lower(btrim(job_content_text)) NOT IN ('nan', 'none', 'null')
        AND job_posted_at_utc IS NOT NULL
        AND job_location_parse_status IN (
            'parsed',
            'country_only',
            'remote',
            'multi_location',
            'city_resolved'
        );
$eligibility_function$;

CREATE OR REPLACE FUNCTION public.set_job_export_eligibility()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $trigger_function$
BEGIN
    NEW.is_export_eligible := public.calculate_job_export_eligibility(
        NEW.is_active,
        NEW.content_text,
        NEW.posted_at_utc,
        NEW.location_parse_status
    );
    RETURN NEW;
END
$trigger_function$;

DROP TRIGGER IF EXISTS trg_jobs_set_export_eligibility ON public.jobs;

CREATE TRIGGER trg_jobs_set_export_eligibility
BEFORE INSERT OR UPDATE OF
    is_active,
    content_text,
    posted_at_utc,
    location_parse_status,
    is_export_eligible
ON public.jobs
FOR EACH ROW
EXECUTE FUNCTION public.set_job_export_eligibility();

\echo Creating the resumable backfill index concurrently...

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_jobs_export_eligibility_backfill_pending
ON public.jobs (id)
WHERE is_export_eligible IS NULL;

CREATE OR REPLACE PROCEDURE public.backfill_job_export_eligibility(batch_size INTEGER)
LANGUAGE plpgsql
AS $backfill_procedure$
DECLARE
    batch_count BIGINT;
    remaining_count BIGINT;
    completed_count BIGINT := 0;
    initial_count BIGINT;
    batch_number BIGINT := 0;
BEGIN
    SELECT count(*)
    INTO initial_count
    FROM public.jobs
    WHERE is_export_eligible IS NULL;

    RAISE NOTICE
        'Eligibility backfill starting: remaining=%, batch_size=%',
        initial_count,
        batch_size;

    LOOP
        WITH batch AS MATERIALIZED (
            SELECT id
            FROM public.jobs
            WHERE is_export_eligible IS NULL
            ORDER BY id
            LIMIT batch_size
            FOR UPDATE SKIP LOCKED
        )
        UPDATE public.jobs AS jobs
        SET is_export_eligible = public.calculate_job_export_eligibility(
            jobs.is_active,
            jobs.content_text,
            jobs.posted_at_utc,
            jobs.location_parse_status
        )
        FROM batch
        WHERE jobs.id = batch.id;

        GET DIAGNOSTICS batch_count = ROW_COUNT;
        EXIT WHEN batch_count = 0;

        batch_number := batch_number + 1;
        completed_count := completed_count + batch_count;

        SELECT count(*)
        INTO remaining_count
        FROM public.jobs
        WHERE is_export_eligible IS NULL;

        RAISE NOTICE
            'Eligibility backfill batch % committed: batch_rows=%, completed_this_run=%, remaining=%',
            batch_number,
            batch_count,
            completed_count,
            remaining_count;

        COMMIT;
    END LOOP;

    RAISE NOTICE
        'Eligibility backfill complete: batches=%, completed_this_run=%, remaining=0',
        batch_number,
        completed_count;
END
$backfill_procedure$;

\echo Backfilling existing jobs in committed batches...

CALL public.backfill_job_export_eligibility(:eligibility_batch_size);

DROP PROCEDURE public.backfill_job_export_eligibility(INTEGER);

DO $backfill_validation$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM public.jobs
        WHERE is_export_eligible IS NULL
    ) THEN
        RAISE EXCEPTION 'Eligibility backfill finished with NULL values remaining';
    END IF;
END
$backfill_validation$;

\echo Creating the stored-eligibility partial index concurrently...

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_jobs_export_eligible_flag_recent
ON public.jobs (posted_at_utc DESC, id DESC)
WHERE is_export_eligible = TRUE;

DO $index_validation$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_index
        WHERE indexrelid = 'public.idx_jobs_export_eligible_flag_recent'::regclass
          AND indisready
          AND indisvalid
    ) THEN
        RAISE EXCEPTION
            'idx_jobs_export_eligible_flag_recent exists but is not ready and valid';
    END IF;
END
$index_validation$;

DROP INDEX CONCURRENTLY IF EXISTS public.idx_jobs_export_eligible_recent;
DROP INDEX CONCURRENTLY IF EXISTS public.idx_jobs_export_eligibility_backfill_pending;

INSERT INTO pipeline_migrations (migration_name)
VALUES ('batched_export_eligibility_v2')
ON CONFLICT (migration_name) DO NOTHING;

\echo Stored export eligibility is backfilled, indexed, trigger-maintained, and recorded.
