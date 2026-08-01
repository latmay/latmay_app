\set ON_ERROR_STOP on

\echo Creating the Latmay content-text eligibility index concurrently...

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_jobs_export_eligible_recent
ON public.jobs (posted_at_utc DESC, id DESC)
WHERE is_active = TRUE
  AND NULLIF(btrim(content_text), '') IS NOT NULL
  AND lower(btrim(content_text)) NOT IN ('nan', 'none', 'null')
  AND posted_at_utc IS NOT NULL
  AND location_parse_status IN (
        'parsed',
        'country_only',
        'remote',
        'multi_location',
        'city_resolved'
  );

DO $migration_validation$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_index
        WHERE indexrelid = 'public.idx_jobs_export_eligible_recent'::regclass
          AND indisready
          AND indisvalid
    ) THEN
        RAISE EXCEPTION
            'idx_jobs_export_eligible_recent exists but is not ready and valid';
    END IF;
END
$migration_validation$;

INSERT INTO pipeline_migrations (migration_name)
VALUES ('content_text_eligibility_partial_index_v1')
ON CONFLICT (migration_name) DO NOTHING;

\echo Content-text eligibility index is ready, valid, and recorded.
