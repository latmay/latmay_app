# Content-text eligibility index

This folder is the source of truth for the definition and operational history
of the `jobs` content-text eligibility column and index.

## What “usable content text” means

A job is eligible only when `content_text` is not SQL `NULL`, is not empty or
whitespace-only, and is not one of the case-insensitive placeholder values
`nan`, `none`, or `null`. Export eligibility additionally requires an active
job, a normalized posting timestamp, and an accepted location parse status.

Migration 001 introduced the expression-based partial index. Migration 002
supersedes it with the stored `is_export_eligible` Boolean, a PostgreSQL trigger
that maintains it, and a partial index over that Boolean. Existing rows are
backfilled in committed, resumable batches with progress notices. PostgreSQL
automatically recalculates the value after relevant inserts and updates. No
application-maintained eligibility state is required.

## Apply once with psql

Run this only against the intended Latmay database and outside an explicit
transaction. `CREATE INDEX CONCURRENTLY` cannot run inside `BEGIN`/`COMMIT`.

```text
\i data_pipeline/migrations/content_text_eligibility/002_add_batched_export_eligibility.sql
```

The script adds the stored column, installs its maintenance trigger, creates a
temporary resumability index, backfills 5,000 rows per committed batch, creates
the final index concurrently, removes temporary/superseded indexes, verifies
readiness, and records `batched_export_eligibility_v2` in
`pipeline_migrations`. It is deliberately not run by normal container startup.
Rerunning after interruption skips rows already committed by earlier batches.

To monitor the build from a second connection:

```sql
SELECT pid, phase, blocks_total, blocks_done, tuples_total, tuples_done
FROM pg_stat_progress_create_index;
```

## Verify later

```text
\i data_pipeline/migrations/content_text_eligibility/verify_content_text_eligibility_index.sql
```

The verification output should show `is_ready`, `is_valid`, and
`migration_recorded` as true. If an interrupted concurrent build leaves an
invalid index, do not assume `IF NOT EXISTS` repaired it: inspect it, drop that
specific index concurrently, and rerun the migration.

The matching application queries live in:

- `data_pipeline/enrichment/add_job_features.py`
- `data_pipeline/export/export_recent_jobs_to_gcs.py`

Keep their eligibility predicate and `posted_at_utc DESC, id DESC` ordering
aligned with the index definition in this folder.
