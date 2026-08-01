from __future__ import annotations

"""
Database schema creation for the unified PostgreSQL-backed Latmay pipeline.

This module creates one jobs table used by all sources. Source-specific data is
kept in raw_json, while common fields are normalized into first-class columns.
Schema updates are additive so existing PostgreSQL databases can be migrated
safely by rerunning the pipeline.
"""

import psycopg

from data_pipeline.common.model_loader import get_enrichment_version


def schema_step(message: str) -> None:
    print(f"schema: {message}", flush=True)


def execute_schema_step(cur: psycopg.Cursor, message: str, sql: str, params=None) -> None:
    schema_step(f"start {message}")
    cur.execute(sql, params)
    schema_step(f"done {message}")


def initialize_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        execute_schema_step(
            cur,
            "create jobs table",
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id BIGSERIAL PRIMARY KEY,
                source_type TEXT NOT NULL,
                source_url TEXT,
                source_job_id TEXT NOT NULL,
                source_company TEXT,
                company_name TEXT,
                title TEXT,
                location_name TEXT,
                location_country TEXT,
                location_region TEXT,
                location_segments JSONB,
                work_arrangement TEXT,
                location_parse_status TEXT,
                location_normalization_version TEXT,
                location_normalized_at_utc TIMESTAMPTZ,
                department_names TEXT,
                office_names TEXT,
                posted_at TEXT,
                posted_at_utc TIMESTAMPTZ,
                updated_at TEXT,
                job_url TEXT,
                apply_url TEXT,
                content_html TEXT,
                content_text TEXT,
                content_text_clean TEXT,
                content_text_clean_version SMALLINT,
                extracted_requirements TEXT,
                requirements_extraction_version TEXT,
                requirements_extracted_at_utc TIMESTAMPTZ,
                years_experience_raw TEXT,
                min_years_experience DOUBLE PRECISION,
                max_years_experience DOUBLE PRECISION,
                experience_type TEXT,
                evidence_text TEXT,
                yoe_extraction_version TEXT,
                yoe_extracted_at_utc TIMESTAMPTZ,
                requires_clearance BOOLEAN,
                clearance_type TEXT,
                clearance_evidence_text TEXT,
                clearance_extraction_version TEXT,
                clearance_extracted_at_utc TIMESTAMPTZ,
                job_word_tokens JSONB,
                job_selected_words JSONB,
                job_selected_word_embeddings JSONB,
                job_phrase_chunks JSONB,
                job_phrase_chunk_embeddings JSONB,
                title_requirements_text TEXT,
                title_requirements_embedding JSONB,
                content_embedding JSONB,
                embedding_model_name TEXT,
                embedding_model_revision TEXT,
                embedding_dim INTEGER,
                embedded_at_utc TIMESTAMPTZ,
                enrichment_version TEXT,
                enrichment_ml_version TEXT,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                is_export_eligible BOOLEAN,
                missing_from_source_at_utc TIMESTAMPTZ,
                content_fetch_failed_count INTEGER NOT NULL DEFAULT 0,
                content_fetch_last_failed_at_utc TIMESTAMPTZ,
                content_fetch_last_error_type TEXT,
                last_seen_at_utc TIMESTAMPTZ,
                first_seen_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
                stale_reason TEXT,
                raw_json JSONB,
                fetched_at_utc TIMESTAMPTZ,
                created_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (source_type, source_job_id)
            )
            """,
        )
        jobs_column_alters = [
            "ADD COLUMN IF NOT EXISTS source_company TEXT",
            "ADD COLUMN IF NOT EXISTS posted_at_utc TIMESTAMPTZ",
            "ADD COLUMN IF NOT EXISTS location_country TEXT",
            "ADD COLUMN IF NOT EXISTS location_region TEXT",
            "ADD COLUMN IF NOT EXISTS location_segments JSONB",
            "ADD COLUMN IF NOT EXISTS work_arrangement TEXT",
            "ADD COLUMN IF NOT EXISTS location_parse_status TEXT",
            "ADD COLUMN IF NOT EXISTS location_normalization_version TEXT",
            "ADD COLUMN IF NOT EXISTS location_normalized_at_utc TIMESTAMPTZ",
            "ADD COLUMN IF NOT EXISTS content_text_clean TEXT",
            "ADD COLUMN IF NOT EXISTS content_text_clean_version SMALLINT",
            "ADD COLUMN IF NOT EXISTS requirements_extraction_version TEXT",
            "ADD COLUMN IF NOT EXISTS requirements_extracted_at_utc TIMESTAMPTZ",
            "ADD COLUMN IF NOT EXISTS yoe_extraction_version TEXT",
            "ADD COLUMN IF NOT EXISTS yoe_extracted_at_utc TIMESTAMPTZ",
            "ADD COLUMN IF NOT EXISTS clearance_extraction_version TEXT",
            "ADD COLUMN IF NOT EXISTS clearance_extracted_at_utc TIMESTAMPTZ",
            "ADD COLUMN IF NOT EXISTS job_word_tokens JSONB",
            "ADD COLUMN IF NOT EXISTS job_selected_words JSONB",
            "ADD COLUMN IF NOT EXISTS job_selected_word_embeddings JSONB",
            "ADD COLUMN IF NOT EXISTS job_phrase_chunks JSONB",
            "ADD COLUMN IF NOT EXISTS job_phrase_chunk_embeddings JSONB",
            "ADD COLUMN IF NOT EXISTS title_requirements_text TEXT",
            "ADD COLUMN IF NOT EXISTS title_requirements_embedding JSONB",
            "ADD COLUMN IF NOT EXISTS content_embedding JSONB",
            "ADD COLUMN IF NOT EXISTS embedding_model_name TEXT",
            "ADD COLUMN IF NOT EXISTS embedding_model_revision TEXT",
            "ADD COLUMN IF NOT EXISTS embedding_dim INTEGER",
            "ADD COLUMN IF NOT EXISTS embedded_at_utc TIMESTAMPTZ",
            "ADD COLUMN IF NOT EXISTS enrichment_version TEXT",
            "ADD COLUMN IF NOT EXISTS enrichment_ml_version TEXT",
            "ADD COLUMN IF NOT EXISTS requires_clearance BOOLEAN",
            "ADD COLUMN IF NOT EXISTS clearance_type TEXT",
            "ADD COLUMN IF NOT EXISTS clearance_evidence_text TEXT",
            "ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE",
            "ADD COLUMN IF NOT EXISTS missing_from_source_at_utc TIMESTAMPTZ",
            "ADD COLUMN IF NOT EXISTS content_fetch_failed_count INTEGER NOT NULL DEFAULT 0",
            "ADD COLUMN IF NOT EXISTS content_fetch_last_failed_at_utc TIMESTAMPTZ",
            "ADD COLUMN IF NOT EXISTS content_fetch_last_error_type TEXT",
            "ADD COLUMN IF NOT EXISTS last_seen_at_utc TIMESTAMPTZ",
            "ADD COLUMN IF NOT EXISTS first_seen_at_utc TIMESTAMPTZ NOT NULL DEFAULT now()",
            "ADD COLUMN IF NOT EXISTS stale_reason TEXT",
        ]
        for idx, column_sql in enumerate(jobs_column_alters, start=1):
            execute_schema_step(
                cur,
                f"alter jobs column {idx}/{len(jobs_column_alters)}: {column_sql}",
                f"ALTER TABLE jobs {column_sql}",
            )

        execute_schema_step(
            cur,
            "backfill jobs.source_company",
            """
            UPDATE jobs
            SET source_company = company_name
            WHERE source_company IS NULL
              AND company_name IS NOT NULL
            """,
        )
        execute_schema_step(
            cur,
            "backfill jobs.content_fetch_failed_count",
            """
            UPDATE jobs
            SET content_fetch_failed_count = 0
            WHERE content_fetch_failed_count IS NULL
            """,
        )
        execute_schema_step(
            cur,
            "create pipeline_migrations table",
            """
            CREATE TABLE IF NOT EXISTS pipeline_migrations (
                migration_name TEXT PRIMARY KEY,
                applied_at_utc TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """,
        )
        execute_schema_step(
            cur,
            "migrate legacy ML enrichment completion state",
            """
            WITH claimed_migration AS (
                INSERT INTO pipeline_migrations (migration_name)
                VALUES ('split_enrichment_versions_ml_v1')
                ON CONFLICT (migration_name) DO NOTHING
                RETURNING migration_name
            )
            UPDATE jobs
            SET enrichment_ml_version = enrichment_version
            WHERE EXISTS (SELECT 1 FROM claimed_migration)
              AND enrichment_ml_version IS NULL
              AND enrichment_version IS NOT NULL
              AND title_requirements_embedding IS NOT NULL
              AND job_selected_words IS NOT NULL
              AND job_selected_word_embeddings IS NOT NULL
              AND job_phrase_chunks IS NOT NULL
              AND job_phrase_chunk_embeddings IS NOT NULL
            """,
        )
        execute_schema_step(
            cur,
            "migrate legacy non-ML preparation completion state",
            """
            WITH claimed_migration AS (
                INSERT INTO pipeline_migrations (migration_name)
                VALUES ('split_enrichment_versions_non_ml_v1')
                ON CONFLICT (migration_name) DO NOTHING
                RETURNING migration_name
            )
            UPDATE jobs
            SET enrichment_version = %s
            WHERE EXISTS (SELECT 1 FROM claimed_migration)
              AND enrichment_version IS NULL
              AND job_word_tokens IS NOT NULL
              AND job_selected_words IS NOT NULL
              AND job_phrase_chunks IS NOT NULL
              AND title_requirements_text IS NOT NULL
            """,
            (get_enrichment_version(),),
        )

        jobs_indexes = [
            ("idx_jobs_source_type", "CREATE INDEX IF NOT EXISTS idx_jobs_source_type ON jobs (source_type)"),
            ("idx_jobs_source_company", "CREATE INDEX IF NOT EXISTS idx_jobs_source_company ON jobs (source_type, source_company)"),
            (
                "idx_jobs_source_company_active",
                "CREATE INDEX IF NOT EXISTS idx_jobs_source_company_active ON jobs (source_type, source_company, is_active)",
            ),
            ("idx_jobs_active", "CREATE INDEX IF NOT EXISTS idx_jobs_active ON jobs (is_active)"),
            ("idx_jobs_posted_at", "CREATE INDEX IF NOT EXISTS idx_jobs_posted_at ON jobs (posted_at)"),
            ("idx_jobs_posted_at_utc", "CREATE INDEX IF NOT EXISTS idx_jobs_posted_at_utc ON jobs (posted_at_utc)"),
            ("idx_jobs_fetched_at", "CREATE INDEX IF NOT EXISTS idx_jobs_fetched_at ON jobs (fetched_at_utc)"),
            ("idx_jobs_last_seen", "CREATE INDEX IF NOT EXISTS idx_jobs_last_seen ON jobs (last_seen_at_utc)"),
            (
                "idx_jobs_content_fetch_backoff",
                "CREATE INDEX IF NOT EXISTS idx_jobs_content_fetch_backoff ON jobs (content_fetch_failed_count, content_fetch_last_failed_at_utc)",
            ),
            ("idx_jobs_company_name", "CREATE INDEX IF NOT EXISTS idx_jobs_company_name ON jobs (company_name)"),
            ("idx_jobs_location_country", "CREATE INDEX IF NOT EXISTS idx_jobs_location_country ON jobs (location_country)"),
            ("idx_jobs_location_region", "CREATE INDEX IF NOT EXISTS idx_jobs_location_region ON jobs (location_country, location_region)"),
            ("idx_jobs_work_arrangement", "CREATE INDEX IF NOT EXISTS idx_jobs_work_arrangement ON jobs (work_arrangement)"),
            ("idx_jobs_location_parse_status", "CREATE INDEX IF NOT EXISTS idx_jobs_location_parse_status ON jobs (location_parse_status)"),
            ("idx_jobs_min_yoe", "CREATE INDEX IF NOT EXISTS idx_jobs_min_yoe ON jobs (min_years_experience)"),
            ("idx_jobs_enrichment_version", "CREATE INDEX IF NOT EXISTS idx_jobs_enrichment_version ON jobs (enrichment_version)"),
            (
                "idx_jobs_enrichment_ml_version",
                "CREATE INDEX IF NOT EXISTS idx_jobs_enrichment_ml_version ON jobs (enrichment_ml_version)",
            ),
            (
                "idx_jobs_requirements_extraction_version",
                "CREATE INDEX IF NOT EXISTS idx_jobs_requirements_extraction_version ON jobs (requirements_extraction_version)",
            ),
            ("idx_jobs_yoe_extraction_version", "CREATE INDEX IF NOT EXISTS idx_jobs_yoe_extraction_version ON jobs (yoe_extraction_version)"),
            (
                "idx_jobs_clearance_extraction_version",
                "CREATE INDEX IF NOT EXISTS idx_jobs_clearance_extraction_version ON jobs (clearance_extraction_version)",
            ),
        ]
        for idx, (index_name, index_sql) in enumerate(jobs_indexes, start=1):
            execute_schema_step(cur, f"create jobs index {idx}/{len(jobs_indexes)}: {index_name}", index_sql)
        execute_schema_step(
            cur,
            "create jobs index idx_jobs_missing_embeddings",
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_missing_embeddings
            ON jobs (id)
            WHERE title_requirements_embedding IS NULL
               OR job_selected_words IS NULL
               OR job_selected_word_embeddings IS NULL
               OR job_phrase_chunks IS NULL
               OR job_phrase_chunk_embeddings IS NULL
            """,
        )

        execute_schema_step(
            cur,
            "create ats_sources table",
            """
            CREATE TABLE IF NOT EXISTS ats_sources (
                source_url TEXT PRIMARY KEY,
                ats TEXT NOT NULL,
                company TEXT,
                last_get_at TIMESTAMPTZ,
                last_http_status_code INTEGER,
                last_error_type TEXT,
                last_error_at TIMESTAMPTZ,
                http_404_streak INTEGER NOT NULL DEFAULT 0
            )
            """,
        )
        ats_source_alters = [
            "ADD COLUMN IF NOT EXISTS last_http_status_code INTEGER",
            "ADD COLUMN IF NOT EXISTS last_error_type TEXT",
            "ADD COLUMN IF NOT EXISTS last_error_at TIMESTAMPTZ",
            "ADD COLUMN IF NOT EXISTS http_404_streak INTEGER NOT NULL DEFAULT 0",
        ]
        for idx, column_sql in enumerate(ats_source_alters, start=1):
            execute_schema_step(
                cur,
                f"alter ats_sources column {idx}/{len(ats_source_alters)}: {column_sql}",
                f"ALTER TABLE ats_sources {column_sql}",
            )
        execute_schema_step(
            cur,
            "backfill ats_sources.http_404_streak",
            """
            UPDATE ats_sources
            SET http_404_streak = 0
            WHERE http_404_streak IS NULL
            """,
        )
        execute_schema_step(
            cur,
            "create ats_sources index idx_ats_sources_ats_last_get",
            "CREATE INDEX IF NOT EXISTS idx_ats_sources_ats_last_get ON ats_sources (ats, last_get_at)",
        )
        execute_schema_step(
            cur,
            "create ats_sources index idx_ats_sources_ats_404_streak",
            "CREATE INDEX IF NOT EXISTS idx_ats_sources_ats_404_streak ON ats_sources (ats, last_http_status_code, http_404_streak, last_get_at)",
        )

        execute_schema_step(
            cur,
            "create resume_profiles table",
            """
            CREATE TABLE IF NOT EXISTS resume_profiles (
                firebase_uid TEXT PRIMARY KEY,
                profile_data JSONB NOT NULL,
                created_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """,
        )

    schema_step("start commit")
    conn.commit()
    schema_step("done commit")
