from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_DIR = REPO_ROOT / "data_pipeline" / "migrations" / "content_text_eligibility"

GENERATED_EXPRESSION_PARTS = (
    "is_active = TRUE",
    "NULLIF(btrim(content_text), '') IS NOT NULL",
    "lower(btrim(content_text)) NOT IN ('nan', 'none', 'null')",
    "posted_at_utc IS NOT NULL",
    "location_parse_status IN (",
)


class ContentTextEligibilityIndexTests(unittest.TestCase):
    def test_initial_migration_preserves_expression_index_history(self) -> None:
        migration_sql = (
            MIGRATION_DIR / "001_create_export_eligible_jobs_partial_index.sql"
        ).read_text(encoding="utf-8")

        self.assertIn("CREATE INDEX CONCURRENTLY", migration_sql)
        self.assertIn("idx_jobs_export_eligible_recent", migration_sql)
        self.assertIn("ON public.jobs (posted_at_utc DESC, id DESC)", migration_sql)
        for predicate_part in GENERATED_EXPRESSION_PARTS:
            self.assertIn(predicate_part, migration_sql)
        self.assertIn("content_text_eligibility_partial_index_v1", migration_sql)

    def test_batched_column_migration_replaces_expression_index(self) -> None:
        migration_sql = (
            MIGRATION_DIR / "002_add_batched_export_eligibility.sql"
        ).read_text(encoding="utf-8")

        self.assertIn("ADD COLUMN IF NOT EXISTS is_export_eligible BOOLEAN", migration_sql)
        self.assertIn("CREATE TRIGGER trg_jobs_set_export_eligibility", migration_sql)
        self.assertIn("job_is_active = TRUE", migration_sql)
        self.assertIn("NULLIF(btrim(job_content_text), '') IS NOT NULL", migration_sql)
        self.assertIn(
            "lower(btrim(job_content_text)) NOT IN ('nan', 'none', 'null')",
            migration_sql,
        )
        self.assertIn("job_posted_at_utc IS NOT NULL", migration_sql)
        self.assertIn("job_location_parse_status IN (", migration_sql)
        self.assertIn("CALL public.backfill_job_export_eligibility", migration_sql)
        self.assertIn("COMMIT;", migration_sql)
        self.assertIn("RAISE NOTICE", migration_sql)
        self.assertIn("CREATE INDEX CONCURRENTLY", migration_sql)
        self.assertIn("idx_jobs_export_eligible_flag_recent", migration_sql)
        self.assertIn("WHERE is_export_eligible = TRUE", migration_sql)
        self.assertIn("DROP INDEX CONCURRENTLY", migration_sql)
        self.assertIn("batched_export_eligibility_v2", migration_sql)

    def test_eligibility_queries_use_generated_boolean(self) -> None:
        source_paths = (
            REPO_ROOT / "data_pipeline" / "enrichment" / "add_job_features.py",
            REPO_ROOT / "data_pipeline" / "export" / "export_recent_jobs_to_gcs.py",
        )

        for source_path in source_paths:
            source = source_path.read_text(encoding="utf-8")
            with self.subTest(source_path=source_path.name):
                self.assertIn("is_export_eligible = TRUE", source)
                self.assertNotIn("NULLIF(btrim(content_text), '')", source)
                self.assertIn("posted_at_utc DESC", source)
                self.assertIn("id DESC", source)

    def test_runtime_schema_does_not_build_the_concurrent_index(self) -> None:
        runtime_schema = (
            REPO_ROOT / "data_pipeline" / "common" / "schema.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("idx_jobs_export_eligible_flag_recent", runtime_schema)


if __name__ == "__main__":
    unittest.main()
