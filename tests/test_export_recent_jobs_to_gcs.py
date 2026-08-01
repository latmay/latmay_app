from __future__ import annotations

import json
import os
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

google_module = types.ModuleType("google")
google_cloud_module = types.ModuleType("google.cloud")
google_storage_module = types.ModuleType("google.cloud.storage")
google_cloud_module.storage = google_storage_module
google_module.cloud = google_cloud_module
sys.modules.setdefault("google", google_module)
sys.modules.setdefault("google.cloud", google_cloud_module)
sys.modules.setdefault("google.cloud.storage", google_storage_module)

from data_pipeline.export.export_recent_jobs_to_gcs import (
    metadata_rows_to_jsonl,
    recent_first_seen_count,
    source_type_counts,
)
from data_pipeline.export import export_recent_jobs_to_gcs as export_mod


class ExportRecentJobsToGcsTest(unittest.TestCase):
    def test_metadata_includes_source_type(self) -> None:
        rows = [
            {
                "id": 123,
                "source_type": "greenhouse",
                "source_job_id": "job-1",
                "source_company": "example",
                "company_name": "Example",
                "content_text": "Job text",
            }
        ]

        payload = json.loads(metadata_rows_to_jsonl(rows).splitlines()[0])

        self.assertEqual(payload["job_id"], "greenhouse:job-1")
        self.assertEqual(payload["source_type"], "greenhouse")
        self.assertEqual(payload["source_company"], "example")

    def test_export_quality_counts(self) -> None:
        now_utc = datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc)
        rows = [
            {
                "source_type": "greenhouse",
                "first_seen_at_utc": now_utc - timedelta(hours=2),
            },
            {
                "source_type": "lever",
                "first_seen_at_utc": (now_utc - timedelta(hours=25)).isoformat(),
            },
            {
                "source_type": "greenhouse",
                "first_seen_at_utc": None,
            },
            {
                "source_type": "",
                "first_seen_at_utc": "not-a-date",
            },
        ]

        self.assertEqual(source_type_counts(rows), {"greenhouse": 2, "lever": 1, "unknown": 1})
        self.assertEqual(recent_first_seen_count(rows, now_utc=now_utc), 1)

    def test_export_can_skip_legacy_fallback_artifacts(self) -> None:
        with (
            patch.dict(
                os.environ,
                {"EXPORT_LEGACY_FALLBACK_ARTIFACTS": "false", "GCS_BUCKET_NAME": "test-bucket"},
            ),
            patch.object(export_mod, "fetch_recent_jobs") as fetch_recent_jobs,
            patch.object(export_mod, "upload_csv_to_gcs") as upload_csv_to_gcs,
            patch.object(export_mod, "upload_text_to_gcs") as upload_text_to_gcs,
            patch.object(export_mod, "upload_bytes_to_gcs") as upload_bytes_to_gcs,
            patch.object(export_mod, "upload_sharded_artifacts_to_gcs", return_value=1000) as upload_shards,
        ):
            exported = export_mod.export_recent_jobs_to_gcs(object())

        self.assertEqual(exported, 1000)
        fetch_recent_jobs.assert_not_called()
        upload_csv_to_gcs.assert_not_called()
        upload_text_to_gcs.assert_not_called()
        upload_bytes_to_gcs.assert_not_called()
        upload_shards.assert_called_once()

    def test_fetch_requires_stored_export_eligibility(self) -> None:
        conn = MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []
        existing_columns = (
            set(export_mod.EXPORT_COLUMNS)
            | export_mod.REQUIRED_COLUMNS
            | export_mod.ARTIFACT_REQUIRED_COLUMNS
            | {"title_requirements_embedding"}
        )

        with patch.object(export_mod, "get_existing_job_columns", return_value=existing_columns):
            rows = export_mod.fetch_recent_jobs_page(conn, limit=100, offset=0)

        self.assertEqual(rows, [])
        sql = cursor.execute.call_args.args[0]
        self.assertIn("WHERE is_export_eligible = TRUE", sql)
        self.assertNotIn("btrim(content_text)", sql)
        self.assertEqual(
            cursor.execute.call_args.args[1],
            ("latmay-features-v1", "latmay-features-v1", 100, 0),
        )
        self.assertIn("location_parse_status", export_mod.ARTIFACT_REQUIRED_COLUMNS)
        self.assertIn("AND enrichment_version = %s", sql)
        self.assertIn("AND enrichment_ml_version = %s", sql)
        self.assertIn("enrichment_ml_version", export_mod.ARTIFACT_REQUIRED_COLUMNS)


if __name__ == "__main__":
    unittest.main()
