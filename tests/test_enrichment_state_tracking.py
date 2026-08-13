from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

psycopg_module = types.ModuleType("psycopg")
psycopg_rows_module = types.ModuleType("psycopg.rows")
psycopg_types_module = types.ModuleType("psycopg.types")
psycopg_json_module = types.ModuleType("psycopg.types.json")
psycopg_module.Connection = object
psycopg_module.connect = object
psycopg_module.Error = Exception
psycopg_module.errors = types.SimpleNamespace(
    QueryCanceled=Exception,
    LockNotAvailable=Exception,
    DeadlockDetected=Exception,
)
psycopg_rows_module.dict_row = object()
psycopg_json_module.Jsonb = lambda value: value
psycopg_types_module.json = psycopg_json_module
sys.modules.setdefault("psycopg", psycopg_module)
sys.modules.setdefault("psycopg.rows", psycopg_rows_module)
sys.modules.setdefault("psycopg.types", psycopg_types_module)
sys.modules.setdefault("psycopg.types.json", psycopg_json_module)

from data_pipeline.common import db, schema
from data_pipeline.enrichment import add_job_features
from data_pipeline.ingestion import fill_missing_content_text


class FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.executemany_calls: list[tuple[str, list[Any]]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    def executemany(self, sql: str, params: list[Any]) -> None:
        self.executemany_calls.append((sql, params))


class FakeConn:
    def __init__(self) -> None:
        self.cursor_obj = FakeCursor()
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commits += 1


class FakeModel:
    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _text in texts], dtype=np.float32)


class EnrichmentStateTrackingTests(unittest.TestCase):
    def test_job_normalization_removes_nul_bytes_recursively(self) -> None:
        normalized = db.normalize_job_record(
            {
                "source_type": "green\x00house",
                "source_job_id": "job\x001",
                "title": "Software\x00 Engineer",
                "content_text": "Python\x00 and SQL",
                "raw_json": {
                    "nested\x00key": ["first\x00value", {"description": "deep\x00value"}],
                },
            }
        )

        self.assertEqual(normalized["source_type"], "greenhouse")
        self.assertEqual(normalized["source_job_id"], "job1")
        self.assertEqual(normalized["title"], "Software Engineer")
        self.assertEqual(normalized["content_text"], "Python and SQL")
        self.assertEqual(
            normalized["raw_json"],
            {"nestedkey": ["firstvalue", {"description": "deepvalue"}]},
        )

    def test_content_fill_removes_nul_bytes_before_update(self) -> None:
        conn = FakeConn()

        fill_missing_content_text.record_content_fetch_success(
            conn,
            1,
            "<p>Null\x00 HTML</p>",
            "Null\x00 text",
        )

        params = conn.cursor_obj.executed[0][1]
        self.assertEqual(params, ("<p>Null HTML</p>", "Null text", False, False, 1))

    def test_schema_uses_one_time_state_split_migrations(self) -> None:
        conn = FakeConn()

        with patch.object(schema, "schema_step"):
            schema.initialize_schema(conn)  # type: ignore[arg-type]

        all_sql = "\n".join(sql for sql, _params in conn.cursor_obj.executed)
        self.assertIn("CREATE TABLE IF NOT EXISTS pipeline_migrations", all_sql)
        self.assertIn("split_enrichment_versions_ml_v1", all_sql)
        self.assertIn("split_enrichment_versions_non_ml_v1", all_sql)
        self.assertIn("ON CONFLICT (migration_name) DO NOTHING", all_sql)

    def test_ingestion_upsert_invalidates_both_versions_on_title_or_content_change(self) -> None:
        conn = FakeConn()
        record = {
            "source_type": "example",
            "source_job_id": "1",
            "title": "Title",
            "content_text": "Content",
        }

        db.upsert_jobs(conn, [record])  # type: ignore[arg-type]

        sql = conn.cursor_obj.executemany_calls[0][0]
        self.assertIn("jobs.title IS DISTINCT FROM EXCLUDED.title", sql)
        self.assertIn("jobs.content_text IS DISTINCT FROM EXCLUDED.content_text", sql)
        self.assertIn("enrichment_version = CASE", sql)
        self.assertIn("enrichment_ml_version = CASE", sql)
        self.assertIn("requirements_extraction_version = CASE", sql)
        self.assertIn("yoe_extraction_version = CASE", sql)
        self.assertIn("clearance_extraction_version = CASE", sql)

    def test_content_fill_invalidates_both_versions(self) -> None:
        conn = FakeConn()

        fill_missing_content_text.record_content_fetch_success(conn, 1, "<p>Content</p>", "Content")

        sql = conn.cursor_obj.executed[0][0]
        self.assertIn("enrichment_version = NULL", sql)
        self.assertIn("enrichment_ml_version = NULL", sql)

    def test_preparation_marks_non_ml_complete_and_ml_incomplete(self) -> None:
        conn = FakeConn()
        payload = {
            "token_lists": [["python"]],
            "selected_words": [["python"]],
            "chunks": [["Python experience required"]],
            "ranking_texts": ["Job title: Engineer"],
        }

        with (
            patch.object(add_job_features, "build_prepared_feature_payload", return_value=payload),
            patch.object(add_job_features, "get_enrichment_version", return_value="features-v2"),
        ):
            updated = add_job_features.update_prepared_feature_rows(conn, [{"id": 1}])

        self.assertEqual(updated, 1)
        sql, params = conn.cursor_obj.executemany_calls[0]
        self.assertIn("enrichment_version = %(enrichment_version)s", sql)
        self.assertIn("enrichment_ml_version = NULL", sql)
        self.assertEqual(params[0]["enrichment_version"], "features-v2")

    def test_embedding_marks_only_ml_stage_complete(self) -> None:
        conn = FakeConn()
        rows = [
            {
                "id": 1,
                "content_text": "Content",
                "job_selected_words": ["python"],
                "job_phrase_chunks": ["Python experience required"],
                "title_requirements_text": "Job title: Engineer",
            }
        ]

        with (
            patch.object(add_job_features, "get_enrichment_version", return_value="features-v2"),
            patch.object(add_job_features, "get_minilm_model_name", return_value="model"),
            patch.object(add_job_features, "get_minilm_model_revision", return_value="revision"),
            patch.object(
                add_job_features,
                "embed_unique_texts",
                side_effect=[
                    [[[1.0, 0.0]]],
                    [[[0.0, 1.0]]],
                ],
            ),
            patch.object(add_job_features, "log_data_quality"),
        ):
            updated = add_job_features.update_embedding_rows(conn, rows, FakeModel())

        self.assertEqual(updated, 1)
        sql, params = conn.cursor_obj.executemany_calls[0]
        self.assertIn("enrichment_ml_version = %(enrichment_ml_version)s", sql)
        self.assertNotIn("enrichment_version = %(enrichment_version)s", sql)
        self.assertEqual(params[0]["enrichment_ml_version"], "features-v2")


if __name__ == "__main__":
    unittest.main()
