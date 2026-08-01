from __future__ import annotations

import unittest
from unittest.mock import patch

from data_pipeline.ingestion import source_loader


class FakeCursor:
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.rows = rows or []
        self.executed_sql: str | None = None
        self.executed_params: object | None = None

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: object | None = None) -> None:
        self.executed_sql = sql
        self.executed_params = params

    def fetchall(self) -> list[dict[str, object]]:
        return self.rows

    def fetchone(self) -> dict[str, object] | None:
        return self.rows[0] if self.rows else None


class FakeConnection:
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.cursor_obj = FakeCursor(rows)
        self.commit_count = 0
        self.rollback_count = 0

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1


class IngestionSourceLoaderTests(unittest.TestCase):
    def test_loads_db_sources_without_404_skip_by_default(self) -> None:
        conn = FakeConnection([{"source_url": "https://example.com/source"}])

        with patch.dict("os.environ", {}, clear=True):
            urls = source_loader.load_ats_source_urls_from_db(conn, "workday")

        self.assertEqual(urls, ["https://example.com/source"])
        self.assertNotIn("http_404_streak", conn.cursor_obj.executed_sql or "")
        self.assertEqual(conn.cursor_obj.executed_params, ["workday"])

    def test_loads_db_sources_with_404_skip_when_enabled(self) -> None:
        conn = FakeConnection([{"source_url": "https://example.com/source"}])

        with patch.dict(
            "os.environ",
            {"SKIP_404_SOURCES": "true", "SKIP_404_STREAK_THRESHOLD": "3"},
            clear=True,
        ):
            urls = source_loader.load_ats_source_urls_from_db(conn, "workday")

        self.assertEqual(urls, ["https://example.com/source"])
        self.assertIn("last_http_status_code = 404", conn.cursor_obj.executed_sql or "")
        self.assertEqual(conn.cursor_obj.executed_params, ["workday", 3])

    def test_counts_skipped_404_sources_only_when_enabled(self) -> None:
        disabled_conn = FakeConnection([{"skipped_count": 7}])
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(source_loader.count_skipped_404_sources_from_db(disabled_conn, "workday"), 0)
        self.assertIsNone(disabled_conn.cursor_obj.executed_sql)

        enabled_conn = FakeConnection([{"skipped_count": 7}])
        with patch.dict(
            "os.environ",
            {"SKIP_404_SOURCES": "true", "SKIP_404_STREAK_THRESHOLD": "3"},
            clear=True,
        ):
            self.assertEqual(source_loader.count_skipped_404_sources_from_db(enabled_conn, "workday"), 7)

        self.assertIn("count(*) AS skipped_count", enabled_conn.cursor_obj.executed_sql or "")
        self.assertEqual(enabled_conn.cursor_obj.executed_params, ["workday", 3])

    def test_update_source_last_get_at_persists_error_metadata(self) -> None:
        conn = FakeConnection()

        source_loader.update_source_last_get_at(
            conn,
            "https://example.com/source",
            http_status_code=404,
            error_type="HTTPError",
        )

        self.assertIn("last_http_status_code", conn.cursor_obj.executed_sql or "")
        self.assertIn("http_404_streak", conn.cursor_obj.executed_sql or "")
        self.assertIn("%(error_type)s::text", conn.cursor_obj.executed_sql or "")
        self.assertIn("%(http_status_code)s::integer", conn.cursor_obj.executed_sql or "")
        self.assertEqual(
            conn.cursor_obj.executed_params,
            {
                "source_url": "https://example.com/source",
                "http_status_code": 404,
                "error_type": "HTTPError",
            },
        )
        self.assertEqual(conn.commit_count, 1)

    def test_rollback_failed_source_attempt_clears_transaction(self) -> None:
        conn = FakeConnection()

        source_loader.rollback_failed_source_attempt(conn)

        self.assertEqual(conn.rollback_count, 1)


if __name__ == "__main__":
    unittest.main()
