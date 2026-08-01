from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.enrichment import add_requirements, add_security_clearance, add_years_experience


class FakeCursor:
    def __init__(self, conn: "FakeConn") -> None:
        self.conn = conn
        self.is_count_query = False
        self.select_params: Any = None

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.conn.executed.append((sql, params))
        if sql.lstrip().upper().startswith("SELECT"):
            self.conn.last_select_params = params
            self.is_count_query = "COUNT(*) AS candidate_count" in sql
            self.select_params = params

    def fetchall(self) -> list[dict[str, Any]]:
        if self.select_params and len(self.select_params) >= 2:
            last_id = int(self.select_params[-2])
            limit = int(self.select_params[-1])
            return [row for row in self.conn.rows if int(row["id"]) > last_id][:limit]
        return self.conn.rows

    def fetchone(self) -> dict[str, int]:
        return {"candidate_count": len(self.conn.rows)}

    def executemany(self, sql: str, params: list[Any]) -> None:
        self.conn.executed.extend((sql, item) for item in params)
        updated_ids = {int(item[-1]) for item in params}
        self.conn.rows = [row for row in self.conn.rows if int(row["id"]) not in updated_ids]


class FakeConn:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, Any]] = []
        self.last_select_params: Any = None
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1

    @property
    def update_params(self) -> list[Any]:
        return [
            params
            for sql, params in self.executed
            if sql.lstrip().upper().startswith("UPDATE")
        ]


class ExtractionVersionTrackingTests(unittest.TestCase):
    def test_no_requirements_found_marks_extraction_complete(self) -> None:
        conn = FakeConn([{"id": 10, "content_text": "Welcome to our company benefits page."}])

        updated = add_requirements.update_requirements(conn)

        self.assertEqual(updated, 0)
        self.assertEqual(
            conn.last_select_params,
            (add_requirements.REQUIREMENTS_EXTRACTION_VERSION, 0, add_requirements.DEFAULT_BATCH_SIZE),
        )
        self.assertIsNone(conn.update_params[0][0])
        self.assertEqual(conn.update_params[0][1], add_requirements.REQUIREMENTS_EXTRACTION_VERSION)
        self.assertEqual(conn.update_params[0][3], 10)

    def test_no_yoe_found_marks_extraction_complete(self) -> None:
        conn = FakeConn([{"id": 20, "content_text": "Python and SQL required.", "extracted_requirements": ""}])

        updated = add_years_experience.update_years_experience(conn)

        self.assertEqual(updated, 0)
        self.assertEqual(
            conn.last_select_params,
            (
                add_years_experience.YOE_EXTRACTION_VERSION,
                20,
                add_years_experience.DEFAULT_YOE_BATCH_SIZE,
            ),
        )
        self.assertEqual(conn.update_params[0][5], add_years_experience.YOE_EXTRACTION_VERSION)
        self.assertEqual(conn.update_params[0][7], 20)

    def test_no_clearance_found_marks_extraction_complete(self) -> None:
        conn = FakeConn([{"id": 30, "content_text": "Python and SQL required.", "extracted_requirements": ""}])

        updated = add_security_clearance.update_security_clearance(conn)

        self.assertEqual(updated, 1)
        self.assertEqual(
            conn.last_select_params,
            (
                add_security_clearance.CLEARANCE_EXTRACTION_VERSION,
                30,
                add_security_clearance.DEFAULT_CLEARANCE_BATCH_SIZE,
            ),
        )
        self.assertEqual(conn.update_params[0][0], False)
        self.assertEqual(conn.update_params[0][3], add_security_clearance.CLEARANCE_EXTRACTION_VERSION)
        self.assertEqual(conn.update_params[0][5], 30)

    def test_zero_candidate_paths_commit_without_batch_select(self) -> None:
        requirements_conn = FakeConn([])
        add_requirements.update_requirements(requirements_conn)
        yoe_conn = FakeConn([])
        add_years_experience.update_years_experience(yoe_conn)
        clearance_conn = FakeConn([])
        add_security_clearance.update_security_clearance(clearance_conn)

        self.assertEqual(
            requirements_conn.last_select_params,
            (add_requirements.REQUIREMENTS_EXTRACTION_VERSION,),
        )
        self.assertEqual(
            yoe_conn.last_select_params,
            (add_years_experience.YOE_EXTRACTION_VERSION,),
        )
        self.assertEqual(
            clearance_conn.last_select_params,
            (add_security_clearance.CLEARANCE_EXTRACTION_VERSION,),
        )
        self.assertEqual(requirements_conn.commits, 1)
        self.assertEqual(yoe_conn.commits, 1)
        self.assertEqual(clearance_conn.commits, 1)
        self.assertEqual(len(requirements_conn.executed), 1)
        self.assertEqual(len(yoe_conn.executed), 1)
        self.assertEqual(len(clearance_conn.executed), 1)

    def test_yoe_version_filter_applies_to_all_text_sources(self) -> None:
        conn = FakeConn([])

        add_years_experience.update_years_experience(conn)

        select_sql = conn.executed[0][0]
        self.assertIn(
            "WHERE (content_text IS NOT NULL OR extracted_requirements IS NOT NULL)",
            select_sql,
        )
        self.assertIn("AND yoe_extraction_version IS DISTINCT FROM %s", select_sql)

    def test_clearance_version_filter_applies_to_all_text_sources(self) -> None:
        conn = FakeConn([])

        add_security_clearance.update_security_clearance(conn)

        select_sql = conn.executed[0][0]
        self.assertIn(
            "WHERE (content_text IS NOT NULL OR extracted_requirements IS NOT NULL)",
            select_sql,
        )
        self.assertIn("AND clearance_extraction_version IS DISTINCT FROM %s", select_sql)


if __name__ == "__main__":
    unittest.main()
