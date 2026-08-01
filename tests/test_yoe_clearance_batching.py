from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.enrichment import add_security_clearance, add_years_experience


class FakeCursor:
    def __init__(self, conn: "FakeConn") -> None:
        self.conn = conn
        self.result: list[dict[str, Any]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        if "COUNT(*) AS candidate_count" in sql:
            self.result = [{"candidate_count": len(self.conn.pending)}]
            return
        last_id = int(params[-2])
        limit = int(params[-1])
        self.result = [row for row in self.conn.pending if int(row["id"]) > last_id][:limit]

    def fetchone(self) -> dict[str, Any]:
        return self.result[0]

    def fetchall(self) -> list[dict[str, Any]]:
        return self.result

    def executemany(self, sql: str, params: list[tuple[Any, ...]]) -> None:
        self.conn.updates.extend(params)
        updated_ids = {int(item[-1]) for item in params}
        self.conn.pending = [row for row in self.conn.pending if int(row["id"]) not in updated_ids]


class FakeConn:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.pending = rows
        self.updates: list[tuple[Any, ...]] = []
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1


class YoeClearanceBatchingTests(unittest.TestCase):
    def test_yoe_drains_all_batches(self) -> None:
        rows = [
            {"id": job_id, "content_text": "Requires 3+ years of experience.", "extracted_requirements": ""}
            for job_id in range(1, 6)
        ]
        conn = FakeConn(rows)

        with patch.dict(os.environ, {"YOE_BATCH_SIZE": "2"}, clear=False):
            updated = add_years_experience.update_years_experience(conn)

        self.assertEqual(updated, 5)
        self.assertEqual(conn.commits, 3)
        self.assertEqual(len(conn.updates), 5)
        self.assertEqual(conn.pending, [])

    def test_clearance_drains_all_batches(self) -> None:
        rows = [
            {
                "id": job_id,
                "content_text": "An active Secret security clearance is required.",
                "extracted_requirements": "",
            }
            for job_id in range(1, 6)
        ]
        conn = FakeConn(rows)

        with patch.dict(os.environ, {"CLEARANCE_BATCH_SIZE": "2"}, clear=False):
            updated = add_security_clearance.update_security_clearance(conn)

        self.assertEqual(updated, 5)
        self.assertEqual(conn.commits, 3)
        self.assertEqual(len(conn.updates), 5)
        self.assertEqual(conn.pending, [])

    def test_batch_sizes_are_independent(self) -> None:
        with patch.dict(
            os.environ,
            {"YOE_BATCH_SIZE": "123", "CLEARANCE_BATCH_SIZE": "456"},
            clear=False,
        ):
            self.assertEqual(add_years_experience.get_batch_size(), 123)
            self.assertEqual(add_security_clearance.get_batch_size(), 456)


if __name__ == "__main__":
    unittest.main()
