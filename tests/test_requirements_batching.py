from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.enrichment import add_requirements


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
        if "SELECT id, content_text" in sql:
            last_id = int(params[-2])
            limit = int(params[-1])
            self.result = [
                row for row in self.conn.pending if int(row["id"]) > last_id
            ][:limit]

    def fetchone(self) -> dict[str, Any]:
        return self.result[0]

    def fetchall(self) -> list[dict[str, Any]]:
        return self.result

    def executemany(self, sql: str, params: list[tuple[Any, ...]]) -> None:
        self.conn.updates.extend(params)
        updated_ids = {int(item[3]) for item in params}
        self.conn.pending = [
            row for row in self.conn.pending if int(row["id"]) not in updated_ids
        ]


class FakeConn:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.pending = rows
        self.updates: list[tuple[Any, ...]] = []
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1


class RequirementsBatchingTests(unittest.TestCase):
    def test_batch_size_prefers_non_ml_setting(self) -> None:
        with patch.dict(
            os.environ,
            {"NON_ML_BATCH_SIZE": "25", "ENRICHMENT_BATCH_SIZE": "500"},
            clear=False,
        ):
            self.assertEqual(add_requirements.get_batch_size(), 25)

    def test_batch_size_falls_back_to_enrichment_setting(self) -> None:
        with patch.dict(os.environ, {"ENRICHMENT_BATCH_SIZE": "75"}, clear=False):
            os.environ.pop("NON_ML_BATCH_SIZE", None)
            self.assertEqual(add_requirements.get_batch_size(), 75)

    def test_processing_is_capped_and_committed_per_batch(self) -> None:
        rows = [
            {"id": job_id, "content_text": "Requirements\nPython experience required"}
            for job_id in range(1, 8)
        ]
        conn = FakeConn(rows)

        with patch.dict(
            os.environ,
            {"NON_ML_BATCH_SIZE": "2", "REQUIREMENTS_MAX_BATCHES": "2"},
            clear=False,
        ):
            updated = add_requirements.update_requirements(conn)

        self.assertEqual(updated, 4)
        self.assertEqual(len(conn.updates), 4)
        self.assertEqual(conn.commits, 2)
        self.assertEqual([row["id"] for row in conn.pending], [5, 6, 7])

    def test_empty_extraction_is_marked_complete_in_batch_update(self) -> None:
        conn = FakeConn([{"id": 1, "content_text": "Welcome to our company."}])

        with patch.dict(
            os.environ,
            {"NON_ML_BATCH_SIZE": "10", "REQUIREMENTS_MAX_BATCHES": "1"},
            clear=False,
        ):
            updated = add_requirements.update_requirements(conn)

        self.assertEqual(updated, 0)
        self.assertEqual(conn.commits, 1)
        self.assertIsNone(conn.updates[0][0])
        self.assertEqual(conn.updates[0][1], add_requirements.REQUIREMENTS_EXTRACTION_VERSION)

    def test_zero_candidates_commit_without_batch_work(self) -> None:
        conn = FakeConn([])

        updated = add_requirements.update_requirements(conn)

        self.assertEqual(updated, 0)
        self.assertEqual(conn.commits, 1)
        self.assertEqual(conn.updates, [])

    def test_requirements_batch_cap_is_independent_from_enrichment_cap(self) -> None:
        with patch.dict(
            os.environ,
            {"REQUIREMENTS_MAX_BATCHES": "3", "ENRICHMENT_MAX_BATCHES": "99"},
            clear=False,
        ):
            self.assertEqual(add_requirements.get_max_batches(), 3)


if __name__ == "__main__":
    unittest.main()
