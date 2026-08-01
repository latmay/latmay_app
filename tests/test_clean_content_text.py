from __future__ import annotations

import unittest
from unittest.mock import patch
from typing import Any

from data_pipeline.enrichment import clean_content_text


class FakeCursor:
    def __init__(self, conn: "FakeConn") -> None:
        self.conn = conn
        self.params: Any = None

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.params = params
        self.conn.executed.append((sql, params))

    def fetchone(self) -> dict[str, int]:
        return {"candidate_count": len(self.conn.rows)}

    def fetchall(self) -> list[dict[str, Any]]:
        last_id = int(self.params[-2])
        limit = int(self.params[-1])
        return [row for row in self.conn.rows if row["id"] > last_id][:limit]

    def executemany(self, sql: str, params: list[Any]) -> None:
        self.conn.updates.extend(params)


class FakeConn:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, Any]] = []
        self.updates: list[Any] = []
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1


class CleanContentTextTests(unittest.TestCase):
    def test_clean_text_repairs_entities_mojibake_unicode_and_spacing(self) -> None:
        raw = (
            "Requirements:&nbsp;&nbsp;Bachelor\u00e2\u20ac\u2122s degree in "
            "\ufb01nance\r\n  \uff30\uff59\uff54\uff48\uff4f\uff4e"
        )

        cleaned = clean_content_text.clean_text(raw)

        self.assertEqual(cleaned, "Requirements: Bachelor's degree in finance\nPython")

    def test_clean_text_is_idempotent_and_preserves_paragraphs(self) -> None:
        cleaned = clean_content_text.clean_text("One\n\n\n\nTwo")

        self.assertEqual(cleaned, "One\n\nTwo")
        self.assertEqual(clean_content_text.clean_text(cleaned), cleaned)

    def test_batches_are_committed_and_versioned(self) -> None:
        conn = FakeConn(
            [
                {"id": 1, "content_text": "A&nbsp;B"},
                {"id": 2, "content_text": "C  D"},
            ]
        )

        with patch.object(clean_content_text, "get_batch_size", return_value=1), \
             patch.object(clean_content_text, "get_max_batches", return_value=2):
            processed = clean_content_text.clean_pending_content_text(conn)

        self.assertEqual(processed, 2)
        self.assertEqual(conn.commits, 2)
        self.assertEqual(
            conn.updates,
            [
                ("A B", clean_content_text.CONTENT_TEXT_CLEAN_VERSION, 1),
                ("C D", clean_content_text.CONTENT_TEXT_CLEAN_VERSION, 2),
            ],
        )

    def test_zero_work_path_commits(self) -> None:
        conn = FakeConn([])

        self.assertEqual(clean_content_text.clean_pending_content_text(conn), 0)
        self.assertEqual(conn.commits, 1)


if __name__ == "__main__":
    unittest.main()
