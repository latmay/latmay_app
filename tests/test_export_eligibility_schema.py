from __future__ import annotations

import unittest

from data_pipeline.common.export_eligibility import require_export_eligibility_schema


class FakeCursor:
    def __init__(self, row: dict[str, bool]) -> None:
        self.row = row
        self.sql = ""
        self.params: dict[str, str] = {}

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, sql: str, params: dict[str, str]) -> None:
        self.sql = sql
        self.params = params

    def fetchone(self) -> dict[str, bool]:
        return self.row


class FakeConnection:
    def __init__(self, row: dict[str, bool]) -> None:
        self.cursor_obj = FakeCursor(row)
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commits += 1


class ExportEligibilitySchemaTests(unittest.TestCase):
    def test_ready_schema_commits_readiness_check(self) -> None:
        conn = FakeConnection(
            {"column_ready": True, "trigger_ready": True, "index_ready": True}
        )

        require_export_eligibility_schema(conn)  # type: ignore[arg-type]

        self.assertEqual(conn.commits, 1)
        self.assertEqual(
            conn.cursor_obj.params["index_name"],
            "public.idx_jobs_export_eligible_flag_recent",
        )

    def test_missing_schema_fails_with_migration_instruction(self) -> None:
        conn = FakeConnection(
            {"column_ready": False, "trigger_ready": False, "index_ready": False}
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "002_add_batched_export_eligibility.sql",
        ):
            require_export_eligibility_schema(conn)  # type: ignore[arg-type]

        self.assertEqual(conn.commits, 1)


if __name__ == "__main__":
    unittest.main()
