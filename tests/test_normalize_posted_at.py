from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.enrichment.normalize_posted_at import (  # noqa: E402
    fetch_rows_to_normalize,
    normalize_posted_at_value,
    reference_time_for_row,
    update_posted_at_utc,
)


class RecordingCursor:
    def __init__(self) -> None:
        self.query = ""
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def execute(self, query, params) -> None:
        self.query = query
        self.params = params

    def fetchall(self) -> list[dict]:
        return []


class RecordingConnection:
    def __init__(self) -> None:
        self.recording_cursor = RecordingCursor()
        self.commits = 0

    def cursor(self) -> RecordingCursor:
        return self.recording_cursor

    def commit(self) -> None:
        self.commits += 1


class NormalizePostedAtTests(unittest.TestCase):
    def test_fetch_only_selects_rows_without_normalized_timestamp(self) -> None:
        conn = RecordingConnection()

        self.assertEqual(fetch_rows_to_normalize(conn, limit=123), [])

        query = " ".join(conn.recording_cursor.query.split())
        self.assertIn("AND posted_at_utc IS NULL", query)
        self.assertNotIn("posted_at !~", query)
        self.assertEqual(conn.recording_cursor.params, (123,))

    def test_no_updates_still_commit_read_transaction(self) -> None:
        conn = RecordingConnection()

        updated, unparseable = update_posted_at_utc(conn, [])

        self.assertEqual((updated, unparseable), (0, 0))
        self.assertEqual(conn.commits, 1)

    def test_parses_iso_timestamp_to_utc(self) -> None:
        parsed = normalize_posted_at_value("2026-06-03T08:00:00-04:00")

        self.assertEqual(parsed, datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc))

    def test_parses_day_abbreviated_month_year(self) -> None:
        parsed = normalize_posted_at_value("08-Jun-2026")

        self.assertEqual(parsed, datetime(2026, 6, 8, 0, 0, tzinfo=timezone.utc))

    def test_parses_full_month_day_year(self) -> None:
        parsed = normalize_posted_at_value("February 20, 2026")

        self.assertEqual(parsed, datetime(2026, 2, 20, 0, 0, tzinfo=timezone.utc))

    def test_parses_full_month_day_year_with_extra_spacing(self) -> None:
        parsed = normalize_posted_at_value("June  8, 2026")

        self.assertEqual(parsed, datetime(2026, 6, 8, 0, 0, tzinfo=timezone.utc))

    def test_parses_relative_days_using_reference_time(self) -> None:
        reference_time = datetime(2026, 6, 7, 18, 30, tzinfo=timezone.utc)

        parsed = normalize_posted_at_value("Posted 9 Days Ago", reference_time=reference_time)

        self.assertEqual(parsed, datetime(2026, 5, 29, 18, 30, tzinfo=timezone.utc))

    def test_parses_relative_days_with_plus_using_reference_time(self) -> None:
        reference_time = datetime(2026, 6, 7, 18, 30, tzinfo=timezone.utc)

        parsed = normalize_posted_at_value("Posted 30+ Days Ago", reference_time=reference_time)

        self.assertEqual(parsed, datetime(2026, 5, 8, 18, 30, tzinfo=timezone.utc))

    def test_parses_relative_today_using_reference_time(self) -> None:
        reference_time = datetime(2026, 6, 7, 18, 30, tzinfo=timezone.utc)

        parsed = normalize_posted_at_value("Posted Today", reference_time=reference_time)

        self.assertEqual(parsed, reference_time)

    def test_uses_fetched_at_as_reference_before_seen_times(self) -> None:
        row = {
            "fetched_at_utc": "2026-06-07T18:30:00Z",
            "last_seen_at_utc": "2026-06-08T18:30:00Z",
            "first_seen_at_utc": "2026-06-09T18:30:00Z",
            "created_at_utc": "2026-06-10T18:30:00Z",
        }

        self.assertEqual(reference_time_for_row(row), datetime(2026, 6, 7, 18, 30, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
