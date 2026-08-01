from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

from recent_posted_filter import format_posted_at_display, is_recent_posted_at, parse_posted_at_utc  # noqa: E402


class RecentPostedFilterTests(unittest.TestCase):
    def test_accepts_posted_at_within_48_hours(self) -> None:
        now_utc = datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc)

        self.assertTrue(
            is_recent_posted_at(
                "2026-06-02T12:00:00Z",
                now_utc=now_utc,
            )
        )

    def test_rejects_old_posted_at(self) -> None:
        now_utc = datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc)

        self.assertFalse(
            is_recent_posted_at(
                "2026-05-31T12:00:00Z",
                now_utc=now_utc,
            )
        )

    def test_rejects_missing_or_unparseable_posted_at(self) -> None:
        now_utc = datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc)

        self.assertFalse(is_recent_posted_at(None, now_utc=now_utc))
        self.assertFalse(is_recent_posted_at("", now_utc=now_utc))
        self.assertFalse(is_recent_posted_at("not a date", now_utc=now_utc))

    def test_parses_timezone_offset_to_utc(self) -> None:
        parsed = parse_posted_at_utc("2026-06-03T08:00:00-04:00")

        self.assertEqual(parsed, datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc))

    def test_formats_recent_posted_at_display(self) -> None:
        now_utc = datetime(2026, 6, 7, 18, 30, tzinfo=timezone.utc)

        self.assertEqual(
            format_posted_at_display("2026-05-29T18:30:00Z", now_utc=now_utc),
            "9 days ago",
        )
        self.assertEqual(
            format_posted_at_display("2026-06-07T12:30:00Z", now_utc=now_utc),
            "6 hours ago",
        )
        self.assertEqual(
            format_posted_at_display("2026-06-06T18:30:00Z", now_utc=now_utc),
            "yesterday",
        )


if __name__ == "__main__":
    unittest.main()
