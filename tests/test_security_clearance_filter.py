from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

from hard_filters.security_clearance_filter import filter_jobs_df_excluding_security_clearance


class SecurityClearanceFilterTests(unittest.TestCase):
    def test_filter_is_noop_when_disabled(self) -> None:
        df = pd.DataFrame(
            [
                {"title": "A", "requires_clearance": True},
                {"title": "B", "requires_clearance": False},
            ]
        )

        filtered = filter_jobs_df_excluding_security_clearance(
            df,
            exclude_security_clearance=False,
        )

        self.assertEqual(filtered["title"].tolist(), ["A", "B"])

    def test_filters_by_requires_clearance_boolean(self) -> None:
        df = pd.DataFrame(
            [
                {"title": "A", "requires_clearance": True},
                {"title": "B", "requires_clearance": False},
            ]
        )

        filtered = filter_jobs_df_excluding_security_clearance(
            df,
            exclude_security_clearance=True,
        )

        self.assertEqual(filtered["title"].tolist(), ["B"])

    def test_filters_by_clearance_text_fields(self) -> None:
        df = pd.DataFrame(
            [
                {"title": "A", "clearance_type": "Top Secret"},
                {"title": "B", "clearance_type": ""},
                {"title": "C", "clearance_evidence_text": "Public Trust required."},
            ]
        )

        filtered = filter_jobs_df_excluding_security_clearance(
            df,
            exclude_security_clearance=True,
        )

        self.assertEqual(filtered["title"].tolist(), ["B"])

    def test_missing_clearance_columns_is_noop(self) -> None:
        df = pd.DataFrame([{"title": "A"}, {"title": "B"}])

        filtered = filter_jobs_df_excluding_security_clearance(
            df,
            exclude_security_clearance=True,
        )

        self.assertEqual(filtered["title"].tolist(), ["A", "B"])


if __name__ == "__main__":
    unittest.main()
