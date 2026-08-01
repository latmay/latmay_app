from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.common.data_quality import (
    count_blank,
    count_distribution,
    duplicate_count,
    length_distribution,
    numeric_distribution,
)


class DataQualityLoggingTest(unittest.TestCase):
    def test_count_helpers(self) -> None:
        records = [
            {"title": "Engineer", "source_job_id": "a"},
            {"title": "", "source_job_id": "a"},
            {"title": None, "source_job_id": "b"},
        ]

        self.assertEqual(count_blank(records, "title"), 2)
        self.assertEqual(duplicate_count(record["source_job_id"] for record in records), 1)

    def test_distributions_are_safe_for_empty_and_numeric_values(self) -> None:
        self.assertEqual(
            numeric_distribution([]),
            {"min": None, "p50": None, "p90": None, "max": None},
        )
        self.assertEqual(numeric_distribution([1, 3, "5"])["p50"], 3)
        self.assertEqual(count_distribution([1, 2, 3], prefix="items_")["items_avg"], 2)
        self.assertEqual(length_distribution(["abc", "abcdef"])["p50_chars"], 4.5)


if __name__ == "__main__":
    unittest.main()
