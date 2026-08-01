from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from data_pipeline.ingestion import budget


class IngestionBudgetTests(unittest.TestCase):
    def test_blank_budget_disables_deadline(self) -> None:
        with patch.dict(os.environ, {"INGESTION_TIME_BUDGET_SECONDS": ""}, clear=True):
            self.assertIsNone(budget.ingestion_time_budget_seconds())
            self.assertFalse(budget.should_stop_for_ingestion_budget("ashby", started_at=0.0, completed_sources=3))

    def test_invalid_or_nonpositive_budget_disables_deadline(self) -> None:
        for raw_value in ["not-a-number", "0", "-1"]:
            with patch.dict(os.environ, {"INGESTION_TIME_BUDGET_SECONDS": raw_value}, clear=True):
                self.assertIsNone(budget.ingestion_time_budget_seconds())

    def test_stops_when_budget_elapsed(self) -> None:
        with (
            patch.dict(os.environ, {"INGESTION_TIME_BUDGET_SECONDS": "10"}, clear=True),
            patch("data_pipeline.ingestion.budget.time.monotonic", return_value=15.0),
        ):
            self.assertTrue(budget.should_stop_for_ingestion_budget("lever", started_at=4.0, completed_sources=2))

    def test_continues_when_budget_remains(self) -> None:
        with (
            patch.dict(os.environ, {"INGESTION_TIME_BUDGET_SECONDS": "10"}, clear=True),
            patch("data_pipeline.ingestion.budget.time.monotonic", return_value=13.9),
        ):
            self.assertFalse(budget.should_stop_for_ingestion_budget("lever", started_at=4.0, completed_sources=2))


if __name__ == "__main__":
    unittest.main()
