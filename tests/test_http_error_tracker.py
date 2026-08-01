from __future__ import annotations

import unittest
from unittest.mock import patch

from data_pipeline.common.data_quality import http_status_code_from_exception
from data_pipeline.ingestion.http_error_tracker import AtsHttp429LimitReached, AtsHttpErrorTracker


class HttpErrorTrackerTests(unittest.TestCase):
    def test_records_429_until_limit(self) -> None:
        with patch.dict("os.environ", {"INGESTION_MAX_429_ERRORS_PER_ATS": "2"}, clear=True):
            tracker = AtsHttpErrorTracker("workable")

        self.assertFalse(tracker.record(429))
        self.assertTrue(tracker.record(429))

    def test_zero_disables_429_stop(self) -> None:
        with patch.dict("os.environ", {"INGESTION_MAX_429_ERRORS_PER_ATS": "0"}, clear=True):
            tracker = AtsHttpErrorTracker("workday")

        self.assertFalse(tracker.record(429))
        self.assertFalse(tracker.record(429))

    def test_limit_exception_reports_http_429(self) -> None:
        exc = AtsHttp429LimitReached("ashby", 4, 4)

        self.assertEqual(http_status_code_from_exception(exc), 429)


if __name__ == "__main__":
    unittest.main()
