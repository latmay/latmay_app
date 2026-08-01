from __future__ import annotations

import os
import sys
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

from ranking_timing import (
    print_ranking_timing_summary,
    record_ranking_timing,
    reset_ranking_timing_collection,
    start_ranking_timing_collection,
)


class RankingTimingSummaryTest(unittest.TestCase):
    def capture_summary(self, env_value: str | None) -> str:
        token = start_ranking_timing_collection()
        try:
            record_ranking_timing("example", time.perf_counter(), jobs=2)
            environment = {} if env_value is None else {"ENABLE_RANKING_TIMING_SUMMARY": env_value}
            with patch.dict(os.environ, environment, clear=False):
                if env_value is None:
                    os.environ.pop("ENABLE_RANKING_TIMING_SUMMARY", None)
                output = StringIO()
                with redirect_stdout(output):
                    print_ranking_timing_summary()
                return output.getvalue()
        finally:
            reset_ranking_timing_collection(token)

    def test_summary_is_enabled_by_default(self) -> None:
        self.assertIn("Ranking timing summary start", self.capture_summary(None))

    def test_summary_can_be_disabled(self) -> None:
        self.assertEqual(self.capture_summary("false"), "")


if __name__ == "__main__":
    unittest.main()
