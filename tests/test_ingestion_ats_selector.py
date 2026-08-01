from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from data_pipeline.ingestion import main_ingest


class IngestionAtsSelectorTests(unittest.TestCase):
    def run_with_steps(self, ingestion_ats: str | None, extra_env: dict[str, str] | None = None) -> list[str]:
        calls: list[str] = []
        steps = {
            "alpha": ("Alpha", "ENABLE_ALPHA_INGESTION", lambda conn: calls.append("alpha")),
            "beta": ("Beta", "ENABLE_BETA_INGESTION", lambda conn: calls.append("beta")),
        }
        env = dict(extra_env or {})
        if ingestion_ats is not None:
            env["INGESTION_ATS"] = ingestion_ats
        with patch.dict(os.environ, env, clear=True), patch.object(main_ingest, "ATS_STEPS", steps):
            main_ingest.run_ats_ingestion_steps(conn=object())
        return calls

    def test_blank_ingestion_ats_runs_enabled_scrapers(self) -> None:
        calls = self.run_with_steps("", {"ENABLE_BETA_INGESTION": "false"})

        self.assertEqual(calls, ["alpha"])

    def test_specific_ingestion_ats_runs_only_that_scraper(self) -> None:
        calls = self.run_with_steps("beta")

        self.assertEqual(calls, ["beta"])

    def test_none_ingestion_ats_skips_all_scrapers(self) -> None:
        calls = self.run_with_steps("none")

        self.assertEqual(calls, [])

    def test_unknown_ingestion_ats_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported INGESTION_ATS"):
            self.run_with_steps("gamma")


if __name__ == "__main__":
    unittest.main()
