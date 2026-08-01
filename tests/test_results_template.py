from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

from main import app


class ResultsTemplateTests(unittest.TestCase):
    def test_empty_results_show_message_without_diagnostics(self) -> None:
        payload = {
            "total_jobs": 100,
            "filtered_jobs": 0,
            "results": [],
            "grouped_results": [],
        }

        with app.test_request_context("/rank"):
            html = app.jinja_env.get_template("results.html").render(
                payload=payload,
                country=None,
                max_required_yoe=None,
                top_k_to_show=10,
                show_ranking_diagnostics=True,
            )

        self.assertIn("No matches found", html)
        self.assertIn(
            "We're sorry, we did not find any jobs matching this particular resume. Please try again tomorrow.",
            html,
        )
        self.assertNotIn("Ranking Metrics", html)
        self.assertNotIn("Total jobs", html)
        self.assertNotIn("Jobs after filters", html)


if __name__ == "__main__":
    unittest.main()
