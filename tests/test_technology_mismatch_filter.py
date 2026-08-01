from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

from ranking_algorithms.technology_mismatch_filter import (
    TechnologyTerm,
    _normalized_aliases,
    match_technologies,
)


class TechnologyMismatchFilterTest(unittest.TestCase):
    def test_safe_one_letter_language_aliases_match(self) -> None:
        terms = [
            TechnologyTerm(category="Programming Languages", name="C", aliases=_normalized_aliases("C")),
            TechnologyTerm(category="Programming Languages", name="R", aliases=_normalized_aliases("R")),
        ]

        matches = match_technologies(
            "Embedded systems with C programming and data analysis in R language.",
            terms,
        )

        self.assertEqual(matches["technologies"], ["C", "R"])
        self.assertEqual(matches["categories"], ["Programming Languages"])

    def test_bare_one_letter_aliases_remain_filtered(self) -> None:
        terms = [
            TechnologyTerm(category="Programming Languages", name="C", aliases=_normalized_aliases("C")),
            TechnologyTerm(category="Programming Languages", name="R", aliases=_normalized_aliases("R")),
        ]

        self.assertNotIn("c", terms[0].aliases)
        self.assertNotIn("r", terms[1].aliases)
        self.assertEqual(match_technologies("Worked with plan C and option R.", terms)["technologies"], [])


if __name__ == "__main__":
    unittest.main()
