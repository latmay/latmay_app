from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.enrichment.add_security_clearance import extract_security_clearance


class SecurityClearanceExtractionTests(unittest.TestCase):
    def test_detects_top_secret_sci_polygraph(self) -> None:
        result = extract_security_clearance(
            "Candidates must possess an active TS/SCI clearance with polygraph."
        )

        self.assertTrue(result["requires_clearance"])
        self.assertEqual(result["clearance_type"], "TS/SCI with Polygraph")
        self.assertIn("TS/SCI", result["clearance_evidence_text"])

    def test_detects_public_trust(self) -> None:
        result = extract_security_clearance("This role requires Public Trust suitability.")

        self.assertTrue(result["requires_clearance"])
        self.assertEqual(result["clearance_type"], "Public Trust")

    def test_detects_ability_to_obtain_clearance_from_requirements(self) -> None:
        result = extract_security_clearance(
            "General job overview.",
            "Must be eligible to obtain a clearance for U.S. government work.",
        )

        self.assertTrue(result["requires_clearance"])
        self.assertEqual(result["clearance_type"], "Ability to Obtain Clearance")

    def test_detects_secret_clearance(self) -> None:
        result = extract_security_clearance("An active Secret clearance is required.")

        self.assertTrue(result["requires_clearance"])
        self.assertEqual(result["clearance_type"], "Secret")

    def test_ignores_negated_clearance_requirement(self) -> None:
        result = extract_security_clearance("No security clearance required for this role.")

        self.assertFalse(result["requires_clearance"])
        self.assertIsNone(result["clearance_type"])
        self.assertIsNone(result["clearance_evidence_text"])

    def test_no_clearance_mention_returns_false(self) -> None:
        result = extract_security_clearance("Experience with Python and SQL is required.")

        self.assertFalse(result["requires_clearance"])
        self.assertIsNone(result["clearance_type"])


if __name__ == "__main__":
    unittest.main()
