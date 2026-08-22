from __future__ import annotations

import unittest

from data_pipeline.ingestion import ashby, greenhouse, lever


class AtsCareerUrlResolutionTests(unittest.TestCase):
    def test_greenhouse_career_page_builds_api_endpoint(self) -> None:
        self.assertEqual(
            greenhouse.greenhouse_endpoint_from_url("https://job-boards.greenhouse.io/acme/jobs/123"),
            "https://boards-api.greenhouse.io/v1/boards/acme/jobs?content=true",
        )
        self.assertEqual(
            greenhouse.board_token_from_url(
                "https://boards-api.greenhouse.io/v1/boards/acme/jobs?content=true"
            ),
            "acme",
        )

    def test_lever_career_page_builds_api_endpoint(self) -> None:
        self.assertEqual(
            lever.lever_endpoint_from_url("https://jobs.lever.co/acme/8c11-job-id"),
            "https://api.lever.co/v0/postings/acme?mode=json",
        )
        self.assertEqual(
            lever.lever_company_slug_from_url("https://api.lever.co/v0/postings/acme?mode=json"),
            "acme",
        )

    def test_ashby_career_page_builds_api_endpoint(self) -> None:
        self.assertEqual(
            ashby.ashby_endpoint_from_url("https://jobs.ashbyhq.com/acme/123"),
            "https://api.ashbyhq.com/posting-api/job-board/acme?includeCompensation=true",
        )
        self.assertEqual(
            ashby.board_token_from_url(
                "https://api.ashbyhq.com/posting-api/job-board/acme?includeCompensation=true"
            ),
            "acme",
        )

    def test_malformed_urls_fail_early_with_provider_specific_messages(self) -> None:
        cases = (
            (greenhouse.greenhouse_endpoint_from_url, "https://job-boards.greenhouse.io", "Greenhouse"),
            (lever.lever_endpoint_from_url, "lever-company", "Lever"),
            (ashby.ashby_endpoint_from_url, "https://example.com/acme", "Ashby"),
        )
        for resolver, url, provider in cases:
            with self.subTest(provider=provider), self.assertRaisesRegex(ValueError, provider):
                resolver(url)


if __name__ == "__main__":
    unittest.main()
