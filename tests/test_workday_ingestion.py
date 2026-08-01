from __future__ import annotations

import sys
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

fake_psycopg = types.ModuleType("psycopg")
fake_psycopg_rows = types.ModuleType("psycopg.rows")
fake_psycopg_types = types.ModuleType("psycopg.types")
fake_psycopg_types_json = types.ModuleType("psycopg.types.json")
fake_psycopg_rows.dict_row = object()
fake_psycopg_types_json.Jsonb = dict
sys.modules.setdefault("psycopg", fake_psycopg)
sys.modules.setdefault("psycopg.rows", fake_psycopg_rows)
sys.modules.setdefault("psycopg.types", fake_psycopg_types)
sys.modules.setdefault("psycopg.types.json", fake_psycopg_types_json)

from data_pipeline.ingestion import workday


class WorkdayIngestionTests(unittest.TestCase):
    def test_parse_workday_site_defaults_locale(self) -> None:
        site_info = workday.parse_workday_site("https://ptc.wd1.myworkdayjobs.com/ptc/")

        self.assertEqual(site_info["scheme"], "https")
        self.assertEqual(site_info["host"], "ptc.wd1.myworkdayjobs.com")
        self.assertEqual(site_info["tenant"], "ptc")
        self.assertEqual(site_info["locale"], "en-US")
        self.assertEqual(site_info["site_slug"], "ptc")
        self.assertEqual(site_info["source_company"], "ptc:ptc")

    def test_parse_workday_site_uses_locale_path_segment(self) -> None:
        site_info = workday.parse_workday_site("https://draper.wd5.myworkdayjobs.com/en-US/Draper_Careers")

        self.assertEqual(site_info["tenant"], "draper")
        self.assertEqual(site_info["locale"], "en-US")
        self.assertEqual(site_info["site_slug"], "Draper_Careers")
        self.assertEqual(site_info["source_company"], "draper:Draper_Careers")

    def test_build_detail_url_uses_public_workday_path(self) -> None:
        site_info = workday.parse_workday_site("https://ptc.wd1.myworkdayjobs.com/ptc/")

        detail_url = workday.build_detail_url(
            site_info,
            "/job/Boston-MA-USA/Strategic-Account-Executive---Onshape_JR111938",
        )

        self.assertEqual(
            detail_url,
            "https://ptc.wd1.myworkdayjobs.com/en-US/ptc/job/Boston-MA-USA/Strategic-Account-Executive---Onshape_JR111938",
        )

    def test_extract_json_ld_jobposting(self) -> None:
        html = """
        <html>
          <script type="application/ld+json">
          {
            "@context": "http://schema.org",
            "@type": "JobPosting",
            "title": "Engineer",
            "description": "<p>Build things</p>",
            "identifier": {"value": "JR123"}
          }
          </script>
        </html>
        """

        jobposting = workday.extract_json_ld_jobposting(html)

        self.assertEqual(jobposting["title"], "Engineer")
        self.assertEqual(jobposting["identifier"]["value"], "JR123")

    def test_existing_content_reuse_requires_unchanged_listing_fields(self) -> None:
        summary = {
            "title": "Engineer",
            "locationsText": "Boston, MA",
            "externalPath": "/job/123",
            "postedOn": "2026-01-01",
        }
        existing_row = {
            "title": "Engineer",
            "location_name": "Boston, MA",
            "job_url": "https://example.com/en-US/site/job/123",
            "content_text": "Already stored",
            "raw_json": {"listing_fingerprint": workday.workday_listing_fingerprint(summary)},
        }

        self.assertTrue(
            workday.can_reuse_existing_content(
                existing_row,
                summary,
                "https://example.com/en-US/site/job/123",
            )
        )

        summary["title"] = "Senior Engineer"
        self.assertFalse(
            workday.can_reuse_existing_content(
                existing_row,
                summary,
                "https://example.com/en-US/site/job/123",
            )
        )

    def test_normalize_workday_job_uses_json_ld_fields(self) -> None:
        site_info = workday.parse_workday_site("https://ptc.wd1.myworkdayjobs.com/ptc/")
        fetched_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

        record = workday.normalize_workday_job(
            {"title": "Listing title", "externalPath": "/job/123", "locationsText": "Remote"},
            site_info,
            fetched_at,
            detail_url="https://ptc.wd1.myworkdayjobs.com/en-US/ptc/job/123",
            json_ld={
                "title": "JSON-LD title",
                "description": "<p>Job description</p>",
                "datePosted": "2026-01-01",
                "identifier": {"value": "JR123"},
                "hiringOrganization": {"name": "PTC"},
                "employmentType": "FULL_TIME",
            },
        )

        self.assertEqual(record["source_type"], "workday")
        self.assertEqual(record["source_job_id"], "JR123")
        self.assertEqual(record["company_name"], "PTC")
        self.assertEqual(record["title"], "JSON-LD title")
        self.assertEqual(record["content_text"], "Job description")
        self.assertEqual(record["raw_json"]["employment_type"], "FULL_TIME")

    def test_collect_workday_summaries_caps_jobs_per_source(self) -> None:
        site_info = workday.parse_workday_site("https://ptc.wd1.myworkdayjobs.com/ptc/")
        original_cap = workday.WORKDAY_MAX_LIST_JOBS_PER_SOURCE
        original_page_size = workday.PAGE_SIZE
        original_fetch = workday.fetch_workday_jobs_page
        calls: list[tuple[int, int]] = []

        def fake_fetch(_site_info, *, limit: int, offset: int):
            calls.append((limit, offset))
            return {
                "total": 5,
                "jobPostings": [
                    {"externalPath": f"/job/{offset + 1}", "title": f"Job {offset + 1}"},
                    {"externalPath": f"/job/{offset + 2}", "title": f"Job {offset + 2}"},
                    {"externalPath": f"/job/{offset + 3}", "title": f"Job {offset + 3}"},
                ],
            }

        try:
            workday.WORKDAY_MAX_LIST_JOBS_PER_SOURCE = 2
            workday.PAGE_SIZE = 3
            workday.fetch_workday_jobs_page = fake_fetch

            summaries = workday.collect_workday_summaries(site_info)
        finally:
            workday.WORKDAY_MAX_LIST_JOBS_PER_SOURCE = original_cap
            workday.PAGE_SIZE = original_page_size
            workday.fetch_workday_jobs_page = original_fetch

        self.assertEqual(len(summaries), 2)
        self.assertEqual([summary["externalPath"] for summary in summaries], ["/job/1", "/job/2"])
        self.assertEqual(calls, [(3, 0)])


if __name__ == "__main__":
    unittest.main()
