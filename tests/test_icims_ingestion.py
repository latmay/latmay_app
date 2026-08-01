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

from data_pipeline.ingestion import icims
from data_pipeline.ingestion.ats_detection import ats_match_from_url, ats_source_entry_url


class IcimsIngestionTests(unittest.TestCase):
    def test_api_url_derives_fixed_path_from_listing_url(self) -> None:
        self.assertEqual(
            icims.api_url_from_listing_url("https://careers.fm.com/careers-home/jobs"),
            "https://careers.fm.com/api/jobs",
        )
        self.assertEqual(
            icims.api_url_from_listing_url("https://jobs.tufts.edu/jobs"),
            "https://jobs.tufts.edu/api/jobs",
        )

    def test_salary_classifier_handles_split_min_max_tags(self) -> None:
        result = icims.classified_tag_values(
            {
                "tags2": "Full/Part-time",
                "tags3": "office/home",
                "tags6": "USD $94,000.00/Yr.",
                "tags7": "USD $120,000.00/Yr.",
            }
        )

        self.assertEqual(result["salary_min"], 94000.0)
        self.assertEqual(result["salary_max"], 120000.0)
        self.assertEqual(result["salary_currency"], "USD")
        self.assertEqual(result["salary_period"], "year")

    def test_salary_classifier_handles_min_mid_max_single_string(self) -> None:
        result = icims.classified_tag_values(
            {
                "tags1": "Hybrid",
                "tags3": "Full-Time",
                "tags5": "Minimum $155,100.00, Midpoint $193,900.00, Maximum $232,600.00",
            }
        )

        self.assertEqual(result["salary_min"], 155100.0)
        self.assertEqual(result["salary_mid"], 193900.0)
        self.assertEqual(result["salary_max"], 232600.0)
        self.assertEqual(result["workplace"], "hybrid")
        self.assertEqual(result["employment_time_from_tags"], "Full-Time")

    def test_normalize_job_uses_shared_icims_fields_and_raw_tags(self) -> None:
        fetched_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        job = {
            "data": {
                "slug": "engineer-123",
                "req_id": "REQ-123",
                "title": "Software Engineer",
                "description": "<p>Build systems.</p>",
                "qualifications": "<p>Python</p>",
                "city": "Boston",
                "state": "MA",
                "country": "US",
                "categories": [{"name": "Engineering"}],
                "employment_type": "Full-Time",
                "posted_date": "2026-01-01",
                "apply_url": "https://tenant.icims.com/jobs/123/apply",
                "hiring_organization": "Example Org",
                "tags1": "Remote",
                "meta_data": {
                    "canonical_url": "https://jobs.example.com/jobs/engineer-123?lang=en-us",
                    "client_code": "example",
                    "ats_code": "icims",
                },
            }
        }

        record = icims.normalize_job(job, "https://jobs.example.com/jobs", fetched_at)

        self.assertEqual(record["source_type"], "icims")
        self.assertEqual(record["source_job_id"], "REQ-123")
        self.assertEqual(record["source_company"], "jobs.example.com")
        self.assertEqual(record["company_name"], "Example Org")
        self.assertEqual(record["department_names"], "Engineering")
        self.assertEqual(record["job_url"], "https://jobs.example.com/jobs/engineer-123?lang=en-us")
        self.assertEqual(record["raw_json"]["raw_tags"], {"tags1": "Remote"})
        self.assertEqual(record["raw_json"]["workplace"], "remote")
        self.assertIn("Build systems", record["content_text"])

    def test_viasat_style_payload_uses_top_level_ats_code_and_list_tags(self) -> None:
        fetched_at = datetime(2026, 6, 16, tzinfo=timezone.utc)
        payload = {
            "jobs": [
                {
                    "data": {
                        "slug": "6322",
                        "client_code": "viasat",
                        "req_id": "6322",
                        "title": "Technical Regulatory and Market Access Lead",
                        "description": "<p>Translate regulatory requirements.</p>",
                        "qualifications": "<ul><li>Telecom experience</li></ul>",
                        "responsibilities": "<p>Support global launch planning.</p>",
                        "location_name": "London",
                        "city": "London",
                        "country": "United Kingdom",
                        "categories": [{"name": "Product Management"}],
                        "posted_date": "2026-06-16T11:36:00+0000",
                        "apply_url": "https://careers-viasat.icims.com/jobs/6322/login",
                        "hiring_organization": "Viasat, Inc.",
                        "tags1": ["Regular"],
                        "tags2": ["None"],
                        "ats_code": "icims",
                        "meta_data": {
                            "icims": {"jps_is_public": True},
                            "canonical_url": "https://careers.viasat.com/jobs/6322?lang=en-us",
                        },
                    }
                }
            ],
            "totalCount": 1,
        }

        icims.validate_icims_payload(payload)
        record = icims.normalize_job(payload["jobs"][0], "https://careers.viasat.com/jobs", fetched_at)

        self.assertEqual(record["source_job_id"], "6322")
        self.assertEqual(record["company_name"], "Viasat, Inc.")
        self.assertEqual(record["department_names"], "Product Management")
        self.assertEqual(record["job_url"], "https://careers.viasat.com/jobs/6322?lang=en-us")
        self.assertEqual(record["raw_json"]["client_code"], "viasat")
        self.assertEqual(record["raw_json"]["raw_tags"], {"tags1": "Regular", "tags2": "None"})
        self.assertIn("Translate regulatory requirements.", record["content_text"])

    def test_ats_detection_recognizes_icims_api_and_preserves_discovered_url(self) -> None:
        match = ats_match_from_url("https://jobs.tufts.edu/api/jobs?page=1")

        self.assertEqual(match["provider"], "icims")
        self.assertEqual(ats_source_entry_url(match), "https://jobs.tufts.edu/api/jobs?page=1")


if __name__ == "__main__":
    unittest.main()
