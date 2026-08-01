from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from data_pipeline.ingestion import workable


class FakeResponse:
    def __init__(
        self,
        *,
        text: str = "",
        payload: object | None = None,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.text = text
        self._payload = payload
        self.status_code = status_code
        self.headers: dict[str, str] = headers or {}

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")


class FakeSession:
    def __init__(self) -> None:
        self.get_urls: list[str] = []
        self.post_calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.get_urls.append(url)
        if url == "https://apply.workable.com/jasmax-1/":
            return FakeResponse(text="<html>No embedded state here</html>")
        if url == "https://apply.workable.com/api/v2/accounts/jasmax-1/jobs/2A7DBFAAC3":
            return FakeResponse(
                payload={
                    "id": 5910825,
                    "shortcode": "2A7DBFAAC3",
                    "title": "Senior Technical Advisor",
                    "state": "published",
                    "published": "2026-06-24T00:00:00.000Z",
                    "location": {"city": "Auckland", "region": "Auckland", "country": "New Zealand"},
                    "locations": [{"city": "Auckland", "region": "Auckland", "country": "New Zealand"}],
                    "department": [],
                    "workplace": "on_site",
                    "description": "<p>Jasmax is seeking a Senior Technical Advisor.</p>",
                    "requirements": "<p>Registered architect.</p>",
                    "benefits": "<p>Professional development.</p>",
                }
            )
        raise AssertionError(f"Unexpected GET {url}")

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.post_calls.append((url, kwargs.get("json") or {}))
        self.post_json = kwargs.get("json")
        return FakeResponse(
            payload={
                "total": 1,
                "results": [
                    {
                        "id": 5910825,
                        "shortcode": "2A7DBFAAC3",
                        "title": "Senior Technical Advisor",
                        "state": "published",
                        "published": "2026-06-24T00:00:00.000Z",
                    }
                ],
            }
        )


class WorkableIngestionTests(unittest.TestCase):
    def test_fetch_workable_page_falls_back_to_modern_api(self) -> None:
        session = FakeSession()

        with (
            patch.object(workable, "MIN_REQUEST_GAP_SECONDS", 0),
            patch.object(workable, "RANDOM_JITTER_MAX_SECONDS", 0),
            patch.object(workable, "_last_request_by_host", {}),
        ):
            jobs, company_title = workable.fetch_workable_page("https://apply.workable.com/jasmax-1/", session)  # type: ignore[arg-type]

        self.assertEqual(company_title, "jasmax-1")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(session.post_calls[0][0], "https://apply.workable.com/api/v3/accounts/jasmax-1/jobs")
        self.assertEqual(
            session.post_calls[0][1],
            {"query": "", "department": [], "location": [], "workplace": [], "worktype": []},
        )
        self.assertEqual(jobs[0]["description"], "<p>Jasmax is seeking a Senior Technical Advisor.</p>")
        self.assertEqual(jobs[0]["url"], "https://apply.workable.com/jasmax-1/j/2A7DBFAAC3/")

    def test_normalizes_modern_api_job_detail(self) -> None:
        fetched_at = datetime(2026, 6, 28, tzinfo=timezone.utc)
        record = workable.normalize_job(
            {
                "id": 5910825,
                "shortcode": "2A7DBFAAC3",
                "title": "Senior Technical Advisor",
                "published": "2026-06-24T00:00:00.000Z",
                "location": {"city": "Auckland", "region": "Auckland", "country": "New Zealand"},
                "description": "<p>Jasmax is seeking a Senior Technical Advisor.</p>",
                "requirements": "<p>Registered architect.</p>",
                "benefits": "<p>Professional development.</p>",
                "url": "https://apply.workable.com/jasmax-1/j/2A7DBFAAC3/",
                "workable_api_version": "v3_list_v2_detail",
            },
            "https://apply.workable.com/jasmax-1/",
            "jasmax-1",
            fetched_at,
        )

        self.assertEqual(record["source_job_id"], "5910825")
        self.assertEqual(record["source_company"], "jasmax-1")
        self.assertEqual(record["posted_at"], "2026-06-24T00:00:00.000Z")
        self.assertEqual(record["location_name"], "Auckland, Auckland, New Zealand")
        self.assertIn("Registered architect.", record["content_text"] or "")
        self.assertEqual(record["job_url"], "https://apply.workable.com/jasmax-1/j/2A7DBFAAC3/")
        self.assertEqual(record["raw_json"]["shortcode"], "2A7DBFAAC3")

    def test_retry_after_response_fails_source_without_sleeping(self) -> None:
        class RetryAfterSession:
            def get(self, url: str, **kwargs: object) -> FakeResponse:
                return FakeResponse(status_code=429, headers={"Retry-After": "83192"})

        with (
            patch.object(workable, "MIN_REQUEST_GAP_SECONDS", 0),
            patch.object(workable, "RANDOM_JITTER_MAX_SECONDS", 0),
            patch.object(workable, "_last_request_by_host", {}),
            patch.object(workable.time, "sleep") as sleep_mock,
        ):
            with self.assertRaises(workable.RetryAfterSourceFailure) as raised:
                workable.polite_get(RetryAfterSession(), "https://apply.workable.com/reno-orthopedic-center/")  # type: ignore[arg-type]

        self.assertEqual(getattr(raised.exception.response, "status_code", None), 429)
        sleep_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
