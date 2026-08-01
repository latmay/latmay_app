from __future__ import annotations

"""
Fetch Ashby public job-board JSON endpoints and upsert normalized jobs into
PostgreSQL. No SQLite databases or CSV files are created.
"""

import os
import random
import re
import time
import hashlib
import json
from datetime import datetime, timezone
from html import unescape
from typing import Any
from urllib.parse import urlparse

import requests

from data_pipeline.common.db import mark_missing_jobs_for_source, upsert_jobs
from data_pipeline.common.data_quality import count_blank, duplicate_count, http_status_code_from_exception, log_data_quality
from data_pipeline.common.timing import log_timing
from data_pipeline.ingestion.budget import ingestion_budget_started_at, should_stop_for_ingestion_budget
from data_pipeline.ingestion.http_error_tracker import AtsHttp429LimitReached, AtsHttpErrorTracker
from data_pipeline.ingestion.source_loader import (
    count_skipped_404_sources_from_db,
    load_ats_source_urls_from_db,
    rollback_failed_source_attempt,
    update_source_last_get_at,
)

REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = "Mozilla/5.0 (compatible; latmay-ashby/1.0)"
MIN_REQUEST_GAP_SECONDS = float(os.environ.get("ASHBY_REQUEST_GAP_SECONDS", os.environ.get("REQUEST_GAP_SECONDS", "2")))
RANDOM_JITTER_MAX_SECONDS = 0.35
BATCH_SOURCE_COUNT = 5
_last_request_time: float | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def sleep_before_request() -> None:
    global _last_request_time

    now = time.monotonic()
    if _last_request_time is not None:
        wait = MIN_REQUEST_GAP_SECONDS + random.uniform(0, RANDOM_JITTER_MAX_SECONDS) - (now - _last_request_time)
        if wait > 0:
            time.sleep(wait)

    _last_request_time = time.monotonic()


def board_token_from_url(url: str) -> str:
    parts = urlparse(url).path.strip("/").split("/")
    if len(parts) >= 3 and parts[0] == "posting-api" and parts[1] == "job-board":
        return parts[2]
    return "unknown"


def strip_html_to_text(html_text: str | None) -> str | None:
    if not html_text:
        return None

    text = unescape(html_text)
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", text)
    text = re.sub(r"(?i)</\s*(p|div|h[1-6]|li)\s*>", "\n", text)
    text = re.sub(r"(?i)<\s*li[^>]*>", "- ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text.replace("\r", ""))
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip() or None


def stable_fingerprint(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def job_listing_fingerprint(record: dict[str, Any]) -> str:
    content_text = record.get("content_text")
    content_hash = (
        hashlib.sha256(content_text.encode("utf-8")).hexdigest()
        if isinstance(content_text, str)
        else None
    )
    return stable_fingerprint(
        {
            "source_job_id": record.get("source_job_id"),
            "title": record.get("title"),
            "location_name": record.get("location_name"),
            "posted_at": record.get("posted_at"),
            "updated_at": record.get("updated_at"),
            "job_url": record.get("job_url"),
            "apply_url": record.get("apply_url"),
            "content_text_hash": content_hash,
        }
    )


def existing_fingerprints_for_source(conn, source_company: str) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT source_job_id, raw_json->>'listing_fingerprint' AS listing_fingerprint
            FROM jobs
            WHERE source_type = %s
              AND source_company = %s
            """,
            ("ashby", source_company),
        )
        rows = cur.fetchall()

    fingerprints: dict[str, str] = {}
    for row in rows:
        fingerprint = row.get("listing_fingerprint")
        source_job_id = str(row.get("source_job_id") or "").strip()
        if source_job_id and isinstance(fingerprint, str):
            fingerprints[source_job_id] = fingerprint
    return fingerprints


def fetch_ashby_json(url: str) -> dict[str, Any]:
    sleep_before_request()
    _t = time.monotonic()
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    log_timing("ashby", board_token_from_url(url), "http_get", time.monotonic() - _t)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Expected Ashby response to be a JSON object.")
    return payload


def normalize_job(job: dict[str, Any], source_url: str, fetched_at_utc: datetime) -> dict[str, Any]:
    board_token = board_token_from_url(source_url)
    address = job.get("address") if isinstance(job.get("address"), dict) else {}
    postal_address = address.get("postalAddress") if isinstance(address.get("postalAddress"), dict) else {}
    secondary_locations = job.get("secondaryLocations") or []
    secondary_location_names = " | ".join(x for x in secondary_locations if isinstance(x, str) and x.strip()) or None

    description_html = job.get("descriptionHtml")
    content_text = job.get("descriptionPlain") or strip_html_to_text(description_html)
    source_job_id = job.get("id") or job.get("jobUrl") or job.get("applyUrl") or f"{source_url}:{job.get('title')}"

    record = {
        "source_type": "ashby",
        "source_url": source_url,
        "source_job_id": str(source_job_id),
        "source_company": board_token,
        "company_name": board_token,
        "title": job.get("title"),
        "location_name": job.get("location"),
        "department_names": job.get("department") or job.get("team"),
        "office_names": secondary_location_names,
        "posted_at": job.get("publishedAt"),
        "updated_at": None,
        "job_url": job.get("jobUrl"),
        "apply_url": job.get("applyUrl"),
        "content_html": description_html,
        "content_text": content_text,
        "raw_json": {"job": job, "postal_address": postal_address},
        "fetched_at_utc": fetched_at_utc,
    }
    record["raw_json"]["listing_fingerprint"] = job_listing_fingerprint(record)
    return record


def collect_jobs_for_source(conn, source_url: str) -> tuple[list[dict[str, Any]], list[str], int, dict[str, Any]]:
    payload = fetch_ashby_json(source_url)
    jobs = payload.get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError("Expected Ashby payload['jobs'] to be a list.")

    fetched_at_utc = utc_now()
    source_company = board_token_from_url(source_url)
    existing_fingerprints = existing_fingerprints_for_source(conn, source_company)
    all_records = [
        normalize_job(job, source_url, fetched_at_utc)
        for job in jobs
        if isinstance(job, dict)
    ]
    records = [
        record
        for record in all_records
        if existing_fingerprints.get(record["source_job_id"]) != record["raw_json"]["listing_fingerprint"]
    ]
    skipped_count = len(all_records) - len(records)
    summary = {
        "fetched": len(all_records),
        "unchanged_skipped": skipped_count,
        "to_upsert": len(records),
        "active_seen": len(all_records),
        "missing_title": count_blank(all_records, "title"),
        "missing_url": count_blank(all_records, "job_url"),
        "missing_location": count_blank(all_records, "location_name"),
        "missing_content_text": count_blank(all_records, "content_text"),
        "duplicate_source_job_ids": duplicate_count(record.get("source_job_id") for record in all_records),
    }
    print(
        f"ashby: fetched {len(all_records)} jobs from {source_company}; "
        f"unchanged_skipped={skipped_count}, to_upsert={len(records)}",
        flush=True,
    )
    return records, [record["source_job_id"] for record in all_records], skipped_count, summary


def flush_batch(conn, records: list[dict[str, Any]], source_type: str) -> int:
    if not records:
        return 0

    count = upsert_jobs(conn, records)
    print(f"{source_type}: inserted/updated {count} rows in batch", flush=True)
    records.clear()
    return count


def run(conn) -> int:
    batch: list[dict[str, Any]] = []
    total_count = 0
    sources_in_batch = 0
    total_marked_missing = 0
    completed_sources = 0
    attempted_sources = 0
    error_sources = 0
    started_at = ingestion_budget_started_at()
    sources = load_ats_source_urls_from_db(conn, "ashby")
    total_sources = len(sources)
    skipped_404_sources = count_skipped_404_sources_from_db(conn, "ashby")
    http_errors = AtsHttpErrorTracker("ashby")

    for source_url in sources:
        if should_stop_for_ingestion_budget("ashby", started_at, completed_sources):
            break
        attempted_sources += 1
        source_company = board_token_from_url(source_url)
        attempt_http_status_code: int | None = None
        attempt_error_type: str | None = None
        stop_after_source = False
        try:
            _t = time.monotonic()
            records, seen_source_job_ids, skipped_count, quality_summary = collect_jobs_for_source(conn, source_url)
            log_timing("ashby", source_company, "fetch", time.monotonic() - _t)

            batch.extend(records)

            _t = time.monotonic()
            marked_missing = mark_missing_jobs_for_source(
                conn,
                source_type="ashby",
                source_company=source_company,
                seen_source_job_ids=seen_source_job_ids,
                seen_at_utc=records[0]["fetched_at_utc"] if records else utc_now(),
            )
            log_timing("ashby", source_company, "mark_missing", time.monotonic() - _t)

            total_marked_missing += marked_missing
            if marked_missing:
                print(f"ashby: marked {marked_missing} jobs missing for {source_company}", flush=True)
            log_data_quality(
                "ingestion",
                source_type="ashby",
                company=source_company,
                inserted_updated="unknown",
                marked_missing=marked_missing,
                request_failures=0,
                **quality_summary,
            )
            sources_in_batch += 1
            completed_sources += 1

            if sources_in_batch >= BATCH_SOURCE_COUNT:
                _t = time.monotonic()
                total_count += flush_batch(conn, batch, "ashby")
                log_timing("ashby", source_company, "flush", time.monotonic() - _t)
                sources_in_batch = 0
        except Exception as exc:
            error_sources += 1
            http_status_code = http_status_code_from_exception(exc)
            attempt_http_status_code = http_status_code
            attempt_error_type = type(exc).__name__
            stop_after_source = isinstance(exc, AtsHttp429LimitReached) or http_errors.record(http_status_code)
            print(
                "ashby: error fetching source: "
                f"board={source_company}, error_type={type(exc).__name__}, "
                f"http_status_code={http_status_code or 'unknown'}",
                flush=True,
            )
            log_data_quality(
                "ingestion",
                source_type="ashby",
                company=source_company,
                fetched=0,
                unchanged_skipped=0,
                to_upsert=0,
                inserted_updated=0,
                marked_missing=0,
                active_seen=0,
                missing_title=0,
                missing_url=0,
                missing_location=0,
                missing_content_text=0,
                duplicate_source_job_ids=0,
                request_failures=1,
                http_status_code=http_status_code,
            )
        finally:
            if attempt_error_type is not None:
                rollback_failed_source_attempt(conn)
            _t = time.monotonic()
            update_source_last_get_at(
                conn,
                source_url,
                http_status_code=attempt_http_status_code,
                error_type=attempt_error_type,
            )
            log_timing("ashby", source_company, "last_get_at", time.monotonic() - _t)
        if stop_after_source:
            break

    total_count += flush_batch(conn, batch, "ashby")
    checked_percent = (attempted_sources / total_sources * 100) if total_sources else 0.0
    print(
        f"ashby: checked {attempted_sources}/{total_sources} sources ({checked_percent:.1f}%); errors={error_sources}",
        flush=True,
    )
    http_errors.print_summary()
    print(f"ashby: skipped {skipped_404_sources} sources due to 404 streak", flush=True)
    print(
        f"ashby: inserted/updated {total_count} rows total; marked {total_marked_missing} missing",
        flush=True,
    )
    return total_count


if __name__ == "__main__":
    from data_pipeline.common.db import connect

    with connect() as connection:
        run(connection)
