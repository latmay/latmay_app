from __future__ import annotations

"""
Fetch Lever public job posting JSON endpoints and upsert normalized jobs into
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
from typing import Any, Iterable
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
from data_pipeline.ingestion.size_limits import MAX_LISTING_RESPONSE_BYTES, read_response_with_limit

REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = "Mozilla/5.0 (compatible; latmay-lever/1.0)"
MIN_REQUEST_GAP_SECONDS = float(os.environ.get("LEVER_REQUEST_GAP_SECONDS", os.environ.get("REQUEST_GAP_SECONDS", "2")))
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


def lever_company_slug_from_url(url: str) -> str:
    parts = urlparse(url).path.strip("/").split("/")
    if len(parts) >= 3 and parts[1] == "postings":
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


def join_nonempty(parts: Iterable[str | None], sep: str = "\n\n") -> str | None:
    cleaned = [part.strip() for part in parts if isinstance(part, str) and part.strip()]
    return sep.join(cleaned) if cleaned else None


def flatten_lists_text(lists_value: list[dict[str, Any]]) -> str | None:
    sections: list[str] = []
    for item in lists_value:
        if not isinstance(item, dict):
            continue
        heading = item.get("text")
        content_text = strip_html_to_text(item.get("content"))
        section = join_nonempty([heading, content_text], sep="\n")
        if section:
            sections.append(section)
    return join_nonempty(sections)


def unix_ms_to_iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).replace(microsecond=0).isoformat()


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
            ("lever", source_company),
        )
        rows = cur.fetchall()

    fingerprints: dict[str, str] = {}
    for row in rows:
        fingerprint = row.get("listing_fingerprint")
        source_job_id = str(row.get("source_job_id") or "").strip()
        if source_job_id and isinstance(fingerprint, str):
            fingerprints[source_job_id] = fingerprint
    return fingerprints


def fetch_lever_json(url: str) -> list[dict[str, Any]]:
    sleep_before_request()
    _t = time.monotonic()
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=REQUEST_TIMEOUT_SECONDS,
        stream=True,
    )
    log_timing("lever", lever_company_slug_from_url(url).rstrip("-"), "http_get", time.monotonic() - _t)
    response.raise_for_status()
    read_response_with_limit(response, MAX_LISTING_RESPONSE_BYTES)
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Expected Lever response to be a JSON list.")
    return payload


def normalize_job(job: dict[str, Any], source_url: str, fetched_at_utc: datetime) -> dict[str, Any]:
    categories = job.get("categories") if isinstance(job.get("categories"), dict) else {}
    lists_value = job.get("lists") if isinstance(job.get("lists"), list) else []

    opening_html = job.get("opening")
    description_html = job.get("description")
    description_body_html = job.get("descriptionBody")
    additional_html = job.get("additional")

    content_html = join_nonempty(
        [opening_html, description_html, description_body_html, additional_html]
        + [item.get("content") for item in lists_value if isinstance(item, dict)]
    )
    content_text = join_nonempty(
        [
            job.get("openingPlain") or strip_html_to_text(opening_html),
            job.get("descriptionPlain") or strip_html_to_text(description_html),
            job.get("descriptionBodyPlain") or strip_html_to_text(description_body_html),
            job.get("additionalPlain") or strip_html_to_text(additional_html),
            flatten_lists_text(lists_value),
        ]
    )
    all_locations = categories.get("allLocations")
    office_names = (
        " | ".join(str(location) for location in all_locations if str(location).strip())
        if isinstance(all_locations, list)
        else None
    )
    source_job_id = job.get("id") or job.get("hostedUrl") or job.get("applyUrl") or f"{source_url}:{job.get('text')}"
    source_company = lever_company_slug_from_url(source_url).rstrip("-")

    raw_json = dict(job)
    record = {
        "source_type": "lever",
        "source_url": source_url,
        "source_job_id": str(source_job_id),
        "source_company": source_company,
        "company_name": source_company,
        "title": job.get("text"),
        "location_name": categories.get("location"),
        "department_names": categories.get("team"),
        "office_names": office_names,
        "posted_at": unix_ms_to_iso(job.get("createdAt") if isinstance(job.get("createdAt"), int) else None),
        "updated_at": None,
        "job_url": job.get("hostedUrl"),
        "apply_url": job.get("applyUrl"),
        "content_html": content_html,
        "content_text": content_text,
        "raw_json": raw_json,
        "fetched_at_utc": fetched_at_utc,
    }
    record["raw_json"]["listing_fingerprint"] = job_listing_fingerprint(record)
    return record


def collect_jobs_for_source(conn, source_url: str) -> tuple[list[dict[str, Any]], list[str], int, dict[str, Any]]:
    jobs = fetch_lever_json(source_url)
    fetched_at_utc = utc_now()
    source_company = lever_company_slug_from_url(source_url).rstrip("-")
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
        f"lever: fetched {len(all_records)} jobs from {source_company}; "
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
    sources = load_ats_source_urls_from_db(conn, "lever")
    total_sources = len(sources)
    skipped_404_sources = count_skipped_404_sources_from_db(conn, "lever")
    http_errors = AtsHttpErrorTracker("lever")

    for source_url in sources:
        if should_stop_for_ingestion_budget("lever", started_at, completed_sources):
            break
        attempted_sources += 1
        source_company = lever_company_slug_from_url(source_url).rstrip("-")
        attempt_http_status_code: int | None = None
        attempt_error_type: str | None = None
        stop_after_source = False
        try:
            _t = time.monotonic()
            records, seen_source_job_ids, skipped_count, quality_summary = collect_jobs_for_source(conn, source_url)
            log_timing("lever", source_company, "fetch", time.monotonic() - _t)

            batch.extend(records)

            _t = time.monotonic()
            marked_missing = mark_missing_jobs_for_source(
                conn,
                source_type="lever",
                source_company=source_company,
                seen_source_job_ids=seen_source_job_ids,
                seen_at_utc=records[0]["fetched_at_utc"] if records else utc_now(),
            )
            log_timing("lever", source_company, "mark_missing", time.monotonic() - _t)

            total_marked_missing += marked_missing
            if marked_missing:
                print(f"lever: marked {marked_missing} jobs missing for {source_company}", flush=True)
            log_data_quality(
                "ingestion",
                source_type="lever",
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
                total_count += flush_batch(conn, batch, "lever")
                log_timing("lever", source_company, "flush", time.monotonic() - _t)
                sources_in_batch = 0
        except Exception as exc:
            error_sources += 1
            http_status_code = http_status_code_from_exception(exc)
            attempt_http_status_code = http_status_code
            attempt_error_type = type(exc).__name__
            stop_after_source = isinstance(exc, AtsHttp429LimitReached) or http_errors.record(http_status_code)
            print(
                "lever: error fetching source: "
                f"company={source_company}, error_type={type(exc).__name__}, "
                f"http_status_code={http_status_code or 'unknown'}",
                flush=True,
            )
            log_data_quality(
                "ingestion",
                source_type="lever",
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
            log_timing("lever", source_company, "last_get_at", time.monotonic() - _t)
        if stop_after_source:
            break

    total_count += flush_batch(conn, batch, "lever")
    checked_percent = (attempted_sources / total_sources * 100) if total_sources else 0.0
    print(
        f"lever: checked {attempted_sources}/{total_sources} sources ({checked_percent:.1f}%); errors={error_sources}",
        flush=True,
    )
    http_errors.print_summary()
    print(f"lever: skipped {skipped_404_sources} sources due to 404 streak", flush=True)
    print(
        f"lever: inserted/updated {total_count} rows total; marked {total_marked_missing} missing",
        flush=True,
    )
    return total_count


if __name__ == "__main__":
    from data_pipeline.common.db import connect

    with connect() as connection:
        run(connection)
