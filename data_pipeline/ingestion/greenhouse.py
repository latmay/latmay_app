from __future__ import annotations

"""
Fetch Greenhouse public job-board JSON endpoints and upsert normalized jobs
into PostgreSQL. No SQLite databases or CSV files are created.
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
from urllib.parse import quote, urlparse

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
USER_AGENT = "Mozilla/5.0 (compatible; latmay-greenhouse/1.0)"
MIN_REQUEST_GAP_SECONDS = float(os.environ.get("GREENHOUSE_REQUEST_GAP_SECONDS", os.environ.get("REQUEST_GAP_SECONDS", "2")))
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


def greenhouse_source(url: str) -> tuple[str, str]:
    """Return (board token, API endpoint) for a Greenhouse page or API URL."""
    parsed = urlparse(str(url or "").strip())
    host = (parsed.hostname or "").lower()
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.scheme not in {"http", "https"} or not host:
        raise ValueError(f"Invalid Greenhouse source URL (expected an absolute URL): {url!r}")

    if host in {"boards.greenhouse.io", "job-boards.greenhouse.io"}:
        token = parts[0] if parts else ""
    elif host == "boards-api.greenhouse.io" and len(parts) >= 3 and parts[:2] == ["v1", "boards"]:
        token = parts[2]
    else:
        raise ValueError(
            "Invalid Greenhouse source URL (expected boards.greenhouse.io/{token}, "
            "job-boards.greenhouse.io/{token}, or boards-api.greenhouse.io/v1/boards/{token}/jobs): "
            f"{url!r}"
        )
    if not token:
        raise ValueError(f"Invalid Greenhouse source URL (missing board token): {url!r}")

    endpoint = f"https://boards-api.greenhouse.io/v1/boards/{quote(token, safe='')}/jobs?content=true"
    return token, endpoint


def board_token_from_url(url: str) -> str:
    return greenhouse_source(url)[0]


def greenhouse_endpoint_from_url(url: str) -> str:
    return greenhouse_source(url)[1]


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


def names_from_objects(items: Iterable[dict[str, Any]] | None, key: str = "name") -> str | None:
    if not items:
        return None
    names = []
    for item in items:
        value = item.get(key) if isinstance(item, dict) else None
        if isinstance(value, str) and value.strip() and value.strip() not in names:
            names.append(value.strip())
    return " | ".join(names) if names else None


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
            ("greenhouse", source_company),
        )
        rows = cur.fetchall()

    fingerprints: dict[str, str] = {}
    for row in rows:
        fingerprint = row.get("listing_fingerprint")
        source_job_id = str(row.get("source_job_id") or "").strip()
        if source_job_id and isinstance(fingerprint, str):
            fingerprints[source_job_id] = fingerprint
    return fingerprints


def fetch_greenhouse_json(url: str) -> dict[str, Any]:
    sleep_before_request()
    _t = time.monotonic()
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=REQUEST_TIMEOUT_SECONDS,
        stream=True,
    )
    log_timing("greenhouse", board_token_from_url(url), "http_get", time.monotonic() - _t)
    response.raise_for_status()
    read_response_with_limit(response, MAX_LISTING_RESPONSE_BYTES)
    _t = time.monotonic()
    payload = response.json()
    log_timing("greenhouse", board_token_from_url(url), "json_parse", time.monotonic() - _t)
    if not isinstance(payload, dict):
        raise ValueError("Expected Greenhouse response to be a JSON object.")
    return payload


def normalize_job(
    job: dict[str, Any],
    source_url: str,
    fetched_at_utc: datetime,
    *,
    include_fingerprint: bool = True,
) -> dict[str, Any]:
    location = job.get("location") if isinstance(job.get("location"), dict) else {}
    content_html = job.get("content")
    source_job_id = job.get("id") or job.get("absolute_url") or f"{source_url}:{job.get('title')}"
    board_token = board_token_from_url(source_url)

    raw_json = dict(job)
    record = {
        "source_type": "greenhouse",
        "source_url": source_url,
        "source_job_id": str(source_job_id),
        "source_company": board_token,
        "company_name": job.get("company_name") or board_token,
        "title": job.get("title"),
        "location_name": location.get("name"),
        "department_names": names_from_objects(job.get("departments")),
        "office_names": names_from_objects(job.get("offices")),
        "posted_at": job.get("first_published"),
        "updated_at": job.get("updated_at"),
        "job_url": job.get("absolute_url"),
        "apply_url": None,
        "content_html": content_html,
        "content_text": strip_html_to_text(content_html),
        "raw_json": raw_json,
        "fetched_at_utc": fetched_at_utc,
    }
    if include_fingerprint:
        record["raw_json"]["listing_fingerprint"] = job_listing_fingerprint(record)
    return record


def collect_jobs_for_source(conn, source_url: str) -> tuple[list[dict[str, Any]], list[str], int, dict[str, Any]]:
    payload = fetch_greenhouse_json(greenhouse_endpoint_from_url(source_url))
    jobs = payload.get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError("Expected Greenhouse payload['jobs'] to be a list.")

    fetched_at_utc = utc_now()
    _t = time.monotonic()
    source_company = board_token_from_url(source_url)
    log_timing("greenhouse", source_company, "source_company", time.monotonic() - _t)
    _t = time.monotonic()
    existing_fingerprints = existing_fingerprints_for_source(conn, source_company)
    log_timing("greenhouse", source_company, "existing_fingerprints_query", time.monotonic() - _t)
    _t = time.monotonic()
    all_records = []
    for job in jobs:
        if isinstance(job, dict):
            all_records.append(normalize_job(job, source_url, fetched_at_utc, include_fingerprint=False))
    log_timing("greenhouse", source_company, "normalize_jobs", time.monotonic() - _t)
    _t = time.monotonic()
    fingerprint_elapsed_s = 0.0
    for record in all_records:
        _fingerprint_t = time.monotonic()
        record["raw_json"]["listing_fingerprint"] = job_listing_fingerprint(record)
        fingerprint_elapsed_s += time.monotonic() - _fingerprint_t
    log_timing("greenhouse", source_company, "make_listing_fingerprint", fingerprint_elapsed_s)
    log_timing("greenhouse", source_company, "listing_fingerprints", time.monotonic() - _t)
    _t = time.monotonic()
    records = []
    for record in all_records:
        if existing_fingerprints.get(record["source_job_id"]) != record["raw_json"]["listing_fingerprint"]:
            records.append(record)
    log_timing("greenhouse", source_company, "skip_unchanged", time.monotonic() - _t)
    skipped_count = len(all_records) - len(records)
    _t = time.monotonic()
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
    log_timing("greenhouse", source_company, "data_quality_summary", time.monotonic() - _t)
    print(
        f"greenhouse: fetched {len(all_records)} jobs from {source_company}; "
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
    sources = load_ats_source_urls_from_db(conn, "greenhouse")
    total_sources = len(sources)
    skipped_404_sources = count_skipped_404_sources_from_db(conn, "greenhouse")
    http_errors = AtsHttpErrorTracker("greenhouse")

    for source_url in sources:
        if should_stop_for_ingestion_budget("greenhouse", started_at, completed_sources):
            break
        attempted_sources += 1
        source_company = board_token_from_url(source_url)
        attempt_http_status_code: int | None = None
        attempt_error_type: str | None = None
        stop_after_source = False
        try:
            _t = time.monotonic()
            records, seen_source_job_ids, skipped_count, quality_summary = collect_jobs_for_source(conn, source_url)
            log_timing("greenhouse", source_company, "fetch", time.monotonic() - _t)

            batch.extend(records)

            _t = time.monotonic()
            marked_missing = mark_missing_jobs_for_source(
                conn,
                source_type="greenhouse",
                source_company=source_company,
                seen_source_job_ids=seen_source_job_ids,
                seen_at_utc=records[0]["fetched_at_utc"] if records else utc_now(),
            )
            log_timing("greenhouse", source_company, "mark_missing", time.monotonic() - _t)

            total_marked_missing += marked_missing
            if marked_missing:
                print(f"greenhouse: marked {marked_missing} jobs missing for {source_company}", flush=True)
            log_data_quality(
                "ingestion",
                source_type="greenhouse",
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
                total_count += flush_batch(conn, batch, "greenhouse")
                log_timing("greenhouse", source_company, "flush", time.monotonic() - _t)
                sources_in_batch = 0
        except Exception as exc:
            error_sources += 1
            http_status_code = http_status_code_from_exception(exc)
            attempt_http_status_code = http_status_code
            attempt_error_type = type(exc).__name__
            stop_after_source = isinstance(exc, AtsHttp429LimitReached) or http_errors.record(http_status_code)
            print(
                "greenhouse: error fetching source: "
                f"board={source_company}, error_type={type(exc).__name__}, "
                f"http_status_code={http_status_code or 'unknown'}",
                flush=True,
            )
            log_data_quality(
                "ingestion",
                source_type="greenhouse",
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
            log_timing("greenhouse", source_company, "last_get_at", time.monotonic() - _t)
        if stop_after_source:
            break

    total_count += flush_batch(conn, batch, "greenhouse")
    checked_percent = (attempted_sources / total_sources * 100) if total_sources else 0.0
    print(
        f"greenhouse: checked {attempted_sources}/{total_sources} sources ({checked_percent:.1f}%); errors={error_sources}",
        flush=True,
    )
    http_errors.print_summary()
    print(f"greenhouse: skipped {skipped_404_sources} sources due to 404 streak", flush=True)
    print(
        f"greenhouse: inserted/updated {total_count} rows total; marked {total_marked_missing} missing",
        flush=True,
    )
    return total_count


if __name__ == "__main__":
    from data_pipeline.common.db import connect

    with connect() as connection:
        run(connection)
