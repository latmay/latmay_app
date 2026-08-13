from __future__ import annotations

"""
Fetch Workable careers listing pages and upsert normalized jobs into PostgreSQL.

Workable embeds all job data as JSON in window.jobBoard.initialState on the
company listing page — a single fetch per company yields all jobs without
per-job HTTP requests.

Supported URL patterns:
  https://jobs.workable.com/company/{id}/jobs-at-{company}
  https://apply.workable.com/{company}/
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from data_pipeline.common.data_quality import count_blank, duplicate_count, http_status_code_from_exception, log_data_quality
from data_pipeline.common.db import mark_missing_jobs_for_source, upsert_jobs
from data_pipeline.common.timing import log_timing
from data_pipeline.ingestion.budget import ingestion_budget_started_at, should_stop_for_ingestion_budget
from data_pipeline.ingestion.http_error_tracker import AtsHttp429LimitReached, AtsHttpErrorTracker
from data_pipeline.ingestion.source_loader import (
    count_skipped_404_sources_from_db,
    load_ats_source_urls_from_db,
    rollback_failed_source_attempt,
    update_source_last_get_at,
)
from data_pipeline.ingestion.size_limits import (
    MAX_DETAIL_RESPONSE_BYTES,
    MAX_LISTING_RESPONSE_BYTES,
    ResponseTooLarge,
    read_response_with_limit,
)

REQUEST_TIMEOUT_SECONDS = int(os.environ.get("WORKABLE_REQUEST_TIMEOUT_SECONDS", "30"))
USER_AGENT = os.environ.get(
    "WORKABLE_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
)
MIN_REQUEST_GAP_SECONDS = float(os.environ.get("WORKABLE_REQUEST_GAP_SECONDS", os.environ.get("REQUEST_GAP_SECONDS", "2.5")))
RANDOM_JITTER_MAX_SECONDS = float(os.environ.get("WORKABLE_REQUEST_JITTER_SECONDS", "0.35"))
MAX_RETRIES = int(os.environ.get("WORKABLE_MAX_RETRIES", "3"))
BATCH_SOURCE_COUNT = 5

_last_request_by_host: dict[str, float] = {}


class RetryAfterSourceFailure(requests.HTTPError):
    """Stop processing this source when Workable asks us to retry much later."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def workable_source_company(source_url: str) -> str:
    """Derive a stable per-company identifier from the listing URL.

    jobs.workable.com/company/{id}/jobs-at-{slug}  →  {slug}
    apply.workable.com/{slug}/                      →  {slug}
    """
    parsed = urlparse(source_url)
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]
    for part in reversed(path_parts):
        if part.startswith("jobs-at-"):
            return part[len("jobs-at-"):]
    if path_parts:
        first = path_parts[0]
        if first not in ("company",):
            return first
    return parsed.netloc.lower()


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
    content_hash = hashlib.sha256(content_text.encode("utf-8")).hexdigest() if isinstance(content_text, str) else None
    return stable_fingerprint({
        "source_job_id": record.get("source_job_id"),
        "title": record.get("title"),
        "location_name": record.get("location_name"),
        "posted_at": record.get("posted_at"),
        "updated_at": record.get("updated_at"),
        "job_url": record.get("job_url"),
        "content_text_hash": content_hash,
    })


def sleep_before_host_request(host: str) -> None:
    now = time.monotonic()
    last = _last_request_by_host.get(host)
    if last is not None:
        wait = MIN_REQUEST_GAP_SECONDS + random.uniform(0, RANDOM_JITTER_MAX_SECONDS) - (now - last)
        if wait > 0:
            time.sleep(wait)
    _last_request_by_host[host] = time.monotonic()


def print_retry_sleep(
    *,
    method: str,
    company: str,
    url: str,
    attempt: int,
    delay: float,
    status_code: int | None = None,
    error_type: str | None = None,
) -> None:
    reason = f"status={status_code}" if status_code is not None else f"error_type={error_type or 'unknown'}"
    print(
        "workable: retrying request after sleep: "
        f"method={method}, company={company}, attempt={attempt + 1}/{MAX_RETRIES + 1}, "
        f"delay_s={delay:.1f}, {reason}, url={url}",
        flush=True,
    )


def raise_retry_after_source_failure(
    *,
    response: requests.Response,
    method: str,
    company: str,
    url: str,
) -> None:
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return
    print(
        "workable: failing source due to Retry-After response: "
        f"method={method}, company={company}, status={response.status_code}, "
        f"retry_after={retry_after}, url={url}",
        flush=True,
    )
    exc = RetryAfterSourceFailure(
        f"Workable returned HTTP {response.status_code} with Retry-After={retry_after}; failing source."
    )
    exc.response = response
    raise exc


def polite_get(
    session: requests.Session,
    url: str,
    *,
    timing_company: str | None = None,
    max_response_bytes: int = MAX_LISTING_RESPONSE_BYTES,
    **kwargs: Any,
) -> requests.Response:
    sleep_before_host_request(urlparse(url).netloc.lower())
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        **(kwargs.pop("headers", {}) or {}),
    }
    last_exc: Exception | None = None
    company = timing_company or workable_source_company(url)
    for attempt in range(MAX_RETRIES + 1):
        try:
            _t = time.monotonic()
            response = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS, stream=True, **kwargs)
            log_timing("workable", company, "http_get", time.monotonic() - _t)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                raise_retry_after_source_failure(response=response, method="GET", company=company, url=url)
                if attempt < MAX_RETRIES:
                    response.close()
                    delay = min(2 ** attempt, 8)
                    print_retry_sleep(
                        method="GET",
                        company=company,
                        url=url,
                        attempt=attempt,
                        delay=delay,
                        status_code=response.status_code,
                    )
                    time.sleep(delay)
                    continue
            response.raise_for_status()
            read_response_with_limit(response, max_response_bytes)
            return response
        except RetryAfterSourceFailure:
            raise
        except ResponseTooLarge:
            raise
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= MAX_RETRIES:
                break
            delay = min(2 ** attempt, 8)
            print_retry_sleep(
                method="GET",
                company=company,
                url=url,
                attempt=attempt,
                delay=delay,
                error_type=type(exc).__name__,
            )
            time.sleep(delay)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"request_failed:{url}")


def polite_post_json(session: requests.Session, url: str, *, json_body: dict[str, Any], referer: str) -> requests.Response:
    sleep_before_host_request(urlparse(url).netloc.lower())
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Content-Type": "application/json",
        "Origin": "https://apply.workable.com",
        "Referer": referer,
    }
    last_exc: Exception | None = None
    company = workable_source_company(referer)
    for attempt in range(MAX_RETRIES + 1):
        try:
            _t = time.monotonic()
            response = session.post(
                url,
                headers=headers,
                json=json_body,
                timeout=REQUEST_TIMEOUT_SECONDS,
                stream=True,
            )
            log_timing("workable", company, "http_post", time.monotonic() - _t)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                raise_retry_after_source_failure(response=response, method="POST", company=company, url=url)
                if attempt < MAX_RETRIES:
                    response.close()
                    delay = min(2 ** attempt, 8)
                    print_retry_sleep(
                        method="POST",
                        company=company,
                        url=url,
                        attempt=attempt,
                        delay=delay,
                        status_code=response.status_code,
                    )
                    time.sleep(delay)
                    continue
            response.raise_for_status()
            read_response_with_limit(response, MAX_LISTING_RESPONSE_BYTES)
            return response
        except RetryAfterSourceFailure:
            raise
        except ResponseTooLarge:
            raise
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= MAX_RETRIES:
                break
            delay = min(2 ** attempt, 8)
            print_retry_sleep(
                method="POST",
                company=company,
                url=url,
                attempt=attempt,
                delay=delay,
                error_type=type(exc).__name__,
            )
            time.sleep(delay)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"request_failed:{url}")


def extract_initial_state(html: str) -> dict[str, Any]:
    """Pull window.jobBoard.initialState out of page HTML via brace-balanced extraction."""
    match = re.search(r"initialState:\s*(\{)", html)
    if not match:
        raise ValueError("window.jobBoard.initialState not found in page HTML.")
    start = match.start(1)
    depth = 0
    for i, ch in enumerate(html[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(html[start: i + 1])
    raise ValueError("Could not find closing brace for initialState JSON.")


def workable_apply_account_slug(source_url: str) -> str | None:
    parsed = urlparse(source_url)
    if parsed.netloc.lower() != "apply.workable.com":
        return None
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]
    if not path_parts or path_parts[0] == "api":
        return None
    return path_parts[0]


def workable_public_job_url(account_slug: str, shortcode: object) -> str | None:
    shortcode_text = str(shortcode or "").strip()
    if not shortcode_text:
        return None
    return f"https://apply.workable.com/{account_slug}/j/{shortcode_text}/"


def fetch_workable_api_jobs(
    source_url: str,
    session: requests.Session,
    http_errors: AtsHttpErrorTracker | None = None,
) -> tuple[list[dict[str, Any]], str]:
    account_slug = workable_apply_account_slug(source_url)
    if not account_slug:
        raise ValueError(f"Workable API fallback requires apply.workable.com account URL: {source_url}")

    referer = f"https://apply.workable.com/{account_slug}/"
    list_url = f"https://apply.workable.com/api/v3/accounts/{account_slug}/jobs"
    list_response = polite_post_json(
        session,
        list_url,
        referer=referer,
        json_body={"query": "", "department": [], "location": [], "workplace": [], "worktype": []},
    )
    payload = list_response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected Workable v3 jobs response to be an object for {source_url}")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Expected Workable v3 jobs response['results'] to be a list for {source_url}")

    jobs: list[dict[str, Any]] = []
    detail_failures = 0
    for job in results:
        if not isinstance(job, dict) or job.get("state") != "published":
            continue
        shortcode = str(job.get("shortcode") or "").strip()
        detail_job = dict(job)
        if shortcode:
            detail_url = f"https://apply.workable.com/api/v2/accounts/{account_slug}/jobs/{shortcode}"
            try:
                detail_response = polite_get(
                    session,
                    detail_url,
                    timing_company=account_slug,
                    max_response_bytes=MAX_DETAIL_RESPONSE_BYTES,
                    headers={"Accept": "application/json, text/plain, */*", "Referer": referer},
                )
                detail_payload = detail_response.json()
                if isinstance(detail_payload, dict):
                    detail_job.update(detail_payload)
                else:
                    detail_failures += 1
            except Exception as exc:
                detail_failures += 1
                http_status_code = http_status_code_from_exception(exc)
                print(
                    "workable: error fetching job detail: "
                    f"company={account_slug}, shortcode={shortcode}, error_type={type(exc).__name__}, "
                    f"http_status_code={http_status_code or 'unknown'}",
                    flush=True,
                )
                if http_errors is not None:
                    http_errors.record_or_raise(http_status_code)
        detail_job["url"] = detail_job.get("url") or workable_public_job_url(account_slug, detail_job.get("shortcode"))
        detail_job["raw_list_job"] = job
        detail_job["workable_api_version"] = "v3_list_v2_detail"
        jobs.append(detail_job)

    company_title = account_slug
    if detail_failures:
        print(f"workable: {detail_failures} detail fetch failures for {account_slug}", flush=True)
    return jobs, company_title


def fetch_workable_page(
    source_url: str,
    session: requests.Session,
    http_errors: AtsHttpErrorTracker | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Fetch listing page; return (published_jobs, company_title)."""
    response = polite_get(session, source_url)
    try:
        initial_state = extract_initial_state(response.text)
    except ValueError as exc:
        if "initialState not found" not in str(exc):
            raise
        return fetch_workable_api_jobs(source_url, session, http_errors=http_errors)

    company_key = next((k for k in initial_state if k.startswith("api/v1/companies/")), None)
    if not company_key:
        raise ValueError(f"No api/v1/companies/ key in initialState for {source_url}")

    entry = initial_state[company_key]
    if entry.get("status") != 200:
        raise ValueError(f"Workable company API status {entry.get('status')} for {source_url}")

    data = entry.get("data") or {}
    company_title: str = data.get("title") or ""
    jobs = [j for j in (data.get("jobs") or []) if isinstance(j, dict) and j.get("state") == "published"]
    return jobs, company_title


def join_location(job: dict[str, Any]) -> str | None:
    loc = job.get("location") or {}
    parts = [loc.get("city"), loc.get("subregion") or loc.get("region"), loc.get("countryName") or loc.get("country")]
    text = ", ".join(str(p).strip() for p in parts if p and str(p).strip())
    return text or None


def normalize_job(job: dict[str, Any], source_url: str, company_title: str, fetched_at_utc: datetime) -> dict[str, Any]:
    source_company = workable_source_company(source_url)
    source_job_id = job.get("id") or job.get("url") or f"{source_url}:{job.get('title')}"

    description_html = job.get("description") or ""
    requirements_html = job.get("requirementsSection") or job.get("requirements") or ""
    benefits_html = job.get("benefitsSection") or job.get("benefits") or ""
    content_html = "\n\n".join(s for s in (description_html, requirements_html, benefits_html) if s) or None

    record: dict[str, Any] = {
        "source_type": "workable",
        "source_url": source_url,
        "source_job_id": str(source_job_id),
        "source_company": source_company,
        "company_name": company_title or source_company,
        "title": job.get("title"),
        "location_name": join_location(job),
        "department_names": job.get("department"),
        "posted_at": job.get("created") or job.get("published"),
        "updated_at": job.get("updated"),
        "job_url": job.get("url"),
        "apply_url": job.get("url"),
        "content_html": content_html,
        "content_text": strip_html_to_text(content_html),
        "raw_json": {
            "id": job.get("id"),
            "shortcode": job.get("shortcode"),
            "department": job.get("department"),
            "employment_type": job.get("employmentType") or job.get("type"),
            "workplace": job.get("workplace"),
            "locations": job.get("locations"),
            "location": job.get("location"),
            "language": job.get("language"),
            "is_featured": job.get("isFeatured"),
            "social_description": job.get("socialSharingDescription"),
            "workable_api_version": job.get("workable_api_version"),
            "raw_list_job": job.get("raw_list_job"),
        },
        "fetched_at_utc": fetched_at_utc,
    }
    record["raw_json"]["listing_fingerprint"] = job_listing_fingerprint(record)
    return record


def existing_fingerprints_for_source(conn, source_company: str) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT source_job_id, raw_json->>'listing_fingerprint' AS listing_fingerprint
            FROM jobs
            WHERE source_type = %s
              AND source_company = %s
            """,
            ("workable", source_company),
        )
        rows = cur.fetchall()
    fingerprints: dict[str, str] = {}
    for row in rows:
        fingerprint = row.get("listing_fingerprint")
        source_job_id = str(row.get("source_job_id") or "").strip()
        if source_job_id and isinstance(fingerprint, str):
            fingerprints[source_job_id] = fingerprint
    return fingerprints


def collect_jobs_for_source(
    conn,
    source_url: str,
    http_errors: AtsHttpErrorTracker | None = None,
) -> tuple[list[dict[str, Any]], list[str], int, dict[str, Any]]:
    with requests.Session() as session:
        jobs, company_title = fetch_workable_page(source_url, session, http_errors=http_errors)
    fetched_at_utc = utc_now()
    source_company = workable_source_company(source_url)
    existing_fingerprints = existing_fingerprints_for_source(conn, source_company)
    all_records = [normalize_job(job, source_url, company_title, fetched_at_utc) for job in jobs]
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
        "request_failures": 0,
    }
    print(
        f"workable: fetched {len(all_records)} jobs from {source_company}; "
        f"unchanged_skipped={skipped_count}, to_upsert={len(records)}",
        flush=True,
    )
    return records, [record["source_job_id"] for record in all_records], skipped_count, summary


def flush_batch(conn, records: list[dict[str, Any]]) -> int:
    if not records:
        return 0
    count = upsert_jobs(conn, records)
    print(f"workable: inserted/updated {count} rows in batch", flush=True)
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
    sources = load_ats_source_urls_from_db(conn, "workable")
    total_sources = len(sources)
    skipped_404_sources = count_skipped_404_sources_from_db(conn, "workable")
    http_errors = AtsHttpErrorTracker("workable")

    for source_url in sources:
        if should_stop_for_ingestion_budget("workable", started_at, completed_sources):
            break
        attempted_sources += 1
        source_company = workable_source_company(source_url)
        attempt_http_status_code: int | None = None
        attempt_error_type: str | None = None
        stop_after_source = False
        try:
            _t = time.monotonic()
            records, seen_source_job_ids, skipped_count, quality_summary = collect_jobs_for_source(
                conn,
                source_url,
                http_errors=http_errors,
            )
            log_timing("workable", source_company, "fetch", time.monotonic() - _t)

            batch.extend(records)

            _t = time.monotonic()
            marked_missing = mark_missing_jobs_for_source(
                conn,
                source_type="workable",
                source_company=source_company,
                seen_source_job_ids=seen_source_job_ids,
                seen_at_utc=records[0]["fetched_at_utc"] if records else utc_now(),
            )
            log_timing("workable", source_company, "mark_missing", time.monotonic() - _t)

            total_marked_missing += marked_missing
            if marked_missing:
                print(f"workable: marked {marked_missing} jobs missing for {source_company}", flush=True)
            log_data_quality(
                "ingestion",
                source_type="workable",
                company=source_company,
                inserted_updated="unknown",
                marked_missing=marked_missing,
                **quality_summary,
            )
            sources_in_batch += 1
            completed_sources += 1
            if sources_in_batch >= BATCH_SOURCE_COUNT:
                _t = time.monotonic()
                total_count += flush_batch(conn, batch)
                log_timing("workable", source_company, "flush", time.monotonic() - _t)
                sources_in_batch = 0
        except Exception as exc:
            error_sources += 1
            http_status_code = http_status_code_from_exception(exc)
            attempt_http_status_code = http_status_code
            attempt_error_type = type(exc).__name__
            stop_after_source = isinstance(exc, AtsHttp429LimitReached) or http_errors.record(http_status_code)
            print(
                "workable: error fetching source: "
                f"company={source_company}, error_type={type(exc).__name__}, "
                f"http_status_code={http_status_code or 'unknown'}: {exc}",
                flush=True,
            )
            log_data_quality(
                "ingestion",
                source_type="workable",
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
            log_timing("workable", source_company, "last_get_at", time.monotonic() - _t)
        if stop_after_source:
            break

    total_count += flush_batch(conn, batch)
    checked_percent = (attempted_sources / total_sources * 100) if total_sources else 0.0
    print(
        f"workable: checked {attempted_sources}/{total_sources} sources ({checked_percent:.1f}%); errors={error_sources}",
        flush=True,
    )
    http_errors.print_summary()
    print(f"workable: skipped {skipped_404_sources} sources due to 404 streak", flush=True)
    print(f"workable: inserted/updated {total_count} rows total; marked {total_marked_missing} missing", flush=True)
    return total_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape Workable careers listing pages to normalized JSONL.",
        epilog=(
            "Examples:\n"
            "  python -m data_pipeline.ingestion.workable https://jobs.workable.com/company/ABC/jobs-at-myco --verbose --pretty --limit 3\n"
            "  python -m data_pipeline.ingestion.workable <url> --raw-state   # dump initialState JSON for inspection"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("urls", nargs="+", help="Workable listing page URL(s)")
    parser.add_argument("--out", help="Write output to file instead of stdout.")
    parser.add_argument("--include-raw", action="store_true", help="Include raw_json field in each output record.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print fetch details to stderr.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON (one formatted block per job).")
    parser.add_argument("--raw-state", action="store_true", help="Dump the raw initialState JSON instead of normalized jobs.")
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="Only output first N jobs.")
    return parser


def cli_main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.raw_state:
        for source_url in args.urls:
            if args.verbose:
                print(f"[debug] fetching : {source_url}", file=sys.stderr)
            with requests.Session() as session:
                response = polite_get(session, source_url)
            initial_state = extract_initial_state(response.text)
            if args.verbose:
                company_key = next((k for k in initial_state if k.startswith("api/v1/companies/")), None)
                if company_key:
                    data = initial_state[company_key].get("data") or {}
                    jobs = data.get("jobs") or []
                    print(f"[debug] company  : {data.get('title')}", file=sys.stderr)
                    print(f"[debug] jobs     : {len(jobs)}", file=sys.stderr)
            text = json.dumps(initial_state, default=str, ensure_ascii=False, indent=2 if args.pretty else None)
            if args.out:
                Path(args.out).write_text(text + "\n", encoding="utf-8")
            else:
                print(text)
        return 0

    fetched_at_utc = utc_now()
    records: list[dict[str, Any]] = []
    for source_url in args.urls:
        if args.verbose:
            print(f"[debug] fetching : {source_url}", file=sys.stderr)
        with requests.Session() as session:
            jobs, company_title = fetch_workable_page(source_url, session)
        if args.verbose:
            print(f"[debug] company  : {company_title}", file=sys.stderr)
            print(f"[debug] jobs     : {len(jobs)} published", file=sys.stderr)
            workplaces = {}
            for j in jobs:
                wp = j.get("workplace") or "unknown"
                workplaces[wp] = workplaces.get(wp, 0) + 1
            print(f"[debug] workplace breakdown: {workplaces}", file=sys.stderr)
        if args.limit is not None:
            jobs = jobs[: args.limit]
        for job in jobs:
            record = normalize_job(job, source_url, company_title, fetched_at_utc)
            if not args.include_raw:
                record = {k: v for k, v in record.items() if k != "raw_json"}
            records.append(record)

    if args.pretty:
        lines = [json.dumps(r, default=str, ensure_ascii=False, indent=2) for r in records]
        output = "\n\n".join(lines) + ("\n" if lines else "")
    else:
        lines = [json.dumps(r, default=str, ensure_ascii=False) for r in records]
        output = "\n".join(lines) + ("\n" if lines else "")

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
