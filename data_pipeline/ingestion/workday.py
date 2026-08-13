from __future__ import annotations

"""
Fetch Workday public job-board JSON endpoints and upsert normalized jobs into
PostgreSQL. No SQLite databases or CSV files are created.
"""

import json
import os
import random
import re
import time
import hashlib
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
from data_pipeline.ingestion.size_limits import (
    MAX_DETAIL_RESPONSE_BYTES,
    MAX_LISTING_RESPONSE_BYTES,
    read_response_with_limit,
)

REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = "Mozilla/5.0 (compatible; latmay-workday/1.0)"
MIN_REQUEST_GAP_SECONDS = float(os.environ.get("WORKDAY_REQUEST_GAP_SECONDS", os.environ.get("REQUEST_GAP_SECONDS", "2")))
RANDOM_JITTER_MAX_SECONDS = 0.35
MAX_503_RETRIES = 2
WORKDAY_MAX_DETAIL_FETCHES_PER_SOURCE = (
    int(os.environ["WORKDAY_MAX_DETAIL_FETCHES_PER_SOURCE"])
    if os.environ.get("WORKDAY_MAX_DETAIL_FETCHES_PER_SOURCE")
    else None
)
WORKDAY_MAX_DETAIL_FETCHES_TOTAL = (
    int(os.environ["WORKDAY_MAX_DETAIL_FETCHES_TOTAL"])
    if os.environ.get("WORKDAY_MAX_DETAIL_FETCHES_TOTAL")
    else None
)
_last_request_time: float | None = None
_detail_fetches_total = 0


def page_size_from_env() -> int:
    try:
        page_size = int(os.environ.get("WORKDAY_PAGE_SIZE", "100"))
    except ValueError:
        return 20
    return page_size if page_size > 0 else 20


PAGE_SIZE = page_size_from_env()


def optional_positive_int_from_env(name: str) -> int | None:
    raw_value = os.environ.get(name)
    if not raw_value:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    return value if value > 0 else None


WORKDAY_MAX_LIST_JOBS_PER_SOURCE = optional_positive_int_from_env("WORKDAY_MAX_LIST_JOBS_PER_SOURCE")


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def stable_fingerprint(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def workday_listing_fingerprint(summary: dict[str, Any]) -> str:
    return stable_fingerprint(
        {
            "externalPath": summary.get("externalPath"),
            "title": summary.get("title"),
            "locationsText": summary.get("locationsText"),
            "postedOn": summary.get("postedOn"),
        }
    )


def raw_json_listing_fingerprint(row: dict[str, Any] | None) -> str | None:
    if not row:
        return None
    raw_json = row.get("raw_json") if isinstance(row.get("raw_json"), dict) else {}
    fingerprint = raw_json.get("listing_fingerprint")
    return fingerprint if isinstance(fingerprint, str) else None


def sleep_before_request() -> None:
    global _last_request_time

    now = time.monotonic()
    if _last_request_time is not None:
        wait = MIN_REQUEST_GAP_SECONDS + random.uniform(0, RANDOM_JITTER_MAX_SECONDS) - (now - _last_request_time)
        if wait > 0:
            time.sleep(wait)

    _last_request_time = time.monotonic()


def workday_request(method: str, url: str, **kwargs: Any) -> requests.Response:
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "user-agent": USER_AGENT,
    }
    headers.update(kwargs.pop("headers", {}))
    headers = {key: value for key, value in headers.items() if value}

    for attempt in range(MAX_503_RETRIES + 1):
        sleep_before_request()
        _t = time.monotonic()
        response = requests.request(
            method,
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
            stream=True,
            **kwargs,
        )
        log_timing("workday", urlparse(url).netloc, "http_get", time.monotonic() - _t)
        if response.status_code != 503 or attempt >= MAX_503_RETRIES:
            response.raise_for_status()
            limit = MAX_DETAIL_RESPONSE_BYTES if method.upper() == "GET" else MAX_LISTING_RESPONSE_BYTES
            read_response_with_limit(response, limit)
            return response

        response.close()
        backoff = (2**attempt) * MIN_REQUEST_GAP_SECONDS + random.uniform(0, RANDOM_JITTER_MAX_SECONDS)
        time.sleep(backoff)

    raise RuntimeError("unreachable Workday request retry state")


def parse_workday_site(site_url: str) -> dict[str, str]:
    parsed = urlparse(site_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid Workday site URL: {site_url}")

    host = parsed.netloc
    tenant = host.split(".", 1)[0]
    path_parts = [part for part in parsed.path.split("/") if part]
    locale = "en-US"
    if path_parts and re.fullmatch(r"[a-z]{2}-[A-Z]{2}", path_parts[0]):
        locale = path_parts[0]
        path_parts = path_parts[1:]
    if not path_parts:
        raise ValueError(f"Workday site URL is missing a site slug: {site_url}")

    site_slug = path_parts[-1]
    return {
        "scheme": parsed.scheme,
        "host": host,
        "tenant": tenant,
        "locale": locale,
        "site_slug": site_slug,
        "source_company": f"{tenant}:{site_slug}",
        "source_url": site_url,
    }


def jobs_endpoint(site_info: dict[str, str]) -> str:
    return (
        f"{site_info['scheme']}://{site_info['host']}"
        f"/wday/cxs/{site_info['tenant']}/{site_info['site_slug']}/jobs"
    )


def build_detail_url(site_info: dict[str, str], external_path: str | None) -> str:
    path = str(external_path or "").strip()
    if not path:
        return f"{site_info['scheme']}://{site_info['host']}/{site_info['locale']}/{site_info['site_slug']}"
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{site_info['scheme']}://{site_info['host']}/{site_info['locale']}/{site_info['site_slug']}{path}"


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


def fetch_workday_jobs_page(site_info: dict[str, str], *, limit: int, offset: int) -> dict[str, Any]:
    response = workday_request(
        "POST",
        jobs_endpoint(site_info),
        json={
            "appliedFacets": {},
            "limit": limit,
            "offset": offset,
            "searchText": "",
        },
    )
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Expected Workday jobs response to be a JSON object.")
    return payload


def collect_workday_summaries(site_info: dict[str, str]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    page_size = PAGE_SIZE
    list_cap_reached = False

    while total is None or offset < total:
        if WORKDAY_MAX_LIST_JOBS_PER_SOURCE is not None and len(summaries) >= WORKDAY_MAX_LIST_JOBS_PER_SOURCE:
            list_cap_reached = True
            break

        try:
            payload = fetch_workday_jobs_page(site_info, limit=page_size, offset=offset)
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if offset == 0 and page_size != 20 and status_code in {400, 413, 422}:
                print(
                    f"workday: retrying {site_info['source_company']} with page_size=20",
                    flush=True,
                )
                page_size = 20
                continue
            raise
        if total is None and isinstance(payload.get("total"), int):
            total = int(payload["total"])

        jobs = payload.get("jobPostings", [])
        if not isinstance(jobs, list) or not jobs:
            break

        valid_jobs = [job for job in jobs if isinstance(job, dict)]
        if WORKDAY_MAX_LIST_JOBS_PER_SOURCE is not None:
            remaining = WORKDAY_MAX_LIST_JOBS_PER_SOURCE - len(summaries)
            if remaining <= 0:
                list_cap_reached = True
                break
            if len(valid_jobs) > remaining:
                valid_jobs = valid_jobs[:remaining]
                list_cap_reached = True
        summaries.extend(valid_jobs)
        offset += len(jobs)
        if list_cap_reached or (total is not None and offset >= total):
            break

    print(
        f"workday: fetched {len(summaries)} summaries from {site_info['source_company']}; "
        f"list_cap_reached={str(list_cap_reached).lower()}",
        flush=True,
    )
    return summaries


def fetch_detail_html(detail_url: str) -> str:
    global _detail_fetches_total

    _detail_fetches_total += 1
    response = workday_request(
        "GET",
        detail_url,
        headers={
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "content-type": None,
        },
    )
    return response.text


def _is_jobposting_type(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() == "jobposting"
    if isinstance(value, list):
        return any(_is_jobposting_type(item) for item in value)
    return False


def _find_jobposting_json_ld(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if _is_jobposting_type(value.get("@type")):
            return value
        graph = value.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                found = _find_jobposting_json_ld(item)
                if found is not None:
                    return found
    if isinstance(value, list):
        for item in value:
            found = _find_jobposting_json_ld(item)
            if found is not None:
                return found
    return None


def extract_json_ld_jobposting(html: str) -> dict[str, Any]:
    for match in re.finditer(
        r'(?is)<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html or "",
    ):
        raw_json = unescape(match.group(1)).strip()
        if not raw_json:
            continue
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        jobposting = _find_jobposting_json_ld(payload)
        if jobposting is not None:
            return jobposting
    return {}


def source_job_id_from_json_ld(json_ld: dict[str, Any]) -> str | None:
    identifier = json_ld.get("identifier")
    if isinstance(identifier, dict):
        value = identifier.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(identifier, str) and identifier.strip():
        return identifier.strip()
    return None


def company_name_from_json_ld(json_ld: dict[str, Any]) -> str | None:
    hiring_org = json_ld.get("hiringOrganization")
    if isinstance(hiring_org, dict):
        name = hiring_org.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def location_from_json_ld(json_ld: dict[str, Any]) -> str | None:
    locations = json_ld.get("jobLocation")
    if isinstance(locations, dict):
        locations = [locations]
    if not isinstance(locations, list):
        return None

    names: list[str] = []
    for location in locations:
        if not isinstance(location, dict):
            continue
        address = location.get("address")
        if isinstance(address, dict):
            parts = [
                address.get("addressLocality"),
                address.get("addressRegion"),
                address.get("addressCountry"),
            ]
            name = ", ".join(str(part).strip() for part in parts if str(part or "").strip())
        else:
            name = str(location.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return " | ".join(names) if names else None


def existing_rows_for_source(conn, source_company: str) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                source_job_id,
                source_url,
                source_company,
                company_name,
                title,
                location_name,
                department_names,
                office_names,
                posted_at,
                updated_at,
                job_url,
                apply_url,
                content_html,
                content_text,
                raw_json
            FROM jobs
            WHERE source_type = %s
              AND source_company = %s
            """,
            ("workday", source_company),
        )
        rows = cur.fetchall()

    for row in rows:
        source_job_id = str(row.get("source_job_id") or "").strip()
        job_url = str(row.get("job_url") or "").strip()
        if source_job_id:
            lookup[f"id:{source_job_id}"] = row
        if job_url:
            lookup[f"url:{job_url}"] = row
    return lookup


def existing_row_for_summary(
    summary: dict[str, Any],
    detail_url: str,
    existing_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    external_path = str(summary.get("externalPath") or "").strip()
    for key in (f"url:{detail_url}", f"id:{external_path}"):
        row = existing_lookup.get(key)
        if row is not None:
            return row
    return None


def source_job_id_for_seen(summary: dict[str, Any], detail_url: str, existing_row: dict[str, Any] | None = None) -> str:
    existing_source_job_id = str((existing_row or {}).get("source_job_id") or "").strip()
    if existing_source_job_id:
        return existing_source_job_id
    external_path = str(summary.get("externalPath") or "").strip()
    return external_path or detail_url


def can_reuse_existing_content(
    existing_row: dict[str, Any] | None,
    summary: dict[str, Any],
    detail_url: str,
) -> bool:
    if not existing_row or not str(existing_row.get("content_text") or "").strip():
        return False
    title = str(summary.get("title") or "").strip()
    location = str(summary.get("locationsText") or "").strip()
    fingerprint = workday_listing_fingerprint(summary)
    return (
        raw_json_listing_fingerprint(existing_row) == fingerprint
        and str(existing_row.get("job_url") or "").strip() == detail_url
        and (not title or str(existing_row.get("title") or "").strip() == title)
        and (not location or str(existing_row.get("location_name") or "").strip() == location)
    )


def normalize_workday_job(
    summary: dict[str, Any],
    site_info: dict[str, str],
    fetched_at_utc: datetime,
    *,
    detail_url: str,
    detail_html: str | None = None,
    json_ld: dict[str, Any] | None = None,
    existing_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    json_ld = json_ld or {}
    listing_fingerprint = workday_listing_fingerprint(summary)
    description = json_ld.get("description") if isinstance(json_ld.get("description"), str) else None
    content_html = description or detail_html or (existing_row or {}).get("content_html")
    content_text = strip_html_to_text(description) or (
        (existing_row or {}).get("content_text") if detail_html is None else strip_html_to_text(detail_html)
    )
    source_job_id = (
        (existing_row or {}).get("source_job_id")
        or source_job_id_from_json_ld(json_ld)
        or summary.get("externalPath")
        or detail_url
    )
    location_name = summary.get("locationsText") or location_from_json_ld(json_ld) or (existing_row or {}).get("location_name")
    company_name = company_name_from_json_ld(json_ld) or (existing_row or {}).get("company_name") or site_info["site_slug"]

    return {
        "source_type": "workday",
        "source_url": site_info["source_url"],
        "source_job_id": str(source_job_id),
        "source_company": site_info["source_company"],
        "company_name": company_name,
        "title": json_ld.get("title") or summary.get("title") or (existing_row or {}).get("title"),
        "location_name": location_name,
        "department_names": None,
        "office_names": None,
        "posted_at": json_ld.get("datePosted") or summary.get("postedOn") or (existing_row or {}).get("posted_at"),
        "updated_at": None,
        "job_url": detail_url,
        "apply_url": None,
        "content_html": content_html,
        "content_text": content_text,
        "raw_json": {
            "listing_fingerprint": listing_fingerprint,
            "summary": summary,
            "json_ld": json_ld,
            "existing_source_job_id": (existing_row or {}).get("source_job_id"),
            "employment_type": json_ld.get("employmentType"),
        },
        "fetched_at_utc": fetched_at_utc,
    }


def can_fetch_detail(source_detail_fetch_count: int) -> bool:
    if WORKDAY_MAX_DETAIL_FETCHES_PER_SOURCE is not None and source_detail_fetch_count >= WORKDAY_MAX_DETAIL_FETCHES_PER_SOURCE:
        return False
    if WORKDAY_MAX_DETAIL_FETCHES_TOTAL is not None and _detail_fetches_total >= WORKDAY_MAX_DETAIL_FETCHES_TOTAL:
        return False
    return True


def collect_jobs_for_source_result(
    conn,
    site_url: str,
    http_errors: AtsHttpErrorTracker | None = None,
) -> dict[str, Any]:
    site_info = parse_workday_site(site_url)
    _t = time.monotonic()
    summaries = collect_workday_summaries(site_info)
    log_timing("workday", site_info["source_company"], "http_list", time.monotonic() - _t)
    existing_lookup = existing_rows_for_source(conn, site_info["source_company"])
    fetched_at_utc = utc_now()
    records: list[dict[str, Any]] = []
    seen_source_job_ids: list[str] = []
    unchanged_skipped_count = 0
    detail_fetch_count = 0
    cap_reached_count = 0

    for summary in summaries:
        external_path = summary.get("externalPath")
        detail_url = build_detail_url(site_info, str(external_path or ""))
        existing_row = existing_row_for_summary(summary, detail_url, existing_lookup)

        if can_reuse_existing_content(existing_row, summary, detail_url):
            seen_source_job_ids.append(source_job_id_for_seen(summary, detail_url, existing_row))
            unchanged_skipped_count += 1
            continue

        if not can_fetch_detail(detail_fetch_count):
            cap_reached_count += 1
            if existing_row and str(existing_row.get("content_text") or "").strip():
                record = normalize_workday_job(
                    summary,
                    site_info,
                    fetched_at_utc,
                    detail_url=detail_url,
                    existing_row=existing_row,
                )
                records.append(record)
                seen_source_job_ids.append(record["source_job_id"])
            else:
                seen_source_job_ids.append(source_job_id_for_seen(summary, detail_url, existing_row))
            continue

        try:
            detail_fetch_count += 1
            detail_html = fetch_detail_html(detail_url)
            json_ld = extract_json_ld_jobposting(detail_html)
        except Exception as exc:
            http_status_code = http_status_code_from_exception(exc)
            print(
                "workday: error fetching job detail: "
                f"source={site_info['source_company']}, error_type={type(exc).__name__}, "
                f"http_status_code={http_status_code or 'unknown'}",
                flush=True,
            )
            if http_errors is not None:
                http_errors.record_or_raise(http_status_code)
            detail_html = None
            json_ld = {}

        record = normalize_workday_job(
            summary,
            site_info,
            fetched_at_utc,
            detail_url=detail_url,
            detail_html=detail_html,
            json_ld=json_ld,
            existing_row=existing_row,
        )
        records.append(record)
        seen_source_job_ids.append(record["source_job_id"])

    print(
        f"workday: normalized {len(records)} jobs from {site_info['source_company']}; "
        f"unchanged_skipped={unchanged_skipped_count}, detail_pages_fetched={detail_fetch_count}, "
        f"cap_reached={cap_reached_count}, to_upsert={len(records)}",
        flush=True,
    )
    return {
        "records": records,
        "seen_source_job_ids": seen_source_job_ids,
        "unchanged_skipped_count": unchanged_skipped_count,
        "detail_fetch_count": detail_fetch_count,
        "cap_reached_count": cap_reached_count,
        "quality_summary": {
            "fetched": len(summaries),
            "unchanged_skipped": unchanged_skipped_count,
            "to_upsert": len(records),
            "active_seen": len(seen_source_job_ids),
            "missing_title": count_blank(records, "title"),
            "missing_url": count_blank(records, "job_url"),
            "missing_location": count_blank(records, "location_name"),
            "missing_content_text": count_blank(records, "content_text"),
            "duplicate_source_job_ids": duplicate_count(seen_source_job_ids),
            "detail_pages_fetched": detail_fetch_count,
            "detail_fetch_cap_reached": cap_reached_count,
        },
    }


def collect_jobs_for_source(conn, site_url: str) -> list[dict[str, Any]]:
    result = collect_jobs_for_source_result(conn, site_url)
    records = result["records"]
    if not isinstance(records, list):
        raise ValueError("Expected Workday collection result records to be a list.")
    return records


def flush_batch(conn, records: list[dict[str, Any]], source_type: str) -> int:
    if not records:
        return 0

    count = upsert_jobs(conn, records)
    print(f"{source_type}: inserted/updated {count} rows in batch", flush=True)
    records.clear()
    return count


def run(conn) -> int:
    total_count = 0
    total_marked_missing = 0
    completed_sources = 0
    attempted_sources = 0
    error_sources = 0
    started_at = ingestion_budget_started_at()
    sites = load_ats_source_urls_from_db(conn, "workday")
    total_sources = len(sites)
    skipped_404_sources = count_skipped_404_sources_from_db(conn, "workday")
    http_errors = AtsHttpErrorTracker("workday")

    for site_url in sites:
        if should_stop_for_ingestion_budget("workday", started_at, completed_sources):
            break
        attempted_sources += 1
        _source_company = "unknown"
        attempt_http_status_code: int | None = None
        attempt_error_type: str | None = None
        stop_after_source = False
        try:
            site_info = parse_workday_site(site_url)
            _source_company = site_info["source_company"]

            _t = time.monotonic()
            result = collect_jobs_for_source_result(conn, site_url, http_errors=http_errors)
            log_timing("workday", _source_company, "fetch", time.monotonic() - _t)

            records = result["records"]
            seen_source_job_ids = result["seen_source_job_ids"]
            seen_at_utc = records[0]["fetched_at_utc"] if records else utc_now()

            _t = time.monotonic()
            inserted_updated = flush_batch(conn, records, "workday")
            log_timing("workday", _source_company, "flush", time.monotonic() - _t)
            total_count += inserted_updated

            _t = time.monotonic()
            marked_missing = mark_missing_jobs_for_source(
                conn,
                source_type="workday",
                source_company=site_info["source_company"],
                seen_source_job_ids=seen_source_job_ids,
                seen_at_utc=seen_at_utc,
            )
            log_timing("workday", _source_company, "mark_missing", time.monotonic() - _t)

            total_marked_missing += marked_missing
            if marked_missing:
                print(
                    f"workday: marked {marked_missing} jobs missing for {site_info['source_company']}",
                    flush=True,
                )
            log_data_quality(
                "ingestion",
                source_type="workday",
                company=site_info["source_company"],
                inserted_updated=inserted_updated,
                marked_missing=marked_missing,
                request_failures=0,
                **result["quality_summary"],
            )
            completed_sources += 1
        except Exception as exc:
            error_sources += 1
            http_status_code = http_status_code_from_exception(exc)
            attempt_http_status_code = http_status_code
            attempt_error_type = type(exc).__name__
            stop_after_source = isinstance(exc, AtsHttp429LimitReached) or http_errors.record(http_status_code)
            if _source_company == "unknown":
                try:
                    _source_company = parse_workday_site(site_url)["source_company"]
                except Exception:
                    pass
            print(
                "workday: error fetching source: "
                f"source={_source_company}, error_type={type(exc).__name__}, "
                f"http_status_code={http_status_code or 'unknown'}",
                flush=True,
            )
            log_data_quality(
                "ingestion",
                source_type="workday",
                company=_source_company,
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
                site_url,
                http_status_code=attempt_http_status_code,
                error_type=attempt_error_type,
            )
            log_timing("workday", _source_company, "last_get_at", time.monotonic() - _t)
        if stop_after_source:
            break

    checked_percent = (attempted_sources / total_sources * 100) if total_sources else 0.0
    print(
        f"workday: checked {attempted_sources}/{total_sources} sources ({checked_percent:.1f}%); errors={error_sources}",
        flush=True,
    )
    http_errors.print_summary()
    print(f"workday: skipped {skipped_404_sources} sources due to 404 streak", flush=True)
    print(
        f"workday: inserted/updated {total_count} rows total; marked {total_marked_missing} missing",
        flush=True,
    )
    return total_count


if __name__ == "__main__":
    from data_pipeline.common.db import connect

    with connect() as connection:
        run(connection)
