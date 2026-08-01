from __future__ import annotations

"""
Fetch modern iCIMS/Jibe/CCC careers JSON APIs and upsert normalized jobs into
PostgreSQL.

Expected source URLs are public listing pages such as:
https://careers.fm.com/careers-home/jobs or https://jobs.tufts.edu/jobs.
The scraper derives https://{host}/api/jobs from the listing URL.
Classic server-rendered iCIMS portals on *.icims.com/jobs/search are not
supported by this module.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

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

REQUEST_TIMEOUT_SECONDS = int(os.environ.get("ICIMS_REQUEST_TIMEOUT_SECONDS", "30"))
USER_AGENT = os.environ.get("ICIMS_USER_AGENT", "latmay-icims/1.0 (+https://latmay.com)")
MIN_REQUEST_GAP_SECONDS = float(os.environ.get("ICIMS_REQUEST_GAP_SECONDS", os.environ.get("REQUEST_GAP_SECONDS", "2.5")))
RANDOM_JITTER_MAX_SECONDS = float(os.environ.get("ICIMS_REQUEST_JITTER_SECONDS", "0.35"))
MAX_RETRIES = int(os.environ.get("ICIMS_MAX_RETRIES", "3"))
DEFAULT_PAGE_SIZE_PROBE = int(os.environ.get("ICIMS_PAGE_SIZE_PROBE", "25"))
BATCH_SOURCE_COUNT = 5

_last_request_by_host: dict[str, float] = {}
_robots_cache: dict[str, RobotFileParser | None] = {}


@dataclass
class FetchContext:
    session: requests.Session = field(default_factory=requests.Session)
    session_mode_by_host: dict[str, str] = field(default_factory=dict)


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def source_host(source_url: str) -> str:
    return urlparse(source_url).netloc.lower()


def source_company_from_url(source_url: str) -> str:
    host = source_host(source_url)
    return host[4:] if host.startswith("www.") else host


def api_url_from_listing_url(source_url: str) -> str:
    parsed = urlparse(source_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid iCIMS source URL: {source_url}")
    return urlunparse((parsed.scheme, parsed.netloc, "/api/jobs", "", "", ""))


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


def names_from_objects(items: Iterable[dict[str, Any]] | None, key: str = "name") -> str | None:
    if not items:
        return None
    names: list[str] = []
    for item in items:
        value = item.get(key) if isinstance(item, dict) else None
        if isinstance(value, str) and value.strip() and value.strip() not in names:
            names.append(value.strip())
    return " | ".join(names) if names else None


def tag_values(data: dict[str, Any]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for key, value in data.items():
        if not re.fullmatch(r"tags\d+", str(key)) or value is None:
            continue
        if isinstance(value, list):
            text = " | ".join(str(item).strip() for item in value if str(item).strip())
        else:
            text = str(value).strip()
        if text:
            tags[str(key)] = text
    return dict(sorted(tags.items(), key=lambda item: int(item[0][4:])))


def parse_money_amount(value: str) -> float | None:
    cleaned = value.replace(",", "")
    match = re.search(r"\$?\s*([0-9]+(?:\.[0-9]+)?)", cleaned)
    if not match:
        return None
    return float(match.group(1))


def classify_salary_from_tags(tags: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "salary_min": None,
        "salary_mid": None,
        "salary_max": None,
        "salary_currency": None,
        "salary_period": None,
    }
    currency_seen = False
    period_seen = False
    unlabeled_amounts: list[float] = []

    for value in tags.values():
        lower = value.lower()
        amounts = [float(item.replace(",", "")) for item in re.findall(r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)", value)]
        if not amounts:
            continue
        if "$" in value or "usd" in lower:
            currency_seen = True
        if re.search(r"\b(yr|year|annual|annually)\b", lower):
            period_seen = True
        elif re.search(r"\b(hr|hour|hourly)\b", lower):
            result["salary_period"] = "hour"

        for label, field_name in (("minimum", "salary_min"), ("min", "salary_min"), ("midpoint", "salary_mid"), ("maximum", "salary_max"), ("max", "salary_max")):
            match = re.search(rf"{label}\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)", lower)
            if match:
                result[field_name] = float(match.group(1).replace(",", ""))
        if not any(word in lower for word in ("minimum", "midpoint", "maximum", "min", "max")):
            unlabeled_amounts.extend(amounts)

    if unlabeled_amounts:
        if result["salary_min"] is None:
            result["salary_min"] = unlabeled_amounts[0]
        if len(unlabeled_amounts) >= 2 and result["salary_max"] is None:
            result["salary_max"] = unlabeled_amounts[-1]

    if currency_seen:
        result["salary_currency"] = "USD"
    if period_seen and result["salary_period"] is None:
        result["salary_period"] = "year"
    return result


def classify_workplace_from_tags(tags: dict[str, str]) -> str | None:
    haystack = " | ".join(tags.values()).lower()
    if re.search(r"\bhybrid\b", haystack):
        return "hybrid"
    if re.search(r"\bremote\b|\bhome office\b", haystack):
        return "remote"
    if re.search(r"\bon[- ]?site\b|\boffice location\b|\bin office\b", haystack):
        return "onsite"
    return None


def classify_employment_time_from_tags(tags: dict[str, str]) -> str | None:
    haystack = " | ".join(tags.values()).lower()
    if re.search(r"\bfull[- ]?time\b", haystack):
        return "Full-Time"
    if re.search(r"\bpart[- ]?time\b", haystack):
        return "Part-Time"
    return None


def classified_tag_values(tags: dict[str, str]) -> dict[str, Any]:
    return {
        **classify_salary_from_tags(tags),
        "workplace": classify_workplace_from_tags(tags),
        "employment_time_from_tags": classify_employment_time_from_tags(tags),
        "unclassified_tags": [
            value
            for value in tags.values()
            if not re.search(r"\$|usd|minimum|midpoint|maximum|remote|hybrid|on[- ]?site|office|full[- ]?time|part[- ]?time", value, re.I)
        ],
    }


def job_data(job: dict[str, Any]) -> dict[str, Any]:
    data = job.get("data") if isinstance(job.get("data"), dict) else job
    if not isinstance(data, dict):
        raise ValueError("Expected iCIMS job to contain a data object.")
    return data


def validate_icims_payload(payload: dict[str, Any]) -> None:
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Expected iCIMS payload['jobs'] to be a list.")
    if jobs:
        data = job_data(jobs[0])
        meta = data.get("meta_data") if isinstance(data.get("meta_data"), dict) else {}
        if data.get("ats_code") != "icims" and meta.get("ats_code") != "icims" and not isinstance(meta.get("icims"), dict):
            raise ValueError("Expected iCIMS job data to include an iCIMS ATS marker.")


def job_public_url(data: dict[str, Any], source_url: str) -> str | None:
    meta = data.get("meta_data") if isinstance(data.get("meta_data"), dict) else {}
    canonical = meta.get("canonical_url")
    if isinstance(canonical, str) and canonical.strip():
        return canonical.strip()
    slug = data.get("slug") or data.get("req_id")
    if slug:
        parsed = urlparse(source_url)
        listing_path = parsed.path.rstrip("/") or "/jobs"
        return urlunparse((parsed.scheme, parsed.netloc, f"{listing_path}/{slug}", "", "lang=en-us", ""))
    return None


def join_location(data: dict[str, Any]) -> str | None:
    for key in ("full_location", "location_name"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    parts = [data.get("city"), data.get("state"), data.get("country")]
    text = ", ".join(str(part).strip() for part in parts if part and str(part).strip())
    return text or None


def job_listing_fingerprint(record: dict[str, Any]) -> str:
    content_text = record.get("content_text")
    content_hash = hashlib.sha256(content_text.encode("utf-8")).hexdigest() if isinstance(content_text, str) else None
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


def normalize_job(job: dict[str, Any], source_url: str, fetched_at_utc: datetime) -> dict[str, Any]:
    data = job_data(job)
    meta = data.get("meta_data") if isinstance(data.get("meta_data"), dict) else {}
    tags = tag_values(data)
    tag_classification = classified_tag_values(tags)
    description_html = data.get("description")
    qualifications_html = data.get("qualifications")
    responsibilities_html = data.get("responsibilities")
    content_html = "\n\n".join(
        str(value)
        for value in (description_html, qualifications_html, responsibilities_html)
        if value
    ) or None
    source_company = source_company_from_url(source_url)
    source_job_id = data.get("req_id") or data.get("slug") or data.get("apply_url") or f"{source_url}:{data.get('title')}"

    raw_json = {
        "job": job,
        "raw_tags": tags,
        "tag_classification": tag_classification,
        "client_code": data.get("client_code") or meta.get("client_code"),
        "language": data.get("language"),
        "languages": data.get("languages"),
        "country_code": data.get("country_code"),
        "postal_code": data.get("postal_code"),
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "street_address": data.get("street_address"),
        "brand": data.get("brand"),
        "location_name": data.get("location_name"),
        "posting_expiry_date": data.get("posting_expiry_date"),
        "salary": {
            "min": tag_classification["salary_min"],
            "mid": tag_classification["salary_mid"],
            "max": tag_classification["salary_max"],
            "currency": tag_classification["salary_currency"],
            "period": tag_classification["salary_period"],
        },
        "workplace": tag_classification["workplace"],
        "employment_time_from_tags": tag_classification["employment_time_from_tags"],
        "unclassified_tags": tag_classification["unclassified_tags"],
    }
    record = {
        "source_type": "icims",
        "source_url": source_url,
        "source_job_id": str(source_job_id),
        "source_company": source_company,
        "company_name": data.get("hiring_organization") or source_company,
        "title": data.get("title"),
        "location_name": join_location(data),
        "department_names": names_from_objects(data.get("categories")),
        "office_names": data.get("location_name") or data.get("brand"),
        "posted_at": data.get("posted_date"),
        "updated_at": None,
        "job_url": job_public_url(data, source_url),
        "apply_url": data.get("apply_url"),
        "content_html": content_html,
        "content_text": strip_html_to_text(content_html),
        "raw_json": raw_json,
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
            ("icims", source_company),
        )
        rows = cur.fetchall()

    fingerprints: dict[str, str] = {}
    for row in rows:
        fingerprint = row.get("listing_fingerprint")
        source_job_id = str(row.get("source_job_id") or "").strip()
        if source_job_id and isinstance(fingerprint, str):
            fingerprints[source_job_id] = fingerprint
    return fingerprints


def robots_root(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def sleep_before_host_request(host: str, crawl_delay: float | None = None) -> None:
    now = time.monotonic()
    minimum_gap = crawl_delay if crawl_delay is not None else MIN_REQUEST_GAP_SECONDS
    last_request = _last_request_by_host.get(host)
    if last_request is not None:
        wait = minimum_gap + random.uniform(0, RANDOM_JITTER_MAX_SECONDS) - (now - last_request)
        if wait > 0:
            time.sleep(wait)
    _last_request_by_host[host] = time.monotonic()


def robots_parser_for(url: str, session: requests.Session) -> RobotFileParser | None:
    root = robots_root(url)
    if root in _robots_cache:
        return _robots_cache[root]
    parser = RobotFileParser()
    parser.set_url(f"{root}/robots.txt")
    try:
        response = session.get(
            f"{root}/robots.txt",
            headers={"User-Agent": USER_AGENT, "Accept": "text/plain,*/*;q=0.8"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            _robots_cache[root] = None
            return None
        parser.parse(response.text.splitlines())
        _robots_cache[root] = parser
        return parser
    except Exception:
        _robots_cache[root] = None
        return None


def robots_allowed(url: str, session: requests.Session) -> tuple[bool, float | None]:
    parser = robots_parser_for(url, session)
    if parser is None:
        return True, None
    return parser.can_fetch(USER_AGENT, url), parser.crawl_delay(USER_AGENT)


def polite_request(session: requests.Session, method: str, url: str, **kwargs: Any) -> requests.Response:
    allowed, crawl_delay = robots_allowed(url, session)
    if not allowed:
        raise RuntimeError(f"robots_disallowed:{url}")
    sleep_before_host_request(urlparse(url).netloc.lower(), crawl_delay)
    headers = {
        "User-Agent": USER_AGENT,
        **(kwargs.pop("headers", {}) or {}),
    }
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            _t = time.monotonic()
            response = session.request(method, url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
            log_timing("icims", source_company_from_url(url), "http_get", time.monotonic() - _t)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                retry_after = response.headers.get("Retry-After")
                if attempt < MAX_RETRIES:
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 8)
                    time.sleep(delay)
                    continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= MAX_RETRIES:
                break
            time.sleep(min(2 ** attempt, 8))
    if last_exc:
        raise last_exc
    raise RuntimeError(f"request_failed:{url}")


def fetch_icims_page(
    source_url: str,
    *,
    page: int,
    locale: str = "en-us",
    context: FetchContext | None = None,
    page_size: int | None = None,
) -> dict[str, Any]:
    context = context or FetchContext()
    api_url = api_url_from_listing_url(source_url)
    params: dict[str, Any] = {
        "page": page,
        "sortBy": "relevance",
        "descending": "false",
        "internal": "false",
    }
    if locale:
        params["lang"] = locale
    if page_size:
        params["limit"] = page_size
    headers = {"Accept": "application/json, text/plain, */*", "Referer": source_url}
    response = polite_request(context.session, "GET", api_url, params=params, headers=headers)
    if not response.text.strip() or response.url != api_url and "/api/jobs" not in response.url:
        context.session_mode_by_host[source_host(source_url)] = "bootstrap"
        polite_request(context.session, "GET", source_url, headers={"Accept": "text/html,application/xhtml+xml"})
        response = polite_request(context.session, "GET", api_url, params=params, headers=headers)
    else:
        context.session_mode_by_host.setdefault(source_host(source_url), "bare")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Expected iCIMS API response to be a JSON object.")
    validate_icims_payload(payload)
    return payload


def collect_all_payload_jobs(source_url: str, *, locale: str = "en-us", context: FetchContext | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context = context or FetchContext()
    first_payload = fetch_icims_page(source_url, page=1, locale=locale, context=context)
    jobs = [job for job in first_payload.get("jobs", []) if isinstance(job, dict)]
    total_count = int(first_payload.get("totalCount") or len(jobs))
    page_size = len(jobs) or 10
    pages_fetched = 1

    page = 2
    while len(jobs) < total_count:
        payload = fetch_icims_page(source_url, page=page, locale=locale, context=context)
        page_jobs = [job for job in payload.get("jobs", []) if isinstance(job, dict)]
        pages_fetched += 1
        if not page_jobs:
            break
        jobs.extend(page_jobs)
        if len(page_jobs) < page_size and len(jobs) >= total_count:
            break
        page += 1

    return jobs, {
        "total_count": total_count,
        "pages_fetched": pages_fetched,
        "page_size": page_size,
        "session_mode": context.session_mode_by_host.get(source_host(source_url), "unknown"),
    }


def collect_jobs_for_source(conn, source_url: str, *, locale: str = "en-us") -> tuple[list[dict[str, Any]], list[str], int, dict[str, Any]]:
    context = FetchContext()
    jobs, fetch_summary = collect_all_payload_jobs(source_url, locale=locale, context=context)
    fetched_at_utc = utc_now()
    source_company = source_company_from_url(source_url)
    existing_fingerprints = existing_fingerprints_for_source(conn, source_company)
    all_records = [normalize_job(job, source_url, fetched_at_utc) for job in jobs]
    records = [
        record
        for record in all_records
        if existing_fingerprints.get(record["source_job_id"]) != record["raw_json"]["listing_fingerprint"]
    ]
    skipped_count = len(all_records) - len(records)
    unclassified_tag_values = sum(len(record["raw_json"].get("unclassified_tags") or []) for record in all_records)
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
        "pages_fetched": fetch_summary["pages_fetched"],
        "session_mode": fetch_summary["session_mode"],
        "unclassified_tag_values": unclassified_tag_values,
    }
    print(
        f"icims: fetched {len(all_records)} jobs from {source_company}; "
        f"pages={fetch_summary['pages_fetched']}, session={fetch_summary['session_mode']}, "
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
    sources = load_ats_source_urls_from_db(conn, "icims")
    total_sources = len(sources)
    skipped_404_sources = count_skipped_404_sources_from_db(conn, "icims")
    http_errors = AtsHttpErrorTracker("icims")

    for source_url in sources:
        if should_stop_for_ingestion_budget("icims", started_at, completed_sources):
            break
        attempted_sources += 1
        source_company = source_company_from_url(source_url)
        attempt_http_status_code: int | None = None
        attempt_error_type: str | None = None
        stop_after_source = False
        try:
            _t = time.monotonic()
            records, seen_source_job_ids, skipped_count, quality_summary = collect_jobs_for_source(conn, source_url)
            log_timing("icims", source_company, "fetch", time.monotonic() - _t)

            batch.extend(records)

            _t = time.monotonic()
            marked_missing = mark_missing_jobs_for_source(
                conn,
                source_type="icims",
                source_company=source_company,
                seen_source_job_ids=seen_source_job_ids,
                seen_at_utc=records[0]["fetched_at_utc"] if records else utc_now(),
            )
            log_timing("icims", source_company, "mark_missing", time.monotonic() - _t)

            total_marked_missing += marked_missing
            if marked_missing:
                print(f"icims: marked {marked_missing} jobs missing for {source_company}", flush=True)
            log_data_quality(
                "ingestion",
                source_type="icims",
                company=source_company,
                inserted_updated="unknown",
                marked_missing=marked_missing,
                **quality_summary,
            )
            sources_in_batch += 1
            completed_sources += 1
            if sources_in_batch >= BATCH_SOURCE_COUNT:
                _t = time.monotonic()
                total_count += flush_batch(conn, batch, "icims")
                log_timing("icims", source_company, "flush", time.monotonic() - _t)
                sources_in_batch = 0
        except Exception as exc:
            error_sources += 1
            http_status_code = http_status_code_from_exception(exc)
            attempt_http_status_code = http_status_code
            attempt_error_type = type(exc).__name__
            stop_after_source = isinstance(exc, AtsHttp429LimitReached) or http_errors.record(http_status_code)
            print(
                "icims: error fetching source: "
                f"company={source_company}, error_type={type(exc).__name__}, "
                f"http_status_code={http_status_code or 'unknown'}",
                flush=True,
            )
            log_data_quality(
                "ingestion",
                source_type="icims",
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
            log_timing("icims", source_company, "last_get_at", time.monotonic() - _t)
        if stop_after_source:
            break

    total_count += flush_batch(conn, batch, "icims")
    checked_percent = (attempted_sources / total_sources * 100) if total_sources else 0.0
    print(
        f"icims: checked {attempted_sources}/{total_sources} sources ({checked_percent:.1f}%); errors={error_sources}",
        flush=True,
    )
    http_errors.print_summary()
    print(f"icims: skipped {skipped_404_sources} sources due to 404 streak", flush=True)
    print(f"icims: inserted/updated {total_count} rows total; marked {total_marked_missing} missing", flush=True)
    return total_count



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape modern iCIMS/Jibe careers APIs to normalized JSONL.",
        epilog="Example: python -m data_pipeline.ingestion.icims https://jobs.tufts.edu/jobs --verbose --pretty --limit 3",
    )
    parser.add_argument("urls", nargs="+", help="Public listing URLs, e.g. https://jobs.tufts.edu/jobs")
    parser.add_argument("--locale", default="en-us", help="Locale/lang parameter to request. Default: en-us.")
    parser.add_argument("--out", help="Path to write output. Defaults to stdout.")
    parser.add_argument("--include-raw", action="store_true", help="Include full raw job JSON in each JSONL record.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print fetch details (API URL, page counts, etc.) to stderr.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON (easier to read, not line-delimited).")
    parser.add_argument("--raw-api", action="store_true", help="Dump the raw API JSON response instead of normalized jobs.")
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="Only output the first N jobs (useful for quick inspection).")
    return parser


def cli_main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    context = FetchContext()

    if args.raw_api:
        payloads = []
        for source_url in args.urls:
            api_url = api_url_from_listing_url(source_url)
            if args.verbose:
                print(f"[debug] source_url : {source_url}", file=sys.stderr)
                print(f"[debug] api_url    : {api_url}", file=sys.stderr)
            payload = fetch_icims_page(source_url, page=1, locale=args.locale, context=context)
            if args.verbose:
                total = payload.get("totalCount", "?")
                print(f"[debug] totalCount : {total}", file=sys.stderr)
            payloads.append(payload)
        out_obj = payloads[0] if len(payloads) == 1 else payloads
        text = json.dumps(out_obj, default=str, ensure_ascii=False, indent=2 if args.pretty else None)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 0

    fetched_at_utc = utc_now()
    records: list[dict] = []
    for source_url in args.urls:
        api_url = api_url_from_listing_url(source_url)
        if args.verbose:
            print(f"[debug] source_url : {source_url}", file=sys.stderr)
            print(f"[debug] api_url    : {api_url}", file=sys.stderr)
        jobs, fetch_summary = collect_all_payload_jobs(source_url, locale=args.locale, context=context)
        if args.verbose:
            print(
                f"[debug] fetched {len(jobs)} jobs | "
                f"pages={fetch_summary['pages_fetched']} | "
                f"session_mode={fetch_summary['session_mode']}",
                file=sys.stderr,
            )
        if args.limit is not None:
            jobs = jobs[: args.limit]
        for job in jobs:
            record = normalize_job(job, source_url, fetched_at_utc)
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
