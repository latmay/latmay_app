from __future__ import annotations

"""
Fill blank jobs.content_text values directly in PostgreSQL.

Rows are selected from the unified jobs table, fetched via job_url, cleaned, and
updated in place. Newer jobs are handled first, and the step stops early by
design when CONTENT_FILL_TIME_BUDGET_SECONDS is reached.
"""

import os
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup

from data_pipeline.common.data_quality import length_distribution, log_data_quality
from data_pipeline.common.db import strip_nul_bytes


MIN_SECONDS_BETWEEN_REQUESTS = float(os.environ.get("CONTENT_FETCH_GAP_SECONDS", "2"))
CONTENT_FILL_TIME_BUDGET_SECONDS = float(os.environ.get("CONTENT_FILL_TIME_BUDGET_SECONDS", "900"))
CONTENT_FILL_MAX_ROWS = int(os.environ["CONTENT_FILL_MAX_ROWS"]) if os.environ.get("CONTENT_FILL_MAX_ROWS") else None
CONTENT_FILL_MAX_FAILURES = int(os.environ.get("CONTENT_FILL_MAX_FAILURES", "3"))
CONTENT_FILL_RETRY_AFTER_HOURS = float(os.environ.get("CONTENT_FILL_RETRY_AFTER_HOURS", "24"))
CONTENT_FILL_INCLUDE_DIRECT_SOURCES = os.environ.get("CONTENT_FILL_INCLUDE_DIRECT_SOURCES", "false").strip().lower() in {
    "true",
    "1",
    "yes",
    "on",
}
CONTENT_FILL_DIRECT_SOURCE_RECENT_HOURS = float(os.environ.get("CONTENT_FILL_DIRECT_SOURCE_RECENT_HOURS", "24"))
DIRECT_CONTENT_SOURCE_TYPES = {"ashby", "greenhouse", "icims", "lever", "workday"}
REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = "Mozilla/5.0 (compatible; latmay-content-fill/1.0)"
_last_request_time: float | None = None


def polite_sleep() -> None:
    global _last_request_time

    now = time.monotonic()
    if _last_request_time is not None:
        wait = MIN_SECONDS_BETWEEN_REQUESTS - (now - _last_request_time)
        if wait > 0:
            time.sleep(wait)

    _last_request_time = time.monotonic()


def fetch_html(url: str) -> str:
    polite_sleep()
    response = requests.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.text


def clean_text_from_node(node: Any) -> str:
    for bad in node.select("script, style, noscript, form, button, input, textarea, select"):
        bad.decompose()

    lines = []
    for line in node.get_text(separator="\n", strip=True).splitlines():
        line = " ".join(line.split())
        if line:
            lines.append(line)

    return "\n".join(lines).strip()


def looks_like_job_description(text: str) -> bool:
    if len(text) < 300:
        return False

    lower = text.lower()
    useful_terms = [
        "qualification",
        "qualifications",
        "requirements",
        "what you'll bring",
        "what you will bring",
        "responsibilities",
        "what you'll do",
        "what you will do",
        "about the role",
        "experience",
        "skills",
    ]
    return any(term in lower for term in useful_terms)


def extract_job_description(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    selectors = [
        "div.job__description.body",
        "div[class*='job__description']",
        "div.job-description",
        "div[class*='job-description']",
        "div.posting-page",
        "div.posting-page-content",
        "section.posting",
        "main",
        "article",
    ]

    candidates: list[str] = []
    for selector in selectors:
        for node in soup.select(selector):
            text = clean_text_from_node(node)
            if looks_like_job_description(text):
                candidates.append(text)

    if candidates:
        return max(candidates, key=len)

    body = soup.body or soup
    text = clean_text_from_node(body)
    return text if looks_like_job_description(text) else ""


def bad_content_reasons(html: str, extracted_text: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    page_text = clean_text_from_node(soup.body or soup)
    lower = page_text.lower()
    reasons: set[str] = set()

    if len(page_text) < 300:
        reasons.add("too_short")
    if any(term in lower for term in ["access denied", "forbidden", "not authorized"]):
        reasons.add("access_denied")
    if any(term in lower for term in ["job no longer available", "position has been filled", "posting is closed"]):
        reasons.add("job_no_longer_available")
    if any(term in lower for term in ["enable javascript", "javascript is required", "requires javascript"]):
        reasons.add("enable_javascript")

    useful_terms = [
        "qualification",
        "qualifications",
        "requirements",
        "responsibilities",
        "experience",
        "skills",
    ]
    if page_text and not any(term in lower for term in useful_terms):
        reasons.add("no_requirements_like_text")
    if page_text and len(set(page_text.lower().split())) < 30:
        reasons.add("boilerplate_only")
    if not extracted_text and not reasons:
        reasons.add("unusable_unknown")

    return reasons


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def record_content_fetch_success(conn, job_id: int, html: str, text: str) -> None:
    html = strip_nul_bytes(html)
    text = strip_nul_bytes(text)
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET content_html = COALESCE(content_html, %s),
                content_text = %s,
                requirements_extraction_version = NULL,
                yoe_extraction_version = NULL,
                clearance_extraction_version = NULL,
                enrichment_version = NULL,
                enrichment_ml_version = NULL,
                content_fetch_failed_count = 0,
                content_fetch_last_failed_at_utc = NULL,
                content_fetch_last_error_type = NULL
            WHERE id = %s
            """,
            (html, text, job_id),
        )


def record_content_fetch_failure(conn, job_id: int, error_type: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET content_fetch_failed_count = COALESCE(content_fetch_failed_count, 0) + 1,
                content_fetch_last_failed_at_utc = %s,
                content_fetch_last_error_type = %s
            WHERE id = %s
            """,
            (utc_now(), error_type, job_id),
        )


def count_backoff_skipped_rows(conn) -> int:
    sql = """
        SELECT COUNT(*) AS skipped_count
        FROM jobs
        WHERE (content_text IS NULL OR btrim(content_text) = '')
          AND job_url IS NOT NULL
          AND btrim(job_url) <> ''
          AND COALESCE(is_active, TRUE) = TRUE
          AND missing_from_source_at_utc IS NULL
          AND COALESCE(content_fetch_failed_count, 0) >= %(max_failures)s
          AND content_fetch_last_failed_at_utc IS NOT NULL
          AND content_fetch_last_failed_at_utc > now() - (%(retry_after_hours)s * interval '1 hour')
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            {
                "max_failures": CONTENT_FILL_MAX_FAILURES,
                "retry_after_hours": CONTENT_FILL_RETRY_AFTER_HOURS,
            },
        )
        row = cur.fetchone()
    return int((row or {}).get("skipped_count") or 0)


def fill_missing_content_text(conn, *, limit: int | None = None) -> int:
    started_at = time.monotonic()
    effective_limit = limit if limit is not None else CONTENT_FILL_MAX_ROWS
    skipped_backoff_count = count_backoff_skipped_rows(conn)
    if skipped_backoff_count:
        print(
            "fill_missing_content_text: skipped "
            f"{skipped_backoff_count} rows due to failure backoff",
            flush=True,
        )

    sql = """
        SELECT id, job_url, source_type
        FROM jobs
        WHERE (content_text IS NULL OR btrim(content_text) = '')
          AND job_url IS NOT NULL
          AND btrim(job_url) <> ''
          AND COALESCE(is_active, TRUE) = TRUE
          AND missing_from_source_at_utc IS NULL
          AND (
            COALESCE(content_fetch_failed_count, 0) < %(max_failures)s
            OR content_fetch_last_failed_at_utc IS NULL
            OR content_fetch_last_failed_at_utc <= now() - (%(retry_after_hours)s * interval '1 hour')
          )
          AND (
            %(include_direct_sources)s
            OR source_type IS NULL
            OR source_type <> ALL(%(direct_source_types)s)
            OR COALESCE(
              CASE
                WHEN posted_at ~ '^\\d{4}-\\d{2}-\\d{2}' THEN posted_at::timestamptz
                ELSE NULL
              END,
              fetched_at_utc,
              created_at_utc
            ) >= now() - (%(direct_source_recent_hours)s * interval '1 hour')
          )
        ORDER BY
          COALESCE(
            CASE
              WHEN posted_at ~ '^\\d{4}-\\d{2}-\\d{2}' THEN posted_at::timestamptz
              ELSE NULL
            END,
            CASE
              WHEN updated_at ~ '^\\d{4}-\\d{2}-\\d{2}' THEN updated_at::timestamptz
              ELSE NULL
            END,
            fetched_at_utc,
            created_at_utc
          ) DESC,
          id DESC
    """
    params: dict[str, Any] = {
        "max_failures": CONTENT_FILL_MAX_FAILURES,
        "retry_after_hours": CONTENT_FILL_RETRY_AFTER_HOURS,
        "include_direct_sources": CONTENT_FILL_INCLUDE_DIRECT_SOURCES,
        "direct_source_types": list(DIRECT_CONTENT_SOURCE_TYPES),
        "direct_source_recent_hours": CONTENT_FILL_DIRECT_SOURCE_RECENT_HOURS,
    }
    if effective_limit is not None:
        sql += " LIMIT %(limit)s"
        params["limit"] = effective_limit

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    filled_count = 0
    failed_count = 0
    unusable_count = 0
    filled_texts: list[str] = []
    bad_reason_counts: Counter[str] = Counter()
    failure_type_counts: Counter[str] = Counter()
    for row in rows:
        elapsed = time.monotonic() - started_at
        if elapsed >= CONTENT_FILL_TIME_BUDGET_SECONDS:
            print(
                "fill_missing_content_text: time budget reached "
                f"after {elapsed:.1f}s; stopping early with {filled_count} rows filled",
                flush=True,
            )
            break

        job_id = row["id"]
        url = row["job_url"]
        try:
            html = fetch_html(url)
            text = extract_job_description(html)
            if not text:
                record_content_fetch_failure(conn, job_id, "UnusableContent")
                unusable_count += 1
                bad_reason_counts.update(bad_content_reasons(html, text))
                print(f"fill_missing_content_text: unusable content job id={job_id}", flush=True)
                continue

            record_content_fetch_success(conn, job_id, html, text)
            filled_count += 1
            filled_texts.append(text)
            print(f"fill_missing_content_text: filled job id={job_id}", flush=True)
        except Exception as exc:
            record_content_fetch_failure(conn, job_id, type(exc).__name__)
            failed_count += 1
            failure_type_counts[type(exc).__name__] += 1
            print(
                "fill_missing_content_text: failed "
                f"job id={job_id} error_type={type(exc).__name__}",
                flush=True,
            )

    conn.commit()
    print(
        "fill_missing_content_text: updated "
        f"{filled_count} rows within {CONTENT_FILL_TIME_BUDGET_SECONDS:.0f}s budget; "
        f"failures={failed_count}, unusable={unusable_count}, selected={len(rows)}",
        flush=True,
    )
    log_data_quality(
        "content_fill",
        selected=len(rows),
        filled=filled_count,
        failed=failed_count,
        unusable=unusable_count,
        skipped_backoff=skipped_backoff_count,
        too_short=bad_reason_counts.get("too_short", 0),
        boilerplate_only=bad_reason_counts.get("boilerplate_only", 0),
        no_requirements_like_text=bad_reason_counts.get("no_requirements_like_text", 0),
        access_denied=bad_reason_counts.get("access_denied", 0),
        job_no_longer_available=bad_reason_counts.get("job_no_longer_available", 0),
        enable_javascript=bad_reason_counts.get("enable_javascript", 0),
        failure_type_counts=dict(sorted(failure_type_counts.items())),
        **length_distribution(filled_texts),
    )
    return filled_count


def run(conn) -> int:
    return fill_missing_content_text(conn)


if __name__ == "__main__":
    from data_pipeline.common.db import connect

    with connect() as connection:
        run(connection)
