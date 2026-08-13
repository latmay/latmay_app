from __future__ import annotations

"""
PostgreSQL connection and upsert helpers for the Latmay data pipeline.

The pipeline uses DATABASE_URL when set. Local development falls back to
LOCAL_DATABASE_URL, which must be set to match your local/Docker Compose
database credentials.
"""

import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from data_pipeline.common.timing import log_timing


JOB_COLUMNS = [
    "source_type",
    "source_url",
    "source_job_id",
    "source_company",
    "company_name",
    "title",
    "location_name",
    "department_names",
    "office_names",
    "posted_at",
    "updated_at",
    "job_url",
    "apply_url",
    "content_html",
    "content_text",
    "raw_json",
    "fetched_at_utc",
    "first_seen_at_utc",
    "last_seen_at_utc",
    "is_active",
    "missing_from_source_at_utc",
    "stale_reason",
]


def get_database_url() -> str:
    if os.environ.get("K_SERVICE") and not os.environ.get("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL is required when running the data pipeline on Cloud Run.")

    if os.environ.get("K_SERVICE"):
        return os.environ["DATABASE_URL"]

    value = os.environ.get("LOCAL_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not value:
        raise RuntimeError("LOCAL_DATABASE_URL (or DATABASE_URL) is required to run the data pipeline locally.")
    return value


def env_timeout_milliseconds(name: str) -> int | None:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return None
    try:
        seconds = float(raw_value)
    except ValueError:
        return None
    if seconds <= 0:
        return None
    return int(seconds * 1000)


def configure_connection_timeouts(conn: psycopg.Connection) -> None:
    statement_timeout_ms = env_timeout_milliseconds("INGESTION_DB_STATEMENT_TIMEOUT_SECONDS")
    lock_timeout_ms = env_timeout_milliseconds("INGESTION_DB_LOCK_TIMEOUT_SECONDS")
    if statement_timeout_ms is None and lock_timeout_ms is None:
        return

    with conn.cursor() as cur:
        if statement_timeout_ms is not None:
            cur.execute("SELECT set_config('statement_timeout', %s, false)", (f"{statement_timeout_ms}ms",))
        if lock_timeout_ms is not None:
            cur.execute("SELECT set_config('lock_timeout', %s, false)", (f"{lock_timeout_ms}ms",))
    conn.commit()


@contextmanager
def connect():
    """
    Open a configured PostgreSQL connection with dict rows and close it after use.

    Timeouts are applied here so every pipeline entry point is protected,
    including enrichment and export jobs as well as ingestion.
    """
    with psycopg.connect(get_database_url(), row_factory=dict_row) as conn:
        configure_connection_timeouts(conn)
        yield conn


def strip_nul_bytes(value: Any) -> Any:
    """Recursively remove NUL characters that PostgreSQL text/JSON cannot store."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {
            strip_nul_bytes(key): strip_nul_bytes(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [strip_nul_bytes(item) for item in value]
    if isinstance(value, tuple):
        return tuple(strip_nul_bytes(item) for item in value)
    return value


def normalize_job_record(record: dict[str, Any]) -> dict[str, Any]:
    from data_pipeline.ingestion.size_limits import limit_job_record_storage

    record = limit_job_record_storage(record)
    sanitized_record = strip_nul_bytes(record)
    normalized = {column: sanitized_record.get(column) for column in JOB_COLUMNS}
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)

    if not normalized["source_type"]:
        raise ValueError("Job record is missing source_type.")
    if not normalized["source_job_id"]:
        raise ValueError("Job record is missing source_job_id.")

    normalized["source_company"] = (
        normalized.get("source_company")
        or sanitized_record.get("company_name")
        or sanitized_record.get("source_url")
        or "unknown"
    )
    normalized["fetched_at_utc"] = normalized.get("fetched_at_utc") or now_utc
    normalized["first_seen_at_utc"] = normalized.get("first_seen_at_utc") or normalized["fetched_at_utc"]
    normalized["last_seen_at_utc"] = normalized.get("last_seen_at_utc") or normalized["fetched_at_utc"]
    normalized["is_active"] = True if normalized.get("is_active") is None else normalized["is_active"]
    normalized["missing_from_source_at_utc"] = None
    normalized["stale_reason"] = None
    normalized["raw_json"] = Jsonb(normalized.get("raw_json") or {})
    return normalized


def upsert_jobs(conn: psycopg.Connection, records: Iterable[dict[str, Any]]) -> int:
    """
    Insert/update normalized job records into the unified jobs table.
    """
    started_at = time.monotonic()
    rows = [normalize_job_record(record) for record in records]
    if not rows:
        return 0
    source_type = str(rows[0].get("source_type") or "unknown")
    log_timing(source_type, "batch", "upsert_normalize_records", time.monotonic() - started_at)

    placeholders = ", ".join(f"%({column})s" for column in JOB_COLUMNS)
    columns_sql = ", ".join(JOB_COLUMNS)
    update_columns = [
        column
        for column in JOB_COLUMNS
        if column not in {"source_type", "source_job_id", "first_seen_at_utc"}
    ]
    update_assignments = [f"{column} = EXCLUDED.{column}" for column in update_columns]
    title_or_content_changed = (
        "jobs.title IS DISTINCT FROM EXCLUDED.title "
        "OR jobs.content_text IS DISTINCT FROM EXCLUDED.content_text"
    )
    content_changed = "jobs.content_text IS DISTINCT FROM EXCLUDED.content_text"
    posted_at_changed = "jobs.posted_at IS DISTINCT FROM EXCLUDED.posted_at"
    update_assignments.extend(
        [
            (
                "enrichment_version = CASE "
                f"WHEN {title_or_content_changed} THEN NULL "
                "ELSE jobs.enrichment_version END"
            ),
            (
                "enrichment_ml_version = CASE "
                f"WHEN {title_or_content_changed} THEN NULL "
                "ELSE jobs.enrichment_ml_version END"
            ),
            (
                "requirements_extraction_version = CASE "
                f"WHEN {content_changed} THEN NULL "
                "ELSE jobs.requirements_extraction_version END"
            ),
            (
                "yoe_extraction_version = CASE "
                f"WHEN {content_changed} THEN NULL "
                "ELSE jobs.yoe_extraction_version END"
            ),
            (
                "clearance_extraction_version = CASE "
                f"WHEN {content_changed} THEN NULL "
                "ELSE jobs.clearance_extraction_version END"
            ),
            (
                "posted_at_utc = CASE "
                f"WHEN {posted_at_changed} THEN NULL "
                "ELSE jobs.posted_at_utc END"
            ),
            (
                "posted_at_normalization_status = CASE "
                f"WHEN {posted_at_changed} THEN NULL "
                "ELSE jobs.posted_at_normalization_status END"
            ),
            (
                "posted_at_normalization_failure_reason = CASE "
                f"WHEN {posted_at_changed} THEN NULL "
                "ELSE jobs.posted_at_normalization_failure_reason END"
            ),
            (
                "posted_at_normalized_at_utc = CASE "
                f"WHEN {posted_at_changed} THEN NULL "
                "ELSE jobs.posted_at_normalized_at_utc END"
            ),
        ]
    )
    update_sql = ", ".join(update_assignments)

    sql = f"""
        INSERT INTO jobs ({columns_sql})
        VALUES ({placeholders})
        ON CONFLICT (source_type, source_job_id)
        DO UPDATE SET
            {update_sql}
    """

    with conn.cursor() as cur:
        started_at = time.monotonic()
        cur.executemany(sql, rows)
        log_timing(source_type, "batch", "upsert_executemany", time.monotonic() - started_at)

    started_at = time.monotonic()
    conn.commit()
    log_timing(source_type, "batch", "upsert_commit", time.monotonic() - started_at)
    return len(rows)


def mark_missing_jobs_for_source(
    conn: psycopg.Connection,
    *,
    source_type: str,
    source_company: str,
    seen_source_job_ids: Iterable[str],
    seen_at_utc: datetime | None = None,
) -> int:
    """
    Soft-deactivate active jobs for one successfully fetched source/company.

    This is intentionally scoped to a single source company so a failed source
    fetch cannot mark unrelated jobs stale.
    """
    seen_ids = [str(job_id) for job_id in seen_source_job_ids if str(job_id).strip()]
    seen_at_utc = seen_at_utc or datetime.now(timezone.utc).replace(microsecond=0)

    if seen_ids:
        sql = """
            UPDATE jobs
            SET
                is_active = FALSE,
                missing_from_source_at_utc = COALESCE(missing_from_source_at_utc, %(seen_at_utc)s),
                stale_reason = 'missing_from_source'
            WHERE source_type = %(source_type)s
              AND source_company = %(source_company)s
              AND is_active = TRUE
              AND NOT (source_job_id = ANY(%(seen_ids)s))
        """
        params = {
            "source_type": source_type,
            "source_company": source_company,
            "seen_ids": seen_ids,
            "seen_at_utc": seen_at_utc,
        }
    else:
        sql = """
            UPDATE jobs
            SET
                is_active = FALSE,
                missing_from_source_at_utc = COALESCE(missing_from_source_at_utc, %(seen_at_utc)s),
                stale_reason = 'missing_from_source'
            WHERE source_type = %(source_type)s
              AND source_company = %(source_company)s
              AND is_active = TRUE
        """
        params = {
            "source_type": source_type,
            "source_company": source_company,
            "seen_at_utc": seen_at_utc,
        }

    with conn.cursor() as cur:
        cur.execute(sql, params)
        marked_count = cur.rowcount

    conn.commit()
    return marked_count
