from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import psycopg


def env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"true", "1", "yes", "on"}


def skip_404_streak_threshold() -> int:
    raw_value = os.environ.get("SKIP_404_STREAK_THRESHOLD", "2").strip()
    try:
        threshold = int(raw_value)
    except ValueError:
        return 2
    return max(threshold, 1)


def load_ats_source_urls_from_db(conn: "psycopg.Connection", ats_type: str) -> list[str]:
    """Return source URLs for this ATS from the DB, ordered by last_get_at ASC NULLS FIRST."""
    params: list[Any] = [ats_type]
    skip_404_sources = env_bool("SKIP_404_SOURCES", False)
    skip_predicate = ""
    if skip_404_sources:
        skip_predicate = """
              AND NOT (
                  last_http_status_code = 404
                  AND http_404_streak >= %s
              )
        """
        params.append(skip_404_streak_threshold())

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT source_url
            FROM ats_sources
            WHERE ats = %s
            {skip_predicate}
            ORDER BY last_get_at ASC NULLS FIRST
            """,
            params,
        )
        rows = cur.fetchall()
    return [row["source_url"] for row in rows]


def count_skipped_404_sources_from_db(conn: "psycopg.Connection", ats_type: str) -> int:
    """Return how many sources would be skipped by the 404-streak filter for this ATS."""
    if not env_bool("SKIP_404_SOURCES", False):
        return 0

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) AS skipped_count
            FROM ats_sources
            WHERE ats = %s
              AND last_http_status_code = 404
              AND http_404_streak >= %s
            """,
            [ats_type, skip_404_streak_threshold()],
        )
        row = cur.fetchone()
    return int((row or {}).get("skipped_count") or 0)


def rollback_failed_source_attempt(conn: "psycopg.Connection") -> None:
    """Clear an aborted transaction before writing source-attempt metadata."""
    try:
        conn.rollback()
    except Exception:
        pass


def update_source_last_get_at(
    conn: "psycopg.Connection",
    source_url: str,
    *,
    http_status_code: int | None = None,
    error_type: str | None = None,
) -> None:
    """Mark a source URL as attempted and persist last HTTP/error metadata."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE ats_sources
            SET
                last_get_at = now(),
                last_http_status_code = %(http_status_code)s::integer,
                last_error_type = %(error_type)s::text,
                last_error_at = CASE
                    WHEN %(error_type)s::text IS NULL THEN NULL
                    ELSE now()
                END,
                http_404_streak = CASE
                    WHEN %(http_status_code)s::integer = 404 THEN http_404_streak + 1
                    ELSE 0
                END
            WHERE source_url = %(source_url)s
            """,
            {
                "source_url": source_url,
                "http_status_code": http_status_code,
                "error_type": error_type,
            },
        )
    conn.commit()
