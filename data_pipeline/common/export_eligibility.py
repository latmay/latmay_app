from __future__ import annotations

"""Schema readiness checks for trigger-maintained export eligibility."""

from typing import Any

import psycopg


EXPORT_ELIGIBILITY_COLUMN = "is_export_eligible"
EXPORT_ELIGIBILITY_INDEX = "idx_jobs_export_eligible_flag_recent"


def require_export_eligibility_schema(conn: psycopg.Connection) -> None:
    """Fail clearly unless the generated eligibility column and index are ready."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'jobs'
                      AND column_name = %(column_name)s
                ) AS column_ready,
                EXISTS (
                    SELECT 1
                    FROM pg_trigger
                    WHERE tgrelid = 'public.jobs'::regclass
                      AND tgname = 'trg_jobs_set_export_eligibility'
                      AND NOT tgisinternal
                      AND tgenabled <> 'D'
                ) AS trigger_ready,
                EXISTS (
                    SELECT 1
                    FROM pg_index
                    WHERE indexrelid = to_regclass(%(index_name)s)
                      AND indisready
                      AND indisvalid
                ) AS index_ready
            """,
            {
                "column_name": EXPORT_ELIGIBILITY_COLUMN,
                "index_name": f"public.{EXPORT_ELIGIBILITY_INDEX}",
            },
        )
        row: dict[str, Any] = cur.fetchone()
    conn.commit()

    if not row["column_ready"] or not row["trigger_ready"] or not row["index_ready"]:
        raise RuntimeError(
            "Stored export eligibility is not ready. Apply migration "
            "002_add_batched_export_eligibility.sql before running this pipeline."
        )
