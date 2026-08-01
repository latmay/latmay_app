from __future__ import annotations

"""Cloud Run Job entry point for lightweight ingestion.

Initializes schema, scrapes Ashby/Greenhouse/Lever/Workday/iCIMS/Workable, soft-flags stale
jobs, and fills missing content_text with the configured time budget.
"""

import os

import psycopg

from data_pipeline.common.db import connect
from data_pipeline.common.data_quality import log_data_quality
from data_pipeline.common.schema import initialize_schema
from data_pipeline.ingestion import ashby, fill_missing_content_text, greenhouse, icims, lever, workable, workday


INGESTION_VERSION = "2026-06-30.1"
CONTINUABLE_DB_STEP_ERRORS = (
    psycopg.errors.QueryCanceled,
    psycopg.errors.LockNotAvailable,
    psycopg.errors.DeadlockDetected,
)
ATS_STEPS = {
    "workable": ("Workable", "ENABLE_WORKABLE_INGESTION", workable.run),
    "workday": ("Workday", "ENABLE_WORKDAY_INGESTION", workday.run),
    "ashby": ("Ashby", "ENABLE_ASHBY_INGESTION", ashby.run),
    "greenhouse": ("Greenhouse", "ENABLE_GREENHOUSE_INGESTION", greenhouse.run),
    "lever": ("Lever", "ENABLE_LEVER_INGESTION", lever.run),
    "icims": ("iCIMS", "ENABLE_ICIMS_INGESTION", icims.run),
}


def env_bool(name: str, default: bool = True) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"true", "1", "yes", "on"}


def rollback_and_log_continuable_step_failure(conn, step_name: str, exc: BaseException) -> None:
    error_type = type(exc).__name__
    sqlstate = getattr(exc, "sqlstate", None) or "unknown"
    print(
        f"{step_name}: database operation canceled; "
        f"error_type={error_type}, sqlstate={sqlstate}; rolling back and continuing",
        flush=True,
    )
    try:
        conn.rollback()
    except Exception as rollback_exc:
        print(
            f"{step_name}: rollback failed after database operation cancel; "
            f"error_type={type(rollback_exc).__name__}; re-raising original failure",
            flush=True,
        )
        raise exc from rollback_exc

    log_data_quality(
        "ingestion_optional_step_failure",
        step=step_name,
        error_type=error_type,
        sqlstate=sqlstate,
        rollback="ok",
        continued=1,
    )
    print(f"{step_name}: skipped remaining work after database timeout/cancel", flush=True)


def run_or_skip(
    step_name: str,
    enabled_env_var: str,
    fn,
    *,
    conn=None,
    continue_on_db_cancel: bool = False,
) -> None:
    if env_bool(enabled_env_var, True):
        print(f"{step_name}: enabled", flush=True)
        try:
            fn()
        except CONTINUABLE_DB_STEP_ERRORS as exc:
            if not continue_on_db_cancel:
                raise
            if conn is None:
                raise
            rollback_and_log_continuable_step_failure(conn, step_name, exc)
    else:
        print(f"{step_name}: skipped because {enabled_env_var}=false", flush=True)


def selected_ingestion_ats() -> str:
    return os.environ.get("INGESTION_ATS", "").strip().lower()


def run_ats_ingestion_steps(conn) -> None:
    selected_ats = selected_ingestion_ats()
    if selected_ats in {"", "all"}:
        print("ingestion: INGESTION_ATS is blank/all; running enabled ATS scrapers", flush=True)
        for step_number, (step_name, enabled_env_var, run_step) in enumerate(ATS_STEPS.values(), start=2):
            print(f"ingestion step {step_number}/8: running {step_name} scraper", flush=True)
            run_or_skip(step_name.lower(), enabled_env_var, lambda run_step=run_step: run_step(conn))
        return

    if selected_ats in {"none", "skip"}:
        print(f"ingestion: skipping ATS scrapers because INGESTION_ATS={selected_ats}", flush=True)
        return

    if selected_ats not in ATS_STEPS:
        supported = ", ".join([*ATS_STEPS.keys(), "all", "none"])
        raise ValueError(f"Unsupported INGESTION_ATS={selected_ats!r}. Supported values: {supported}.")

    step_name, _enabled_env_var, run_step = ATS_STEPS[selected_ats]
    print(f"ingestion: running only {step_name} scraper because INGESTION_ATS={selected_ats}", flush=True)
    run_step(conn)


def print_posted_at_quality_summary(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE NULLIF(btrim(posted_at), '') IS NULL
                       OR lower(btrim(posted_at)) IN ('nan', 'none', 'null')
                ) AS missing_posted_at,
                COUNT(*) FILTER (
                    WHERE NULLIF(btrim(posted_at), '') IS NOT NULL
                      AND lower(btrim(posted_at)) NOT IN ('nan', 'none', 'null')
                ) AS has_posted_at,
                COUNT(*) AS total_jobs
            FROM jobs
            WHERE is_active = TRUE
            """
        )
        row = cur.fetchone()

    missing_posted_at = int(row["missing_posted_at"] or 0)
    has_posted_at = int(row["has_posted_at"] or 0)
    total_jobs = int(row["total_jobs"] or 0)
    percent_missing = (missing_posted_at / total_jobs * 100) if total_jobs else 0.0
    log_data_quality(
        "ingestion_posted_at_summary",
        active_jobs=total_jobs,
        has_posted_at=has_posted_at,
        missing_posted_at=missing_posted_at,
        percent_missing_posted_at=percent_missing,
    )


def run_ingestion_job() -> None:
    print(f"ingestion: version {INGESTION_VERSION}", flush=True)
    print("ingestion: using configured database", flush=True)
    with connect() as conn:
        print("ingestion step 1/8: initializing schema", flush=True)
        initialize_schema(conn)
        run_ats_ingestion_steps(conn)
        print("ingestion step 8/8: filling missing content_text", flush=True)
        run_or_skip(
            "content_fill",
            "ENABLE_CONTENT_FILL",
            lambda: fill_missing_content_text.run(conn),
            conn=conn,
            continue_on_db_cancel=True,
        )
        print_posted_at_quality_summary(conn)
    print("ingestion: finished successfully", flush=True)


if __name__ == "__main__":
    run_ingestion_job()
