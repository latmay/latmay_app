from __future__ import annotations

"""Cloud Run Job entry point for non-ML PostgreSQL-backed enrichment."""

import signal
from contextlib import contextmanager
from types import FrameType
from typing import Iterator

import psycopg

from data_pipeline.common.db import connect
from data_pipeline.common.export_eligibility import require_export_eligibility_schema
from data_pipeline.common.schema import initialize_schema
from data_pipeline.enrichment import (
    add_job_features,
    add_requirements,
    add_security_clearance,
    add_years_experience,
    clean_content_text,
    normalize_location,
    normalize_posted_at,
)


@contextmanager
def cancel_connection_on_sigterm(conn: psycopg.Connection) -> Iterator[None]:
    """Cancel and close the active database connection during graceful shutdown."""
    previous_handler = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(signum: int, _frame: FrameType | None) -> None:
        print("non-ml enrichment: received SIGTERM; cancelling database work", flush=True)
        try:
            conn.cancel()
        except Exception:
            print("non-ml enrichment: database cancellation failed", flush=True)
        try:
            conn.rollback()
        except Exception:
            print("non-ml enrichment: database rollback failed", flush=True)
        try:
            conn.close()
        except Exception:
            print("non-ml enrichment: database close failed", flush=True)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def run_non_ml_enrichment_job() -> None:
    print("non-ml enrichment: using configured database", flush=True)
    print(
        "non-ml enrichment: steps run in order: clean content text -> normalize posted_at -> requirements -> years experience -> "
        "security clearance -> normalize location -> prepare ranking feature text",
        flush=True,
    )
    with connect() as conn:
        with cancel_connection_on_sigterm(conn):
            print("non-ml enrichment step 1/8: initializing schema", flush=True)
            initialize_schema(conn)
            print("non-ml enrichment step 2/8: cleaning content text", flush=True)
            clean_content_text.run(conn)
            print("non-ml enrichment step 3/8: normalizing posted_at timestamps", flush=True)
            normalize_posted_at.run(conn)
            print("non-ml enrichment step 4/8: extracting requirements", flush=True)
            add_requirements.run(conn)
            print("non-ml enrichment step 5/8: extracting years of experience", flush=True)
            add_years_experience.run(conn)
            print("non-ml enrichment step 6/8: extracting security clearance requirements", flush=True)
            add_security_clearance.run(conn)
            print("non-ml enrichment step 7/8: normalizing locations", flush=True)
            normalize_location.run(conn)
            print("non-ml enrichment step 8/8: preparing non-model ranking features", flush=True)
            require_export_eligibility_schema(conn)
            add_job_features.run_preparation(conn)
    print("non-ml enrichment: finished successfully", flush=True)


if __name__ == "__main__":
    run_non_ml_enrichment_job()
