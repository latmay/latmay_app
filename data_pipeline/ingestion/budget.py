from __future__ import annotations

import os
import time


def ingestion_time_budget_seconds() -> float | None:
    raw_value = os.environ.get("INGESTION_TIME_BUDGET_SECONDS", "").strip()
    if not raw_value:
        return None
    try:
        seconds = float(raw_value)
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def ingestion_budget_started_at() -> float:
    return time.monotonic()


def should_stop_for_ingestion_budget(source_type: str, started_at: float, completed_sources: int) -> bool:
    budget_seconds = ingestion_time_budget_seconds()
    if budget_seconds is None:
        return False

    elapsed_seconds = time.monotonic() - started_at
    if elapsed_seconds < budget_seconds:
        return False

    print(
        f"{source_type}: stopping before next source because INGESTION_TIME_BUDGET_SECONDS="
        f"{budget_seconds:.0f} was reached; elapsed_seconds={elapsed_seconds:.1f}; "
        f"completed_sources={completed_sources}",
        flush=True,
    )
    return True
