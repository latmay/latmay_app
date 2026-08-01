from __future__ import annotations

import os

_ENABLED = os.environ.get("INGESTION_TIMING_DEBUG", "").strip().lower() in ("1", "true", "yes")


def log_timing(ats: str, company: str, step: str, elapsed_s: float) -> None:
    if not _ENABLED:
        return
    print(f"TIMING ats={ats} company={company} step={step} elapsed_s={elapsed_s:.3f}s", flush=True)
