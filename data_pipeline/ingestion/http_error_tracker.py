from __future__ import annotations

import os
from collections import Counter


DEFAULT_MAX_429_ERRORS_PER_ATS = 4
MAX_429_ERRORS_ENV = "INGESTION_MAX_429_ERRORS_PER_ATS"


class AtsHttp429LimitReached(RuntimeError):
    def __init__(self, ats_name: str, count: int, limit: int) -> None:
        super().__init__(f"{ats_name} stopped after {count} HTTP 429 errors; {MAX_429_ERRORS_ENV}={limit}")
        self.ats_name = ats_name
        self.count = count
        self.limit = limit
        self.http_status_code = 429


def max_429_errors_per_ats() -> int:
    raw_value = os.environ.get(MAX_429_ERRORS_ENV, str(DEFAULT_MAX_429_ERRORS_PER_ATS))
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_429_ERRORS_PER_ATS
    return value


class AtsHttpErrorTracker:
    def __init__(self, ats_name: str) -> None:
        self.ats_name = ats_name
        self.max_429_errors = max_429_errors_per_ats()
        self.counts: Counter[str] = Counter()
        self._printed_stop = False

    def record(self, status_code: int | None) -> bool:
        key = str(status_code) if isinstance(status_code, int) else "unknown"
        self.counts[key] += 1
        if status_code == 429 and self.max_429_errors > 0 and self.counts["429"] >= self.max_429_errors:
            self.print_429_stop()
            return True
        return False

    def record_or_raise(self, status_code: int | None) -> None:
        if self.record(status_code):
            raise AtsHttp429LimitReached(self.ats_name, self.counts["429"], self.max_429_errors)

    def print_429_stop(self) -> None:
        if self._printed_stop:
            return
        print(
            f"{self.ats_name}: stopping early after {self.counts['429']} HTTP 429 errors; "
            f"{MAX_429_ERRORS_ENV}={self.max_429_errors}",
            flush=True,
        )
        self._printed_stop = True

    def print_summary(self) -> None:
        if not self.counts:
            summary = "none"
        else:
            keys = sorted((key for key in self.counts if key != "unknown"), key=lambda value: int(value))
            if "unknown" in self.counts:
                keys.append("unknown")
            summary = ", ".join(f"{key}={self.counts[key]}" for key in keys)
        print(f"{self.ats_name}: http error summary: {summary}", flush=True)
