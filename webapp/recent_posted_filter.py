from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta, timezone
from typing import Any


RECENT_POSTED_HOURS = int(os.environ.get("RECENT_POSTED_HOURS", "48"))
POSTED_AT_COLUMN = "posted_at"
POSTED_AT_UTC_COLUMN = "posted_at_utc"


def parse_posted_at_utc(value: Any) -> datetime | None:
    """
    Parse a source-provided job posting timestamp into UTC.
    """
    if value is None:
        return None

    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()

    if isinstance(value, datetime):
        posted_at = value
    elif isinstance(value, date):
        posted_at = datetime.combine(value, time.min)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None

        if stripped.endswith("Z"):
            stripped = f"{stripped[:-1]}+00:00"

        try:
            posted_at = datetime.fromisoformat(stripped)
        except ValueError:
            return None
    else:
        return None

    if posted_at.tzinfo is None:
        return posted_at.replace(tzinfo=timezone.utc)

    return posted_at.astimezone(timezone.utc)


def is_recent_posted_at(value: Any, *, now_utc: datetime | None = None, hours: int = RECENT_POSTED_HOURS) -> bool:
    posted_at = parse_posted_at_utc(value)
    if posted_at is None:
        return False

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    return now_utc - timedelta(hours=hours) <= posted_at <= now_utc


def format_posted_at_display(value: Any, *, now_utc: datetime | None = None) -> str:
    posted_at = parse_posted_at_utc(value)
    if posted_at is None:
        return ""

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    posted_at = posted_at.astimezone(timezone.utc)
    elapsed = now_utc - posted_at
    if elapsed.total_seconds() < 0:
        return f"{posted_at:%b} {posted_at.day}, {posted_at.year}"

    elapsed_hours = int(elapsed.total_seconds() // 3600)
    elapsed_days = elapsed.days

    if elapsed_hours < 1:
        return "just now"
    if elapsed_hours < 24:
        return "1 hour ago" if elapsed_hours == 1 else f"{elapsed_hours} hours ago"
    if elapsed_days == 1:
        return "yesterday"
    if elapsed_days < 30:
        return f"{elapsed_days} days ago"
    if elapsed_days < 365:
        elapsed_months = max(1, elapsed_days // 30)
        return "1 month ago" if elapsed_months == 1 else f"{elapsed_months} months ago"

    return f"{posted_at:%b} {posted_at.day}, {posted_at.year}"


def filter_recently_posted_jobs_df(
    df: Any,
    *,
    enabled: bool,
    posted_at_column: str = POSTED_AT_COLUMN,
    now_utc: datetime | None = None,
    hours: int = RECENT_POSTED_HOURS,
) -> Any:
    if not enabled:
        return df

    effective_posted_at_column = POSTED_AT_UTC_COLUMN if POSTED_AT_UTC_COLUMN in df.columns else posted_at_column
    if effective_posted_at_column not in df.columns:
        return df.iloc[0:0].copy().reset_index(drop=True)

    mask = df[effective_posted_at_column].map(
        lambda value: is_recent_posted_at(value, now_utc=now_utc, hours=hours)
    )
    return df[mask].reset_index(drop=True)
