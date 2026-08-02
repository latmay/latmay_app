from __future__ import annotations

"""
Normalize source-native posted_at strings into posted_at_utc.

The raw posted_at column is intentionally preserved for debugging. This module
adds a canonical timestamp that export and ranking can use for recency ordering.
"""

import os
import re
import time as monotonic_time
from datetime import date, datetime, time, timedelta, timezone
from typing import Any


def env_positive_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default


POSTED_AT_NORMALIZATION_BATCH_SIZE = env_positive_int("POSTED_AT_NORMALIZATION_BATCH_SIZE", 500)

RELATIVE_POSTED_RE = re.compile(
    r"^\s*(?:posted\s+)?(?:(today|yesterday)|(\d+)\+?\s+(minute|hour|day|week|month|year)s?\s+ago)\s*$",
    re.IGNORECASE,
)

ABSOLUTE_POSTED_AT_FORMATS = (
    "%d-%b-%Y",
    "%B %d, %Y",
)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_absolute_posted_at(value: Any) -> datetime | None:
    if value is None:
        return None

    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()

    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"

    try:
        return ensure_utc(datetime.fromisoformat(text))
    except ValueError:
        pass

    for posted_at_format in ABSOLUTE_POSTED_AT_FORMATS:
        try:
            return datetime.strptime(text, posted_at_format).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    return None


def parse_relative_posted_at(value: Any, reference_time: datetime | None) -> datetime | None:
    if reference_time is None or not isinstance(value, str):
        return None

    match = RELATIVE_POSTED_RE.match(value)
    if not match:
        return None

    reference_time = ensure_utc(reference_time)
    named_day, amount_text, unit = match.groups()
    if named_day:
        amount = 0 if named_day.lower() == "today" else 1
        return reference_time - timedelta(days=amount)

    amount = int(amount_text)
    unit = unit.lower()
    if unit == "minute":
        delta = timedelta(minutes=amount)
    elif unit == "hour":
        delta = timedelta(hours=amount)
    elif unit == "day":
        delta = timedelta(days=amount)
    elif unit == "week":
        delta = timedelta(weeks=amount)
    elif unit == "month":
        delta = timedelta(days=30 * amount)
    elif unit == "year":
        delta = timedelta(days=365 * amount)
    else:
        return None

    return reference_time - delta


def reference_time_for_row(row: dict[str, Any]) -> datetime | None:
    for column in ("fetched_at_utc", "last_seen_at_utc", "first_seen_at_utc", "created_at_utc"):
        value = parse_absolute_posted_at(row.get(column))
        if value is not None:
            return value
    return None


def normalize_posted_at_value(value: Any, *, reference_time: datetime | None = None) -> datetime | None:
    absolute = parse_absolute_posted_at(value)
    if absolute is not None:
        return absolute
    return parse_relative_posted_at(value, reference_time)


def fetch_rows_to_normalize(conn, *, limit: int = 500, after_id: int = 0) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                id,
                posted_at,
                fetched_at_utc,
                last_seen_at_utc,
                first_seen_at_utc,
                created_at_utc
            FROM jobs
            WHERE posted_at IS NOT NULL
              AND NULLIF(btrim(posted_at), '') IS NOT NULL
              AND posted_at_utc IS NULL
              AND posted_at_normalization_failure_reason IS NULL
              AND id > %s
            ORDER BY id
            LIMIT %s
            """,
            (after_id, limit),
        )
        return cur.fetchall()


def update_posted_at_utc(conn, rows: list[dict[str, Any]]) -> tuple[int, int]:
    updates: list[tuple[datetime, int]] = []
    failures: list[tuple[str, int]] = []
    for row in rows:
        normalized = normalize_posted_at_value(
            row.get("posted_at"),
            reference_time=reference_time_for_row(row),
        )
        if normalized is None:
            reason = "unsupported_format_or_missing_relative_reference"
            failures.append((reason, int(row["id"])))
            continue
        updates.append((normalized, int(row["id"])))

    if updates:
        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE jobs
                SET
                    posted_at_utc = %s,
                    posted_at_normalization_status = 'normalized',
                    posted_at_normalization_failure_reason = NULL,
                    posted_at_normalized_at_utc = now()
                WHERE id = %s
                """,
                updates,
            )
    if failures:
        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE jobs
                SET
                    posted_at_normalization_status = 'unparseable',
                    posted_at_normalization_failure_reason = %s,
                    posted_at_normalized_at_utc = now()
                WHERE id = %s
                """,
                failures,
            )

    conn.commit()

    return len(updates), len(failures)


def run(conn) -> int:
    total_checked = 0
    total_updated = 0
    total_unparseable = 0
    batch_number = 0
    after_id = 0

    print(
        f"normalize_posted_at: starting batch_size={POSTED_AT_NORMALIZATION_BATCH_SIZE}",
        flush=True,
    )
    while True:
        select_started = monotonic_time.monotonic()
        rows = fetch_rows_to_normalize(
            conn,
            limit=POSTED_AT_NORMALIZATION_BATCH_SIZE,
            after_id=after_id,
        )
        select_seconds = monotonic_time.monotonic() - select_started
        if not rows:
            break

        batch_number += 1
        after_id = int(rows[-1]["id"])
        print(
            "normalize_posted_at: selected "
            f"batch={batch_number}, rows={len(rows)}, through_id={after_id}, "
            f"duration_seconds={select_seconds:.3f}",
            flush=True,
        )

        update_started = monotonic_time.monotonic()
        updated, unparseable = update_posted_at_utc(conn, rows)
        update_seconds = monotonic_time.monotonic() - update_started
        total_checked += len(rows)
        total_updated += updated
        total_unparseable += unparseable
        print(
            "normalize_posted_at: committed "
            f"batch={batch_number}, checked={len(rows)}, updated={updated}, "
            f"unparseable={unparseable}, duration_seconds={update_seconds:.3f}, "
            f"total_checked={total_checked}",
            flush=True,
        )

    print(
        "normalize_posted_at: finished "
        f"batches={batch_number}, checked={total_checked}, updated={total_updated}, "
        f"unparseable={total_unparseable}",
        flush=True,
    )
    return total_updated
