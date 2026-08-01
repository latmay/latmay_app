from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Sequence


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "unknown"
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if isinstance(value, dict):
        items = ",".join(f"{key}:{value[key]}" for key in sorted(value))
        return "{" + items + "}"
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def log_data_quality(stage: str, **fields: Any) -> None:
    parts = [f"DATA_QUALITY stage={stage}"]
    parts.extend(f"{key}={format_value(value)}" for key, value in fields.items())
    print(" ".join(parts), flush=True)


def http_status_code_from_exception(exc: BaseException) -> int | None:
    direct_status_code = getattr(exc, "http_status_code", None)
    if isinstance(direct_status_code, int):
        return direct_status_code
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return int(status_code) if isinstance(status_code, int) else None


def percentile(values: Sequence[float], percent: float) -> float | None:
    numbers = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not numbers:
        return None
    if len(numbers) == 1:
        return numbers[0]
    position = (len(numbers) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return numbers[int(position)]
    weight = position - lower
    return numbers[lower] * (1 - weight) + numbers[upper] * weight


def numeric_distribution(values: Iterable[Any]) -> dict[str, Any]:
    numbers: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numbers.append(number)

    if not numbers:
        return {"min": None, "p50": None, "p90": None, "max": None}

    return {
        "min": min(numbers),
        "p50": percentile(numbers, 50),
        "p90": percentile(numbers, 90),
        "max": max(numbers),
    }


def length_distribution(values: Iterable[Any], *, prefix: str = "") -> dict[str, Any]:
    lengths = [len(str(value).strip()) for value in values if str(value or "").strip()]
    if not lengths:
        return {
            f"{prefix}avg_chars": None,
            f"{prefix}p10_chars": None,
            f"{prefix}p50_chars": None,
            f"{prefix}p90_chars": None,
        }
    return {
        f"{prefix}avg_chars": sum(lengths) / len(lengths),
        f"{prefix}p10_chars": percentile(lengths, 10),
        f"{prefix}p50_chars": percentile(lengths, 50),
        f"{prefix}p90_chars": percentile(lengths, 90),
    }


def count_distribution(values: Iterable[int], *, prefix: str = "") -> dict[str, Any]:
    counts = [int(value) for value in values]
    if not counts:
        return {
            f"{prefix}avg": None,
            f"{prefix}p10": None,
            f"{prefix}p50": None,
            f"{prefix}p90": None,
        }
    return {
        f"{prefix}avg": sum(counts) / len(counts),
        f"{prefix}p10": percentile(counts, 10),
        f"{prefix}p50": percentile(counts, 50),
        f"{prefix}p90": percentile(counts, 90),
    }


def count_blank(records: Iterable[dict[str, Any]], field: str) -> int:
    return sum(1 for record in records if not str(record.get(field) or "").strip())


def duplicate_count(values: Iterable[Any]) -> int:
    cleaned = [str(value).strip() for value in values if str(value or "").strip()]
    counts = Counter(cleaned)
    return sum(count - 1 for count in counts.values() if count > 1)


def count_by_source_type(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(record.get("source_type") or "unknown") for record in records)
    return dict(sorted(counts.items()))
