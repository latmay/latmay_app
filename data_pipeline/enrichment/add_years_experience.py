from __future__ import annotations

"""
Extract years-of-experience metadata and update jobs directly in PostgreSQL.

Pure regex extraction functions are kept separate from database update logic.
The extraction order is:
1. content_text
2. extracted_requirements
3. looser requirements-only fallback
"""

import os
import re
from datetime import datetime, timezone
from typing import Any

from data_pipeline.common.data_quality import log_data_quality, numeric_distribution

YOE_EXTRACTION_VERSION = "yoe-extractor-v1"
DEFAULT_YOE_BATCH_SIZE = 500


YEARS_EXPERIENCE_PATTERNS = [
    re.compile(
        r"(?P<min_years>\d+(?:\.\d+)?)\s*(?:-|to|–|—)\s*(?P<max_years>\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:minimum|min\.?)\s+(?:of\s+)?(?P<years>\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", re.IGNORECASE),
    re.compile(r"at\s+least\s+(?P<years>\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", re.IGNORECASE),
    re.compile(r"requires?\s+(?:at\s+least\s+)?(?P<years>\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)", re.IGNORECASE),
    re.compile(
        r"(?P<years>\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+of\s+"
        r"(?:relevant\s+|related\s+|professional\s+|industry\s+|work\s+|hands[-\s]?on\s+|clinical\s+practice\s+|product\s+management\s+)?experience",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<years>\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+"
        r"(?:of\s+)?(?:clinical\s+practice|product\s+management|account\s+management|relationship\s+management|partnership\s+development|leadership|engineering|sales|marketing|operations)?\s*experience",
        re.IGNORECASE,
    ),
    re.compile(r"(?P<years>\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:in|with)\b", re.IGNORECASE),
    re.compile(r"(?P<years>\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:required|preferred)", re.IGNORECASE),
    re.compile(r"(?P<years>\d+(?:\.\d+)?)\s*(?:or\s+more|or\s+more\s+years)\b", re.IGNORECASE),
]

REQUIREMENTS_PLUS_FALLBACK_PATTERN = re.compile(r"(?P<years>\d+(?:\.\d+)?)\s*\+", re.IGNORECASE)
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+|\n+|(?:\s*[;•]\s*)")


def empty_result() -> dict[str, Any]:
    return {
        "years_experience_raw": None,
        "min_years_experience": None,
        "max_years_experience": None,
        "experience_type": None,
        "evidence_text": None,
    }


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_missing_extraction(result: dict[str, Any]) -> bool:
    return result.get("min_years_experience") is None


def sentence_candidates(text: str) -> list[str]:
    chunks = SENTENCE_SPLIT_REGEX.split(text)
    candidates: list[str] = []

    for chunk in chunks:
        clean = normalize_whitespace(chunk)
        if not clean:
            continue

        lowered = clean.lower()
        if ("year" in lowered or "yrs" in lowered) and (
            "experience" in lowered
            or "required" in lowered
            or "preferred" in lowered
            or "minimum" in lowered
            or "at least" in lowered
            or "requirement" in lowered
            or "qualification" in lowered
            or "practice" in lowered
            or "management" in lowered
            or "leadership" in lowered
            or "with " in lowered
            or " in " in lowered
        ):
            candidates.append(clean)

    return candidates


def infer_experience_type(candidate: str) -> str:
    lowered = candidate.lower()
    if any(word in lowered for word in ["required", "must have", "minimum", "at least", "requirement"]):
        return "required"
    if "preferred" in lowered or "nice to have" in lowered or "plus" in lowered:
        return "preferred"
    return "unspecified"


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.rstrip("+"))
    except ValueError:
        return None


def result_from_value(value: str, candidate: str) -> dict[str, Any]:
    min_years = None
    max_years = None

    if "-" in value:
        min_years, max_years = value.split("-", 1)
    else:
        num_match = re.search(r"\d+(?:\.\d+)?", value)
        if num_match:
            min_years = num_match.group(0)

    return {
        "years_experience_raw": value,
        "min_years_experience": parse_float(min_years),
        "max_years_experience": parse_float(max_years),
        "experience_type": infer_experience_type(candidate),
        "evidence_text": candidate,
    }


def extract_years_experience_from_text(text_value: Any) -> dict[str, Any]:
    if text_value is None:
        return empty_result()

    text = str(text_value)
    candidates = sentence_candidates(text) or [normalize_whitespace(text)]

    for candidate in candidates:
        for pattern in YEARS_EXPERIENCE_PATTERNS:
            match = pattern.search(candidate)
            if not match:
                continue

            if "min_years" in match.groupdict() and "max_years" in match.groupdict():
                value = f"{match.group('min_years')}-{match.group('max_years')}"
            else:
                value = match.group("years")
                if "+" in match.group(0) and not value.endswith("+"):
                    value = f"{value}+"

            return result_from_value(value=value, candidate=candidate)

    return empty_result()


def extract_years_experience_from_requirements_fallback(requirements_text: Any) -> dict[str, Any]:
    if requirements_text is None:
        return empty_result()

    for chunk in SENTENCE_SPLIT_REGEX.split(str(requirements_text)):
        candidate = normalize_whitespace(chunk)
        if not candidate:
            continue

        match = REQUIREMENTS_PLUS_FALLBACK_PATTERN.search(candidate)
        if match:
            return result_from_value(value=f"{match.group('years')}+", candidate=candidate)

    return empty_result()


def extract_best_years_experience(content_text: Any, extracted_requirements: Any) -> dict[str, Any]:
    result = extract_years_experience_from_text(content_text)
    if not is_missing_extraction(result):
        return result

    result = extract_years_experience_from_text(extracted_requirements)
    if not is_missing_extraction(result):
        return result

    return extract_years_experience_from_requirements_fallback(extracted_requirements)


def get_batch_size() -> int:
    return max(1, int(os.environ.get("YOE_BATCH_SIZE", str(DEFAULT_YOE_BATCH_SIZE))))


def update_years_experience(conn, *, only_missing: bool = True) -> int:
    where_clause = "WHERE (content_text IS NOT NULL OR extracted_requirements IS NOT NULL)"
    params: tuple[str, ...] = ()
    if only_missing:
        where_clause += " AND yoe_extraction_version IS DISTINCT FROM %s"
        params = (YOE_EXTRACTION_VERSION,)

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS candidate_count FROM jobs {where_clause}", params)
        total_candidates = int(cur.fetchone()["candidate_count"])

    batch_size = get_batch_size()
    print(
        f"add_years_experience: {total_candidates} rows need YOE extraction; "
        f"processing in batches of {batch_size} until complete",
        flush=True,
    )

    updated_count = 0
    processed_count = 0
    extracted_min_years: list[float] = []
    extracted_at = datetime.now(timezone.utc).replace(microsecond=0)
    last_id = 0
    batch_number = 0

    if total_candidates <= 0:
        conn.commit()
        print(
            "add_years_experience: completed 0 rows in 0 batches; extracted=0, missing=0",
            flush=True,
        )
        log_data_quality(
            "yoe",
            selected=0,
            updated=0,
            has_yoe=0,
            missing_yoe=0,
            batches=0,
            **numeric_distribution([]),
        )
        return 0

    while True:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, content_text, extracted_requirements
                FROM jobs
                {where_clause}
                  AND id > %s
                ORDER BY id
                LIMIT %s
                """,
                (*params, last_id, batch_size),
            )
            rows = cur.fetchall()

        if not rows:
            break

        batch_number += 1
        print(
            f"add_years_experience: starting batch {batch_number}; "
            f"rows={len(rows)}, processed={processed_count}/{total_candidates}",
            flush=True,
        )
        update_rows: list[tuple[Any, ...]] = []
        batch_extracted = 0
        batch_missing = 0
        for batch_processed, row in enumerate(rows, start=1):
            result = extract_best_years_experience(row["content_text"], row["extracted_requirements"])
            if is_missing_extraction(result):
                batch_missing += 1
                update_rows.append(
                    (
                        None,
                        None,
                        None,
                        None,
                        None,
                        YOE_EXTRACTION_VERSION,
                        extracted_at,
                        row["id"],
                    )
                )
            else:
                update_rows.append(
                    (
                        result["years_experience_raw"],
                        result["min_years_experience"],
                        result["max_years_experience"],
                        result["experience_type"],
                        result["evidence_text"],
                        YOE_EXTRACTION_VERSION,
                        extracted_at,
                        row["id"],
                    )
                )
                updated_count += 1
                batch_extracted += 1
                extracted_min_years.append(float(result["min_years_experience"]))

            if batch_processed % 250 == 0 and batch_processed < len(rows):
                print(
                    f"add_years_experience: batch {batch_number} progress "
                    f"{batch_processed}/{len(rows)}",
                    flush=True,
                )

        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE jobs
                SET years_experience_raw = %s,
                    min_years_experience = %s,
                    max_years_experience = %s,
                    experience_type = %s,
                    evidence_text = %s,
                    yoe_extraction_version = %s,
                    yoe_extracted_at_utc = %s
                WHERE id = %s
                """,
                update_rows,
            )
        conn.commit()
        processed_count += len(rows)
        last_id = int(rows[-1]["id"])
        print(
            f"add_years_experience: committed batch {batch_number}; "
            f"batch_rows={len(rows)}, extracted={batch_extracted}, missing={batch_missing}, "
            f"processed={processed_count}/{total_candidates}",
            flush=True,
        )

    print(
        f"add_years_experience: completed {processed_count} rows in {batch_number} batches; "
        f"extracted={updated_count}, missing={processed_count - updated_count}",
        flush=True,
    )
    log_data_quality(
        "yoe",
        selected=processed_count,
        updated=updated_count,
        has_yoe=len(extracted_min_years),
        missing_yoe=processed_count - len(extracted_min_years),
        batches=batch_number,
        **numeric_distribution(extracted_min_years),
    )
    return updated_count


def run(conn) -> int:
    return update_years_experience(conn)


if __name__ == "__main__":
    from data_pipeline.common.db import connect

    with connect() as connection:
        run(connection)
