from __future__ import annotations

"""
Extract candidate requirements from jobs.content_text and update
jobs.extracted_requirements directly in PostgreSQL.
"""

import os
import re
from datetime import datetime, timezone
from typing import Iterable

from data_pipeline.common.data_quality import length_distribution, log_data_quality

REQUIREMENTS_EXTRACTION_VERSION = "requirements-extractor-v1"
DEFAULT_BATCH_SIZE = 500
DEFAULT_MAX_BATCHES = 1


REQUIREMENT_SECTION_HEADINGS = [
    r"requirements?",
    r"qualifications?",
    r"minimum qualifications?",
    r"basic qualifications?",
    r"preferred qualifications?",
    r"nice to have",
    r"nice-to-have",
    r"what you(?:'|')ll need",
    r"what you will need",
    r"what we(?:'|')re looking for",
    r"what we are looking for",
    r"who you are",
    r"about you",
    r"you have",
    r"skills(?: and experience)?",
    r"experience(?: and skills)?",
    r"candidate profile",
    r"desired skills",
    r"preferred skills",
    r"must have",
]

NON_REQUIREMENT_SECTION_HEADINGS = [
    r"responsibilities?",
    r"what you(?:'|')ll do",
    r"what you will do",
    r"about the role",
    r"about us",
    r"benefits?",
    r"compensation",
    r"salary",
    r"equal opportunity",
    r"eeo",
    r"legal",
    r"privacy",
    r"application",
    r"how to apply",
    r"work environment",
    r"location",
]

REQUIREMENT_CUE_PATTERNS = [
    r"\b\d+\+?\s*(?:years|yrs)\b",
    r"\bexperience with\b",
    r"\bexperience in\b",
    r"\bexperience working with\b",
    r"\bproficiency in\b",
    r"\bproficient in\b",
    r"\bfamiliarity with\b",
    r"\bknowledge of\b",
    r"\bunderstanding of\b",
    r"\bbackground in\b",
    r"\bdegree in\b",
    r"\bbachelor(?:'|')s\b",
    r"\bmaster(?:'|')s\b",
    r"\bph\.?d\.?\b",
    r"\bmust have\b",
    r"\brequired\b",
    r"\bpreferred\b",
    r"\bstrong\b.*\bskills\b",
    r"\bexcellent\b.*\bskills\b",
    r"\bability to\b",
    r"\bable to\b",
    r"\bcomfortable with\b",
    r"\bhands-on\b",
    r"\bexpertise in\b",
    r"\bworking knowledge of\b",
]

HARD_REQUIREMENT_PATTERNS = [
    r"(?:minimum of\s*)?\d+\+?\s*(?:years|yrs)\s+(?:of\s+)?(?:professional\s+)?experience[^.\n]*",
    r"(?:bachelor(?:'|')s|master(?:'|')s|ph\.?d\.?)[^.\n]*",
    r"(?:must be|ability to obtain|active|current)[^.\n]*(?:security clearance|clearance)[^.\n]*",
    r"(?:must be|authorized to|eligible to)[^.\n]*(?:work in|work for|employment in|united states|u\.s\.)[^.\n]*",
    r"(?:experience|proficiency|expertise|knowledge|familiarity|background)\s+(?:with|in|of)[^.\n]*",
]

NOISE_PATTERNS = [
    r"\bequal opportunity\b",
    r"\beeoc?\b",
    r"\bwe are an equal\b",
    r"\bbenefits\b",
    r"\bmedical\b",
    r"\bdental\b",
    r"\bvision\b",
    r"\b401k\b",
    r"\bpaid time off\b",
    r"\bcompensation\b",
    r"\bsalary range\b",
    r"\bprivacy policy\b",
    r"\baccommodation\b",
    r"\bbackground check\b",
]


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_heading(line: str, heading_patterns: Iterable[str]) -> bool:
    normalized = line.strip().lower().strip(":").strip()
    if len(normalized) > 80:
        return False
    return any(re.fullmatch(pattern, normalized, flags=re.IGNORECASE) for pattern in heading_patterns)


def extract_requirement_like_sections(text: str) -> list[str]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    extracted: list[str] = []
    in_requirement_section = False

    for line in lines:
        if is_heading(line, REQUIREMENT_SECTION_HEADINGS):
            in_requirement_section = True
            continue
        if is_heading(line, NON_REQUIREMENT_SECTION_HEADINGS):
            in_requirement_section = False
            continue
        if in_requirement_section:
            extracted.append(line)

    return extracted


def split_into_candidate_units(text: str) -> list[str]:
    rough_lines: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        pieces = re.split(r"\s*[-*•●▪◦]\s*", line)
        rough_lines.extend(piece.strip() for piece in pieces if piece.strip())

    units: list[str] = []
    for line in rough_lines:
        if len(line) > 300:
            units.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", line) if part.strip())
        else:
            units.append(line)
    return units


def contains_requirement_cue(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in REQUIREMENT_CUE_PATTERNS)


def is_obviously_not_requirement(text: str) -> bool:
    lowered = text.lower()
    if any(re.search(pattern, lowered) for pattern in NOISE_PATTERNS):
        return True
    return len(text) > 700


def extract_requirement_like_sentences(text: str) -> list[str]:
    return [
        unit
        for unit in split_into_candidate_units(text)
        if contains_requirement_cue(unit) and not is_obviously_not_requirement(unit)
    ]


def extract_hard_requirement_patterns(text: str) -> list[str]:
    extracted: list[str] = []
    for pattern in HARD_REQUIREMENT_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        extracted.extend(match.strip() for match in matches if match.strip())
    return extracted


def clean_one_requirement_item(item: str) -> str:
    item = item.strip()
    item = re.sub(r"^[\-–—*•●▪◦\s]+", "", item)
    item = re.sub(r"^\d+[\).\s]+", "", item)
    item = re.sub(r"\s+", " ", item)
    return item.strip(" ;")


def normalize_for_deduplication(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_requirement_items(items: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()

    for item in items:
        item = clean_one_requirement_item(item)
        if not item or len(item) < 8 or is_obviously_not_requirement(item):
            continue

        key = normalize_for_deduplication(item)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)

    return cleaned


def extract_candidate_requirements(job_text: str) -> str:
    clean_text = normalize_text(job_text)
    extracted_items: list[str] = []
    extracted_items.extend(extract_requirement_like_sections(clean_text))
    extracted_items.extend(extract_requirement_like_sentences(clean_text))
    extracted_items.extend(extract_hard_requirement_patterns(clean_text))
    return "\n".join(clean_requirement_items(extracted_items))


def get_batch_size() -> int:
    return max(
        1,
        int(
            os.environ.get(
                "NON_ML_BATCH_SIZE",
                os.environ.get("ENRICHMENT_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)),
            )
        ),
    )


def get_max_batches() -> int:
    return max(1, int(os.environ.get("REQUIREMENTS_MAX_BATCHES", str(DEFAULT_MAX_BATCHES))))


def update_requirements(conn, *, only_missing: bool = True) -> int:
    where_clause = "WHERE content_text IS NOT NULL AND btrim(content_text) <> ''"
    params: tuple[str, ...] = ()
    if only_missing:
        where_clause += " AND requirements_extraction_version IS DISTINCT FROM %s"
        params = (REQUIREMENTS_EXTRACTION_VERSION,)

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS candidate_count FROM jobs {where_clause}", params)
        total_candidates = int(cur.fetchone()["candidate_count"])

    batch_size = get_batch_size()
    max_batches = get_max_batches()
    run_limit = batch_size * max_batches
    print(
        f"add_requirements: {total_candidates} rows need requirements extraction; "
        f"processing up to {run_limit} this run "
        f"({max_batches} batches of {batch_size})",
        flush=True,
    )

    updated_count = 0
    processed_count = 0
    extracted_values: list[str] = []
    empty_count = 0
    short_count = 0
    generic_count = 0
    extracted_at = datetime.now(timezone.utc).replace(microsecond=0)
    last_id = 0
    batches_completed = 0

    if total_candidates <= 0:
        conn.commit()
        print(
            "add_requirements: completed run; processed=0/0, extracted=0, empty=0, "
            "batches=0, remaining=0",
            flush=True,
        )
        log_data_quality(
            "requirements",
            selected=0,
            pending_at_start=0,
            remaining=0,
            batches=0,
            updated=0,
            empty=0,
            short=0,
            generic=0,
            missing_after_extract=0,
            **length_distribution([]),
        )
        return 0

    for batch_number in range(1, max_batches + 1):
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, content_text
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

        print(
            f"add_requirements: starting batch {batch_number}/{max_batches}; "
            f"rows={len(rows)}, processed={processed_count}/{total_candidates}",
            flush=True,
        )

        update_rows: list[tuple[object, str, datetime, int]] = []
        batch_updated = 0
        batch_empty = 0
        for batch_processed, row in enumerate(rows, start=1):
            requirements = extract_candidate_requirements(row["content_text"])
            if not requirements:
                empty_count += 1
                batch_empty += 1
                requirements_value = None
            else:
                requirements_value = requirements
                updated_count += 1
                batch_updated += 1
                extracted_values.append(requirements)
                if len(requirements) < 120:
                    short_count += 1
                if len(requirements.splitlines()) <= 1 and len(requirements) < 200:
                    generic_count += 1

            update_rows.append(
                (
                    requirements_value,
                    REQUIREMENTS_EXTRACTION_VERSION,
                    extracted_at,
                    row["id"],
                )
            )
            if batch_processed % 250 == 0 and batch_processed < len(rows):
                print(
                    f"add_requirements: batch {batch_number}/{max_batches} progress "
                    f"{batch_processed}/{len(rows)}",
                    flush=True,
                )

        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE jobs
                SET extracted_requirements = %s,
                    requirements_extraction_version = %s,
                    requirements_extracted_at_utc = %s
                WHERE id = %s
                """,
                update_rows,
            )
        conn.commit()

        processed_count += len(rows)
        last_id = int(rows[-1]["id"])
        batches_completed += 1
        print(
            f"add_requirements: committed batch {batch_number}/{max_batches}; "
            f"batch_rows={len(rows)}, extracted={batch_updated}, empty={batch_empty}, "
            f"processed={processed_count}/{total_candidates}",
            flush=True,
        )

        if len(rows) < batch_size:
            break

    remaining = max(0, total_candidates - processed_count)
    print(
        f"add_requirements: completed run; processed={processed_count}/{total_candidates}, "
        f"extracted={updated_count}, empty={empty_count}, batches={batches_completed}, "
        f"remaining={remaining}",
        flush=True,
    )
    log_data_quality(
        "requirements",
        selected=processed_count,
        pending_at_start=total_candidates,
        remaining=remaining,
        batches=batches_completed,
        updated=updated_count,
        empty=empty_count,
        short=short_count,
        generic=generic_count,
        missing_after_extract=empty_count,
        **length_distribution(extracted_values),
    )
    return updated_count


def run(conn) -> int:
    return update_requirements(conn)


if __name__ == "__main__":
    from data_pipeline.common.db import connect

    with connect() as connection:
        run(connection)
