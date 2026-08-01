from __future__ import annotations

"""
Extract U.S. security clearance requirements and update jobs directly in PostgreSQL.

Pure regex extraction functions are kept separate from database update logic.
"""

import os
import re
from datetime import datetime, timezone
from typing import Any

from data_pipeline.common.data_quality import log_data_quality

CLEARANCE_EXTRACTION_VERSION = "clearance-extractor-v1"
DEFAULT_CLEARANCE_BATCH_SIZE = 500


SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+|\n+|(?:\s*[;•]\s*)")

CLEARANCE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "TS/SCI with Polygraph",
        re.compile(r"\b(?:ts\s*/\s*sci|top secret\s*/\s*sci)\b[^.\n;]{0,80}\bpoly(?:graph)?\b", re.IGNORECASE),
    ),
    (
        "Top Secret with Polygraph",
        re.compile(r"\btop secret\b[^.\n;]{0,80}\bpoly(?:graph)?\b", re.IGNORECASE),
    ),
    (
        "TS/SCI",
        re.compile(r"\b(?:ts\s*/\s*sci|top secret\s*/\s*sci)\b", re.IGNORECASE),
    ),
    (
        "Top Secret",
        re.compile(r"\b(?:top secret|ts)\s+(?:security\s+)?clearance\b", re.IGNORECASE),
    ),
    (
        "Secret",
        re.compile(r"\bsecret\s+(?:security\s+)?clearance\b", re.IGNORECASE),
    ),
    (
        "SCI",
        re.compile(r"\bsci\b", re.IGNORECASE),
    ),
    (
        "Polygraph",
        re.compile(r"\bpoly(?:graph)?\b", re.IGNORECASE),
    ),
    (
        "Public Trust",
        re.compile(r"\bpublic trust\b", re.IGNORECASE),
    ),
    (
        "Ability to Obtain Clearance",
        re.compile(
            r"\b(?:(?:ability|eligible|eligibility)\s+to\s+obtain|able\s+to\s+obtain)\b[^.\n;]{0,80}\bclearance\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Security Clearance",
        re.compile(
            r"\b(?:security clearance|active clearance|current clearance|u\.?s\.?\s+government clearance)\b",
            re.IGNORECASE,
        ),
    ),
]

CLEARANCE_CONTEXT_PATTERN = re.compile(
    r"\b(?:clearance|security clearance|public trust|polygraph|ts\s*/\s*sci|top secret|secret clearance|sci)\b",
    re.IGNORECASE,
)

NEGATED_CLEARANCE_PATTERN = re.compile(
    r"\b(?:no|not|without)\s+(?:security\s+)?clearance\s+(?:required|needed|necessary)\b",
    re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sentence_candidates(*texts: Any) -> list[str]:
    candidates: list[str] = []
    for text_value in texts:
        if text_value is None:
            continue
        for chunk in SENTENCE_SPLIT_REGEX.split(str(text_value)):
            candidate = normalize_whitespace(chunk)
            if candidate:
                candidates.append(candidate)
    return candidates


def infer_clearance_type(text: str) -> str | None:
    for clearance_type, pattern in CLEARANCE_PATTERNS:
        if pattern.search(text):
            return clearance_type
    return None


def extract_security_clearance(content_text: Any, extracted_requirements: Any = None) -> dict[str, Any]:
    for candidate in sentence_candidates(content_text, extracted_requirements):
        if NEGATED_CLEARANCE_PATTERN.search(candidate):
            continue
        if not CLEARANCE_CONTEXT_PATTERN.search(candidate):
            continue

        clearance_type = infer_clearance_type(candidate)
        if clearance_type is None:
            continue

        return {
            "requires_clearance": True,
            "clearance_type": clearance_type,
            "clearance_evidence_text": candidate[:500],
        }

    return {
        "requires_clearance": False,
        "clearance_type": None,
        "clearance_evidence_text": None,
    }


def get_batch_size() -> int:
    return max(1, int(os.environ.get("CLEARANCE_BATCH_SIZE", str(DEFAULT_CLEARANCE_BATCH_SIZE))))


def update_security_clearance(conn, *, only_missing: bool = True) -> int:
    where_clause = "WHERE (content_text IS NOT NULL OR extracted_requirements IS NOT NULL)"
    params: tuple[str, ...] = ()
    if only_missing:
        where_clause += " AND clearance_extraction_version IS DISTINCT FROM %s"
        params = (CLEARANCE_EXTRACTION_VERSION,)

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS candidate_count FROM jobs {where_clause}", params)
        total_candidates = int(cur.fetchone()["candidate_count"])

    batch_size = get_batch_size()
    print(
        f"add_security_clearance: {total_candidates} rows need clearance extraction; "
        f"processing in batches of {batch_size} until complete",
        flush=True,
    )

    updated_count = 0
    processed_count = 0
    requires_clearance_count = 0
    extracted_at = datetime.now(timezone.utc).replace(microsecond=0)
    last_id = 0
    batch_number = 0

    if total_candidates <= 0:
        conn.commit()
        print(
            "add_security_clearance: completed 0 rows in 0 batches; requires_clearance=0",
            flush=True,
        )
        log_data_quality(
            "security_clearance",
            selected=0,
            updated=0,
            requires_clearance=0,
            no_clearance=0,
            batches=0,
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
            f"add_security_clearance: starting batch {batch_number}; "
            f"rows={len(rows)}, processed={processed_count}/{total_candidates}",
            flush=True,
        )
        update_rows: list[tuple[Any, ...]] = []
        batch_requires_clearance = 0
        for batch_processed, row in enumerate(rows, start=1):
            result = extract_security_clearance(row["content_text"], row["extracted_requirements"])
            if result["requires_clearance"]:
                requires_clearance_count += 1
                batch_requires_clearance += 1
            update_rows.append(
                (
                    result["requires_clearance"],
                    result["clearance_type"],
                    result["clearance_evidence_text"],
                    CLEARANCE_EXTRACTION_VERSION,
                    extracted_at,
                    row["id"],
                )
            )
            if batch_processed % 250 == 0 and batch_processed < len(rows):
                print(
                    f"add_security_clearance: batch {batch_number} progress "
                    f"{batch_processed}/{len(rows)}",
                    flush=True,
                )

        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE jobs
                SET requires_clearance = %s,
                    clearance_type = %s,
                    clearance_evidence_text = %s,
                    clearance_extraction_version = %s,
                    clearance_extracted_at_utc = %s
                WHERE id = %s
                """,
                update_rows,
            )
        conn.commit()
        updated_count += len(rows)
        processed_count += len(rows)
        last_id = int(rows[-1]["id"])
        print(
            f"add_security_clearance: committed batch {batch_number}; "
            f"batch_rows={len(rows)}, requires_clearance={batch_requires_clearance}, "
            f"processed={processed_count}/{total_candidates}",
            flush=True,
        )

    print(
        f"add_security_clearance: completed {processed_count} rows in {batch_number} batches; "
        f"requires_clearance={requires_clearance_count}",
        flush=True,
    )
    log_data_quality(
        "security_clearance",
        selected=processed_count,
        updated=updated_count,
        requires_clearance=requires_clearance_count,
        no_clearance=processed_count - requires_clearance_count,
        batches=batch_number,
    )
    return updated_count


def run(conn) -> int:
    return update_security_clearance(conn)


if __name__ == "__main__":
    from data_pipeline.common.db import connect

    with connect() as connection:
        run(connection)
