from __future__ import annotations

"""Create a conservative, versioned cleaned copy of jobs.content_text."""

import html
import os
import re
import unicodedata
from typing import Any

import ftfy


CONTENT_TEXT_CLEAN_VERSION = 1
DEFAULT_BATCH_SIZE = 500
DEFAULT_MAX_BATCHES = 1000


def _positive_int_from_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default


def get_batch_size() -> int:
    return _positive_int_from_env(
        "CONTENT_TEXT_CLEAN_BATCH_SIZE",
        _positive_int_from_env("NON_ML_BATCH_SIZE", DEFAULT_BATCH_SIZE),
    )


def get_max_batches() -> int:
    return _positive_int_from_env("CONTENT_TEXT_CLEAN_MAX_BATCHES", DEFAULT_MAX_BATCHES)


def clean_text(value: str | None) -> str | None:
    """Normalize text while retaining meaningful line and paragraph boundaries."""
    if value is None:
        return None

    text = html.unescape(value)
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\u201c": '"',
                "\u201d": '"',
            }
        )
    )
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_pending_content_text(conn: Any) -> int:
    """Clean pending rows in committed, resumable batches."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS candidate_count
            FROM jobs
            WHERE content_text IS NOT NULL
              AND content_text_clean_version IS DISTINCT FROM %s
            """,
            (CONTENT_TEXT_CLEAN_VERSION,),
        )
        total_candidates = int(cur.fetchone()["candidate_count"])

    batch_size = get_batch_size()
    max_batches = get_max_batches()
    run_limit = batch_size * max_batches
    print(
        f"clean_content_text: {total_candidates} rows need cleaning; "
        f"processing up to {run_limit} this run "
        f"({max_batches} batches of {batch_size})",
        flush=True,
    )

    if total_candidates == 0:
        conn.commit()
        print("clean_content_text: completed run; processed=0/0, batches=0, remaining=0", flush=True)
        return 0

    processed = 0
    batches_completed = 0
    last_id = 0
    for batch_number in range(1, max_batches + 1):
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content_text
                FROM jobs
                WHERE content_text IS NOT NULL
                  AND content_text_clean_version IS DISTINCT FROM %s
                  AND id > %s
                ORDER BY id
                LIMIT %s
                """,
                (CONTENT_TEXT_CLEAN_VERSION, last_id, batch_size),
            )
            rows = cur.fetchall()

        if not rows:
            break

        updates = [
            (clean_text(row["content_text"]), CONTENT_TEXT_CLEAN_VERSION, row["id"])
            for row in rows
        ]
        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE jobs
                SET content_text_clean = %s,
                    content_text_clean_version = %s
                WHERE id = %s
                """,
                updates,
            )
        conn.commit()

        processed += len(rows)
        batches_completed += 1
        last_id = int(rows[-1]["id"])
        print(
            f"clean_content_text: committed batch {batch_number}/{max_batches}; "
            f"batch_rows={len(rows)}, processed={processed}/{total_candidates}",
            flush=True,
        )
        if len(rows) < batch_size:
            break

    remaining = max(0, total_candidates - processed)
    print(
        f"clean_content_text: completed run; processed={processed}/{total_candidates}, "
        f"batches={batches_completed}, remaining={remaining}",
        flush=True,
    )
    return processed


def run(conn: Any) -> int:
    return clean_pending_content_text(conn)


if __name__ == "__main__":
    from data_pipeline.common.db import connect

    with connect() as connection:
        run(connection)
