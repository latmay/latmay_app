from __future__ import annotations

"""Precompute reusable job-side ranking features in PostgreSQL.

This enrichment is idempotent: by default it updates rows with missing or stale
feature metadata, and leaves already-current rows alone.
"""

import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
from psycopg.types.json import Jsonb
from sklearn.feature_extraction.text import TfidfVectorizer

from data_pipeline.common.data_quality import count_distribution, length_distribution, log_data_quality
from data_pipeline.common.model_loader import (
    get_enrichment_version,
    get_minilm_model_name,
    get_minilm_model_revision,
    load_minilm_model,
)
from data_pipeline.enrichment.feature_text import (
    build_stopword_set,
    maybe_remove_frequent_words,
    maybe_remove_stopwords,
    phrase_chunks,
    tokenize_text,
)


DEFAULT_BATCH_SIZE = 500
DEFAULT_MAX_BATCHES = 1
DEFAULT_MAX_WORDS_PER_JOB = 50
DEFAULT_TOP_TFIDF_FRACTION = 0.25
DEFAULT_EMBEDDING_CLEANUP_RETENTION_DAYS = 30
DEFAULT_EMBEDDING_CLEANUP_LIMIT = 3000
DEFAULT_RECENT_JOBS_EXPORT_LIMIT = 1000
DEFAULT_RANKING_ARTIFACT_SHARD_COUNT = 1
DEFAULT_RANKING_ARTIFACT_JOBS_PER_SHARD = 1000


def get_batch_size() -> int:
    return max(1, int(os.environ.get("ENRICHMENT_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))))


def get_non_ml_batch_size() -> int:
    return max(1, int(os.environ.get("NON_ML_BATCH_SIZE", os.environ.get("ENRICHMENT_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))))


def get_max_batches() -> int:
    return max(1, int(os.environ.get("ENRICHMENT_MAX_BATCHES", str(DEFAULT_MAX_BATCHES))))


def get_ml_max_batches() -> int:
    return max(1, int(os.environ.get("ML_ENRICHMENT_MAX_BATCHES", str(DEFAULT_MAX_BATCHES))))


def get_max_words_per_job() -> int:
    return max(1, int(os.environ.get("JOB_FEATURE_MAX_WORDS", str(DEFAULT_MAX_WORDS_PER_JOB))))


def should_embed_content_text() -> bool:
    return os.environ.get("EMBED_CONTENT_TEXT", "false").lower() in {"1", "true", "yes"}


def should_cleanup_embeddings() -> bool:
    return os.environ.get("ENABLE_EMBEDDING_CLEANUP", "false").lower() in {"1", "true", "yes", "on"}


def get_positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name, str(default)).strip()
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return parsed


def get_nonnegative_int_env(name: str, default: int) -> int:
    value = os.environ.get(name, str(default)).strip()
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed < 0:
        raise ValueError(f"{name} cannot be negative.")
    return parsed


def get_embedding_cleanup_retention_days() -> int:
    return get_nonnegative_int_env("EMBEDDING_CLEANUP_RETENTION_DAYS", DEFAULT_EMBEDDING_CLEANUP_RETENTION_DAYS)


def get_embedding_cleanup_limit() -> int:
    return get_positive_int_env("EMBEDDING_CLEANUP_LIMIT", DEFAULT_EMBEDDING_CLEANUP_LIMIT)


def get_export_embedding_keep_count() -> int:
    recent_export_limit = get_positive_int_env("RECENT_JOBS_EXPORT_LIMIT", DEFAULT_RECENT_JOBS_EXPORT_LIMIT)
    shard_count = get_positive_int_env("RANKING_ARTIFACT_SHARD_COUNT", DEFAULT_RANKING_ARTIFACT_SHARD_COUNT)
    jobs_per_shard = get_positive_int_env(
        "RANKING_ARTIFACT_JOBS_PER_SHARD",
        DEFAULT_RANKING_ARTIFACT_JOBS_PER_SHARD,
    )
    return max(recent_export_limit, shard_count * jobs_per_shard)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_title_requirements_text(row: dict[str, Any]) -> str:
    title = clean_text(row.get("title"))
    requirements = clean_text(row.get("extracted_requirements"))
    content = clean_text(row.get("content_text"))
    body = requirements or content
    return f"Job title: {title}\nRequirements and description: {body}".strip()


def preprocess_words(text: str) -> list[str]:
    stopwords = build_stopword_set(
        custom_stopwords={"preferred", "required", "qualification", "qualifications"},
        use_stopword_filter=True,
    )
    tokens = tokenize_text(text)
    tokens = maybe_remove_stopwords(tokens, stopword_set=stopwords, use_stopword_filter=True)
    tokens = maybe_remove_frequent_words(tokens, use_frequent_word_filter=False, max_count=None)
    return tokens


def select_tfidf_words(texts: list[str], *, max_words_per_job: int) -> list[list[str]]:
    tokenized_docs = [preprocess_words(text) for text in texts]
    joined_docs = [" ".join(tokens) for tokens in tokenized_docs]

    if not any(joined_docs):
        return [tokens[:max_words_per_job] for tokens in tokenized_docs]

    try:
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[a-z][a-z']+\b")
        matrix = vectorizer.fit_transform(joined_docs)
        feature_names = np.asarray(vectorizer.get_feature_names_out())
    except ValueError:
        return [tokens[:max_words_per_job] for tokens in tokenized_docs]

    selected: list[list[str]] = []
    for row_idx, tokens in enumerate(tokenized_docs):
        if not tokens:
            selected.append([])
            continue

        row = matrix.getrow(row_idx)
        if row.nnz == 0:
            selected.append(tokens[:max_words_per_job])
            continue

        n_keep = max(1, int(row.nnz * DEFAULT_TOP_TFIDF_FRACTION))
        n_keep = min(max_words_per_job, n_keep)
        ranked_indices = row.indices[np.argsort(row.data)[::-1]][:n_keep]
        words = [str(word) for word in feature_names[ranked_indices]]
        selected.append(words)

    return selected


def embed_unique_texts(
    model,
    grouped_texts: list[list[str]],
    *,
    batch_size: int = 64,
    normalize_embeddings: bool = False,
) -> list[list[list[float]]]:
    unique_texts = sorted({text for group in grouped_texts for text in group if text})
    if not unique_texts:
        return [[] for _ in grouped_texts]

    encoded = model.encode(
        unique_texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )
    lookup = {
        text: np.asarray(vector, dtype=np.float32).tolist()
        for text, vector in zip(unique_texts, encoded)
    }
    return [[lookup[text] for text in group if text in lookup] for group in grouped_texts]


def count_rows_with_complete_embeddings(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS row_count
            FROM jobs
            WHERE title_requirements_embedding IS NOT NULL
              AND job_selected_words IS NOT NULL
              AND job_selected_word_embeddings IS NOT NULL
              AND job_phrase_chunks IS NOT NULL
              AND job_phrase_chunk_embeddings IS NOT NULL
            """
        )
        row = cur.fetchone()
    return int(row["row_count"])


def cleanup_old_embeddings(conn) -> int:
    if not should_cleanup_embeddings():
        print("embedding cleanup: disabled by ENABLE_EMBEDDING_CLEANUP=false", flush=True)
        return 0

    retention_days = get_embedding_cleanup_retention_days()
    cleanup_limit = get_embedding_cleanup_limit()
    export_keep_count = get_export_embedding_keep_count()
    embedded_count = count_rows_with_complete_embeddings(conn)
    removable_budget = max(0, embedded_count - export_keep_count)
    cleanup_count = min(cleanup_limit, removable_budget)

    print(
        "embedding cleanup: start "
        f"embedded_jobs={embedded_count}, export_keep_count={export_keep_count}, "
        f"retention_days={retention_days}, limit={cleanup_limit}, planned_max={cleanup_count}",
        flush=True,
    )

    if cleanup_count <= 0:
        print(
            "embedding cleanup: skipped because embedded job count does not exceed export keep count",
            flush=True,
        )
        return 0

    with conn.cursor() as cur:
        cur.execute(
            """
            WITH cleanup_candidates AS (
                SELECT id
                FROM jobs
                WHERE title_requirements_embedding IS NOT NULL
                  AND job_selected_words IS NOT NULL
                  AND job_selected_word_embeddings IS NOT NULL
                  AND job_phrase_chunks IS NOT NULL
                  AND job_phrase_chunk_embeddings IS NOT NULL
                  AND COALESCE(
                        posted_at_utc,
                        CASE
                          WHEN posted_at ~ '^\\d{4}-\\d{2}-\\d{2}' THEN posted_at::timestamptz
                          ELSE NULL
                        END,
                        fetched_at_utc,
                        created_at_utc
                      ) < now() - (%(retention_days)s * interval '1 day')
                ORDER BY
                  COALESCE(
                    posted_at_utc,
                    CASE
                      WHEN posted_at ~ '^\\d{4}-\\d{2}-\\d{2}' THEN posted_at::timestamptz
                      ELSE NULL
                    END,
                    fetched_at_utc,
                    created_at_utc
                  ) ASC,
                  id ASC
                LIMIT %(cleanup_count)s
            )
            UPDATE jobs
            SET
                job_word_tokens = NULL,
                job_selected_words = NULL,
                job_selected_word_embeddings = NULL,
                job_phrase_chunks = NULL,
                job_phrase_chunk_embeddings = NULL,
                title_requirements_text = NULL,
                title_requirements_embedding = NULL,
                content_embedding = NULL,
                embedding_model_name = NULL,
                embedding_model_revision = NULL,
                embedding_dim = NULL,
                embedded_at_utc = NULL,
                enrichment_version = NULL,
                enrichment_ml_version = NULL
            WHERE id IN (SELECT id FROM cleanup_candidates)
            """,
            {
                "retention_days": retention_days,
                "cleanup_count": cleanup_count,
            },
        )
        cleared_count = cur.rowcount

    conn.commit()
    print(
        "embedding cleanup: finished "
        f"cleared_embeddings={cleared_count}, embedded_jobs_before={embedded_count}, "
        f"export_keep_count={export_keep_count}",
        flush=True,
    )
    return int(cleared_count)


def fetch_rows_to_prepare(conn, *, limit: int, version: str) -> list[dict[str, Any]]:
    export_keep_count = get_export_embedding_keep_count()
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH export_eligible_jobs AS (
                -- Uses the stored/generated export-eligibility migration.
                SELECT id
                FROM jobs
                WHERE is_export_eligible = TRUE
                ORDER BY posted_at_utc DESC, id DESC
                LIMIT %(export_keep_count)s
            )
            SELECT
                jobs.id,
                jobs.title,
                jobs.content_text,
                jobs.extracted_requirements,
                jobs.enrichment_version,
                jobs.enrichment_ml_version
            FROM jobs
            JOIN export_eligible_jobs ON export_eligible_jobs.id = jobs.id
            WHERE
              (
                    jobs.job_word_tokens IS NULL
                 OR jobs.job_selected_words IS NULL
                 OR jobs.job_phrase_chunks IS NULL
                 OR jobs.title_requirements_text IS NULL
                 OR jobs.enrichment_version IS DISTINCT FROM %(version)s
              )
            ORDER BY jobs.posted_at_utc DESC, jobs.id DESC
            LIMIT %(limit)s
            """,
            {
                "export_keep_count": export_keep_count,
                "limit": limit,
                "version": version,
            },
        )
        return cur.fetchall()


def count_rows_to_prepare(conn, *, version: str) -> int:
    export_keep_count = get_export_embedding_keep_count()
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH export_eligible_jobs AS (
                -- Uses the stored/generated export-eligibility migration.
                SELECT id
                FROM jobs
                WHERE is_export_eligible = TRUE
                ORDER BY posted_at_utc DESC, id DESC
                LIMIT %(export_keep_count)s
            )
            SELECT COUNT(*) AS row_count
            FROM jobs
            JOIN export_eligible_jobs ON export_eligible_jobs.id = jobs.id
            WHERE (
                    jobs.job_word_tokens IS NULL
                 OR jobs.job_selected_words IS NULL
                 OR jobs.job_phrase_chunks IS NULL
                 OR jobs.title_requirements_text IS NULL
                 OR jobs.enrichment_version IS DISTINCT FROM %(version)s
              )
            """,
            {
                "export_keep_count": export_keep_count,
                "version": version,
            },
        )
        row = cur.fetchone()
    return int(row["row_count"])


def fetch_rows_to_enrich(conn, *, limit: int, model_name: str, model_revision: str, version: str) -> list[dict[str, Any]]:
    export_keep_count = get_export_embedding_keep_count()
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH export_eligible_jobs AS (
                -- Uses the stored/generated export-eligibility migration.
                SELECT id
                FROM jobs
                WHERE is_export_eligible = TRUE
                ORDER BY posted_at_utc DESC, id DESC
                LIMIT %(export_keep_count)s
            )
            SELECT
                jobs.id,
                jobs.content_text,
                jobs.job_selected_words,
                jobs.job_phrase_chunks,
                jobs.title_requirements_text,
                jobs.embedding_model_name,
                jobs.embedding_model_revision,
                jobs.enrichment_version,
                jobs.enrichment_ml_version
            FROM jobs
            JOIN export_eligible_jobs ON export_eligible_jobs.id = jobs.id
            WHERE
              (
                    jobs.job_selected_word_embeddings IS NULL
                 OR jobs.job_phrase_chunk_embeddings IS NULL
                 OR jobs.title_requirements_embedding IS NULL
                 OR jobs.embedding_model_name IS DISTINCT FROM %(model_name)s
                 OR jobs.embedding_model_revision IS DISTINCT FROM %(model_revision)s
                 OR jobs.enrichment_ml_version IS DISTINCT FROM %(version)s
              )
              AND jobs.enrichment_version = %(version)s
              AND jobs.job_selected_words IS NOT NULL
              AND jobs.job_phrase_chunks IS NOT NULL
              AND jobs.title_requirements_text IS NOT NULL
            ORDER BY jobs.posted_at_utc DESC, jobs.id DESC
            LIMIT %(limit)s
            """,
            {
                "export_keep_count": export_keep_count,
                "limit": limit,
                "model_name": model_name,
                "model_revision": model_revision,
                "version": version,
            },
        )
        return cur.fetchall()


def count_rows_to_enrich(conn, *, model_name: str, model_revision: str, version: str) -> int:
    export_keep_count = get_export_embedding_keep_count()
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH export_eligible_jobs AS (
                -- Uses the stored/generated export-eligibility migration.
                SELECT id
                FROM jobs
                WHERE is_export_eligible = TRUE
                ORDER BY posted_at_utc DESC, id DESC
                LIMIT %(export_keep_count)s
            )
            SELECT COUNT(*) AS row_count
            FROM jobs
            JOIN export_eligible_jobs ON export_eligible_jobs.id = jobs.id
            WHERE (
                    jobs.job_selected_word_embeddings IS NULL
                 OR jobs.job_phrase_chunk_embeddings IS NULL
                 OR jobs.title_requirements_embedding IS NULL
                 OR jobs.embedding_model_name IS DISTINCT FROM %(model_name)s
                 OR jobs.embedding_model_revision IS DISTINCT FROM %(model_revision)s
                 OR jobs.enrichment_ml_version IS DISTINCT FROM %(version)s
              )
              AND jobs.enrichment_version = %(version)s
              AND jobs.job_selected_words IS NOT NULL
              AND jobs.job_phrase_chunks IS NOT NULL
              AND jobs.title_requirements_text IS NOT NULL
            """,
            {
                "export_keep_count": export_keep_count,
                "model_name": model_name,
                "model_revision": model_revision,
                "version": version,
            },
        )
        row = cur.fetchone()
    return int(row["row_count"])


def build_prepared_feature_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_words = get_max_words_per_job()
    ranking_texts = [build_title_requirements_text(row) for row in rows]
    content_texts = [clean_text(row.get("content_text")) for row in rows]
    token_lists = [preprocess_words(text) for text in content_texts]
    selected_words = select_tfidf_words(content_texts, max_words_per_job=max_words)
    chunks = [
        phrase_chunks(clean_text(row.get("extracted_requirements")) or clean_text(row.get("content_text")))
        for row in rows
    ]
    selected_word_counts = [len(words) for words in selected_words]
    phrase_chunk_counts = [len(job_chunks) for job_chunks in chunks]
    total_selected_words = sum(selected_word_counts)
    total_phrase_chunks = sum(phrase_chunk_counts)
    unique_selected_words = len({word for words in selected_words for word in words})
    unique_phrase_chunks = len({chunk for job_chunks in chunks for chunk in job_chunks})

    print(
        "add_job_features: prepared "
        f"{len(rows)} rows, {total_selected_words} selected words "
        f"({unique_selected_words} unique), {total_phrase_chunks} phrase chunks "
        f"({unique_phrase_chunks} unique)",
        flush=True,
    )

    return {
        "ranking_texts": ranking_texts,
        "content_texts": content_texts,
        "token_lists": token_lists,
        "selected_words": selected_words,
        "chunks": chunks,
        "selected_word_counts": selected_word_counts,
        "phrase_chunk_counts": phrase_chunk_counts,
        "unique_selected_words": unique_selected_words,
        "unique_phrase_chunks": unique_phrase_chunks,
    }


def update_prepared_feature_rows(conn, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0

    payload = build_prepared_feature_payload(rows)
    updates: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        updates.append(
            {
                "id": row["id"],
                "job_word_tokens": Jsonb(payload["token_lists"][idx]),
                "job_selected_words": Jsonb(payload["selected_words"][idx]),
                "job_phrase_chunks": Jsonb(payload["chunks"][idx]),
                "title_requirements_text": payload["ranking_texts"][idx],
                "enrichment_version": get_enrichment_version(),
            }
        )

    print(f"add_job_features: writing {len(updates)} prepared feature rows to PostgreSQL", flush=True)
    with conn.cursor() as cur:
        cur.executemany(
            """
            UPDATE jobs
            SET
                job_word_tokens = %(job_word_tokens)s,
                job_selected_words = %(job_selected_words)s,
                job_selected_word_embeddings = NULL,
                job_phrase_chunks = %(job_phrase_chunks)s,
                job_phrase_chunk_embeddings = NULL,
                title_requirements_text = %(title_requirements_text)s,
                title_requirements_embedding = NULL,
                content_embedding = NULL,
                embedding_model_name = NULL,
                embedding_model_revision = NULL,
                embedding_dim = NULL,
                embedded_at_utc = NULL,
                enrichment_version = %(enrichment_version)s,
                enrichment_ml_version = NULL
            WHERE id = %(id)s
            """,
            updates,
        )

    conn.commit()
    return len(updates)


def update_embedding_rows(conn, rows: list[dict[str, Any]], model) -> int:
    if not rows:
        return 0

    model_name = get_minilm_model_name()
    model_revision = get_minilm_model_revision()
    version = get_enrichment_version()
    embedded_at = datetime.now(timezone.utc).replace(microsecond=0)

    ranking_texts = [clean_text(row.get("title_requirements_text")) for row in rows]
    content_texts = [clean_text(row.get("content_text")) for row in rows]
    selected_words = [list(row.get("job_selected_words") or []) for row in rows]
    chunks = [list(row.get("job_phrase_chunks") or []) for row in rows]
    selected_word_counts = [len(words) for words in selected_words]
    phrase_chunk_counts = [len(job_chunks) for job_chunks in chunks]
    unique_selected_words = len({word for words in selected_words for word in words})
    unique_phrase_chunks = len({chunk for job_chunks in chunks for chunk in job_chunks})

    print(
        "add_job_features: embedding "
        f"{len(rows)} prepared rows, {sum(selected_word_counts)} selected words "
        f"({unique_selected_words} unique), {sum(phrase_chunk_counts)} phrase chunks "
        f"({unique_phrase_chunks} unique)",
        flush=True,
    )
    print("add_job_features: encoding unique selected words", flush=True)
    word_embeddings = embed_unique_texts(
        model,
        selected_words,
        batch_size=64,
        normalize_embeddings=False,
    )
    print("add_job_features: encoding unique phrase chunks", flush=True)
    phrase_embeddings = embed_unique_texts(
        model,
        chunks,
        batch_size=64,
        normalize_embeddings=False,
    )

    print("add_job_features: encoding title+requirements embeddings", flush=True)
    title_embeddings = model.encode(
        ranking_texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    content_embeddings = None
    if should_embed_content_text():
        print("add_job_features: encoding optional content_text embeddings", flush=True)
        content_embeddings = model.encode(
            content_texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    title_embedding_matrix = np.asarray(title_embeddings, dtype=np.float32)
    embedding_dim = int(title_embedding_matrix.shape[1]) if title_embedding_matrix.ndim == 2 else "unknown"
    log_data_quality(
        "features",
        rows=len(rows),
        empty_words=sum(1 for count in selected_word_counts if count == 0),
        empty_phrases=sum(1 for count in phrase_chunk_counts if count == 0),
        unique_selected_words=unique_selected_words,
        unique_phrase_chunks=unique_phrase_chunks,
        embedding_rows=title_embedding_matrix.shape[0] if title_embedding_matrix.ndim >= 1 else "unknown",
        embedding_dim=embedding_dim,
        model_name=model_name,
        model_revision=model_revision,
        enrichment_version=version,
        **count_distribution(selected_word_counts, prefix="selected_words_"),
        **count_distribution(phrase_chunk_counts, prefix="phrase_chunks_"),
        **length_distribution(content_texts, prefix="content_"),
        **length_distribution(ranking_texts, prefix="title_requirements_"),
    )

    updates: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        title_embedding = np.asarray(title_embeddings[idx], dtype=np.float32)
        if len(selected_words[idx]) != len(word_embeddings[idx]):
            raise RuntimeError(f"Word embedding count mismatch for job id={row['id']}")
        if len(chunks[idx]) != len(phrase_embeddings[idx]):
            raise RuntimeError(f"Phrase embedding count mismatch for job id={row['id']}")
        content_embedding = (
            np.asarray(content_embeddings[idx], dtype=np.float32).tolist()
            if content_embeddings is not None
            else None
        )
        updates.append(
            {
                "id": row["id"],
                "job_selected_word_embeddings": Jsonb(word_embeddings[idx]),
                "job_phrase_chunk_embeddings": Jsonb(phrase_embeddings[idx]),
                "title_requirements_embedding": Jsonb(title_embedding.tolist()),
                "content_embedding": Jsonb(content_embedding) if content_embedding is not None else None,
                "embedding_model_name": model_name,
                "embedding_model_revision": model_revision,
                "embedding_dim": int(title_embedding.shape[0]),
                "embedded_at_utc": embedded_at,
                "enrichment_ml_version": version,
            }
        )

    print(f"add_job_features: writing {len(updates)} embedded feature rows to PostgreSQL", flush=True)
    with conn.cursor() as cur:
        cur.executemany(
            """
            UPDATE jobs
            SET
                job_selected_word_embeddings = %(job_selected_word_embeddings)s,
                job_phrase_chunk_embeddings = %(job_phrase_chunk_embeddings)s,
                title_requirements_embedding = %(title_requirements_embedding)s,
                content_embedding = COALESCE(%(content_embedding)s, content_embedding),
                embedding_model_name = %(embedding_model_name)s,
                embedding_model_revision = %(embedding_model_revision)s,
                embedding_dim = %(embedding_dim)s,
                embedded_at_utc = %(embedded_at_utc)s,
                enrichment_ml_version = %(enrichment_ml_version)s
            WHERE id = %(id)s
            """,
            updates,
        )

    conn.commit()
    return len(updates)


def update_feature_rows(conn, rows: list[dict[str, Any]], model) -> int:
    if not rows:
        return 0

    payload = build_prepared_feature_payload(rows)
    prepared_rows = []
    for idx, row in enumerate(rows):
        prepared_rows.append(
            {
                **row,
                "job_selected_words": payload["selected_words"][idx],
                "job_phrase_chunks": payload["chunks"][idx],
                "title_requirements_text": payload["ranking_texts"][idx],
            }
        )

    update_prepared_feature_rows(conn, rows)
    return update_embedding_rows(conn, prepared_rows, model)


def run_preparation(conn) -> int:
    batch_size = get_non_ml_batch_size()
    max_batches = get_max_batches()
    version = get_enrichment_version()

    total_pending = count_rows_to_prepare(conn, version=version)
    print(
        f"add_job_features: {total_pending} rows need non-model feature preparation; "
        f"processing up to {batch_size * max_batches} this run "
        f"({max_batches} batches of {batch_size})",
        flush=True,
    )

    if total_pending <= 0:
        conn.commit()
        print("add_job_features: no rows needed non-model feature preparation", flush=True)
        return 0

    total_count = 0
    for batch_number in range(1, max_batches + 1):
        rows = fetch_rows_to_prepare(conn, limit=batch_size, version=version)
        if not rows:
            print(
                f"add_job_features: no rows remained before preparation batch {batch_number}/{max_batches}",
                flush=True,
            )
            break

        print(
            f"add_job_features: starting preparation batch {batch_number}/{max_batches} with {len(rows)} rows",
            flush=True,
        )
        count = update_prepared_feature_rows(conn, rows)
        total_count += count
        print(
            f"add_job_features: finished preparation batch {batch_number}/{max_batches}; "
            f"batch_rows={count}, total_prepared={total_count}",
            flush=True,
        )

        if count < batch_size:
            break

    print(f"add_job_features: prepared {total_count} rows with non-model ranking features", flush=True)
    return total_count


def run_embeddings(conn) -> int:
    batch_size = get_batch_size()
    max_batches = get_ml_max_batches()
    model_name = get_minilm_model_name()
    model_revision = get_minilm_model_revision()
    version = get_enrichment_version()

    total_pending = count_rows_to_enrich(
        conn,
        model_name=model_name,
        model_revision=model_revision,
        version=version,
    )
    print(
        f"add_job_features: {total_pending} prepared rows need embeddings; "
        f"processing up to {batch_size * max_batches} this run "
        f"({max_batches} batches of {batch_size})",
        flush=True,
    )

    if total_pending <= 0:
        conn.commit()
        print("add_job_features: no prepared rows needed embeddings", flush=True)
        cleanup_old_embeddings(conn)
        return 0

    print(
        f"add_job_features: loading MiniLM for up to {min(total_pending, batch_size * max_batches)} prepared rows",
        flush=True,
    )
    model = load_minilm_model()

    total_count = 0
    for batch_number in range(1, max_batches + 1):
        rows = fetch_rows_to_enrich(
            conn,
            limit=batch_size,
            model_name=model_name,
            model_revision=model_revision,
            version=version,
        )
        if not rows:
            print(
                f"add_job_features: no prepared rows remained before embedding batch {batch_number}/{max_batches}",
                flush=True,
            )
            break

        print(
            f"add_job_features: starting embedding batch {batch_number}/{max_batches} with {len(rows)} rows",
            flush=True,
        )
        count = update_embedding_rows(conn, rows, model)
        total_count += count
        print(
            f"add_job_features: finished embedding batch {batch_number}/{max_batches}; "
            f"batch_rows={count}, total_enriched={total_count}",
            flush=True,
        )

        if count < batch_size:
            break

    print(
        f"add_job_features: enriched {total_count} rows with job-side precomputed embeddings",
        flush=True,
    )
    cleanup_old_embeddings(conn)
    return total_count


def run(conn) -> int:
    prepared_count = run_preparation(conn)
    embedded_count = run_embeddings(conn)
    return prepared_count + embedded_count
