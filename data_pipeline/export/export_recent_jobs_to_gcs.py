from __future__ import annotations

"""
Export recent PostgreSQL jobs to ranking-serving artifacts.

The primary serving artifacts are JSONL metadata plus compressed NumPy arrays.
The CSV is still written as a backward-compatible debug/fallback artifact.
"""

import csv
import json
import os
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from io import BytesIO, StringIO
from typing import Any

import numpy as np
from google.cloud import storage

from data_pipeline.common.model_loader import get_enrichment_version


DEFAULT_GCS_JOBS_BLOB_NAME = "job_posting_data/combined_jobs_filtered.csv"
DEFAULT_GCS_METADATA_BLOB_NAME = "job_posting_data/jobs_metadata.jsonl"
DEFAULT_GCS_EMBEDDINGS_BLOB_NAME = "job_posting_data/job_embeddings.npz"
DEFAULT_GCS_RANKING_SHARDS_PREFIX = "job_posting_data/shards"
DEFAULT_EXPORT_LIMIT = 1000
DEFAULT_RANKING_ARTIFACT_SHARD_COUNT = 1
DEFAULT_RANKING_ARTIFACT_JOBS_PER_SHARD = 1000
DEFAULT_EXPORT_LEGACY_FALLBACK_ARTIFACTS = False

EXPORT_COLUMNS = [
    "id",
    "source_type",
    "source_url",
    "source_job_id",
    "source_company",
    "company_name",
    "title",
    "location_name",
    "location_country",
    "location_region",
    "location_segments",
    "work_arrangement",
    "location_parse_status",
    "location_normalization_version",
    "location_normalized_at_utc",
    "department_names",
    "office_names",
    "posted_at",
    "posted_at_utc",
    "updated_at",
    "job_url",
    "apply_url",
    "content_text",
    "extracted_requirements",
    "years_experience_raw",
    "min_years_experience",
    "max_years_experience",
    "experience_type",
    "evidence_text",
    "requires_clearance",
    "clearance_type",
    "clearance_requirement_type",
    "clearance_evidence_text",
    "job_selected_words",
    "job_selected_word_embeddings",
    "job_phrase_chunks",
    "job_phrase_chunk_embeddings",
    "embedding_model_name",
    "embedding_model_revision",
    "embedding_dim",
    "embedded_at_utc",
    "enrichment_version",
    "enrichment_ml_version",
    "is_active",
    "missing_from_source_at_utc",
    "last_seen_at_utc",
    "first_seen_at_utc",
    "stale_reason",
]

REQUIRED_COLUMNS = {"content_text"}
ARTIFACT_REQUIRED_COLUMNS = {
    "id",
    "source_type",
    "source_job_id",
    "content_text",
    "job_selected_words",
    "job_selected_word_embeddings",
    "job_phrase_chunks",
    "job_phrase_chunk_embeddings",
    "title_requirements_embedding",
    "is_active",
    "missing_from_source_at_utc",
    "posted_at_utc",
    "location_parse_status",
    "enrichment_version",
    "enrichment_ml_version",
}


def get_export_limit() -> int:
    value = os.environ.get("RECENT_JOBS_EXPORT_LIMIT", str(DEFAULT_EXPORT_LIMIT))
    try:
        limit = int(value)
    except ValueError as exc:
        raise ValueError("RECENT_JOBS_EXPORT_LIMIT must be an integer.") from exc

    if limit <= 0:
        raise ValueError("RECENT_JOBS_EXPORT_LIMIT must be greater than zero.")

    return limit


def get_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def should_export_legacy_fallback_artifacts() -> bool:
    return get_bool_env("EXPORT_LEGACY_FALLBACK_ARTIFACTS", DEFAULT_EXPORT_LEGACY_FALLBACK_ARTIFACTS)


def get_gcs_destination() -> tuple[str, str]:
    bucket_name = os.environ.get("GCS_BUCKET_NAME", "").strip()
    blob_name = os.environ.get("GCS_JOBS_BLOB_NAME", DEFAULT_GCS_JOBS_BLOB_NAME).strip()

    if not bucket_name:
        raise RuntimeError("GCS_BUCKET_NAME is required for recent jobs export.")
    if not blob_name:
        raise RuntimeError("GCS_JOBS_BLOB_NAME is required for recent jobs export.")

    return bucket_name, blob_name


def get_artifact_destinations() -> tuple[str, str, str]:
    bucket_name, csv_blob_name = get_gcs_destination()
    metadata_blob_name = os.environ.get("GCS_METADATA_BLOB_NAME", DEFAULT_GCS_METADATA_BLOB_NAME).strip()
    embeddings_blob_name = os.environ.get("GCS_EMBEDDINGS_BLOB_NAME", DEFAULT_GCS_EMBEDDINGS_BLOB_NAME).strip()

    if not metadata_blob_name:
        raise RuntimeError("GCS_METADATA_BLOB_NAME cannot be empty.")
    if not embeddings_blob_name:
        raise RuntimeError("GCS_EMBEDDINGS_BLOB_NAME cannot be empty.")

    return bucket_name, metadata_blob_name, embeddings_blob_name


def get_positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc

    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return parsed


def get_shard_export_config() -> tuple[int, int, str]:
    shard_count = get_positive_int_env(
        "RANKING_ARTIFACT_SHARD_COUNT",
        DEFAULT_RANKING_ARTIFACT_SHARD_COUNT,
    )
    jobs_per_shard = get_positive_int_env(
        "RANKING_ARTIFACT_JOBS_PER_SHARD",
        DEFAULT_RANKING_ARTIFACT_JOBS_PER_SHARD,
    )
    prefix = os.environ.get("GCS_RANKING_SHARDS_PREFIX", DEFAULT_GCS_RANKING_SHARDS_PREFIX).strip().strip("/")
    if not prefix:
        raise RuntimeError("GCS_RANKING_SHARDS_PREFIX cannot be empty.")

    return shard_count, jobs_per_shard, prefix


def get_existing_job_columns(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'jobs'
            """
        )
        return {row["column_name"] for row in cur.fetchall()}


def fetch_recent_jobs(conn, *, limit: int) -> list[dict[str, Any]]:
    return fetch_recent_jobs_page(conn, limit=limit, offset=0)


def fetch_recent_jobs_page(conn, *, limit: int, offset: int) -> list[dict[str, Any]]:
    existing_columns = get_existing_job_columns(conn)
    missing_required = (REQUIRED_COLUMNS | ARTIFACT_REQUIRED_COLUMNS) - existing_columns
    if missing_required:
        raise RuntimeError(f"jobs table is missing required export columns: {sorted(missing_required)}")

    selected_columns = [column for column in EXPORT_COLUMNS if column in existing_columns]
    selected_columns.append("title_requirements_embedding")
    select_sql = ", ".join(selected_columns)
    enrichment_version = get_enrichment_version()

    with conn.cursor() as cur:
        cur.execute(
            f"""
            -- Uses the stored/generated export-eligibility migration.
            SELECT {select_sql}
            FROM jobs
            WHERE is_export_eligible = TRUE
              AND title_requirements_embedding IS NOT NULL
              AND job_selected_words IS NOT NULL
              AND job_selected_word_embeddings IS NOT NULL
              AND job_phrase_chunks IS NOT NULL
              AND job_phrase_chunk_embeddings IS NOT NULL
              AND enrichment_version = %s
              AND enrichment_ml_version = %s
            ORDER BY
              posted_at_utc DESC,
              id DESC
            LIMIT %s
            OFFSET %s
            """,
            (enrichment_version, enrichment_version, limit, offset),
        )
        rows = cur.fetchall()

    exported_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=offset):
        row_dict = {column: row.get(column, "") for column in EXPORT_COLUMNS}
        row_dict["csv_row_index"] = index
        row_dict["title_requirements_embedding"] = row.get("title_requirements_embedding")
        exported_rows.append(row_dict)

    return exported_rows


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    output = StringIO()
    csv_columns = [
        column
        for column in EXPORT_COLUMNS
        if column not in {"job_selected_word_embeddings", "job_phrase_chunk_embeddings"}
    ]
    writer = csv.DictWriter(output, fieldnames=csv_columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def upload_csv_to_gcs(csv_text: str, *, bucket_name: str, blob_name: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(csv_text, content_type="text/csv")


def metadata_rows_to_jsonl(
    rows: list[dict[str, Any]],
    *,
    artifact_shard: int | None = None,
) -> str:
    lines: list[str] = []
    for artifact_row_index, row in enumerate(rows):
        job_id = f"{row.get('source_type')}:{row.get('source_job_id')}"
        payload = {
            "job_id": job_id,
            "db_id": row.get("id"),
            "csv_row_index": row.get("csv_row_index"),
            "artifact_shard": artifact_shard if artifact_shard is not None else row.get("artifact_shard", 0),
            "artifact_row_index": row.get("artifact_row_index", artifact_row_index),
            "title": row.get("title"),
            "company": row.get("company_name"),
            "company_name": row.get("company_name"),
            "source_type": row.get("source_type"),
            "source_company": row.get("source_company"),
            "location": row.get("location_name"),
            "location_name": row.get("location_name"),
            "location_country": row.get("location_country"),
            "location_region": row.get("location_region"),
            "location_segments": row.get("location_segments") or [],
            "work_arrangement": row.get("work_arrangement"),
            "location_parse_status": row.get("location_parse_status"),
            "location_normalization_version": row.get("location_normalization_version"),
            "location_normalized_at_utc": row.get("location_normalized_at_utc"),
            "posted_at": row.get("posted_at"),
            "posted_at_utc": row.get("posted_at_utc"),
            "url": row.get("job_url") or row.get("apply_url"),
            "job_url": row.get("job_url"),
            "content_text": row.get("content_text"),
            "extracted_requirements": row.get("extracted_requirements"),
            "job_selected_words": row.get("job_selected_words") or [],
            "job_phrase_chunks": row.get("job_phrase_chunks") or [],
            "years_experience_raw": row.get("years_experience_raw"),
            "min_years_experience": row.get("min_years_experience"),
            "max_years_experience": row.get("max_years_experience"),
            "experience_type": row.get("experience_type"),
            "evidence_text": row.get("evidence_text"),
            "requires_clearance": row.get("requires_clearance"),
            "clearance_type": row.get("clearance_type"),
            "clearance_requirement_type": row.get("clearance_requirement_type"),
            "clearance_evidence_text": row.get("clearance_evidence_text"),
            "is_active": row.get("is_active"),
            "missing_from_source_at_utc": row.get("missing_from_source_at_utc"),
            "last_seen_at_utc": row.get("last_seen_at_utc"),
            "first_seen_at_utc": row.get("first_seen_at_utc"),
            "stale_reason": row.get("stale_reason"),
            "embedding_model_name": row.get("embedding_model_name"),
            "embedding_model_revision": row.get("embedding_model_revision"),
            "embedding_dim": row.get("embedding_dim"),
            "embedded_at_utc": row.get("embedded_at_utc"),
            "enrichment_version": row.get("enrichment_version"),
            "enrichment_ml_version": row.get("enrichment_ml_version"),
        }
        lines.append(json.dumps(payload, default=str, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def source_type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("source_type") or "unknown") for row in rows)
    return dict(sorted(counts.items()))


def is_first_seen_within_last_24_hours(row: dict[str, Any], *, now_utc: datetime) -> bool:
    value = row.get("first_seen_at_utc")
    if value is None:
        return False

    if isinstance(value, datetime):
        first_seen = value
    else:
        try:
            first_seen = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return False

    if first_seen.tzinfo is None:
        first_seen = first_seen.replace(tzinfo=timezone.utc)
    else:
        first_seen = first_seen.astimezone(timezone.utc)

    return first_seen >= now_utc - timedelta(hours=24)


def recent_first_seen_count(rows: list[dict[str, Any]], *, now_utc: datetime) -> int:
    return sum(1 for row in rows if is_first_seen_within_last_24_hours(row, now_utc=now_utc))


def print_export_quality_summary(
    *,
    label: str,
    rows: list[dict[str, Any]],
    now_utc: datetime,
    shard_index: int | None = None,
) -> None:
    prefix = "export_recent_jobs_to_gcs: export quality summary"
    shard_text = f" shard={shard_index:05d}" if shard_index is not None else ""
    print(
        f"{prefix}:{shard_text} "
        f"label={label}, rows={len(rows)}, "
        f"new_first_seen_24h={recent_first_seen_count(rows, now_utc=now_utc)}, "
        f"source_type_counts={source_type_counts(rows)}",
        flush=True,
    )


def print_export_timing(stage: str, started_at: float, **fields: Any) -> None:
    metadata = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {metadata}" if metadata else ""
    print(
        f"EXPORT_TIMING stage={stage} elapsed_seconds={time.perf_counter() - started_at:.3f}{suffix}",
        flush=True,
    )


def rows_to_npz_bytes(rows: list[dict[str, Any]]) -> bytes:
    embeddings = []
    word_embedding_groups = []
    phrase_embedding_groups = []
    job_ids = []
    db_ids = []

    for row in rows:
        embedding = row.get("title_requirements_embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError(f"Missing title_requirements_embedding for exported job id={row.get('id')}")
        word_embeddings = row.get("job_selected_word_embeddings")
        phrase_embeddings = row.get("job_phrase_chunk_embeddings")
        if not isinstance(word_embeddings, list):
            raise RuntimeError(f"Missing job_selected_word_embeddings for exported job id={row.get('id')}")
        if not isinstance(phrase_embeddings, list):
            raise RuntimeError(f"Missing job_phrase_chunk_embeddings for exported job id={row.get('id')}")
        embeddings.append(np.asarray(embedding, dtype=np.float32))
        word_embedding_groups.append(np.asarray(word_embeddings, dtype=np.float32).reshape((-1, len(embedding))))
        phrase_embedding_groups.append(np.asarray(phrase_embeddings, dtype=np.float32).reshape((-1, len(embedding))))
        job_ids.append(f"{row.get('source_type')}:{row.get('source_job_id')}")
        db_ids.append(int(row["id"]))

    title_requirements_embeddings = np.vstack(embeddings).astype(np.float32)
    word_offsets = offsets_from_groups(word_embedding_groups)
    phrase_offsets = offsets_from_groups(phrase_embedding_groups)
    word_flat = flatten_embedding_groups(word_embedding_groups, title_requirements_embeddings.shape[1])
    phrase_flat = flatten_embedding_groups(phrase_embedding_groups, title_requirements_embeddings.shape[1])
    if word_offsets[-1] != len(word_flat):
        raise RuntimeError("Word embedding offsets do not align with flattened word embeddings.")
    if phrase_offsets[-1] != len(phrase_flat):
        raise RuntimeError("Phrase embedding offsets do not align with flattened phrase embeddings.")
    output = BytesIO()
    np.savez(
        output,
        title_requirements_embeddings=title_requirements_embeddings,
        job_selected_word_embeddings=word_flat,
        job_selected_word_offsets=word_offsets,
        job_phrase_chunk_embeddings=phrase_flat,
        job_phrase_chunk_offsets=phrase_offsets,
        job_ids=np.asarray(job_ids),
        db_ids=np.asarray(db_ids, dtype=np.int64),
    )
    return output.getvalue()


def offsets_from_groups(groups: list[np.ndarray]) -> np.ndarray:
    offsets = [0]
    total = 0
    for group in groups:
        total += int(group.shape[0])
        offsets.append(total)
    return np.asarray(offsets, dtype=np.int64)


def flatten_embedding_groups(groups: list[np.ndarray], embedding_dim: int) -> np.ndarray:
    non_empty = [group.astype(np.float32) for group in groups if group.size]
    if not non_empty:
        return np.empty((0, embedding_dim), dtype=np.float32)
    return np.vstack(non_empty).astype(np.float32)


def upload_text_to_gcs(text: str, *, bucket_name: str, blob_name: str, content_type: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(text, content_type=content_type)


def upload_bytes_to_gcs(data: bytes, *, bucket_name: str, blob_name: str, content_type: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)


def shard_blob_names(prefix: str, shard_index: int) -> tuple[str, str]:
    return (
        f"{prefix}/jobs_metadata_{shard_index:05d}.jsonl",
        f"{prefix}/job_embeddings_{shard_index:05d}.npz",
    )


def upload_sharded_artifacts_to_gcs(
    conn,
    *,
    bucket_name: str,
    shard_count: int,
    jobs_per_shard: int,
    prefix: str,
) -> int:
    uploaded = 0
    uploaded_rows = 0
    total_recent_first_seen_24h = 0
    total_source_type_counts: Counter[str] = Counter()
    now_utc = datetime.now(timezone.utc)
    for shard_index in range(shard_count):
        step_started_at = time.perf_counter()
        shard_rows = fetch_recent_jobs_page(
            conn,
            limit=jobs_per_shard,
            offset=shard_index * jobs_per_shard,
        )
        print_export_timing(
            "shard_db_fetch",
            step_started_at,
            shard=f"{shard_index:05d}",
            rows=len(shard_rows),
            offset=shard_index * jobs_per_shard,
            limit=jobs_per_shard,
        )
        if not shard_rows:
            break
        for artifact_row_index, row in enumerate(shard_rows):
            row["artifact_shard"] = shard_index
            row["artifact_row_index"] = artifact_row_index
        print_export_quality_summary(
            label="shard",
            rows=shard_rows,
            now_utc=now_utc,
            shard_index=shard_index,
        )
        metadata_blob_name, embeddings_blob_name = shard_blob_names(prefix, shard_index)
        step_started_at = time.perf_counter()
        metadata_text = metadata_rows_to_jsonl(shard_rows, artifact_shard=shard_index)
        print_export_timing(
            "shard_metadata_jsonl_build",
            step_started_at,
            shard=f"{shard_index:05d}",
            rows=len(shard_rows),
            bytes=len(metadata_text.encode("utf-8")),
        )
        step_started_at = time.perf_counter()
        upload_text_to_gcs(
            metadata_text,
            bucket_name=bucket_name,
            blob_name=metadata_blob_name,
            content_type="application/x-ndjson",
        )
        print_export_timing(
            "shard_metadata_gcs_upload",
            step_started_at,
            shard=f"{shard_index:05d}",
            bytes=len(metadata_text.encode("utf-8")),
        )
        step_started_at = time.perf_counter()
        embeddings_bytes = rows_to_npz_bytes(shard_rows)
        print_export_timing(
            "shard_npz_build",
            step_started_at,
            shard=f"{shard_index:05d}",
            rows=len(shard_rows),
            bytes=len(embeddings_bytes),
        )
        step_started_at = time.perf_counter()
        upload_bytes_to_gcs(
            embeddings_bytes,
            bucket_name=bucket_name,
            blob_name=embeddings_blob_name,
            content_type="application/octet-stream",
        )
        print_export_timing(
            "shard_embeddings_gcs_upload",
            step_started_at,
            shard=f"{shard_index:05d}",
            bytes=len(embeddings_bytes),
        )
        uploaded += 1
        uploaded_rows += len(shard_rows)
        total_recent_first_seen_24h += recent_first_seen_count(shard_rows, now_utc=now_utc)
        total_source_type_counts.update(source_type_counts(shard_rows))
        print(
            "export_recent_jobs_to_gcs: finished exporting shard "
            f"{uploaded}/{shard_count} "
            f"index={shard_index:05d} rows={len(shard_rows)} to "
            f"gs://{bucket_name}/{metadata_blob_name}, "
            f"gs://{bucket_name}/{embeddings_blob_name}",
            flush=True,
        )
        del shard_rows

    print(
        "export_recent_jobs_to_gcs: finished exporting sharded artifacts "
        f"shards={uploaded}, rows={uploaded_rows}, "
        f"new_first_seen_24h={total_recent_first_seen_24h}, "
        f"source_type_counts={dict(sorted(total_source_type_counts.items()))}",
        flush=True,
    )
    return uploaded_rows


def export_recent_jobs_to_gcs(conn) -> int:
    limit = get_export_limit()
    bucket_name, blob_name = get_gcs_destination()
    artifact_bucket_name, metadata_blob_name, embeddings_blob_name = get_artifact_destinations()
    shard_count, jobs_per_shard, shards_prefix = get_shard_export_config()
    fallback_rows: list[dict[str, Any]] = []
    export_legacy_fallback = should_export_legacy_fallback_artifacts()
    if export_legacy_fallback:
        fallback_limit = min(limit, jobs_per_shard)
        if limit > jobs_per_shard:
            print(
                "export_recent_jobs_to_gcs: legacy fallback artifact export capped at "
                f"{fallback_limit} rows to avoid holding more than one shard in memory "
                f"(RECENT_JOBS_EXPORT_LIMIT={limit}).",
                flush=True,
            )
        fallback_rows = fetch_recent_jobs(conn, limit=fallback_limit)

        if not fallback_rows:
            raise RuntimeError("No usable enriched jobs were available to export.")

        now_utc = datetime.now(timezone.utc)
        print_export_quality_summary(
            label="legacy_fallback",
            rows=fallback_rows,
            now_utc=now_utc,
        )
        upload_csv_to_gcs(
            rows_to_csv(fallback_rows),
            bucket_name=bucket_name,
            blob_name=blob_name,
        )
        upload_text_to_gcs(
            metadata_rows_to_jsonl(fallback_rows),
            bucket_name=artifact_bucket_name,
            blob_name=metadata_blob_name,
            content_type="application/x-ndjson",
        )
        upload_bytes_to_gcs(
            rows_to_npz_bytes(fallback_rows),
            bucket_name=artifact_bucket_name,
            blob_name=embeddings_blob_name,
            content_type="application/octet-stream",
        )
    else:
        print(
            "export_recent_jobs_to_gcs: skipping legacy fallback CSV/single-artifact export "
            "because EXPORT_LEGACY_FALLBACK_ARTIFACTS=false.",
            flush=True,
        )
    uploaded_shard_rows = upload_sharded_artifacts_to_gcs(
        conn,
        bucket_name=artifact_bucket_name,
        shard_count=shard_count,
        jobs_per_shard=jobs_per_shard,
        prefix=shards_prefix,
    )
    if uploaded_shard_rows <= 0:
        raise RuntimeError("No usable enriched jobs were available to export as shards.")

    print(
        "export_recent_jobs_to_gcs: exported "
        f"fallback_enabled={export_legacy_fallback}, fallback_rows={len(fallback_rows)}, "
        f"shard_rows={uploaded_shard_rows}, jobs_per_shard={jobs_per_shard}",
        flush=True,
    )
    return uploaded_shard_rows


def run(conn) -> int:
    return export_recent_jobs_to_gcs(conn)


if __name__ == "__main__":
    from data_pipeline.common.db import connect

    with connect() as connection:
        run(connection)
