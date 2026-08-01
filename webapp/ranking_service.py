from __future__ import annotations


"""
Loads job data and runs the ranking pipeline.

Behavior:
- If running on Cloud Run (K_SERVICE set): download CSV from GCS bucket.
- Otherwise: load local sample CSV from the data folder.

Pipeline:
1. Load and clean jobs.
2. Apply country + precomputed experience filters.
3. Cluster jobs by requirements and keep top clusters.
4. Rank remaining jobs via multi-stage pipeline, ending with resume phrase coverage and optional LLM filtering.
5. Return top results with metadata.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from google.cloud import storage

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

from hard_filters.country_filter import filter_jobs_df_by_country
from hard_filters.security_clearance_filter import filter_jobs_df_excluding_security_clearance
from hard_filters.years_experience_filter import filter_jobs_df_by_max_required_yoe
from recent_posted_filter import (
    RECENT_POSTED_HOURS,
    filter_recently_posted_jobs_df,
    format_posted_at_display,
    parse_posted_at_utc,
)
try:
    from ranking_timing import (
        print_ranking_timing_summary,
        record_ranking_timing,
        reset_ranking_timing_collection,
        start_ranking_timing_collection,
    )
except ImportError:
    from webapp.ranking_timing import (
        print_ranking_timing_summary,
        record_ranking_timing,
        reset_ranking_timing_collection,
        start_ranking_timing_collection,
    )


try:
    from model_loader import load_cross_encoder_model, load_minilm_model
except ImportError:
    from webapp.model_loader import load_cross_encoder_model, load_minilm_model
from ranking_algorithms.requirements_embedding_clustering import cluster_jobs_by_requirements_df
from multi_stage_rankings import rank_jobs_multi_stage, run_seniority_filter_operation
from reduction_policy import reduce_job_ids


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

RUNNING_ON_CLOUD_RUN = bool(os.environ.get("K_SERVICE"))


def env_nonnegative_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} cannot be negative.")
    return parsed


LOCAL_JOBS_CSV_PATH = Path(
    os.environ.get(
        "LOCAL_JOBS_CSV_PATH",
        DATA_DIR / "sample_combined_jobs_filtered_with_requirements.csv",
    )
)

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "")
GCS_JOBS_BLOB_NAME = os.environ.get(
    "GCS_JOBS_BLOB_NAME",
    "job_posting_data/combined_jobs_filtered.csv",
)
GCS_METADATA_BLOB_NAME = os.environ.get(
    "GCS_METADATA_BLOB_NAME",
    "job_posting_data/jobs_metadata.jsonl",
)
GCS_EMBEDDINGS_BLOB_NAME = os.environ.get(
    "GCS_EMBEDDINGS_BLOB_NAME",
    "job_posting_data/job_embeddings.npz",
)
GCS_RANKING_SHARDS_PREFIX = os.environ.get("GCS_RANKING_SHARDS_PREFIX", "job_posting_data/shards").strip().strip("/")

DOWNLOADED_GCS_CSV_PATH = Path("/tmp/combined_jobs_filtered.csv")
DOWNLOADED_GCS_METADATA_PATH = Path("/tmp/jobs_metadata.jsonl")
DOWNLOADED_GCS_EMBEDDINGS_PATH = Path("/tmp/job_embeddings.npz")
DOWNLOADED_GCS_SHARDS_DIR = Path("/tmp/ranking_shards")

TEXT_COLUMN = os.environ.get("TEXT_COLUMN", "content_text")
LOCATION_COLUMN = os.environ.get("LOCATION_COLUMN", "location_name")

MULTI_STAGE_WORD_KEEP = int(os.environ.get("MULTI_STAGE_WORD_KEEP", os.environ.get("STAGE1_KEEP", "40")))
MULTI_STAGE_PHRASE_KEEP = int(os.environ.get("MULTI_STAGE_PHRASE_KEEP", os.environ.get("STAGE2_KEEP", "15")))
CROSS_ENCODER_UNION_TOP_K_PER_RANKER = int(os.environ.get("CROSS_ENCODER_UNION_TOP_K_PER_RANKER", "25"))
ENABLE_CROSS_ENCODER = os.environ.get("ENABLE_CROSS_ENCODER", "true").strip().lower() in {"1", "true", "yes", "on"}
PIPELINE_POOR_MATCH_MAX_RANK = int(os.environ.get("PIPELINE_POOR_MATCH_MAX_RANK", "150"))
MAHALANOBIS_INPUT_TOP_K = int(os.environ.get("MAHALANOBIS_INPUT_TOP_K", "10"))
MAHALANOBIS_REMOVE_BOTTOM_FRACTION = float(os.environ.get("MAHALANOBIS_REMOVE_BOTTOM_FRACTION", "0.66"))
MULTI_METRIC_BAD_FIT_BOTTOM_FRACTION = float(os.environ.get("MULTI_METRIC_BAD_FIT_BOTTOM_FRACTION", "0.25"))
ENABLE_TECHNOLOGY_MISMATCH_FILTER = os.environ.get(
    "ENABLE_TECHNOLOGY_MISMATCH_FILTER",
    "true",
).lower() in {"1", "true", "yes"}
TECH_FILTER_MIN_JOB_TYPES = int(os.environ.get("TECH_FILTER_MIN_JOB_TYPES", "3"))
TECH_FILTER_MAX_JOB_TYPE_OVERLAP_RATIO = float(os.environ.get("TECH_FILTER_MAX_JOB_TYPE_OVERLAP_RATIO", "0.333333"))
RESUME_DATASET_DIR = os.environ.get("RESUME_DATASET_DIR", "").strip()
RESUME_PHRASE_DISTANCE_FLAG_PERCENTILE = float(os.environ.get("RESUME_PHRASE_DISTANCE_FLAG_PERCENTILE", "10"))
RESUME_PHRASE_BAD_MATCH_PERCENTILE = float(os.environ.get("RESUME_PHRASE_BAD_MATCH_PERCENTILE", "90"))
RESUME_PHRASE_JOB_FLAG_FRACTION = float(os.environ.get("RESUME_PHRASE_JOB_FLAG_FRACTION", "0.30"))
RESUME_PHRASE_JOB_COVERAGE_COMPARISON_JOBS = int(
    os.environ.get("RESUME_PHRASE_JOB_COVERAGE_COMPARISON_JOBS", "100")
)
RESUME_PHRASE_JOB_COVERAGE_FLAG_PERCENTILE = float(
    os.environ.get("RESUME_PHRASE_JOB_COVERAGE_FLAG_PERCENTILE", "10")
)
RESUME_PHRASE_JOB_COVERAGE_BAD_MATCH_PERCENTILE = float(
    os.environ.get("RESUME_PHRASE_JOB_COVERAGE_BAD_MATCH_PERCENTILE", "90")
)
RESUME_PHRASE_COVERAGE_REMOVE_BOTTOM_GOOD_FIT_FRACTION = float(
    os.environ.get("RESUME_PHRASE_COVERAGE_REMOVE_BOTTOM_GOOD_FIT_FRACTION", "0")
)
RESUME_PHRASE_COVERAGE_REMOVE_TOP_BAD_MATCH_FRACTION = float(
    os.environ.get("RESUME_PHRASE_COVERAGE_REMOVE_TOP_BAD_MATCH_FRACTION", "0")
)
RESUME_PHRASE_JOB_COVERAGE_REMOVE_BOTTOM_GOOD_FIT_FRACTION = float(
    os.environ.get("RESUME_PHRASE_JOB_COVERAGE_REMOVE_BOTTOM_GOOD_FIT_FRACTION", "0")
)
RESUME_PHRASE_JOB_COVERAGE_REMOVE_TOP_BAD_MATCH_FRACTION = float(
    os.environ.get("RESUME_PHRASE_JOB_COVERAGE_REMOVE_TOP_BAD_MATCH_FRACTION", "0")
)

USE_REQUIREMENTS_CLUSTERING = os.environ.get("USE_REQUIREMENTS_CLUSTERING", "true").lower() in {"1", "true", "yes"}
USE_RANKING_ARTIFACTS = os.environ.get("USE_RANKING_ARTIFACTS", "true").lower() in {"1", "true", "yes"}
USE_SHARDED_RANKING_ARTIFACTS = os.environ.get(
    "USE_SHARDED_RANKING_ARTIFACTS",
    "true",
).lower() in {"1", "true", "yes"}
SHARDED_PIPELINE_MIN_CANDIDATES_AFTER_SENIORITY = int(
    os.environ.get("SHARDED_PIPELINE_MIN_CANDIDATES_AFTER_SENIORITY", "250")
)
SHARD_DOWNLOAD_WORKERS = env_nonnegative_int("SHARD_DOWNLOAD_WORKERS", 4)
SHARD_PREFETCH_BATCH_SIZE = env_nonnegative_int("SHARD_PREFETCH_BATCH_SIZE", 4)
LOCAL_RANKING_SHARDS_DIR = Path(os.environ.get("LOCAL_RANKING_SHARDS_DIR", DATA_DIR / "shards"))
REQUIREMENTS_COLUMN = os.environ.get("REQUIREMENTS_COLUMN", "extracted_requirements")
TITLE_COLUMN = os.environ.get("TITLE_COLUMN", "title")
REQUIREMENTS_CLUSTERING_METHOD = os.environ.get("REQUIREMENTS_CLUSTERING_METHOD", "kmeans")
REQUIREMENTS_N_CLUSTERS = int(os.environ.get("REQUIREMENTS_N_CLUSTERS", "40"))
REQUIREMENTS_CLUSTER_KEEP_FRACTION = float(os.environ.get("REQUIREMENTS_CLUSTER_KEEP_FRACTION", "0.99"))
REQUIREMENTS_CLUSTER_EPS = float(os.environ.get("REQUIREMENTS_CLUSTER_EPS", "0.25"))
REQUIREMENTS_CLUSTER_MIN_SAMPLES = int(os.environ.get("REQUIREMENTS_CLUSTER_MIN_SAMPLES", "5"))
REQUIREMENTS_CLUSTER_METRIC = os.environ.get("REQUIREMENTS_CLUSTER_METRIC", "cosine")
REQUIREMENTS_CLUSTER_RANDOM_STATE = int(os.environ.get("REQUIREMENTS_CLUSTER_RANDOM_STATE", "0"))
REQUIREMENTS_CLUSTER_BATCH_SIZE = int(os.environ.get("REQUIREMENTS_CLUSTER_BATCH_SIZE", "64"))
REQUIREMENTS_CLUSTER_NORMALIZE_EMBEDDINGS = os.environ.get(
    "REQUIREMENTS_CLUSTER_NORMALIZE_EMBEDDINGS",
    "true",
).lower() in {"1", "true", "yes"}
REQUIREMENTS_CLUSTER_MIN_REMAINING_JOBS = env_nonnegative_int("REQUIREMENTS_CLUSTER_MIN_REMAINING_JOBS", 0)
REQUIREMENTS_EMBEDDING_MAX_CANDIDATES = int(
    os.environ.get("REQUIREMENTS_EMBEDDING_MAX_CANDIDATES", "100")
)
if REQUIREMENTS_EMBEDDING_MAX_CANDIDATES <= 0:
    raise ValueError("REQUIREMENTS_EMBEDDING_MAX_CANDIDATES must be positive.")
ENABLE_SENIORITY_FILTER_JOB_PRINTS = os.environ.get(
    "ENABLE_SENIORITY_FILTER_JOB_PRINTS",
    "false",
).lower() in {"1", "true", "yes", "on"}
ENABLE_UNPARSEABLE_POSTED_AT_PRINTS = os.environ.get(
    "ENABLE_UNPARSEABLE_POSTED_AT_PRINTS",
    "false",
).lower() in {"1", "true", "yes", "on"}
UNPARSEABLE_POSTED_AT_PRINT_LIMIT = int(os.environ.get("UNPARSEABLE_POSTED_AT_PRINT_LIMIT", "25"))


_jobs_df: pd.DataFrame | None = None
_title_requirements_embeddings: np.ndarray | None = None
_job_selected_word_embeddings: np.ndarray | None = None
_job_selected_word_offsets: np.ndarray | None = None
_job_phrase_chunk_embeddings: np.ndarray | None = None
_job_phrase_chunk_offsets: np.ndarray | None = None
_minilm_model: SentenceTransformer | None = None
_cross_encoder_model: CrossEncoder | None = None


@dataclass(frozen=True)
class RankingShard:
    index: int
    metadata_blob_name: str | None
    embeddings_blob_name: str | None
    metadata_path: Path | None = None
    embeddings_path: Path | None = None


def print_ranking_timing(step_name: str, started_at: float, **metadata: Any) -> None:
    record_ranking_timing(step_name, started_at, **metadata)


def safe_log_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def print_candidate_titles_and_companies(label: str, df: pd.DataFrame) -> None:
    jobs = [
        {
            "title": safe_log_text(row.get(TITLE_COLUMN, "")),
            "company": safe_log_text(row.get("company_name", "") or row.get("source_company", "")),
        }
        for _, row in df.iterrows()
    ]
    print(
        "Sharded ranking candidate snapshot: "
        f"label={label}, count={len(jobs)}, jobs={json.dumps(jobs, ensure_ascii=False)}",
        flush=True,
    )


def print_seniority_filter_job_snapshot(label: str, df: pd.DataFrame) -> None:
    if not ENABLE_SENIORITY_FILTER_JOB_PRINTS:
        return
    print_candidate_titles_and_companies(f"seniority_filter_{label}", df)


def get_max_csv_rows() -> int | None:
    value = os.environ.get("MAX_CSV_ROWS", "").strip()
    if not value:
        return None

    try:
        max_rows = int(value)
    except ValueError as exc:
        raise ValueError("MAX_CSV_ROWS must be a positive integer when set.") from exc

    if max_rows <= 0:
        raise ValueError("MAX_CSV_ROWS must be a positive integer when set.")

    return max_rows


def validate_jobs_df_columns(df: pd.DataFrame) -> None:
    required_columns = {TEXT_COLUMN}

    if USE_REQUIREMENTS_CLUSTERING:
        required_columns.update({REQUIREMENTS_COLUMN, TITLE_COLUMN})

    missing_columns = sorted(column for column in required_columns if column not in df.columns)
    if missing_columns:
        raise ValueError(
            f"Jobs CSV is missing required columns: {missing_columns}. "
            f"Found: {list(df.columns)}"
        )


def ensure_jobs_csv_available() -> Path:
    """
    Local:
        Use LOCAL_JOBS_CSV_PATH or the checked-in sample CSV under data/.

    Cloud Run:
        Must use GCS bucket CSV. No local fallback.
    """
    if RUNNING_ON_CLOUD_RUN:
        if not GCS_BUCKET_NAME:
            raise RuntimeError("Running on Cloud Run, but GCS_BUCKET_NAME is not set.")
        if not GCS_JOBS_BLOB_NAME:
            raise RuntimeError("Running on Cloud Run, but GCS_JOBS_BLOB_NAME is not set.")

        DOWNLOADED_GCS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_JOBS_BLOB_NAME)

        if not blob.exists():
            raise FileNotFoundError(
                f"Cloud Run requires GCS CSV, but file was not found: "
                f"gs://{GCS_BUCKET_NAME}/{GCS_JOBS_BLOB_NAME}"
            )

        blob.download_to_filename(str(DOWNLOADED_GCS_CSV_PATH))
        return DOWNLOADED_GCS_CSV_PATH

    if not LOCAL_JOBS_CSV_PATH.exists():
        raise FileNotFoundError(f"Local jobs CSV not found: {LOCAL_JOBS_CSV_PATH}")

    return LOCAL_JOBS_CSV_PATH


def download_gcs_blob(blob_name: str, destination_path: Path) -> Path | None:
    if not GCS_BUCKET_NAME:
        raise RuntimeError("Running on Cloud Run, but GCS_BUCKET_NAME is not set.")

    started_at = time.perf_counter()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() and destination_path.stat().st_size > 0:
        bytes_on_disk = destination_path.stat().st_size
        print(
            "Using cached GCS artifact: "
            f"blob={blob_name}, path={destination_path}, bytes={bytes_on_disk}",
            flush=True,
        )
        print_ranking_timing("gcs_blob_available", started_at, blob=blob_name, cache_hit=True, bytes=bytes_on_disk)
        return destination_path

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        print_ranking_timing("gcs_blob_available", started_at, blob=blob_name, cache_hit=False, found=False)
        return None

    blob.download_to_filename(str(destination_path))
    bytes_on_disk = destination_path.stat().st_size if destination_path.exists() else 0
    print_ranking_timing("gcs_blob_available", started_at, blob=blob_name, cache_hit=False, found=True, bytes=bytes_on_disk)
    return destination_path


def shard_index_from_name(name: str, pattern: str) -> int | None:
    match = re.search(pattern, name)
    if not match:
        return None
    return int(match.group(1))


def discover_gcs_ranking_shards() -> list[RankingShard]:
    if not GCS_BUCKET_NAME:
        raise RuntimeError("GCS_BUCKET_NAME is required for sharded ranking artifacts.")
    if not GCS_RANKING_SHARDS_PREFIX:
        raise RuntimeError("GCS_RANKING_SHARDS_PREFIX cannot be empty.")

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    prefix = f"{GCS_RANKING_SHARDS_PREFIX}/"
    metadata_by_index: dict[int, str] = {}
    embeddings_by_index: dict[int, str] = {}

    for blob in client.list_blobs(bucket, prefix=prefix):
        metadata_index = shard_index_from_name(blob.name, r"jobs_metadata_(\d+)\.jsonl$")
        if metadata_index is not None:
            metadata_by_index[metadata_index] = blob.name
            continue
        embeddings_index = shard_index_from_name(blob.name, r"job_embeddings_(\d+)\.npz$")
        if embeddings_index is not None:
            embeddings_by_index[embeddings_index] = blob.name

    shard_indexes = sorted(set(metadata_by_index) & set(embeddings_by_index))
    shards = [
        RankingShard(
            index=index,
            metadata_blob_name=metadata_by_index[index],
            embeddings_blob_name=embeddings_by_index[index],
        )
        for index in shard_indexes
    ]
    print(f"Sharded ranking artifacts discovered: shards={len(shards)}, prefix=gs://{GCS_BUCKET_NAME}/{prefix}", flush=True)
    return shards


def discover_local_ranking_shards() -> list[RankingShard]:
    if not LOCAL_RANKING_SHARDS_DIR.exists():
        return []

    metadata_by_index = {
        int(path.stem.rsplit("_", 1)[-1]): path
        for path in LOCAL_RANKING_SHARDS_DIR.glob("jobs_metadata_*.jsonl")
        if shard_index_from_name(path.name, r"jobs_metadata_(\d+)\.jsonl$") is not None
    }
    embeddings_by_index = {
        int(path.stem.rsplit("_", 1)[-1]): path
        for path in LOCAL_RANKING_SHARDS_DIR.glob("job_embeddings_*.npz")
        if shard_index_from_name(path.name, r"job_embeddings_(\d+)\.npz$") is not None
    }
    shards = [
        RankingShard(index=index, metadata_blob_name=None, embeddings_blob_name=None, metadata_path=metadata_by_index[index], embeddings_path=embeddings_by_index[index])
        for index in sorted(set(metadata_by_index) & set(embeddings_by_index))
    ]
    print(f"Local sharded ranking artifacts discovered: shards={len(shards)}, dir={LOCAL_RANKING_SHARDS_DIR}", flush=True)
    return shards


def discover_ranking_shards() -> list[RankingShard]:
    started_at = time.perf_counter()
    source = "gcs" if RUNNING_ON_CLOUD_RUN else "local"
    if RUNNING_ON_CLOUD_RUN:
        shards = discover_gcs_ranking_shards()
    else:
        shards = discover_local_ranking_shards()
    print_ranking_timing("sharded_rank_discover_shards", started_at, source=source, shards=len(shards))
    return shards


def ensure_shard_files_available(shard: RankingShard) -> tuple[Path, Path]:
    started_at = time.perf_counter()
    if shard.metadata_path and shard.embeddings_path:
        print_ranking_timing(
            "sharded_rank_shard_files_available",
            started_at,
            shard=f"{shard.index:05d}",
            source="local",
            cache_hit=True,
        )
        return shard.metadata_path, shard.embeddings_path

    if not shard.metadata_blob_name or not shard.embeddings_blob_name:
        raise RuntimeError(f"Shard {shard.index:05d} is missing GCS blob names.")

    metadata_path = DOWNLOADED_GCS_SHARDS_DIR / f"jobs_metadata_{shard.index:05d}.jsonl"
    embeddings_path = DOWNLOADED_GCS_SHARDS_DIR / f"job_embeddings_{shard.index:05d}.npz"
    metadata_was_cached = metadata_path.exists() and metadata_path.stat().st_size > 0
    embeddings_was_cached = embeddings_path.exists() and embeddings_path.stat().st_size > 0
    downloaded_metadata = download_gcs_blob(shard.metadata_blob_name, metadata_path)
    downloaded_embeddings = download_gcs_blob(shard.embeddings_blob_name, embeddings_path)
    if not downloaded_metadata or not downloaded_embeddings:
        raise FileNotFoundError(f"Could not download complete ranking shard {shard.index:05d}.")

    print_ranking_timing(
        "sharded_rank_shard_files_available",
        started_at,
        shard=f"{shard.index:05d}",
        source="gcs",
        metadata_cache_hit=metadata_was_cached,
        embeddings_cache_hit=embeddings_was_cached,
    )
    return downloaded_metadata, downloaded_embeddings


def prefetch_shard_files(shards: list[RankingShard], *, max_workers: int, batch_number: int) -> None:
    if not shards or max_workers <= 0:
        return

    started_at = time.perf_counter()
    workers = min(max_workers, len(shards))
    shard_labels = ",".join(f"{shard.index:05d}" for shard in shards)
    print(
        "Sharded ranking shard prefetch start: "
        f"batch={batch_number}, shards={len(shards)}, workers={workers}, shard_indexes={shard_labels}",
        flush=True,
    )
    try:
        if workers == 1:
            for shard in shards:
                ensure_shard_files_available(shard)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_shard = {
                    pool.submit(copy_context().run, ensure_shard_files_available, shard): shard for shard in shards
                }
                for future in as_completed(future_to_shard):
                    shard = future_to_shard[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(
                            "Sharded ranking shard prefetch failed: "
                            f"batch={batch_number}, shard={shard.index:05d}, error_type={type(exc).__name__}",
                            flush=True,
                        )
                        raise
    finally:
        print_ranking_timing(
            "sharded_rank_shard_prefetch",
            started_at,
            batch=batch_number,
            shards=len(shards),
            workers=workers,
        )


def iter_prefetched_shards(shards: list[RankingShard]) -> Iterator[RankingShard]:
    batch_size = max(1, SHARD_PREFETCH_BATCH_SIZE)
    if SHARD_DOWNLOAD_WORKERS <= 1 or batch_size <= 1 or len(shards) <= 1:
        print(
            "Sharded ranking shard prefetch disabled: "
            f"shards={len(shards)}, workers={SHARD_DOWNLOAD_WORKERS}, batch_size={batch_size}",
            flush=True,
        )
        yield from shards
        return

    for batch_number, start in enumerate(range(0, len(shards), batch_size), start=1):
        batch = shards[start : start + batch_size]
        prefetch_shard_files(batch, max_workers=SHARD_DOWNLOAD_WORKERS, batch_number=batch_number)
        yield from batch


def read_metadata_records(metadata_path: Path, *, shard_index: int) -> list[dict[str, Any]]:
    started_at = time.perf_counter()
    records: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for artifact_row_index, line in enumerate(f):
            if line.strip():
                record = json.loads(line)
                record.setdefault("artifact_shard", shard_index)
                record.setdefault("artifact_row_index", artifact_row_index)
                records.append(record)
    print_ranking_timing("sharded_rank_metadata_parse", started_at, shard=f"{shard_index:05d}", rows=len(records))
    return records


def posted_at_range_for_records(records: list[dict[str, Any]]) -> tuple[datetime | None, datetime | None]:
    posted_times = [
        posted_at
        for record in records
        if (posted_at := parse_posted_at_utc(record.get("posted_at_utc") or record.get("posted_at"))) is not None
    ]
    if not posted_times:
        return None, None
    return min(posted_times), max(posted_times)


def print_unparseable_posted_at_examples(*, shard_index: int, records: list[dict[str, Any]]) -> None:
    if not ENABLE_UNPARSEABLE_POSTED_AT_PRINTS:
        return

    examples: list[str] = []
    unparseable_count = 0
    max_examples = max(0, UNPARSEABLE_POSTED_AT_PRINT_LIMIT)
    for record in records:
        if parse_posted_at_utc(record.get("posted_at_utc")) is not None:
            continue

        posted_at = record.get("posted_at")
        text = safe_log_text(posted_at)
        if not text:
            continue
        if parse_posted_at_utc(posted_at) is not None:
            continue
        unparseable_count += 1
        if len(examples) < max_examples:
            examples.append(text)

    if unparseable_count:
        print(
            "Sharded ranking unparseable posted_at examples: "
            f"shard={shard_index:05d}, count={unparseable_count}, "
            f"printed={len(examples)}, examples={json.dumps(examples, ensure_ascii=False)}",
            flush=True,
        )


def ensure_ranking_artifacts_available() -> tuple[Path, Path] | None:
    if not USE_RANKING_ARTIFACTS:
        return None

    if RUNNING_ON_CLOUD_RUN:
        metadata_path = download_gcs_blob(GCS_METADATA_BLOB_NAME, DOWNLOADED_GCS_METADATA_PATH)
        embeddings_path = download_gcs_blob(GCS_EMBEDDINGS_BLOB_NAME, DOWNLOADED_GCS_EMBEDDINGS_PATH)
        if metadata_path and embeddings_path:
            return metadata_path, embeddings_path
        print("Ranking artifacts not found in GCS; falling back to CSV.")
        return None

    metadata_path = Path(os.environ.get("LOCAL_JOBS_METADATA_PATH", DATA_DIR / "jobs_metadata.jsonl"))
    embeddings_path = Path(os.environ.get("LOCAL_JOB_EMBEDDINGS_PATH", DATA_DIR / "job_embeddings.npz"))
    if metadata_path.exists() and embeddings_path.exists():
        return metadata_path, embeddings_path

    return None


def load_jobs_from_artifacts(metadata_path: Path, embeddings_path: Path) -> pd.DataFrame:
    global _job_phrase_chunk_embeddings
    global _job_phrase_chunk_offsets
    global _job_selected_word_embeddings
    global _job_selected_word_offsets
    global _title_requirements_embeddings

    records: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"Ranking metadata artifact has no rows: {metadata_path}")

    max_rows = get_max_csv_rows()
    if max_rows is not None:
        records = records[:max_rows]

    npz = np.load(embeddings_path, allow_pickle=False)
    embeddings = np.asarray(npz["title_requirements_embeddings"], dtype=np.float32)
    embeddings = embeddings[: len(records)]

    if len(records) != len(embeddings):
        raise ValueError(
            "Ranking artifacts are misaligned: "
            f"{len(records)} metadata rows vs {len(embeddings)} embedding rows."
        )

    df = pd.DataFrame(records)
    df["artifact_row_index"] = np.arange(len(df), dtype=np.int64)
    if "csv_row_index" not in df.columns:
        df["csv_row_index"] = df["artifact_row_index"]

    _title_requirements_embeddings = embeddings
    if {
        "job_selected_word_embeddings",
        "job_selected_word_offsets",
        "job_phrase_chunk_embeddings",
        "job_phrase_chunk_offsets",
    }.issubset(set(npz.files)):
        _job_selected_word_embeddings = np.asarray(npz["job_selected_word_embeddings"], dtype=np.float32)
        _job_selected_word_offsets = np.asarray(npz["job_selected_word_offsets"], dtype=np.int64)[: len(records) + 1]
        _job_phrase_chunk_embeddings = np.asarray(npz["job_phrase_chunk_embeddings"], dtype=np.float32)
        _job_phrase_chunk_offsets = np.asarray(npz["job_phrase_chunk_offsets"], dtype=np.int64)[: len(records) + 1]
        if len(_job_selected_word_offsets) != len(records) + 1:
            raise ValueError("job_selected_word_offsets does not align with metadata rows.")
        if len(_job_phrase_chunk_offsets) != len(records) + 1:
            raise ValueError("job_phrase_chunk_offsets does not align with metadata rows.")
        print(
            "Loaded precomputed ranking embeddings: "
            f"{len(_job_selected_word_embeddings)} word vectors, "
            f"{len(_job_phrase_chunk_embeddings)} phrase vectors."
        )
    else:
        _job_selected_word_embeddings = None
        _job_selected_word_offsets = None
        _job_phrase_chunk_embeddings = None
        _job_phrase_chunk_offsets = None
        print("Ranking artifacts do not include word/phrase embeddings; runtime fallback may embed job-side text.")
    print(f"Using ranking artifacts: {metadata_path}, {embeddings_path}")
    return df


def get_jobs_df() -> pd.DataFrame:
    global _jobs_df

    if _jobs_df is None:
        artifact_paths = ensure_ranking_artifacts_available()
        if artifact_paths:
            df = load_jobs_from_artifacts(*artifact_paths)
        else:
            jobs_csv_path = ensure_jobs_csv_available()
            print(f"Using jobs CSV: {jobs_csv_path}")

            # Optional Cloud Run/local safety valve for large CSVs.
            df = pd.read_csv(jobs_csv_path, nrows=get_max_csv_rows())

        validate_jobs_df_columns(df)

        df = df.copy()
        df["csv_row_index"] = df.index
        if "min_years_experience" not in df.columns:
            df["min_years_experience"] = np.nan
        df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).map(str.strip)
        df = df[df[TEXT_COLUMN] != ""].reset_index(drop=True)

        if df.empty:
            raise ValueError("CSV has no non-empty job descriptions.")

        _jobs_df = df

    return _jobs_df.copy()


def get_minilm_model() -> SentenceTransformer:
    global _minilm_model

    if _minilm_model is None:
        _minilm_model = load_minilm_model()

    return _minilm_model


def get_cross_encoder_model() -> CrossEncoder:
    global _cross_encoder_model

    if _cross_encoder_model is None:
        _cross_encoder_model = load_cross_encoder_model()

    return _cross_encoder_model


def get_cross_encoder_model_or_none() -> CrossEncoder | None:
    if not ENABLE_CROSS_ENCODER:
        print("Cross-encoder disabled by ENABLE_CROSS_ENCODER=false.")
        return None

    try:
        return get_cross_encoder_model()
    except Exception as exc:
        print(
            "ALERT: Cross-encoder model unavailable; final ranking will use fallback policy. "
            f"error_type={type(exc).__name__}",
            flush=True,
        )
        return None


def apply_hard_filters(
    df: pd.DataFrame,
    *,
    country: str | None,
    state: str | None,
    max_required_yoe: float | None,
    exclude_security_clearance: bool = False,
    require_recent_posted: bool = False,
) -> pd.DataFrame:
    filtered = df.copy()

    if (country and country.strip()) or (state and state.strip()):
        filtered = filter_jobs_df_by_country(
            jobs_df=filtered,
            country=country.strip() if country else None,
            state=state.strip() if state else None,
            location_column=LOCATION_COLUMN,
        )

    filtered = filter_jobs_df_by_max_required_yoe(filtered, max_required_yoe)
    filtered = filter_jobs_df_excluding_security_clearance(
        filtered,
        exclude_security_clearance=exclude_security_clearance,
    )
    if require_recent_posted:
        jobs_before_recent_posted = len(filtered)
        filtered = filter_recently_posted_jobs_df(
            filtered,
            enabled=True,
            hours=RECENT_POSTED_HOURS,
        )
        print(
            "Hard filter recent posted complete: "
            f"enabled=True, hours={RECENT_POSTED_HOURS}, "
            f"jobs_before={jobs_before_recent_posted}, jobs_after={len(filtered)}, "
            f"removed={jobs_before_recent_posted - len(filtered)}",
            flush=True,
        )
    filtered = filtered.reset_index(drop=True)

    return filtered


def apply_requirements_cluster_filter(
    df: pd.DataFrame,
    *,
    resume_text: str,
    precomputed_resume_embedding: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not USE_REQUIREMENTS_CLUSTERING:
        return df.copy().reset_index(drop=True), {"enabled": False}

    if not 0 < REQUIREMENTS_CLUSTER_KEEP_FRACTION <= 1:
        raise ValueError("REQUIREMENTS_CLUSTER_KEEP_FRACTION must be in (0, 1].")

    if _title_requirements_embeddings is not None and "artifact_row_index" in df.columns:
        artifact_indices = df["artifact_row_index"].astype(int).to_numpy()
        job_embeddings = _title_requirements_embeddings[artifact_indices]
        resume_embedding = (
            np.asarray(precomputed_resume_embedding, dtype=np.float32)
            if precomputed_resume_embedding is not None
            else get_minilm_model().encode(
                [resume_text],
                normalize_embeddings=REQUIREMENTS_CLUSTER_NORMALIZE_EMBEDDINGS,
                show_progress_bar=False,
            )[0].astype(np.float32)
        )
        distances = 1.0 - np.clip(job_embeddings @ resume_embedding, -1.0, 1.0)
        ranked_positions = np.argsort(distances).tolist()
        operation_result = {
            "operation_name": "requirements_embedding",
            "status": "ok",
            "ranked_job_ids": ranked_positions,
            "job_metrics": {
                int(position): {
                    "rank": rank,
                    "score": float(distances[position]),
                    "score_direction": "lower_is_better",
                    "raw_metrics": {
                        "requirements_embedding_distance": float(distances[position]),
                    },
                }
                for rank, position in enumerate(ranked_positions, start=1)
            },
            "error": None,
        }
        keep_positions = reduce_job_ids(
            current_job_ids=list(range(len(df))),
            operation_result=operation_result,
            reduction_policies={
                "requirements_embedding": {
                    "top_n": REQUIREMENTS_EMBEDDING_MAX_CANDIDATES,
                    "top_fraction": REQUIREMENTS_CLUSTER_KEEP_FRACTION,
                    "min_remaining_jobs": REQUIREMENTS_CLUSTER_MIN_REMAINING_JOBS,
                }
            },
        )
        keep_positions = keep_positions[:REQUIREMENTS_EMBEDDING_MAX_CANDIDATES]

        reduced_df = df.iloc[keep_positions].copy()
        reduced_df["requirements_embedding_distance"] = [float(distances[position]) for position in keep_positions]
        reduced_df["requirements_cluster_rank"] = [
            operation_result["job_metrics"][int(position)]["rank"] for position in keep_positions
        ]
        reduced_df["requirements_cluster_distance"] = reduced_df["requirements_embedding_distance"]
        reduced_df["requirements_cluster_size"] = len(df)
        reduced_df = reduced_df.sort_values("requirements_embedding_distance").reset_index(drop=True)

        metadata = {
            "enabled": True,
            "method": "precomputed_title_requirements_embedding",
            "jobs_before_cluster_filter": len(df),
            "jobs_after_cluster_filter": len(reduced_df),
            "kept_fraction": REQUIREMENTS_CLUSTER_KEEP_FRACTION,
            "max_candidates": REQUIREMENTS_EMBEDDING_MAX_CANDIDATES,
        }
        return reduced_df, metadata

    if precomputed_resume_embedding is not None:
        raise RuntimeError("Cached resume ranking requires precomputed title/requirements job embeddings.")

    cluster_result = cluster_jobs_by_requirements_df(
        jobs_df=df,
        resume_text=resume_text,
        model=get_minilm_model(),
        requirements_column=REQUIREMENTS_COLUMN,
        title_column=TITLE_COLUMN,
        clustering_method=REQUIREMENTS_CLUSTERING_METHOD,
        eps=REQUIREMENTS_CLUSTER_EPS,
        min_samples=REQUIREMENTS_CLUSTER_MIN_SAMPLES,
        n_clusters=min(REQUIREMENTS_N_CLUSTERS, len(df)),
        metric=REQUIREMENTS_CLUSTER_METRIC,
        random_state=REQUIREMENTS_CLUSTER_RANDOM_STATE,
        embedding_batch_size=REQUIREMENTS_CLUSTER_BATCH_SIZE,
        normalize_embeddings=REQUIREMENTS_CLUSTER_NORMALIZE_EMBEDDINGS,
    )

    clusters = cluster_result["clusters"]

    if not clusters:
        raise ValueError("Requirements clustering produced no ranked clusters.")

    rank_by_label = {
        int(cluster["cluster_label"]): int(cluster["cluster_rank"])
        for cluster in clusters
    }
    dist_by_label = {
        int(cluster["cluster_label"]): float(cluster["resume_to_centroid_distance"])
        for cluster in clusters
    }
    size_by_label = {
        int(cluster["cluster_label"]): int(cluster["cluster_size"])
        for cluster in clusters
    }

    valid_df = cluster_result["valid_df"].copy()
    ranked_job_ids = sorted(
        [
            position
            for position in range(len(valid_df))
            if int(valid_df.iloc[position]["cluster_label"]) in rank_by_label
        ],
        key=lambda position: (
            rank_by_label.get(int(valid_df.iloc[position]["cluster_label"]), len(clusters) + 1),
            position,
        ),
    )
    operation_result = {
        "operation_name": "requirements_clustering",
        "status": "ok",
        "ranked_job_ids": ranked_job_ids,
        "job_metrics": {
            int(position): {
                "rank": rank_by_label[int(valid_df.iloc[position]["cluster_label"])],
                "score": dist_by_label[int(valid_df.iloc[position]["cluster_label"])],
                "score_direction": "lower_is_better",
                "raw_metrics": {
                    "cluster_label": int(valid_df.iloc[position]["cluster_label"]),
                    "cluster_size": size_by_label[int(valid_df.iloc[position]["cluster_label"])],
                    "resume_to_centroid_distance": dist_by_label[int(valid_df.iloc[position]["cluster_label"])],
                },
            }
            for position in ranked_job_ids
        },
        "error": None,
    }
    keep_positions = reduce_job_ids(
        current_job_ids=list(range(len(valid_df))),
        operation_result=operation_result,
        reduction_policies={
            "requirements_clustering": {
                "top_cluster_fraction": REQUIREMENTS_CLUSTER_KEEP_FRACTION,
                "min_remaining_jobs": REQUIREMENTS_CLUSTER_MIN_REMAINING_JOBS,
            }
        },
    )
    kept_labels = {
        int(valid_df.iloc[position]["cluster_label"])
        for position in keep_positions
    }

    reduced_df = valid_df.iloc[keep_positions].copy()

    reduced_df["requirements_cluster_rank"] = reduced_df["cluster_label"].map(rank_by_label)
    reduced_df["requirements_cluster_distance"] = reduced_df["cluster_label"].map(dist_by_label)
    reduced_df["requirements_cluster_size"] = reduced_df["cluster_label"].map(size_by_label)
    reduced_df = reduced_df.reset_index(drop=True)

    if reduced_df.empty:
        raise ValueError("No jobs remained after requirements cluster filtering.")

    cluster_rankings = [
        {
            "cluster_rank": int(cluster["cluster_rank"]),
            "cluster_label": int(cluster["cluster_label"]),
            "cluster_size": int(cluster["cluster_size"]),
            "resume_to_centroid_distance": float(cluster["resume_to_centroid_distance"]),
            "kept": int(cluster["cluster_label"]) in kept_labels,
        }
        for cluster in clusters
    ]

    metadata = {
        "enabled": True,
        "method": cluster_result["clustering_method"],
        "requested_clusters": REQUIREMENTS_N_CLUSTERS,
        "actual_clusters": len(clusters),
        "kept_clusters": len(kept_labels),
        "removed_clusters": len(clusters) - len(kept_labels),
        "jobs_before_cluster_filter": len(df),
        "jobs_after_cluster_filter": len(reduced_df),
        "noise_jobs": int(cluster_result["noise_count"]),
        "cluster_rankings": cluster_rankings,
    }

    return reduced_df, metadata


def apply_requirements_embedding_prefilter(
    df: pd.DataFrame,
    *,
    resume_text: str,
    title_requirements_embeddings: np.ndarray,
    precomputed_resume_embedding: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    if not USE_REQUIREMENTS_CLUSTERING:
        return df.copy().reset_index(drop=True), {"enabled": False}, title_requirements_embeddings

    if not 0 < REQUIREMENTS_CLUSTER_KEEP_FRACTION <= 1:
        raise ValueError("REQUIREMENTS_CLUSTER_KEEP_FRACTION must be in (0, 1].")

    resume_embedding = (
        np.asarray(precomputed_resume_embedding, dtype=np.float32)
        if precomputed_resume_embedding is not None
        else get_minilm_model().encode(
            [resume_text],
            normalize_embeddings=REQUIREMENTS_CLUSTER_NORMALIZE_EMBEDDINGS,
            show_progress_bar=False,
        )[0].astype(np.float32)
    )
    distances = 1.0 - np.clip(title_requirements_embeddings @ resume_embedding, -1.0, 1.0)
    ranked_positions = np.argsort(distances).tolist()
    operation_result = {
        "operation_name": "requirements_embedding",
        "status": "ok",
        "ranked_job_ids": ranked_positions,
        "job_metrics": {
            int(position): {
                "rank": rank,
                "score": float(distances[position]),
                "score_direction": "lower_is_better",
                "raw_metrics": {"requirements_embedding_distance": float(distances[position])},
            }
            for rank, position in enumerate(ranked_positions, start=1)
        },
        "error": None,
    }
    keep_positions = reduce_job_ids(
        current_job_ids=list(range(len(df))),
        operation_result=operation_result,
        reduction_policies={
            "requirements_embedding": {
                "top_n": REQUIREMENTS_EMBEDDING_MAX_CANDIDATES,
                "top_fraction": REQUIREMENTS_CLUSTER_KEEP_FRACTION,
                "min_remaining_jobs": REQUIREMENTS_CLUSTER_MIN_REMAINING_JOBS,
            }
        },
    )
    keep_positions = keep_positions[:REQUIREMENTS_EMBEDDING_MAX_CANDIDATES]

    reduced_df = df.iloc[keep_positions].copy()
    reduced_df["_kept_position"] = keep_positions
    reduced_df["requirements_embedding_distance"] = [float(distances[position]) for position in keep_positions]
    reduced_df["requirements_cluster_rank"] = [
        operation_result["job_metrics"][int(position)]["rank"] for position in keep_positions
    ]
    reduced_df["requirements_cluster_distance"] = reduced_df["requirements_embedding_distance"]
    reduced_df["requirements_cluster_size"] = len(df)
    reduced_df = reduced_df.sort_values("requirements_embedding_distance").reset_index(drop=True)
    kept_embeddings = title_requirements_embeddings[reduced_df["_kept_position"].astype(int).to_numpy()].astype(
        np.float32,
        copy=True,
    )
    reduced_df = reduced_df.drop(columns=["_kept_position"])

    return (
        reduced_df,
        {
            "enabled": True,
            "method": "precomputed_title_requirements_embedding",
            "jobs_before_cluster_filter": len(df),
            "jobs_after_cluster_filter": len(reduced_df),
            "kept_fraction": REQUIREMENTS_CLUSTER_KEEP_FRACTION,
            "max_candidates": REQUIREMENTS_EMBEDDING_MAX_CANDIDATES,
        },
        kept_embeddings,
    )


def run_seniority_prefilter(
    df: pd.DataFrame,
    *,
    resume_text: str,
    title_requirements_embeddings: np.ndarray,
    precomputed_resume_embedding: np.ndarray | None = None,
    precomputed_anchor_embeddings: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if df.empty:
        return df, title_requirements_embeddings

    job_requirements = (
        df[REQUIREMENTS_COLUMN].fillna("").astype(str).tolist()
        if REQUIREMENTS_COLUMN in df.columns
        else df[TEXT_COLUMN].fillna("").astype(str).tolist()
    )
    job_titles = df[TITLE_COLUMN].fillna("").astype(str).tolist() if TITLE_COLUMN in df.columns else None
    job_min_years_experience = (
        df["min_years_experience"].tolist()
        if "min_years_experience" in df.columns
        else None
    )
    seniority_result = run_seniority_filter_operation(
        job_ids=list(range(len(df))),
        resume_text=resume_text,
        minilm_model=(
            None
            if precomputed_resume_embedding is not None and precomputed_anchor_embeddings is not None
            else get_minilm_model()
        ),
        job_titles=job_titles,
        job_requirements=job_requirements,
        job_min_years_experience=job_min_years_experience,
        precomputed_title_requirements_embeddings=title_requirements_embeddings,
        max_gap=float(os.environ.get("SENIORITY_FILTER_MAX_GAP", "1.5")),
        max_junior_gap=float(os.environ.get("SENIORITY_FILTER_MAX_JUNIOR_GAP", "10.0")),
        enabled=os.environ.get("ENABLE_SENIORITY_FILTER", "true").lower() in {"1", "true", "yes"},
        level_probability_alpha=float(os.environ.get("SENIORITY_FILTER_LEVEL_PROBABILITY_ALPHA", "3.0")),
        batch_size=128,
        precomputed_resume_embedding=precomputed_resume_embedding,
        precomputed_anchor_embeddings=precomputed_anchor_embeddings,
    )
    survivors = reduce_job_ids(
        current_job_ids=list(range(len(df))),
        operation_result=seniority_result,
        reduction_policies={"seniority_filter": {"filter_raw_metric": "is_filtered", "exclude_value": True}},
    )
    survivor_set = set(survivors)
    removed = [index for index in range(len(df)) if index not in survivor_set]
    print_seniority_filter_job_snapshot("removed", df.iloc[removed].copy().reset_index(drop=True))
    print_seniority_filter_job_snapshot("remaining", df.iloc[survivors].copy().reset_index(drop=True))
    return df.iloc[survivors].copy().reset_index(drop=True), title_requirements_embeddings[survivors].astype(np.float32, copy=True)


def parse_list_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if pd.isna(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [part.strip() for part in text.split("|") if part.strip()]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    return []


def precomputed_lists_from_column(df: pd.DataFrame, column: str) -> list[list[str]] | None:
    if column not in df.columns:
        return None
    values = [parse_list_value(value) for value in df[column].tolist()]
    return values if any(values) else None


def embedding_matrices_for_filtered_rows(
    df: pd.DataFrame,
    embeddings: np.ndarray | None,
    offsets: np.ndarray | None,
) -> list[np.ndarray] | None:
    if embeddings is None or offsets is None or "artifact_row_index" not in df.columns:
        return None

    matrices: list[np.ndarray] = []
    for artifact_index in df["artifact_row_index"].astype(int).tolist():
        start = int(offsets[artifact_index])
        end = int(offsets[artifact_index + 1])
        matrices.append(np.asarray(embeddings[start:end], dtype=np.float32))
    return matrices


def load_shard_title_embeddings(
    embeddings_path: Path,
    *,
    row_count: int,
    shard_index: int | None = None,
) -> np.ndarray:
    started_at = time.perf_counter()
    with np.load(embeddings_path, allow_pickle=False) as npz:
        embeddings = np.asarray(npz["title_requirements_embeddings"], dtype=np.float32)[:row_count].copy()
    if len(embeddings) != row_count:
        raise ValueError(
            f"Shard title embeddings are misaligned: {row_count} metadata rows vs {len(embeddings)} embedding rows."
        )
    print_ranking_timing(
        "sharded_rank_title_embedding_load",
        started_at,
        shard=f"{shard_index:05d}" if shard_index is not None else "unknown",
        rows=row_count,
        embedding_dim=embeddings.shape[1] if embeddings.ndim == 2 else 0,
    )
    return embeddings


def load_selected_embedding_matrices(
    embeddings_path: Path,
    *,
    artifact_row_indexes: list[int],
    shard_index: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    started_at = time.perf_counter()
    word_matrices: list[np.ndarray] = []
    phrase_matrices: list[np.ndarray] = []

    with np.load(embeddings_path, allow_pickle=False) as npz:
        word_embeddings = np.asarray(npz["job_selected_word_embeddings"], dtype=np.float32)
        word_offsets = np.asarray(npz["job_selected_word_offsets"], dtype=np.int64)
        phrase_embeddings = np.asarray(npz["job_phrase_chunk_embeddings"], dtype=np.float32)
        phrase_offsets = np.asarray(npz["job_phrase_chunk_offsets"], dtype=np.int64)

        for artifact_row_index in artifact_row_indexes:
            word_start = int(word_offsets[artifact_row_index])
            word_end = int(word_offsets[artifact_row_index + 1])
            phrase_start = int(phrase_offsets[artifact_row_index])
            phrase_end = int(phrase_offsets[artifact_row_index + 1])
            word_matrices.append(word_embeddings[word_start:word_end].astype(np.float32, copy=True))
            phrase_matrices.append(phrase_embeddings[phrase_start:phrase_end].astype(np.float32, copy=True))

    print_ranking_timing(
        "sharded_rank_selected_embedding_load",
        started_at,
        shard=f"{shard_index:05d}" if shard_index is not None else "unknown",
        selected_jobs=len(artifact_row_indexes),
        word_matrices=len(word_matrices),
        phrase_matrices=len(phrase_matrices),
    )
    return word_matrices, phrase_matrices


def title_requirements_embeddings_for_filtered_rows(df: pd.DataFrame) -> np.ndarray | None:
    if _title_requirements_embeddings is None or "artifact_row_index" not in df.columns:
        return None

    artifact_indices = df["artifact_row_index"].astype(int).to_numpy()
    return np.asarray(_title_requirements_embeddings[artifact_indices], dtype=np.float32)


def full_artifact_phrase_comparison_inputs(
    df: pd.DataFrame,
    limit: int,
) -> tuple[list[int], list[list[str]] | None, list[np.ndarray] | None]:
    if limit <= 0:
        raise ValueError("RESUME_PHRASE_JOB_COVERAGE_COMPARISON_JOBS must be positive.")

    comparison_df = df.head(min(limit, len(df))).copy()
    if "artifact_row_index" not in comparison_df.columns:
        print("Resume phrase job coverage comparison skipped: full artifact row indices are unavailable.")
        return [], None, None

    chunks = precomputed_lists_from_column(comparison_df, "job_phrase_chunks")
    embeddings = embedding_matrices_for_filtered_rows(
        comparison_df,
        _job_phrase_chunk_embeddings,
        _job_phrase_chunk_offsets,
    )
    if chunks is None or embeddings is None:
        print("Resume phrase job coverage comparison skipped: full artifact phrase data is unavailable.")
        return [], None, None

    return list(range(len(comparison_df))), chunks, embeddings


def job_url_from_row(row: pd.Series) -> str:
    for col in ["job_url", "absolute_url", "url"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()

    return ""


def safe_row_value(row: pd.Series, col: str) -> Any:
    if col not in row:
        return ""

    value = row[col]

    if pd.isna(value):
        return ""

    return value


def package_results(
    *,
    results: list[dict[str, Any]],
    more_results: list[dict[str, Any]] | None = None,
    filtered_df: pd.DataFrame,
    total_jobs: int,
    top_k_to_show: int,
    cluster_filter: dict[str, Any] | None = None,
    include_cross_encoder_metrics: bool = True,
) -> dict[str, Any]:
    print(
        "Package results start: "
        f"results={len(results)}, more_results={len(more_results or [])}, "
        f"filtered_rows={len(filtered_df)}, include_cross_encoder_metrics={include_cross_encoder_metrics}",
        flush=True,
    )

    def package_rows(
        rows: list[dict[str, Any]],
        *,
        limit: int | None = None,
        excluded_job_indexes: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        packaged: list[dict[str, Any]] = []
        excluded_job_indexes = excluded_job_indexes or set()
        selected_rows = rows[:limit] if limit is not None else rows

        for result in selected_rows:
            job_idx = int(result["job_index"])
            if job_idx in excluded_job_indexes:
                continue
            if job_idx < 0 or job_idx >= len(filtered_df):
                print(
                    "ALERT: package_results job index out of range: "
                    f"job_index={job_idx}, filtered_rows={len(filtered_df)}, "
                    f"result_keys={sorted(result.keys())}",
                    flush=True,
                )
            meta = filtered_df.iloc[job_idx]

            packaged.append(
                {
                    "final_rank": result.get("final_rank"),
                    "title": safe_row_value(meta, "title"),
                    "company": safe_row_value(meta, "company_name"),
                    "location": safe_row_value(meta, LOCATION_COLUMN),
                    "posted_at": safe_row_value(meta, "posted_at"),
                    "posted_at_utc": safe_row_value(meta, "posted_at_utc"),
                    "posted_at_display": format_posted_at_display(safe_row_value(meta, "posted_at_utc")),
                    "csv_row_index": safe_row_value(meta, "csv_row_index"),
                    "url": job_url_from_row(meta),
                    "stage1_rank": result.get("stage1_rank"),
                    "stage1_word_wasserstein_distance": result.get("stage1_word_wasserstein_distance"),
                    "stage2_rank": result.get("stage2_rank"),
                    "stage2_phrase_wasserstein_distance": result.get("stage2_phrase_wasserstein_distance"),
                    "seniority_score": result.get("seniority_score"),
                    "seniority_gap": result.get("seniority_gap"),
                    "job_title_seniority_floor": result.get("job_title_seniority_floor"),
                    "job_title_seniority_floor_label": result.get("job_title_seniority_floor_label"),
                    "job_title_seniority_floor_applied": result.get("job_title_seniority_floor_applied"),
                    "job_title_seniority_ceiling": result.get("job_title_seniority_ceiling"),
                    "job_title_seniority_ceiling_label": result.get("job_title_seniority_ceiling_label"),
                    "job_title_seniority_ceiling_applied": result.get("job_title_seniority_ceiling_applied"),
                    "job_yoe_seniority_floor": result.get("job_yoe_seniority_floor"),
                    "job_yoe_seniority_floor_label": result.get("job_yoe_seniority_floor_label"),
                    "job_yoe_seniority_floor_applied": result.get("job_yoe_seniority_floor_applied"),
                    "mahalanobis_distance": result.get("mahalanobis_distance"),
                    "technology_overlap_score": result.get("technology_overlap_score"),
                    "technology_filter_removed": result.get("technology_filter_removed"),
                    "technology_filter_reason": result.get("technology_filter_reason"),
                    "job_technologies": result.get("job_technologies") or [],
                    "job_technology_categories": result.get("job_technology_categories") or [],
                    "resume_technologies": result.get("resume_technologies") or [],
                    "resume_technology_categories": result.get("resume_technology_categories") or [],
                    "technology_category_overlap": result.get("technology_category_overlap") or [],
                    "resume_phrase_job_coverage_percent_flagged": result.get(
                        "resume_phrase_job_coverage_percent_flagged"
                    ),
                    "resume_phrase_coverage_bad_match_percent": result.get(
                        "resume_phrase_coverage_bad_match_percent"
                    ),
                    "resume_phrase_job_coverage_bad_match_percent": result.get(
                        "resume_phrase_job_coverage_bad_match_percent"
                    ),
                    "years_experience_raw": safe_row_value(meta, "years_experience_raw"),
                    "min_years_experience": safe_row_value(meta, "min_years_experience"),
                    "experience_type": safe_row_value(meta, "experience_type"),
                    "evidence_text": safe_row_value(meta, "evidence_text"),
                    "requirements_cluster_label": safe_row_value(meta, "cluster_label"),
                    "requirements_cluster_rank": safe_row_value(meta, "requirements_cluster_rank"),
                    "requirements_cluster_distance": safe_row_value(meta, "requirements_cluster_distance"),
                    "requirements_cluster_size": safe_row_value(meta, "requirements_cluster_size"),
                    "job_description_preview": str(safe_row_value(meta, TEXT_COLUMN))[:2500],
                }
            )
            if include_cross_encoder_metrics:
                packaged[-1]["cross_encoder_score"] = result.get("cross_encoder_score")

        return packaged

    def group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: list[dict[str, Any]] = []
        company_groups: dict[str, dict[str, Any]] = {}

        for row in rows:
            company = str(row.get("company") or "Unknown company").strip() or "Unknown company"
            group = company_groups.get(company)
            if group is None:
                group = {
                    "company": company,
                    "results": [],
                    "result_count": 0,
                    "best_rank": row.get("final_rank"),
                    "rank_range": "",
                }
                company_groups[company] = group
                grouped.append(group)

            group["results"].append(row)
            group["result_count"] += 1

        for group in grouped:
            ranks = [
                row.get("final_rank")
                for row in group["results"]
                if row.get("final_rank") is not None
            ]
            if ranks:
                group["best_rank"] = min(ranks)
                group["rank_range"] = (
                    f"#{min(ranks)}"
                    if min(ranks) == max(ranks)
                    else f"#{min(ranks)}-#{max(ranks)}"
                )

        return grouped

    packaged_rows = package_rows(results, limit=top_k_to_show)
    top_job_indexes = {int(row["job_index"]) for row in results[:top_k_to_show] if "job_index" in row}
    more_packaged_rows = package_rows(
        more_results or [],
        excluded_job_indexes=top_job_indexes,
    )

    grouped_rows = group_rows(packaged_rows)
    more_grouped_rows = group_rows(more_packaged_rows)

    return {
        "total_jobs": total_jobs,
        "filtered_jobs": len(filtered_df),
        "requirements_cluster_filter": cluster_filter or {"enabled": False},
        "results": packaged_rows,
        "grouped_results": grouped_rows,
        "more_results": more_packaged_rows,
        "more_grouped_results": more_grouped_rows,
    }


def run_multi_stage_and_package(
    *,
    resume_text: str,
    filtered_df: pd.DataFrame,
    total_jobs: int,
    top_k_to_show: int,
    cluster_filter: dict[str, Any],
    full_jobs_df: pd.DataFrame | None = None,
    precomputed_title_requirements_embeddings: np.ndarray | None = None,
    precomputed_word_embeddings: list[np.ndarray] | None = None,
    precomputed_phrase_embeddings: list[np.ndarray] | None = None,
    precomputed_resume_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    step_started_at = time.perf_counter()
    job_descriptions = filtered_df[TEXT_COLUMN].tolist()
    job_titles = (
        filtered_df[TITLE_COLUMN].fillna("").astype(str).tolist()
        if TITLE_COLUMN in filtered_df.columns
        else None
    )
    job_requirements = (
        filtered_df[REQUIREMENTS_COLUMN].fillna("").astype(str).tolist()
        if REQUIREMENTS_COLUMN in filtered_df.columns
        else [""] * len(filtered_df)
    )
    job_companies = (
        filtered_df["company_name"].fillna("").astype(str).tolist()
        if "company_name" in filtered_df.columns
        else None
    )
    job_min_years_experience = (
        filtered_df["min_years_experience"].tolist()
        if "min_years_experience" in filtered_df.columns
        else None
    )
    precomputed_word_lists = precomputed_lists_from_column(filtered_df, "job_selected_words")
    if precomputed_title_requirements_embeddings is None:
        precomputed_title_requirements_embeddings = title_requirements_embeddings_for_filtered_rows(filtered_df)
    precomputed_phrase_chunks = precomputed_lists_from_column(filtered_df, "job_phrase_chunks")
    if precomputed_word_embeddings is None:
        precomputed_word_embeddings = embedding_matrices_for_filtered_rows(
            filtered_df,
            _job_selected_word_embeddings,
            _job_selected_word_offsets,
        )
    if precomputed_phrase_embeddings is None:
        precomputed_phrase_embeddings = embedding_matrices_for_filtered_rows(
            filtered_df,
            _job_phrase_chunk_embeddings,
            _job_phrase_chunk_offsets,
        )
    if full_jobs_df is None and precomputed_phrase_chunks is not None and precomputed_phrase_embeddings is not None:
        comparison_count = min(RESUME_PHRASE_JOB_COVERAGE_COMPARISON_JOBS, len(filtered_df))
        comparison_job_ids = list(range(comparison_count))
        comparison_phrase_chunks = precomputed_phrase_chunks[:comparison_count]
        comparison_phrase_embeddings = precomputed_phrase_embeddings[:comparison_count]
    else:
        comparison_source_df = full_jobs_df if full_jobs_df is not None else filtered_df
        (
            comparison_job_ids,
            comparison_phrase_chunks,
            comparison_phrase_embeddings,
        ) = full_artifact_phrase_comparison_inputs(
            comparison_source_df,
            RESUME_PHRASE_JOB_COVERAGE_COMPARISON_JOBS,
        )
    requirements_cluster_distances = (
        filtered_df["requirements_cluster_distance"].tolist()
        if "requirements_cluster_distance" in filtered_df.columns
        else None
    )
    print_ranking_timing("prepare_ranking_inputs", step_started_at, jobs=len(job_descriptions))

    if precomputed_word_embeddings is not None and precomputed_phrase_embeddings is not None:
        print("Using precomputed job-side word and phrase embeddings for ranking.")
    else:
        print("Precomputed job-side word/phrase embeddings unavailable; ranking may embed job-side text at runtime.")

    step_started_at = time.perf_counter()
    ranking_output = rank_jobs_multi_stage(
        resume_text=resume_text,
        job_descriptions=job_descriptions,
        job_titles=job_titles,
        job_requirements=job_requirements,
        job_min_years_experience=job_min_years_experience,
        job_companies=job_companies,
        minilm_model=get_minilm_model() if precomputed_resume_profile is None else None,
        cross_encoder_model=get_cross_encoder_model_or_none() if precomputed_resume_profile is None else None,
        enable_cross_encoder=ENABLE_CROSS_ENCODER and precomputed_resume_profile is None,
        word_keep_n=min(MULTI_STAGE_WORD_KEEP, len(job_descriptions)),
        phrase_keep_n=min(MULTI_STAGE_PHRASE_KEEP, len(job_descriptions)),
        word_custom_stopwords={"preferred", "required", "qualification", "qualifications"},
        word_use_stopword_filter=True,
        word_use_frequent_word_filter=False,
        word_max_count=4,
        word_use_tfidf_filter=True,
        word_top_tfidf_fraction=0.25,
        word_max_words_per_job=50,
        word_deduplicate_resume_words=True,
        word_n_projections=50,
        word_random_state=0,
        word_embedding_batch_size=128,
        phrase_n_projections=64,
        phrase_random_state=0,
        phrase_batch_size=64,
        phrase_normalize_embeddings=False,
        phrase_min_chunk_words=3,
        phrase_max_chunk_words=24,
        phrase_include_sentences=True,
        phrase_include_sentence_windows=True,
        ce_chunk_size_words=220,
        ce_chunk_overlap_words=40,
        ce_aggregation="top_k_mean",
        ce_top_k=2,
        ce_batch_size=32,
        cross_encoder_union_top_k_per_ranker=CROSS_ENCODER_UNION_TOP_K_PER_RANKER,
        poor_match_max_rank_per_step=PIPELINE_POOR_MATCH_MAX_RANK,
        mahalanobis_input_top_k=MAHALANOBIS_INPUT_TOP_K,
        mahalanobis_remove_bottom_fraction=MAHALANOBIS_REMOVE_BOTTOM_FRACTION,
        multi_metric_bad_fit_bottom_fraction=MULTI_METRIC_BAD_FIT_BOTTOM_FRACTION,
        enable_technology_mismatch_filter=ENABLE_TECHNOLOGY_MISMATCH_FILTER,
        tech_filter_min_job_types=TECH_FILTER_MIN_JOB_TYPES,
        tech_filter_max_job_type_overlap_ratio=TECH_FILTER_MAX_JOB_TYPE_OVERLAP_RATIO,
        precomputed_job_word_lists=precomputed_word_lists,
        precomputed_title_requirements_embeddings=precomputed_title_requirements_embeddings,
        precomputed_job_phrase_chunks=precomputed_phrase_chunks,
        precomputed_job_word_embeddings=precomputed_word_embeddings,
        precomputed_job_phrase_embeddings=precomputed_phrase_embeddings,
        resume_phrase_job_coverage_comparison_job_ids=comparison_job_ids,
        resume_phrase_job_coverage_comparison_phrase_chunks=comparison_phrase_chunks,
        resume_phrase_job_coverage_comparison_phrase_embeddings=comparison_phrase_embeddings,
        requirements_cluster_distances=requirements_cluster_distances,
        resume_phrase_coverage_dataset_dir=RESUME_DATASET_DIR,
        resume_phrase_coverage_flag_percentile=RESUME_PHRASE_DISTANCE_FLAG_PERCENTILE,
        resume_phrase_coverage_bad_match_percentile=RESUME_PHRASE_BAD_MATCH_PERCENTILE,
        resume_phrase_coverage_job_flag_fraction=RESUME_PHRASE_JOB_FLAG_FRACTION,
        resume_phrase_coverage_min_chunk_words=3,
        resume_phrase_coverage_max_chunk_words=24,
        resume_phrase_coverage_include_sentences=True,
        resume_phrase_coverage_include_sentence_windows=True,
        resume_phrase_coverage_batch_size=64,
        resume_phrase_coverage_normalize_embeddings=False,
        resume_phrase_job_coverage_flag_percentile=RESUME_PHRASE_JOB_COVERAGE_FLAG_PERCENTILE,
        resume_phrase_job_coverage_bad_match_percentile=RESUME_PHRASE_JOB_COVERAGE_BAD_MATCH_PERCENTILE,
        resume_phrase_coverage_remove_bottom_good_fit_fraction=RESUME_PHRASE_COVERAGE_REMOVE_BOTTOM_GOOD_FIT_FRACTION,
        resume_phrase_coverage_remove_top_bad_match_fraction=RESUME_PHRASE_COVERAGE_REMOVE_TOP_BAD_MATCH_FRACTION,
        resume_phrase_job_coverage_remove_bottom_good_fit_fraction=(
            RESUME_PHRASE_JOB_COVERAGE_REMOVE_BOTTOM_GOOD_FIT_FRACTION
        ),
        resume_phrase_job_coverage_remove_top_bad_match_fraction=(
            RESUME_PHRASE_JOB_COVERAGE_REMOVE_TOP_BAD_MATCH_FRACTION
        ),
        return_operation_results=True,
        precomputed_resume_profile=precomputed_resume_profile,
    )
    results = ranking_output["results"]
    more_results = ranking_output.get("pre_llm_results", [])
    print_ranking_timing("multi_stage_ranking", step_started_at, results=len(results))

    step_started_at = time.perf_counter()
    payload = package_results(
        results=results,
        more_results=more_results,
        filtered_df=filtered_df,
        total_jobs=total_jobs,
        top_k_to_show=top_k_to_show,
        cluster_filter=cluster_filter,
        include_cross_encoder_metrics=ENABLE_CROSS_ENCODER,
    )
    print_ranking_timing("package_results", step_started_at, results=len(payload["results"]))
    return payload


def load_candidate_embeddings_from_shards(
    candidate_df: pd.DataFrame,
    *,
    shards_by_index: dict[int, RankingShard],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    word_embeddings_by_position: dict[int, np.ndarray] = {}
    phrase_embeddings_by_position: dict[int, np.ndarray] = {}

    for shard_index, shard_rows in candidate_df.groupby("artifact_shard", sort=True):
        shard = shards_by_index[int(shard_index)]
        _, embeddings_path = ensure_shard_files_available(shard)
        artifact_row_indexes = shard_rows["artifact_row_index"].astype(int).tolist()
        word_matrices, phrase_matrices = load_selected_embedding_matrices(
            embeddings_path,
            artifact_row_indexes=artifact_row_indexes,
            shard_index=int(shard_index),
        )
        for position, word_matrix, phrase_matrix in zip(shard_rows.index.tolist(), word_matrices, phrase_matrices):
            word_embeddings_by_position[int(position)] = word_matrix
            phrase_embeddings_by_position[int(position)] = phrase_matrix
        print(
            "Sharded ranking embeddings loaded: "
            f"shard={int(shard_index):05d}, selected_jobs={len(shard_rows)}",
            flush=True,
        )
        del word_matrices, phrase_matrices

    positions = list(range(len(candidate_df)))
    return (
        [word_embeddings_by_position[position] for position in positions],
        [phrase_embeddings_by_position[position] for position in positions],
    )


def run_sharded_resume_ranking(
    *,
    resume_text: str,
    country: str | None,
    state: str | None,
    max_required_yoe: float | None,
    exclude_security_clearance: bool,
    require_recent_posted: bool,
    top_k_to_show: int,
    precomputed_resume_profile: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not USE_RANKING_ARTIFACTS or not USE_SHARDED_RANKING_ARTIFACTS:
        return None

    shards = discover_ranking_shards()
    if not shards:
        print("Sharded ranking artifacts not found; falling back to existing artifact/CSV loading.", flush=True)
        return None

    print(
        "Sharded ranking start: "
        f"shards={len(shards)}, top_k_to_show={top_k_to_show}, "
        f"country_filter={bool(country)}, state_filter={bool(state)}, "
        f"max_required_yoe_set={max_required_yoe is not None}, "
        f"exclude_security_clearance={exclude_security_clearance}",
        flush=True,
    )
    total_jobs = 0
    candidate_frames: list[pd.DataFrame] = []
    candidate_title_embeddings: list[np.ndarray] = []
    cluster_filter: dict[str, Any] = {"enabled": USE_REQUIREMENTS_CLUSTERING, "method": "sharded_requirements_prefilter"}
    recent_cutoff_utc = datetime.now(timezone.utc) - timedelta(hours=RECENT_POSTED_HOURS)

    for shard in iter_prefetched_shards(shards):
        shard_started_at = time.perf_counter()
        print(f"Sharded ranking shard start: shard={shard.index:05d}", flush=True)
        metadata_path, embeddings_path = ensure_shard_files_available(shard)
        records = read_metadata_records(metadata_path, shard_index=shard.index)
        if not records:
            print(f"Sharded ranking shard skipped: shard={shard.index:05d}, rows=0", flush=True)
            continue
        total_jobs += len(records)
        shard_df = pd.DataFrame(records)
        print(
            "Sharded ranking shard metadata loaded: "
            f"shard={shard.index:05d}, rows={len(records)}, columns={len(shard_df.columns)}",
            flush=True,
        )
        print_unparseable_posted_at_examples(shard_index=shard.index, records=records)
        stop_after_current_shard_for_recent_posted = False
        if require_recent_posted:
            oldest_posted_at, newest_posted_at = posted_at_range_for_records(records)
            if newest_posted_at is None:
                print(
                    "Sharded ranking recent-posted shard stop skipped: "
                    f"shard={shard.index:05d}, reason=no_parseable_posted_at",
                    flush=True,
                )
            elif newest_posted_at < recent_cutoff_utc:
                print(
                    "Sharded ranking recent-posted shard stop: "
                    f"shard={shard.index:05d}, reason=newest_job_too_old, "
                    f"newest_posted_at={newest_posted_at.isoformat()}, "
                    f"cutoff={recent_cutoff_utc.isoformat()}",
                    flush=True,
                )
                break
            elif oldest_posted_at is not None and oldest_posted_at < recent_cutoff_utc:
                stop_after_current_shard_for_recent_posted = True
                print(
                    "Sharded ranking recent-posted boundary shard: "
                    f"shard={shard.index:05d}, "
                    f"oldest_posted_at={oldest_posted_at.isoformat()}, "
                    f"cutoff={recent_cutoff_utc.isoformat()}",
                    flush=True,
                )
        validate_jobs_df_columns(shard_df)
        shard_df["_shard_local_row_index"] = np.arange(len(shard_df), dtype=np.int64)
        shard_df["csv_row_index"] = shard_df.get("csv_row_index", pd.Series(range(len(shard_df))))
        shard_df[TEXT_COLUMN] = shard_df[TEXT_COLUMN].fillna("").astype(str).map(str.strip)
        shard_df = shard_df[shard_df[TEXT_COLUMN] != ""].reset_index(drop=True)
        title_embeddings = load_shard_title_embeddings(
            embeddings_path,
            row_count=len(records),
            shard_index=shard.index,
        )
        print(
            "Sharded ranking shard title embeddings loaded: "
            f"shard={shard.index:05d}, embeddings={len(title_embeddings)}",
            flush=True,
        )

        filtered_df = apply_hard_filters(
            shard_df,
            country=country,
            state=state,
            max_required_yoe=max_required_yoe,
            exclude_security_clearance=exclude_security_clearance,
            require_recent_posted=require_recent_posted,
        )
        if filtered_df.empty:
            print(
                f"Sharded ranking early filters: shard={shard.index:05d}, rows={len(records)}, survivors=0, accumulated={sum(len(frame) for frame in candidate_frames)}",
                flush=True,
            )
            if stop_after_current_shard_for_recent_posted:
                print(
                    "Sharded ranking recent-posted stopping after boundary shard: "
                    f"shard={shard.index:05d}, accumulated={sum(len(frame) for frame in candidate_frames)}",
                    flush=True,
                )
                break
            continue

        title_embeddings = title_embeddings[filtered_df["_shard_local_row_index"].astype(int).to_numpy()].astype(
            np.float32,
            copy=True,
        )
        filtered_df = filtered_df.reset_index(drop=True)
        print(
            "Sharded ranking shard hard filters complete: "
            f"shard={shard.index:05d}, survivors={len(filtered_df)}",
            flush=True,
        )
        filtered_df, title_embeddings = run_seniority_prefilter(
            filtered_df,
            resume_text=resume_text,
            title_requirements_embeddings=title_embeddings,
            precomputed_resume_embedding=(
                np.asarray(precomputed_resume_profile["overall_embedding"], dtype=np.float32)
                if precomputed_resume_profile is not None
                else None
            ),
            precomputed_anchor_embeddings=(
                np.asarray(precomputed_resume_profile["seniority_anchor_embeddings"], dtype=np.float32)
                if precomputed_resume_profile is not None
                else None
            ),
        )
        print(
            "Sharded ranking shard seniority prefilter complete: "
            f"shard={shard.index:05d}, survivors={len(filtered_df)}",
            flush=True,
        )
        if not filtered_df.empty:
            candidate_frames.append(filtered_df)
            candidate_title_embeddings.append(title_embeddings)
        accumulated = sum(len(frame) for frame in candidate_frames)
        print(
            "Sharded ranking early filters: "
            f"shard={shard.index:05d}, rows={len(records)}, survivors={len(filtered_df)}, accumulated={accumulated}",
            flush=True,
        )
        print_ranking_timing(
            "sharded_rank_shard_total",
            shard_started_at,
            shard=f"{shard.index:05d}",
            survivors=len(filtered_df),
            accumulated=accumulated,
        )
        if accumulated >= SHARDED_PIPELINE_MIN_CANDIDATES_AFTER_SENIORITY:
            break
        if stop_after_current_shard_for_recent_posted:
            print(
                "Sharded ranking recent-posted stopping after boundary shard: "
                f"shard={shard.index:05d}, accumulated={accumulated}",
                flush=True,
            )
            break

    if not candidate_frames:
        payload = package_results(
            results=[],
            filtered_df=pd.DataFrame(),
            total_jobs=total_jobs,
            top_k_to_show=top_k_to_show,
            cluster_filter=cluster_filter,
            include_cross_encoder_metrics=ENABLE_CROSS_ENCODER,
        )
        return payload

    step_started_at = time.perf_counter()
    candidate_df = pd.concat(candidate_frames, ignore_index=True)
    compact_title_embeddings = np.vstack(candidate_title_embeddings).astype(np.float32)
    candidate_df = candidate_df.reset_index(drop=True)
    candidate_df["candidate_row_index"] = candidate_df.index
    print_ranking_timing(
        "sharded_rank_candidate_assembly",
        step_started_at,
        candidates=len(candidate_df),
        title_embeddings=len(compact_title_embeddings),
    )
    print(
        "Sharded ranking final candidate assembly complete: "
        f"candidates={len(candidate_df)}, title_embeddings={len(compact_title_embeddings)}",
        flush=True,
    )
    print_candidate_titles_and_companies("before_requirements_embedding_prefilter", candidate_df)
    step_started_at = time.perf_counter()
    candidate_df, cluster_filter, compact_title_embeddings = apply_requirements_embedding_prefilter(
        candidate_df,
        resume_text=resume_text,
        title_requirements_embeddings=compact_title_embeddings,
        precomputed_resume_embedding=(
            np.asarray(precomputed_resume_profile["overall_embedding"], dtype=np.float32)
            if precomputed_resume_profile is not None
            else None
        ),
    )
    candidate_df = candidate_df.reset_index(drop=True)
    candidate_df["candidate_row_index"] = candidate_df.index
    print(
        "Sharded ranking requirements prefilter complete: "
        f"survivors={len(candidate_df)}, method={cluster_filter.get('method')}",
        flush=True,
    )
    print_ranking_timing(
        "sharded_rank_requirements_prefilter",
        step_started_at,
        survivors=len(candidate_df),
        method=cluster_filter.get("method"),
    )
    print_candidate_titles_and_companies("after_requirements_embedding_prefilter", candidate_df)

    shards_by_index = {shard.index: shard for shard in shards}
    print("Sharded ranking loading selected word/phrase embeddings.", flush=True)
    step_started_at = time.perf_counter()
    word_embeddings, phrase_embeddings = load_candidate_embeddings_from_shards(
        candidate_df,
        shards_by_index=shards_by_index,
    )
    print_ranking_timing(
        "sharded_rank_selected_embeddings_total",
        step_started_at,
        word_matrices=len(word_embeddings),
        phrase_matrices=len(phrase_embeddings),
    )
    print(
        "Sharded ranking selected embeddings loaded: "
        f"word_matrices={len(word_embeddings)}, phrase_matrices={len(phrase_embeddings)}",
        flush=True,
    )
    return run_multi_stage_and_package(
        resume_text=resume_text,
        filtered_df=candidate_df,
        total_jobs=total_jobs,
        top_k_to_show=top_k_to_show,
        cluster_filter=cluster_filter,
        full_jobs_df=None,
        precomputed_title_requirements_embeddings=compact_title_embeddings,
        precomputed_word_embeddings=word_embeddings,
        precomputed_phrase_embeddings=phrase_embeddings,
        precomputed_resume_profile=precomputed_resume_profile,
    )


def run_resume_ranking(
    *,
    resume_text: str,
    country: str | None,
    state: str | None,
    max_required_yoe: float | None,
    exclude_security_clearance: bool = False,
    require_recent_posted: bool = False,
    top_k_to_show: int = 10,
    precomputed_resume_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timing_token = start_ranking_timing_collection()
    try:
        return _run_resume_ranking_impl(
            resume_text=resume_text,
            country=country,
            state=state,
            max_required_yoe=max_required_yoe,
            exclude_security_clearance=exclude_security_clearance,
            require_recent_posted=require_recent_posted,
            top_k_to_show=top_k_to_show,
            precomputed_resume_profile=precomputed_resume_profile,
        )
    finally:
        print_ranking_timing_summary()
        reset_ranking_timing_collection(timing_token)


def _run_resume_ranking_impl(
    *,
    resume_text: str,
    country: str | None,
    state: str | None,
    max_required_yoe: float | None,
    exclude_security_clearance: bool = False,
    require_recent_posted: bool = False,
    top_k_to_show: int = 10,
    precomputed_resume_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pipeline_started_at = time.perf_counter()

    sharded_payload = run_sharded_resume_ranking(
        resume_text=resume_text,
        country=country,
        state=state,
        max_required_yoe=max_required_yoe,
        exclude_security_clearance=exclude_security_clearance,
        require_recent_posted=require_recent_posted,
        top_k_to_show=top_k_to_show,
        precomputed_resume_profile=precomputed_resume_profile,
    )
    if sharded_payload is not None:
        print_ranking_timing("total_resume_ranking", pipeline_started_at, results=len(sharded_payload["results"]))
        return sharded_payload

    step_started_at = time.perf_counter()
    jobs_df = get_jobs_df()
    total_jobs = len(jobs_df)
    print_ranking_timing("load_jobs", step_started_at, jobs=total_jobs)

    step_started_at = time.perf_counter()
    filtered_df = apply_hard_filters(
        jobs_df,
        country=country,
        state=state,
        max_required_yoe=max_required_yoe,
        exclude_security_clearance=exclude_security_clearance,
        require_recent_posted=require_recent_posted,
    )
    print_ranking_timing(
        "hard_filters",
        step_started_at,
        jobs_before=total_jobs,
        jobs_after=len(filtered_df),
    )

    if filtered_df.empty:
        payload = package_results(
            results=[],
            filtered_df=filtered_df,
            total_jobs=total_jobs,
            top_k_to_show=top_k_to_show,
            cluster_filter={"enabled": False},
            include_cross_encoder_metrics=ENABLE_CROSS_ENCODER,
        )
        print_ranking_timing("total_resume_ranking", pipeline_started_at, results=0)
        return payload

    step_started_at = time.perf_counter()
    filtered_df, cluster_filter = apply_requirements_cluster_filter(
        filtered_df,
        resume_text=resume_text,
        precomputed_resume_embedding=(
            np.asarray(precomputed_resume_profile["overall_embedding"], dtype=np.float32)
            if precomputed_resume_profile is not None
            else None
        ),
    )
    print_ranking_timing(
        "requirements_cluster_filter",
        step_started_at,
        enabled=cluster_filter.get("enabled", False),
        jobs_after=len(filtered_df),
    )

    step_started_at = time.perf_counter()
    job_descriptions = filtered_df[TEXT_COLUMN].tolist()
    job_titles = (
        filtered_df[TITLE_COLUMN].fillna("").astype(str).tolist()
        if TITLE_COLUMN in filtered_df.columns
        else None
    )
    job_requirements = (
        filtered_df[REQUIREMENTS_COLUMN].fillna("").astype(str).tolist()
        if REQUIREMENTS_COLUMN in filtered_df.columns
        else [""] * len(filtered_df)
    )
    job_companies = (
        filtered_df["company_name"].fillna("").astype(str).tolist()
        if "company_name" in filtered_df.columns
        else None
    )
    job_min_years_experience = (
        filtered_df["min_years_experience"].tolist()
        if "min_years_experience" in filtered_df.columns
        else None
    )
    precomputed_word_lists = precomputed_lists_from_column(filtered_df, "job_selected_words")
    precomputed_title_requirements_embeddings = title_requirements_embeddings_for_filtered_rows(filtered_df)
    precomputed_phrase_chunks = precomputed_lists_from_column(filtered_df, "job_phrase_chunks")
    precomputed_word_embeddings = embedding_matrices_for_filtered_rows(
        filtered_df,
        _job_selected_word_embeddings,
        _job_selected_word_offsets,
    )
    precomputed_phrase_embeddings = embedding_matrices_for_filtered_rows(
        filtered_df,
        _job_phrase_chunk_embeddings,
        _job_phrase_chunk_offsets,
    )
    (
        comparison_job_ids,
        comparison_phrase_chunks,
        comparison_phrase_embeddings,
    ) = full_artifact_phrase_comparison_inputs(
        jobs_df,
        RESUME_PHRASE_JOB_COVERAGE_COMPARISON_JOBS,
    )
    requirements_cluster_distances = (
        filtered_df["requirements_cluster_distance"].tolist()
        if "requirements_cluster_distance" in filtered_df.columns
        else None
    )
    print_ranking_timing("prepare_ranking_inputs", step_started_at, jobs=len(job_descriptions))

    if precomputed_word_embeddings is not None and precomputed_phrase_embeddings is not None:
        print("Using precomputed job-side word and phrase embeddings for ranking.")
    else:
        print("Precomputed job-side word/phrase embeddings unavailable; ranking may embed job-side text at runtime.")

    step_started_at = time.perf_counter()
    ranking_output = rank_jobs_multi_stage(
        resume_text=resume_text,
        job_descriptions=job_descriptions,
        job_titles=job_titles,
        job_requirements=job_requirements,
        job_min_years_experience=job_min_years_experience,
        job_companies=job_companies,
        minilm_model=get_minilm_model() if precomputed_resume_profile is None else None,
        cross_encoder_model=get_cross_encoder_model_or_none() if precomputed_resume_profile is None else None,
        enable_cross_encoder=ENABLE_CROSS_ENCODER and precomputed_resume_profile is None,
        word_keep_n=min(MULTI_STAGE_WORD_KEEP, len(job_descriptions)),
        phrase_keep_n=min(MULTI_STAGE_PHRASE_KEEP, len(job_descriptions)),
        word_custom_stopwords={"preferred", "required", "qualification", "qualifications"},
        word_use_stopword_filter=True,
        word_use_frequent_word_filter=False,
        word_max_count=4,
        word_use_tfidf_filter=True,
        word_top_tfidf_fraction=0.25,
        word_max_words_per_job=50,
        word_deduplicate_resume_words=True,
        word_n_projections=50,
        word_random_state=0,
        word_embedding_batch_size=128,
        phrase_n_projections=64,
        phrase_random_state=0,
        phrase_batch_size=64,
        phrase_normalize_embeddings=False,
        phrase_min_chunk_words=3,
        phrase_max_chunk_words=24,
        phrase_include_sentences=True,
        phrase_include_sentence_windows=True,
        ce_chunk_size_words=220,
        ce_chunk_overlap_words=40,
        ce_aggregation="top_k_mean",
        ce_top_k=2,
        ce_batch_size=32,
        cross_encoder_union_top_k_per_ranker=CROSS_ENCODER_UNION_TOP_K_PER_RANKER,
        poor_match_max_rank_per_step=PIPELINE_POOR_MATCH_MAX_RANK,
        mahalanobis_input_top_k=MAHALANOBIS_INPUT_TOP_K,
        mahalanobis_remove_bottom_fraction=MAHALANOBIS_REMOVE_BOTTOM_FRACTION,
        multi_metric_bad_fit_bottom_fraction=MULTI_METRIC_BAD_FIT_BOTTOM_FRACTION,
        enable_technology_mismatch_filter=ENABLE_TECHNOLOGY_MISMATCH_FILTER,
        tech_filter_min_job_types=TECH_FILTER_MIN_JOB_TYPES,
        tech_filter_max_job_type_overlap_ratio=TECH_FILTER_MAX_JOB_TYPE_OVERLAP_RATIO,
        precomputed_job_word_lists=precomputed_word_lists,
        precomputed_title_requirements_embeddings=precomputed_title_requirements_embeddings,
        precomputed_job_phrase_chunks=precomputed_phrase_chunks,
        precomputed_job_word_embeddings=precomputed_word_embeddings,
        precomputed_job_phrase_embeddings=precomputed_phrase_embeddings,
        resume_phrase_job_coverage_comparison_job_ids=comparison_job_ids,
        resume_phrase_job_coverage_comparison_phrase_chunks=comparison_phrase_chunks,
        resume_phrase_job_coverage_comparison_phrase_embeddings=comparison_phrase_embeddings,
        requirements_cluster_distances=requirements_cluster_distances,
        resume_phrase_coverage_dataset_dir=RESUME_DATASET_DIR,
        resume_phrase_coverage_flag_percentile=RESUME_PHRASE_DISTANCE_FLAG_PERCENTILE,
        resume_phrase_coverage_bad_match_percentile=RESUME_PHRASE_BAD_MATCH_PERCENTILE,
        resume_phrase_coverage_job_flag_fraction=RESUME_PHRASE_JOB_FLAG_FRACTION,
        resume_phrase_coverage_min_chunk_words=3,
        resume_phrase_coverage_max_chunk_words=24,
        resume_phrase_coverage_include_sentences=True,
        resume_phrase_coverage_include_sentence_windows=True,
        resume_phrase_coverage_batch_size=64,
        resume_phrase_coverage_normalize_embeddings=False,
        resume_phrase_job_coverage_flag_percentile=RESUME_PHRASE_JOB_COVERAGE_FLAG_PERCENTILE,
        resume_phrase_job_coverage_bad_match_percentile=RESUME_PHRASE_JOB_COVERAGE_BAD_MATCH_PERCENTILE,
        resume_phrase_coverage_remove_bottom_good_fit_fraction=RESUME_PHRASE_COVERAGE_REMOVE_BOTTOM_GOOD_FIT_FRACTION,
        resume_phrase_coverage_remove_top_bad_match_fraction=RESUME_PHRASE_COVERAGE_REMOVE_TOP_BAD_MATCH_FRACTION,
        resume_phrase_job_coverage_remove_bottom_good_fit_fraction=(
            RESUME_PHRASE_JOB_COVERAGE_REMOVE_BOTTOM_GOOD_FIT_FRACTION
        ),
        resume_phrase_job_coverage_remove_top_bad_match_fraction=(
            RESUME_PHRASE_JOB_COVERAGE_REMOVE_TOP_BAD_MATCH_FRACTION
        ),
        return_operation_results=True,
        precomputed_resume_profile=precomputed_resume_profile,
    )
    results = ranking_output["results"]
    more_results = ranking_output.get("pre_llm_results", [])
    print_ranking_timing("multi_stage_ranking", step_started_at, results=len(results))

    step_started_at = time.perf_counter()
    payload = package_results(
        results=results,
        more_results=more_results,
        filtered_df=filtered_df,
        total_jobs=total_jobs,
        top_k_to_show=top_k_to_show,
        cluster_filter=cluster_filter,
        include_cross_encoder_metrics=ENABLE_CROSS_ENCODER,
    )
    print_ranking_timing("package_results", step_started_at, results=len(payload["results"]))
    print_ranking_timing("total_resume_ranking", pipeline_started_at, results=len(payload["results"]))
    return payload
