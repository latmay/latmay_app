from __future__ import annotations


"""
Lightweight sibling to ranking_service.py for the front-page "recent jobs" feed.

Unlike ranking_service.py, this module never imports pandas/numpy/sentence-transformers
and never touches the embeddings (.npz) artifacts -- it only downloads and parses the
small jobs_metadata.jsonl shard files, so browsing recent jobs stays fast and doesn't
pay for the ranking pipeline's heavy imports.

Shards are exported (see data_pipeline/export/export_recent_jobs_to_gcs.py) as
sequential slices of jobs ordered by posted_at_utc DESC, so shard 0 holds the most
recent jobs, shard 1 the next-most-recent, and so on.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

from google.cloud import storage

from recent_posted_filter import format_posted_at_display, parse_posted_at_utc


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

RUNNING_ON_CLOUD_RUN = bool(os.environ.get("K_SERVICE"))

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "")
GCS_RANKING_SHARDS_PREFIX = os.environ.get("GCS_RANKING_SHARDS_PREFIX", "job_posting_data/shards").strip().strip("/")
LOCAL_RANKING_SHARDS_DIR = Path(os.environ.get("LOCAL_RANKING_SHARDS_DIR", DATA_DIR / "shards"))
DOWNLOADED_GCS_SHARDS_DIR = Path("/tmp/ranking_shards")

METADATA_NAME_PATTERN = r"jobs_metadata_(\d+)\.jsonl$"

# Module-level caches (process lifetime, no invalidation) mirroring the caching
# style already used in ranking_service.py for downloaded ranking artifacts.
_shard_metadata_locations: dict[int, str] | dict[int, Path] | None = None
_shard_records_cache: dict[int, list[dict[str, Any]]] = {}


def shard_index_from_name(name: str) -> int | None:
    match = re.search(METADATA_NAME_PATTERN, name)
    if not match:
        return None
    return int(match.group(1))


def discover_gcs_shard_metadata_locations() -> dict[int, str]:
    if not GCS_BUCKET_NAME or not GCS_RANKING_SHARDS_PREFIX:
        return {}

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    prefix = f"{GCS_RANKING_SHARDS_PREFIX}/"
    locations: dict[int, str] = {}
    for blob in client.list_blobs(bucket, prefix=prefix):
        index = shard_index_from_name(blob.name)
        if index is not None:
            locations[index] = blob.name
    return locations


def discover_local_shard_metadata_locations() -> dict[int, Path]:
    if not LOCAL_RANKING_SHARDS_DIR.exists():
        return {}

    return {
        index: path
        for path in LOCAL_RANKING_SHARDS_DIR.glob("jobs_metadata_*.jsonl")
        if (index := shard_index_from_name(path.name)) is not None
    }


def get_shard_metadata_locations() -> dict[int, str] | dict[int, Path]:
    global _shard_metadata_locations

    if _shard_metadata_locations is None:
        _shard_metadata_locations = (
            discover_gcs_shard_metadata_locations()
            if RUNNING_ON_CLOUD_RUN
            else discover_local_shard_metadata_locations()
        )
    return _shard_metadata_locations


def download_metadata_blob(blob_name: str, destination_path: Path) -> Path | None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() and destination_path.stat().st_size > 0:
        return destination_path

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        return None

    blob.download_to_filename(str(destination_path))
    return destination_path


def read_metadata_records(metadata_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def get_recent_jobs_shard(shard_index: int) -> list[dict[str, Any]] | None:
    """
    Return metadata records for one shard, or None if the shard does not exist.

    Downloads (and caches on disk under /tmp) only that shard's metadata file --
    never the paired embeddings artifact used by ranking.
    """
    if shard_index in _shard_records_cache:
        return _shard_records_cache[shard_index]

    locations = get_shard_metadata_locations()
    if shard_index not in locations:
        return None

    if RUNNING_ON_CLOUD_RUN:
        metadata_path = download_metadata_blob(
            locations[shard_index],
            DOWNLOADED_GCS_SHARDS_DIR / f"jobs_metadata_{shard_index:05d}.jsonl",
        )
        if metadata_path is None:
            return None
    else:
        metadata_path = locations[shard_index]

    records = read_metadata_records(metadata_path)
    _shard_records_cache[shard_index] = records
    return records


def display_fields_for_job(record: dict[str, Any]) -> dict[str, Any]:
    posted_at_value = record.get("posted_at_utc") or record.get("posted_at")
    posted_at_utc = parse_posted_at_utc(posted_at_value)
    return {
        "title": str(record.get("title") or "").strip(),
        "company": str(
            record.get("company_name") or record.get("company") or record.get("source_company") or ""
        ).strip(),
        "location": str(record.get("location_name") or record.get("location") or "").strip(),
        "work_arrangement": str(record.get("work_arrangement") or "").strip(),
        "url": str(record.get("url") or record.get("job_url") or record.get("apply_url") or "").strip(),
        "posted_display": format_posted_at_display(posted_at_utc) if posted_at_utc else "",
    }


def get_recent_jobs_page(shard_index: int) -> dict[str, Any] | None:
    """
    Return {"shard", "jobs", "has_more"} for one shard, or None if it doesn't exist.

    "jobs" holds only display fields (no requirements text, no embeddings row
    indexes) since this feed is for browsing, not ranking.
    """
    records = get_recent_jobs_shard(shard_index)
    if records is None:
        return None

    jobs = [display_fields_for_job(record) for record in records]
    jobs = [job for job in jobs if job["title"]]

    locations = get_shard_metadata_locations()
    has_more = (shard_index + 1) in locations

    return {"shard": shard_index, "jobs": jobs, "has_more": has_more}
