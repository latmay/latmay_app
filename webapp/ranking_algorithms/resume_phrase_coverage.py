from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
try:
    from ranking_timing import record_ranking_timing
except ImportError:
    from webapp.ranking_timing import record_ranking_timing

from ranking_algorithms.phrases_wasserstein_rankings import embed_texts, phrase_chunks


OPERATION_NAME = "resume_phrase_coverage"
PERSISTENT_EXAMPLE_CACHE_VERSION = "resume_phrase_examples_v1"
_EXAMPLE_CACHE: dict[tuple[str, int, int, bool, bool, int, bool], tuple[list[str], np.ndarray]] = {}


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _timing(step: str, start: float, **fields: Any) -> None:
    record_ranking_timing(f"resume_phrase_coverage_{step}", start, **fields)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def _cosine_distances(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    query_norm = _normalize_rows(query.reshape(1, -1))
    candidate_norms = _normalize_rows(candidates)
    if query_norm.shape[1] != candidate_norms.shape[1]:
        raise ValueError("Embedding dimensions do not match for cosine distance.")
    similarities = np.clip(candidate_norms @ query_norm[0], -1.0, 1.0)
    return 1.0 - similarities


def _result(
    status: str,
    *,
    ranked_job_ids: list[int] | None = None,
    job_metrics: dict[int, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "operation_name": OPERATION_NAME,
        "status": status,
        "ranked_job_ids": ranked_job_ids or [],
        "job_metrics": job_metrics or {},
        "error": error,
    }


def _rank_scores(score_rows: Sequence[tuple[int, float]]) -> dict[int, int]:
    ranks: dict[int, int] = {}
    previous_score: float | None = None
    previous_rank = 0

    for position, (job_id, score) in enumerate(score_rows, start=1):
        if previous_score is None or score != previous_score:
            previous_score = score
            previous_rank = position
        ranks[job_id] = previous_rank

    return ranks


def _read_example_resume_texts(resume_dataset_dir: str | Path) -> list[str]:
    dataset_dir = Path(resume_dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        print(
            "Resume phrase coverage skipped: "
            f"RESUME_DATASET_DIR is not a directory: {dataset_dir}",
            flush=True,
        )
        return []

    texts: list[str] = []
    for path in sorted(dataset_dir.rglob("*.md")):
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                texts.append(text)

    if not texts:
        print(
            "Resume phrase coverage skipped: "
            f"no Markdown resumes found in {dataset_dir}",
            flush=True,
        )

    return texts


def _resume_dataset_files(resume_dataset_dir: str | Path) -> list[Path]:
    dataset_dir = Path(resume_dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return []
    return [path for path in sorted(dataset_dir.rglob("*.md")) if path.is_file()]


def _resume_dataset_hash(resume_dataset_dir: str | Path) -> str:
    dataset_dir = Path(resume_dataset_dir)
    digest = hashlib.sha256()
    for path in _resume_dataset_files(dataset_dir):
        relative = path.relative_to(dataset_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _example_cache_path() -> Path:
    configured = os.environ.get("RESUME_PHRASE_EXAMPLE_CACHE_PATH", "").strip()
    if configured:
        return Path(configured)
    cache_dir = Path(os.environ.get("MODEL_CACHE_DIR", "/app/model_cache"))
    return cache_dir / "resume_phrase_coverage_examples.npz"


def _model_identity() -> dict[str, str]:
    return {
        "model_name": os.environ.get("MINILM_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        "model_revision": (
            os.environ.get("MINILM_MODEL_REVISION", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41").strip()
            or "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
        ),
    }


def _example_cache_metadata(
    *,
    resume_dataset_dir: str | Path,
    min_chunk_words: int,
    max_chunk_words: int,
    include_sentences: bool,
    include_sentence_windows: bool,
    batch_size: int,
    normalize_embeddings: bool,
) -> dict[str, Any]:
    return {
        "cache_version": PERSISTENT_EXAMPLE_CACHE_VERSION,
        "dataset_hash": _resume_dataset_hash(resume_dataset_dir),
        "min_chunk_words": int(min_chunk_words),
        "max_chunk_words": int(max_chunk_words),
        "include_sentences": bool(include_sentences),
        "include_sentence_windows": bool(include_sentence_windows),
        "batch_size": int(batch_size),
        "normalize_embeddings": bool(normalize_embeddings),
        **_model_identity(),
    }


def _metadata_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _load_persistent_example_cache(expected_metadata: dict[str, Any]) -> tuple[list[str], np.ndarray] | None:
    path = _example_cache_path()
    if not path.exists():
        print(
            "Resume phrase coverage example cache miss: "
            f"path={path}, reason=missing_file",
            flush=True,
        )
        return None

    try:
        with np.load(path, allow_pickle=False) as npz:
            metadata = json.loads(str(npz["metadata_json"].item()))
            if not _metadata_matches(metadata, expected_metadata):
                print(
                    "Resume phrase coverage example cache miss: "
                    f"path={path}, reason=metadata_mismatch",
                    flush=True,
                )
                return None
            chunks = [str(chunk) for chunk in npz["chunks"].tolist()]
            embeddings = np.asarray(npz["embeddings"], dtype=np.float32)
    except Exception as exc:
        print(
            "Resume phrase coverage example cache miss: "
            f"path={path}, reason=load_failed, error_type={type(exc).__name__}",
            flush=True,
        )
        return None

    print(
        "Resume phrase coverage example cache hit: "
        f"path={path}, chunks={len(chunks)}, embedding_shape={embeddings.shape}",
        flush=True,
    )
    return chunks, embeddings


def _save_persistent_example_cache(chunks: list[str], embeddings: np.ndarray, metadata: dict[str, Any]) -> None:
    path = _example_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            chunks=np.asarray(chunks, dtype=str),
            embeddings=np.asarray(embeddings, dtype=np.float32),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    except Exception as exc:
        print(
            "Resume phrase coverage example cache save skipped: "
            f"path={path}, error_type={type(exc).__name__}",
            flush=True,
        )
        return

    print(
        "Resume phrase coverage example cache saved: "
        f"path={path}, chunks={len(chunks)}, embedding_shape={np.asarray(embeddings).shape}",
        flush=True,
    )


def _embed_example_resume_phrases(
    *,
    model: Any,
    resume_dataset_dir: str | Path,
    min_chunk_words: int,
    max_chunk_words: int,
    include_sentences: bool,
    include_sentence_windows: bool,
    batch_size: int,
    normalize_embeddings: bool,
    cache: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    started = perf_counter()
    dataset_dir = str(Path(resume_dataset_dir).resolve())
    cache_key = (
        dataset_dir,
        min_chunk_words,
        max_chunk_words,
        include_sentences,
        include_sentence_windows,
        batch_size,
        normalize_embeddings,
    )
    cached = _EXAMPLE_CACHE.get(cache_key)
    if cached is not None:
        _timing("example_memory_cache_hit", started, chunks=len(cached[0]))
        return cached

    metadata_started = perf_counter()
    expected_metadata = _example_cache_metadata(
        resume_dataset_dir=resume_dataset_dir,
        min_chunk_words=min_chunk_words,
        max_chunk_words=max_chunk_words,
        include_sentences=include_sentences,
        include_sentence_windows=include_sentence_windows,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    _timing(
        "example_cache_metadata",
        metadata_started,
        dataset_hash=expected_metadata["dataset_hash"][:12],
    )

    load_started = perf_counter()
    persistent = _load_persistent_example_cache(expected_metadata)
    _timing(
        "example_persistent_cache_lookup",
        load_started,
        hit=persistent is not None,
    )
    if persistent is not None:
        _EXAMPLE_CACHE[cache_key] = persistent
        _timing("example_setup_total", started, source="persistent_cache", chunks=len(persistent[0]))
        return persistent

    read_started = perf_counter()
    example_texts = _read_example_resume_texts(resume_dataset_dir)
    _timing("example_dataset_read", read_started, resumes=len(example_texts))

    chunk_started = perf_counter()
    example_chunks: list[str] = []
    seen: set[str] = set()
    for text in example_texts:
        for chunk in phrase_chunks(
            text,
            min_words=min_chunk_words,
            max_words=max_chunk_words,
            include_sentences=include_sentences,
            include_sentence_windows=include_sentence_windows,
        ):
            key = chunk.lower()
            if key not in seen:
                seen.add(key)
                example_chunks.append(chunk)
    _timing("example_chunking", chunk_started, chunks=len(example_chunks))

    if not example_chunks:
        embeddings = np.empty((0, 0), dtype=np.float32)
    else:
        embed_started = perf_counter()
        embeddings = embed_texts(
            model=model,
            texts=example_chunks,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            cache=cache,
        )
        _timing("example_embedding", embed_started, chunks=len(example_chunks))

    _EXAMPLE_CACHE[cache_key] = (example_chunks, embeddings)
    _save_persistent_example_cache(example_chunks, embeddings, expected_metadata)
    _timing("example_setup_total", started, source="computed", chunks=len(example_chunks))
    return example_chunks, embeddings


def build_persistent_example_resume_cache(
    *,
    model: Any,
    resume_dataset_dir: str | Path,
    min_chunk_words: int = 3,
    max_chunk_words: int = 24,
    include_sentences: bool = True,
    include_sentence_windows: bool = True,
    batch_size: int = 64,
    normalize_embeddings: bool = False,
) -> tuple[list[str], np.ndarray]:
    """
    Build and save the example-resume phrase cache for baked container images.
    """
    cache: dict[str, np.ndarray] = {}
    _EXAMPLE_CACHE.clear()
    return _embed_example_resume_phrases(
        model=model,
        resume_dataset_dir=resume_dataset_dir,
        min_chunk_words=min_chunk_words,
        max_chunk_words=max_chunk_words,
        include_sentences=include_sentences,
        include_sentence_windows=include_sentence_windows,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        cache=cache,
    )


def run_resume_phrase_coverage_operation(
    *,
    job_ids: Sequence[int],
    job_titles: Sequence[str] | None,
    job_companies: Sequence[str] | None,
    resume_text: str,
    minilm_model: Any,
    precomputed_job_phrase_chunks: Sequence[Sequence[str]] | None,
    precomputed_job_phrase_embeddings: Sequence[np.ndarray] | None,
    precomputed_user_phrase_chunks: Sequence[str] | None = None,
    precomputed_user_phrase_embeddings: np.ndarray | None = None,
    resume_dataset_dir: str | Path | None,
    flag_percentile: float = 10.0,
    bad_match_percentile: float = 90.0,
    job_flag_fraction: float = 0.30,
    min_chunk_words: int = 3,
    max_chunk_words: int = 24,
    include_sentences: bool = True,
    include_sentence_windows: bool = True,
    batch_size: int = 64,
    normalize_embeddings: bool = False,
    max_flagged_phrases_to_print: int = 5,
) -> dict[str, Any]:
    if not resume_dataset_dir:
        message = "RESUME_DATASET_DIR is not set."
        print(f"Resume phrase coverage skipped: {message}", flush=True)
        return _result("skipped", error=message)

    if not 0 <= flag_percentile <= 100:
        raise ValueError("flag_percentile must be in [0, 100].")
    if not 0 <= bad_match_percentile <= 100:
        raise ValueError("bad_match_percentile must be in [0, 100].")
    if not 0 <= job_flag_fraction <= 1:
        raise ValueError("job_flag_fraction must be in [0, 1].")

    if precomputed_job_phrase_chunks is None or precomputed_job_phrase_embeddings is None:
        message = "precomputed job phrase chunks/embeddings are unavailable."
        print(
            "Resume phrase coverage skipped: "
            f"{message}",
            flush=True,
        )
        return _result("skipped", error=message)

    operation_started = perf_counter()
    cache: dict[str, np.ndarray] = {}
    user_chunk_started = perf_counter()
    user_chunks = (
        [str(chunk).strip() for chunk in precomputed_user_phrase_chunks if str(chunk).strip()]
        if precomputed_user_phrase_chunks is not None
        else phrase_chunks(
            resume_text,
            min_words=min_chunk_words,
            max_words=max_chunk_words,
            include_sentences=include_sentences,
            include_sentence_windows=include_sentence_windows,
        )
    )
    _timing("user_chunking", user_chunk_started, chunks=len(user_chunks))
    if not user_chunks:
        message = "user resume produced no phrase chunks."
        print(f"Resume phrase coverage skipped: {message}", flush=True)
        return _result("skipped", error=message)

    user_embed_started = perf_counter()
    if precomputed_user_phrase_embeddings is not None:
        user_embeddings = np.asarray(precomputed_user_phrase_embeddings, dtype=np.float32)
    else:
        user_embeddings = embed_texts(
            model=minilm_model,
            texts=user_chunks,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            cache=cache,
        )
    _timing("user_embedding", user_embed_started, chunks=len(user_chunks))
    if len(user_embeddings) != len(user_chunks):
        message = "precomputed user phrase chunks/embeddings are misaligned."
        print(f"Resume phrase coverage skipped: {message}", flush=True)
        return _result("skipped", error=message)

    example_started = perf_counter()
    example_chunks, example_embeddings = _embed_example_resume_phrases(
        model=minilm_model,
        resume_dataset_dir=resume_dataset_dir,
        min_chunk_words=min_chunk_words,
        max_chunk_words=max_chunk_words,
        include_sentences=include_sentences,
        include_sentence_windows=include_sentence_windows,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        cache=cache,
    )
    _timing("example_setup", example_started, chunks=len(example_chunks))
    if len(example_chunks) == 0 or len(example_embeddings) == 0:
        return _result("skipped", error="example resume phrases are unavailable.")

    coverage_rows: list[dict[str, Any]] = []
    print_job_details = _env_bool("ENABLE_RESUME_PHRASE_COVERAGE_JOB_PRINTS", True)
    print(
        "Resume phrase coverage starting: "
        f"jobs={len(job_ids)}, "
        f"user_phrases={len(user_chunks)}, "
        f"example_resume_phrases={len(example_chunks)}, "
        f"flag_percentile={flag_percentile}, "
        f"bad_match_percentile={bad_match_percentile}, "
        f"job_flag_fraction={job_flag_fraction}",
        flush=True,
    )

    loop_started = perf_counter()
    phrase_comparisons = 0
    for job_id in job_ids:
        job_index = int(job_id)
        if job_index < 0 or job_index >= len(precomputed_job_phrase_chunks):
            continue

        job_chunks = [
            str(chunk).strip()
            for chunk in precomputed_job_phrase_chunks[job_index]
            if str(chunk).strip()
        ]
        job_embeddings = np.asarray(precomputed_job_phrase_embeddings[job_index], dtype=np.float32)
        phrase_count = min(len(job_chunks), len(job_embeddings))
        if phrase_count == 0:
            if print_job_details:
                print(
                    "\nResume phrase coverage: "
                    f"job_index={job_index}, total_phrases=0, skipped_empty_job_phrases=True",
                    flush=True,
                )
            coverage_rows.append(
                {
                    "job_id": job_index,
                    "total_phrases": 0,
                    "flagged_phrases": 0,
                    "percent_flagged": 0.0,
                    "job_flagged": False,
                    "mean_flagged_percentile": None,
                    "mean_flagged_distance": None,
                    "bad_match_phrases": 0,
                    "bad_match_percent": 0.0,
                    "mean_bad_match_percentile": None,
                    "mean_bad_match_distance": None,
                    "strongest_flagged_phrases": [],
                    "strongest_bad_match_phrases": [],
                }
            )
            continue

        flagged_phrases: list[dict[str, Any]] = []
        bad_match_phrases: list[dict[str, Any]] = []
        for phrase_index in range(phrase_count):
            phrase_comparisons += 1
            job_embedding = job_embeddings[phrase_index]
            user_distances = _cosine_distances(job_embedding, user_embeddings)
            closest_user_index = int(np.argmin(user_distances))
            closest_user_distance = float(user_distances[closest_user_index])

            example_distances = _cosine_distances(job_embedding, example_embeddings)
            percentile = float(100.0 * np.mean(example_distances <= closest_user_distance))
            if percentile <= flag_percentile:
                flagged_phrases.append(
                    {
                        "job_phrase": job_chunks[phrase_index],
                        "closest_user_resume_phrase_index": closest_user_index,
                        "closest_distance": closest_user_distance,
                        "percentile": percentile,
                    }
                )
            if percentile >= bad_match_percentile:
                bad_match_phrases.append(
                    {
                        "job_phrase": job_chunks[phrase_index],
                        "closest_user_resume_phrase_index": closest_user_index,
                        "closest_distance": closest_user_distance,
                        "percentile": percentile,
                    }
                )

        percent_flagged = len(flagged_phrases) / phrase_count
        bad_match_percent = len(bad_match_phrases) / phrase_count
        job_flagged = percent_flagged >= job_flag_fraction
        mean_flagged_percentile = (
            float(np.mean([float(row["percentile"]) for row in flagged_phrases]))
            if flagged_phrases
            else None
        )
        mean_flagged_distance = (
            float(np.mean([float(row["closest_distance"]) for row in flagged_phrases]))
            if flagged_phrases
            else None
        )
        mean_bad_match_percentile = (
            float(np.mean([float(row["percentile"]) for row in bad_match_phrases]))
            if bad_match_phrases
            else None
        )
        mean_bad_match_distance = (
            float(np.mean([float(row["closest_distance"]) for row in bad_match_phrases]))
            if bad_match_phrases
            else None
        )
        title = (
            str(job_titles[job_index]).strip()
            if job_titles is not None and job_index < len(job_titles)
            else ""
        )
        company = (
            str(job_companies[job_index]).strip()
            if job_companies is not None and job_index < len(job_companies)
            else ""
        )
        if print_job_details:
            print(
                "\nResume phrase coverage: "
                f"title={title or '(unknown title)'}, "
                f"company={company or '(unknown company)'}, "
                f"total_phrases={phrase_count}, "
                f"flagged_phrases={len(flagged_phrases)}, "
                f"percent_flagged={percent_flagged:.3f}, "
                f"bad_match_phrases={len(bad_match_phrases)}, "
                f"bad_match_percent={bad_match_percent:.3f}, "
                f"job_would_be_flagged={job_flagged}",
                flush=True,
            )

        strongest = sorted(
            flagged_phrases,
            key=lambda row: (float(row["percentile"]), float(row["closest_distance"])),
        )[:max_flagged_phrases_to_print]
        strongest_bad_match = sorted(
            bad_match_phrases,
            key=lambda row: (-float(row["percentile"]), -float(row["closest_distance"])),
        )[:max_flagged_phrases_to_print]

        coverage_rows.append(
            {
                "job_id": job_index,
                "title": title,
                "company": company,
                "total_phrases": phrase_count,
                "flagged_phrases": len(flagged_phrases),
                "percent_flagged": percent_flagged,
                "job_flagged": job_flagged,
                "mean_flagged_percentile": mean_flagged_percentile,
                "mean_flagged_distance": mean_flagged_distance,
                "bad_match_phrases": len(bad_match_phrases),
                "bad_match_percent": bad_match_percent,
                "mean_bad_match_percentile": mean_bad_match_percentile,
                "mean_bad_match_distance": mean_bad_match_distance,
                "strongest_flagged_phrases": strongest,
                "strongest_bad_match_phrases": strongest_bad_match,
            }
        )

    _timing(
        "coverage_loop",
        loop_started,
        jobs=len(job_ids),
        scored_jobs=len(coverage_rows),
        job_phrases=phrase_comparisons,
        example_phrases=len(example_chunks),
    )

    rank_started = perf_counter()
    ranked_rows = sorted(
        coverage_rows,
        key=lambda row: (
            -float(row["percent_flagged"]),
            -int(row["flagged_phrases"]),
            float(row["mean_flagged_percentile"])
            if row["mean_flagged_percentile"] is not None
            else float("inf"),
            float(row["mean_flagged_distance"])
            if row["mean_flagged_distance"] is not None
            else float("inf"),
            int(row["job_id"]),
        ),
    )
    ranked_job_ids = [int(row["job_id"]) for row in ranked_rows]
    ranks = _rank_scores([(int(row["job_id"]), float(row["percent_flagged"])) for row in ranked_rows])
    overall_bad_match_percent = (
        float(np.mean([float(row["bad_match_percent"]) for row in coverage_rows]))
        if coverage_rows
        else 0.0
    )
    print(
        "Resume phrase coverage overall bad-match percent: "
        f"{overall_bad_match_percent:.3f}",
        flush=True,
    )
    _timing("ranking_finalize", rank_started, rows=len(ranked_rows))
    _timing("operation_total", operation_started, jobs=len(job_ids), rows=len(ranked_rows))

    return _result(
        "ok",
        ranked_job_ids=ranked_job_ids,
        job_metrics={
            int(row["job_id"]): {
                "rank": ranks[int(row["job_id"])],
                "score": float(row["percent_flagged"]),
                "score_direction": "higher_is_better",
                "raw_metrics": {
                    "total_phrases": int(row["total_phrases"]),
                    "flagged_phrases": int(row["flagged_phrases"]),
                    "percent_flagged": float(row["percent_flagged"]),
                    "job_flagged": bool(row["job_flagged"]),
                    "mean_flagged_percentile": row["mean_flagged_percentile"],
                    "mean_flagged_distance": row["mean_flagged_distance"],
                    "bad_match_phrases": int(row["bad_match_phrases"]),
                    "bad_match_percent": float(row["bad_match_percent"]),
                    "mean_bad_match_percentile": row["mean_bad_match_percentile"],
                    "mean_bad_match_distance": row["mean_bad_match_distance"],
                    "strongest_flagged_phrases": row["strongest_flagged_phrases"],
                    "strongest_bad_match_phrases": row["strongest_bad_match_phrases"],
                    "bad_match_percentile": float(bad_match_percentile),
                },
            }
            for row in ranked_rows
        },
    )
