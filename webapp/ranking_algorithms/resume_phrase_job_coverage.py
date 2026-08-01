from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

from ranking_algorithms.phrases_wasserstein_rankings import embed_texts, phrase_chunks
from ranking_algorithms.resume_phrase_coverage import _cosine_distances


OPERATION_NAME = "resume_phrase_job_coverage"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


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


def compute_resume_phrase_inputs(
    *,
    resume_text: str,
    minilm_model: Any,
    min_chunk_words: int,
    max_chunk_words: int,
    include_sentences: bool,
    include_sentence_windows: bool,
    batch_size: int,
    normalize_embeddings: bool,
) -> tuple[list[str], np.ndarray]:
    chunks = phrase_chunks(
        resume_text,
        min_words=min_chunk_words,
        max_words=max_chunk_words,
        include_sentences=include_sentences,
        include_sentence_windows=include_sentence_windows,
    )
    if not chunks:
        return [], np.empty((0, 0), dtype=np.float32)

    embeddings = embed_texts(
        model=minilm_model,
        texts=chunks,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        cache={},
    )
    return chunks, np.asarray(embeddings, dtype=np.float32)


def _clean_chunks(chunks: Sequence[str]) -> list[str]:
    return [str(chunk).strip() for chunk in chunks if str(chunk).strip()]


def _valid_phrase_group(
    chunks: Sequence[str],
    embeddings: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    clean_chunks = _clean_chunks(chunks)
    matrix = np.asarray(embeddings, dtype=np.float32)
    phrase_count = min(len(clean_chunks), len(matrix))
    return clean_chunks[:phrase_count], matrix[:phrase_count]


def _comparison_phrase_matrix(
    *,
    candidate_job_id: int,
    comparison_job_ids: Sequence[int],
    candidate_embeddings: np.ndarray,
    comparison_phrase_chunks: Sequence[Sequence[str]],
    comparison_phrase_embeddings: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    matrices = [np.asarray(candidate_embeddings, dtype=np.float32)]
    labels = [np.ones(len(candidate_embeddings), dtype=bool)]

    for comparison_job_id in comparison_job_ids:
        if comparison_job_id < 0 or comparison_job_id >= len(comparison_phrase_embeddings):
            continue

        if comparison_job_id >= len(comparison_phrase_chunks):
            continue

        _, matrix = _valid_phrase_group(
            comparison_phrase_chunks[comparison_job_id],
            np.asarray(comparison_phrase_embeddings[comparison_job_id], dtype=np.float32),
        )
        if len(matrix) == 0:
            continue
        matrices.append(matrix)
        labels.append(np.zeros(len(matrix), dtype=bool))

    return np.vstack(matrices), np.concatenate(labels)


def run_resume_phrase_job_coverage_operation(
    *,
    job_ids: Sequence[int],
    job_titles: Sequence[str] | None,
    job_companies: Sequence[str] | None,
    resume_text: str,
    minilm_model: Any,
    precomputed_job_phrase_chunks: Sequence[Sequence[str]] | None,
    precomputed_job_phrase_embeddings: Sequence[np.ndarray] | None,
    comparison_job_ids: Sequence[int] | None,
    comparison_job_phrase_chunks: Sequence[Sequence[str]] | None,
    comparison_job_phrase_embeddings: Sequence[np.ndarray] | None,
    precomputed_user_phrase_chunks: Sequence[str] | None = None,
    precomputed_user_phrase_embeddings: np.ndarray | None = None,
    flag_percentile: float = 10.0,
    bad_match_percentile: float = 90.0,
    min_chunk_words: int = 3,
    max_chunk_words: int = 24,
    include_sentences: bool = True,
    include_sentence_windows: bool = True,
    batch_size: int = 64,
    normalize_embeddings: bool = False,
) -> dict[str, Any]:
    if not 0 <= flag_percentile <= 100:
        raise ValueError("flag_percentile must be in [0, 100].")
    if not 0 <= bad_match_percentile <= 100:
        raise ValueError("bad_match_percentile must be in [0, 100].")

    if precomputed_job_phrase_chunks is None or precomputed_job_phrase_embeddings is None:
        message = "candidate job phrase chunks/embeddings are unavailable."
        print(f"Resume phrase job coverage skipped: {message}", flush=True)
        return _result("skipped", error=message)
    if comparison_job_ids is None or comparison_job_phrase_chunks is None or comparison_job_phrase_embeddings is None:
        message = "comparison job phrase chunks/embeddings are unavailable."
        print(f"Resume phrase job coverage skipped: {message}", flush=True)
        return _result("skipped", error=message)

    if precomputed_user_phrase_chunks is not None and precomputed_user_phrase_embeddings is not None:
        user_chunks = _clean_chunks(precomputed_user_phrase_chunks)
        user_embeddings = np.asarray(precomputed_user_phrase_embeddings, dtype=np.float32)
    else:
        user_chunks, user_embeddings = compute_resume_phrase_inputs(
            resume_text=resume_text,
            minilm_model=minilm_model,
            min_chunk_words=min_chunk_words,
            max_chunk_words=max_chunk_words,
            include_sentences=include_sentences,
            include_sentence_windows=include_sentence_windows,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )
    user_phrase_count = min(len(user_chunks), len(user_embeddings))
    if user_phrase_count == 0:
        message = "user resume produced no phrase chunks."
        print(f"Resume phrase job coverage skipped: {message}", flush=True)
        return _result("skipped", error=message)
    user_chunks = user_chunks[:user_phrase_count]
    user_embeddings = user_embeddings[:user_phrase_count]

    print(
        "Resume phrase job coverage starting: "
        f"jobs={len(job_ids)}, "
        f"user_phrases={user_phrase_count}, "
        f"comparison_jobs={len(comparison_job_ids)}, "
        f"flag_percentile={flag_percentile}, "
        f"bad_match_percentile={bad_match_percentile}",
        flush=True,
    )

    job_metrics: dict[int, Any] = {}
    bad_match_percents: list[float] = []
    print_job_details = _env_bool("ENABLE_RESUME_PHRASE_COVERAGE_JOB_PRINTS", True)
    for rank, job_id in enumerate(job_ids, start=1):
        candidate_job_id = int(job_id)
        if candidate_job_id < 0 or candidate_job_id >= len(precomputed_job_phrase_chunks):
            continue

        candidate_chunks, candidate_embeddings = _valid_phrase_group(
            precomputed_job_phrase_chunks[candidate_job_id],
            np.asarray(precomputed_job_phrase_embeddings[candidate_job_id], dtype=np.float32),
        )
        if len(candidate_chunks) == 0:
            percent_flagged = 0.0
            flagged_count = 0
            mean_flagged_percentile = None
            mean_flagged_distance = None
            bad_match_count = 0
            bad_match_percent = 0.0
            mean_bad_match_percentile = None
            mean_bad_match_distance = None
        else:
            all_embeddings, candidate_mask = _comparison_phrase_matrix(
                candidate_job_id=candidate_job_id,
                comparison_job_ids=comparison_job_ids,
                candidate_embeddings=candidate_embeddings,
                comparison_phrase_chunks=comparison_job_phrase_chunks,
                comparison_phrase_embeddings=comparison_job_phrase_embeddings,
            )

            flagged_percentiles: list[float] = []
            flagged_distances: list[float] = []
            bad_match_percentiles: list[float] = []
            bad_match_distances: list[float] = []
            for user_embedding in user_embeddings:
                distances = _cosine_distances(user_embedding, all_embeddings)
                candidate_distances = distances[candidate_mask]
                if len(candidate_distances) == 0:
                    continue

                closest_candidate_distance = float(np.min(candidate_distances))
                percentile = float(100.0 * np.mean(distances <= closest_candidate_distance))
                if percentile <= flag_percentile:
                    flagged_percentiles.append(percentile)
                    flagged_distances.append(closest_candidate_distance)
                if percentile >= bad_match_percentile:
                    bad_match_percentiles.append(percentile)
                    bad_match_distances.append(closest_candidate_distance)

            flagged_count = len(flagged_percentiles)
            percent_flagged = flagged_count / user_phrase_count
            mean_flagged_percentile = float(np.mean(flagged_percentiles)) if flagged_percentiles else None
            mean_flagged_distance = float(np.mean(flagged_distances)) if flagged_distances else None
            bad_match_count = len(bad_match_percentiles)
            bad_match_percent = bad_match_count / user_phrase_count
            mean_bad_match_percentile = float(np.mean(bad_match_percentiles)) if bad_match_percentiles else None
            mean_bad_match_distance = float(np.mean(bad_match_distances)) if bad_match_distances else None
        bad_match_percents.append(float(bad_match_percent))

        title = (
            str(job_titles[candidate_job_id]).strip()
            if job_titles is not None and candidate_job_id < len(job_titles)
            else ""
        )
        company = (
            str(job_companies[candidate_job_id]).strip()
            if job_companies is not None and candidate_job_id < len(job_companies)
            else ""
        )
        if print_job_details:
            print(
                "Resume phrase job coverage: "
                f"title={title or '(unknown title)'}, "
                f"company={company or '(unknown company)'}, "
                f"total_resume_phrases={user_phrase_count}, "
                f"flagged_resume_phrases={flagged_count}, "
                f"percent_flagged={percent_flagged:.3f}, "
                f"bad_match_resume_phrases={bad_match_count}, "
                f"bad_match_percent={bad_match_percent:.3f}",
                flush=True,
            )

        job_metrics[candidate_job_id] = {
            "rank": rank,
            "score": float(percent_flagged),
            "score_direction": "higher_is_better",
            "raw_metrics": {
                "total_resume_phrases": int(user_phrase_count),
                "flagged_resume_phrases": int(flagged_count),
                "percent_flagged": float(percent_flagged),
                "comparison_jobs": int(len(comparison_job_ids)),
                "flag_percentile": float(flag_percentile),
                "mean_flagged_percentile": mean_flagged_percentile,
                "mean_flagged_distance": mean_flagged_distance,
                "bad_match_resume_phrases": int(bad_match_count),
                "bad_match_percent": float(bad_match_percent),
                "bad_match_percentile": float(bad_match_percentile),
                "mean_bad_match_percentile": mean_bad_match_percentile,
                "mean_bad_match_distance": mean_bad_match_distance,
            },
        }

    ranked_job_ids = sorted(
        job_metrics,
        key=lambda job_id: (-float(job_metrics[job_id]["score"]), int(job_id)),
    )
    for rank, job_id in enumerate(ranked_job_ids, start=1):
        job_metrics[job_id]["rank"] = rank
    overall_bad_match_percent = float(np.mean(bad_match_percents)) if bad_match_percents else 0.0
    print(
        "Resume phrase job coverage overall bad-match percent: "
        f"{overall_bad_match_percent:.3f}",
        flush=True,
    )

    return _result(
        "ok",
        ranked_job_ids=ranked_job_ids,
        job_metrics=job_metrics,
    )
