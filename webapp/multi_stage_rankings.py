from __future__ import annotations

"""
Multi-stage resume -> job ranking pipeline.

The pipeline wraps unit ranking operations in a standard result object, applies
separate reduction policies between operations, and chooses the final ordering
with a final ranking policy.
"""

import os
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

try:
    from ranking_timing import record_ranking_timing
except ImportError:
    from webapp.ranking_timing import record_ranking_timing

try:
    from model_loader import load_cross_encoder_model, load_minilm_model
except ImportError:
    from webapp.model_loader import load_cross_encoder_model, load_minilm_model

from ranking_algorithms.cross_encoder_scoring import compare_texts_cross_encoder
from ranking_algorithms.mahalanobis_outlier_ranking import (
    build_mahalanobis_candidate_job_ids,
    rank_mahalanobis_outliers,
)
from ranking_algorithms.multi_metric_bad_fit_filter import run_multi_metric_bad_fit_filter
from ranking_algorithms.pair_independent_bad_match_filter import run_pair_independent_bad_match_filter
from ranking_algorithms.phrases_wasserstein_rankings import rank_job_descriptions_by_phrase_sliced_wasserstein
from ranking_algorithms.resume_phrase_coverage import run_resume_phrase_coverage_operation
from ranking_algorithms.resume_phrase_job_coverage import (
    compute_resume_phrase_inputs,
    run_resume_phrase_job_coverage_operation,
)
from ranking_algorithms.seniority_filter import run_seniority_filter
from ranking_algorithms.technology_mismatch_filter import run_technology_mismatch_filter
from ranking_algorithms.words_wasserstein_rankings import main_rank_job_descriptions_by_wasserstein
from final_ranking_policy import choose_final_job_order
from reduction_policy import reduce_job_ids


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

TEST_RESUME_PATH = DATA_DIR / "sample_resume"
TEST_JOBS_CSV_PATH = DATA_DIR / "sample_combined_jobs_filtered_with_requirements.csv"
TEST_JOB_TEXT_COLUMN = "extracted_requirements"

LOWER_IS_BETTER = "lower_is_better"
HIGHER_IS_BETTER = "higher_is_better"


def run_llm_bad_match_filter(**kwargs: Any) -> dict[str, Any]:
    """Import the optional paid-LLM dependency only when that stage is enabled."""
    if not env_bool("ENABLE_LLM_BAD_MATCH_FILTER", False):
        return make_skipped_result(
            "llm_bad_match_filter",
            "Disabled by ENABLE_LLM_BAD_MATCH_FILTER=false.",
        ) | {"ranked_job_ids": list(kwargs.get("job_ids") or [])}

    from ranking_algorithms.llm_bad_match_filter import run_llm_bad_match_filter as run_enabled_filter

    return run_enabled_filter(**kwargs)


def print_ranking_timing(step_name: str, started_at: float, **metadata: Any) -> None:
    record_ranking_timing(step_name, started_at, **metadata)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def env_nonnegative_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} cannot be negative.")
    return parsed


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return float(value)


def safe_log_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def print_pair_independent_forest_removals(
    removed_job_ids: Sequence[int],
    *,
    job_titles: Sequence[str] | None,
    operation_result: Mapping[str, Any],
) -> None:
    metrics = operation_result.get("job_metrics", {}) or {}
    print(
        "Pair-independent bad-match forest complete: "
        f"removed_jobs={len(removed_job_ids)}",
        flush=True,
    )
    for job_id in removed_job_ids:
        job_index = int(job_id)
        raw_metrics = ((metrics.get(job_id) or {}).get("raw_metrics") or {})
        probability = raw_metrics.get("semantic_bad_match_probability")
        threshold = raw_metrics.get("semantic_bad_match_threshold")
        title = (
            safe_log_text(job_titles[job_index])
            if job_titles is not None and 0 <= job_index < len(job_titles)
            else ""
        )
        # Titles are non-sensitive labels, but keep logs compact and single-line.
        title = " ".join(title.split())[:160]
        print(
            "Pair-independent bad-match forest removed job: "
            f"job_id={job_id}, title={title!r}, "
            f"probability={float(probability):.6f}, threshold={float(threshold):.6f}",
            flush=True,
        )


def print_seniority_filter_job_snapshot(
    label: str,
    job_ids: Sequence[int],
    *,
    job_titles: Sequence[str] | None,
    job_companies: Sequence[str] | None,
    seniority_result: Mapping[str, Any],
) -> None:
    if not env_bool("ENABLE_SENIORITY_FILTER_JOB_PRINTS", False):
        return

    metrics = seniority_result.get("job_metrics", {}) or {}
    jobs = []
    for job_id in job_ids:
        title = job_titles[int(job_id)] if job_titles is not None and int(job_id) < len(job_titles) else ""
        company = job_companies[int(job_id)] if job_companies is not None and int(job_id) < len(job_companies) else ""
        raw_metrics = ((metrics.get(int(job_id)) or {}).get("raw_metrics") or {})
        jobs.append(
            {
                "job_id": int(job_id),
                "title": safe_log_text(title),
                "company": safe_log_text(company),
                "seniority_gap": raw_metrics.get("seniority_gap"),
                "is_too_senior": raw_metrics.get("is_too_senior"),
                "is_too_junior": raw_metrics.get("is_too_junior"),
            }
        )
    print(
        "Seniority filter job snapshot: "
        f"label={label}, count={len(jobs)}, jobs={json.dumps(jobs, ensure_ascii=False)}",
        flush=True,
    )


def sort_job_ids_by_resume_phrase_coverage(
    job_ids: Sequence[int],
    coverage_result: Mapping[str, Any],
) -> list[int]:
    metrics = coverage_result.get("job_metrics", {})

    def sort_key(job_id: int) -> tuple[float, int]:
        metric = metrics.get(job_id) or {}
        raw_metrics = metric.get("raw_metrics") or {}
        return (-float(raw_metrics.get("percent_flagged", 0.0)), int(job_id))

    return sorted([int(job_id) for job_id in job_ids], key=sort_key)


def _validate_fraction(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1].")


def _removal_count(total: int, fraction: float) -> int:
    if total <= 0 or fraction <= 0:
        return 0
    return min(total, max(1, int(total * fraction)))


def reduce_by_coverage_quality(
    *,
    current_job_ids: Sequence[int],
    operation_result: Mapping[str, Any],
    remove_bottom_good_fit_fraction: float,
    remove_top_bad_match_fraction: float,
    min_remaining_jobs: int = 0,
) -> list[int]:
    _validate_fraction("remove_bottom_good_fit_fraction", remove_bottom_good_fit_fraction)
    _validate_fraction("remove_top_bad_match_fraction", remove_top_bad_match_fraction)
    if min_remaining_jobs < 0:
        raise ValueError("min_remaining_jobs cannot be negative.")

    current_set = {int(job_id) for job_id in current_job_ids}
    if operation_result.get("status") != "ok":
        return [int(job_id) for job_id in current_job_ids]

    ranked_job_ids = [
        int(job_id)
        for job_id in operation_result.get("ranked_job_ids", [])
        if int(job_id) in current_set
    ]
    if not ranked_job_ids:
        return [int(job_id) for job_id in current_job_ids]

    metrics = operation_result.get("job_metrics", {}) or {}
    removable_by_good_fit = _removal_count(len(ranked_job_ids), remove_bottom_good_fit_fraction)
    removable_by_bad_match = _removal_count(len(ranked_job_ids), remove_top_bad_match_fraction)
    removed_ids: set[int] = set()

    if removable_by_good_fit:
        by_good_fit = sorted(
            ranked_job_ids,
            key=lambda job_id: (
                float(((metrics.get(job_id) or {}).get("raw_metrics") or {}).get("percent_flagged", 0.0)),
                job_id,
            ),
        )
        removed_ids.update(by_good_fit[:removable_by_good_fit])

    if removable_by_bad_match:
        by_bad_match = sorted(
            ranked_job_ids,
            key=lambda job_id: (
                -float(((metrics.get(job_id) or {}).get("raw_metrics") or {}).get("bad_match_percent", 0.0)),
                job_id,
            ),
        )
        removed_ids.update(by_bad_match[:removable_by_bad_match])

    kept_job_ids = [job_id for job_id in ranked_job_ids if job_id not in removed_ids]
    floor = min(min_remaining_jobs, len(ranked_job_ids))
    if floor > 0 and len(kept_job_ids) < floor:
        seen = set(kept_job_ids)
        for job_id in ranked_job_ids:
            if job_id not in seen:
                kept_job_ids.append(job_id)
                seen.add(job_id)
            if len(kept_job_ids) >= floor:
                break
        removed_ids = {job_id for job_id in ranked_job_ids if job_id not in set(kept_job_ids)}
        print(
            "Coverage quality min-remaining guard applied: "
            f"operation={operation_result.get('operation_name') or 'coverage_quality'}, "
            f"requested_min={min_remaining_jobs}, floor={floor}, jobs_after={len(kept_job_ids)}",
            flush=True,
        )

    if removed_ids:
        operation_name = str(operation_result.get("operation_name") or "coverage_quality")
        print(
            "Coverage quality reduction: "
            f"operation={operation_name}, jobs_before={len(ranked_job_ids)}, "
            f"removed={len(removed_ids)}, jobs_after={len(ranked_job_ids) - len(removed_ids)}, "
            f"remove_bottom_good_fit_fraction={remove_bottom_good_fit_fraction}, "
            f"remove_top_bad_match_fraction={remove_top_bad_match_fraction}",
            flush=True,
        )

    return kept_job_ids


def order_job_ids_by_operation(
    job_ids: Sequence[int],
    operation_result: Mapping[str, Any],
) -> tuple[list[int], str | None]:
    operation_name = str(operation_result.get("operation_name") or "")
    current_set = set(job_ids)
    if operation_result.get("status") == "ok":
        ranked_ids = [
            int(job_id)
            for job_id in operation_result.get("ranked_job_ids", [])
            if job_id in current_set
        ]
        if ranked_ids:
            return ranked_ids, operation_name

    return [int(job_id) for job_id in job_ids], None


def read_text_file(file_path: str | Path) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    return path.read_text(encoding="utf-8")


def load_job_descriptions_from_csv(
    csv_path: str | Path,
    text_column: str = "content_text",
) -> pd.DataFrame:
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV.")

    df = df.copy()
    df["csv_row_index"] = df.index
    df[text_column] = df[text_column].fillna("").astype(str).map(str.strip)
    df = df[df[text_column] != ""].reset_index(drop=True)

    if df.empty:
        raise ValueError("No non-empty job descriptions found in CSV.")

    return df


def safe_error_label(error: Exception | str) -> str:
    if isinstance(error, Exception):
        return f"{type(error).__name__}: redacted"
    return "Error: redacted"


def log_operation_failure(operation_name: str, error: Exception | str) -> None:
    error_type = type(error).__name__ if isinstance(error, Exception) else "Error"
    print(
        f"ALERT: {operation_name} failed: error_type={error_type}",
        flush=True,
    )


def make_failed_result(operation_name: str, error: Exception | str) -> dict[str, Any]:
    return {
        "operation_name": operation_name,
        "status": "failed",
        "ranked_job_ids": [],
        "job_metrics": {},
        "error": safe_error_label(error),
    }


def make_skipped_result(operation_name: str, reason: str) -> dict[str, Any]:
    return {
        "operation_name": operation_name,
        "status": "skipped",
        "ranked_job_ids": [],
        "job_metrics": {},
        "error": reason,
    }


def _rank_with_ties(
    rows: Sequence[tuple[Any, float]],
    *,
    reverse: bool,
) -> dict[Any, int]:
    sorted_rows = sorted(rows, key=lambda row: row[1], reverse=reverse)
    ranks: dict[Any, int] = {}
    previous_score: float | None = None
    previous_rank = 0

    for position, (job_id, score) in enumerate(sorted_rows, start=1):
        if previous_score is None or score != previous_score:
            previous_rank = position
            previous_score = score
        ranks[job_id] = previous_rank

    return ranks


def _subset_by_job_ids(
    values: Sequence[Any] | None,
    job_ids: Sequence[int],
) -> list[Any] | None:
    if values is None:
        return None
    return [values[int(job_id)] for job_id in job_ids]


def _standard_wasserstein_result(
    *,
    operation_name: str,
    raw_results: Sequence[Mapping[str, Any]],
    local_job_ids: Sequence[Any],
    score_key: str,
    raw_metric_key: str,
) -> dict[str, Any]:
    ranked_job_ids: list[Any] = []
    scores_for_ties: list[tuple[Any, float]] = []
    rows_by_job_id: dict[Any, Mapping[str, Any]] = {}

    for row in raw_results:
        local_idx = int(row["job_index"])
        job_id = local_job_ids[local_idx]
        score = float(row[score_key])
        ranked_job_ids.append(job_id)
        scores_for_ties.append((job_id, score))
        rows_by_job_id[job_id] = row

    tied_ranks = _rank_with_ties(scores_for_ties, reverse=False)
    job_metrics: dict[Any, dict[str, Any]] = {}

    for job_id in ranked_job_ids:
        row = rows_by_job_id[job_id]
        score = float(row[score_key])
        job_metrics[job_id] = {
            "rank": tied_ranks[job_id],
            "score": score,
            "score_direction": LOWER_IS_BETTER,
            "raw_metrics": {
                raw_metric_key: score,
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"job_index", "rank", score_key, "job_description"}
                },
            },
        }

    ranked_job_ids.sort(key=lambda job_id: job_metrics[job_id]["rank"])
    return {
        "operation_name": operation_name,
        "status": "ok",
        "ranked_job_ids": ranked_job_ids,
        "job_metrics": job_metrics,
        "error": None,
    }


def run_word_wasserstein_operation(
    *,
    job_ids: Sequence[int],
    job_descriptions: Sequence[str],
    resume_text: str,
    minilm_model: SentenceTransformer,
    custom_stopwords: Iterable[str] | None,
    use_stopword_filter: bool,
    use_frequent_word_filter: bool,
    max_count: int | None,
    use_tfidf_filter: bool,
    top_tfidf_fraction: float | None,
    max_words_per_job: int | None,
    deduplicate_resume_words: bool,
    n_projections: int,
    random_state: int | None,
    embedding_batch_size: int,
    precomputed_job_word_lists: Sequence[Sequence[str]] | None = None,
    precomputed_job_word_embeddings: Sequence[np.ndarray] | None = None,
    precomputed_resume_words: Sequence[str] | None = None,
    precomputed_resume_embeddings: np.ndarray | None = None,
) -> dict[str, Any]:
    operation_name = "word_sliced_wasserstein"
    try:
        local_job_descriptions = _subset_by_job_ids(job_descriptions, job_ids) or []
        raw_results = main_rank_job_descriptions_by_wasserstein(
            job_descriptions=local_job_descriptions,
            resume=resume_text,
            model=minilm_model,
            custom_stopwords=custom_stopwords,
            use_stopword_filter=use_stopword_filter,
            use_frequent_word_filter=use_frequent_word_filter,
            max_count=max_count,
            use_tfidf_filter=use_tfidf_filter,
            top_tfidf_fraction=top_tfidf_fraction,
            max_words_per_job=max_words_per_job,
            deduplicate_resume_words=deduplicate_resume_words,
            n_projections=n_projections,
            random_state=random_state,
            embedding_batch_size=embedding_batch_size,
            return_debug_info=False,
            precomputed_job_word_lists=_subset_by_job_ids(precomputed_job_word_lists, job_ids),
            precomputed_job_embeddings=_subset_by_job_ids(precomputed_job_word_embeddings, job_ids),
            precomputed_resume_words=precomputed_resume_words,
            precomputed_resume_embeddings=precomputed_resume_embeddings,
        )
        print("Word Wasserstein operation complete")
        return _standard_wasserstein_result(
            operation_name=operation_name,
            raw_results=raw_results,
            local_job_ids=job_ids,
            score_key="distance",
            raw_metric_key="word_wasserstein_distance",
        )
    except Exception as exc:
        log_operation_failure(operation_name, exc)
        return make_failed_result(operation_name, exc)


def run_phrase_wasserstein_operation(
    *,
    job_ids: Sequence[int],
    job_descriptions: Sequence[str],
    resume_text: str,
    minilm_model: SentenceTransformer,
    n_projections: int,
    random_state: int,
    batch_size: int,
    normalize_embeddings: bool,
    min_chunk_words: int,
    max_chunk_words: int,
    include_sentences: bool,
    include_sentence_windows: bool,
    precomputed_job_phrase_chunks: Sequence[Sequence[str]] | None = None,
    precomputed_job_phrase_embeddings: Sequence[np.ndarray] | None = None,
    precomputed_resume_phrase_embeddings: np.ndarray | None = None,
) -> dict[str, Any]:
    operation_name = "phrase_sliced_wasserstein"
    try:
        local_job_descriptions = _subset_by_job_ids(job_descriptions, job_ids) or []
        raw_results = rank_job_descriptions_by_phrase_sliced_wasserstein(
            model=minilm_model,
            job_descriptions=local_job_descriptions,
            resume=resume_text,
            n_projections=n_projections,
            random_state=random_state,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            min_chunk_words=min_chunk_words,
            max_chunk_words=max_chunk_words,
            include_sentences=include_sentences,
            include_sentence_windows=include_sentence_windows,
            precomputed_job_phrase_chunks=_subset_by_job_ids(precomputed_job_phrase_chunks, job_ids),
            precomputed_job_phrase_embeddings=_subset_by_job_ids(precomputed_job_phrase_embeddings, job_ids),
            precomputed_resume_phrase_embeddings=precomputed_resume_phrase_embeddings,
        )
        print("Phrase Wasserstein operation complete")
        return _standard_wasserstein_result(
            operation_name=operation_name,
            raw_results=raw_results,
            local_job_ids=job_ids,
            score_key="distance",
            raw_metric_key="phrase_wasserstein_distance",
        )
    except Exception as exc:
        log_operation_failure(operation_name, exc)
        return make_failed_result(operation_name, exc)


def run_cross_encoder_operation(
    *,
    job_ids: Sequence[int],
    job_descriptions: Sequence[str],
    resume_text: str,
    cross_encoder_model: CrossEncoder | None,
    chunk_size_words: int,
    chunk_overlap_words: int,
    aggregation: str,
    top_k: int,
    batch_size: int,
) -> dict[str, Any]:
    operation_name = "cross_encoder"
    if cross_encoder_model is None:
        print("ALERT: cross_encoder skipped because no model was provided.")
        return make_skipped_result(operation_name, "No cross-encoder model was provided.")

    try:
        score_rows: list[tuple[Any, float]] = []
        raw_scores: dict[Any, float] = {}

        for job_id in job_ids:
            score = compare_texts_cross_encoder(
                text_a=resume_text,
                text_b=job_descriptions[int(job_id)],
                model=cross_encoder_model,
                chunk_size_words=chunk_size_words,
                chunk_overlap_words=chunk_overlap_words,
                aggregation=aggregation,
                top_k=top_k,
                batch_size=batch_size,
            )
            raw_scores[job_id] = float(score)
            score_rows.append((job_id, float(score)))

        tied_ranks = _rank_with_ties(score_rows, reverse=True)
        ranked_job_ids = [
            job_id
            for job_id, _ in sorted(score_rows, key=lambda row: row[1], reverse=True)
        ]
        job_metrics = {
            job_id: {
                "rank": tied_ranks[job_id],
                "score": raw_scores[job_id],
                "score_direction": HIGHER_IS_BETTER,
                "raw_metrics": {"cross_encoder_score": raw_scores[job_id]},
            }
            for job_id in ranked_job_ids
        }
        print("Cross-encoder operation complete")
        return {
            "operation_name": operation_name,
            "status": "ok",
            "ranked_job_ids": ranked_job_ids,
            "job_metrics": job_metrics,
            "error": None,
        }
    except Exception as exc:
        log_operation_failure(operation_name, exc)
        return make_failed_result(operation_name, exc)


def run_mahalanobis_operation(
    *,
    candidate_job_ids: Sequence[int],
    operation_results: Sequence[Mapping[str, Any]],
    requirements_cluster_distances: Sequence[float | None] | None,
    job_descriptions: Sequence[str],
    resume_text: str,
    cross_encoder_model: CrossEncoder | None,
    ce_chunk_size_words: int,
    ce_chunk_overlap_words: int,
    ce_aggregation: str,
    ce_top_k: int,
    ce_batch_size: int,
    scoring_mode: str,
) -> dict[str, Any]:
    operation_name = "mahalanobis_outlier"
    try:
        result = rank_mahalanobis_outliers(
            candidate_job_ids=candidate_job_ids,
            operation_results=operation_results,
            requirements_cluster_distances=requirements_cluster_distances,
            job_descriptions=job_descriptions,
            resume_text=resume_text,
            cross_encoder_model=cross_encoder_model,
            ce_chunk_size_words=ce_chunk_size_words,
            ce_chunk_overlap_words=ce_chunk_overlap_words,
            ce_aggregation=ce_aggregation,
            ce_top_k=ce_top_k,
            ce_batch_size=ce_batch_size,
            scoring_mode=scoring_mode,
        )
        print("Mahalanobis outlier operation complete")
        return result
    except Exception as exc:
        log_operation_failure(operation_name, exc)
        return make_failed_result(operation_name, exc)


def run_multi_metric_bad_fit_operation(
    *,
    job_ids: Sequence[int],
    operation_results: Sequence[Mapping[str, Any]],
    requirements_cluster_distances: Sequence[float | None] | None,
    bottom_fraction: float,
) -> dict[str, Any]:
    operation_name = "multi_metric_bad_fit_filter"
    try:
        result = run_multi_metric_bad_fit_filter(
            job_ids=job_ids,
            operation_results=operation_results,
            requirements_cluster_distances=requirements_cluster_distances,
            bottom_fraction=bottom_fraction,
        )
        print("Multi-metric bad-fit filter complete")
        return result
    except Exception as exc:
        log_operation_failure(operation_name, exc)
        return make_failed_result(operation_name, exc)


def run_technology_mismatch_operation(
    *,
    job_ids: Sequence[int],
    job_descriptions: Sequence[str],
    resume_text: str,
    enabled: bool,
    min_job_types: int,
    max_job_type_overlap_ratio: float,
    precomputed_resume_matches: dict[str, Any] | None = None,
) -> dict[str, Any]:
    operation_name = "technology_mismatch_filter"
    try:
        result = run_technology_mismatch_filter(
            job_ids=job_ids,
            job_descriptions=job_descriptions,
            resume_text=resume_text,
            enabled=enabled,
            min_job_types=min_job_types,
            max_job_type_overlap_ratio=max_job_type_overlap_ratio,
            precomputed_resume_matches=precomputed_resume_matches,
        )
        return result
    except Exception as exc:
        log_operation_failure(operation_name, exc)
        return make_failed_result(operation_name, exc)


def run_seniority_filter_operation(
    *,
    job_ids: Sequence[int],
    resume_text: str,
    minilm_model: SentenceTransformer,
    job_titles: Sequence[str] | None,
    job_requirements: Sequence[str] | None,
    job_min_years_experience: Sequence[Any] | None,
    precomputed_title_requirements_embeddings: np.ndarray | None,
    max_gap: float,
    max_junior_gap: float,
    enabled: bool,
    level_probability_alpha: float,
    batch_size: int,
    precomputed_resume_embedding: np.ndarray | None = None,
    precomputed_anchor_embeddings: np.ndarray | None = None,
) -> dict[str, Any]:
    operation_name = "seniority_filter"
    try:
        kwargs = dict(
            job_ids=job_ids,
            resume_text=resume_text,
            minilm_model=minilm_model,
            job_titles=job_titles,
            job_requirements=job_requirements,
            job_min_years_experience=job_min_years_experience,
            precomputed_title_requirements_embeddings=precomputed_title_requirements_embeddings,
            max_gap=max_gap,
            max_junior_gap=max_junior_gap,
            enabled=enabled,
            level_probability_alpha=level_probability_alpha,
            batch_size=batch_size,
        )
        if precomputed_resume_embedding is not None:
            kwargs["precomputed_resume_embedding"] = precomputed_resume_embedding
        if precomputed_anchor_embeddings is not None:
            kwargs["precomputed_anchor_embeddings"] = precomputed_anchor_embeddings
        return run_seniority_filter(**kwargs)
    except Exception as exc:
        log_operation_failure(operation_name, exc)
        return make_failed_result(operation_name, exc)


def run_resume_phrase_coverage_ranking_operation(
    *,
    job_ids: Sequence[int],
    job_titles: Sequence[str] | None,
    job_companies: Sequence[str] | None,
    resume_text: str,
    minilm_model: SentenceTransformer,
    precomputed_job_phrase_chunks: Sequence[Sequence[str]] | None,
    precomputed_job_phrase_embeddings: Sequence[np.ndarray] | None,
    precomputed_user_phrase_chunks: Sequence[str] | None,
    precomputed_user_phrase_embeddings: np.ndarray | None,
    resume_dataset_dir: str | Path | None,
    flag_percentile: float,
    bad_match_percentile: float,
    job_flag_fraction: float,
    min_chunk_words: int,
    max_chunk_words: int,
    include_sentences: bool,
    include_sentence_windows: bool,
    batch_size: int,
    normalize_embeddings: bool,
) -> dict[str, Any]:
    operation_name = "resume_phrase_coverage"
    try:
        return run_resume_phrase_coverage_operation(
            job_ids=job_ids,
            job_titles=job_titles,
            job_companies=job_companies,
            resume_text=resume_text,
            minilm_model=minilm_model,
            precomputed_job_phrase_chunks=precomputed_job_phrase_chunks,
            precomputed_job_phrase_embeddings=precomputed_job_phrase_embeddings,
            precomputed_user_phrase_chunks=precomputed_user_phrase_chunks,
            precomputed_user_phrase_embeddings=precomputed_user_phrase_embeddings,
            resume_dataset_dir=resume_dataset_dir,
            flag_percentile=flag_percentile,
            bad_match_percentile=bad_match_percentile,
            job_flag_fraction=job_flag_fraction,
            min_chunk_words=min_chunk_words,
            max_chunk_words=max_chunk_words,
            include_sentences=include_sentences,
            include_sentence_windows=include_sentence_windows,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )
    except Exception as exc:
        log_operation_failure(operation_name, exc)
        return make_failed_result(operation_name, exc)


def run_resume_phrase_job_coverage_ranking_operation(
    *,
    job_ids: Sequence[int],
    job_titles: Sequence[str] | None,
    job_companies: Sequence[str] | None,
    resume_text: str,
    minilm_model: SentenceTransformer,
    precomputed_job_phrase_chunks: Sequence[Sequence[str]] | None,
    precomputed_job_phrase_embeddings: Sequence[np.ndarray] | None,
    comparison_job_ids: Sequence[int] | None,
    comparison_job_phrase_chunks: Sequence[Sequence[str]] | None,
    comparison_job_phrase_embeddings: Sequence[np.ndarray] | None,
    precomputed_user_phrase_chunks: Sequence[str] | None,
    precomputed_user_phrase_embeddings: np.ndarray | None,
    flag_percentile: float,
    bad_match_percentile: float,
    min_chunk_words: int,
    max_chunk_words: int,
    include_sentences: bool,
    include_sentence_windows: bool,
    batch_size: int,
    normalize_embeddings: bool,
) -> dict[str, Any]:
    operation_name = "resume_phrase_job_coverage"
    try:
        return run_resume_phrase_job_coverage_operation(
            job_ids=job_ids,
            job_titles=job_titles,
            job_companies=job_companies,
            resume_text=resume_text,
            minilm_model=minilm_model,
            precomputed_job_phrase_chunks=precomputed_job_phrase_chunks,
            precomputed_job_phrase_embeddings=precomputed_job_phrase_embeddings,
            comparison_job_ids=comparison_job_ids,
            comparison_job_phrase_chunks=comparison_job_phrase_chunks,
            comparison_job_phrase_embeddings=comparison_job_phrase_embeddings,
            precomputed_user_phrase_chunks=precomputed_user_phrase_chunks,
            precomputed_user_phrase_embeddings=precomputed_user_phrase_embeddings,
            flag_percentile=flag_percentile,
            bad_match_percentile=bad_match_percentile,
            min_chunk_words=min_chunk_words,
            max_chunk_words=max_chunk_words,
            include_sentences=include_sentences,
            include_sentence_windows=include_sentence_windows,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )
    except Exception as exc:
        log_operation_failure(operation_name, exc)
        return make_failed_result(operation_name, exc)


def _metric_for_job(
    result_by_name: Mapping[str, Mapping[str, Any]],
    operation_name: str,
    job_id: Any,
    metric_name: str,
) -> Any:
    result = result_by_name.get(operation_name, {})
    metrics = result.get("job_metrics", {})
    job_metric = metrics.get(job_id) or {}
    raw_metrics = job_metric.get("raw_metrics", {})
    return raw_metrics.get(metric_name)


def _rank_for_job(
    result_by_name: Mapping[str, Mapping[str, Any]],
    operation_name: str,
    job_id: Any,
) -> Any:
    result = result_by_name.get(operation_name, {})
    metrics = result.get("job_metrics", {})
    job_metric = metrics.get(job_id) or {}
    return job_metric.get("rank")


def build_legacy_final_rows(
    *,
    final_job_ids: Sequence[int],
    job_descriptions: Sequence[str],
    operation_results: Sequence[Mapping[str, Any]],
    final_operation_name: str | None,
) -> list[dict[str, Any]]:
    result_by_name = {
        str(result.get("operation_name")): result
        for result in operation_results
        if result.get("status") == "ok"
    }
    final_rows: list[dict[str, Any]] = []

    for final_rank, job_id in enumerate(final_job_ids, start=1):
        job_index = int(job_id)
        cross_encoder_score = _metric_for_job(
            result_by_name,
            "cross_encoder",
            job_id,
            "cross_encoder_score",
        )
        final_score = None
        if final_operation_name is not None:
            final_operation = result_by_name.get(final_operation_name, {})
            final_metric = (final_operation.get("job_metrics", {}).get(job_id) or {})
            final_score = final_metric.get("score")

        final_rows.append(
            {
                "job_id": job_id,
                "job_index": job_index,
                "job_description": job_descriptions[job_index],
                "stage1_rank": _rank_for_job(result_by_name, "word_sliced_wasserstein", job_id),
                "stage1_word_wasserstein_distance": _metric_for_job(
                    result_by_name,
                    "word_sliced_wasserstein",
                    job_id,
                    "word_wasserstein_distance",
                ),
                "stage2_rank": _rank_for_job(result_by_name, "phrase_sliced_wasserstein", job_id),
                "stage2_phrase_wasserstein_distance": _metric_for_job(
                    result_by_name,
                    "phrase_sliced_wasserstein",
                    job_id,
                    "phrase_wasserstein_distance",
                ),
                "cross_encoder_score": cross_encoder_score,
                "seniority_score": _metric_for_job(
                    result_by_name,
                    "seniority_filter",
                    job_id,
                    "job_seniority_score",
                ),
                "seniority_gap": _metric_for_job(
                    result_by_name,
                    "seniority_filter",
                    job_id,
                    "seniority_gap",
                ),
                "job_title_seniority_floor": _metric_for_job(
                    result_by_name,
                    "seniority_filter",
                    job_id,
                    "job_title_seniority_floor",
                ),
                "job_title_seniority_floor_label": _metric_for_job(
                    result_by_name,
                    "seniority_filter",
                    job_id,
                    "job_title_seniority_floor_label",
                ),
                "job_title_seniority_floor_applied": _metric_for_job(
                    result_by_name,
                    "seniority_filter",
                    job_id,
                    "job_title_seniority_floor_applied",
                ),
                "job_title_seniority_ceiling": _metric_for_job(
                    result_by_name,
                    "seniority_filter",
                    job_id,
                    "job_title_seniority_ceiling",
                ),
                "job_title_seniority_ceiling_label": _metric_for_job(
                    result_by_name,
                    "seniority_filter",
                    job_id,
                    "job_title_seniority_ceiling_label",
                ),
                "job_title_seniority_ceiling_applied": _metric_for_job(
                    result_by_name,
                    "seniority_filter",
                    job_id,
                    "job_title_seniority_ceiling_applied",
                ),
                "job_yoe_seniority_floor": _metric_for_job(
                    result_by_name,
                    "seniority_filter",
                    job_id,
                    "job_yoe_seniority_floor",
                ),
                "job_yoe_seniority_floor_label": _metric_for_job(
                    result_by_name,
                    "seniority_filter",
                    job_id,
                    "job_yoe_seniority_floor_label",
                ),
                "job_yoe_seniority_floor_applied": _metric_for_job(
                    result_by_name,
                    "seniority_filter",
                    job_id,
                    "job_yoe_seniority_floor_applied",
                ),
                "mahalanobis_distance": _metric_for_job(
                    result_by_name,
                    "mahalanobis_outlier",
                    job_id,
                    "mahalanobis_distance",
                ),
                "technology_overlap_score": _metric_for_job(
                    result_by_name,
                    "technology_mismatch_filter",
                    job_id,
                    "technology_overlap_score",
                ),
                "technology_filter_removed": _metric_for_job(
                    result_by_name,
                    "technology_mismatch_filter",
                    job_id,
                    "technology_filter_removed",
                ),
                "technology_filter_reason": _metric_for_job(
                    result_by_name,
                    "technology_mismatch_filter",
                    job_id,
                    "technology_filter_reason",
                ),
                "job_technologies": _metric_for_job(
                    result_by_name,
                    "technology_mismatch_filter",
                    job_id,
                    "job_technologies",
                ),
                "job_technology_categories": _metric_for_job(
                    result_by_name,
                    "technology_mismatch_filter",
                    job_id,
                    "job_technology_categories",
                ),
                "resume_technologies": _metric_for_job(
                    result_by_name,
                    "technology_mismatch_filter",
                    job_id,
                    "resume_technologies",
                ),
                "resume_technology_categories": _metric_for_job(
                    result_by_name,
                    "technology_mismatch_filter",
                    job_id,
                    "resume_technology_categories",
                ),
                "technology_category_overlap": _metric_for_job(
                    result_by_name,
                    "technology_mismatch_filter",
                    job_id,
                    "technology_category_overlap",
                ),
                "resume_phrase_job_coverage_percent_flagged": _metric_for_job(
                    result_by_name,
                    "resume_phrase_job_coverage",
                    job_id,
                    "percent_flagged",
                ),
                "resume_phrase_coverage_bad_match_percent": _metric_for_job(
                    result_by_name,
                    "resume_phrase_coverage",
                    job_id,
                    "bad_match_percent",
                ),
                "resume_phrase_job_coverage_bad_match_percent": _metric_for_job(
                    result_by_name,
                    "resume_phrase_job_coverage",
                    job_id,
                    "bad_match_percent",
                ),
                "final_score": final_score,
                "final_rank": final_rank,
            }
        )

    return final_rows


def rank_jobs_multi_stage(
    *,
    resume_text: str,
    job_descriptions: Sequence[str],
    minilm_model: SentenceTransformer,
    cross_encoder_model: CrossEncoder | None,
    enable_cross_encoder: bool = True,
    job_titles: Sequence[str] | None = None,
    job_requirements: Sequence[str] | None = None,
    job_min_years_experience: Sequence[Any] | None = None,
    job_companies: Sequence[str] | None = None,
    job_ids: Sequence[int] | None = None,
    word_keep_n: int = 50,
    phrase_keep_n: int = 15,
    word_custom_stopwords: Iterable[str] | None = None,
    word_use_stopword_filter: bool = True,
    word_use_frequent_word_filter: bool = True,
    word_max_count: int | None = 4,
    word_use_tfidf_filter: bool = True,
    word_top_tfidf_fraction: float | None = 0.10,
    word_max_words_per_job: int | None = 30,
    word_deduplicate_resume_words: bool = True,
    word_n_projections: int = 50,
    word_random_state: int | None = 0,
    word_embedding_batch_size: int = 128,
    phrase_n_projections: int = 64,
    phrase_random_state: int = 0,
    phrase_batch_size: int = 64,
    phrase_normalize_embeddings: bool = False,
    phrase_min_chunk_words: int = 3,
    phrase_max_chunk_words: int = 24,
    phrase_include_sentences: bool = True,
    phrase_include_sentence_windows: bool = True,
    ce_chunk_size_words: int = 180,
    ce_chunk_overlap_words: int = 40,
    ce_aggregation: str = "top_k_mean",
    ce_top_k: int = 3,
    ce_batch_size: int = 32,
    cross_encoder_union_top_k_per_ranker: int = 25,
    poor_match_max_rank_per_step: int = 150,
    mahalanobis_input_top_k: int = 10,
    mahalanobis_remove_bottom_fraction: float = 0.66,
    multi_metric_bad_fit_bottom_fraction: float = 0.25,
    enable_technology_mismatch_filter: bool = True,
    tech_filter_min_job_types: int = 3,
    tech_filter_max_job_type_overlap_ratio: float = 0.333333,
    precomputed_job_word_lists: Sequence[Sequence[str]] | None = None,
    precomputed_title_requirements_embeddings: np.ndarray | None = None,
    precomputed_job_phrase_chunks: Sequence[Sequence[str]] | None = None,
    precomputed_job_word_embeddings: Sequence[np.ndarray] | None = None,
    precomputed_job_phrase_embeddings: Sequence[np.ndarray] | None = None,
    resume_phrase_job_coverage_comparison_job_ids: Sequence[int] | None = None,
    resume_phrase_job_coverage_comparison_phrase_chunks: Sequence[Sequence[str]] | None = None,
    resume_phrase_job_coverage_comparison_phrase_embeddings: Sequence[np.ndarray] | None = None,
    requirements_cluster_distances: Sequence[float | None] | None = None,
    resume_phrase_coverage_dataset_dir: str | Path | None = None,
    resume_phrase_coverage_flag_percentile: float = 10.0,
    resume_phrase_coverage_bad_match_percentile: float = 90.0,
    resume_phrase_coverage_job_flag_fraction: float = 0.30,
    resume_phrase_coverage_min_chunk_words: int = 3,
    resume_phrase_coverage_max_chunk_words: int = 24,
    resume_phrase_coverage_include_sentences: bool = True,
    resume_phrase_coverage_include_sentence_windows: bool = True,
    resume_phrase_coverage_batch_size: int = 64,
    resume_phrase_coverage_normalize_embeddings: bool = False,
    resume_phrase_job_coverage_flag_percentile: float = 10.0,
    resume_phrase_job_coverage_bad_match_percentile: float = 90.0,
    resume_phrase_coverage_remove_bottom_good_fit_fraction: float = 0.0,
    resume_phrase_coverage_remove_top_bad_match_fraction: float = 0.0,
    resume_phrase_job_coverage_remove_bottom_good_fit_fraction: float = 0.0,
    resume_phrase_job_coverage_remove_top_bad_match_fraction: float = 0.0,
    return_operation_results: bool = False,
    precomputed_resume_profile: dict[str, Any] | None = None,
    all_candidates_through_all_metrics: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
    pipeline_started_at = time.perf_counter()

    if precomputed_resume_profile is None and (not isinstance(resume_text, str) or not resume_text.strip()):
        raise ValueError("resume_text must be a non-empty string.")

    cached_profile = precomputed_resume_profile or {}
    cached_overall_embedding = (
        np.asarray(cached_profile["overall_embedding"], dtype=np.float32)
        if cached_profile.get("overall_embedding") is not None
        else None
    )
    cached_word_embeddings = (
        np.asarray(cached_profile["word_embeddings"], dtype=np.float32)
        if cached_profile.get("word_embeddings") is not None
        else None
    )
    cached_phrase_embeddings = (
        np.asarray(cached_profile["phrase_embeddings"], dtype=np.float32)
        if cached_profile.get("phrase_embeddings") is not None
        else None
    )
    cached_phrase_chunks = (
        [f"cached-phrase-{index}" for index in range(len(cached_phrase_embeddings))]
        if cached_phrase_embeddings is not None
        else None
    )

    if not job_descriptions:
        raise ValueError("job_descriptions cannot be empty.")

    if precomputed_resume_profile is not None and (
        precomputed_job_word_embeddings is None
        or precomputed_job_phrase_embeddings is None
        or precomputed_title_requirements_embeddings is None
    ):
        raise RuntimeError("Cached resume ranking requires all precomputed job-side embedding artifacts.")

    if job_titles is not None and len(job_titles) != len(job_descriptions):
        raise ValueError("job_titles must match job_descriptions length when provided.")

    if job_requirements is not None and len(job_requirements) != len(job_descriptions):
        raise ValueError("job_requirements must match job_descriptions length when provided.")

    if job_min_years_experience is not None and len(job_min_years_experience) != len(job_descriptions):
        raise ValueError("job_min_years_experience must match job_descriptions length when provided.")

    if job_companies is not None and len(job_companies) != len(job_descriptions):
        raise ValueError("job_companies must match job_descriptions length when provided.")

    if precomputed_title_requirements_embeddings is not None and len(precomputed_title_requirements_embeddings) != len(job_descriptions):
        raise ValueError("precomputed_title_requirements_embeddings must match job_descriptions length when provided.")

    if word_keep_n <= 0:
        raise ValueError("word_keep_n must be positive.")

    if phrase_keep_n <= 0:
        raise ValueError("phrase_keep_n must be positive.")

    if enable_cross_encoder and cross_encoder_union_top_k_per_ranker <= 0:
        raise ValueError("cross_encoder_union_top_k_per_ranker must be positive.")

    if poor_match_max_rank_per_step <= 0:
        raise ValueError("poor_match_max_rank_per_step must be positive.")

    if mahalanobis_input_top_k <= 0:
        raise ValueError("mahalanobis_input_top_k must be positive.")

    if not 0 <= mahalanobis_remove_bottom_fraction < 1:
        raise ValueError("mahalanobis_remove_bottom_fraction must be in [0, 1).")

    if not 0 <= multi_metric_bad_fit_bottom_fraction < 1:
        raise ValueError("multi_metric_bad_fit_bottom_fraction must be in [0, 1).")

    if tech_filter_min_job_types < 0:
        raise ValueError("tech_filter_min_job_types cannot be negative.")

    if not 0 <= tech_filter_max_job_type_overlap_ratio <= 1:
        raise ValueError("tech_filter_max_job_type_overlap_ratio must be in [0, 1].")
    _validate_fraction(
        "resume_phrase_coverage_remove_bottom_good_fit_fraction",
        resume_phrase_coverage_remove_bottom_good_fit_fraction,
    )
    _validate_fraction(
        "resume_phrase_coverage_remove_top_bad_match_fraction",
        resume_phrase_coverage_remove_top_bad_match_fraction,
    )
    _validate_fraction(
        "resume_phrase_job_coverage_remove_bottom_good_fit_fraction",
        resume_phrase_job_coverage_remove_bottom_good_fit_fraction,
    )
    _validate_fraction(
        "resume_phrase_job_coverage_remove_top_bad_match_fraction",
        resume_phrase_job_coverage_remove_top_bad_match_fraction,
    )

    current_job_ids = list(job_ids) if job_ids is not None else list(range(len(job_descriptions)))
    if not current_job_ids:
        raise ValueError("job_ids cannot be empty.")
    all_job_ids = list(current_job_ids)

    def reduce_or_keep_all(
        candidate_ids: Sequence[int],
        operation_result: Mapping[str, Any],
    ) -> list[int]:
        if all_candidates_through_all_metrics:
            return [int(job_id) for job_id in candidate_ids]
        return reduce_job_ids(
            current_job_ids=candidate_ids,
            operation_result=operation_result,
            reduction_policies=reduction_policies,
        )

    llm_job_requirements = job_requirements if job_requirements is not None else job_descriptions
    enable_seniority_filter = env_bool("ENABLE_SENIORITY_FILTER", True)
    enable_word_wasserstein = env_bool("ENABLE_WORD_WASSERSTEIN", True)
    seniority_filter_max_gap = env_float("SENIORITY_FILTER_MAX_GAP", 1.5)
    seniority_filter_max_junior_gap = env_float("SENIORITY_FILTER_MAX_JUNIOR_GAP", 10.0)
    seniority_filter_level_probability_alpha = env_float("SENIORITY_FILTER_LEVEL_PROBABILITY_ALPHA", 3.0)
    final_ranking_mode = os.environ.get("FINAL_RANKING_MODE", "minimax").strip().lower()
    mahalanobis_scoring_mode = os.environ.get("MAHALANOBIS_SCORING_MODE", "distance_outlier").strip().lower()
    mahalanobis_min_remaining_jobs = env_nonnegative_int("MAHALANOBIS_MIN_REMAINING_JOBS", 0)
    multi_metric_bad_fit_min_remaining_jobs = env_nonnegative_int("MULTI_METRIC_BAD_FIT_MIN_REMAINING_JOBS", 0)
    technology_mismatch_min_remaining_jobs = env_nonnegative_int("TECHNOLOGY_MISMATCH_MIN_REMAINING_JOBS", 0)
    resume_phrase_coverage_min_remaining_jobs = env_nonnegative_int("RESUME_PHRASE_COVERAGE_MIN_REMAINING_JOBS", 0)
    resume_phrase_job_coverage_min_remaining_jobs = env_nonnegative_int(
        "RESUME_PHRASE_JOB_COVERAGE_MIN_REMAINING_JOBS",
        0,
    )
    llm_bad_match_min_remaining_jobs = env_nonnegative_int("LLM_BAD_MATCH_MIN_REMAINING_JOBS", 0)

    reduction_policies = {
        "seniority_filter": {"filter_raw_metric": "is_filtered", "exclude_value": True},
        "word_sliced_wasserstein": {"top_n": poor_match_max_rank_per_step},
        "phrase_sliced_wasserstein": {"top_n": poor_match_max_rank_per_step},
        "cross_encoder": {"top_n": poor_match_max_rank_per_step},
        "mahalanobis_outlier": {
            "top_fraction": 1.0 - mahalanobis_remove_bottom_fraction,
            "min_remaining_jobs": mahalanobis_min_remaining_jobs,
        },
        "multi_metric_bad_fit_filter": {
            "keep_score_equals": 1,
            "min_remaining_jobs": multi_metric_bad_fit_min_remaining_jobs,
        },
        "technology_mismatch_filter": {
            "keep_score_equals": 1,
            "min_remaining_jobs": technology_mismatch_min_remaining_jobs,
        },
        "resume_phrase_coverage": {"top_fraction": 1.0},
        "llm_bad_match_filter": {
            "keep_score_equals": 1,
            "min_remaining_jobs": llm_bad_match_min_remaining_jobs,
        },
    }
    operation_results: list[dict[str, Any]] = []
    user_phrase_chunks: list[str] | None = None
    user_phrase_embeddings: np.ndarray | None = None

    step_started_at = time.perf_counter()
    jobs_before_step = len(current_job_ids)
    seniority_result = run_seniority_filter_operation(
        job_ids=current_job_ids,
        resume_text=resume_text,
        minilm_model=minilm_model,
        job_titles=job_titles,
        job_requirements=llm_job_requirements,
        job_min_years_experience=job_min_years_experience,
        precomputed_title_requirements_embeddings=precomputed_title_requirements_embeddings,
        max_gap=seniority_filter_max_gap,
        max_junior_gap=seniority_filter_max_junior_gap,
        enabled=enable_seniority_filter,
        level_probability_alpha=seniority_filter_level_probability_alpha,
        batch_size=word_embedding_batch_size,
        precomputed_resume_embedding=cached_overall_embedding,
        precomputed_anchor_embeddings=(
            np.asarray(cached_profile["seniority_anchor_embeddings"], dtype=np.float32)
            if cached_profile.get("seniority_anchor_embeddings") is not None
            else None
        ),
    )
    operation_results.append(seniority_result)
    seniority_input_job_ids = list(current_job_ids)
    current_job_ids = reduce_or_keep_all(current_job_ids, seniority_result)
    current_job_id_set = set(current_job_ids)
    seniority_removed_job_ids = [
        job_id
        for job_id in seniority_input_job_ids
        if job_id not in current_job_id_set
    ]
    print_seniority_filter_job_snapshot(
        "removed",
        seniority_removed_job_ids,
        job_titles=job_titles,
        job_companies=job_companies,
        seniority_result=seniority_result,
    )
    print_seniority_filter_job_snapshot(
        "remaining",
        current_job_ids,
        job_titles=job_titles,
        job_companies=job_companies,
        seniority_result=seniority_result,
    )
    print_ranking_timing(
        "seniority_filter",
        step_started_at,
        jobs_before=jobs_before_step,
        jobs_after=len(current_job_ids),
        enabled=enable_seniority_filter,
        max_gap=seniority_filter_max_gap,
        max_junior_gap=seniority_filter_max_junior_gap,
        status=seniority_result.get("status"),
    )
    if not current_job_ids:
        print_ranking_timing("rank_jobs_multi_stage_total", pipeline_started_at, final_jobs=0)
        if return_operation_results:
            return {
                "results": [],
                "pre_llm_results": [],
                "operation_results": operation_results,
                "final_operation_name": "seniority_filter",
                "remaining_job_ids": [],
            }
        return []

    step_started_at = time.perf_counter()
    jobs_before_step = len(current_job_ids)
    if enable_word_wasserstein:
        word_result = run_word_wasserstein_operation(
            job_ids=current_job_ids,
            job_descriptions=job_descriptions,
            resume_text=resume_text,
            minilm_model=minilm_model,
            custom_stopwords=word_custom_stopwords,
            use_stopword_filter=word_use_stopword_filter,
            use_frequent_word_filter=word_use_frequent_word_filter,
            max_count=word_max_count,
            use_tfidf_filter=word_use_tfidf_filter,
            top_tfidf_fraction=word_top_tfidf_fraction,
            max_words_per_job=word_max_words_per_job,
            deduplicate_resume_words=word_deduplicate_resume_words,
            n_projections=word_n_projections,
            random_state=word_random_state,
            embedding_batch_size=word_embedding_batch_size,
            precomputed_job_word_lists=precomputed_job_word_lists,
            precomputed_job_word_embeddings=precomputed_job_word_embeddings,
            precomputed_resume_words=(
                [f"cached-word-{index}" for index in range(len(cached_word_embeddings))]
                if cached_word_embeddings is not None
                else None
            ),
            precomputed_resume_embeddings=cached_word_embeddings,
        )
    else:
        print("Word Wasserstein ranking step skipped: ENABLE_WORD_WASSERSTEIN=false.", flush=True)
        word_result = make_skipped_result(
            "word_sliced_wasserstein",
            "Disabled by ENABLE_WORD_WASSERSTEIN=false.",
        )
    operation_results.append(word_result)
    current_job_ids = reduce_or_keep_all(current_job_ids, word_result)
    print_ranking_timing(
        "word_sliced_wasserstein",
        step_started_at,
        jobs_before=jobs_before_step,
        jobs_after=len(current_job_ids),
        enabled=enable_word_wasserstein,
        status=word_result.get("status"),
    )

    step_started_at = time.perf_counter()
    jobs_before_step = len(current_job_ids)
    phrase_result = run_phrase_wasserstein_operation(
        job_ids=current_job_ids,
        job_descriptions=job_descriptions,
        resume_text=resume_text,
        minilm_model=minilm_model,
        n_projections=phrase_n_projections,
        random_state=phrase_random_state,
        batch_size=phrase_batch_size,
        normalize_embeddings=phrase_normalize_embeddings,
        min_chunk_words=phrase_min_chunk_words,
        max_chunk_words=phrase_max_chunk_words,
        include_sentences=phrase_include_sentences,
        include_sentence_windows=phrase_include_sentence_windows,
        precomputed_job_phrase_chunks=precomputed_job_phrase_chunks,
        precomputed_job_phrase_embeddings=precomputed_job_phrase_embeddings,
        precomputed_resume_phrase_embeddings=cached_phrase_embeddings,
    )
    operation_results.append(phrase_result)
    current_job_ids = reduce_or_keep_all(current_job_ids, phrase_result)
    print_ranking_timing(
        "phrase_sliced_wasserstein",
        step_started_at,
        jobs_before=jobs_before_step,
        jobs_after=len(current_job_ids),
        status=phrase_result.get("status"),
    )

    if enable_cross_encoder and precomputed_resume_profile is None:
        # Cross-encoder receives only the top N jobs that survived prior poor-match cutoffs.
        cross_encoder_job_ids = (
            list(current_job_ids)
            if all_candidates_through_all_metrics
            else current_job_ids[:min(cross_encoder_union_top_k_per_ranker, len(current_job_ids))]
        )

        step_started_at = time.perf_counter()
        jobs_before_step = len(cross_encoder_job_ids)
        cross_encoder_result = run_cross_encoder_operation(
            job_ids=cross_encoder_job_ids,
            job_descriptions=job_descriptions,
            resume_text=resume_text,
            cross_encoder_model=cross_encoder_model,
            chunk_size_words=ce_chunk_size_words,
            chunk_overlap_words=ce_chunk_overlap_words,
            aggregation=ce_aggregation,
            top_k=ce_top_k,
            batch_size=ce_batch_size,
        )
        operation_results.append(cross_encoder_result)
        current_job_ids = reduce_or_keep_all(cross_encoder_job_ids, cross_encoder_result)
        print_ranking_timing(
            "cross_encoder",
            step_started_at,
            jobs_before=jobs_before_step,
            jobs_after=len(current_job_ids),
            status=cross_encoder_result.get("status"),
        )
    else:
        print("Cross-encoder ranking step skipped: ENABLE_CROSS_ENCODER=false.", flush=True)

    step_started_at = time.perf_counter()
    mahalanobis_candidate_ids = (
        list(all_job_ids)
        if all_candidates_through_all_metrics
        else build_mahalanobis_candidate_job_ids(
            operation_results=operation_results,
            top_k_per_ranker=mahalanobis_input_top_k,
        )
    )
    if not mahalanobis_candidate_ids:
        mahalanobis_candidate_ids = list(current_job_ids)
    print_ranking_timing(
        "build_mahalanobis_candidates",
        step_started_at,
        candidate_jobs=len(mahalanobis_candidate_ids),
    )

    step_started_at = time.perf_counter()
    jobs_before_step = len(mahalanobis_candidate_ids)
    mahalanobis_result = run_mahalanobis_operation(
        candidate_job_ids=mahalanobis_candidate_ids,
        operation_results=operation_results,
        requirements_cluster_distances=requirements_cluster_distances,
        job_descriptions=job_descriptions,
        resume_text=resume_text,
        cross_encoder_model=cross_encoder_model if enable_cross_encoder else None,
        ce_chunk_size_words=ce_chunk_size_words,
        ce_chunk_overlap_words=ce_chunk_overlap_words,
        ce_aggregation=ce_aggregation,
        ce_top_k=ce_top_k,
        ce_batch_size=ce_batch_size,
        scoring_mode=mahalanobis_scoring_mode,
    )
    operation_results.append(mahalanobis_result)
    current_job_ids = reduce_or_keep_all(mahalanobis_candidate_ids, mahalanobis_result)
    print_ranking_timing(
        "mahalanobis_outlier",
        step_started_at,
        jobs_before=jobs_before_step,
        jobs_after=len(current_job_ids),
        status=mahalanobis_result.get("status"),
        scoring_mode=mahalanobis_scoring_mode,
    )

    step_started_at = time.perf_counter()
    jobs_before_step = len(current_job_ids)
    multi_metric_bad_fit_result = run_multi_metric_bad_fit_operation(
        job_ids=current_job_ids,
        operation_results=operation_results,
        requirements_cluster_distances=requirements_cluster_distances,
        bottom_fraction=multi_metric_bad_fit_bottom_fraction,
    )
    operation_results.append(multi_metric_bad_fit_result)
    current_job_ids = reduce_or_keep_all(current_job_ids, multi_metric_bad_fit_result)
    print_ranking_timing(
        "multi_metric_bad_fit_filter",
        step_started_at,
        jobs_before=jobs_before_step,
        jobs_after=len(current_job_ids),
        status=multi_metric_bad_fit_result.get("status"),
    )

    step_started_at = time.perf_counter()
    jobs_before_step = len(current_job_ids)
    technology_mismatch_result = run_technology_mismatch_operation(
        job_ids=current_job_ids,
        job_descriptions=job_descriptions,
        resume_text=resume_text,
        enabled=enable_technology_mismatch_filter,
        min_job_types=tech_filter_min_job_types,
        max_job_type_overlap_ratio=tech_filter_max_job_type_overlap_ratio,
        precomputed_resume_matches=(
            {
                "technologies": cached_profile.get("technologies", []),
                "categories": cached_profile.get("technology_categories", []),
            }
            if precomputed_resume_profile is not None
            else None
        ),
    )
    operation_results.append(technology_mismatch_result)
    current_job_ids = reduce_or_keep_all(current_job_ids, technology_mismatch_result)
    print_ranking_timing(
        "technology_mismatch_filter",
        step_started_at,
        jobs_before=jobs_before_step,
        jobs_after=len(current_job_ids),
        status=technology_mismatch_result.get("status"),
    )

    if cached_phrase_embeddings is not None:
        user_phrase_chunks = cached_phrase_chunks
        user_phrase_embeddings = cached_phrase_embeddings
    elif precomputed_job_phrase_chunks is not None and precomputed_job_phrase_embeddings is not None:
        try:
            user_phrase_chunks, user_phrase_embeddings = compute_resume_phrase_inputs(
                resume_text=resume_text,
                minilm_model=minilm_model,
                min_chunk_words=resume_phrase_coverage_min_chunk_words,
                max_chunk_words=resume_phrase_coverage_max_chunk_words,
                include_sentences=resume_phrase_coverage_include_sentences,
                include_sentence_windows=resume_phrase_coverage_include_sentence_windows,
                batch_size=resume_phrase_coverage_batch_size,
                normalize_embeddings=resume_phrase_coverage_normalize_embeddings,
            )
        except Exception as exc:
            print(
                "ALERT: failed to precompute user resume phrase embeddings: "
                f"error_type={type(exc).__name__}",
                flush=True,
            )
            user_phrase_chunks = None
            user_phrase_embeddings = None

    step_started_at = time.perf_counter()
    jobs_before_step = len(current_job_ids)
    resume_phrase_coverage_result = run_resume_phrase_coverage_ranking_operation(
        job_ids=current_job_ids,
        job_titles=job_titles,
        job_companies=job_companies,
        resume_text=resume_text,
        minilm_model=minilm_model,
        precomputed_job_phrase_chunks=precomputed_job_phrase_chunks,
        precomputed_job_phrase_embeddings=precomputed_job_phrase_embeddings,
        precomputed_user_phrase_chunks=user_phrase_chunks,
        precomputed_user_phrase_embeddings=user_phrase_embeddings,
        resume_dataset_dir=resume_phrase_coverage_dataset_dir,
        flag_percentile=resume_phrase_coverage_flag_percentile,
        bad_match_percentile=resume_phrase_coverage_bad_match_percentile,
        job_flag_fraction=resume_phrase_coverage_job_flag_fraction,
        min_chunk_words=resume_phrase_coverage_min_chunk_words,
        max_chunk_words=resume_phrase_coverage_max_chunk_words,
        include_sentences=resume_phrase_coverage_include_sentences,
        include_sentence_windows=resume_phrase_coverage_include_sentence_windows,
        batch_size=resume_phrase_coverage_batch_size,
        normalize_embeddings=resume_phrase_coverage_normalize_embeddings,
    )
    operation_results.append(resume_phrase_coverage_result)
    if not all_candidates_through_all_metrics:
        current_job_ids = reduce_by_coverage_quality(
            current_job_ids=[int(job_id) for job_id in current_job_ids],
            operation_result=resume_phrase_coverage_result,
            remove_bottom_good_fit_fraction=resume_phrase_coverage_remove_bottom_good_fit_fraction,
            remove_top_bad_match_fraction=resume_phrase_coverage_remove_top_bad_match_fraction,
            min_remaining_jobs=resume_phrase_coverage_min_remaining_jobs,
        )
    print_ranking_timing(
        "resume_phrase_coverage",
        step_started_at,
        jobs_before=jobs_before_step,
        jobs_after=len(current_job_ids),
        status=resume_phrase_coverage_result.get("status"),
        remove_bottom_good_fit_fraction=resume_phrase_coverage_remove_bottom_good_fit_fraction,
        remove_top_bad_match_fraction=resume_phrase_coverage_remove_top_bad_match_fraction,
        min_remaining_jobs=resume_phrase_coverage_min_remaining_jobs,
    )

    step_started_at = time.perf_counter()
    jobs_before_step = len(current_job_ids)
    resume_phrase_job_coverage_result = run_resume_phrase_job_coverage_ranking_operation(
        job_ids=current_job_ids,
        job_titles=job_titles,
        job_companies=job_companies,
        resume_text=resume_text,
        minilm_model=minilm_model,
        precomputed_job_phrase_chunks=precomputed_job_phrase_chunks,
        precomputed_job_phrase_embeddings=precomputed_job_phrase_embeddings,
        comparison_job_ids=resume_phrase_job_coverage_comparison_job_ids,
        comparison_job_phrase_chunks=resume_phrase_job_coverage_comparison_phrase_chunks,
        comparison_job_phrase_embeddings=resume_phrase_job_coverage_comparison_phrase_embeddings,
        precomputed_user_phrase_chunks=user_phrase_chunks,
        precomputed_user_phrase_embeddings=user_phrase_embeddings,
        flag_percentile=resume_phrase_job_coverage_flag_percentile,
        bad_match_percentile=resume_phrase_job_coverage_bad_match_percentile,
        min_chunk_words=resume_phrase_coverage_min_chunk_words,
        max_chunk_words=resume_phrase_coverage_max_chunk_words,
        include_sentences=resume_phrase_coverage_include_sentences,
        include_sentence_windows=resume_phrase_coverage_include_sentence_windows,
        batch_size=resume_phrase_coverage_batch_size,
        normalize_embeddings=resume_phrase_coverage_normalize_embeddings,
    )
    operation_results.append(resume_phrase_job_coverage_result)
    if not all_candidates_through_all_metrics:
        current_job_ids = reduce_by_coverage_quality(
            current_job_ids=[int(job_id) for job_id in current_job_ids],
            operation_result=resume_phrase_job_coverage_result,
            remove_bottom_good_fit_fraction=resume_phrase_job_coverage_remove_bottom_good_fit_fraction,
            remove_top_bad_match_fraction=resume_phrase_job_coverage_remove_top_bad_match_fraction,
            min_remaining_jobs=resume_phrase_job_coverage_min_remaining_jobs,
        )
    print_ranking_timing(
        "resume_phrase_job_coverage",
        step_started_at,
        jobs_before=jobs_before_step,
        jobs_after=len(current_job_ids),
        status=resume_phrase_job_coverage_result.get("status"),
        remove_bottom_good_fit_fraction=resume_phrase_job_coverage_remove_bottom_good_fit_fraction,
        remove_top_bad_match_fraction=resume_phrase_job_coverage_remove_top_bad_match_fraction,
        min_remaining_jobs=resume_phrase_job_coverage_min_remaining_jobs,
    )

    pre_llm_job_ids, pre_llm_operation_name = order_job_ids_by_operation(
        current_job_ids,
        resume_phrase_job_coverage_result,
    )
    pre_llm_rows = build_legacy_final_rows(
        final_job_ids=pre_llm_job_ids,
        job_descriptions=job_descriptions,
        operation_results=operation_results,
        final_operation_name=pre_llm_operation_name,
    )

    # When enabled, LLM screens coverage candidates by descending flagged phrase fraction.
    llm_enabled = env_bool("ENABLE_LLM_BAD_MATCH_FILTER", False) and precomputed_resume_profile is None
    if llm_enabled:
        llm_max_jobs = env_int("LLM_BAD_MATCH_MAX_JOBS", 5)
        llm_job_ids = sort_job_ids_by_resume_phrase_coverage(
            current_job_ids,
            resume_phrase_coverage_result,
        )[:llm_max_jobs]
    else:
        llm_job_ids = list(current_job_ids)
    step_started_at = time.perf_counter()
    jobs_before_step = len(llm_job_ids)
    if precomputed_resume_profile is not None:
        llm_bad_match_result = make_skipped_result(
            "llm_bad_match_filter",
            "Cached resume profiles do not retain the raw text required by the LLM filter.",
        )
        llm_bad_match_result["ranked_job_ids"] = list(llm_job_ids)
    else:
        llm_bad_match_result = run_llm_bad_match_filter(
            job_ids=llm_job_ids,
            job_titles=job_titles,
            job_requirements=llm_job_requirements,
            resume_text=resume_text,
        )
    operation_results.append(llm_bad_match_result)
    current_job_ids = reduce_or_keep_all(llm_job_ids, llm_bad_match_result)
    print_ranking_timing(
        "llm_bad_match_filter",
        step_started_at,
        jobs_before=jobs_before_step,
        jobs_after=len(current_job_ids),
        enabled=llm_enabled,
        status=llm_bad_match_result.get("status"),
    )
    llm_filtered_job_ids = {
        int(job_id)
        for job_id, metric in (llm_bad_match_result.get("job_metrics", {}) or {}).items()
        if (metric.get("raw_metrics") or {}).get("is_bad_match") is True
    }
    pre_llm_rows = [
        row
        for row in pre_llm_rows
        if int(row.get("job_id")) not in llm_filtered_job_ids
    ]

    step_started_at = time.perf_counter()
    forest_enabled = env_bool("ENABLE_PAIR_INDEPENDENT_BAD_MATCH_FOREST", False)
    jobs_before_step = len(current_job_ids)
    if forest_enabled:
        pair_independent_forest_result = run_pair_independent_bad_match_filter(
            job_ids=current_job_ids,
            operation_results=operation_results,
            requirements_cluster_distances=requirements_cluster_distances,
        )
        if pair_independent_forest_result.get("status") == "ok":
            forest_filtered_job_id_set = {
                job_id
                for job_id, metric in pair_independent_forest_result.get("job_metrics", {}).items()
                if (metric.get("raw_metrics") or {}).get("semantic_bad_match_flagged") is True
            }
            forest_filtered_job_ids = [
                job_id for job_id in current_job_ids if job_id in forest_filtered_job_id_set
            ]
            print_pair_independent_forest_removals(
                forest_filtered_job_ids,
                job_titles=job_titles,
                operation_result=pair_independent_forest_result,
            )
            current_job_ids = [
                job_id
                for job_id in current_job_ids
                if job_id not in forest_filtered_job_id_set
            ]
            pre_llm_rows = [
                row for row in pre_llm_rows if row.get("job_id") not in forest_filtered_job_id_set
            ]
    else:
        pair_independent_forest_result = make_skipped_result(
            "pair_independent_bad_match_forest",
            "Disabled by ENABLE_PAIR_INDEPENDENT_BAD_MATCH_FOREST=false.",
        )
        pair_independent_forest_result["ranked_job_ids"] = list(current_job_ids)
    operation_results.append(pair_independent_forest_result)
    print_ranking_timing(
        "pair_independent_bad_match_forest",
        step_started_at,
        jobs_before=jobs_before_step,
        jobs_after=len(current_job_ids),
        enabled=forest_enabled,
        status=pair_independent_forest_result.get("status"),
    )

    step_started_at = time.perf_counter()
    final_job_ids, final_operation_name = choose_final_job_order(
        remaining_job_ids=current_job_ids,
        operation_results=operation_results,
        ranking_mode=final_ranking_mode,
    )
    final_rows = build_legacy_final_rows(
        final_job_ids=final_job_ids,
        job_descriptions=job_descriptions,
        operation_results=operation_results,
        final_operation_name=final_operation_name,
    )
    print_ranking_timing(
        "final_ordering",
        step_started_at,
        remaining_jobs=len(current_job_ids),
        final_jobs=len(final_rows),
        final_operation=final_operation_name,
        final_ranking_mode=final_ranking_mode,
    )
    print_ranking_timing("rank_jobs_multi_stage_total", pipeline_started_at, final_jobs=len(final_rows))

    if return_operation_results:
        return {
            "results": final_rows,
            "pre_llm_results": pre_llm_rows,
            "operation_results": operation_results,
            "final_operation_name": final_operation_name,
            "remaining_job_ids": list(current_job_ids),
        }

    return final_rows


def main() -> None:
    resume_text = read_text_file(TEST_RESUME_PATH)
    jobs_df = load_job_descriptions_from_csv(
        csv_path=TEST_JOBS_CSV_PATH,
        text_column=TEST_JOB_TEXT_COLUMN,
    )
    job_descriptions = jobs_df[TEST_JOB_TEXT_COLUMN].tolist()
    job_titles = (
        jobs_df["title"].fillna("").astype(str).tolist()
        if "title" in jobs_df.columns
        else None
    )

    minilm_model = load_minilm_model()
    enable_cross_encoder = env_bool("ENABLE_CROSS_ENCODER", True)
    cross_encoder_model = load_cross_encoder_model() if enable_cross_encoder else None

    results = rank_jobs_multi_stage(
        resume_text=resume_text,
        job_descriptions=job_descriptions,
        job_titles=job_titles,
        job_requirements=job_descriptions,
        minilm_model=minilm_model,
        cross_encoder_model=cross_encoder_model,
        enable_cross_encoder=enable_cross_encoder,
        word_keep_n=100,
        phrase_keep_n=25,
        word_custom_stopwords={"preferred", "required", "qualification", "qualifications"},
        word_use_stopword_filter=True,
        word_use_frequent_word_filter=False,
        word_max_count=100,
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
    )

    print("\n===== FINAL TOP JOBS AFTER MULTI-STAGE RANKING =====\n")
    for row in results:
        job_idx = row["job_index"]
        meta = jobs_df.iloc[job_idx]
        company = str(meta["company_name"]).strip() if "company_name" in jobs_df.columns else ""
        title = str(meta["title"]).strip() if "title" in jobs_df.columns else ""
        location = str(meta["location_name"]).strip() if "location_name" in jobs_df.columns else ""

        print(
            f"final_rank={row['final_rank']:>2} | "
            f"job_index={job_idx:>4} | "
            f"csv_row_index={int(meta['csv_row_index']):>5} | "
            f"company={company} | "
            f"title={title} | "
            f"location={location} | "
            f"stage1_rank={row['stage1_rank']} | "
            f"stage1_dist={row['stage1_word_wasserstein_distance']} | "
            f"stage2_rank={row['stage2_rank']} | "
            f"stage2_dist={row['stage2_phrase_wasserstein_distance']} | "
            f"ce_score={row['cross_encoder_score']}"
        )


if __name__ == "__main__":
    main()
