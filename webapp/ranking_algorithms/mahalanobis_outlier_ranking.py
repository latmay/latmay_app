from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


OPERATION_NAME = "mahalanobis_outlier"
LOWER_IS_BETTER = "lower_is_better"
HIGHER_IS_BETTER = "higher_is_better"
SCORING_MODE_DISTANCE_OUTLIER = "distance_outlier"
SCORING_MODE_GOOD_DIRECTION_PROJECTION = "good_direction_projection"
SCORING_MODES = {
    SCORING_MODE_DISTANCE_OUTLIER,
    SCORING_MODE_GOOD_DIRECTION_PROJECTION,
}
FEATURE_DIRECTIONS = {
    "requirements_cluster_distance": -1.0,
    "word_wasserstein_distance": -1.0,
    "phrase_wasserstein_distance": -1.0,
    "cross_encoder_score": 1.0,
}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number):
        return None

    return number


def _raw_metric(
    operation_results_by_name: Mapping[str, Mapping[str, Any]],
    operation_name: str,
    job_id: Any,
    metric_name: str,
) -> float | None:
    result = operation_results_by_name.get(operation_name) or {}
    metric = (result.get("job_metrics") or {}).get(job_id) or {}
    raw_metrics = metric.get("raw_metrics") or {}
    return _safe_float(raw_metrics.get(metric_name))


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


def _get_cross_encoder_result(operation_results: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for result in operation_results:
        if result.get("operation_name") == "cross_encoder" and isinstance(result, dict):
            return result

    return None


def _cross_encoder_score_from_result(
    cross_encoder_result: Mapping[str, Any] | None,
    job_id: Any,
) -> float | None:
    if not cross_encoder_result:
        return None

    metric = (cross_encoder_result.get("job_metrics") or {}).get(job_id) or {}
    raw_metrics = metric.get("raw_metrics") or {}
    return _safe_float(raw_metrics.get("cross_encoder_score"))


def _refresh_cross_encoder_ranking(cross_encoder_result: dict[str, Any]) -> None:
    job_metrics = cross_encoder_result.get("job_metrics") or {}
    score_rows: list[tuple[Any, float]] = []

    for job_id, metric in job_metrics.items():
        raw_metrics = metric.get("raw_metrics") or {}
        score = _safe_float(raw_metrics.get("cross_encoder_score"))
        if score is not None:
            score_rows.append((job_id, score))

    if not score_rows:
        return

    ranks = _rank_with_ties(score_rows, reverse=True)
    ranked_job_ids = [
        job_id
        for job_id, _ in sorted(score_rows, key=lambda row: row[1], reverse=True)
    ]

    for job_id, score in score_rows:
        job_metrics[job_id] = {
            "rank": ranks[job_id],
            "score": score,
            "score_direction": HIGHER_IS_BETTER,
            "raw_metrics": {"cross_encoder_score": score},
        }

    cross_encoder_result["status"] = "ok"
    cross_encoder_result["ranked_job_ids"] = ranked_job_ids
    cross_encoder_result["job_metrics"] = job_metrics
    cross_encoder_result["error"] = None


def _fill_missing_cross_encoder_scores(
    *,
    candidate_job_ids: Sequence[Any],
    operation_results: Sequence[Mapping[str, Any]],
    job_descriptions: Sequence[str] | None,
    resume_text: str | None,
    cross_encoder_model: Any | None,
    chunk_size_words: int,
    chunk_overlap_words: int,
    aggregation: str,
    top_k: int,
    batch_size: int,
) -> None:
    if cross_encoder_model is None or job_descriptions is None or resume_text is None:
        return

    cross_encoder_result = _get_cross_encoder_result(operation_results)
    if cross_encoder_result is None:
        return

    missing_job_ids = [
        job_id
        for job_id in candidate_job_ids
        if _cross_encoder_score_from_result(cross_encoder_result, job_id) is None
    ]
    if not missing_job_ids:
        return

    from ranking_algorithms.cross_encoder_scoring import compare_texts_cross_encoder

    job_metrics = cross_encoder_result.get("job_metrics") or {}
    for job_id in missing_job_ids:
        job_index = int(job_id)
        score = compare_texts_cross_encoder(
            text_a=resume_text,
            text_b=job_descriptions[job_index],
            model=cross_encoder_model,
            chunk_size_words=chunk_size_words,
            chunk_overlap_words=chunk_overlap_words,
            aggregation=aggregation,
            top_k=top_k,
            batch_size=batch_size,
        )
        job_metrics[job_id] = {
            "rank": None,
            "score": float(score),
            "score_direction": HIGHER_IS_BETTER,
            "raw_metrics": {"cross_encoder_score": float(score)},
        }

    cross_encoder_result["job_metrics"] = job_metrics
    _refresh_cross_encoder_ranking(cross_encoder_result)


def _top_ranked_job_ids(
    operation_results_by_name: Mapping[str, Mapping[str, Any]],
    operation_name: str,
    *,
    top_k: int,
) -> list[Any]:
    result = operation_results_by_name.get(operation_name) or {}
    if result.get("status") != "ok":
        return []

    return list(result.get("ranked_job_ids") or [])[:top_k]


def build_mahalanobis_candidate_job_ids(
    operation_results: Sequence[Mapping[str, Any]],
    *,
    top_k_per_ranker: int = 10,
) -> list[Any]:
    """
    Build a stable union of the strongest jobs from the ranking operations.
    """
    if top_k_per_ranker <= 0:
        raise ValueError("top_k_per_ranker must be positive.")

    result_by_name = {
        str(result.get("operation_name")): result
        for result in operation_results
        if result.get("status") == "ok"
    }
    operation_names = [
        "word_sliced_wasserstein",
        "phrase_sliced_wasserstein",
        "cross_encoder",
    ]
    selected: list[Any] = []
    seen: set[Any] = set()

    for operation_name in operation_names:
        for job_id in _top_ranked_job_ids(result_by_name, operation_name, top_k=top_k_per_ranker):
            if job_id not in seen:
                selected.append(job_id)
                seen.add(job_id)

    return selected


def _candidate_features(
    *,
    candidate_job_ids: Sequence[Any],
    operation_results: Sequence[Mapping[str, Any]],
    requirements_cluster_distances: Sequence[float | None] | None,
) -> tuple[np.ndarray, list[str], list[dict[str, float | None]]]:
    result_by_name = {
        str(result.get("operation_name")): result
        for result in operation_results
        if result.get("status") == "ok"
    }

    raw_rows: list[dict[str, float | None]] = []
    for job_id in candidate_job_ids:
        job_index = int(job_id)
        requirements_distance = None
        if requirements_cluster_distances is not None and job_index < len(requirements_cluster_distances):
            requirements_distance = _safe_float(requirements_cluster_distances[job_index])

        raw_rows.append(
            {
                "requirements_cluster_distance": requirements_distance,
                "word_wasserstein_distance": _raw_metric(
                    result_by_name,
                    "word_sliced_wasserstein",
                    job_id,
                    "word_wasserstein_distance",
                ),
                "phrase_wasserstein_distance": _raw_metric(
                    result_by_name,
                    "phrase_sliced_wasserstein",
                    job_id,
                    "phrase_wasserstein_distance",
                ),
                "cross_encoder_score": _raw_metric(
                    result_by_name,
                    "cross_encoder",
                    job_id,
                    "cross_encoder_score",
                ),
            }
        )

    feature_names = [
        name
        for name in [
            "requirements_cluster_distance",
            "word_wasserstein_distance",
            "phrase_wasserstein_distance",
            "cross_encoder_score",
        ]
        if any(row[name] is not None for row in raw_rows)
    ]

    if not feature_names:
        raise ValueError("No Mahalanobis feature coordinates are available.")

    matrix = np.empty((len(raw_rows), len(feature_names)), dtype=np.float64)
    for col_idx, feature_name in enumerate(feature_names):
        values = np.array(
            [
                np.nan if row[feature_name] is None else float(row[feature_name])
                for row in raw_rows
            ],
            dtype=np.float64,
        )
        values = values * FEATURE_DIRECTIONS[feature_name]
        mean_value = float(np.nanmean(values))
        values = np.where(np.isnan(values), mean_value, values)
        std_value = float(np.std(values))
        if std_value == 0.0:
            matrix[:, col_idx] = 0.0
        else:
            matrix[:, col_idx] = (values - mean_value) / std_value

    return matrix, feature_names, raw_rows


def rank_mahalanobis_outliers(
    *,
    candidate_job_ids: Sequence[Any],
    operation_results: Sequence[Mapping[str, Any]],
    requirements_cluster_distances: Sequence[float | None] | None = None,
    job_descriptions: Sequence[str] | None = None,
    resume_text: str | None = None,
    cross_encoder_model: Any | None = None,
    ce_chunk_size_words: int = 180,
    ce_chunk_overlap_words: int = 40,
    ce_aggregation: str = "top_k_mean",
    ce_top_k: int = 3,
    ce_batch_size: int = 32,
    scoring_mode: str = SCORING_MODE_DISTANCE_OUTLIER,
) -> dict[str, Any]:
    """
    Rank candidates using either symmetric outlier distance or a signed good-direction score.
    """
    candidate_job_ids = list(candidate_job_ids)
    if not candidate_job_ids:
        raise ValueError("candidate_job_ids cannot be empty.")
    scoring_mode = scoring_mode.strip().lower()
    if scoring_mode not in SCORING_MODES:
        raise ValueError(f"Unsupported Mahalanobis scoring mode: {scoring_mode!r}.")

    _fill_missing_cross_encoder_scores(
        candidate_job_ids=candidate_job_ids,
        operation_results=operation_results,
        job_descriptions=job_descriptions,
        resume_text=resume_text,
        cross_encoder_model=cross_encoder_model,
        chunk_size_words=ce_chunk_size_words,
        chunk_overlap_words=ce_chunk_overlap_words,
        aggregation=ce_aggregation,
        top_k=ce_top_k,
        batch_size=ce_batch_size,
    )

    features, feature_names, raw_feature_rows = _candidate_features(
        candidate_job_ids=candidate_job_ids,
        operation_results=operation_results,
        requirements_cluster_distances=requirements_cluster_distances,
    )

    center = features.mean(axis=0)
    centered = features - center
    if len(candidate_job_ids) <= 1:
        distances = np.zeros(len(candidate_job_ids), dtype=np.float64)
    else:
        covariance = np.cov(features, rowvar=False)
        covariance = np.atleast_2d(covariance)
        inverse_covariance = np.linalg.pinv(covariance)
        squared_distances = np.einsum("ij,jk,ik->i", centered, inverse_covariance, centered)
        distances = np.sqrt(np.maximum(squared_distances, 0.0))

    good_direction_scores = features.mean(axis=1)
    if scoring_mode == SCORING_MODE_GOOD_DIRECTION_PROJECTION:
        scores = good_direction_scores
    else:
        scores = distances

    ranked_pairs = sorted(
        zip(candidate_job_ids, scores.tolist(), strict=True),
        key=lambda row: row[1],
        reverse=True,
    )
    ranked_job_ids = [job_id for job_id, _ in ranked_pairs]
    job_metrics: dict[Any, dict[str, Any]] = {}
    raw_features_by_job_id = dict(zip(candidate_job_ids, raw_feature_rows, strict=True))
    distances_by_job_id = dict(zip(candidate_job_ids, distances.tolist(), strict=True))
    good_direction_scores_by_job_id = dict(zip(candidate_job_ids, good_direction_scores.tolist(), strict=True))

    for rank, (job_id, score) in enumerate(ranked_pairs, start=1):
        raw_features = raw_features_by_job_id[job_id]
        job_metrics[job_id] = {
            "rank": rank,
            "score": float(score),
            "score_direction": HIGHER_IS_BETTER,
            "raw_metrics": {
                "mahalanobis_distance": float(distances_by_job_id[job_id]),
                "mahalanobis_good_direction_score": float(good_direction_scores_by_job_id[job_id]),
                "mahalanobis_scoring_mode": scoring_mode,
                "mahalanobis_feature_names": feature_names,
                "mahalanobis_raw_features": raw_features,
            },
        }

    return {
        "operation_name": OPERATION_NAME,
        "status": "ok",
        "ranked_job_ids": ranked_job_ids,
        "job_metrics": job_metrics,
        "error": None,
    }
