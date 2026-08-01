from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ranking_algorithms.mahalanobis_outlier_ranking import FEATURE_DIRECTIONS


OPERATION_NAME = "multi_metric_bad_fit_filter"
HIGHER_IS_BETTER = "higher_is_better"


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
    result_by_name: Mapping[str, Mapping[str, Any]],
    operation_name: str,
    job_id: Any,
    metric_name: str,
) -> float | None:
    result = result_by_name.get(operation_name) or {}
    metric = (result.get("job_metrics") or {}).get(job_id) or {}
    raw_metrics = metric.get("raw_metrics") or {}
    return _safe_float(raw_metrics.get(metric_name))


def _metric_values_for_job(
    *,
    job_id: Any,
    operation_results_by_name: Mapping[str, Mapping[str, Any]],
    requirements_cluster_distances: Sequence[float | None] | None,
) -> dict[str, float | None]:
    job_index = int(job_id)
    requirements_distance = None
    if requirements_cluster_distances is not None and job_index < len(requirements_cluster_distances):
        requirements_distance = _safe_float(requirements_cluster_distances[job_index])

    return {
        "requirements_cluster_distance": requirements_distance,
        "word_wasserstein_distance": _raw_metric(
            operation_results_by_name,
            "word_sliced_wasserstein",
            job_id,
            "word_wasserstein_distance",
        ),
        "phrase_wasserstein_distance": _raw_metric(
            operation_results_by_name,
            "phrase_sliced_wasserstein",
            job_id,
            "phrase_wasserstein_distance",
        ),
        "cross_encoder_score": _raw_metric(
            operation_results_by_name,
            "cross_encoder",
            job_id,
            "cross_encoder_score",
        ),
    }


def run_multi_metric_bad_fit_filter(
    *,
    job_ids: Sequence[Any],
    operation_results: Sequence[Mapping[str, Any]],
    requirements_cluster_distances: Sequence[float | None] | None = None,
    bottom_fraction: float = 0.25,
    min_bottom_metric_count: int = 2,
) -> dict[str, Any]:
    if not 0 <= bottom_fraction < 1:
        raise ValueError("bottom_fraction must be in [0, 1).")
    if min_bottom_metric_count <= 0:
        raise ValueError("min_bottom_metric_count must be positive.")

    job_ids = list(job_ids)
    result_by_name = {
        str(result.get("operation_name")): result
        for result in operation_results
        if result.get("status") == "ok"
    }
    metric_values_by_job_id = {
        job_id: _metric_values_for_job(
            job_id=job_id,
            operation_results_by_name=result_by_name,
            requirements_cluster_distances=requirements_cluster_distances,
        )
        for job_id in job_ids
    }

    bottom_metrics_by_job_id: dict[Any, set[str]] = {job_id: set() for job_id in job_ids}
    for metric_name, direction in FEATURE_DIRECTIONS.items():
        available_rows = [
            (job_id, float(value) * direction)
            for job_id, values in metric_values_by_job_id.items()
            if (value := values.get(metric_name)) is not None
        ]
        if not available_rows or bottom_fraction == 0:
            continue

        bottom_count = max(1, int(len(available_rows) * bottom_fraction))
        bottom_rows = sorted(available_rows, key=lambda row: row[1])[:bottom_count]
        for job_id, _ in bottom_rows:
            bottom_metrics_by_job_id[job_id].add(metric_name)

    job_metrics: dict[Any, dict[str, Any]] = {}
    kept_job_ids: list[Any] = []
    removed_job_ids: list[Any] = []
    for job_id in job_ids:
        bottom_metrics = sorted(bottom_metrics_by_job_id[job_id])
        available_metrics = sorted(
            metric_name
            for metric_name, value in metric_values_by_job_id[job_id].items()
            if value is not None
        )
        is_bad_fit = len(bottom_metrics) >= min_bottom_metric_count
        score = 0.0 if is_bad_fit else 1.0
        if is_bad_fit:
            removed_job_ids.append(job_id)
        else:
            kept_job_ids.append(job_id)

        job_metrics[job_id] = {
            "rank": 2 if is_bad_fit else 1,
            "score": score,
            "score_direction": HIGHER_IS_BETTER,
            "raw_metrics": {
                "is_multi_metric_bad_fit": is_bad_fit,
                "bottom_metric_count": len(bottom_metrics),
                "bottom_metrics": bottom_metrics,
                "available_metrics": available_metrics,
                "metric_values": metric_values_by_job_id[job_id],
            },
        }

    return {
        "operation_name": OPERATION_NAME,
        "status": "ok",
        "ranked_job_ids": kept_job_ids + removed_job_ids,
        "job_metrics": job_metrics,
        "error": None,
    }
