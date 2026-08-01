"""Final semantic bad-match filter using only pair-local and fixed-reference features."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


OPERATION_NAME = "pair_independent_bad_match_forest"
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "rf_depth7_three_class_hill_climbed.joblib"


SOURCE_FEATURES = {
    "word_distance": ("word_sliced_wasserstein", "word_wasserstein_distance"),
    "phrase_distance": ("phrase_sliced_wasserstein", "phrase_wasserstein_distance"),
    "coverage_percent_flagged": ("resume_phrase_job_coverage", "percent_flagged"),
    "coverage_mean_flagged_distance": ("resume_phrase_job_coverage", "mean_flagged_distance"),
    "coverage_mean_flagged_percentile": ("resume_phrase_job_coverage", "mean_flagged_percentile"),
    "coverage_bad_match_percent": ("resume_phrase_job_coverage", "bad_match_percent"),
    "coverage_flagged_resume_phrases": ("resume_phrase_job_coverage", "flagged_resume_phrases"),
    "coverage_bad_match_resume_phrases": ("resume_phrase_job_coverage", "bad_match_resume_phrases"),
    "coverage_total_resume_phrases": ("resume_phrase_job_coverage", "total_resume_phrases"),
    "resume_seniority_score": ("seniority_filter", "resume_seniority_score"),
    "resume_seniority_std": ("seniority_filter", "resume_seniority_std"),
    "job_seniority_score": ("seniority_filter", "job_seniority_score"),
    "job_raw_seniority_score": ("seniority_filter", "job_raw_seniority_score"),
    "job_seniority_std": ("seniority_filter", "job_seniority_std"),
    "seniority_gap": ("seniority_filter", "seniority_gap"),
    "seniority_abs_gap": ("seniority_filter", "seniority_abs_gap"),
    "job_title_seniority_floor": ("seniority_filter", "job_title_seniority_floor"),
    "job_yoe_seniority_floor": ("seniority_filter", "job_yoe_seniority_floor"),
    "job_title_floor_applied": ("seniority_filter", "job_title_seniority_floor_applied"),
    "job_title_ceiling_applied": ("seniority_filter", "job_title_seniority_ceiling_applied"),
    "job_yoe_floor_applied": ("seniority_filter", "job_yoe_seniority_floor_applied"),
    "is_too_senior": ("seniority_filter", "is_too_senior"),
    "is_too_junior": ("seniority_filter", "is_too_junior"),
    "technology_overlap": ("technology_mismatch_filter", "technology_overlap_score"),
    "technology_job_type_count": ("technology_mismatch_filter", "job_type_count"),
    "technology_overlap_type_count": ("technology_mismatch_filter", "overlap_type_count"),
    "technology_would_remove": ("technology_mismatch_filter", "technology_filter_would_remove"),
}

VECTOR_FEATURES = {
    "resume_level_probability": ("seniority_filter", "resume_level_probability_distribution"),
    "job_level_probability": ("seniority_filter", "job_level_probability_distribution"),
    "resume_level_similarity": ("seniority_filter", "resume_level_similarities"),
    "job_level_similarity": ("seniority_filter", "job_level_similarities"),
}


def _model_path() -> Path:
    cache_dir = Path(os.environ.get("MODEL_CACHE_DIR", BASE_DIR / "model_cache"))
    return cache_dir / os.environ.get("PAIR_INDEPENDENT_BAD_MATCH_MODEL", DEFAULT_MODEL_NAME)


@lru_cache(maxsize=1)
def load_model_artifact() -> dict[str, Any]:
    artifact = joblib.load(_model_path())
    if not isinstance(artifact, dict) or not {"model", "features"} <= artifact.keys():
        raise ValueError("Pair-independent bad-match artifact has an invalid schema.")
    if "bad_threshold" not in artifact and "threshold" not in artifact:
        raise ValueError("Pair-independent bad-match artifact has an invalid schema.")
    return artifact


def _bad_probabilities_and_threshold(
    artifact: Mapping[str, Any], probability_matrix: np.ndarray
) -> tuple[np.ndarray, float, list[str] | None]:
    if "bad_threshold" not in artifact:
        if probability_matrix.shape[1] < 2:
            raise ValueError("Binary bad-match model did not return two probability columns.")
        return probability_matrix[:, 1], float(artifact["threshold"]), None

    model = artifact["model"]
    estimator = getattr(model, "named_steps", {}).get("model")
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError("Three-class bad-match model does not expose fitted classes.")
    class_names = [str(label) for label in classes]
    try:
        bad_index = class_names.index("really_bad")
    except ValueError as exc:
        raise ValueError("Three-class bad-match model has no 'really_bad' class.") from exc
    if probability_matrix.shape[1] != len(class_names):
        raise ValueError("Three-class probability columns do not match fitted classes.")
    return probability_matrix[:, bad_index], float(artifact["bad_threshold"]), class_names


def _result_by_name(operation_results: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(result.get("operation_name")): result
        for result in operation_results
        if result.get("status") == "ok"
    }


def _raw_metric(
    results: Mapping[str, Mapping[str, Any]], operation_name: str, job_id: Any, metric_name: str
) -> Any:
    operation = results.get(operation_name) or {}
    metric = (operation.get("job_metrics") or {}).get(job_id) or {}
    return (metric.get("raw_metrics") or {}).get(metric_name)


def _number(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    try:
        parsed = float(value)
        return parsed if np.isfinite(parsed) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _vector(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple, np.ndarray)):
        return []
    return [_number(item) for item in value]


def build_feature_frame(
    *,
    job_ids: Sequence[Any],
    operation_results: Sequence[Mapping[str, Any]],
    requirements_cluster_distances: Sequence[float | None] | None,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    results = _result_by_name(operation_results)
    rows: list[dict[str, float]] = []
    for job_id in job_ids:
        row: dict[str, float] = {}
        for name, (operation_name, metric_name) in SOURCE_FEATURES.items():
            row[name] = _number(_raw_metric(results, operation_name, job_id, metric_name))

        job_index = int(job_id)
        requirements_distance = None
        if requirements_cluster_distances is not None and 0 <= job_index < len(requirements_cluster_distances):
            requirements_distance = requirements_cluster_distances[job_index]
        row["requirements_embedding_distance"] = _number(requirements_distance)

        for prefix, (operation_name, metric_name) in VECTOR_FEATURES.items():
            values = _vector(_raw_metric(results, operation_name, job_id, metric_name))
            for index in range(6):
                row[f"{prefix}_{index}"] = values[index] if index < len(values) else np.nan
            if "probability" in prefix:
                finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
                row[f"{prefix}_max"] = float(finite.max()) if finite.size else np.nan
                row[f"{prefix}_entropy"] = (
                    float(-np.sum(finite * np.log(np.maximum(finite, 1e-12))))
                    if finite.size
                    else np.nan
                )

        word = row["word_distance"]
        phrase = row["phrase_distance"]
        requirements = row["requirements_embedding_distance"]
        row["semantic_distance_geomean"] = (
            float(np.sqrt(max(word, 0.0) * max(phrase, 0.0)))
            if np.isfinite(word) and np.isfinite(phrase)
            else np.nan
        )
        row["semantic_distance_ratio_log"] = (
            float(np.log((word + 1e-9) / (phrase + 1e-9)))
            if np.isfinite(word) and np.isfinite(phrase) and word + 1e-9 > 0 and phrase + 1e-9 > 0
            else np.nan
        )
        semantic_values = np.asarray([word, phrase, requirements], dtype=float)
        finite_semantic = semantic_values[np.isfinite(semantic_values)]
        row["semantic_requirements_mean"] = float(finite_semantic.mean()) if finite_semantic.size else np.nan
        row["semantic_requirements_max"] = float(finite_semantic.max()) if finite_semantic.size else np.nan
        seniority_abs_gap = row["seniority_abs_gap"]
        row["seniority_fit_exp"] = (
            float(np.exp(-max(seniority_abs_gap, 0.0))) if np.isfinite(seniority_abs_gap) else np.nan
        )
        technology = row["technology_overlap"]
        geomean = row["semantic_distance_geomean"]
        row["technology_x_semantic_similarity"] = (
            float(technology * np.exp(-max(geomean, 0.0)))
            if np.isfinite(technology) and np.isfinite(geomean)
            else np.nan
        )
        row["requirements_x_seniority_badness"] = (
            float(requirements * (1.0 + seniority_abs_gap))
            if np.isfinite(requirements) and np.isfinite(seniority_abs_gap)
            else np.nan
        )
        coverage_values = np.asarray(
            [row["coverage_percent_flagged"], row["coverage_bad_match_percent"]], dtype=float
        )
        finite_coverage = coverage_values[np.isfinite(coverage_values)]
        row["coverage_badness_mean"] = float(finite_coverage.mean()) if finite_coverage.size else np.nan
        rows.append(row)

    return pd.DataFrame(rows, index=list(job_ids)).reindex(columns=list(feature_names))


def run_pair_independent_bad_match_filter(
    *,
    job_ids: Sequence[Any],
    operation_results: Sequence[Mapping[str, Any]],
    requirements_cluster_distances: Sequence[float | None] | None,
) -> dict[str, Any]:
    candidate_ids = list(job_ids)
    if not candidate_ids:
        return {
            "operation_name": OPERATION_NAME,
            "status": "ok",
            "ranked_job_ids": [],
            "job_metrics": {},
            "error": None,
        }
    try:
        artifact = load_model_artifact()
        features = list(artifact["features"])
        frame = build_feature_frame(
            job_ids=candidate_ids,
            operation_results=operation_results,
            requirements_cluster_distances=requirements_cluster_distances,
            feature_names=features,
        )
        probability_matrix = artifact["model"].predict_proba(frame)
        probabilities, threshold, class_names = _bad_probabilities_and_threshold(
            artifact, probability_matrix
        )
        kept_ids: list[Any] = []
        removed_ids: list[Any] = []
        job_metrics: dict[Any, dict[str, Any]] = {}
        for job_id, probability, class_probability_row, missing_count in zip(
            candidate_ids,
            probabilities,
            probability_matrix,
            frame.isna().sum(axis=1).tolist(),
            strict=True,
        ):
            flagged = bool(probability >= threshold)
            (removed_ids if flagged else kept_ids).append(job_id)
            job_metrics[job_id] = {
                "rank": 2 if flagged else 1,
                "score": float(1.0 - probability),
                "score_direction": "higher_is_better",
                "raw_metrics": {
                    "semantic_bad_match_probability": float(probability),
                    "semantic_bad_match_threshold": threshold,
                    "semantic_bad_match_flagged": flagged,
                    "semantic_class_probabilities": (
                        {
                            class_name: float(class_probability)
                            for class_name, class_probability in zip(
                                class_names, class_probability_row, strict=True
                            )
                        }
                        if class_names is not None
                        else None
                    ),
                    "missing_feature_count": int(missing_count),
                },
            }
        return {
            "operation_name": OPERATION_NAME,
            "status": "ok",
            "ranked_job_ids": kept_ids + removed_ids,
            "job_metrics": job_metrics,
            "error": None,
        }
    except Exception as exc:
        print(
            f"ALERT: {OPERATION_NAME} failed open: error_type={type(exc).__name__}",
            flush=True,
        )
        return {
            "operation_name": OPERATION_NAME,
            "status": "failed",
            "ranked_job_ids": candidate_ids,
            "job_metrics": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
