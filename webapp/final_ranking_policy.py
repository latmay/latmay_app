from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


FINAL_RANKING_MODE_MINIMAX = "minimax"
FINAL_RANKING_MODE_RECIPROCAL_RANK_FUSION = "reciprocal_rank_fusion"
FINAL_RANKING_MODES = {
    FINAL_RANKING_MODE_MINIMAX,
    FINAL_RANKING_MODE_RECIPROCAL_RANK_FUSION,
}
FINAL_ORDER_OPERATION_BY_MODE = {
    FINAL_RANKING_MODE_MINIMAX: "minimax_rank",
    FINAL_RANKING_MODE_RECIPROCAL_RANK_FUSION: "reciprocal_rank_fusion",
}
RRF_K = 60.0
IGNORED_FINAL_RANKING_OPERATIONS = {
    "llm_bad_match_filter",
    "mahalanobis_outlier",
    "multi_metric_bad_fit_filter",
    "pair_independent_bad_match_forest",
    "seniority_filter",
    "technology_mismatch_filter",
}


def _usable_rank(value: Any) -> float | None:
    try:
        rank = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rank):
        return None
    return rank


def _job_id_tie_breaker(job_id: Any) -> tuple[int, int | str]:
    try:
        return (0, int(job_id))
    except (TypeError, ValueError):
        return (1, str(job_id))


def _dense_normalized_ranks(operation_ranks: Mapping[Any, float]) -> dict[Any, int]:
    normalized_ranks: dict[Any, int] = {}
    previous_rank: float | None = None
    normalized_rank = 0

    for job_id, rank in sorted(operation_ranks.items(), key=lambda item: (item[1], _job_id_tie_breaker(item[0]))):
        if previous_rank is None or rank != previous_rank:
            normalized_rank += 1
            previous_rank = rank
        normalized_ranks[job_id] = normalized_rank

    return normalized_ranks


def _normalized_ranks_by_job_id(
    *,
    remaining_ids: Sequence[Any],
    operation_results: Sequence[Mapping[str, Any]],
) -> dict[Any, list[float]]:
    ranks_by_job_id: dict[Any, list[float]] = {job_id: [] for job_id in remaining_ids}

    for result in operation_results:
        if result.get("status") != "ok":
            continue
        operation_name = str(result.get("operation_name") or "")
        if operation_name in IGNORED_FINAL_RANKING_OPERATIONS:
            continue

        job_metrics = result.get("job_metrics") or {}
        operation_ranks: dict[Any, float] = {}
        for job_id in remaining_ids:
            metric = job_metrics.get(job_id) or {}
            rank = _usable_rank(metric.get("rank"))
            if rank is None:
                continue
            operation_ranks[job_id] = rank

        if not operation_ranks:
            continue

        for job_id, rank in _dense_normalized_ranks(operation_ranks).items():
            ranks_by_job_id[job_id].append(rank)

    return ranks_by_job_id


def _choose_minimax_order(
    *,
    remaining_ids: Sequence[Any],
    ranks_by_job_id: Mapping[Any, Sequence[float]],
) -> list[Any]:
    return sorted(
        remaining_ids,
        key=lambda job_id: (
            not ranks_by_job_id[job_id],
            sorted(ranks_by_job_id[job_id], reverse=True),
            _job_id_tie_breaker(job_id),
        ),
    )


def _choose_reciprocal_rank_fusion_order(
    *,
    remaining_ids: Sequence[Any],
    ranks_by_job_id: Mapping[Any, Sequence[float]],
) -> list[Any]:
    def rrf_score(job_id: Any) -> float:
        return sum(1.0 / (RRF_K + rank) for rank in ranks_by_job_id[job_id])

    return sorted(
        remaining_ids,
        key=lambda job_id: (
            not ranks_by_job_id[job_id],
            -rrf_score(job_id),
            sorted(ranks_by_job_id[job_id], reverse=True),
            _job_id_tie_breaker(job_id),
        ),
    )


def choose_final_job_order(
    *,
    remaining_job_ids: Sequence[Any],
    operation_results: Sequence[Mapping[str, Any]],
    ranking_mode: str = FINAL_RANKING_MODE_MINIMAX,
) -> tuple[list[Any], str | None]:
    """
    Final candidates are ordered using normalized prior ranks.
    """
    ranking_mode = ranking_mode.strip().lower()
    if ranking_mode not in FINAL_RANKING_MODES:
        raise ValueError(f"Unsupported final ranking mode: {ranking_mode!r}.")

    remaining_ids = list(remaining_job_ids)
    ranks_by_job_id = _normalized_ranks_by_job_id(
        remaining_ids=remaining_ids,
        operation_results=operation_results,
    )

    if any(ranks for ranks in ranks_by_job_id.values()):
        if ranking_mode == FINAL_RANKING_MODE_RECIPROCAL_RANK_FUSION:
            ranked_ids = _choose_reciprocal_rank_fusion_order(
                remaining_ids=remaining_ids,
                ranks_by_job_id=ranks_by_job_id,
            )
        else:
            ranked_ids = _choose_minimax_order(
                remaining_ids=remaining_ids,
                ranks_by_job_id=ranks_by_job_id,
            )
        return ranked_ids, FINAL_ORDER_OPERATION_BY_MODE[ranking_mode]

    print(
        "ALERT: Final ranking fallback in use: "
        "no usable prior ranks found; returning jobs in stable/current order."
    )
    return remaining_ids, None
