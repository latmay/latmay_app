from __future__ import annotations

from typing import Any, Mapping, Sequence


def _policy_count(
    *,
    total: int,
    top_n: int | None = None,
    top_fraction: float | None = None,
) -> int:
    if total <= 0:
        return 0

    counts: list[int] = []

    if top_n is not None:
        if top_n <= 0:
            raise ValueError("top_n must be positive when set.")
        counts.append(top_n)

    if top_fraction is not None:
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1] when set.")
        counts.append(max(1, int(total * top_fraction)))

    if not counts:
        return total

    return min(total, min(counts))


def _reduce_by_cluster_fraction(
    *,
    current_job_ids: Sequence[Any],
    operation_result: Mapping[str, Any],
    top_cluster_fraction: float,
) -> list[Any]:
    if not 0 < top_cluster_fraction <= 1:
        raise ValueError("top_cluster_fraction must be in (0, 1] when set.")

    ranked_job_ids = operation_result.get("ranked_job_ids", [])
    job_metrics = operation_result.get("job_metrics", {})
    cluster_rows: list[tuple[Any, int, Any]] = []

    for job_id in ranked_job_ids:
        metric = job_metrics.get(job_id) or {}
        raw_metrics = metric.get("raw_metrics") or {}
        cluster_label = raw_metrics.get("cluster_label")
        if cluster_label is None:
            continue
        cluster_rows.append((cluster_label, int(metric.get("rank", 0)), job_id))

    if not cluster_rows:
        print("ALERT: Cluster reduction found no cluster labels; passing jobs through unchanged.")
        return list(current_job_ids)

    cluster_rank_by_label: dict[Any, int] = {}
    for cluster_label, rank, _ in cluster_rows:
        if cluster_label not in cluster_rank_by_label:
            cluster_rank_by_label[cluster_label] = rank
        else:
            cluster_rank_by_label[cluster_label] = min(cluster_rank_by_label[cluster_label], rank)

    ranked_cluster_labels = [
        label
        for label, _ in sorted(cluster_rank_by_label.items(), key=lambda row: row[1])
    ]
    n_keep_clusters = max(1, int(len(ranked_cluster_labels) * top_cluster_fraction))
    kept_labels = set(ranked_cluster_labels[:n_keep_clusters])
    current_set = set(current_job_ids)

    return [
        job_id
        for job_id in ranked_job_ids
        if job_id in current_set and (job_metrics.get(job_id) or {}).get("raw_metrics", {}).get("cluster_label") in kept_labels
    ]


def _nonnegative_int(value: Any, *, field_name: str) -> int:
    parsed = int(value or 0)
    if parsed < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return parsed


def _enforce_min_remaining(
    *,
    current_job_ids: Sequence[Any],
    proposed_job_ids: Sequence[Any],
    ranked_job_ids: Sequence[Any],
    min_remaining_jobs: int,
    operation_name: str,
) -> list[Any]:
    current_ids = list(current_job_ids)
    proposed_ids = list(proposed_job_ids)
    if min_remaining_jobs <= 0:
        return proposed_ids

    floor = min(min_remaining_jobs, len(current_ids))
    if len(proposed_ids) >= floor:
        return proposed_ids

    current_set = set(current_ids)
    expanded = list(proposed_ids)
    seen = set(expanded)
    for job_id in list(ranked_job_ids) + current_ids:
        if job_id in current_set and job_id not in seen:
            expanded.append(job_id)
            seen.add(job_id)
        if len(expanded) >= floor:
            break

    print(
        "Reduction min-remaining guard applied: "
        f"operation={operation_name or 'unknown operation'}, "
        f"requested_min={min_remaining_jobs}, floor={floor}, "
        f"proposed_jobs={len(proposed_ids)}, guarded_jobs={len(expanded)}",
        flush=True,
    )
    return expanded


def reduce_job_ids(
    *,
    current_job_ids: Sequence[Any],
    operation_result: Mapping[str, Any],
    reduction_policies: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Any]:
    """
    Decide which stable job IDs continue after a ranking operation.

    Failed or skipped operations pass the current job IDs through unchanged.
    Operations without an explicit policy also pass all current jobs through.
    """
    current_ids = list(current_job_ids)
    operation_name = str(operation_result.get("operation_name", ""))
    status = operation_result.get("status")

    if status != "ok":
        print(
            f"ALERT: Reduction skipped for {operation_name or 'unknown operation'} "
            f"because status={status!r}; passing {len(current_ids)} jobs through."
        )
        return current_ids

    policy = (reduction_policies or {}).get(operation_name)
    if not policy:
        return current_ids

    min_remaining_jobs = _nonnegative_int(policy.get("min_remaining_jobs", 0), field_name="min_remaining_jobs")

    if "top_cluster_fraction" in policy:
        reduced_ids = _reduce_by_cluster_fraction(
            current_job_ids=current_ids,
            operation_result=operation_result,
            top_cluster_fraction=float(policy["top_cluster_fraction"]),
        )
        return _enforce_min_remaining(
            current_job_ids=current_ids,
            proposed_job_ids=reduced_ids,
            ranked_job_ids=operation_result.get("ranked_job_ids", []),
            min_remaining_jobs=min_remaining_jobs,
            operation_name=operation_name,
        )

    current_set = set(current_ids)
    ranked_job_ids = [
        job_id
        for job_id in operation_result.get("ranked_job_ids", [])
        if job_id in current_set
    ]

    if "filter_raw_metric" in policy:
        metric_name = str(policy["filter_raw_metric"])
        exclude_value = policy.get("exclude_value", True)
        job_metrics = operation_result.get("job_metrics", {})
        kept_job_ids = [
            job_id
            for job_id in ranked_job_ids
            if ((job_metrics.get(job_id) or {}).get("raw_metrics") or {}).get(metric_name) != exclude_value
        ]
        return _enforce_min_remaining(
            current_job_ids=current_ids,
            proposed_job_ids=kept_job_ids,
            ranked_job_ids=ranked_job_ids,
            min_remaining_jobs=min_remaining_jobs,
            operation_name=operation_name,
        )

    if "keep_score_equals" in policy:
        expected_score = float(policy["keep_score_equals"])
        job_metrics = operation_result.get("job_metrics", {})
        kept_job_ids = [
            job_id
            for job_id in ranked_job_ids
            if float((job_metrics.get(job_id) or {}).get("score", float("nan"))) == expected_score
        ]
        return _enforce_min_remaining(
            current_job_ids=current_ids,
            proposed_job_ids=kept_job_ids,
            ranked_job_ids=ranked_job_ids,
            min_remaining_jobs=min_remaining_jobs,
            operation_name=operation_name,
        )

    if not ranked_job_ids:
        print(
            f"ALERT: Reduction for {operation_name} found no ranked jobs; "
            f"passing {len(current_ids)} jobs through."
        )
        return current_ids

    keep_count = _policy_count(
        total=len(ranked_job_ids),
        top_n=policy.get("top_n"),
        top_fraction=policy.get("top_fraction"),
    )
    return _enforce_min_remaining(
        current_job_ids=current_ids,
        proposed_job_ids=ranked_job_ids[:keep_count],
        ranked_job_ids=ranked_job_ids,
        min_remaining_jobs=min_remaining_jobs,
        operation_name=operation_name,
    )


def union_top_ranked_job_ids(
    *,
    current_job_ids: Sequence[Any],
    operation_results: Sequence[Mapping[str, Any]],
    operation_names: Sequence[str],
    top_k_per_ranker: int,
) -> list[Any]:
    """Build a stable-ID union from the top K successful rankings in policy order."""
    if top_k_per_ranker <= 0:
        raise ValueError("top_k_per_ranker must be positive.")

    current_ids = list(current_job_ids)
    current_set = set(current_ids)
    result_by_name = {
        str(result.get("operation_name")): result
        for result in operation_results
        if result.get("status") == "ok"
    }
    selected: list[Any] = []
    seen: set[Any] = set()

    for operation_name in operation_names:
        result = result_by_name.get(operation_name)
        if not result:
            continue

        ranked_ids = [
            job_id
            for job_id in result.get("ranked_job_ids", [])
            if job_id in current_set
        ][:top_k_per_ranker]
        for job_id in ranked_ids:
            if job_id not in seen:
                selected.append(job_id)
                seen.add(job_id)

    if not selected:
        print("ALERT: Union policy found no successful rankings; passing jobs through unchanged.")
        return current_ids

    return selected
