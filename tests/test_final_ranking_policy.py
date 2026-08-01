from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

from final_ranking_policy import choose_final_job_order


def operation_result(
    name: str,
    ranks_by_job_id: dict[int, int | float | str],
    *,
    status: str = "ok",
) -> dict[str, object]:
    return {
        "operation_name": name,
        "status": status,
        "ranked_job_ids": list(ranks_by_job_id),
        "job_metrics": {
            job_id: {
                "rank": rank,
                "score": 1.0,
            }
            for job_id, rank in ranks_by_job_id.items()
        },
        "error": None,
    }


class FinalRankingPolicyTest(unittest.TestCase):
    def test_orders_by_smallest_worst_rank(self) -> None:
        ranked_ids, operation_name = choose_final_job_order(
            remaining_job_ids=[101, 202],
            operation_results=[
                operation_result("word_sliced_wasserstein", {101: 1, 202: 3}),
                operation_result("phrase_sliced_wasserstein", {101: 8, 202: 5}),
                operation_result("cross_encoder", {101: 12, 202: 20}),
            ],
        )

        self.assertEqual(ranked_ids, [101, 202])
        self.assertEqual(operation_name, "minimax_rank")

    def test_uses_second_worst_rank_then_job_id_for_ties(self) -> None:
        ranked_ids, operation_name = choose_final_job_order(
            remaining_job_ids=[30, 20, 10],
            operation_results=[
                operation_result("word_sliced_wasserstein", {30: 1, 20: 2, 10: 2}),
                operation_result("phrase_sliced_wasserstein", {30: 7, 20: 7, 10: 7}),
                operation_result("cross_encoder", {30: 5, 20: 4, 10: 4}),
            ],
        )

        self.assertEqual(ranked_ids, [10, 20, 30])
        self.assertEqual(operation_name, "minimax_rank")

    def test_normalizes_sparse_operation_ranks_before_minimax(self) -> None:
        ranked_ids, operation_name = choose_final_job_order(
            remaining_job_ids=[101, 202, 303],
            operation_results=[
                operation_result("word_sliced_wasserstein", {101: 4, 202: 8, 303: 13}),
                operation_result("phrase_sliced_wasserstein", {303: 1, 202: 2, 101: 3}),
            ],
        )

        self.assertEqual(ranked_ids, [202, 101, 303])
        self.assertEqual(operation_name, "minimax_rank")

    def test_reciprocal_rank_fusion_uses_normalized_ranks(self) -> None:
        operation_results = [
            operation_result("word_sliced_wasserstein", {1: 10, 2: 20, 3: 30, 4: 40}),
            operation_result("phrase_sliced_wasserstein", {1: 10, 2: 20, 3: 30, 4: 40}),
            operation_result("cross_encoder", {3: 10, 2: 20, 4: 30, 1: 40}),
        ]

        minimax_ids, minimax_operation_name = choose_final_job_order(
            remaining_job_ids=[1, 2, 3, 4],
            operation_results=operation_results,
        )
        rrf_ids, rrf_operation_name = choose_final_job_order(
            remaining_job_ids=[1, 2, 3, 4],
            operation_results=operation_results,
            ranking_mode="reciprocal_rank_fusion",
        )

        self.assertEqual(minimax_ids[:2], [2, 3])
        self.assertEqual(minimax_operation_name, "minimax_rank")
        self.assertEqual(rrf_ids[:2], [1, 2])
        self.assertEqual(rrf_operation_name, "reciprocal_rank_fusion")

    def test_rejects_unknown_final_ranking_mode(self) -> None:
        with self.assertRaises(ValueError):
            choose_final_job_order(
                remaining_job_ids=[1],
                operation_results=[operation_result("word_sliced_wasserstein", {1: 1})],
                ranking_mode="surprise_me",
            )

    def test_ignores_filters_failed_results_and_missing_ranks(self) -> None:
        ranked_ids, operation_name = choose_final_job_order(
            remaining_job_ids=[1, 2, 3],
            operation_results=[
                operation_result("technology_mismatch_filter", {1: 99, 2: 1, 3: 1}),
                operation_result("word_sliced_wasserstein", {1: 4, 2: 2}),
                operation_result("phrase_sliced_wasserstein", {1: "bad", 2: 3, 3: 1}),
                operation_result("cross_encoder", {1: 1, 2: 5, 3: 2}, status="failed"),
            ],
        )

        self.assertEqual(ranked_ids, [3, 1, 2])
        self.assertEqual(operation_name, "minimax_rank")

    def test_preserves_stable_order_when_no_usable_ranks_exist(self) -> None:
        ranked_ids, operation_name = choose_final_job_order(
            remaining_job_ids=[3, 1, 2],
            operation_results=[
                operation_result("seniority_filter", {3: 1, 1: 2, 2: 1}),
                operation_result("word_sliced_wasserstein", {}, status="failed"),
            ],
        )

        self.assertEqual(ranked_ids, [3, 1, 2])
        self.assertIsNone(operation_name)


if __name__ == "__main__":
    unittest.main()
