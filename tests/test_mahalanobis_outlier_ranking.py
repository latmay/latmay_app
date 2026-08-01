from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

from ranking_algorithms.mahalanobis_outlier_ranking import rank_mahalanobis_outliers


def cross_encoder_result(scores_by_job_id: dict[int, float]) -> dict[str, object]:
    return {
        "operation_name": "cross_encoder",
        "status": "ok",
        "ranked_job_ids": sorted(scores_by_job_id, key=scores_by_job_id.get, reverse=True),
        "job_metrics": {
            job_id: {
                "rank": rank,
                "score": score,
                "raw_metrics": {"cross_encoder_score": score},
            }
            for rank, (job_id, score) in enumerate(
                sorted(scores_by_job_id.items(), key=lambda item: item[1], reverse=True),
                start=1,
            )
        },
    }


class MahalanobisOutlierRankingTest(unittest.TestCase):
    def test_good_direction_projection_prefers_good_outlier_over_bad_outlier(self) -> None:
        result = rank_mahalanobis_outliers(
            candidate_job_ids=[0, 1, 2],
            operation_results=[cross_encoder_result({0: 0.0, 1: 10.0, 2: 5.0})],
            scoring_mode="good_direction_projection",
        )

        self.assertEqual(result["ranked_job_ids"], [1, 2, 0])
        metrics = result["job_metrics"]  # type: ignore[index]
        self.assertGreater(
            metrics[1]["raw_metrics"]["mahalanobis_good_direction_score"],  # type: ignore[index]
            metrics[0]["raw_metrics"]["mahalanobis_good_direction_score"],  # type: ignore[index]
        )
        self.assertEqual(metrics[1]["raw_metrics"]["mahalanobis_scoring_mode"], "good_direction_projection")  # type: ignore[index]

    def test_distance_outlier_mode_remains_available(self) -> None:
        result = rank_mahalanobis_outliers(
            candidate_job_ids=[0, 1, 2],
            operation_results=[cross_encoder_result({0: 0.0, 1: 10.0, 2: 5.0})],
            scoring_mode="distance_outlier",
        )

        self.assertEqual(result["ranked_job_ids"][:2], [0, 1])
        metrics = result["job_metrics"]  # type: ignore[index]
        self.assertEqual(metrics[0]["raw_metrics"]["mahalanobis_scoring_mode"], "distance_outlier")  # type: ignore[index]
        self.assertAlmostEqual(
            metrics[0]["raw_metrics"]["mahalanobis_distance"],  # type: ignore[index]
            metrics[1]["raw_metrics"]["mahalanobis_distance"],  # type: ignore[index]
        )

    def test_rejects_unknown_scoring_mode(self) -> None:
        with self.assertRaises(ValueError):
            rank_mahalanobis_outliers(
                candidate_job_ids=[0],
                operation_results=[cross_encoder_result({0: 1.0})],
                scoring_mode="surprise_me",
            )


if __name__ == "__main__":
    unittest.main()
