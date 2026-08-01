from __future__ import annotations

import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

from ranking_algorithms import pair_independent_bad_match_filter as subject
from multi_stage_rankings import print_pair_independent_forest_removals


class FakeModel:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities
        self.seen = None

    def predict_proba(self, frame):
        self.seen = frame.copy()
        probabilities = np.asarray(self.probabilities, dtype=float)
        return np.column_stack([1.0 - probabilities, probabilities])


class FakeThreeClassEstimator:
    classes_ = np.asarray(["really_bad", "really_good", "unclear"])


class FakeThreeClassModel:
    def __init__(self, probability_matrix: list[list[float]]) -> None:
        self.probability_matrix = np.asarray(probability_matrix, dtype=float)
        self.named_steps = {"model": FakeThreeClassEstimator()}
        self.seen = None

    def predict_proba(self, frame):
        self.seen = frame.copy()
        return self.probability_matrix


def operation(name: str, metrics_by_job: dict[int, dict[str, object]]) -> dict[str, object]:
    return {
        "operation_name": name,
        "status": "ok",
        "ranked_job_ids": list(metrics_by_job),
        "job_metrics": {
            job_id: {"raw_metrics": raw_metrics}
            for job_id, raw_metrics in metrics_by_job.items()
        },
    }


class PairIndependentBadMatchFilterTest(unittest.TestCase):
    def test_removal_log_includes_count_title_probability_and_threshold(self) -> None:
        result = {
            "job_metrics": {
                1: {"raw_metrics": {
                    "semantic_bad_match_probability": 0.7123456,
                    "semantic_bad_match_threshold": 0.313164,
                }}
            }
        }
        output = StringIO()

        with redirect_stdout(output):
            print_pair_independent_forest_removals(
                [1],
                job_titles=["kept", "Example\nJob Title"],
                operation_result=result,
            )

        logged = output.getvalue()
        self.assertIn("removed_jobs=1", logged)
        self.assertIn("title='Example Job Title'", logged)
        self.assertIn("probability=0.712346", logged)
        self.assertIn("threshold=0.313164", logged)

    def test_builds_pair_local_and_fixed_reference_features(self) -> None:
        operations = [
            operation("word_sliced_wasserstein", {0: {"word_wasserstein_distance": 0.4}}),
            operation("phrase_sliced_wasserstein", {0: {"phrase_wasserstein_distance": 0.9}}),
            operation("resume_phrase_job_coverage", {0: {
                "percent_flagged": 0.2,
                "bad_match_percent": 0.6,
                "mean_flagged_distance": 0.3,
                "mean_flagged_percentile": 8.0,
                "flagged_resume_phrases": 2,
                "bad_match_resume_phrases": 6,
                "total_resume_phrases": 10,
            }}),
            operation("seniority_filter", {0: {
                "seniority_abs_gap": 1.5,
                "resume_level_probability_distribution": [0.1, 0.2, 0.3, 0.2, 0.1, 0.1],
                "job_level_probability_distribution": [0.0, 0.1, 0.2, 0.3, 0.3, 0.1],
                "resume_level_similarities": [1, 2, 3, 4, 5, 6],
                "job_level_similarities": [6, 5, 4, 3, 2, 1],
            }}),
            operation("technology_mismatch_filter", {0: {"technology_overlap_score": 0.5}}),
        ]
        features = [
            "word_distance", "phrase_distance", "requirements_embedding_distance",
            "coverage_badness_mean", "semantic_distance_geomean",
            "resume_level_probability_2", "resume_level_probability_entropy",
        ]

        frame = subject.build_feature_frame(
            job_ids=[0],
            operation_results=operations,
            requirements_cluster_distances=[0.7],
            feature_names=features,
        )

        self.assertEqual(list(frame.columns), features)
        self.assertAlmostEqual(frame.loc[0, "requirements_embedding_distance"], 0.7)
        self.assertAlmostEqual(frame.loc[0, "coverage_badness_mean"], 0.4)
        self.assertAlmostEqual(frame.loc[0, "semantic_distance_geomean"], 0.6)
        self.assertAlmostEqual(frame.loc[0, "resume_level_probability_2"], 0.3)
        self.assertTrue(np.isfinite(frame.loc[0, "resume_level_probability_entropy"]))

    def test_removes_probabilities_at_or_above_artifact_threshold(self) -> None:
        model = FakeModel([0.2, 0.7])
        artifact = {"model": model, "features": ["word_distance"], "threshold": 0.5}
        operations = [operation("word_sliced_wasserstein", {
            0: {"word_wasserstein_distance": 0.1},
            1: {"word_wasserstein_distance": 0.9},
        })]

        with patch.object(subject, "load_model_artifact", return_value=artifact):
            result = subject.run_pair_independent_bad_match_filter(
                job_ids=[0, 1],
                operation_results=operations,
                requirements_cluster_distances=None,
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["ranked_job_ids"], [0, 1])
        self.assertFalse(result["job_metrics"][0]["raw_metrics"]["semantic_bad_match_flagged"])
        self.assertTrue(result["job_metrics"][1]["raw_metrics"]["semantic_bad_match_flagged"])
        self.assertEqual(list(model.seen.columns), ["word_distance"])

    def test_three_class_model_removes_only_by_named_bad_class_and_bad_threshold(self) -> None:
        model = FakeThreeClassModel([
            [0.60, 0.10, 0.30],
            [0.20, 0.70, 0.10],
            [0.40, 0.05, 0.55],
        ])
        artifact = {
            "model": model,
            "features": ["word_distance"],
            "bad_threshold": 0.48125,
            "good_threshold": 0.153125,
        }
        operations = [operation("word_sliced_wasserstein", {
            0: {"word_wasserstein_distance": 0.9},
            1: {"word_wasserstein_distance": 0.1},
            2: {"word_wasserstein_distance": 0.5},
        })]

        with patch.object(subject, "load_model_artifact", return_value=artifact):
            result = subject.run_pair_independent_bad_match_filter(
                job_ids=[0, 1, 2],
                operation_results=operations,
                requirements_cluster_distances=None,
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["ranked_job_ids"], [1, 2, 0])
        self.assertTrue(result["job_metrics"][0]["raw_metrics"]["semantic_bad_match_flagged"])
        self.assertFalse(result["job_metrics"][1]["raw_metrics"]["semantic_bad_match_flagged"])
        self.assertFalse(result["job_metrics"][2]["raw_metrics"]["semantic_bad_match_flagged"])
        self.assertEqual(
            result["job_metrics"][0]["raw_metrics"]["semantic_class_probabilities"],
            {"really_bad": 0.6, "really_good": 0.1, "unclear": 0.3},
        )

    def test_empty_candidate_set_does_not_load_model(self) -> None:
        with patch.object(subject, "load_model_artifact") as loader:
            result = subject.run_pair_independent_bad_match_filter(
                job_ids=[], operation_results=[], requirements_cluster_distances=None
            )

        loader.assert_not_called()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["ranked_job_ids"], [])

    def test_model_failure_returns_candidates_unchanged(self) -> None:
        with patch.object(subject, "load_model_artifact", side_effect=ValueError("broken")):
            result = subject.run_pair_independent_bad_match_filter(
                job_ids=[4, 8],
                operation_results=[],
                requirements_cluster_distances=None,
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["ranked_job_ids"], [4, 8])
        self.assertEqual(result["job_metrics"], {})


if __name__ == "__main__":
    unittest.main()
