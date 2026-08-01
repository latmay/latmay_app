from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

from ranking_algorithms.seniority_filter import run_seniority_filter
from reduction_policy import reduce_job_ids


def basis(index: int) -> np.ndarray:
    vector = np.zeros(6, dtype=np.float32)
    vector[index] = 1.0
    return vector


class FakeMiniLM:
    def __init__(self) -> None:
        self.encoded_batches: list[list[str]] = []

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        self.encoded_batches.append(list(texts))
        return np.asarray([self.vector_for_text(text) for text in texts], dtype=np.float32)

    def vector_for_text(self, text: str) -> np.ndarray:
        lower = text.lower()
        if "principal" in lower or "manager" in lower:
            return basis(5)
        if "lead" in lower:
            return basis(4)
        if "senior" in lower:
            return basis(3)
        if "mid" in lower:
            return basis(2)
        if "junior" in lower:
            return basis(1)
        if "entry" in lower:
            return basis(0)
        return basis(2)


class SeniorityFilterTests(unittest.TestCase):
    def test_disabled_filter_returns_skipped_passthrough_ids(self) -> None:
        model = FakeMiniLM()

        result = run_seniority_filter(
            job_ids=[0, 1],
            resume_text="mid resume",
            minilm_model=model,  # type: ignore[arg-type]
            enabled=False,
        )

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["ranked_job_ids"], [0, 1])
        self.assertEqual(model.encoded_batches, [])

    def test_job_more_than_max_gap_above_resume_is_removed_by_reduction(self) -> None:
        result = run_seniority_filter(
            job_ids=[0, 1, 2],
            resume_text="mid resume",
            minilm_model=FakeMiniLM(),  # type: ignore[arg-type]
            job_titles=["Junior engineer", "Mid-level engineer", "Lead engineer"],
            job_requirements=["junior", "mid", "lead"],
            max_gap=1.5,
            enabled=True,
            level_probability_alpha=3.0,
        )

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["job_metrics"][2]["raw_metrics"]["is_filtered"])  # type: ignore[index]
        kept = reduce_job_ids(
            current_job_ids=[0, 1, 2],
            operation_result=result,
            reduction_policies={
                "seniority_filter": {"filter_raw_metric": "is_filtered", "exclude_value": True}
            },
        )
        self.assertEqual(kept, [0, 1])

    def test_reduction_min_remaining_extends_top_fraction_cut(self) -> None:
        result = {
            "operation_name": "mahalanobis_outlier",
            "status": "ok",
            "ranked_job_ids": [4, 3, 2, 1, 0],
            "job_metrics": {},
        }

        kept = reduce_job_ids(
            current_job_ids=[0, 1, 2, 3, 4],
            operation_result=result,
            reduction_policies={
                "mahalanobis_outlier": {
                    "top_fraction": 0.2,
                    "min_remaining_jobs": 3,
                }
            },
        )

        self.assertEqual(kept, [4, 3, 2])

    def test_reduction_min_remaining_extends_score_filter(self) -> None:
        result = {
            "operation_name": "multi_metric_bad_fit_filter",
            "status": "ok",
            "ranked_job_ids": [0, 1, 2, 3],
            "job_metrics": {
                0: {"score": 1},
                1: {"score": 0},
                2: {"score": 0},
                3: {"score": 0},
            },
        }

        kept = reduce_job_ids(
            current_job_ids=[0, 1, 2, 3],
            operation_result=result,
            reduction_policies={
                "multi_metric_bad_fit_filter": {
                    "keep_score_equals": 1,
                    "min_remaining_jobs": 3,
                }
            },
        )

        self.assertEqual(kept, [0, 1, 2])

    def test_negative_max_gap_requires_resume_to_be_more_senior_than_job(self) -> None:
        result = run_seniority_filter(
            job_ids=[0, 1],
            resume_text="senior resume",
            minilm_model=FakeMiniLM(),  # type: ignore[arg-type]
            job_titles=["Mid-level engineer", "Senior engineer"],
            job_requirements=["mid", "senior"],
            max_gap=-0.5,
            enabled=True,
            level_probability_alpha=3.0,
        )

        self.assertFalse(result["job_metrics"][0]["raw_metrics"]["is_filtered"])  # type: ignore[index]
        self.assertTrue(result["job_metrics"][1]["raw_metrics"]["is_filtered"])  # type: ignore[index]

    def test_job_more_than_max_junior_gap_below_resume_is_removed(self) -> None:
        result = run_seniority_filter(
            job_ids=[0, 1],
            resume_text="senior resume",
            minilm_model=FakeMiniLM(),  # type: ignore[arg-type]
            job_titles=["Junior engineer", "Senior engineer"],
            job_requirements=["junior", "senior"],
            max_gap=1.5,
            max_junior_gap=1.5,
            enabled=True,
            level_probability_alpha=3.0,
        )

        self.assertTrue(result["job_metrics"][0]["raw_metrics"]["is_too_junior"])  # type: ignore[index]
        self.assertTrue(result["job_metrics"][0]["raw_metrics"]["is_filtered"])  # type: ignore[index]
        self.assertFalse(result["job_metrics"][1]["raw_metrics"]["is_filtered"])  # type: ignore[index]

    def test_precomputed_title_requirements_embeddings_are_used_for_jobs(self) -> None:
        model = FakeMiniLM()
        precomputed = np.vstack([basis(1), basis(2), basis(4)])

        result = run_seniority_filter(
            job_ids=[0, 1, 2],
            resume_text="mid resume",
            minilm_model=model,  # type: ignore[arg-type]
            job_titles=["should not be embedded"] * 3,
            job_requirements=["should not be embedded"] * 3,
            precomputed_title_requirements_embeddings=precomputed,
            max_gap=1.5,
            enabled=True,
        )

        self.assertEqual(result["status"], "ok")
        encoded_texts = [text for batch in model.encoded_batches for text in batch]
        self.assertFalse(any(text.startswith("Job title:") for text in encoded_texts))
        self.assertTrue(result["job_metrics"][2]["raw_metrics"]["is_filtered"])  # type: ignore[index]
        self.assertIn("job_level_probability_distribution", result["job_metrics"][1]["raw_metrics"])  # type: ignore[index]

    def test_title_signal_raises_job_score_to_floor(self) -> None:
        result = run_seniority_filter(
            job_ids=[0],
            resume_text="mid resume",
            minilm_model=FakeMiniLM(),  # type: ignore[arg-type]
            job_titles=["Senior engineer"],
            job_requirements=["mid"],
            precomputed_title_requirements_embeddings=np.vstack([basis(2)]),
            max_gap=0.5,
            enabled=True,
            level_probability_alpha=3.0,
        )

        raw_metrics = result["job_metrics"][0]["raw_metrics"]  # type: ignore[index]
        self.assertEqual(raw_metrics["job_raw_seniority_label"], "mid")
        self.assertEqual(raw_metrics["job_seniority_label"], "senior")
        self.assertTrue(raw_metrics["job_title_seniority_floor_applied"])
        self.assertIsNone(raw_metrics["job_yoe_seniority_floor"])
        self.assertFalse(raw_metrics["job_yoe_seniority_floor_applied"])
        self.assertTrue(raw_metrics["is_too_senior"])

    def test_yoe_signal_raises_job_score_to_floor(self) -> None:
        result = run_seniority_filter(
            job_ids=[0],
            resume_text="mid resume",
            minilm_model=FakeMiniLM(),  # type: ignore[arg-type]
            job_titles=["Software engineer"],
            job_requirements=["mid"],
            job_min_years_experience=[8],
            precomputed_title_requirements_embeddings=np.vstack([basis(2)]),
            max_gap=0.5,
            enabled=True,
            level_probability_alpha=3.0,
        )

        raw_metrics = result["job_metrics"][0]["raw_metrics"]  # type: ignore[index]
        self.assertEqual(raw_metrics["job_raw_seniority_label"], "mid")
        self.assertEqual(raw_metrics["job_seniority_label"], "lead")
        self.assertIsNone(raw_metrics["job_title_seniority_floor"])
        self.assertFalse(raw_metrics["job_title_seniority_floor_applied"])
        self.assertEqual(raw_metrics["job_yoe_seniority_floor"], 4.0)
        self.assertEqual(raw_metrics["job_yoe_seniority_floor_label"], "lead")
        self.assertTrue(raw_metrics["job_yoe_seniority_floor_applied"])
        self.assertTrue(raw_metrics["is_too_senior"])

    def test_internship_title_caps_job_score_to_entry(self) -> None:
        result = run_seniority_filter(
            job_ids=[0],
            resume_text="mid resume",
            minilm_model=FakeMiniLM(),  # type: ignore[arg-type]
            job_titles=["Internship - Photonics Design"],
            job_requirements=["lead"],
            precomputed_title_requirements_embeddings=np.vstack([basis(4)]),
            max_gap=1.5,
            enabled=True,
            level_probability_alpha=3.0,
        )

        raw_metrics = result["job_metrics"][0]["raw_metrics"]  # type: ignore[index]
        self.assertEqual(raw_metrics["job_raw_seniority_label"], "lead")
        self.assertEqual(raw_metrics["job_seniority_label"], "entry")
        self.assertEqual(raw_metrics["job_title_seniority_floor"], 0.0)
        self.assertFalse(raw_metrics["job_title_seniority_floor_applied"])
        self.assertEqual(raw_metrics["job_title_seniority_ceiling"], 0.0)
        self.assertEqual(raw_metrics["job_title_seniority_ceiling_label"], "entry")
        self.assertTrue(raw_metrics["job_title_seniority_ceiling_applied"])
        self.assertFalse(raw_metrics["is_too_senior"])

    def test_fallback_embedding_path_embeds_title_and_requirements(self) -> None:
        model = FakeMiniLM()

        result = run_seniority_filter(
            job_ids=[0],
            resume_text="mid resume",
            minilm_model=model,  # type: ignore[arg-type]
            job_titles=["Senior engineer"],
            job_requirements=["Senior: owns complex projects"],
            enabled=True,
        )

        self.assertEqual(result["status"], "ok")
        encoded_texts = [text for batch in model.encoded_batches for text in batch]
        self.assertTrue(any(text.startswith("Job title: Senior engineer") for text in encoded_texts))


if __name__ == "__main__":
    unittest.main()
