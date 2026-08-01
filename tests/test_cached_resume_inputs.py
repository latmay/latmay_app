from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

from ranking_algorithms.phrases_wasserstein_rankings import rank_job_descriptions_by_phrase_sliced_wasserstein
from ranking_algorithms.seniority_filter import LEVEL_PHRASES, run_seniority_filter
from ranking_algorithms.words_wasserstein_rankings import main_rank_job_descriptions_by_wasserstein
from resume_profiles import build_resume_profile


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            value = float(sum(ord(char) for char in str(text)) % 17 + 1)
            vector = np.array([value, value + 1, value + 2, value + 3], dtype=np.float32)
            if kwargs.get("normalize_embeddings"):
                vector = vector / np.linalg.norm(vector)
            rows.append(vector)
        return np.vstack(rows)


class NoEncodeModel:
    def encode(self, *_args, **_kwargs):
        raise AssertionError("cached ranking attempted to encode text")


class CachedResumeInputTests(unittest.TestCase):
    def test_profile_contains_derived_inputs_but_not_resume_words_or_phrases(self) -> None:
        profile = build_resume_profile(
            "Python developer building distributed AWS services with PostgreSQL and Docker.",
            FakeEmbeddingModel(),
        )

        self.assertNotIn("resume_text", profile)
        self.assertNotIn("word_terms", profile)
        self.assertNotIn("phrase_chunks", profile)
        self.assertGreater(profile["word_count"], 0)
        self.assertGreater(profile["phrase_count"], 0)
        self.assertEqual(len(profile["word_embeddings"]), profile["word_count"])
        self.assertEqual(len(profile["phrase_embeddings"]), profile["phrase_count"])

    def test_word_ranking_uses_cached_resume_embeddings_without_encoding(self) -> None:
        results = main_rank_job_descriptions_by_wasserstein(
            ["first", "second"],
            "",
            NoEncodeModel(),
            precomputed_job_word_lists=[["first"], ["second"]],
            precomputed_job_embeddings=[
                np.array([[1, 0, 0, 0]], dtype=np.float32),
                np.array([[0, 1, 0, 0]], dtype=np.float32),
            ],
            precomputed_resume_words=["cached"],
            precomputed_resume_embeddings=np.array([[1, 0, 0, 0]], dtype=np.float32),
        )
        self.assertEqual(results[0]["job_index"], 0)

    def test_phrase_ranking_uses_cached_resume_embeddings_without_encoding(self) -> None:
        results = rank_job_descriptions_by_phrase_sliced_wasserstein(
            model=NoEncodeModel(),
            job_descriptions=["first", "second"],
            resume="",
            precomputed_job_phrase_chunks=[["first"], ["second"]],
            precomputed_job_phrase_embeddings=[
                np.array([[1, 0, 0, 0]], dtype=np.float32),
                np.array([[0, 1, 0, 0]], dtype=np.float32),
            ],
            precomputed_resume_phrase_embeddings=np.array([[1, 0, 0, 0]], dtype=np.float32),
        )
        self.assertEqual(results[0]["job_index"], 0)

    def test_seniority_uses_cached_resume_and_anchor_embeddings_without_encoding(self) -> None:
        anchor_embeddings = np.eye(len(LEVEL_PHRASES), 8, dtype=np.float32)
        result = run_seniority_filter(
            job_ids=[0],
            resume_text="",
            minilm_model=NoEncodeModel(),
            job_titles=["Engineer"],
            job_requirements=["Build systems"],
            precomputed_title_requirements_embeddings=np.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32),
            precomputed_resume_embedding=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            precomputed_anchor_embeddings=anchor_embeddings,
        )
        self.assertEqual(result["status"], "ok")

    def test_cached_word_and_phrase_distances_match_fresh_resume_inputs(self) -> None:
        resume_text = "Python developer building distributed services and databases for customers."
        model = FakeEmbeddingModel()
        profile = build_resume_profile(resume_text, model)
        job_embeddings = [
            np.array([[1, 2, 3, 4]], dtype=np.float32),
            np.array([[4, 3, 2, 1]], dtype=np.float32),
        ]

        fresh_words = main_rank_job_descriptions_by_wasserstein(
            ["first", "second"],
            resume_text,
            model,
            custom_stopwords={"preferred", "required", "qualification", "qualifications"},
            use_frequent_word_filter=False,
            precomputed_job_word_lists=[["first"], ["second"]],
            precomputed_job_embeddings=job_embeddings,
        )
        cached_words = main_rank_job_descriptions_by_wasserstein(
            ["first", "second"],
            "",
            NoEncodeModel(),
            precomputed_job_word_lists=[["first"], ["second"]],
            precomputed_job_embeddings=job_embeddings,
            precomputed_resume_words=[f"cached-{index}" for index in range(profile["word_count"])],
            precomputed_resume_embeddings=np.asarray(profile["word_embeddings"], dtype=np.float32),
        )
        self.assertTrue(
            np.allclose(
                sorted(row["distance"] for row in fresh_words),
                sorted(row["distance"] for row in cached_words),
            )
        )

        fresh_phrases = rank_job_descriptions_by_phrase_sliced_wasserstein(
            model=model,
            job_descriptions=["first", "second"],
            resume=resume_text,
            precomputed_job_phrase_chunks=[["first"], ["second"]],
            precomputed_job_phrase_embeddings=job_embeddings,
        )
        cached_phrases = rank_job_descriptions_by_phrase_sliced_wasserstein(
            model=NoEncodeModel(),
            job_descriptions=["first", "second"],
            resume="",
            precomputed_job_phrase_chunks=[["first"], ["second"]],
            precomputed_job_phrase_embeddings=job_embeddings,
            precomputed_resume_phrase_embeddings=np.asarray(profile["phrase_embeddings"], dtype=np.float32),
        )
        self.assertTrue(
            np.allclose(
                sorted(row["distance"] for row in fresh_phrases),
                sorted(row["distance"] for row in cached_phrases),
            )
        )


if __name__ == "__main__":
    unittest.main()
