from __future__ import annotations

import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

sys.modules.setdefault("pandas", types.SimpleNamespace())
sys.modules.setdefault(
    "sentence_transformers",
    types.SimpleNamespace(SentenceTransformer=object),
)
sys.modules.setdefault("scipy", types.SimpleNamespace())
sys.modules.setdefault(
    "scipy.stats",
    types.SimpleNamespace(wasserstein_distance=lambda *_args, **_kwargs: 0.0),
)

import numpy as np  # noqa: E402
from ranking_algorithms import resume_phrase_coverage  # noqa: E402


class ResumePhraseExampleCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        resume_phrase_coverage._EXAMPLE_CACHE.clear()

    def tearDown(self) -> None:
        resume_phrase_coverage._EXAMPLE_CACHE.clear()

    def test_example_phrase_cache_is_saved_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_dir = tmp / "resume_dataset"
            dataset_dir.mkdir()
            (dataset_dir / "example.md").write_text("Built APIs. Led data pipeline work.", encoding="utf-8")
            cache_path = tmp / "examples.npz"
            calls = {"embeds": 0}

            def fake_embed_texts(**kwargs: object) -> np.ndarray:
                calls["embeds"] += 1
                texts = kwargs["texts"]
                return np.ones((len(texts), 2), dtype=np.float32)

            env = {
                "RESUME_PHRASE_EXAMPLE_CACHE_PATH": str(cache_path),
                "MINILM_MODEL_NAME": "test-minilm",
                "MINILM_MODEL_REVISION": "test-revision",
            }

            with (
                patch.dict("os.environ", env, clear=False),
                patch.object(resume_phrase_coverage, "embed_texts", fake_embed_texts),
                redirect_stdout(io.StringIO()),
            ):
                first_chunks, first_embeddings = resume_phrase_coverage.build_persistent_example_resume_cache(
                    model=object(),
                    resume_dataset_dir=dataset_dir,
                    min_chunk_words=1,
                    max_chunk_words=12,
                )
                resume_phrase_coverage._EXAMPLE_CACHE.clear()
                second_chunks, second_embeddings = resume_phrase_coverage.build_persistent_example_resume_cache(
                    model=object(),
                    resume_dataset_dir=dataset_dir,
                    min_chunk_words=1,
                    max_chunk_words=12,
                )

            self.assertTrue(cache_path.exists())
            self.assertEqual(calls["embeds"], 1)
            self.assertEqual(first_chunks, second_chunks)
            np.testing.assert_array_equal(first_embeddings, second_embeddings)


if __name__ == "__main__":
    unittest.main()
