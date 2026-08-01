from __future__ import annotations

import os
from pathlib import Path

from model_loader import load_minilm_model
from ranking_algorithms.resume_phrase_coverage import build_persistent_example_resume_cache


def main() -> None:
    webapp_dir = Path(__file__).resolve().parents[1]
    resume_dataset_dir = Path(
        os.environ.get("RESUME_DATASET_DIR", webapp_dir / "webapp_data" / "resume_dataset")
    )
    chunks, embeddings = build_persistent_example_resume_cache(
        model=load_minilm_model(),
        resume_dataset_dir=resume_dataset_dir,
        min_chunk_words=3,
        max_chunk_words=24,
        include_sentences=True,
        include_sentence_windows=True,
        batch_size=64,
        normalize_embeddings=False,
    )
    print(
        "Resume phrase coverage baked cache build complete: "
        f"resume_dataset_dir={resume_dataset_dir}, "
        f"chunks={len(chunks)}, "
        f"embedding_shape={embeddings.shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
