from __future__ import annotations

"""Smoke test the local ranking path against JSONL/NPZ serving artifacts."""

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
WEBAPP_DIR = ROOT / "webapp"

os.environ["USE_RANKING_ARTIFACTS"] = "true"
os.environ["LOCAL_JOBS_METADATA_PATH"] = str(DATA_DIR / "jobs_metadata.jsonl")
os.environ["LOCAL_JOB_EMBEDDINGS_PATH"] = str(DATA_DIR / "job_embeddings.npz")

sys.path.insert(0, str(WEBAPP_DIR))

from ranking_service import run_resume_ranking  # noqa: E402


def main() -> None:
    resume_path = DATA_DIR / "sample_resume"
    if not resume_path.exists():
        raise FileNotFoundError(f"Missing sample resume: {resume_path}")

    payload = run_resume_ranking(
        resume_text=resume_path.read_text(encoding="utf-8"),
        country=None,
        state=None,
        max_required_yoe=None,
        top_k_to_show=5,
    )

    results = payload.get("results") or []
    if not results:
        raise RuntimeError("Smoke test failed: ranking returned no results.")

    print("Top 5 local artifact ranking results:")
    for row in results[:5]:
        print(
            f"{row.get('final_rank')}. "
            f"{row.get('title')} | {row.get('company')} | "
            f"score={row.get('cross_encoder_score')}"
        )


if __name__ == "__main__":
    main()
