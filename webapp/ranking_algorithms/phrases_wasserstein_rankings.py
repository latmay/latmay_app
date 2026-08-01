from __future__ import annotations
"""
Rank job descriptions against a resume by:
1. splitting both texts into phrase-sized chunks,
2. embedding those chunks into point clouds with a preloaded model,
3. computing sliced Wasserstein distance between the resume cloud and each job cloud.

Inputs
------
- model: a preloaded sentence-transformers style model with .encode(...)
- job_descriptions: list[str] of raw job description texts
- resume: raw resume text
- n_projections: number of random projections used in sliced Wasserstein distance

Output
------
- list of dicts (sorted by increasing distance), each containing:
    - rank
    - job_index
    - distance (lower = closer match)
    - job_description
"""



import html
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sentence_transformers import SentenceTransformer


# ============================================================
# EASY-TO-EDIT PATH CONFIG
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT.parent / "data"


MODEL_CACHE_DIR = PROJECT_ROOT / "model_cache"

# These are only for testing in main().
TEST_RESUME_PATH = DATA_DIR / "sample_resume"
TEST_JOBS_CSV_PATH = DATA_DIR / "sample_combined_jobs_filtered.csv"


# ============================================================
# MODEL LOADING
# ============================================================

def load_minilm_model() -> SentenceTransformer:
    """
    Load MiniLM model from the hardcoded model cache directory.
    """
    print("Loading MiniLM model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        cache_folder=str(MODEL_CACHE_DIR),
        local_files_only=True,
    )

    print("Model loaded.")
    return model


# ============================================================
# TEXT CLEANING AND PHRASE CHUNKING
# ============================================================

_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_CLAUSE_SPLIT_RE = re.compile(
    r"(?:\s*[;:]\s+)|"
    r"(?:\s+--\s+)|"
    r"(?:\s+\u2013\s+)|"
    r"(?:\s+\u2014\s+)|"
    r"(?:\s*,\s+(?=(?:and|or|but|while|where|which|that|using|including|with|to)\b))",
    flags=re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """
    Basic cleanup for resume and job text.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n([•\-])\s*", r". \1 ", text)

    parts = _SENTENCE_SPLIT_RE.split(text)
    return [normalize_text(p) for p in parts if normalize_text(p)]


def split_clauses(sentence: str) -> List[str]:
    """
    Split a sentence into smaller phrase-like units.
    """
    raw_parts = _CLAUSE_SPLIT_RE.split(sentence)
    parts = [normalize_text(p) for p in raw_parts if normalize_text(p)]

    merged: List[str] = []
    for part in parts:
        word_count = len(part.split())
        if merged and word_count < 4:
            merged[-1] = f"{merged[-1]} {part}".strip()
        else:
            merged.append(part)

    return merged


def phrase_chunks(
    text: str,
    min_words: int = 3,
    max_words: int = 24,
    include_sentences: bool = True,
    include_sentence_windows: bool = True,
) -> List[str]:
    """
    Produce phrase/clause chunks from text.
    """
    text = normalize_text(text)
    if not text:
        return []

    sentences = split_sentences(text)
    chunks: List[str] = []

    for sent in sentences:
        clauses = split_clauses(sent)

        for clause in clauses:
            wc = len(clause.split())

            if min_words <= wc <= max_words:
                chunks.append(clause)
            elif wc > max_words:
                words = clause.split()
                step = max_words
                for i in range(0, len(words), step):
                    sub = " ".join(words[i:i + max_words])
                    sub_wc = len(sub.split())
                    if min_words <= sub_wc <= max_words:
                        chunks.append(sub)

        if include_sentences:
            wc = len(sent.split())
            if min_words <= wc <= max_words:
                chunks.append(sent)

    if include_sentence_windows and len(sentences) >= 2:
        for i in range(len(sentences) - 1):
            window = f"{sentences[i]} {sentences[i + 1]}".strip()
            wc = len(window.split())

            if min_words <= wc <= max_words:
                chunks.append(window)
            elif wc > max_words:
                words = window.split()
                sub = " ".join(words[:max_words])
                if len(sub.split()) >= min_words:
                    chunks.append(sub)

    seen = set()
    deduped: List[str] = []

    for chunk in chunks:
        key = chunk.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(chunk)

    return deduped


# ============================================================
# EMBEDDING HELPERS
# ============================================================

def embed_texts(
    model: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 64,
    normalize_embeddings: bool = False,
    cache: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    Embed a sequence of texts with a sentence-transformers style model.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    if cache is None:
        cache = {}

    missing = [text for text in texts if text not in cache]

    if missing:
        encoded = model.encode(
            missing,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )

        for text, vec in zip(missing, encoded):
            cache[text] = np.asarray(vec, dtype=np.float32)

    arr = np.vstack([cache[text] for text in texts]).astype(np.float32)
    return arr


# ============================================================
# WASSERSTEIN
# ============================================================

def random_projection_matrix(
    *,
    dim: int,
    n_projections: int,
    random_state: int = 0,
) -> np.ndarray:
    if n_projections <= 0:
        raise ValueError("n_projections must be positive.")

    rng = np.random.default_rng(random_state)
    directions = rng.normal(size=(n_projections, dim)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    valid = norms[:, 0] > 0
    if not np.any(valid):
        raise ValueError("No valid projections generated.")

    directions = directions[valid] / norms[valid]
    return directions.T


def sorted_projected_cloud(point_cloud: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    return np.sort(point_cloud @ projection_matrix, axis=0)


def sliced_wasserstein_distance_from_sorted_projections(
    *,
    sorted_x_projections: np.ndarray,
    Y: np.ndarray,
    projection_matrix: np.ndarray,
) -> float:
    if Y.ndim != 2:
        raise ValueError("Y must be a 2D array.")
    if Y.shape[0] == 0:
        return float("inf")
    if Y.shape[1] != projection_matrix.shape[0]:
        raise ValueError("X and Y must have the same embedding dimension.")

    sorted_y_projections = sorted_projected_cloud(Y, projection_matrix)
    distances = [
        wasserstein_distance(sorted_x_projections[:, projection_idx], sorted_y_projections[:, projection_idx])
        for projection_idx in range(projection_matrix.shape[1])
    ]
    return float(np.mean(distances))


def sliced_wasserstein_distance(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_projections: int = 64,
    random_state: int = 0,
) -> float:
    """
    Compute average 1D Wasserstein distance over random projections.
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays.")

    if X.shape[0] == 0 or Y.shape[0] == 0:
        return float("inf")

    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same embedding dimension.")

    projection_matrix = random_projection_matrix(
        dim=X.shape[1],
        n_projections=n_projections,
        random_state=random_state,
    )
    sorted_x_projections = sorted_projected_cloud(X, projection_matrix)
    return sliced_wasserstein_distance_from_sorted_projections(
        sorted_x_projections=sorted_x_projections,
        Y=Y,
        projection_matrix=projection_matrix,
    )


# ============================================================
# PRIMARY FUNCTION OTHER SCRIPTS SHOULD CALL
# ============================================================

def rank_job_descriptions_by_phrase_sliced_wasserstein(
    model: Any,
    job_descriptions: Sequence[str],
    resume: str,
    *,
    n_projections: int = 64,
    random_state: int = 0,
    batch_size: int = 64,
    normalize_embeddings: bool = False,
    min_chunk_words: int = 3,
    max_chunk_words: int = 24,
    include_sentences: bool = True,
    include_sentence_windows: bool = True,
    precomputed_job_phrase_chunks: Sequence[Sequence[str]] | None = None,
    precomputed_job_phrase_embeddings: Sequence[np.ndarray] | None = None,
    precomputed_resume_phrase_embeddings: np.ndarray | None = None,
) -> List[Dict[str, Any]]:
    """
    Primary reusable function.

    Returns
    -------
    list of dicts (sorted by increasing distance), each containing:
        - rank
        - job_index
        - distance (lower = closer match)
        - job_description
    """
    if precomputed_resume_phrase_embeddings is None and (not isinstance(resume, str) or not resume.strip()):
        raise ValueError("resume must be a non-empty string.")

    if not job_descriptions:
        raise ValueError("job_descriptions cannot be empty.")

    cache: Dict[str, np.ndarray] = {}

    if precomputed_resume_phrase_embeddings is not None:
        resume_emb = np.asarray(precomputed_resume_phrase_embeddings, dtype=np.float32)
        if resume_emb.ndim != 2 or len(resume_emb) == 0:
            raise ValueError("precomputed_resume_phrase_embeddings must be a non-empty matrix.")
    else:
        resume_chunks = phrase_chunks(
            resume,
            min_words=min_chunk_words,
            max_words=max_chunk_words,
            include_sentences=include_sentences,
            include_sentence_windows=include_sentence_windows,
        )
        if not resume_chunks:
            raise ValueError("No resume chunks were produced.")
        resume_emb = embed_texts(
            model=model,
            texts=resume_chunks,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            cache=cache,
        )
    projection_matrix = random_projection_matrix(
        dim=resume_emb.shape[1],
        n_projections=n_projections,
        random_state=random_state,
    )
    sorted_resume_projections = sorted_projected_cloud(resume_emb, projection_matrix)

    results: List[Dict[str, Any]] = []

    if precomputed_job_phrase_chunks is not None and len(precomputed_job_phrase_chunks) != len(job_descriptions):
        raise ValueError("precomputed_job_phrase_chunks must match job_descriptions length.")
    if precomputed_job_phrase_embeddings is not None and len(precomputed_job_phrase_embeddings) != len(job_descriptions):
        raise ValueError("precomputed_job_phrase_embeddings must match job_descriptions length.")

    if precomputed_job_phrase_embeddings is not None:
        print("phrases_wasserstein: using precomputed job-side embeddings; embedding resume chunks only.")
    else:
        print("phrases_wasserstein: precomputed job-side embeddings unavailable; embedding job and resume chunks.")

    for idx, job_description in enumerate(job_descriptions):
        if precomputed_job_phrase_chunks is not None:
            jd_chunks = [
                str(chunk).strip()
                for chunk in precomputed_job_phrase_chunks[idx]
                if str(chunk).strip()
            ]
        else:
            jd_chunks = phrase_chunks(
                job_description,
                min_words=min_chunk_words,
                max_words=max_chunk_words,
                include_sentences=include_sentences,
                include_sentence_windows=include_sentence_windows,
            )

        if not jd_chunks:
            results.append(
                {
                    "job_index": idx,
                    "distance": float("inf"),
                    "job_description": job_description,
                }
            )
            continue

        if precomputed_job_phrase_embeddings is not None:
            jd_emb = np.asarray(precomputed_job_phrase_embeddings[idx], dtype=np.float32)
        else:
            jd_emb = embed_texts(
                model=model,
                texts=jd_chunks,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                cache=cache,
            )

        if len(jd_emb) == 0:
            results.append(
                {
                    "job_index": idx,
                    "distance": float("inf"),
                    "job_description": job_description,
                }
            )
            continue

        distance = sliced_wasserstein_distance_from_sorted_projections(
            sorted_x_projections=sorted_resume_projections,
            Y=jd_emb,
            projection_matrix=projection_matrix,
        )

        results.append(
            {
                "job_index": idx,
                "distance": distance,
                "job_description": job_description,
            }
        )

    results.sort(key=lambda row: row["distance"])

    for rank, row in enumerate(results, start=1):
        row["rank"] = rank

    return results


# ============================================================
# FILE HELPERS FOR TESTING
# ============================================================

def read_text_file(file_path: str | Path) -> str:
    """
    Read a UTF-8 text file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    return path.read_text(encoding="utf-8")


def load_job_descriptions_from_csv(
    csv_path: str | Path,
    text_column: str = "content_text",
) -> pd.DataFrame:
    """
    Load job descriptions from CSV and preserve metadata for testing output.
    """
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV.")

    df = df.copy()
    df["csv_row_index"] = df.index
    df[text_column] = df[text_column].fillna("").astype(str).map(str.strip)
    df = df[df[text_column] != ""].reset_index(drop=True)

    if df.empty:
        raise ValueError("No non-empty job descriptions found in CSV.")

    return df


# ============================================================
# TESTING ENTRY POINT
# ============================================================

def main() -> None:
    """
    Test harness so you can verify the script works on the TXT and CSV
    in the same folder.
    """
    resume_text = read_text_file(TEST_RESUME_PATH)
    jobs_df = load_job_descriptions_from_csv(
        csv_path=TEST_JOBS_CSV_PATH,
        text_column="content_text",
    )

    job_descriptions = jobs_df["content_text"].tolist()
    model = load_minilm_model()

    results = rank_job_descriptions_by_phrase_sliced_wasserstein(
        model=model,
        job_descriptions=job_descriptions,
        resume=resume_text,
        n_projections=64,
        random_state=0,
        batch_size=64,
        normalize_embeddings=False,
        min_chunk_words=3,
        max_chunk_words=24,
        include_sentences=True,
        include_sentence_windows=True,
    )

    top_k = 20

    print("\n===== TOP JOBS BY PHRASE SLICED WASSERSTEIN DISTANCE =====\n")

    for row in results[:top_k]:
        job_idx = row["job_index"]
        meta = jobs_df.iloc[job_idx]

        company = str(meta["company_name"]).strip() if "company_name" in jobs_df.columns else ""
        title = str(meta["title"]).strip() if "title" in jobs_df.columns else ""
        location = str(meta["location_name"]).strip() if "location_name" in jobs_df.columns else ""

        print(
            f"rank={row['rank']:>2} | "
            f"job_index={job_idx:>4} | "
            f"csv_row_index={int(meta['csv_row_index']):>5} | "
            f"distance={row['distance']:.6f} | "
            f"company={company} | "
            f"title={title} | "
            f"location={location}"
        )

    if results:
        best = results[0]
        best_idx = best["job_index"]
        best_meta = jobs_df.iloc[best_idx]

        print("\n===== BEST MATCH DETAILS =====\n")
        print(f"rank: {best['rank']}")
        print(f"job_index: {best_idx}")
        print(f"csv_row_index: {int(best_meta['csv_row_index'])}")

        if "company_name" in jobs_df.columns:
            print(f"company_name: {best_meta['company_name']}")
        if "title" in jobs_df.columns:
            print(f"title: {best_meta['title']}")
        if "location_name" in jobs_df.columns:
            print(f"location_name: {best_meta['location_name']}")
        if "absolute_url" in jobs_df.columns:
            print(f"absolute_url: {best_meta['absolute_url']}")

        print(f"distance: {best['distance']:.6f}")


if __name__ == "__main__":
    main()
