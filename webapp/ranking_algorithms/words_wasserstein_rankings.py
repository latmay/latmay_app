"""
Flexible sliced-Wasserstein job ranking script.

Purpose
-------
Rank job descriptions against a resume using word-embedding geometry, with
optional preprocessing steps that can each be turned on or off:
1. stopword removal
2. frequent-word filtering
3. TF-IDF-based job-word selection

Primary function for other scripts
----------------------------------
main_rank_job_descriptions_by_wasserstein(...)

Testing entry point
-------------------
main()

Notes
-----
- This script is intentionally structured so hardcoded paths are easy to swap
  later for relative paths.
- Model loading uses a hardcoded cache path exactly as requested.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


import numpy as np
import pandas as pd

from scipy.stats import wasserstein_distance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ============================================================
# EASY-TO-EDIT PATH CONFIG
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT.parent / "data"

MODEL_CACHE_DIR = PROJECT_ROOT / "model_cache"

# These are only for the test harness in main().
TEST_RESUME_PATH = DATA_DIR / "sample_resume"
TEST_JOBS_CSV_PATH = DATA_DIR / "sample_combined_jobs_filtered.csv"


# ============================================================
# MODEL LOADING
# ============================================================

def load_minilm_model() -> SentenceTransformer:
    """
    Load MiniLM model from the hardcoded model cache directory.

    This is kept in one place so it is easy to switch to a relative path later.
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
# TEXT PREPROCESSING
# ============================================================

def tokenize_text(text: str) -> list[str]:
    """
    Lowercase and tokenize into alphabetic/apostrophe-containing words.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def build_stopword_set(
    custom_stopwords: Iterable[str] | None = None,
    *,
    use_stopword_filter: bool = True,
) -> set[str]:
    if not use_stopword_filter:
        return set()

    sklearn_sw = set(ENGLISH_STOP_WORDS)
    custom_sw = {w.lower() for w in (custom_stopwords or [])}
    return sklearn_sw | custom_sw


def maybe_remove_stopwords(
    tokens: list[str],
    *,
    stopword_set: set[str],
    use_stopword_filter: bool,
) -> list[str]:
    """
    Optionally remove stopwords.
    """
    if not use_stopword_filter:
        return tokens
    return [tok for tok in tokens if tok not in stopword_set]


def maybe_remove_frequent_words(
    tokens: list[str],
    *,
    use_frequent_word_filter: bool,
    max_count: int | None,
) -> list[str]:
    """
    Optionally remove words whose within-document frequency exceeds max_count.

    If filtering is disabled, or max_count is None, this step does nothing.
    """
    if not use_frequent_word_filter or max_count is None:
        return tokens

    if max_count < 1:
        raise ValueError("max_count must be >= 1 when frequent-word filtering is enabled.")

    freq = Counter(tokens)
    return [tok for tok in tokens if freq[tok] <= max_count]


def preprocess_resume_words(
    resume: str,
    *,
    custom_stopwords: Iterable[str] | None = None,
    use_stopword_filter: bool = True,
    use_frequent_word_filter: bool = True,
    max_count: int | None = 4,
    deduplicate_resume_words: bool = True,
) -> list[str]:
    """
    Preprocess the resume into a word list.

    Steps can be individually turned on/off:
    - stopword removal
    - frequent-word removal

    Optionally deduplicates words at the end.
    """
    stopword_set = build_stopword_set(
        custom_stopwords=custom_stopwords,
        use_stopword_filter=use_stopword_filter,
    )

    tokens = tokenize_text(resume)
    tokens = maybe_remove_stopwords(
        tokens,
        stopword_set=stopword_set,
        use_stopword_filter=use_stopword_filter,
    )
    tokens = maybe_remove_frequent_words(
        tokens,
        use_frequent_word_filter=use_frequent_word_filter,
        max_count=max_count,
    )

    if deduplicate_resume_words:
        return list(dict.fromkeys(tokens))

    return tokens


def preprocess_job_descriptions(
    job_descriptions: Sequence[str],
    *,
    custom_stopwords: Iterable[str] | None = None,
    use_stopword_filter: bool = True,
    use_frequent_word_filter: bool = True,
    max_count: int | None = 4,
    use_tfidf_filter: bool = True,
    top_tfidf_fraction: float | None = 0.10,
    max_words_per_job: int | None = 30,
) -> list[list[str]]:
    """
    Preprocess job descriptions into word lists.

    Steps can be individually turned on/off:
    - stopword removal
    - frequent-word removal
    - TF-IDF filtering

    Behavior:
    - If use_tfidf_filter is False, TF-IDF is skipped entirely.
    - If top_tfidf_fraction is None, TF-IDF is skipped entirely.
    """
    stopword_set = build_stopword_set(
        custom_stopwords=custom_stopwords,
        use_stopword_filter=use_stopword_filter,
    )

    processed_token_lists: list[list[str]] = []

    for jd in job_descriptions:
        tokens = tokenize_text(jd)
        tokens = maybe_remove_stopwords(
            tokens,
            stopword_set=stopword_set,
            use_stopword_filter=use_stopword_filter,
        )
        tokens = maybe_remove_frequent_words(
            tokens,
            use_frequent_word_filter=use_frequent_word_filter,
            max_count=max_count,
        )
        processed_token_lists.append(tokens)

    should_use_tfidf = use_tfidf_filter and (top_tfidf_fraction is not None)

    if not should_use_tfidf:
        final_lists: list[list[str]] = []
        for tokens in processed_token_lists:
            if max_words_per_job is not None:
                final_lists.append(tokens[:max_words_per_job])
            else:
                final_lists.append(tokens)
        return final_lists

    if top_tfidf_fraction is None or top_tfidf_fraction <= 0:
        raise ValueError("top_tfidf_fraction must be > 0 when TF-IDF filtering is enabled.")

    docs_for_tfidf = [" ".join(tokens) for tokens in processed_token_lists]
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[a-z]+(?:'[a-z]+)?\b")
    tfidf_matrix = vectorizer.fit_transform(docs_for_tfidf)
    vocab = np.array(vectorizer.get_feature_names_out())

    final_word_lists: list[list[str]] = []

    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(i)

        if row.nnz == 0:
            final_word_lists.append([])
            continue

        scores = row.toarray().ravel()
        nonzero_idx = np.flatnonzero(scores)

        if len(nonzero_idx) == 0:
            final_word_lists.append([])
            continue

        n_keep = max(1, int(np.ceil(top_tfidf_fraction * len(nonzero_idx))))
        if max_words_per_job is not None:
            n_keep = min(n_keep, max_words_per_job)

        ranked_idx = nonzero_idx[np.argsort(scores[nonzero_idx])[::-1]]
        keep_idx = ranked_idx[:n_keep]

        selected_words = vocab[keep_idx].tolist()
        final_word_lists.append(selected_words)

    return final_word_lists


# ============================================================
# EMBEDDING HELPERS
# ============================================================

def build_embedding_lookup(
    words: Iterable[str],
    model: SentenceTransformer,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    """
    Embed each unique word once and return a lookup table.
    """
    unique_words = sorted(set(words))

    if not unique_words:
        return {}

    vectors = model.encode(
        unique_words,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    return {word: vec for word, vec in zip(unique_words, vectors)}


def point_cloud_from_lookup(
    words: Sequence[str],
    lookup: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Build a 2D point cloud array from a word -> embedding lookup.
    """
    vectors = [lookup[w] for w in words if w in lookup]

    if not vectors:
        raise ValueError("No vectors available to build point cloud.")

    return np.vstack(vectors)


# ============================================================
# WASSERSTEIN
# ============================================================

def random_projection_matrix(
    *,
    dim: int,
    n_projections: int,
    random_state: int | None = 0,
) -> np.ndarray:
    if n_projections <= 0:
        raise ValueError("No valid projections generated.")

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
    if Y.shape[1] != projection_matrix.shape[0]:
        raise ValueError("Point clouds must have same embedding dimension.")

    sorted_y_projections = sorted_projected_cloud(Y, projection_matrix)
    distances = [
        wasserstein_distance(sorted_x_projections[:, projection_idx], sorted_y_projections[:, projection_idx])
        for projection_idx in range(projection_matrix.shape[1])
    ]
    if not distances:
        raise ValueError("No valid projections generated.")
    return float(np.mean(distances))


def sliced_wasserstein_distance(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 50,
    random_state: int | None = 0,
) -> float:
    """
    Compute sliced Wasserstein distance between two point clouds.
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must both be 2D arrays.")

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Point clouds must have same embedding dimension.")

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

def main_rank_job_descriptions_by_wasserstein(
    job_descriptions: Sequence[str],
    resume: str,
    model: SentenceTransformer,
    *,
    custom_stopwords: Iterable[str] | None = None,
    use_stopword_filter: bool = True,
    use_frequent_word_filter: bool = True,
    max_count: int | None = 4,
    use_tfidf_filter: bool = True,
    top_tfidf_fraction: float | None = 0.10,
    max_words_per_job: int | None = 30,
    deduplicate_resume_words: bool = True,
    n_projections: int = 50,
    random_state: int | None = 0,
    embedding_batch_size: int = 128,
    return_debug_info: bool = False,
    precomputed_job_word_lists: Sequence[Sequence[str]] | None = None,
    precomputed_job_embeddings: Sequence[np.ndarray] | None = None,
    precomputed_resume_words: Sequence[str] | None = None,
    precomputed_resume_embeddings: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """
    Primary ranking function for reuse by other scripts.

    Inputs
    ------
    job_descriptions : Sequence[str]
        Raw job description texts.
    resume : str
        Raw resume text.
    model : SentenceTransformer
        Preloaded MiniLM model.

    Optional preprocessing controls
    -------------------------------
    use_stopword_filter : bool
        If False, stopword filtering is skipped.
    use_frequent_word_filter : bool
        If False, frequent-word filtering is skipped.
    max_count : int | None
        Frequency threshold for frequent-word filtering.
        If None, frequent-word filtering does nothing.
    use_tfidf_filter : bool
        If False, TF-IDF filtering is skipped.
    top_tfidf_fraction : float | None
        Fraction of top-TFIDF words to keep per job.
        If None, TF-IDF filtering does nothing.

    Outputs
    -------
    list[dict[str, Any]]
        Sorted by increasing Wasserstein distance.
        Lower distance = closer match.
    """
    if not job_descriptions:
        raise ValueError("job_descriptions cannot be empty.")

    if precomputed_resume_embeddings is None and (not isinstance(resume, str) or not resume.strip()):
        raise ValueError("resume must be a non-empty string.")

    resume_words = (
        [str(word) for word in precomputed_resume_words or [] if str(word).strip()]
        if precomputed_resume_embeddings is not None
        else preprocess_resume_words(
            resume=resume,
            custom_stopwords=custom_stopwords,
            use_stopword_filter=use_stopword_filter,
            use_frequent_word_filter=use_frequent_word_filter,
            max_count=max_count,
            deduplicate_resume_words=deduplicate_resume_words,
        )
    )

    if not resume_words:
        raise ValueError("Resume has no remaining words after preprocessing.")

    if precomputed_job_word_lists is not None:
        if len(precomputed_job_word_lists) != len(job_descriptions):
            raise ValueError("precomputed_job_word_lists must match job_descriptions length.")
        job_word_lists = [[str(word) for word in words if str(word).strip()] for words in precomputed_job_word_lists]
    else:
        job_word_lists = preprocess_job_descriptions(
            job_descriptions=job_descriptions,
            custom_stopwords=custom_stopwords,
            use_stopword_filter=use_stopword_filter,
            use_frequent_word_filter=use_frequent_word_filter,
            max_count=max_count,
            use_tfidf_filter=use_tfidf_filter,
            top_tfidf_fraction=top_tfidf_fraction,
            max_words_per_job=max_words_per_job,
        )

    if precomputed_job_embeddings is not None and len(precomputed_job_embeddings) != len(job_descriptions):
        raise ValueError("precomputed_job_embeddings must match job_descriptions length.")

    valid_job_rows: list[dict[str, Any]] = []
    all_job_words: list[str] = []

    for idx, words in enumerate(job_word_lists):
        job_embedding_matrix = None
        if precomputed_job_embeddings is not None:
            job_embedding_matrix = np.asarray(precomputed_job_embeddings[idx], dtype=np.float32)

        if not words or (job_embedding_matrix is not None and len(job_embedding_matrix) == 0):
            continue

        valid_job_rows.append(
            {
                "job_index": idx,
                "words": words,
                "job_embedding_matrix": job_embedding_matrix,
                "job_description": job_descriptions[idx],
            }
        )
        all_job_words.extend(words)

    if not valid_job_rows:
        raise ValueError("No valid job descriptions remained after preprocessing.")

    if precomputed_resume_embeddings is not None:
        X_resume = np.asarray(precomputed_resume_embeddings, dtype=np.float32)
        if X_resume.ndim != 2 or len(X_resume) != len(resume_words):
            raise ValueError("precomputed resume words and embeddings must align.")
        embedding_lookup = {}
    else:
        if precomputed_job_embeddings is not None:
            print("words_wasserstein: using precomputed job-side embeddings; embedding resume words only.")
            all_words_to_embed = list(set(resume_words))
        else:
            print("words_wasserstein: precomputed job-side embeddings unavailable; embedding job and resume words.")
            all_words_to_embed = list(set(all_job_words) | set(resume_words))
        embedding_lookup = build_embedding_lookup(
            all_words_to_embed,
            model=model,
            batch_size=embedding_batch_size,
        )

    if precomputed_resume_embeddings is None:
        X_resume = point_cloud_from_lookup(resume_words, embedding_lookup)
    projection_matrix = random_projection_matrix(
        dim=X_resume.shape[1],
        n_projections=n_projections,
        random_state=random_state,
    )
    sorted_resume_projections = sorted_projected_cloud(X_resume, projection_matrix)

    results: list[dict[str, Any]] = []

    for row in valid_job_rows:
        if row["job_embedding_matrix"] is not None:
            Y_job = row["job_embedding_matrix"]
        else:
            Y_job = point_cloud_from_lookup(row["words"], embedding_lookup)

        dist = sliced_wasserstein_distance_from_sorted_projections(
            sorted_x_projections=sorted_resume_projections,
            Y=Y_job,
            projection_matrix=projection_matrix,
        )

        result_row: dict[str, Any] = {
            "job_index": row["job_index"],
            "distance": dist,
            "job_description": row["job_description"],
        }

        if return_debug_info:
            result_row["n_resume_words"] = len(resume_words)
            result_row["n_job_words"] = len(row["words"])
            result_row["resume_words"] = resume_words
            result_row["job_words"] = row["words"]

        results.append(result_row)

    if not results:
        raise ValueError("No valid job descriptions remained after ranking.")

    results.sort(key=lambda x: x["distance"])

    for rank, row in enumerate(results, start=1):
        row["rank"] = rank

    return results


# ============================================================
# CSV / FILE HELPERS FOR TESTING
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
    Load CSV and ensure the chosen text column exists.
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
    Test harness so you can quickly verify the script works.
    """
    resume_text = read_text_file(TEST_RESUME_PATH)
    jobs_df = load_job_descriptions_from_csv(
        csv_path=TEST_JOBS_CSV_PATH,
        text_column="content_text",
    )

    job_descriptions = jobs_df["content_text"].tolist()
    model = load_minilm_model()

    results = main_rank_job_descriptions_by_wasserstein(
        job_descriptions=job_descriptions,
        resume=resume_text,
        model=model,
        custom_stopwords={"preferred", "required", "qualification", "qualifications"},
        use_stopword_filter=True,
        use_frequent_word_filter=False,
        max_count=4,
        use_tfidf_filter=True,
        top_tfidf_fraction=0.25,
        max_words_per_job=50,
        deduplicate_resume_words=True,
        n_projections=50,
        random_state=0,
        embedding_batch_size=128,
        return_debug_info=False,
    )

    top_k = 100
    print("\n===== TOP JOBS BY FLEXIBLE SLICED WASSERSTEIN DISTANCE =====\n")

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
