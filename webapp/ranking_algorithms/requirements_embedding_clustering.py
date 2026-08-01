from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, KMeans


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT.parent / "data"

CSV_PATH = DATA_DIR / "sample_combined_jobs_filtered_with_requirements.csv"
RESUME_PATH = DATA_DIR / "sample_resume"
MODEL_CACHE_DIR = PROJECT_ROOT / "model_cache"

REQUIREMENTS_COLUMN = "extracted_requirements"
TITLE_COLUMN = "title"

# Options: "dbscan", "kmeans", "agglomerative", "birch"
CLUSTERING_METHOD = "kmeans"
N_CLUSTERS = 40


def load_minilm_model() -> SentenceTransformer:
    print("Loading MiniLM locally...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        cache_folder=str(MODEL_CACHE_DIR),
        local_files_only=True,
    )

    print("MiniLM loaded.")
    return model


def embed_texts(
    texts: list[str],
    model: SentenceTransformer,
    *,
    batch_size: int = 64,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )


def make_cluster_labels(
    embeddings: np.ndarray,
    *,
    method: str = CLUSTERING_METHOD,
    eps: float = 0.25,
    min_samples: int = 5,
    n_clusters: int = N_CLUSTERS,
    metric: str = "cosine",
    random_state: int = 0,
) -> np.ndarray:
    method = method.lower()

    if method == "dbscan":
        clusterer = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
        )
        return clusterer.fit_predict(embeddings)

    if method == "kmeans":
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
        )
        return clusterer.fit_predict(embeddings)

    if method == "agglomerative":
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=metric,
            linkage="average",
        )
        return clusterer.fit_predict(embeddings)

    if method == "birch":
        clusterer = Birch(
            n_clusters=n_clusters,
        )
        return clusterer.fit_predict(embeddings)

    raise ValueError(f"Unknown clustering method: {method}")


def cluster_jobs_by_requirements_df(
    jobs_df: pd.DataFrame,
    resume_text: str,
    model: SentenceTransformer,
    *,
    requirements_column: str = REQUIREMENTS_COLUMN,
    title_column: str = TITLE_COLUMN,
    clustering_method: str = CLUSTERING_METHOD,
    eps: float = 0.25,
    min_samples: int = 5,
    n_clusters: int = N_CLUSTERS,
    metric: str = "cosine",
    random_state: int = 0,
    embedding_batch_size: int = 64,
    normalize_embeddings: bool = True,
) -> dict[str, Any]:
    """
    Cluster an already-loaded jobs DataFrame using embeddings of title + requirements.

    This is the reusable version for the web ranking pipeline. It preserves an
    existing csv_row_index column when present, so later ranking stages can still
    map reduced rows back to the original CSV.
    """
    if requirements_column not in jobs_df.columns:
        raise ValueError(f"Missing column: {requirements_column}")

    if title_column not in jobs_df.columns:
        raise ValueError(f"Missing column: {title_column}")

    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError("resume_text must be a non-empty string.")

    df = jobs_df.copy()
    if "csv_row_index" not in df.columns:
        df["csv_row_index"] = df.index

    req_texts = df[requirements_column].fillna("").astype(str).map(str.strip)
    title_texts = df[title_column].fillna("").astype(str).map(str.strip)

    combined_texts = (
        "Job title: " + title_texts + "\n"
        "Requirements: " + req_texts
    ).map(str.strip)

    valid_mask = combined_texts != ""
    valid_df = df[valid_mask].copy().reset_index(drop=True)
    valid_texts = combined_texts[valid_mask].tolist()

    if not valid_texts:
        raise ValueError("No non-empty title/requirements text found.")

    effective_n_clusters = min(n_clusters, len(valid_texts))
    if clustering_method.lower() in {"kmeans", "agglomerative", "birch"} and effective_n_clusters < 1:
        raise ValueError("n_clusters must be at least 1.")

    job_embeddings = embed_texts(
        valid_texts,
        model,
        batch_size=embedding_batch_size,
        normalize_embeddings=normalize_embeddings,
    )

    resume_embedding = embed_texts(
        [resume_text],
        model,
        batch_size=embedding_batch_size,
        normalize_embeddings=normalize_embeddings,
    )[0]

    cluster_labels = make_cluster_labels(
        job_embeddings,
        method=clustering_method,
        eps=eps,
        min_samples=min_samples,
        n_clusters=effective_n_clusters,
        metric=metric,
        random_state=random_state,
    )

    valid_df["cluster_label"] = cluster_labels

    cluster_summaries: list[dict[str, Any]] = []

    for label in sorted(set(cluster_labels)):
        if label == -1:
            continue

        cluster_mask = cluster_labels == label
        cluster_embeddings = job_embeddings[cluster_mask]
        cluster_rows = valid_df[cluster_mask].copy()

        centroid = cluster_embeddings.mean(axis=0)
        resume_to_centroid_distance = float(np.linalg.norm(resume_embedding - centroid))

        cluster_summaries.append(
            {
                "cluster_label": int(label),
                "cluster_size": int(cluster_mask.sum()),
                "resume_to_centroid_distance": resume_to_centroid_distance,
                "csv_row_indices": cluster_rows["csv_row_index"].tolist(),
                "valid_df_indices": cluster_rows.index.tolist(),
                "centroid_embedding": centroid,
                "job_embeddings": cluster_embeddings,
                "jobs_df": cluster_rows,
            }
        )

    cluster_summaries.sort(key=lambda row: row["resume_to_centroid_distance"])

    for cluster_rank, cluster in enumerate(cluster_summaries, start=1):
        cluster["cluster_rank"] = cluster_rank

    return {
        "df": df,
        "valid_df": valid_df,
        "job_embeddings": job_embeddings,
        "resume_embedding": resume_embedding,
        "cluster_labels": cluster_labels,
        "clusters": cluster_summaries,
        "noise_count": int(np.sum(cluster_labels == -1)),
        "clustering_method": clustering_method,
        "n_clusters": effective_n_clusters,
    }


def cluster_jobs_by_requirements(
    csv_path: str | Path,
    resume_path: str | Path,
    model: SentenceTransformer,
    *,
    requirements_column: str = REQUIREMENTS_COLUMN,
    title_column: str = TITLE_COLUMN,
    clustering_method: str = CLUSTERING_METHOD,
    eps: float = 0.25,
    min_samples: int = 5,
    n_clusters: int = N_CLUSTERS,
    metric: str = "cosine",
    random_state: int = 0,
    embedding_batch_size: int = 64,
    normalize_embeddings: bool = True,
) -> dict[str, Any]:
    """
    File-based wrapper for standalone testing.
    """
    df = pd.read_csv(Path(csv_path))
    resume_text = Path(resume_path).read_text(encoding="utf-8")

    return cluster_jobs_by_requirements_df(
        jobs_df=df,
        resume_text=resume_text,
        model=model,
        requirements_column=requirements_column,
        title_column=title_column,
        clustering_method=clustering_method,
        eps=eps,
        min_samples=min_samples,
        n_clusters=n_clusters,
        metric=metric,
        random_state=random_state,
        embedding_batch_size=embedding_batch_size,
        normalize_embeddings=normalize_embeddings,
    )


def main() -> None:
    """
    Testing only.
    """

    model = load_minilm_model()

    result = cluster_jobs_by_requirements(
        csv_path=CSV_PATH,
        resume_path=RESUME_PATH,
        model=model,
        clustering_method=CLUSTERING_METHOD,
        n_clusters=N_CLUSTERS,
        eps=0.25,
        min_samples=5,
        metric="cosine",
    )

    clusters = result["clusters"]

    print("\n===== CLUSTER SUMMARY =====")
    print(f"Clustering method: {result['clustering_method']}")
    print(f"Number of requested clusters: {result['n_clusters']}")
    print(f"Number of returned clusters: {len(clusters)}")
    print(f"Noise jobs: {result['noise_count']}")

    rng = np.random.default_rng(0)

    for cluster_rank, cluster in enumerate(clusters, start=1):
        jobs_df = cluster["jobs_df"]

        print("\n" + "=" * 80)
        print(
            f"Cluster rank: {cluster_rank} | "
            f"cluster_label: {cluster['cluster_label']} | "
            f"size: {cluster['cluster_size']} | "
            f"resume_distance: {cluster['resume_to_centroid_distance']:.6f}"
        )
        print("=" * 80)

        sample_n = min(8, len(jobs_df))
        sampled_positions = rng.choice(len(jobs_df), size=sample_n, replace=False)
        sampled_jobs = jobs_df.iloc[sampled_positions]

        for _, row in sampled_jobs.iterrows():
            company = row.get("company_name", "")
            title = row.get("title", "")
            location = row.get("location_name", "")
            csv_row_index = row.get("csv_row_index", "")

            print(
                f"csv_row_index={csv_row_index} | "
                f"company={company} | "
                f"title={title} | "
                f"location={location}"
            )


if __name__ == "__main__":
    main()
