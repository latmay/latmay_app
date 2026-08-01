"""
Cross-encoder text comparison module using a preloaded model.

Purpose:
- Computes a semantic relevance score between two texts (e.g., resume vs job description)
  using a provided cross-encoder model.
- Handles long texts by splitting them into chunks and aggregating chunk-pair scores.

Inputs:
- text_a (str) OR file_a (str): First text or path to text file
- text_b (str) OR file_b (str): Second text or path to text file
- model (CrossEncoder): PRELOADED cross-encoder model (must be passed in)

Optional chunking params:
- chunk_size_words (int): Approximate number of words per chunk
- chunk_overlap_words (int): Overlap between consecutive chunks
- aggregation (str): How to combine chunk-pair scores
    - "max"
    - "mean"
    - "top_k_mean"
- top_k (int): Number of top chunk-pair scores to average if aggregation="top_k_mean"

Outputs:
- float: Relevance score
  Higher = more relevant / better match
  Lower = less relevant

Notes:
- This module does NOT load models; caller must provide one.
- Designed for reuse in pipelines where model is loaded once.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from sentence_transformers import CrossEncoder


def read_text_file(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    return path.read_text(encoding="utf-8")


def chunk_text_by_words(
    text: str,
    chunk_size_words: int = 180,
    chunk_overlap_words: int = 40,
) -> List[str]:
    """
    Split text into overlapping word chunks.

    Parameters
    ----------
    text : str
        Input text.
    chunk_size_words : int
        Approximate number of words per chunk.
    chunk_overlap_words : int
        Number of overlapping words between consecutive chunks.

    Returns
    -------
    List[str]
        List of text chunks.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    words = text.split()

    if not words:
        return []

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive.")

    if chunk_overlap_words < 0:
        raise ValueError("chunk_overlap_words cannot be negative.")

    if chunk_overlap_words >= chunk_size_words:
        raise ValueError("chunk_overlap_words must be smaller than chunk_size_words.")

    step = chunk_size_words - chunk_overlap_words
    chunks = []

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size_words]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))

        if start + chunk_size_words >= len(words):
            break

    return chunks


def aggregate_scores(
    scores: List[float],
    aggregation: str = "top_k_mean",
    top_k: int = 3,
) -> float:
    """
    Aggregate chunk-pair scores into a single score.

    Parameters
    ----------
    scores : List[float]
        Chunk-pair scores.
    aggregation : str
        Aggregation strategy: "max", "mean", or "top_k_mean".
    top_k : int
        Number of top scores to average when using "top_k_mean".

    Returns
    -------
    float
        Aggregated score.
    """
    if not scores:
        raise ValueError("scores cannot be empty.")

    if aggregation == "max":
        return float(max(scores))

    if aggregation == "mean":
        return float(sum(scores) / len(scores))

    if aggregation == "top_k_mean":
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        sorted_scores = sorted(scores, reverse=True)
        k = min(top_k, len(sorted_scores))
        return float(sum(sorted_scores[:k]) / k)

    raise ValueError("aggregation must be one of: 'max', 'mean', 'top_k_mean'")


def compare_texts_cross_encoder(
    text_a: str,
    text_b: str,
    model: CrossEncoder,
    chunk_size_words: int = 180,
    chunk_overlap_words: int = 40,
    aggregation: str = "top_k_mean",
    top_k: int = 3,
    batch_size: int = 32,
) -> float:
    """
    Compare two text strings using a cross-encoder with chunking for long inputs.

    Parameters
    ----------
    text_a : str
        First text.
    text_b : str
        Second text.
    model : CrossEncoder
        A preloaded cross-encoder model.
    chunk_size_words : int
        Approximate number of words per chunk.
    chunk_overlap_words : int
        Overlap between consecutive chunks.
    aggregation : str
        How to combine chunk-pair scores: "max", "mean", or "top_k_mean".
    top_k : int
        Number of top chunk-pair scores to average if aggregation="top_k_mean".
    batch_size : int
        Batch size for model.predict.

    Returns
    -------
    float
        Aggregated cross-encoder relevance score.
    """
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        raise TypeError("text_a and text_b must both be strings.")

    if model is None:
        raise ValueError("model must be a preloaded CrossEncoder instance.")

    text_a = text_a.strip()
    text_b = text_b.strip()

    if not text_a or not text_b:
        raise ValueError("text_a and text_b must both be non-empty.")

    chunks_a = chunk_text_by_words(
        text=text_a,
        chunk_size_words=chunk_size_words,
        chunk_overlap_words=chunk_overlap_words,
    )
    chunks_b = chunk_text_by_words(
        text=text_b,
        chunk_size_words=chunk_size_words,
        chunk_overlap_words=chunk_overlap_words,
    )

    if not chunks_a:
        raise ValueError("text_a produced no chunks.")
    if not chunks_b:
        raise ValueError("text_b produced no chunks.")

    pairs = [(a, b) for a in chunks_a for b in chunks_b]
    scores = model.predict(pairs, batch_size=batch_size)

    return aggregate_scores(
        scores=[float(s) for s in scores],
        aggregation=aggregation,
        top_k=top_k,
    )


def compare_text_files_cross_encoder(
    file_a: str,
    file_b: str,
    model: CrossEncoder,
    chunk_size_words: int = 180,
    chunk_overlap_words: int = 40,
    aggregation: str = "top_k_mean",
    top_k: int = 3,
    batch_size: int = 32,
) -> float:
    """
    Read two text files and compare them using a cross-encoder with chunking.

    Parameters
    ----------
    file_a : str
        Path to first text file.
    file_b : str
        Path to second text file.
    model : CrossEncoder
        A preloaded cross-encoder model.
    chunk_size_words : int
        Approximate number of words per chunk.
    chunk_overlap_words : int
        Overlap between consecutive chunks.
    aggregation : str
        How to combine chunk-pair scores: "max", "mean", or "top_k_mean".
    top_k : int
        Number of top chunk-pair scores to average if aggregation="top_k_mean".
    batch_size : int
        Batch size for model.predict.

    Returns
    -------
    float
        Aggregated cross-encoder relevance score.
    """
    text_a = read_text_file(file_a)
    text_b = read_text_file(file_b)

    return compare_texts_cross_encoder(
        text_a=text_a,
        text_b=text_b,
        model=model,
        chunk_size_words=chunk_size_words,
        chunk_overlap_words=chunk_overlap_words,
        aggregation=aggregation,
        top_k=top_k,
        batch_size=batch_size,
    )