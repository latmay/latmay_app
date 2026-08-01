from __future__ import annotations

"""Build and persist derived resume-ranking profiles without storing resume text."""

import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import psycopg

from model_loader import get_minilm_model_name, get_minilm_model_revision
from ranking_algorithms.phrases_wasserstein_rankings import embed_texts, phrase_chunks
from ranking_algorithms.seniority_filter import LEVEL_PHRASES
from ranking_algorithms.technology_mismatch_filter import load_technology_terms, match_technologies
from ranking_algorithms.words_wasserstein_rankings import build_embedding_lookup, preprocess_resume_words


PROFILE_VERSION = "resume-profile-v1"
WORD_CUSTOM_STOPWORDS = {"preferred", "required", "qualification", "qualifications"}


def get_database_url() -> str:
    value = os.environ.get("DATABASE_URL") or os.environ.get("LOCAL_DATABASE_URL")
    if not value:
        raise RuntimeError("DATABASE_URL is required for saved resume profiles.")
    return value


def build_resume_profile(resume_text: str, model: Any) -> dict[str, Any]:
    """Compute every resume-side input used by the embedding-only cached path."""
    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError("Resume text is required.")

    overall_embedding = model.encode(
        [resume_text],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0].astype(np.float32)

    words = preprocess_resume_words(
        resume_text,
        custom_stopwords=WORD_CUSTOM_STOPWORDS,
        use_stopword_filter=True,
        use_frequent_word_filter=False,
        max_count=4,
        deduplicate_resume_words=True,
    )
    if not words:
        raise ValueError("Resume produced no usable ranking words.")
    word_lookup = build_embedding_lookup(words, model=model, batch_size=128)
    word_embeddings = np.vstack([word_lookup[word] for word in words]).astype(np.float32)

    chunks = phrase_chunks(
        resume_text,
        min_words=3,
        max_words=24,
        include_sentences=True,
        include_sentence_windows=True,
    )
    if not chunks:
        raise ValueError("Resume produced no usable ranking phrases.")
    phrase_embeddings = embed_texts(
        model=model,
        texts=chunks,
        batch_size=64,
        normalize_embeddings=False,
    ).astype(np.float32)

    seniority_anchor_embeddings = model.encode(
        LEVEL_PHRASES,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    technology_matches = match_technologies(resume_text, load_technology_terms())
    return {
        "profile_version": PROFILE_VERSION,
        "model_name": get_minilm_model_name(),
        "model_revision": get_minilm_model_revision(),
        "embedding_dimension": int(overall_embedding.shape[0]),
        "overall_embedding": overall_embedding.tolist(),
        "word_count": int(len(words)),
        "word_embeddings": word_embeddings.tolist(),
        "phrase_count": int(len(phrase_embeddings)),
        "phrase_embeddings": phrase_embeddings.tolist(),
        "seniority_anchor_embeddings": seniority_anchor_embeddings.tolist(),
        "technologies": technology_matches["technologies"],
        "technology_categories": technology_matches["categories"],
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def profile_is_current(profile: dict[str, Any]) -> bool:
    return (
        profile.get("profile_version") == PROFILE_VERSION
        and profile.get("model_name") == get_minilm_model_name()
        and profile.get("model_revision") == get_minilm_model_revision()
        and isinstance(profile.get("overall_embedding"), list)
        and isinstance(profile.get("word_embeddings"), list)
        and isinstance(profile.get("phrase_embeddings"), list)
        and isinstance(profile.get("seniority_anchor_embeddings"), list)
    )


def save_resume_profile(firebase_uid: str, profile: dict[str, Any]) -> None:
    from psycopg.types.json import Jsonb

    with psycopg.connect(get_database_url()) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO resume_profiles (firebase_uid, profile_data, created_at_utc, updated_at_utc)
            VALUES (%s, %s, now(), now())
            ON CONFLICT (firebase_uid) DO UPDATE SET
                profile_data = EXCLUDED.profile_data,
                updated_at_utc = now()
            """,
            (firebase_uid, Jsonb(profile)),
        )


def load_resume_profile(firebase_uid: str) -> dict[str, Any] | None:
    with psycopg.connect(get_database_url()) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT profile_data, updated_at_utc FROM resume_profiles WHERE firebase_uid = %s",
            (firebase_uid,),
        )
        row = cur.fetchone()
    if not row or not isinstance(row[0], dict):
        return None
    profile = dict(row[0])
    profile["updated_at"] = row[1].isoformat() if row[1] else None
    return profile


def delete_resume_profile(firebase_uid: str) -> bool:
    with psycopg.connect(get_database_url()) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM resume_profiles WHERE firebase_uid = %s", (firebase_uid,))
        return cur.rowcount > 0
