from __future__ import annotations

"""Centralized ML model configuration for enrichment jobs.

Models are expected to be baked into the container image. When LOCAL_FILES_ONLY
is true, missing model files raise loudly instead of downloading at runtime.
"""

import os
from pathlib import Path
from typing import Any


DEFAULT_MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MINILM_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_MODEL_REVISION = "4bebbd56fc380a66525f95b03d4ec1a4b41a4f1e"
DEFAULT_ENRICHMENT_VERSION = "latmay-features-v1"


def get_model_cache_dir() -> Path:
    return Path(os.environ.get("MODEL_CACHE_DIR", "/app/model_cache"))


def get_local_files_only() -> bool:
    return os.environ.get("LOCAL_FILES_ONLY", "true").lower() in {"1", "true", "yes"}


def get_minilm_model_name() -> str:
    return os.environ.get("MINILM_MODEL_NAME", DEFAULT_MINILM_MODEL_NAME)


def get_cross_encoder_model_name() -> str:
    return os.environ.get("CROSS_ENCODER_MODEL_NAME", DEFAULT_CROSS_ENCODER_MODEL_NAME)


def get_minilm_model_revision() -> str:
    return os.environ.get("MINILM_MODEL_REVISION", DEFAULT_MINILM_MODEL_REVISION).strip() or DEFAULT_MINILM_MODEL_REVISION


def get_cross_encoder_model_revision() -> str:
    return (
        os.environ.get("CROSS_ENCODER_MODEL_REVISION", DEFAULT_CROSS_ENCODER_MODEL_REVISION).strip()
        or DEFAULT_CROSS_ENCODER_MODEL_REVISION
    )


def get_enrichment_version() -> str:
    return os.environ.get("ENRICHMENT_VERSION", DEFAULT_ENRICHMENT_VERSION)


def _model_kwargs() -> dict[str, Any]:
    cache_dir = get_model_cache_dir()
    if get_local_files_only() and not cache_dir.exists():
        raise FileNotFoundError(
            f"MODEL_CACHE_DIR does not exist: {cache_dir}. "
            "Build the ML base image so models are cached before runtime."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "cache_folder": str(cache_dir),
        "local_files_only": get_local_files_only(),
    }
    revision = get_minilm_model_revision()
    if revision:
        kwargs["revision"] = revision
    return kwargs


def load_minilm_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_minilm_model_name(), **_model_kwargs())


def load_cross_encoder_model():
    from sentence_transformers import CrossEncoder

    cache_dir = get_model_cache_dir()
    if get_local_files_only() and not cache_dir.exists():
        raise FileNotFoundError(
            f"MODEL_CACHE_DIR does not exist: {cache_dir}. "
            "Build the ML base image so models are cached before runtime."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "automodel_args": {"cache_dir": str(cache_dir)},
        "tokenizer_args": {"cache_dir": str(cache_dir)},
        "local_files_only": get_local_files_only(),
    }
    revision = get_cross_encoder_model_revision()
    if revision:
        kwargs["revision"] = revision
    return CrossEncoder(
        get_cross_encoder_model_name(),
        **kwargs,
    )
