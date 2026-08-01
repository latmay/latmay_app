from __future__ import annotations

"""Shared model loading/configuration for the ranking web app."""

import os
from pathlib import Path
import time

from sentence_transformers import CrossEncoder, SentenceTransformer

try:
    from ranking_timing import record_ranking_timing
except ImportError:
    from webapp.ranking_timing import record_ranking_timing


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MINILM_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_MODEL_REVISION = "4bebbd56fc380a66525f95b03d4ec1a4b41a4f1e"
MODEL_CACHE_DIAGNOSTIC_MAX_FILES = int(os.environ.get("MODEL_CACHE_DIAGNOSTIC_MAX_FILES", "20"))


def get_model_cache_dir() -> Path:
    return Path(os.environ.get("MODEL_CACHE_DIR", BASE_DIR / "model_cache"))


def get_local_files_only() -> bool:
    return os.environ.get("LOCAL_FILES_ONLY", "false").lower() in {"1", "true", "yes"}


def get_minilm_model_name() -> str:
    return os.environ.get("MINILM_MODEL_NAME", DEFAULT_MINILM_MODEL_NAME)


def get_minilm_model_revision() -> str:
    return os.environ.get("MINILM_MODEL_REVISION", DEFAULT_MINILM_MODEL_REVISION).strip() or DEFAULT_MINILM_MODEL_REVISION


def get_cross_encoder_model_name() -> str:
    return os.environ.get("CROSS_ENCODER_MODEL_NAME", DEFAULT_CROSS_ENCODER_MODEL_NAME)


def get_cross_encoder_model_revision() -> str:
    return (
        os.environ.get("CROSS_ENCODER_MODEL_REVISION", DEFAULT_CROSS_ENCODER_MODEL_REVISION).strip()
        or DEFAULT_CROSS_ENCODER_MODEL_REVISION
    )


def assert_cache_available(cache_dir: Path) -> None:
    if get_local_files_only() and not cache_dir.exists():
        raise FileNotFoundError(
            f"MODEL_CACHE_DIR does not exist: {cache_dir}. "
            "Build from the ML base image so models are cached before runtime."
        )


def _cache_diagnostic_summary(cache_dir: Path, model_name: str) -> str:
    """
    Return safe cache metadata only: counts and filenames, never model contents.
    """
    if not cache_dir.exists():
        return "cache_exists=False"

    files = []
    dirs = []
    for child in cache_dir.rglob("*"):
        relative = child.relative_to(cache_dir).as_posix()
        if child.is_dir():
            dirs.append(relative)
        else:
            files.append(relative)

    model_token = "models--" + model_name.replace("/", "--")
    model_dirs = [name for name in dirs if model_token in name]
    model_files = [name for name in files if model_token in name]
    interesting_suffixes = (
        "config.json",
        "modules.json",
        "model.safetensors",
        "pytorch_model.bin",
        "sentence_bert_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    interesting = [
        name
        for name in model_files
        if name.endswith(interesting_suffixes)
    ][:MODEL_CACHE_DIAGNOSTIC_MAX_FILES]

    return (
        "cache_exists=True, "
        f"top_level_entries={len(list(cache_dir.iterdir()))}, "
        f"total_files={len(files)}, total_dirs={len(dirs)}, "
        f"model_dirs={len(model_dirs)}, model_files={len(model_files)}, "
        f"interesting_model_files={interesting}"
    )


def log_model_load_context(
    model_label: str,
    model_name: str,
    cache_dir: Path,
    *,
    revision: str = "",
) -> None:
    revision_text = revision or "default"
    print(
        "Model load context: "
        f"model_label={model_label}, "
        f"model_name={model_name}, "
        f"revision={revision_text}, "
        f"cache_dir={cache_dir}, "
        f"local_files_only={get_local_files_only()}, "
        f"{_cache_diagnostic_summary(cache_dir, model_name)}",
        flush=True,
    )


def load_minilm_model() -> SentenceTransformer:
    cache_dir = get_model_cache_dir()
    model_name = get_minilm_model_name()
    revision = get_minilm_model_revision()
    assert_cache_available(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "cache_folder": str(cache_dir),
        "local_files_only": get_local_files_only(),
    }
    if revision:
        kwargs["revision"] = revision

    log_model_load_context("minilm", model_name, cache_dir, revision=revision)
    started_at = time.perf_counter()
    try:
        model = SentenceTransformer(model_name, **kwargs)
    except Exception as exc:
        record_ranking_timing("model_load_minilm", started_at, status="failed", error_type=type(exc).__name__)
        print(
            "ALERT: model load failed: "
            f"model_label=minilm, error_type={type(exc).__name__}",
            flush=True,
        )
        log_model_load_context("minilm", model_name, cache_dir, revision=revision)
        raise
    record_ranking_timing("model_load_minilm", started_at, status="success")
    print("Model load complete: model_label=minilm", flush=True)
    return model


def load_cross_encoder_model() -> CrossEncoder:
    cache_dir = get_model_cache_dir()
    model_name = get_cross_encoder_model_name()
    revision = get_cross_encoder_model_revision()
    assert_cache_available(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "automodel_args": {"cache_dir": str(cache_dir)},
        "tokenizer_args": {"cache_dir": str(cache_dir)},
        "local_files_only": get_local_files_only(),
    }
    if revision:
        kwargs["revision"] = revision

    log_model_load_context("cross_encoder", model_name, cache_dir, revision=revision)
    started_at = time.perf_counter()
    try:
        model = CrossEncoder(model_name, **kwargs)
    except Exception as exc:
        record_ranking_timing("model_load_cross_encoder", started_at, status="failed", error_type=type(exc).__name__)
        print(
            "ALERT: model load failed: "
            f"model_label=cross_encoder, error_type={type(exc).__name__}",
            flush=True,
        )
        log_model_load_context("cross_encoder", model_name, cache_dir, revision=revision)
        raise
    record_ranking_timing("model_load_cross_encoder", started_at, status="success")
    print("Model load complete: model_label=cross_encoder", flush=True)
    return model
