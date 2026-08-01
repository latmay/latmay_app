from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


WEBAPP_DIR = Path(__file__).resolve().parents[1]
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

from model_loader import (  # noqa: E402
    get_minilm_model_name,
    get_minilm_model_revision,
    load_cross_encoder_model,
    load_minilm_model,
)
from multi_stage_rankings import rank_jobs_multi_stage  # noqa: E402
from ranking_algorithms.mahalanobis_outlier_ranking import rank_mahalanobis_outliers  # noqa: E402
from ranking_algorithms.multi_metric_bad_fit_filter import run_multi_metric_bad_fit_filter  # noqa: E402
from ranking_algorithms.phrases_wasserstein_rankings import embed_texts, phrase_chunks  # noqa: E402
from ranking_algorithms.words_wasserstein_rankings import preprocess_job_descriptions  # noqa: E402


DEFAULT_DATA_DIR = WEBAPP_DIR / "webapp_data" / "test_data"
WORD_CUSTOM_STOPWORDS = {"preferred", "required", "qualification", "qualifications"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _pack_matrices(matrices: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    nonempty = [np.asarray(matrix, dtype=np.float32) for matrix in matrices]
    width = next((matrix.shape[1] for matrix in nonempty if matrix.ndim == 2 and len(matrix)), 0)
    for matrix in nonempty:
        offsets.append(offsets[-1] + len(matrix))
    flat = np.vstack([matrix for matrix in nonempty if len(matrix)]) if offsets[-1] else np.empty((0, width), np.float32)
    return flat.astype(np.float32), np.asarray(offsets, dtype=np.int64)


def _unpack_matrices(flat: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(flat[offsets[i]:offsets[i + 1]], dtype=np.float32) for i in range(len(offsets) - 1)]


def _encode_with_progress(
    model: Any,
    texts: Sequence[str],
    *,
    batch_size: int,
    stage: str,
    normalize_embeddings: bool = False,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batches.append(np.asarray(model.encode(
            texts[start:end], batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=normalize_embeddings,
        ), dtype=np.float32))
        print(
            f"[harness] {stage}: embedded items {start + 1}-{end}/{total}; "
            f"{total - end} remain.",
            flush=True,
        )
    return np.vstack(batches).astype(np.float32) if batches else np.empty((0, 0), np.float32)


def _load_jobs(data_dir: Path) -> tuple[list[str], list[dict[str, Any]]]:
    manifest = _read_json(data_dir / "jobs" / "manifest.json")
    jobs: list[dict[str, Any]] = []
    ids: list[str] = []
    for entry in manifest["jobs"]:
        jobs.append(_read_json(data_dir / "jobs" / entry["file"]))
        ids.append(str(entry["job_id"]))
    return ids, jobs


def _load_or_build_job_embeddings(
    *, data_dir: Path, job_paths: Sequence[Path], descriptions: Sequence[str],
    titles: Sequence[str], requirements: Sequence[str], model: Any,
) -> tuple[list[list[str]], list[np.ndarray], list[list[str]], list[np.ndarray], np.ndarray]:
    cache_dir = data_dir / "generated" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "job_embeddings_metadata.json"
    arrays_path = cache_dir / "job_embeddings.npz"
    expected = {
        "dataset_sha256": _dataset_hash(job_paths),
        "model_name": get_minilm_model_name(),
        "model_revision": get_minilm_model_revision(),
        "job_count": len(descriptions),
        "cache_version": 1,
    }
    if metadata_path.exists() and arrays_path.exists() and _read_json(metadata_path) == expected:
        print(f"[harness] Loading cached job embeddings for {len(descriptions)} jobs.", flush=True)
        arrays = np.load(arrays_path, allow_pickle=False)
        word_lists = _read_json(cache_dir / "job_word_lists.json")
        phrase_lists = _read_json(cache_dir / "job_phrase_chunks.json")
        return (
            word_lists,
            _unpack_matrices(arrays["word_embeddings"], arrays["word_offsets"]),
            phrase_lists,
            _unpack_matrices(arrays["phrase_embeddings"], arrays["phrase_offsets"]),
            np.asarray(arrays["title_requirements_embeddings"], dtype=np.float32),
        )

    print(f"[harness] No current embedding cache; embedding {len(descriptions)} jobs (3 stages).", flush=True)
    word_lists = preprocess_job_descriptions(
        descriptions, custom_stopwords=WORD_CUSTOM_STOPWORDS, use_stopword_filter=True,
        use_frequent_word_filter=False, max_count=4, use_tfidf_filter=True,
        top_tfidf_fraction=0.25, max_words_per_job=50,
    )
    for index, words in enumerate(word_lists, start=1):
        print(
            f"[harness] Word stage: prepared job {index}/{len(word_lists)}; "
            f"{len(word_lists) - index} jobs remain; selected_words={len(words)}.",
            flush=True,
        )
    unique_words = sorted({word for words in word_lists for word in words})
    encoded_words = _encode_with_progress(
        model, unique_words, batch_size=128, stage="Word stage",
    )
    lookup = dict(zip(unique_words, encoded_words, strict=True))
    word_matrices = [np.vstack([lookup[word] for word in words]).astype(np.float32) if words else np.empty((0, 0), np.float32) for words in word_lists]
    print("[harness] Embedding progress: word embeddings complete; 2 stages remain.", flush=True)
    phrase_lists: list[list[str]] = []
    phrase_matrices: list[np.ndarray] = []
    for index, text in enumerate(descriptions, start=1):
        chunks = phrase_chunks(text, min_words=3, max_words=24)
        phrase_lists.append(chunks)
        phrase_matrices.append(embed_texts(model, chunks, batch_size=64))
        print(
            f"[harness] Phrase stage: embedded job {index}/{len(descriptions)}; "
            f"{len(descriptions) - index} jobs remain; phrase_chunks={len(chunks)}.",
            flush=True,
        )
    print("[harness] Embedding progress: phrase embeddings complete; 1 stage remains.", flush=True)
    title_requirement_texts = [f"Job title: {title}\nRequirements: {requirement}".strip() for title, requirement in zip(titles, requirements, strict=True)]
    title_embeddings = _encode_with_progress(
        model, title_requirement_texts, batch_size=128,
        stage="Title/requirements stage", normalize_embeddings=True,
    )
    word_flat, word_offsets = _pack_matrices(word_matrices)
    phrase_flat, phrase_offsets = _pack_matrices(phrase_matrices)
    np.savez_compressed(arrays_path, word_embeddings=word_flat, word_offsets=word_offsets,
                        phrase_embeddings=phrase_flat, phrase_offsets=phrase_offsets,
                        title_requirements_embeddings=title_embeddings)
    (cache_dir / "job_word_lists.json").write_text(json.dumps(word_lists), encoding="utf-8")
    (cache_dir / "job_phrase_chunks.json").write_text(json.dumps(phrase_lists), encoding="utf-8")
    metadata_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    print(f"[harness] Embedding cache saved to {cache_dir}.", flush=True)
    return word_lists, word_matrices, phrase_lists, phrase_matrices, title_embeddings


def _metric_rows(resume_id: str, job_ids: Sequence[str], jobs: Sequence[dict[str, Any]], output: dict[str, Any]) -> list[dict[str, Any]]:
    operation_results = output["operation_results"]
    rows: list[dict[str, Any]] = []
    for index, (job_id, job) in enumerate(zip(job_ids, jobs, strict=True)):
        row: dict[str, Any] = {"resume_id": resume_id, "job_id": job_id, "source_job_id": job.get("id", ""),
                               "title": job.get("title", ""), "company_name": job.get("company_name", "")}
        for result in operation_results:
            name = str(result.get("operation_name", "unknown"))
            metric = (result.get("job_metrics") or {}).get(index) or {}
            row[f"{name}__status"] = result.get("status", "")
            row[f"{name}__rank"] = metric.get("rank", "")
            row[f"{name}__score"] = metric.get("score", "")
            for key, value in (metric.get("raw_metrics") or {}).items():
                row[f"{name}__{key}"] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        rows.append(row)
    return rows


def _write_checkpoint_state(path: Path, state: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _append_checkpoint_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _requirements_embedding_result(distances: np.ndarray) -> dict[str, Any]:
    ranked_ids = np.argsort(distances).tolist()
    return {
        "operation_name": "requirements_embedding",
        "status": "ok",
        "ranked_job_ids": ranked_ids,
        "job_metrics": {
            int(job_id): {
                "rank": rank,
                "score": float(distances[job_id]),
                "score_direction": "lower_is_better",
                "raw_metrics": {"requirements_embedding_distance": float(distances[job_id])},
            }
            for rank, job_id in enumerate(ranked_ids, start=1)
        },
        "error": None,
    }


def _add_operation_to_rows(rows: Sequence[dict[str, Any]], result: dict[str, Any]) -> None:
    name = str(result["operation_name"])
    metrics = result.get("job_metrics") or {}
    for job_index, row in enumerate(rows):
        metric = metrics.get(job_index) or {}
        row[f"{name}__status"] = result.get("status", "")
        row[f"{name}__rank"] = metric.get("rank", "")
        row[f"{name}__score"] = metric.get("score", "")
        for key, value in (metric.get("raw_metrics") or {}).items():
            row[f"{name}__{key}"] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value


def _saved_source_results(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    specifications = (
        ("word_sliced_wasserstein", "word_wasserstein_distance"),
        ("phrase_sliced_wasserstein", "phrase_wasserstein_distance"),
        ("cross_encoder", "cross_encoder_score"),
    )
    results: list[dict[str, Any]] = []
    for operation_name, metric_name in specifications:
        job_metrics: dict[int, dict[str, Any]] = {}
        for job_index, row in enumerate(rows):
            value = row.get(f"{operation_name}__{metric_name}")
            if value not in (None, ""):
                job_metrics[job_index] = {"raw_metrics": {metric_name: float(value)}}
        if job_metrics:
            results.append({"operation_name": operation_name, "status": "ok", "job_metrics": job_metrics})
    return results


def _write_rows_as_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary_path.replace(path)


def backfill_requirements_metric(data_dir: Path, source_path: Path | None = None) -> Path:
    results_dir = data_dir / "results"
    if source_path is None:
        candidates = sorted(results_dir.glob("all_pair_metrics_*.jsonl"), key=lambda path: path.stat().st_mtime)
        candidates = [path for path in candidates if "checkpoint" not in path.name and "enriched" not in path.name]
        if not candidates:
            raise FileNotFoundError(f"No completed all-pair JSONL result found under {results_dir}.")
        source_path = candidates[-1]
    source_path = source_path.resolve()
    rows = _checkpoint_rows(source_path)
    resume_manifest = _read_json(data_dir / "resumes" / "manifest.json")
    job_ids, _ = _load_jobs(data_dir)
    expected_rows = len(resume_manifest) * len(job_ids)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} saved pairing rows; found {len(rows)} in {source_path}.")

    arrays_path = data_dir / "generated" / "embeddings" / "job_embeddings.npz"
    if not arrays_path.exists():
        raise FileNotFoundError(f"Cached job embeddings are required for backfill: {arrays_path}")
    arrays = np.load(arrays_path, allow_pickle=False)
    job_embeddings = np.asarray(arrays["title_requirements_embeddings"], dtype=np.float32)
    if len(job_embeddings) != len(job_ids):
        raise ValueError("Cached title/requirements embeddings do not match the test job count.")

    model = load_minilm_model()
    rows_by_resume = {str(entry["resume_id"]): [] for entry in resume_manifest}
    for row in rows:
        rows_by_resume.setdefault(str(row.get("resume_id")), []).append(row)
    for resume_number, entry in enumerate(resume_manifest, start=1):
        resume_id = str(entry["resume_id"])
        resume_rows = rows_by_resume[resume_id]
        if len(resume_rows) != len(job_ids):
            raise ValueError(f"Resume {resume_id} has {len(resume_rows)} rows; expected {len(job_ids)}.")
        print(f"[harness] Backfill resume {resume_number}/{len(resume_manifest)} ({resume_id}); recomputing 3 metric groups.", flush=True)
        resume_text = (data_dir / "resumes" / entry["file"]).read_text(encoding="utf-8")
        resume_embedding = np.asarray(model.encode(
            [resume_text], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True,
        )[0], dtype=np.float32)
        distances = 1.0 - np.clip(job_embeddings @ resume_embedding, -1.0, 1.0)
        requirements_result = _requirements_embedding_result(distances)
        _add_operation_to_rows(resume_rows, requirements_result)
        source_results = _saved_source_results(resume_rows)
        scoring_mode = str(resume_rows[0].get("mahalanobis_outlier__mahalanobis_scoring_mode") or "distance_outlier")
        mahalanobis_result = rank_mahalanobis_outliers(
            candidate_job_ids=list(range(len(job_ids))), operation_results=source_results,
            requirements_cluster_distances=distances.tolist(), scoring_mode=scoring_mode,
        )
        _add_operation_to_rows(resume_rows, mahalanobis_result)
        bad_fit_result = run_multi_metric_bad_fit_filter(
            job_ids=list(range(len(job_ids))),
            operation_results=[*source_results, mahalanobis_result],
            requirements_cluster_distances=distances.tolist(), bottom_fraction=0.25,
        )
        _add_operation_to_rows(resume_rows, bad_fit_result)
        print(f"[harness] Backfill resume {resume_number}/{len(resume_manifest)} complete; {len(resume_manifest) - resume_number} remain.", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_jsonl = results_dir / f"all_pair_metrics_enriched_{stamp}.jsonl"
    _append_checkpoint_rows(output_jsonl, rows)
    output_csv = results_dir / f"all_pair_metrics_enriched_{stamp}.csv"
    _write_rows_as_csv(output_csv, rows)
    print(f"[harness] Backfill complete. Wrote enriched metrics to {output_csv}.", flush=True)
    return output_csv


def run_harness(data_dir: Path, *, enable_cross_encoder: bool) -> Path:
    if os.environ.get("ENABLE_LLM_BAD_MATCH_FILTER", "false").lower() in {"1", "true", "yes", "on"}:
        print("[harness] Paid LLM filter is disabled for test-harness runs.", flush=True)
    os.environ["ENABLE_LLM_BAD_MATCH_FILTER"] = "false"
    resume_manifest = _read_json(data_dir / "resumes" / "manifest.json")
    job_ids, jobs = _load_jobs(data_dir)
    job_paths = [data_dir / "jobs" / entry["file"] for entry in _read_json(data_dir / "jobs" / "manifest.json")["jobs"]]
    descriptions = [str(job.get("content_text") or job.get("extracted_requirements") or "") for job in jobs]
    requirements = [str(job.get("extracted_requirements") or job.get("content_text") or "") for job in jobs]
    titles = [str(job.get("title") or "") for job in jobs]
    companies = [str(job.get("company_name") or "") for job in jobs]
    min_yoe = [job.get("min_years_experience") or None for job in jobs]
    total_pairs = len(resume_manifest) * len(jobs)
    results_dir = data_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    resume_paths = [data_dir / "resumes" / entry["file"] for entry in resume_manifest]
    run_identity = {
        "jobs_sha256": _dataset_hash(job_paths),
        "resumes_sha256": _dataset_hash(resume_paths),
        "model_name": get_minilm_model_name(),
        "model_revision": get_minilm_model_revision(),
        "cross_encoder_enabled": enable_cross_encoder,
        "checkpoint_version": 2,
    }
    run_key = hashlib.sha256(json.dumps(run_identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    checkpoint_path = results_dir / f"all_pair_metrics_checkpoint_{run_key}.jsonl"
    state_path = results_dir / f"all_pair_metrics_checkpoint_{run_key}.state.json"
    state = _read_json(state_path) if state_path.exists() else {**run_identity, "completed_resume_ids": []}
    checkpoint_rows = _checkpoint_rows(checkpoint_path) if checkpoint_path.exists() else []
    row_counts: dict[str, int] = {}
    for row in checkpoint_rows:
        row_resume_id = str(row.get("resume_id") or "")
        row_counts[row_resume_id] = row_counts.get(row_resume_id, 0) + 1
    completed_resume_ids = {
        resume_id for resume_id, count in row_counts.items() if count == len(jobs)
    }
    incomplete_resume_ids = set(row_counts) - completed_resume_ids
    if incomplete_resume_ids:
        print(
            f"[harness] Discarding incomplete checkpoint rows for resumes: "
            f"{', '.join(sorted(incomplete_resume_ids))}.",
            flush=True,
        )
        checkpoint_rows = [
            row for row in checkpoint_rows
            if str(row.get("resume_id") or "") in completed_resume_ids
        ]
        temporary_checkpoint = checkpoint_path.with_suffix(".jsonl.tmp")
        with temporary_checkpoint.open("w", encoding="utf-8") as handle:
            for row in checkpoint_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_checkpoint.replace(checkpoint_path)
    state["completed_resume_ids"] = sorted(completed_resume_ids)
    _write_checkpoint_state(state_path, state)
    print(f"[harness] Test set: {len(resume_manifest)} resumes x {len(jobs)} jobs = {total_pairs} pairings.", flush=True)
    if completed_resume_ids:
        print(
            f"[harness] Resuming checkpoint: {len(completed_resume_ids)}/{len(resume_manifest)} "
            f"resumes already stored; {len(resume_manifest) - len(completed_resume_ids)} remain.",
            flush=True,
        )
    model = load_minilm_model()
    word_lists, word_embeddings, phrase_lists, phrase_embeddings, title_embeddings = _load_or_build_job_embeddings(
        data_dir=data_dir, job_paths=job_paths, descriptions=descriptions, titles=titles,
        requirements=requirements, model=model,
    )
    cross_encoder = load_cross_encoder_model() if enable_cross_encoder else None
    started = time.perf_counter()
    resumes_processed_this_run = 0
    for resume_number, entry in enumerate(resume_manifest, start=1):
        resume_id = str(entry["resume_id"])
        if resume_id in completed_resume_ids:
            print(f"[harness] Resume {resume_number}/{len(resume_manifest)} ({resume_id}) already checkpointed; skipping.", flush=True)
            continue
        completed_pairs = len(completed_resume_ids) * len(jobs)
        print(
            f"[harness] Resume {resume_number}/{len(resume_manifest)} ({resume_id}); "
            f"{len(resume_manifest) - resume_number} resumes remain after this one; "
            f"{total_pairs - completed_pairs} pairings remain.",
            flush=True,
        )
        resume_text = (data_dir / "resumes" / entry["file"]).read_text(encoding="utf-8")
        resume_embedding = np.asarray(model.encode(
            [resume_text], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True,
        )[0], dtype=np.float32)
        requirements_distances = 1.0 - np.clip(title_embeddings @ resume_embedding, -1.0, 1.0)
        output = rank_jobs_multi_stage(
            resume_text=resume_text,
            job_descriptions=descriptions, job_titles=titles, job_requirements=requirements,
            job_min_years_experience=min_yoe, job_companies=companies, minilm_model=model,
            cross_encoder_model=cross_encoder, enable_cross_encoder=enable_cross_encoder,
            job_ids=list(range(len(jobs))), word_custom_stopwords=WORD_CUSTOM_STOPWORDS,
            word_use_frequent_word_filter=False, word_top_tfidf_fraction=0.25,
            word_max_words_per_job=50, precomputed_job_word_lists=word_lists,
            precomputed_job_word_embeddings=word_embeddings,
            precomputed_job_phrase_chunks=phrase_lists,
            precomputed_job_phrase_embeddings=phrase_embeddings,
            precomputed_title_requirements_embeddings=title_embeddings,
            resume_phrase_job_coverage_comparison_job_ids=list(range(len(jobs))),
            resume_phrase_job_coverage_comparison_phrase_chunks=phrase_lists,
            resume_phrase_job_coverage_comparison_phrase_embeddings=phrase_embeddings,
            requirements_cluster_distances=requirements_distances.tolist(),
            all_candidates_through_all_metrics=True, return_operation_results=True,
        )
        output["operation_results"].append(_requirements_embedding_result(requirements_distances))
        resume_rows = _metric_rows(resume_id, job_ids, jobs, output)
        _append_checkpoint_rows(checkpoint_path, resume_rows)
        completed_resume_ids.add(resume_id)
        state["completed_resume_ids"] = sorted(completed_resume_ids)
        _write_checkpoint_state(state_path, state)
        resumes_processed_this_run += 1
        elapsed = time.perf_counter() - started
        completed_pairs = len(completed_resume_ids) * len(jobs)
        remaining_resumes = len(resume_manifest) - len(completed_resume_ids)
        remaining_seconds = elapsed / resumes_processed_this_run * remaining_resumes
        print(
            f"[harness] Checkpoint saved: {checkpoint_path.name}; "
            f"completed {completed_pairs}/{total_pairs} pairings ({completed_pairs / total_pairs:.1%}); "
            f"elapsed={elapsed:.1f}s, estimated_remaining={remaining_seconds:.1f}s.",
            flush=True,
        )

    all_rows = _checkpoint_rows(checkpoint_path)
    if len(all_rows) != total_pairs:
        raise RuntimeError(f"Checkpoint has {len(all_rows)} rows; expected {total_pairs} before finalization.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = results_dir / f"all_pair_metrics_{stamp}.csv"
    temporary_output_path = output_path.with_suffix(".csv.tmp")
    fieldnames = list(dict.fromkeys(key for row in all_rows for key in row))
    with temporary_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary_output_path.replace(output_path)
    archived_checkpoint = results_dir / f"all_pair_metrics_{stamp}.jsonl"
    checkpoint_path.replace(archived_checkpoint)
    state_path.replace(results_dir / f"all_pair_metrics_{stamp}.state.json")
    print(
        f"[harness] Complete. Wrote {len(all_rows)} pairing rows to {output_path}; "
        f"archived checkpoint to {archived_checkpoint}.",
        flush=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every test resume through every local ranking metric.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--disable-cross-encoder", action="store_true", help="Skip the slower local cross-encoder metric.")
    parser.add_argument(
        "--backfill-requirements-metric", action="store_true",
        help="Enrich saved results with requirements distance and consistent derived metrics without rerunning the pipeline.",
    )
    parser.add_argument("--source-result", type=Path, help="Completed JSONL to enrich; defaults to the latest completed harness JSONL.")
    args = parser.parse_args()
    if args.backfill_requirements_metric:
        backfill_requirements_metric(args.data_dir.resolve(), args.source_result)
    else:
        if args.source_result is not None:
            parser.error("--source-result requires --backfill-requirements-metric.")
        run_harness(args.data_dir.resolve(), enable_cross_encoder=not args.disable_cross_encoder)


if __name__ == "__main__":
    main()
