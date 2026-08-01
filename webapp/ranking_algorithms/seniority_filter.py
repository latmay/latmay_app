from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


OPERATION_NAME = "seniority_filter"
LOWER_IS_BETTER = "lower_is_better"

LEVEL_LABELS: tuple[str, ...] = (
    "entry",
    "junior",
    "mid",
    "senior",
    "lead",
    "principal/manager",
)

LEVEL_PHRASES: tuple[str, ...] = (
    "Entry level: little or no professional experience. Learns basic tools and workflows. "
    "Assists others and works on clearly defined tasks with close guidance.",
    "Junior: 0 to 2 years of experience. Handles small tasks, fixes bugs, follows established processes, "
    "and usually needs guidance for larger or ambiguous work.",
    "Mid-level: 2 to 5 years of experience. Independently completes projects, uses standard tools well, "
    "solves routine problems, and contributes to implementation decisions.",
    "Senior: 5 to 8 years of experience. Owns complex projects, handles ambiguity, makes technical decisions, "
    "improves processes, and may mentor less experienced teammates.",
    "Lead: 8 or more years of experience. Leads projects across multiple people or teams, coordinates work, "
    "reviews others' work, and helps set technical or operational direction.",
    "Principal or manager: 10 or more years of experience. Owns strategy, leads major systems or teams, "
    "influences roadmaps, and makes organization-level decisions.",
)

LEVEL_NUMERIC_SCORES = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.float32)

TITLE_SENIORITY_FLOORS: tuple[tuple[re.Pattern[str], float, str], ...] = (
    (re.compile(r"\b(?:principal|director|head|vp|vice president|manager)\b", re.IGNORECASE), 5.0, "principal/manager"),
    (re.compile(r"\b(?:lead|staff)\b", re.IGNORECASE), 4.0, "lead"),
    (re.compile(r"\b(?:senior|sr\.?)\b", re.IGNORECASE), 3.0, "senior"),
    (re.compile(r"\b(?:mid|mid-level|intermediate)\b", re.IGNORECASE), 2.0, "mid"),
    (re.compile(r"\b(?:junior|jr\.?)\b", re.IGNORECASE), 1.0, "junior"),
    (re.compile(r"\b(?:entry|entry-level|intern|internship)\b", re.IGNORECASE), 0.0, "entry"),
)

TITLE_SENIORITY_CEILINGS: tuple[tuple[re.Pattern[str], float, str], ...] = (
    (re.compile(r"\b(?:intern|internship)\b", re.IGNORECASE), 0.0, "entry"),
)

YOE_SENIORITY_FLOORS: tuple[tuple[float, float, str], ...] = (
    (10.0, 5.0, "principal/manager"),
    (8.0, 4.0, "lead"),
    (5.0, 3.0, "senior"),
    (3.0, 2.0, "mid"),
    (1.0, 1.0, "junior"),
    (0.0, 0.0, "entry"),
)


def _result(
    status: str,
    *,
    ranked_job_ids: list[int] | None = None,
    job_metrics: dict[int, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "operation_name": OPERATION_NAME,
        "status": status,
        "ranked_job_ids": ranked_job_ids or [],
        "job_metrics": job_metrics or {},
        "error": error,
    }


def _as_2d_float_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")
    return array


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def _embed_texts(
    texts: Sequence[str],
    minilm_model: "SentenceTransformer",
    *,
    batch_size: int,
) -> np.ndarray:
    return _as_2d_float_array(
        minilm_model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    )


def _similarity_distribution(similarities: np.ndarray, *, alpha: float) -> np.ndarray:
    sims = np.asarray(similarities, dtype=np.float32)
    min_sim = float(np.min(sims))
    if min_sim < 0:
        sims = sims - min_sim

    total = float(np.sum(sims))
    if total <= 0:
        probs = np.ones_like(sims, dtype=np.float32) / len(sims)
    else:
        probs = sims / total

    sharpened = probs ** alpha
    sharpened_total = float(np.sum(sharpened))
    if sharpened_total <= 0:
        return np.ones_like(probs, dtype=np.float32) / len(probs)

    return sharpened / sharpened_total


def _closest_label_from_score(score: float) -> str:
    idx = int(np.clip(round(score), 0, len(LEVEL_LABELS) - 1))
    return LEVEL_LABELS[idx]


def _title_seniority_floor(title: str) -> tuple[float | None, str | None]:
    normalized_title = str(title or "")
    for pattern, floor, label in TITLE_SENIORITY_FLOORS:
        if pattern.search(normalized_title):
            return floor, label
    return None, None


def _title_seniority_ceiling(title: str) -> tuple[float | None, str | None]:
    normalized_title = str(title or "")
    for pattern, ceiling, label in TITLE_SENIORITY_CEILINGS:
        if pattern.search(normalized_title):
            return ceiling, label
    return None, None


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def _yoe_seniority_floor(min_years_experience: Any) -> tuple[float | None, str | None]:
    yoe = _coerce_optional_float(min_years_experience)
    if yoe is None:
        return None, None
    for threshold, floor, label in YOE_SENIORITY_FLOORS:
        if yoe >= threshold:
            return floor, label
    return None, None


def _score_embedding(
    embedding: np.ndarray,
    anchor_embeddings: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, float, str, np.ndarray, np.ndarray]:
    normalized = _normalize_rows(np.asarray([embedding], dtype=np.float32))[0]
    similarities = anchor_embeddings @ normalized
    probabilities = _similarity_distribution(similarities, alpha=alpha)
    score = float(np.sum(probabilities * LEVEL_NUMERIC_SCORES))
    variance = float(np.sum(probabilities * (LEVEL_NUMERIC_SCORES - score) ** 2))
    std = float(np.sqrt(variance))
    return score, std, _closest_label_from_score(score), probabilities, similarities


def _job_texts(
    job_ids: Sequence[int],
    *,
    job_titles: Sequence[str] | None,
    job_requirements: Sequence[str] | None,
) -> list[str]:
    texts: list[str] = []
    for job_id in job_ids:
        idx = int(job_id)
        title = str(job_titles[idx]) if job_titles is not None and idx < len(job_titles) else ""
        requirements = (
            str(job_requirements[idx])
            if job_requirements is not None and idx < len(job_requirements)
            else ""
        )
        texts.append(f"Job title: {title}\nRequirements: {requirements}".strip())
    return texts


def _select_precomputed_embeddings(
    job_ids: Sequence[int],
    precomputed_title_requirements_embeddings: Any,
) -> np.ndarray:
    embeddings = _as_2d_float_array(precomputed_title_requirements_embeddings)
    int_job_ids = [int(job_id) for job_id in job_ids]

    if len(embeddings) == len(int_job_ids):
        return embeddings

    max_job_id = max(int_job_ids, default=-1)
    if max_job_id >= len(embeddings):
        raise ValueError("precomputed_title_requirements_embeddings do not cover the requested job IDs.")

    return embeddings[int_job_ids]


def _rank_with_ties(rows: Sequence[tuple[int, float]]) -> dict[int, int]:
    ranks: dict[int, int] = {}
    previous_score: float | None = None
    previous_rank = 0

    for position, (job_id, score) in enumerate(sorted(rows, key=lambda row: (row[1], row[0])), start=1):
        if previous_score is None or score != previous_score:
            previous_score = score
            previous_rank = position
        ranks[job_id] = previous_rank

    return ranks


def run_seniority_filter(
    job_ids: Sequence[int],
    resume_text: str,
    minilm_model: "SentenceTransformer",
    job_titles: Sequence[str] | None = None,
    job_requirements: Sequence[str] | None = None,
    job_min_years_experience: Sequence[Any] | None = None,
    precomputed_title_requirements_embeddings: Any = None,
    max_gap: float = 1.5,
    max_junior_gap: float = 10.0,
    enabled: bool = True,
    level_probability_alpha: float = 3.0,
    batch_size: int = 64,
    precomputed_resume_embedding: Any = None,
    precomputed_anchor_embeddings: Any = None,
) -> dict[str, Any]:
    if not enabled:
        return _result(
            "skipped",
            ranked_job_ids=[int(job_id) for job_id in job_ids],
            error="Seniority filter is disabled.",
        )

    try:
        if level_probability_alpha <= 0:
            raise ValueError("level_probability_alpha must be positive.")
        if max_junior_gap < 0:
            raise ValueError("max_junior_gap must be non-negative.")

        int_job_ids = [int(job_id) for job_id in job_ids]
        if not int_job_ids:
            return _result("ok", ranked_job_ids=[], job_metrics={})
        if precomputed_resume_embedding is None and (not isinstance(resume_text, str) or not resume_text.strip()):
            raise ValueError("resume_text must be a non-empty string.")

        anchor_embeddings = _normalize_rows(
            np.asarray(precomputed_anchor_embeddings, dtype=np.float32)
            if precomputed_anchor_embeddings is not None
            else _embed_texts(LEVEL_PHRASES, minilm_model, batch_size=batch_size)
        )
        resume_embedding = (
            np.asarray(precomputed_resume_embedding, dtype=np.float32)
            if precomputed_resume_embedding is not None
            else _embed_texts([resume_text], minilm_model, batch_size=batch_size)[0]
        )
        (
            resume_score,
            resume_std,
            resume_label,
            resume_probabilities,
            resume_similarities,
        ) = _score_embedding(
            resume_embedding,
            anchor_embeddings,
            alpha=level_probability_alpha,
        )

        if precomputed_title_requirements_embeddings is not None:
            job_embeddings = _select_precomputed_embeddings(
                int_job_ids,
                precomputed_title_requirements_embeddings,
            )
        else:
            job_embeddings = _embed_texts(
                _job_texts(
                    int_job_ids,
                    job_titles=job_titles,
                    job_requirements=job_requirements,
                ),
                minilm_model,
                batch_size=batch_size,
            )

        job_embeddings = _normalize_rows(job_embeddings)
        score_rows: list[tuple[int, float]] = []
        job_score_by_id: dict[int, float] = {}
        job_std_by_id: dict[int, float] = {}
        job_label_by_id: dict[int, str] = {}
        job_raw_score_by_id: dict[int, float] = {}
        job_raw_label_by_id: dict[int, str] = {}
        job_title_floor_by_id: dict[int, float | None] = {}
        job_title_floor_label_by_id: dict[int, str | None] = {}
        job_title_floor_applied_by_id: dict[int, bool] = {}
        job_title_ceiling_by_id: dict[int, float | None] = {}
        job_title_ceiling_label_by_id: dict[int, str | None] = {}
        job_title_ceiling_applied_by_id: dict[int, bool] = {}
        job_yoe_floor_by_id: dict[int, float | None] = {}
        job_yoe_floor_label_by_id: dict[int, str | None] = {}
        job_yoe_floor_applied_by_id: dict[int, bool] = {}
        job_probabilities_by_id: dict[int, list[float]] = {}
        job_similarities_by_id: dict[int, list[float]] = {}
        filtered_by_id: dict[int, bool] = {}
        too_senior_by_id: dict[int, bool] = {}
        too_junior_by_id: dict[int, bool] = {}

        for offset, job_id in enumerate(int_job_ids):
            (
                job_score,
                job_std,
                job_label,
                job_probabilities,
                job_similarities,
            ) = _score_embedding(
                job_embeddings[offset],
                anchor_embeddings,
                alpha=level_probability_alpha,
            )
            title = str(job_titles[job_id]) if job_titles is not None and job_id < len(job_titles) else ""
            title_floor, title_floor_label = _title_seniority_floor(title)
            title_ceiling, title_ceiling_label = _title_seniority_ceiling(title)
            min_yoe = (
                job_min_years_experience[job_id]
                if job_min_years_experience is not None and job_id < len(job_min_years_experience)
                else None
            )
            yoe_floor, yoe_floor_label = _yoe_seniority_floor(min_yoe)
            raw_job_score = float(job_score)
            raw_job_label = job_label
            title_floor_applied = title_floor is not None and job_score < title_floor
            yoe_floor_applied = yoe_floor is not None and job_score < yoe_floor
            effective_floor = max(
                (floor for floor in (title_floor, yoe_floor) if floor is not None),
                default=None,
            )
            if effective_floor is not None and job_score < effective_floor:
                job_score = float(effective_floor)
                job_label = _closest_label_from_score(job_score)
            title_ceiling_applied = title_ceiling is not None and job_score > title_ceiling
            if title_ceiling is not None and job_score > title_ceiling:
                job_score = float(title_ceiling)
                job_label = _closest_label_from_score(job_score)
            gap = float(job_score - resume_score)
            is_too_senior = gap > max_gap
            is_too_junior = gap < -max_junior_gap
            job_score_by_id[job_id] = float(job_score)
            job_std_by_id[job_id] = float(job_std)
            job_label_by_id[job_id] = job_label
            job_raw_score_by_id[job_id] = raw_job_score
            job_raw_label_by_id[job_id] = raw_job_label
            job_title_floor_by_id[job_id] = title_floor
            job_title_floor_label_by_id[job_id] = title_floor_label
            job_title_floor_applied_by_id[job_id] = bool(title_floor_applied)
            job_title_ceiling_by_id[job_id] = title_ceiling
            job_title_ceiling_label_by_id[job_id] = title_ceiling_label
            job_title_ceiling_applied_by_id[job_id] = bool(title_ceiling_applied)
            job_yoe_floor_by_id[job_id] = yoe_floor
            job_yoe_floor_label_by_id[job_id] = yoe_floor_label
            job_yoe_floor_applied_by_id[job_id] = bool(yoe_floor_applied)
            job_probabilities_by_id[job_id] = [float(value) for value in job_probabilities]
            job_similarities_by_id[job_id] = [float(value) for value in job_similarities]
            too_senior_by_id[job_id] = bool(is_too_senior)
            too_junior_by_id[job_id] = bool(is_too_junior)
            filtered_by_id[job_id] = bool(is_too_senior or is_too_junior)
            score_rows.append((job_id, gap))

        ranks = _rank_with_ties(score_rows)
        ranked_job_ids = [job_id for job_id, _ in sorted(score_rows, key=lambda row: (row[1], row[0]))]
        job_metrics = {
            job_id: {
                "rank": ranks[job_id],
                "score": float(job_score_by_id[job_id] - resume_score),
                "score_direction": LOWER_IS_BETTER,
                "raw_metrics": {
                    "resume_seniority_score": float(resume_score),
                    "resume_seniority_std": float(resume_std),
                    "resume_seniority_label": resume_label,
                    "resume_level_probability_distribution": [
                        float(value) for value in resume_probabilities
                    ],
                    "resume_level_similarities": [
                        float(value) for value in resume_similarities
                    ],
                    "job_seniority_score": float(job_score_by_id[job_id]),
                    "job_raw_seniority_score": float(job_raw_score_by_id[job_id]),
                    "job_seniority_std": float(job_std_by_id[job_id]),
                    "job_seniority_label": job_label_by_id[job_id],
                    "job_raw_seniority_label": job_raw_label_by_id[job_id],
                    "job_title_seniority_floor": job_title_floor_by_id[job_id],
                    "job_title_seniority_floor_label": job_title_floor_label_by_id[job_id],
                    "job_title_seniority_floor_applied": job_title_floor_applied_by_id[job_id],
                    "job_title_seniority_ceiling": job_title_ceiling_by_id[job_id],
                    "job_title_seniority_ceiling_label": job_title_ceiling_label_by_id[job_id],
                    "job_title_seniority_ceiling_applied": job_title_ceiling_applied_by_id[job_id],
                    "job_yoe_seniority_floor": job_yoe_floor_by_id[job_id],
                    "job_yoe_seniority_floor_label": job_yoe_floor_label_by_id[job_id],
                    "job_yoe_seniority_floor_applied": job_yoe_floor_applied_by_id[job_id],
                    "job_level_probability_distribution": job_probabilities_by_id[job_id],
                    "job_level_similarities": job_similarities_by_id[job_id],
                    "seniority_gap": float(job_score_by_id[job_id] - resume_score),
                    "seniority_abs_gap": abs(float(job_score_by_id[job_id] - resume_score)),
                    "seniority_filter_max_gap": float(max_gap),
                    "seniority_filter_max_junior_gap": float(max_junior_gap),
                    "seniority_filter_level_probability_alpha": float(level_probability_alpha),
                    "is_too_senior": too_senior_by_id[job_id],
                    "is_too_junior": too_junior_by_id[job_id],
                    "is_filtered": bool(filtered_by_id[job_id]),
                },
            }
            for job_id in ranked_job_ids
        }

        return _result("ok", ranked_job_ids=ranked_job_ids, job_metrics=job_metrics)
    except Exception as exc:
        print(
            f"ALERT: {OPERATION_NAME} failed: error_type={type(exc).__name__}",
            flush=True,
        )
        return _result("failed", error=f"{type(exc).__name__}: redacted")
