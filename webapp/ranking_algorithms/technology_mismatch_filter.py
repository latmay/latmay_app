from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


OPERATION_NAME = "technology_mismatch_filter"
HIGHER_IS_BETTER = "higher_is_better"
DEFAULT_TECHNOLOGIES_PATH = Path(__file__).resolve().parent.parent / "webapp_data" / "technologies.md"
TECHNOLOGY_TERMS_PATH_ENV = "TECHNOLOGY_TERMS_MARKDOWN_PATH"


COMMON_ALIASES = {
    "C": ["C programming", "C language"],
    "R": ["R programming", "R language"],
    "Node.js": ["Nodejs", "Node JS"],
    "Next.js": ["Nextjs", "Next JS"],
    "Vue.js": ["Vuejs", "Vue JS"],
    "Nuxt.js": ["Nuxtjs", "Nuxt JS"],
    "AngularJS": ["Angular JS"],
    "NestJS": ["Nest JS"],
    "PostgreSQL": ["Postgres", "Postgre SQL"],
    "Google Cloud Platform": ["GCP", "Google Cloud"],
    "Google Cloud": ["GCP", "Google Cloud Platform"],
    "Amazon Web Services": ["AWS"],
    "AWS": ["Amazon Web Services"],
    "Microsoft Azure": ["Azure"],
    "Visual Studio Code": ["VS Code", "VSCode"],
    "Windows Subsystem for Linux": ["WSL"],
    "Large Language Model": ["LLM", "Large Language Models"],
    "OpenAI GPT": ["GPT", "ChatGPT"],
    "Meta Llama": ["Llama"],
    "Cloud Firestore": ["Firestore"],
    "Firebase Realtime Database": ["Realtime Database"],
    "Microsoft SQL Server": ["SQL Server", "MSSQL"],
    "Microsoft Access": ["MS Access"],
    "Amazon Redshift": ["Redshift"],
    "Amazon Bedrock": ["Bedrock"],
}


@dataclass(frozen=True)
class TechnologyTerm:
    category: str
    name: str
    aliases: tuple[str, ...]


def _normalize_search_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace(".js", " js")
    text = text.replace(".net", " dotnet")
    text = re.sub(r"(?<=\w)[/#-](?=\w)", " ", text)
    text = re.sub(r"[^a-z0-9+#]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return f" {text.strip()} "


def _alias_variants(term: str) -> set[str]:
    variants = {term, *COMMON_ALIASES.get(term, [])}
    expanded: set[str] = set()
    for value in variants:
        value = value.strip()
        if not value:
            continue
        expanded.add(value)
        expanded.add(value.replace(".", ""))
        expanded.add(value.replace(".", " "))
        expanded.add(value.replace("-", " "))
        expanded.add(value.replace("/", " "))
    return expanded


def _normalized_aliases(term: str) -> tuple[str, ...]:
    aliases: set[str] = set()
    for variant in _alias_variants(term):
        normalized = _normalize_search_text(variant).strip()
        compact = normalized.replace(" ", "")
        if normalized:
            aliases.add(normalized)
        if compact and compact != normalized:
            aliases.add(compact)

    # Avoid noisy one-letter language/platform false positives like "C", "R", or "X".
    aliases = {alias for alias in aliases if not re.fullmatch(r"[a-z0-9]", alias)}
    return tuple(sorted(aliases, key=lambda alias: (-len(alias), alias)))


def load_technology_terms(path: str | Path = DEFAULT_TECHNOLOGIES_PATH) -> list[TechnologyTerm]:
    path_override = os.environ.get(TECHNOLOGY_TERMS_PATH_ENV)
    markdown_path = Path(path_override) if path_override else Path(path)
    if not markdown_path.exists():
        data_dir = markdown_path.parent
        if data_dir.exists() and data_dir.is_dir():
            visible_files = sorted(child.name for child in data_dir.iterdir())
            data_dir_debug = f"{data_dir} exists with files={visible_files}"
        else:
            data_dir_debug = f"{data_dir} exists={data_dir.exists()} is_dir={data_dir.is_dir()}"
        raise FileNotFoundError(
            "Technology terms Markdown not found: "
            f"{markdown_path} "
            f"(cwd={Path.cwd()}, module_file={Path(__file__).resolve()}, "
            f"path_override_set={bool(path_override)}, {data_dir_debug})"
        )

    terms: list[TechnologyTerm] = []
    current_category: str | None = None
    for raw_line in markdown_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        heading = re.match(r"^\\?#\s+(.+)$", line)
        if heading:
            current_category = heading.group(1).replace("\\_", "_").strip()
            continue

        if current_category is None:
            continue

        term_name = line.lstrip("-*").strip().replace("\\_", "_")
        if not term_name:
            continue

        aliases = _normalized_aliases(term_name)
        if aliases:
            terms.append(TechnologyTerm(category=current_category, name=term_name, aliases=aliases))

    if not terms:
        raise ValueError(f"No technology terms loaded from {markdown_path}")

    return terms


def _make_unavailable_result(
    *,
    job_ids: Sequence[Any],
    enabled: bool,
    reason: str,
) -> dict[str, Any]:
    job_metrics = {
        job_id: {
            "rank": 1,
            "score": 1.0,
            "score_direction": HIGHER_IS_BETTER,
            "raw_metrics": {
                "technology_overlap_score": None,
                "technology_filter_enabled": enabled,
                "technology_filter_removed": False,
                "technology_filter_would_remove": False,
                "technology_filter_reason": reason,
                "job_technologies": [],
                "job_technology_categories": [],
                "resume_technologies": [],
                "resume_technology_categories": [],
                "technology_category_overlap": [],
                "job_type_count": 0,
                "overlap_type_count": 0,
            },
        }
        for job_id in job_ids
    }
    return {
        "operation_name": OPERATION_NAME,
        "status": "ok",
        "ranked_job_ids": list(job_ids),
        "job_metrics": job_metrics,
        "error": None,
    }


def match_technologies(text: str, terms: Sequence[TechnologyTerm]) -> dict[str, Any]:
    normalized_text = _normalize_search_text(text)
    matched_technologies: set[str] = set()
    matched_categories: set[str] = set()

    for term in terms:
        for alias in term.aliases:
            if " " in alias:
                matched = f" {alias} " in normalized_text
            else:
                matched = re.search(
                    rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                    normalized_text,
                ) is not None

            if matched:
                matched_technologies.add(term.name)
                matched_categories.add(term.category)
                break

    return {
        "technologies": sorted(matched_technologies),
        "categories": sorted(matched_categories),
    }


def run_technology_mismatch_filter(
    *,
    job_ids: Sequence[Any],
    job_descriptions: Sequence[str],
    resume_text: str,
    enabled: bool,
    terms_path: str | Path = DEFAULT_TECHNOLOGIES_PATH,
    min_job_types: int = 3,
    max_job_type_overlap_ratio: float = 0.333333,
    precomputed_resume_matches: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if min_job_types < 0:
        raise ValueError("min_job_types cannot be negative.")
    if not 0 <= max_job_type_overlap_ratio <= 1:
        raise ValueError("max_job_type_overlap_ratio must be in [0, 1].")

    try:
        terms = load_technology_terms(terms_path)
    except FileNotFoundError as exc:
        print(
            "ALERT: Technology mismatch analysis unavailable for this request: "
            f"error_type={type(exc).__name__}",
            flush=True,
        )
        return _make_unavailable_result(
            job_ids=job_ids,
            enabled=enabled,
            reason="technology_terms_markdown_missing",
        )

    resume_matches = precomputed_resume_matches or match_technologies(resume_text, terms)
    resume_categories = set(resume_matches["categories"])

    ranked_job_ids: list[Any] = []
    job_metrics: dict[Any, dict[str, Any]] = {}
    removed_count = 0

    for job_id in job_ids:
        job_index = int(job_id)
        job_matches = match_technologies(job_descriptions[job_index], terms)
        job_categories = set(job_matches["categories"])
        overlap_categories = sorted(job_categories & resume_categories)
        job_type_count = len(job_categories)
        overlap_type_count = len(overlap_categories)
        technology_overlap_score = (
            overlap_type_count / job_type_count
            if job_type_count > 0
            else 1.0
        )
        would_filter = (
            job_type_count >= min_job_types
            and technology_overlap_score <= max_job_type_overlap_ratio
        )
        is_filtered = enabled and would_filter
        if is_filtered:
            removed_count += 1

        if not enabled:
            print(
                "Technology mismatch diagnostic: "
                f"job_id={job_id}, "
                f"resume_type_count={len(resume_categories)}, "
                f"job_type_count={job_type_count}, "
                f"overlap_type_count={overlap_type_count}, "
                f"technology_overlap_score={technology_overlap_score:.6f}, "
                f"would_filter={would_filter}",
                flush=True,
            )

        score = 0.0 if is_filtered else 1.0
        ranked_job_ids.append(job_id)
        job_metrics[job_id] = {
            "rank": 2 if is_filtered else 1,
            "score": score,
            "score_direction": HIGHER_IS_BETTER,
            "raw_metrics": {
                "technology_overlap_score": float(technology_overlap_score),
                "technology_filter_enabled": enabled,
                "technology_filter_removed": is_filtered,
                "technology_filter_would_remove": would_filter,
                "technology_filter_reason": (
                    "filtered_low_category_overlap"
                    if is_filtered
                    else "would_filter_but_disabled"
                    if would_filter
                    else "passed"
                ),
                "job_technologies": job_matches["technologies"],
                "job_technology_categories": job_matches["categories"],
                "resume_technologies": resume_matches["technologies"],
                "resume_technology_categories": resume_matches["categories"],
                "technology_category_overlap": overlap_categories,
                "job_type_count": job_type_count,
                "overlap_type_count": overlap_type_count,
                "min_job_types": min_job_types,
                "max_job_type_overlap_ratio": max_job_type_overlap_ratio,
            },
        }

    print(
        "Technology mismatch filter complete: "
        f"enabled={enabled}, "
        f"jobs_checked={len(job_ids)}, "
        f"removed={removed_count}, "
        f"resume_categories={len(resume_categories)}",
        flush=True,
    )

    return {
        "operation_name": OPERATION_NAME,
        "status": "ok",
        "ranked_job_ids": ranked_job_ids,
        "job_metrics": job_metrics,
        "error": None,
    }
