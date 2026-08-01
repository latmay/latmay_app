from __future__ import annotations

import os
from pprint import pformat
from typing import Any, Sequence

from openai import OpenAI
from pydantic import BaseModel


OPERATION_NAME = "llm_bad_match_filter"
DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_REASONING_EFFORT = "minimal"

SYSTEM_PROMPT = """You are an early hiring screen.

Resume:
{resume}

Mark is_bad_match=true when the resume would likely be rejected early due to missing experience, skills, domain background, seniority, production experience, or similar requirements.

If the job appears to require substantially more direct professional experience, domain-specific experience, leadership experience, or production experience than the resume demonstrates, mark is_bad_match=true even if the candidate shows strong raw intelligence, academic ability, or adjacent technical skills.

Be conservative about experience gaps. Prefer false positives over false negatives when the resume lacks direct evidence of successfully performing similar work at the required level, scale, or domain.

Treat senior/staff/principal/manager roles as bad matches unless the resume clearly demonstrates comparable prior responsibility and experience.

Also mark is_bad_match=true when the job clearly makes poor use of the candidate’s core skills, technical strengths,
background, or demonstrated interests.

Mark is_bad_match=false when the candidate is not ruled out yet."""


class BadMatchDecision(BaseModel):
    is_bad_match: bool


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def clip(text: str, max_chars: int | None) -> str:
    return text if max_chars is None else text[:max_chars]


def user_prompt(title: str, requirements: str, max_chars: int | None) -> str:
    return (
        "JOB TITLE:\n"
        f"{clip(title, max_chars)}\n\n"
        "JOB REQUIREMENTS:\n"
        f"{clip(requirements, max_chars)}"
    )


def preview(value: Any, max_chars: int = 2000) -> str:
    text = pformat(value, width=120)
    return text if len(text) <= max_chars else text[:max_chars] + "...<truncated>"


def response_debug_dict(response: Any) -> dict[str, Any]:
    return {
        "status": getattr(response, "status", None),
        "error": getattr(response, "error", None),
        "incomplete_details": getattr(response, "incomplete_details", None),
        "usage": usage_dict(response),
    }


def usage_value(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    input_details = usage_value(usage, "input_tokens_details")
    output_details = usage_value(usage, "output_tokens_details")
    return {
        "input_tokens": usage_value(usage, "input_tokens"),
        "cached_input_tokens": usage_value(input_details, "cached_tokens"),
        "output_tokens": usage_value(usage, "output_tokens"),
        "reasoning_tokens": usage_value(output_details, "reasoning_tokens"),
        "total_tokens": usage_value(usage, "total_tokens"),
    }


def score_to_ranks(scores: dict[int, float]) -> dict[int, int]:
    ranks: dict[int, int] = {}
    previous_score: float | None = None
    previous_rank = 0

    for position, (job_id, score) in enumerate(
        sorted(scores.items(), key=lambda row: row[1], reverse=True),
        1,
    ):
        if score != previous_score:
            previous_score = score
            previous_rank = position
        ranks[job_id] = previous_rank

    return ranks


def result(
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


def decide_bad_match(
    *,
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    resume: str,
    title: str,
    requirements: str,
    max_chars: int | None,
    max_output_tokens: int,
) -> tuple[bool, dict[str, Any]]:
    response = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT.format(resume=clip(resume, max_chars))},
            {"role": "user", "content": user_prompt(title, requirements, max_chars)},
        ],
        text_format=BadMatchDecision,
    )
    if response.output_parsed is None:
        print("ALERT: llm_bad_match_filter unparsed OpenAI response:", flush=True)
        print(preview(response_debug_dict(response)), flush=True)
        raise ValueError("OpenAI response did not parse into BadMatchDecision.")
    return response.output_parsed.is_bad_match, usage_dict(response)


def run_llm_bad_match_filter(
    *,
    job_ids: Sequence[int],
    job_requirements: Sequence[str],
    resume_text: str,
    job_titles: Sequence[str] | None = None,
    enabled: bool | None = None,
    api_key: str | None = None,
    model: str | None = None,
    max_jobs: int | None = None,
    max_passed_jobs: int | None = None,
    timeout_seconds: int | None = None,
    max_input_chars: int | None = None,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    enabled = env_bool("ENABLE_LLM_BAD_MATCH_FILTER", False) if enabled is None else enabled
    if not enabled:
        return result("skipped", error="LLM bad match filter is disabled.")

    api_key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return result("skipped", error="OPENAI_API_KEY is not set.")

    model = model or os.environ.get("LLM_BAD_MATCH_MODEL", DEFAULT_MODEL).strip()
    reasoning_effort = reasoning_effort or os.environ.get(
        "LLM_BAD_MATCH_REASONING_EFFORT",
        DEFAULT_REASONING_EFFORT,
    ).strip() or DEFAULT_REASONING_EFFORT
    max_jobs = max_jobs if max_jobs is not None else env_int("LLM_BAD_MATCH_MAX_JOBS", 5)
    max_passed_jobs = (
        max_passed_jobs
        if max_passed_jobs is not None
        else env_int("LLM_BAD_MATCH_MAX_PASSED_JOBS", 2)
    )
    timeout_seconds = timeout_seconds if timeout_seconds is not None else env_int("LLM_BAD_MATCH_TIMEOUT_SECONDS", 20)
    max_input_chars = max_input_chars if max_input_chars is not None else env_int("LLM_BAD_MATCH_MAX_INPUT_CHARS", None)
    max_output_tokens = (
        max_output_tokens
        if max_output_tokens is not None
        else env_int("LLM_BAD_MATCH_MAX_OUTPUT_TOKENS", 64)
    )

    if max_jobs is None or max_passed_jobs is None or timeout_seconds is None or max_output_tokens is None:
        raise ValueError("LLM bad match numeric settings must be positive integers.")

    screened_ids = [int(job_id) for job_id in job_ids[:max_jobs]]
    client = OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)
    bad_match_by_job: dict[int, bool] = {}
    usage_by_job: dict[int, dict[str, Any]] = {}
    scores: dict[int, float] = {}
    accepted_job_ids: list[int] = []

    try:
        for job_id in screened_ids:
            is_bad_match, call_usage = decide_bad_match(
                client=client,
                model=model,
                reasoning_effort=reasoning_effort,
                resume=resume_text,
                title="" if job_titles is None else job_titles[job_id],
                requirements=job_requirements[job_id],
                max_chars=max_input_chars,
                max_output_tokens=max_output_tokens,
            )
            bad_match_by_job[job_id] = is_bad_match
            usage_by_job[job_id] = call_usage
            scores[job_id] = 0.0 if is_bad_match else 1.0
            print(
                "LLM bad match filter call: "
                f"job_id={job_id}, "
                f"is_bad_match={is_bad_match}, "
                f"usage={call_usage}",
                flush=True,
            )
            if not is_bad_match:
                accepted_job_ids.append(job_id)
                if len(accepted_job_ids) >= max_passed_jobs:
                    break
    except Exception as exc:
        print(
            "ALERT: llm_bad_match_filter failed: "
            f"error_type={type(exc).__name__}",
            flush=True,
        )
        return result("failed", error=f"{type(exc).__name__}: redacted")

    ranks = score_to_ranks(scores)
    ranked_job_ids = accepted_job_ids
    processed_ids = list(scores)
    bad_match_count = sum(1 for job_id in processed_ids if bad_match_by_job[job_id] is True)
    screened_pass_count = len(accepted_job_ids)
    total_usage = {
        key: sum((usage_by_job[job_id] or {}).get(key) or 0 for job_id in processed_ids)
        for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")
    }

    print(
        "LLM bad match filter complete: "
        f"api_calls={len(processed_ids)}, "
        f"screened={len(processed_ids)}, "
        f"bad_matches_filtered={bad_match_count}, "
        f"screened_passed={screened_pass_count}, "
        f"accepted_job_ids={accepted_job_ids}, "
        f"usage_totals={total_usage}",
        flush=True,
    )

    return result(
        "ok",
        ranked_job_ids=ranked_job_ids,
        job_metrics={
            job_id: {
                "rank": ranks[job_id],
                "score": score,
                "score_direction": "higher_is_better",
                "raw_metrics": {
                    "is_bad_match": bad_match_by_job[job_id],
                    "screened": True,
                    "openai_usage": usage_by_job[job_id],
                },
            }
            for job_id, score in scores.items()
        },
    )
