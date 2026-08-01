from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

if "openai" not in sys.modules:
    fake_openai_module = types.ModuleType("openai")

    class _ImportOnlyOpenAI:
        def __init__(self, **_: object) -> None:
            pass

    fake_openai_module.OpenAI = _ImportOnlyOpenAI
    sys.modules["openai"] = fake_openai_module

if "pydantic" not in sys.modules:
    fake_pydantic_module = types.ModuleType("pydantic")

    class _ImportOnlyBaseModel:
        pass

    fake_pydantic_module.BaseModel = _ImportOnlyBaseModel
    sys.modules["pydantic"] = fake_pydantic_module

from ranking_algorithms import llm_bad_match_filter as llm_filter
import multi_stage_rankings as multi_stage


class FakeOpenAI:
    def __init__(self, **_: object) -> None:
        pass


def operation_result(
    name: str,
    job_ids: list[int],
    status: str = "ok",
    percent_flagged_by_job: dict[int, float] | None = None,
    bad_match_job_ids: set[int] | None = None,
) -> dict[str, object]:
    percent_flagged_by_job = percent_flagged_by_job or {}
    bad_match_job_ids = bad_match_job_ids or set()
    return {
        "operation_name": name,
        "status": status,
        "ranked_job_ids": list(job_ids),
        "job_metrics": {
            int(job_id): {
                "rank": rank,
                "score": 1.0,
                "score_direction": "higher_is_better",
                "raw_metrics": {
                    "percent_flagged": percent_flagged_by_job.get(int(job_id), 0.0),
                    "is_bad_match": int(job_id) in bad_match_job_ids,
                },
            }
            for rank, job_id in enumerate(job_ids, start=1)
        },
        "error": None,
    }


class LlmBadMatchFilterTests(unittest.TestCase):
    def run_filter_with_decisions(
        self,
        decisions: list[bool],
        *,
        max_passed_jobs: int | None = None,
    ) -> tuple[dict[str, object], list[int]]:
        calls: list[int] = []

        def fake_decide_bad_match(**kwargs: object) -> tuple[bool, dict[str, int]]:
            job_id = int(str(kwargs["requirements"]).split()[-1])
            calls.append(job_id)
            return decisions[len(calls) - 1], {"total_tokens": 1}

        with (
            patch.object(llm_filter, "OpenAI", FakeOpenAI),
            patch.object(llm_filter, "decide_bad_match", side_effect=fake_decide_bad_match),
        ):
            result = llm_filter.run_llm_bad_match_filter(
                job_ids=[0, 1, 2],
                job_titles=["title 0", "title 1", "title 2"],
                job_requirements=["requirements 0", "requirements 1", "requirements 2"],
                resume_text="resume",
                enabled=True,
                api_key="test-key",
                max_jobs=3,
                max_passed_jobs=max_passed_jobs,
                timeout_seconds=1,
                max_output_tokens=8,
            )

        return result, calls

    def test_stops_after_default_max_passed_jobs(self) -> None:
        result, calls = self.run_filter_with_decisions([False, True, False])

        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["ranked_job_ids"], [0, 2])

    def test_later_jobs_are_not_processed_after_max_passed_jobs(self) -> None:
        result, calls = self.run_filter_with_decisions([False, False, True])

        self.assertEqual(calls, [0, 1])
        self.assertEqual(result["ranked_job_ids"], [0, 1])

    def test_max_passed_jobs_can_be_one(self) -> None:
        result, calls = self.run_filter_with_decisions([False, False, True], max_passed_jobs=1)

        self.assertEqual(calls, [0])
        self.assertEqual(result["ranked_job_ids"], [0])

    def test_all_filtered_returns_no_accepted_jobs(self) -> None:
        result, calls = self.run_filter_with_decisions([True, True, True])

        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(result["ranked_job_ids"], [])
        self.assertTrue(
            all(
                metric["score"] == 0.0
                for metric in result["job_metrics"].values()  # type: ignore[union-attr]
            )
        )

    def test_disabled_llm_does_not_create_client_or_call_decider(self) -> None:
        with (
            patch.object(llm_filter, "OpenAI", side_effect=AssertionError("OpenAI should not be constructed")),
            patch.object(llm_filter, "decide_bad_match", side_effect=AssertionError("LLM should not be called")),
        ):
            result = llm_filter.run_llm_bad_match_filter(
                job_ids=[0, 1, 2],
                job_titles=["title 0", "title 1", "title 2"],
                job_requirements=["requirements 0", "requirements 1", "requirements 2"],
                resume_text="resume",
                enabled=False,
            )

        self.assertEqual(result["status"], "skipped")


class MultiStageLlmHandoffTests(unittest.TestCase):
    def run_mocked_pipeline(
        self,
        *,
        llm_enabled: bool,
        llm_status: str = "ok",
        llm_ranked_ids: list[int] | None = None,
        llm_bad_match_ids: set[int] | None = None,
        max_jobs: int = 4,
        return_operation_results: bool = False,
        all_candidates_through_all_metrics: bool = False,
    ) -> tuple[list[dict[str, object]] | dict[str, object], list[int]]:
        seen_llm_job_ids: list[int] = []
        llm_bad_match_ids = llm_bad_match_ids or set()

        def passthrough(name: str):
            return lambda **kwargs: operation_result(
                name,
                list(kwargs.get("job_ids") or kwargs.get("candidate_job_ids")),
            )

        def fake_llm(**kwargs: object) -> dict[str, object]:
            seen_llm_job_ids.extend(list(kwargs["job_ids"]))  # type: ignore[arg-type]
            if llm_enabled:
                ranked_ids = [seen_llm_job_ids[0]] if llm_ranked_ids is None else llm_ranked_ids
                llm_metric_ids = list(dict.fromkeys([*seen_llm_job_ids, *ranked_ids]))
                return operation_result(
                    "llm_bad_match_filter",
                    llm_metric_ids,
                    llm_status,
                    bad_match_job_ids=llm_bad_match_ids,
                ) | {"ranked_job_ids": ranked_ids}
            return operation_result("llm_bad_match_filter", seen_llm_job_ids, status="skipped")

        env_value = "true" if llm_enabled else "false"
        with (
            patch.dict(
                os.environ,
                {
                    "ENABLE_LLM_BAD_MATCH_FILTER": env_value,
                    "LLM_BAD_MATCH_MAX_JOBS": str(max_jobs),
                },
            ),
            patch.object(multi_stage, "run_word_wasserstein_operation", passthrough("word_sliced_wasserstein")),
            patch.object(multi_stage, "run_phrase_wasserstein_operation", passthrough("phrase_sliced_wasserstein")),
            patch.object(multi_stage, "run_cross_encoder_operation", passthrough("cross_encoder")),
            patch.object(multi_stage, "build_mahalanobis_candidate_job_ids", return_value=[0, 1, 2, 3, 4]),
            patch.object(
                multi_stage,
                "run_mahalanobis_operation",
                lambda **_: operation_result("mahalanobis_outlier", [4, 3, 2, 1, 0]),
            ),
            patch.object(
                multi_stage,
                "run_multi_metric_bad_fit_operation",
                passthrough("multi_metric_bad_fit_filter"),
            ),
            patch.object(
                multi_stage,
                "run_technology_mismatch_operation",
                passthrough("technology_mismatch_filter"),
            ),
            patch.object(
                multi_stage,
                "run_resume_phrase_coverage_ranking_operation",
                lambda **_: operation_result(
                    "resume_phrase_coverage",
                    [2, 0, 4, 1, 3],
                    percent_flagged_by_job={
                        0: 0.60,
                        1: 0.10,
                        2: 0.35,
                        3: 0.80,
                        4: 0.45,
                    },
                ),
            ),
            patch.object(multi_stage, "run_llm_bad_match_filter", fake_llm),
        ):
            rows = multi_stage.rank_jobs_multi_stage(
                resume_text="resume text",
                job_descriptions=["job 0", "job 1", "job 2", "job 3", "job 4"],
                minilm_model=object(),
                cross_encoder_model=None,
                word_keep_n=5,
                phrase_keep_n=5,
                poor_match_max_rank_per_step=1 if all_candidates_through_all_metrics else 5,
                mahalanobis_remove_bottom_fraction=0.0,
                multi_metric_bad_fit_bottom_fraction=0.0,
                return_operation_results=return_operation_results,
                all_candidates_through_all_metrics=all_candidates_through_all_metrics,
            )

        return rows, seen_llm_job_ids

    def test_all_candidates_mode_bypasses_inter_stage_reductions(self) -> None:
        output, seen_llm_job_ids = self.run_mocked_pipeline(
            llm_enabled=False,
            return_operation_results=True,
            all_candidates_through_all_metrics=True,
        )

        self.assertEqual(seen_llm_job_ids, [0, 1, 2, 3, 4])
        results_by_name = {
            result["operation_name"]: result
            for result in output["operation_results"]
        }
        for operation_name in (
            "word_sliced_wasserstein",
            "phrase_sliced_wasserstein",
            "mahalanobis_outlier",
            "multi_metric_bad_fit_filter",
            "technology_mismatch_filter",
        ):
            self.assertEqual(set(results_by_name[operation_name]["job_metrics"]), {0, 1, 2, 3, 4})

    def test_pipeline_sends_up_to_max_jobs_sorted_by_percent_flagged_to_enabled_llm(self) -> None:
        rows, seen_llm_job_ids = self.run_mocked_pipeline(llm_enabled=True, max_jobs=4)

        self.assertEqual(seen_llm_job_ids, [3, 0, 4, 2])
        self.assertEqual([row["job_index"] for row in rows], [3])

    def test_pipeline_uses_available_jobs_when_less_than_llm_max(self) -> None:
        rows, seen_llm_job_ids = self.run_mocked_pipeline(llm_enabled=True, max_jobs=10)

        self.assertEqual(seen_llm_job_ids, [3, 0, 4, 2, 1])
        self.assertEqual([row["job_index"] for row in rows], [3])

    def test_pipeline_returns_no_jobs_when_llm_rejects_all_screened_jobs(self) -> None:
        rows, seen_llm_job_ids = self.run_mocked_pipeline(
            llm_enabled=True,
            llm_status="ok",
            llm_ranked_ids=[],
            max_jobs=3,
        )

        self.assertEqual(seen_llm_job_ids, [3, 0, 4])
        self.assertEqual(rows, [])

    def test_pre_llm_results_exclude_jobs_rejected_by_llm(self) -> None:
        output, seen_llm_job_ids = self.run_mocked_pipeline(
            llm_enabled=True,
            llm_ranked_ids=[3],
            llm_bad_match_ids={0, 4},
            max_jobs=3,
            return_operation_results=True,
        )

        self.assertEqual(seen_llm_job_ids, [3, 0, 4])
        self.assertEqual([row["job_index"] for row in output["results"]], [3])  # type: ignore[index]
        self.assertEqual(
            [row["job_index"] for row in output["pre_llm_results"]],  # type: ignore[index]
            [3, 2, 1],
        )

    def test_pipeline_preserves_disabled_llm_coverage_order(self) -> None:
        rows, seen_llm_job_ids = self.run_mocked_pipeline(llm_enabled=False)

        self.assertEqual(seen_llm_job_ids, [2, 0, 4, 1, 3])
        self.assertEqual([row["job_index"] for row in rows], [2, 0, 4, 1, 3])

    def test_pipeline_skips_word_wasserstein_when_disabled(self) -> None:
        mahalanobis_operation_results: list[list[dict[str, object]]] = []
        multi_metric_operation_results: list[list[dict[str, object]]] = []

        def passthrough(name: str):
            return lambda **kwargs: operation_result(
                name,
                list(kwargs.get("job_ids") or kwargs.get("candidate_job_ids")),
            )

        def fake_mahalanobis(**kwargs: object) -> dict[str, object]:
            operation_results = list(kwargs["operation_results"])  # type: ignore[arg-type]
            mahalanobis_operation_results.append(operation_results)
            return operation_result("mahalanobis_outlier", list(kwargs["candidate_job_ids"]))  # type: ignore[arg-type]

        def fake_multi_metric(**kwargs: object) -> dict[str, object]:
            operation_results = list(kwargs["operation_results"])  # type: ignore[arg-type]
            multi_metric_operation_results.append(operation_results)
            return operation_result("multi_metric_bad_fit_filter", list(kwargs["job_ids"]))  # type: ignore[arg-type]

        with (
            patch.dict(
                os.environ,
                {
                    "ENABLE_LLM_BAD_MATCH_FILTER": "false",
                    "ENABLE_WORD_WASSERSTEIN": "false",
                },
            ),
            patch.object(
                multi_stage,
                "run_word_wasserstein_operation",
                side_effect=AssertionError("word Wasserstein should be skipped"),
            ),
            patch.object(multi_stage, "run_phrase_wasserstein_operation", passthrough("phrase_sliced_wasserstein")),
            patch.object(multi_stage, "run_cross_encoder_operation", passthrough("cross_encoder")),
            patch.object(multi_stage, "build_mahalanobis_candidate_job_ids", return_value=[0, 1, 2]),
            patch.object(multi_stage, "run_mahalanobis_operation", fake_mahalanobis),
            patch.object(multi_stage, "run_multi_metric_bad_fit_operation", fake_multi_metric),
            patch.object(
                multi_stage,
                "run_technology_mismatch_operation",
                passthrough("technology_mismatch_filter"),
            ),
            patch.object(
                multi_stage,
                "run_resume_phrase_coverage_ranking_operation",
                lambda **_: operation_result("resume_phrase_coverage", [2, 0, 1]),
            ),
            patch.object(
                multi_stage,
                "run_resume_phrase_job_coverage_ranking_operation",
                lambda **_: operation_result("resume_phrase_job_coverage", [2, 0, 1]),
            ),
            patch.object(
                multi_stage,
                "run_llm_bad_match_filter",
                lambda **kwargs: operation_result("llm_bad_match_filter", list(kwargs["job_ids"]), status="skipped"),
            ),
        ):
            output = multi_stage.rank_jobs_multi_stage(
                resume_text="resume text",
                job_descriptions=["job 0", "job 1", "job 2"],
                minilm_model=object(),
                cross_encoder_model=None,
                word_keep_n=3,
                phrase_keep_n=3,
                poor_match_max_rank_per_step=3,
                mahalanobis_remove_bottom_fraction=0.0,
                multi_metric_bad_fit_bottom_fraction=0.0,
                return_operation_results=True,
            )

        word_result = next(
            result
            for result in output["operation_results"]  # type: ignore[index]
            if result["operation_name"] == "word_sliced_wasserstein"
        )
        self.assertEqual(word_result["status"], "skipped")
        self.assertTrue(mahalanobis_operation_results)
        self.assertTrue(multi_metric_operation_results)
        self.assertEqual(
            [
                result["status"]
                for result in mahalanobis_operation_results[0]
                if result["operation_name"] == "word_sliced_wasserstein"
            ],
            ["skipped"],
        )
        self.assertEqual(
            [
                result["status"]
                for result in multi_metric_operation_results[0]
                if result["operation_name"] == "word_sliced_wasserstein"
            ],
            ["skipped"],
        )
        self.assertTrue(
            all(
                row.get("stage1_word_wasserstein_distance") is None
                for row in output["results"]  # type: ignore[index]
            )
        )

    def test_pipeline_skips_cross_encoder_when_disabled(self) -> None:
        mahalanobis_models: list[object] = []

        def passthrough(name: str):
            return lambda **kwargs: operation_result(
                name,
                list(kwargs.get("job_ids") or kwargs.get("candidate_job_ids")),
            )

        def fake_mahalanobis(**kwargs: object) -> dict[str, object]:
            mahalanobis_models.append(kwargs["cross_encoder_model"])
            return operation_result("mahalanobis_outlier", list(kwargs["candidate_job_ids"]))  # type: ignore[arg-type]

        with (
            patch.dict(os.environ, {"ENABLE_LLM_BAD_MATCH_FILTER": "false"}),
            patch.object(multi_stage, "run_word_wasserstein_operation", passthrough("word_sliced_wasserstein")),
            patch.object(multi_stage, "run_phrase_wasserstein_operation", passthrough("phrase_sliced_wasserstein")),
            patch.object(
                multi_stage,
                "run_cross_encoder_operation",
                side_effect=AssertionError("cross-encoder should be skipped"),
            ),
            patch.object(multi_stage, "build_mahalanobis_candidate_job_ids", return_value=[0, 1, 2]),
            patch.object(multi_stage, "run_mahalanobis_operation", fake_mahalanobis),
            patch.object(
                multi_stage,
                "run_multi_metric_bad_fit_operation",
                passthrough("multi_metric_bad_fit_filter"),
            ),
            patch.object(
                multi_stage,
                "run_technology_mismatch_operation",
                passthrough("technology_mismatch_filter"),
            ),
            patch.object(
                multi_stage,
                "run_resume_phrase_coverage_ranking_operation",
                lambda **_: operation_result("resume_phrase_coverage", [2, 0, 1]),
            ),
            patch.object(
                multi_stage,
                "run_llm_bad_match_filter",
                lambda **kwargs: operation_result("llm_bad_match_filter", list(kwargs["job_ids"]), status="skipped"),
            ),
        ):
            rows = multi_stage.rank_jobs_multi_stage(
                resume_text="resume text",
                job_descriptions=["job 0", "job 1", "job 2"],
                minilm_model=object(),
                cross_encoder_model=object(),
                enable_cross_encoder=False,
                word_keep_n=3,
                phrase_keep_n=3,
                poor_match_max_rank_per_step=3,
                mahalanobis_remove_bottom_fraction=0.0,
                multi_metric_bad_fit_bottom_fraction=0.0,
            )

        self.assertEqual(mahalanobis_models, [None])
        self.assertEqual([row["job_index"] for row in rows], [2, 0, 1])


if __name__ == "__main__":
    unittest.main()
