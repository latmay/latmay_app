from __future__ import annotations

import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

import main as web_main  # noqa: E402
from ranking_algorithms import llm_bad_match_filter as llm_filter  # noqa: E402
from ranking_algorithms import resume_phrase_coverage  # noqa: E402


class PrivacySafetyTests(unittest.TestCase):
    def test_rank_exception_does_not_log_or_render_resume_text(self) -> None:
        sensitive_resume = "Jane Candidate jane@example.com 555-1212 secret resume body"

        def boom(**_: object) -> dict[str, object]:
            raise ValueError(f"bad input: {sensitive_resume}")

        fake_ranking_service = types.SimpleNamespace(run_resume_ranking=boom)
        stdout = io.StringIO()

        with (
            patch.dict(sys.modules, {"ranking_service": fake_ranking_service}),
            web_main.app.test_client() as client,
            redirect_stdout(stdout),
        ):
            response = client.post("/rank", data={"resume_text": sensitive_resume})

        body = response.get_data(as_text=True)
        logs = stdout.getvalue()

        self.assertEqual(response.status_code, 500)
        self.assertNotIn(sensitive_resume, body)
        self.assertNotIn(sensitive_resume, logs)
        self.assertNotIn("jane@example.com", body)
        self.assertNotIn("jane@example.com", logs)
        self.assertIn("error_type=ValueError", logs)

    def test_openai_response_debug_dict_excludes_response_bodies(self) -> None:
        response = types.SimpleNamespace(
            status="incomplete",
            error="schema mismatch",
            incomplete_details={"reason": "max_output_tokens"},
            usage=None,
            output_text="Jane Candidate jane@example.com",
            output_parsed=None,
            output=[{"content": "Jane Candidate jane@example.com"}],
        )

        debug = llm_filter.response_debug_dict(response)

        self.assertNotIn("output_text", debug)
        self.assertNotIn("output_parsed", debug)
        self.assertNotIn("output", debug)

    def test_resume_phrase_coverage_metrics_do_not_store_user_phrases(self) -> None:
        user_phrase = "Jane Candidate private project"
        job_phrase = "private project"

        with patch.object(
            resume_phrase_coverage,
            "_embed_example_resume_phrases",
            return_value=(["example phrase"], [[0.0, 1.0]]),
        ):
            result = resume_phrase_coverage.run_resume_phrase_coverage_operation(
                job_ids=[0],
                job_titles=["Engineer"],
                job_companies=["Example"],
                resume_text=user_phrase,
                minilm_model=object(),
                precomputed_job_phrase_chunks=[[job_phrase]],
                precomputed_job_phrase_embeddings=[[[1.0, 0.0]]],
                precomputed_user_phrase_chunks=[user_phrase],
                precomputed_user_phrase_embeddings=[[1.0, 0.0]],
                resume_dataset_dir="unused",
                flag_percentile=100.0,
                job_flag_fraction=0.0,
            )

        strongest = result["job_metrics"][0]["raw_metrics"]["strongest_flagged_phrases"]

        self.assertTrue(strongest)
        self.assertNotIn("closest_user_resume_phrase", strongest[0])
        self.assertEqual(strongest[0]["closest_user_resume_phrase_index"], 0)
        self.assertNotIn(user_phrase, repr(result))


if __name__ == "__main__":
    unittest.main()
