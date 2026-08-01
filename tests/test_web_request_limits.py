from __future__ import annotations

import sys
import types
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

from flask import g


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))

import main as web_main  # noqa: E402


class WebRequestLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        web_main.app.config["TESTING"] = True
        web_main.app.config["RATELIMIT_ENABLED"] = True
        web_main.limiter.reset()

    def test_request_body_size_limit_rejects_before_ranking(self) -> None:
        run_resume_ranking = Mock()
        fake_ranking_service = types.SimpleNamespace(run_resume_ranking=run_resume_ranking)
        oversized_resume = "x" * (web_main.MAX_REQUEST_BYTES + 1)

        with (
            patch.dict(sys.modules, {"ranking_service": fake_ranking_service}),
            web_main.app.test_client() as client,
        ):
            response = client.post(
                "/rank",
                data={"resume_text": oversized_resume},
                environ_overrides={"REMOTE_ADDR": "203.0.113.10"},
            )

        self.assertEqual(response.status_code, 413)
        self.assertIn("That request is too large", response.get_data(as_text=True))
        run_resume_ranking.assert_not_called()

    def test_resume_length_limit_rejects_before_ranking(self) -> None:
        run_resume_ranking = Mock()
        fake_ranking_service = types.SimpleNamespace(run_resume_ranking=run_resume_ranking)
        long_resume = "x" * (web_main.MAX_RESUME_CHARS + 1)

        with (
            patch.dict(sys.modules, {"ranking_service": fake_ranking_service}),
            web_main.app.test_client() as client,
        ):
            response = client.post(
                "/rank",
                data={"resume_text": long_resume},
                environ_overrides={"REMOTE_ADDR": "203.0.113.11"},
            )

        self.assertEqual(response.status_code, 413)
        self.assertIn("Resume text is too long", response.get_data(as_text=True))
        run_resume_ranking.assert_not_called()

    def test_rank_endpoint_rate_limit(self) -> None:
        with web_main.app.test_client() as client:
            responses = [
                client.post(
                    "/rank",
                    data={"resume_text": ""},
                    environ_overrides={"REMOTE_ADDR": "203.0.113.12"},
                )
                for _ in range(11)
            ]

        self.assertEqual([response.status_code for response in responses[:10]], [400] * 10)
        self.assertEqual(responses[10].status_code, 429)
        self.assertIn("Too many ranking requests", responses[10].get_data(as_text=True))

    def test_environment_configuration_is_applied(self) -> None:
        with patch.dict("os.environ", {"MAX_REQUEST_BYTES": "12345"}):
            self.assertEqual(web_main.int_from_env("MAX_REQUEST_BYTES", 1), 12345)

        self.assertEqual(web_main.app.config["MAX_CONTENT_LENGTH"], web_main.MAX_REQUEST_BYTES)
        self.assertGreater(web_main.MAX_REQUEST_BYTES, web_main.MAX_RESUME_CHARS)
        self.assertTrue(web_main.RANK_RATE_LIMIT)

    def test_valid_rank_request_still_runs_ranking(self) -> None:
        payload = {"results": [], "grouped_results": [], "total_jobs": 0, "filtered_jobs": 0}
        run_resume_ranking = Mock(return_value=payload)
        fake_ranking_service = types.SimpleNamespace(run_resume_ranking=run_resume_ranking)

        with (
            patch.dict(sys.modules, {"ranking_service": fake_ranking_service}),
            patch.object(web_main, "render_template", return_value="ok"),
            web_main.app.test_client() as client,
        ):
            response = client.post(
                "/rank",
                data={"resume_text": "Python engineer resume"},
                environ_overrides={"REMOTE_ADDR": "203.0.113.13"},
            )

        self.assertEqual(response.status_code, 200)
        run_resume_ranking.assert_called_once()
        self.assertFalse(run_resume_ranking.call_args.kwargs["require_recent_posted"])

    def test_saved_resume_rank_uses_verified_users_profile_without_resume_text(self) -> None:
        import resume_profiles

        payload = {"results": [], "grouped_results": [], "total_jobs": 0, "filtered_jobs": 0}
        run_resume_ranking = Mock(return_value=payload)
        fake_ranking_service = types.SimpleNamespace(run_resume_ranking=run_resume_ranking)
        cached_profile = {
            "profile_version": resume_profiles.PROFILE_VERSION,
            "model_name": resume_profiles.get_minilm_model_name(),
            "model_revision": resume_profiles.get_minilm_model_revision(),
            "overall_embedding": [[1.0]],
            "word_embeddings": [[1.0]],
            "phrase_embeddings": [[1.0]],
            "seniority_anchor_embeddings": [[1.0]],
        }

        with (
            patch.dict(sys.modules, {"ranking_service": fake_ranking_service}),
            patch.object(web_main, "verify_firebase_id_token", return_value={"uid": "user-123"}),
            patch.object(resume_profiles, "load_resume_profile", return_value=cached_profile) as load_profile,
            patch.object(web_main, "render_template", return_value="ok"),
            web_main.app.test_client() as client,
        ):
            response = client.post(
                "/rank",
                data={"use_saved_resume": "on"},
                headers={"Authorization": "Bearer valid-token"},
            )

        self.assertEqual(response.status_code, 200)
        load_profile.assert_called_once_with("user-123")
        run_resume_ranking.assert_called_once()
        self.assertEqual(run_resume_ranking.call_args.kwargs["resume_text"], "")
        self.assertIs(run_resume_ranking.call_args.kwargs["precomputed_resume_profile"], cached_profile)

    def test_anonymous_request_has_no_firebase_identity(self) -> None:
        with web_main.app.test_request_context("/"):
            response = web_main.establish_firebase_identity()

            self.assertIsNone(response)
            self.assertIsNone(g.firebase_uid)
            self.assertIsNone(g.firebase_user)

    def test_valid_bearer_token_attaches_firebase_identity(self) -> None:
        decoded_token = {"uid": "firebase-user-123", "email": "user@example.com"}
        with (
            patch.object(web_main, "verify_firebase_id_token", return_value=decoded_token) as verify_token,
            web_main.app.test_request_context(
                "/rank",
                headers={"Authorization": "Bearer valid-token"},
            ),
        ):
            response = web_main.establish_firebase_identity()

            self.assertIsNone(response)
            self.assertEqual(g.firebase_uid, "firebase-user-123")
            self.assertEqual(g.firebase_user, decoded_token)
            verify_token.assert_called_once_with("valid-token")

    def test_invalid_bearer_token_is_rejected(self) -> None:
        with (
            patch.object(web_main, "verify_firebase_id_token", side_effect=ValueError("invalid")),
            web_main.app.test_request_context(
                "/rank",
                headers={"Authorization": "Bearer invalid-token"},
            ),
        ):
            response, status_code = web_main.establish_firebase_identity()

            self.assertEqual(status_code, 401)
            self.assertIn("Invalid or expired", response.get_json()["error"])
            self.assertIsNone(g.firebase_uid)

    def test_delete_account_removes_profile_then_firebase_user(self) -> None:
        import resume_profiles

        calls: list[str] = []
        with (
            patch.object(
                web_main,
                "verify_firebase_id_token",
                return_value={"uid": "user-123", "auth_time": 950},
            ),
            patch.object(web_main.time, "time", return_value=1000),
            patch.object(
                resume_profiles,
                "delete_resume_profile",
                side_effect=lambda uid: calls.append(f"profile:{uid}"),
            ),
            patch.object(
                web_main,
                "delete_firebase_user",
                side_effect=lambda uid: calls.append(f"firebase:{uid}"),
            ),
            web_main.app.test_client() as client,
        ):
            response = client.delete("/account", headers={"Authorization": "Bearer valid-token"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"deleted": True})
        self.assertEqual(calls, ["profile:user-123", "firebase:user-123"])

    def test_delete_account_requires_recent_authentication(self) -> None:
        import resume_profiles

        with (
            patch.object(
                web_main,
                "verify_firebase_id_token",
                return_value={"uid": "user-123", "auth_time": 1},
            ),
            patch.object(web_main.time, "time", return_value=1000),
            patch.object(resume_profiles, "delete_resume_profile") as delete_profile,
            patch.object(web_main, "delete_firebase_user") as delete_user,
            web_main.app.test_client() as client,
        ):
            response = client.delete("/account", headers={"Authorization": "Bearer valid-token"})

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json()["code"], "recent_login_required")
        delete_profile.assert_not_called()
        delete_user.assert_not_called()

    def test_delete_account_stops_if_profile_deletion_fails(self) -> None:
        import resume_profiles

        with (
            patch.object(
                web_main,
                "verify_firebase_id_token",
                return_value={"uid": "user-123", "auth_time": 950},
            ),
            patch.object(web_main.time, "time", return_value=1000),
            patch.object(
                resume_profiles,
                "delete_resume_profile",
                side_effect=RuntimeError("database unavailable"),
            ),
            patch.object(web_main, "delete_firebase_user") as delete_user,
            web_main.app.test_client() as client,
        ):
            response = client.delete("/account", headers={"Authorization": "Bearer valid-token"})

        self.assertEqual(response.status_code, 500)
        self.assertIn("account was not deleted", response.get_json()["error"])
        delete_user.assert_not_called()

    def test_rank_endpoint_accepts_resume_upload(self) -> None:
        payload = {"results": [], "grouped_results": [], "total_jobs": 0, "filtered_jobs": 0}
        run_resume_ranking = Mock(return_value=payload)
        fake_ranking_service = types.SimpleNamespace(run_resume_ranking=run_resume_ranking)

        with (
            patch.dict(sys.modules, {"ranking_service": fake_ranking_service}),
            patch.object(web_main, "extract_docx_text", return_value="Uploaded resume text"),
            patch.object(web_main, "render_template", return_value="ok"),
            web_main.app.test_client() as client,
        ):
            response = client.post(
                "/rank",
                data={"resume_file": (BytesIO(b"docx test"), "resume.docx")},
                content_type="multipart/form-data",
                environ_overrides={"REMOTE_ADDR": "203.0.113.15"},
            )

        self.assertEqual(response.status_code, 200)
        run_resume_ranking.assert_called_once()
        self.assertEqual(run_resume_ranking.call_args.kwargs["resume_text"], "Uploaded resume text")

    def test_recent_posted_checkbox_is_passed_to_ranking(self) -> None:
        payload = {"results": [], "grouped_results": [], "total_jobs": 0, "filtered_jobs": 0}
        run_resume_ranking = Mock(return_value=payload)
        fake_ranking_service = types.SimpleNamespace(run_resume_ranking=run_resume_ranking)

        with (
            patch.dict(sys.modules, {"ranking_service": fake_ranking_service}),
            patch.object(web_main, "render_template", return_value="ok"),
            web_main.app.test_client() as client,
        ):
            response = client.post(
                "/rank",
                data={
                    "resume_text": "Python engineer resume",
                    "require_recent_posted": "on",
                },
                environ_overrides={"REMOTE_ADDR": "203.0.113.14"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(run_resume_ranking.call_args.kwargs["require_recent_posted"])

    def test_resume_preview_extracts_pdf_text_into_textarea(self) -> None:
        with (
            patch.object(web_main, "extract_pdf_text", return_value="Extracted PDF resume"),
            web_main.app.test_client() as client,
        ):
            response = client.post(
                "/resume-preview",
                data={"resume_file": (BytesIO(b"%PDF test"), "resume.pdf")},
                content_type="multipart/form-data",
            )

        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Resume text extracted", body)
        self.assertIn("Extracted PDF resume", body)

    def test_resume_preview_extracts_docx_text_into_textarea(self) -> None:
        with (
            patch.object(web_main, "extract_docx_text", return_value="Extracted DOCX resume"),
            web_main.app.test_client() as client,
        ):
            response = client.post(
                "/resume-preview",
                data={"resume_file": (BytesIO(b"docx test"), "resume.docx")},
                content_type="multipart/form-data",
            )

        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Resume text extracted", body)
        self.assertIn("Extracted DOCX resume", body)

    def test_resume_preview_rejects_unsupported_file_type(self) -> None:
        with web_main.app.test_client() as client:
            response = client.post(
                "/resume-preview",
                data={"resume_file": (BytesIO(b"plain text"), "resume.txt")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Please upload a PDF or DOCX resume file.", response.get_data(as_text=True))

    def test_resume_preview_rejects_oversized_resume_file(self) -> None:
        with (
            patch.object(web_main, "MAX_RESUME_FILE_BYTES", 4),
            web_main.app.test_client() as client,
        ):
            response = client.post(
                "/resume-preview",
                data={"resume_file": (BytesIO(b"12345"), "resume.docx")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Resume file is too large", response.get_data(as_text=True))

    def test_homepage_renders_saved_resume_controls(self) -> None:
        with web_main.app.test_client() as client:
            response = client.get("/")

        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Process and save current resume", body)
        self.assertIn("use_saved_resume", body)
        self.assertIn("/resume-profile", body)


if __name__ == "__main__":
    unittest.main()
