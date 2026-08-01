from __future__ import annotations

import unittest
from unittest.mock import patch

from data_pipeline.common import model_loader


class ModelRevisionPinningTests(unittest.TestCase):
    def test_minilm_revision_is_pinned_when_unset_or_blank(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                model_loader.get_minilm_model_revision(),
                "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            )

        with patch.dict("os.environ", {"MINILM_MODEL_REVISION": ""}, clear=True):
            self.assertEqual(
                model_loader.get_minilm_model_revision(),
                "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            )

    def test_cross_encoder_revision_is_pinned_when_unset_or_blank(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                model_loader.get_cross_encoder_model_revision(),
                "4bebbd56fc380a66525f95b03d4ec1a4b41a4f1e",
            )

        with patch.dict("os.environ", {"CROSS_ENCODER_MODEL_REVISION": ""}, clear=True):
            self.assertEqual(
                model_loader.get_cross_encoder_model_revision(),
                "4bebbd56fc380a66525f95b03d4ec1a4b41a4f1e",
            )

    def test_explicit_revision_override_still_works(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MINILM_MODEL_REVISION": "minilm-test-revision",
                "CROSS_ENCODER_MODEL_REVISION": "cross-test-revision",
            },
            clear=True,
        ):
            self.assertEqual(model_loader.get_minilm_model_revision(), "minilm-test-revision")
            self.assertEqual(model_loader.get_cross_encoder_model_revision(), "cross-test-revision")


if __name__ == "__main__":
    unittest.main()
