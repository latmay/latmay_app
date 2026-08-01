from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.generate_cloudrun_env_yaml import parse_env_file


class GenerateCloudRunEnvYamlTests(unittest.TestCase):
    def test_includes_firebase_web_api_key_but_excludes_other_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "FIREBASE_WEB_API_KEY=firebase-public-value\n"
                "OPENAI_API_KEY=private-value\n"
                "EXAMPLE_API_KEY=private-value\n"
                "NORMAL_SETTING=enabled\n",
                encoding="utf-8",
            )

            parsed = parse_env_file(env_path)

        self.assertIn("FIREBASE_WEB_API_KEY", parsed)
        self.assertEqual(parsed["NORMAL_SETTING"], "enabled")
        self.assertNotIn("OPENAI_API_KEY", parsed)
        self.assertNotIn("EXAMPLE_API_KEY", parsed)


if __name__ == "__main__":
    unittest.main()
