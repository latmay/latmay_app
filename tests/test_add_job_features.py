from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

sentence_transformers_module = types.ModuleType("sentence_transformers")
sentence_transformers_module.SentenceTransformer = object
sentence_transformers_module.CrossEncoder = object
sys.modules.setdefault("sentence_transformers", sentence_transformers_module)

psycopg_module = types.ModuleType("psycopg")
psycopg_types_module = types.ModuleType("psycopg.types")
psycopg_json_module = types.ModuleType("psycopg.types.json")
psycopg_rows_module = types.ModuleType("psycopg.rows")
psycopg_errors_module = types.SimpleNamespace(
    QueryCanceled=Exception,
    LockNotAvailable=Exception,
    DeadlockDetected=Exception,
)
psycopg_json_module.Jsonb = lambda value: value
psycopg_types_module.json = psycopg_json_module
psycopg_rows_module.dict_row = object()
psycopg_module.errors = psycopg_errors_module
psycopg_module.types = psycopg_types_module
psycopg_module.rows = psycopg_rows_module
sys.modules.setdefault("psycopg", psycopg_module)
sys.modules.setdefault("psycopg.types", psycopg_types_module)
sys.modules.setdefault("psycopg.types.json", psycopg_json_module)
sys.modules.setdefault("psycopg.rows", psycopg_rows_module)

sklearn_module = types.ModuleType("sklearn")
sklearn_feature_extraction_module = types.ModuleType("sklearn.feature_extraction")
sklearn_text_module = types.ModuleType("sklearn.feature_extraction.text")
sklearn_text_module.TfidfVectorizer = object
sklearn_text_module.ENGLISH_STOP_WORDS = frozenset()
sklearn_feature_extraction_module.text = sklearn_text_module
sklearn_module.feature_extraction = sklearn_feature_extraction_module
sys.modules.setdefault("sklearn", sklearn_module)
sys.modules.setdefault("sklearn.feature_extraction", sklearn_feature_extraction_module)
sys.modules.setdefault("sklearn.feature_extraction.text", sklearn_text_module)

phrases_module = types.ModuleType("webapp.ranking_algorithms.phrases_wasserstein_rankings")
phrases_module.phrase_chunks = lambda text: []
words_module = types.ModuleType("webapp.ranking_algorithms.words_wasserstein_rankings")
words_module.build_stopword_set = lambda *args, **kwargs: set()
words_module.maybe_remove_frequent_words = lambda words, *args, **kwargs: words
words_module.maybe_remove_stopwords = lambda words, *args, **kwargs: words
words_module.tokenize_text = lambda text: []
ranking_algorithms_module = types.ModuleType("webapp.ranking_algorithms")
ranking_algorithms_module.phrases_wasserstein_rankings = phrases_module
ranking_algorithms_module.words_wasserstein_rankings = words_module
webapp_module = types.ModuleType("webapp")
webapp_module.ranking_algorithms = ranking_algorithms_module
sys.modules.setdefault("webapp", webapp_module)
sys.modules.setdefault("webapp.ranking_algorithms", ranking_algorithms_module)
sys.modules.setdefault("webapp.ranking_algorithms.phrases_wasserstein_rankings", phrases_module)
sys.modules.setdefault("webapp.ranking_algorithms.words_wasserstein_rankings", words_module)

from data_pipeline.enrichment import add_job_features  # noqa: E402


class FakeCursor:
    def __init__(self, row: dict[str, int] | None = None) -> None:
        self.row = row or {"row_count": 0}
        self.sql = ""
        self.params = {}

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, sql: str, params: dict | None = None) -> None:
        self.sql = sql
        self.params = params or {}

    def fetchall(self) -> list[dict]:
        return []

    def fetchone(self) -> dict:
        return self.row


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.cursor_instance = cursor

    def cursor(self) -> FakeCursor:
        return self.cursor_instance


class AddJobFeaturesTests(unittest.TestCase):
    successful_location_status_sql = """
                  AND location_parse_status IN (
                        'parsed',
                        'country_only',
                        'remote',
                        'multi_location',
                        'city_resolved'
                  )
    """.strip()

    def test_non_ml_batch_size_prefers_dedicated_env_var(self) -> None:
        with patch.dict(
            os.environ,
            {"NON_ML_BATCH_SIZE": "25", "ENRICHMENT_BATCH_SIZE": "500"},
            clear=False,
        ):
            self.assertEqual(add_job_features.get_non_ml_batch_size(), 25)

    def test_non_ml_batch_size_falls_back_to_enrichment_batch_size(self) -> None:
        with patch.dict(os.environ, {"ENRICHMENT_BATCH_SIZE": "75"}, clear=False):
            os.environ.pop("NON_ML_BATCH_SIZE", None)
            self.assertEqual(add_job_features.get_non_ml_batch_size(), 75)

    def test_ml_max_batches_has_independent_setting(self) -> None:
        with patch.dict(
            os.environ,
            {"ML_ENRICHMENT_MAX_BATCHES": "6", "ENRICHMENT_MAX_BATCHES": "30"},
            clear=False,
        ):
            self.assertEqual(add_job_features.get_ml_max_batches(), 6)
            self.assertEqual(add_job_features.get_max_batches(), 30)

    def test_fetch_rows_to_enrich_only_considers_newest_export_eligible_jobs(self) -> None:
        cursor = FakeCursor()
        conn = FakeConnection(cursor)

        with patch.object(add_job_features, "get_export_embedding_keep_count", return_value=1234):
            add_job_features.fetch_rows_to_enrich(
                conn,
                limit=50,
                model_name="model",
                model_revision="revision",
                version="version",
            )

        self.assertIn("WITH export_eligible_jobs AS", cursor.sql)
        self.assertIn("WHERE is_export_eligible = TRUE", cursor.sql)
        self.assertNotIn("btrim(content_text)", cursor.sql)
        self.assertIn("ORDER BY posted_at_utc DESC, id DESC", cursor.sql)
        self.assertIn("JOIN export_eligible_jobs", cursor.sql)
        self.assertEqual(cursor.params["export_keep_count"], 1234)
        self.assertEqual(cursor.params["limit"], 50)

    def test_count_rows_to_enrich_only_counts_newest_export_eligible_jobs(self) -> None:
        cursor = FakeCursor({"row_count": 7})
        conn = FakeConnection(cursor)

        with patch.object(add_job_features, "get_export_embedding_keep_count", return_value=1234):
            count = add_job_features.count_rows_to_enrich(
                conn,
                model_name="model",
                model_revision="revision",
                version="version",
            )

        self.assertEqual(count, 7)
        self.assertIn("WITH export_eligible_jobs AS", cursor.sql)
        self.assertIn("WHERE is_export_eligible = TRUE", cursor.sql)
        self.assertNotIn("btrim(content_text)", cursor.sql)
        self.assertIn("ORDER BY posted_at_utc DESC, id DESC", cursor.sql)
        self.assertIn("JOIN export_eligible_jobs", cursor.sql)
        self.assertEqual(cursor.params["export_keep_count"], 1234)

    def test_fetch_rows_to_prepare_uses_non_model_feature_fields(self) -> None:
        cursor = FakeCursor()
        conn = FakeConnection(cursor)

        with patch.object(add_job_features, "get_export_embedding_keep_count", return_value=1234):
            add_job_features.fetch_rows_to_prepare(conn, limit=50, version="version")

        self.assertIn("WITH export_eligible_jobs AS", cursor.sql)
        self.assertIn("jobs.job_word_tokens IS NULL", cursor.sql)
        self.assertIn("jobs.job_selected_words IS NULL", cursor.sql)
        self.assertIn("jobs.job_phrase_chunks IS NULL", cursor.sql)
        self.assertIn("jobs.title_requirements_text IS NULL", cursor.sql)
        self.assertNotIn("load_minilm", cursor.sql.lower())
        self.assertEqual(cursor.params["export_keep_count"], 1234)
        self.assertEqual(cursor.params["limit"], 50)

    def test_fetch_rows_to_enrich_requires_prepared_feature_fields(self) -> None:
        cursor = FakeCursor()
        conn = FakeConnection(cursor)

        with patch.object(add_job_features, "get_export_embedding_keep_count", return_value=1234):
            add_job_features.fetch_rows_to_enrich(
                conn,
                limit=50,
                model_name="model",
                model_revision="revision",
                version="version",
            )

        self.assertIn("jobs.job_selected_word_embeddings IS NULL", cursor.sql)
        self.assertIn("jobs.job_phrase_chunk_embeddings IS NULL", cursor.sql)
        self.assertIn("jobs.title_requirements_embedding IS NULL", cursor.sql)
        self.assertIn("jobs.enrichment_ml_version IS DISTINCT FROM %(version)s", cursor.sql)
        self.assertIn("AND jobs.enrichment_version = %(version)s", cursor.sql)
        self.assertIn("AND jobs.job_selected_words IS NOT NULL", cursor.sql)
        self.assertIn("AND jobs.job_phrase_chunks IS NOT NULL", cursor.sql)
        self.assertIn("AND jobs.title_requirements_text IS NOT NULL", cursor.sql)


if __name__ == "__main__":
    unittest.main()
