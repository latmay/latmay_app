from __future__ import annotations

import sys
import unittest
import os
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.enrichment.normalize_location import (  # noqa: E402
    LOCATION_NORMALIZATION_VERSION,
    get_location_normalization_batch_size,
    parse_location_name,
    update_location_normalization,
)


class FakeCursor:
    def __init__(self, conn: "FakeConn") -> None:
        self.conn = conn
        self.result_rows: list[dict[str, object]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> None:
        if sql.strip().upper().startswith("SELECT"):
            self.conn.select_sql = sql
            self.conn.select_params = params
            if "COUNT(*)" in sql:
                self.result_rows = [{"candidate_count": len(self.conn.rows)}]
            else:
                last_id = int(params[-2])
                limit = int(params[-1])
                self.result_rows = [
                    row for row in self.conn.rows if int(row["id"]) > last_id
                ][:limit]
            return
        self.conn.update_sql.append(sql)
        self.conn.update_params.append(params)

    def executemany(self, sql: str, params_seq: list[tuple[object, ...]]) -> None:
        for params in params_seq:
            self.conn.update_sql.append(sql)
            self.conn.update_params.append(params)

    def fetchall(self) -> list[dict[str, object]]:
        return self.result_rows

    def fetchone(self) -> dict[str, object]:
        return self.result_rows[0]


class FakeConn:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.select_sql = ""
        self.select_params: tuple[object, ...] = ()
        self.update_sql: list[str] = []
        self.update_params: list[tuple[object, ...]] = []
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1


class NormalizeLocationTests(unittest.TestCase):
    def test_parses_us_city_state_without_guessing_city(self) -> None:
        parsed = parse_location_name("Boston, MA")

        self.assertEqual(parsed.location_country, "US")
        self.assertEqual(parsed.location_region, "MA")
        self.assertEqual(parsed.location_segments, [{"country": "US", "region": "MA"}])
        self.assertEqual(parsed.work_arrangement, None)
        self.assertEqual(parsed.location_parse_status, "parsed")

    def test_parses_common_us_state_abbreviation(self) -> None:
        parsed = parse_location_name("Cambridge, Mass.")

        self.assertEqual(parsed.location_country, "US")
        self.assertEqual(parsed.location_region, "MA")
        self.assertEqual(parsed.location_parse_status, "parsed")

    def test_city_only_resolves_to_most_populous_geonames_match(self) -> None:
        parsed = parse_location_name("Boston")

        self.assertEqual(parsed.location_country, "US")
        self.assertEqual(parsed.location_region, "MA")
        self.assertEqual(parsed.location_segments, [{"country": "US", "region": "MA"}])
        self.assertEqual(parsed.location_parse_status, "city_resolved")

    def test_non_us_city_resolution_does_not_store_fips_region(self) -> None:
        parsed = parse_location_name("London")

        self.assertEqual(parsed.location_country, "GB")
        self.assertIsNone(parsed.location_region)
        self.assertEqual(parsed.location_segments, [{"country": "GB", "region": None}])
        self.assertEqual(parsed.location_parse_status, "city_resolved")

    def test_city_resolution_handles_non_us_non_gb_city(self) -> None:
        parsed = parse_location_name("Stockholm")

        self.assertEqual(parsed.location_country, "SE")
        self.assertIsNone(parsed.location_region)
        self.assertEqual(parsed.location_segments, [{"country": "SE", "region": None}])
        self.assertEqual(parsed.location_parse_status, "city_resolved")

    def test_unresolvable_city_stays_ambiguous(self) -> None:
        parsed = parse_location_name("Notacityzzzz")

        self.assertIsNone(parsed.location_country)
        self.assertIsNone(parsed.location_region)
        self.assertEqual(parsed.location_segments, [{"country": None, "region": None}])
        self.assertEqual(parsed.location_parse_status, "ambiguous")

    def test_city_resolution_applies_to_each_segment(self) -> None:
        parsed = parse_location_name("Boston | Stockholm")

        self.assertEqual(parsed.location_country, "US")
        self.assertEqual(parsed.location_region, "MA")
        self.assertEqual(
            parsed.location_segments,
            [
                {"country": "US", "region": "MA"},
                {"country": "SE", "region": None},
            ],
        )
        self.assertEqual(parsed.location_parse_status, "city_resolved")

    def test_bare_subdivision_like_code_is_not_treated_as_country(self) -> None:
        parsed = parse_location_name("MA")

        self.assertIsNone(parsed.location_country)
        self.assertIsNone(parsed.location_region)
        self.assertEqual(parsed.location_parse_status, "ambiguous")

    def test_country_only_uses_iso_alpha2(self) -> None:
        parsed = parse_location_name("United States")

        self.assertEqual(parsed.location_country, "US")
        self.assertIsNone(parsed.location_region)
        self.assertEqual(parsed.location_parse_status, "country_only")

    def test_remote_country_only_uses_remote_status(self) -> None:
        parsed = parse_location_name("Remote - United States")

        self.assertEqual(parsed.location_country, "US")
        self.assertIsNone(parsed.location_region)
        self.assertEqual(parsed.work_arrangement, "remote")
        self.assertEqual(parsed.location_parse_status, "remote")

    def test_remote_without_country_keeps_unknown_segment(self) -> None:
        parsed = parse_location_name("Remote")

        self.assertIsNone(parsed.location_country)
        self.assertIsNone(parsed.location_region)
        self.assertEqual(parsed.location_segments, [{"country": None, "region": None}])
        self.assertEqual(parsed.work_arrangement, "remote")
        self.assertEqual(parsed.location_parse_status, "remote")

    def test_parses_canadian_subdivision_with_country(self) -> None:
        parsed = parse_location_name("Toronto, Ontario, Canada")

        self.assertEqual(parsed.location_country, "CA")
        self.assertEqual(parsed.location_region, "ON")
        self.assertEqual(parsed.location_parse_status, "parsed")

    def test_parses_iso_3166_2_subdivision_code(self) -> None:
        parsed = parse_location_name("US-MA")

        self.assertEqual(parsed.location_country, "US")
        self.assertEqual(parsed.location_region, "MA")
        self.assertEqual(parsed.location_parse_status, "parsed")

    def test_multi_location_segments_are_preserved(self) -> None:
        parsed = parse_location_name("Boston, MA | Toronto, Ontario, Canada")

        self.assertEqual(parsed.location_country, "US")
        self.assertEqual(parsed.location_region, "MA")
        self.assertEqual(
            parsed.location_segments,
            [
                {"country": "US", "region": "MA"},
                {"country": "CA", "region": "ON"},
            ],
        )
        self.assertEqual(parsed.location_parse_status, "multi_location")

    def test_hybrid_signal_does_not_imply_onsite(self) -> None:
        parsed = parse_location_name("Hybrid - Boston, MA")

        self.assertEqual(parsed.location_country, "US")
        self.assertEqual(parsed.location_region, "MA")
        self.assertEqual(parsed.work_arrangement, "hybrid")
        self.assertEqual(parsed.location_parse_status, "parsed")

    def test_missing_location_is_marked_missing(self) -> None:
        parsed = parse_location_name("")

        self.assertIsNone(parsed.location_country)
        self.assertIsNone(parsed.location_region)
        self.assertEqual(parsed.location_segments, [])
        self.assertEqual(parsed.location_parse_status, "missing")

    def test_update_location_normalization_preserves_location_name(self) -> None:
        conn = FakeConn([{"id": 1, "location_name": "Boston, MA"}])

        updated = update_location_normalization(conn)

        self.assertEqual(updated, 1)
        self.assertEqual(
            conn.select_params,
            (LOCATION_NORMALIZATION_VERSION, 1, 500),
        )
        self.assertEqual(conn.commits, 1)
        self.assertNotIn("location_name =", conn.update_sql[0])
        params = conn.update_params[0]
        self.assertEqual(params[0], "US")
        self.assertEqual(params[1], "MA")
        self.assertEqual(params[4], "parsed")
        self.assertEqual(params[5], LOCATION_NORMALIZATION_VERSION)
        self.assertEqual(params[7], 1)

    def test_update_location_normalization_commits_each_batch(self) -> None:
        conn = FakeConn(
            [
                {"id": 1, "location_name": "Boston, MA"},
                {"id": 2, "location_name": "Toronto, Ontario, Canada"},
                {"id": 3, "location_name": "Stockholm"},
            ]
        )

        updated = update_location_normalization(conn, batch_size=2)

        self.assertEqual(updated, 3)
        self.assertEqual(conn.commits, 2)
        self.assertEqual([params[7] for params in conn.update_params], [1, 2, 3])

    def test_zero_candidates_commit_without_batch_select(self) -> None:
        conn = FakeConn([])

        updated = update_location_normalization(conn)

        self.assertEqual(updated, 0)
        self.assertEqual(conn.commits, 1)
        self.assertIn("COUNT(*)", conn.select_sql)
        self.assertEqual(conn.select_params, (LOCATION_NORMALIZATION_VERSION,))

    def test_location_batch_size_prefers_non_ml_batch_size(self) -> None:
        with patch.dict(
            os.environ,
            {"NON_ML_BATCH_SIZE": "25", "ENRICHMENT_BATCH_SIZE": "500"},
            clear=False,
        ):
            self.assertEqual(get_location_normalization_batch_size(), 25)

    def test_update_location_normalization_rejects_invalid_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be at least 1"):
            update_location_normalization(FakeConn([]), batch_size=0)


if __name__ == "__main__":
    unittest.main()
