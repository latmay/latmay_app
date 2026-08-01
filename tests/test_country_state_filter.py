import sys
import unittest
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = PROJECT_ROOT / "webapp"
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))


if pd is not None:
    from hard_filters.country_filter import filter_jobs_df_by_country, location_matches_us_state  # noqa: E402


@unittest.skipIf(pd is None, "pandas is not installed")
class CountryStateFilterTests(unittest.TestCase):
    def test_location_matches_us_state_name_and_abbrev(self) -> None:
        self.assertTrue(location_matches_us_state("San Francisco, CA, United States", "California"))
        self.assertTrue(location_matches_us_state("San Francisco, California, United States", "CA"))
        self.assertTrue(location_matches_us_state("Boston, Massachusetts, United States", "MA"))
        self.assertFalse(location_matches_us_state("Austin, TX, United States", "California"))

    def test_filter_jobs_df_by_country_and_state(self) -> None:
        df = pd.DataFrame(
            [
                {"title": "A", "location_name": "San Francisco, CA, United States"},
                {"title": "B", "location_name": "Boston, Massachusetts, United States"},
                {"title": "C", "location_name": "Toronto, Canada"},
            ]
        )

        filtered = filter_jobs_df_by_country(df, country="United States", state="CA")

        self.assertEqual(filtered["title"].tolist(), ["A"])

    def test_state_filter_is_ignored_for_non_us_country(self) -> None:
        df = pd.DataFrame(
            [
                {"title": "A", "location_name": "Toronto, Canada"},
                {"title": "B", "location_name": "Vancouver, Canada"},
            ]
        )

        filtered = filter_jobs_df_by_country(df, country="Canada", state="CA")

        self.assertEqual(filtered["title"].tolist(), ["A", "B"])

    def test_normalized_country_and_region_are_preferred(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "title": "A",
                    "location_name": "Toronto, Canada",
                    "location_country": "US",
                    "location_region": "CA",
                    "location_segments": [{"country": "US", "region": "CA"}],
                },
                {
                    "title": "B",
                    "location_name": "San Francisco, CA, United States",
                    "location_country": "CA",
                    "location_region": "ON",
                    "location_segments": [{"country": "CA", "region": "ON"}],
                },
            ]
        )

        filtered = filter_jobs_df_by_country(df, country="United States", state="CA")

        self.assertEqual(filtered["title"].tolist(), ["A"])

    def test_multi_location_segments_match_any_country_and_us_state(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "title": "A",
                    "location_name": "Toronto | Boston",
                    "location_country": "CA",
                    "location_region": "ON",
                    "location_segments": [
                        {"country": "CA", "region": "ON"},
                        {"country": "US", "region": "MA"},
                    ],
                },
                {
                    "title": "B",
                    "location_name": "Toronto, Canada",
                    "location_country": "CA",
                    "location_region": "ON",
                    "location_segments": [{"country": "CA", "region": "ON"}],
                },
            ]
        )

        filtered = filter_jobs_df_by_country(df, country="US", state="MA")

        self.assertEqual(filtered["title"].tolist(), ["A"])

    def test_csv_stringified_segments_are_supported(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "title": "A",
                    "location_name": "Toronto | Boston",
                    "location_segments": (
                        "[{'country': 'CA', 'region': 'ON'}, "
                        "{'country': 'US', 'region': 'MA'}]"
                    ),
                }
            ]
        )

        filtered = filter_jobs_df_by_country(df, country="United States", state="MA")

        self.assertEqual(filtered["title"].tolist(), ["A"])

    def test_missing_normalized_fields_fall_back_to_location_name(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "title": "A",
                    "location_name": "San Francisco, CA, United States",
                },
                {
                    "title": "B",
                    "location_name": "Toronto, Canada",
                    "location_country": "CA",
                    "location_region": "ON",
                    "location_segments": [{"country": "CA", "region": "ON"}],
                },
            ]
        )

        filtered = filter_jobs_df_by_country(df, country="United States", state="CA")

        self.assertEqual(filtered["title"].tolist(), ["A"])


if __name__ == "__main__":
    unittest.main()
