from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


COUNTRY_ALIASES = {
    "india": ["india"],
    "canada": ["canada"],
    "united kingdom": [
        "united kingdom", "uk", "u.k.", "england", "scotland", "wales",
        "northern ireland", "britain", "great britain",
    ],
    "germany": ["germany"],
    "france": ["france"],
    "australia": ["australia"],
    "ireland": ["ireland", "ie"],
    "finland": ["finland"],
    "spain": ["spain"],
    "denmark": ["denmark"],
    "hungary": ["hungary"],
    "slovakia": ["slovakia"],
    "czechia": ["czechia", "czech republic"],
    "south korea": ["south korea", "korea"],
    "taiwan": ["taiwan"],
    "japan": ["japan"],
    "singapore": ["singapore"],
    "brazil": ["brazil"],
    "belgium": ["belgium"],
    "netherlands": ["netherlands", "the netherlands"],
    "chile": ["chile"],
    "argentina": ["argentina"],
    "colombia": ["colombia"],
    "mexico": ["mexico"],
    "uruguay": ["uruguay"],
    "sweden": ["sweden"],
    "poland": ["poland"],
    "romania": ["romania"],
    "croatia": ["croatia"],
    "malta": ["malta"],
    "cyprus": ["cyprus"],
    "bulgaria": ["bulgaria"],
    "lithuania": ["lithuania"],
    "israel": ["israel"],
    "united arab emirates": ["united arab emirates", "uae"],
    "nigeria": ["nigeria"],
}

COUNTRY_ISO_CODES = {
    "argentina": "AR",
    "australia": "AU",
    "belgium": "BE",
    "brazil": "BR",
    "bulgaria": "BG",
    "canada": "CA",
    "chile": "CL",
    "colombia": "CO",
    "croatia": "HR",
    "cyprus": "CY",
    "czechia": "CZ",
    "denmark": "DK",
    "finland": "FI",
    "france": "FR",
    "germany": "DE",
    "hungary": "HU",
    "india": "IN",
    "ireland": "IE",
    "israel": "IL",
    "japan": "JP",
    "lithuania": "LT",
    "malta": "MT",
    "mexico": "MX",
    "netherlands": "NL",
    "nigeria": "NG",
    "poland": "PL",
    "romania": "RO",
    "singapore": "SG",
    "slovakia": "SK",
    "south korea": "KR",
    "spain": "ES",
    "sweden": "SE",
    "taiwan": "TW",
    "united arab emirates": "AE",
    "united kingdom": "GB",
    "united states": "US",
    "uruguay": "UY",
}

US_STATE_NAMES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming", "district of columbia",
}

US_STATE_ABBREVS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
}

US_STATE_ABBREV_TO_NAME = {
    "AL": "alabama",
    "AK": "alaska",
    "AZ": "arizona",
    "AR": "arkansas",
    "CA": "california",
    "CO": "colorado",
    "CT": "connecticut",
    "DE": "delaware",
    "FL": "florida",
    "GA": "georgia",
    "HI": "hawaii",
    "ID": "idaho",
    "IL": "illinois",
    "IN": "indiana",
    "IA": "iowa",
    "KS": "kansas",
    "KY": "kentucky",
    "LA": "louisiana",
    "ME": "maine",
    "MD": "maryland",
    "MA": "massachusetts",
    "MI": "michigan",
    "MN": "minnesota",
    "MS": "mississippi",
    "MO": "missouri",
    "MT": "montana",
    "NE": "nebraska",
    "NV": "nevada",
    "NH": "new hampshire",
    "NJ": "new jersey",
    "NM": "new mexico",
    "NY": "new york",
    "NC": "north carolina",
    "ND": "north dakota",
    "OH": "ohio",
    "OK": "oklahoma",
    "OR": "oregon",
    "PA": "pennsylvania",
    "RI": "rhode island",
    "SC": "south carolina",
    "SD": "south dakota",
    "TN": "tennessee",
    "TX": "texas",
    "UT": "utah",
    "VT": "vermont",
    "VA": "virginia",
    "WA": "washington",
    "WV": "west virginia",
    "WI": "wisconsin",
    "WY": "wyoming",
    "DC": "district of columbia",
}

US_STATE_NAME_TO_ABBREV = {
    state_name: state_abbrev.lower()
    for state_abbrev, state_name in US_STATE_ABBREV_TO_NAME.items()
}

US_COUNTRY_KEYS = {"united states", "usa", "us", "u.s.", "u.s.a."}

US_MAJOR_CITIES = {
    "san francisco",
    "new york",
    "new york city",
    "boston",
    "chicago",
    "los angeles",
    "seattle",
    "austin",
    "denver",
    "atlanta",
    "miami",
    "dallas",
    "houston",
    "phoenix",
    "san diego",
    "san jose",
    "washington",
    "washington dc",
    "washington d.c.",
    "baltimore",
    "boulder",
    "pittsburgh",
    "cincinnati",
    "des moines",
    "menlo park",
    "mountain view",
    "redwood city",
    "palo alto",
    "redmond",
    "long beach",
    "cape canaveral",
    "starbase",
    "bastrop",
    "hawthorne",
    "mcgregor",
    "provo",
    "charlotte",
    "raleigh",
    "durham",
    "somerville",
    "billerica",
    "framingham",
    "quincy",
    "lexington",
    "waltham",
    "reston",
    "mclean",
    "irvine",
    "costa mesa",
    "santa ana",
    "san carlos",
    "foster city",
    "south san francisco",
    "sunnyvale",
    "los altos",
    "tucson",
    "albuquerque",
    "st. louis",
    "saint louis",
}

AMBIGUOUS_NON_LOCATIONS = {
    "", "remote", "hybrid", "distributed", "in-office", "in office", "n/a",
    "worldwide", "home based - worldwide", "water/ww",
}


def normalize_location_text(text: str) -> str:
    """
    Lowercase and normalize punctuation/spacing so matching is easier.
    """
    if not text:
        return ""

    text = str(text).strip().lower()

    replacements = {
        "–": "-",
        "—": "-",
        "_": " ",
        "(": " ",
        ")": " ",
        "[": " ",
        "]": " ",
        "{": " ",
        "}": " ",
        "/": " / ",
        "|": " ; ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_location_segments(location: str) -> list[str]:
    """
    Split multi-location strings into smaller segments.
    """
    if not location:
        return []

    normalized = normalize_location_text(location)
    segments = re.split(r"\s*;\s*", normalized)
    segments = [seg.strip(" ,;-") for seg in segments if seg.strip(" ,;-")]

    return segments


def has_whole_word(text: str, pattern: str) -> bool:
    """
    Whole-word match, tolerant of punctuation around the term.
    """
    escaped = re.escape(pattern.lower())
    return re.search(rf"(?<![a-z]){escaped}(?![a-z])", text) is not None


def segment_has_us_state_name(segment: str) -> bool:
    return any(has_whole_word(segment, state_name) for state_name in US_STATE_NAMES)


def segment_has_us_state_abbrev(segment: str) -> bool:
    """
    Match U.S. state abbreviations in common location formats such as:
    - Boston, MA
    - NY - New York City
    - UT-Cottonwood Heights
    - Office: Provo, UT
    - Remote - UT
    - CA - San Francisco
    """
    for abbr in US_STATE_ABBREVS:
        escaped = re.escape(abbr.lower())
        pattern = rf"(?:^|[\s,;:/()\-]){escaped}(?:$|[\s,;:/()\-])"
        if re.search(pattern, segment):
            return True
    return False


def country_supports_us_state_filter(country: str | None) -> bool:
    if country is None or not str(country).strip():
        return True

    return str(country).strip().lower() in US_COUNTRY_KEYS


def normalize_us_state_filter(state: str | None) -> tuple[str, str] | None:
    if state is None or not str(state).strip():
        return None

    state_text = normalize_location_text(state)
    if state_text in US_STATE_NAMES:
        return state_text, US_STATE_NAME_TO_ABBREV.get(state_text, "")

    state_abbrev = re.sub(r"[^A-Za-z]", "", str(state)).upper()
    state_name = US_STATE_ABBREV_TO_NAME.get(state_abbrev)
    if state_name:
        return state_name, state_abbrev.lower()

    return state_text, ""


def segment_matches_us_state(segment: str, state_filter: tuple[str, str]) -> bool:
    state_name, state_abbrev = state_filter
    if state_name and has_whole_word(segment, state_name):
        return True

    if state_abbrev:
        escaped = re.escape(state_abbrev)
        pattern = rf"(?:^|[\s,;:/()\-]){escaped}(?:$|[\s,;:/()\-])"
        return re.search(pattern, segment) is not None

    return False


def location_matches_us_state(location: str, state: str | None) -> bool:
    state_filter = normalize_us_state_filter(state)
    if state_filter is None:
        return True

    segments = split_location_segments(location)
    if not segments:
        return False

    for segment in segments:
        if segment in AMBIGUOUS_NON_LOCATIONS:
            continue

        if segment_matches_us_state(segment, state_filter):
            return True

    return False


def segment_has_explicit_us_term(segment: str) -> bool:
    explicit_us_patterns = [
        r"(?<![a-z])united states(?![a-z])",
        r"(?<![a-z])usa(?![a-z])",
        r"(?<![a-z])u\.s\.(?![a-z])",
        r"(?<![a-z])u\.s\.a\.(?![a-z])",
        r"(?<![a-z])us remote(?![a-z])",
        r"(?<![a-z])remote us(?![a-z])",
        r"(?<![a-z])remote - us(?![a-z])",
        r"(?<![a-z])remote-us(?![a-z])",
        r"(?<![a-z])u\.s\. remote(?![a-z])",
        r"(?<![a-z])remote - u\.s\.(?![a-z])",
    ]
    return any(re.search(pattern, segment) for pattern in explicit_us_patterns)


def segment_has_us_city(segment: str) -> bool:
    return any(has_whole_word(segment, city) for city in US_MAJOR_CITIES)


def location_matches_united_states(location: str) -> bool:
    """
    Return True if the location string appears to be in the United States.

    Order of checks:
    1. explicit U.S. terms
    2. full state names
    3. state abbreviations
    4. curated U.S. city list
    """
    if not location:
        return False

    segments = split_location_segments(location)
    if not segments:
        return False

    for segment in segments:
        if segment in AMBIGUOUS_NON_LOCATIONS:
            continue

        if segment_has_explicit_us_term(segment):
            return True

        if segment_has_us_state_name(segment):
            return True

        if segment_has_us_state_abbrev(segment):
            return True

        if segment_has_us_city(segment):
            return True

    return False


def location_matches_non_us_country(location: str, country_key: str) -> bool:
    """
    Match a non-U.S. country using aliases across all subsegments of the location string.
    """
    if not location:
        return False

    aliases = COUNTRY_ALIASES.get(country_key, [country_key])
    segments = split_location_segments(location)
    if not segments:
        return False

    for segment in segments:
        if segment in AMBIGUOUS_NON_LOCATIONS:
            continue

        for alias in aliases:
            if has_whole_word(segment, alias):
                return True

    return False


def location_matches_country(location: str, country: str | None) -> bool:
    """
    Return True if the location matches the requested country.

    If country is None or blank, this returns True so no filtering occurs.
    """
    if country is None or not str(country).strip():
        return True

    if not location:
        return False

    country_key = str(country).strip().lower()

    if country_key in US_COUNTRY_KEYS:
        return location_matches_united_states(location)

    return location_matches_non_us_country(location, country_key)


def country_filter_to_iso(country: str | None) -> str | None:
    if country is None or not str(country).strip():
        return None

    country_key = normalize_location_text(country)
    compact_code = re.sub(r"[^A-Za-z]", "", str(country)).upper()
    if len(compact_code) == 2 and compact_code in COUNTRY_ISO_CODES.values():
        return compact_code
    if country_key in US_COUNTRY_KEYS:
        return "US"
    if country_key in COUNTRY_ISO_CODES:
        return COUNTRY_ISO_CODES[country_key]

    for canonical_name, aliases in COUNTRY_ALIASES.items():
        if country_key in {normalize_location_text(alias) for alias in aliases}:
            return COUNTRY_ISO_CODES.get(canonical_name)
    return None


def normalized_location_segments(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return []
    else:
        return []

    if not isinstance(parsed, list):
        return []
    return [segment for segment in parsed if isinstance(segment, dict)]


def normalized_code(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip().upper()


def normalized_country_match(row: pd.Series, country_iso: str) -> bool | None:
    segments = normalized_location_segments(row.get("location_segments"))
    segment_countries = {
        normalized_code(segment.get("country"))
        for segment in segments
        if normalized_code(segment.get("country"))
    }
    if segment_countries:
        return country_iso in segment_countries

    normalized_country = normalized_code(row.get("location_country"))
    if normalized_country:
        return normalized_country == country_iso
    return None


def normalized_us_state_match(row: pd.Series, state_abbrev: str) -> bool | None:
    segments = normalized_location_segments(row.get("location_segments"))
    us_regions = {
        normalized_code(segment.get("region"))
        for segment in segments
        if normalized_code(segment.get("country")) == "US"
        and normalized_code(segment.get("region"))
    }
    if us_regions:
        return state_abbrev in us_regions

    normalized_country = normalized_code(row.get("location_country"))
    normalized_region = normalized_code(row.get("location_region"))
    if normalized_country == "US" and normalized_region:
        return normalized_region == state_abbrev
    return None


def filter_jobs_df_by_country(
    jobs_df: pd.DataFrame,
    country: str | None,
    state: str | None = None,
    location_column: str = "location_name",
) -> pd.DataFrame:
    """
    Filter an existing jobs dataframe by country and optional U.S. state.
    """
    if location_column not in jobs_df.columns:
        raise ValueError(
            f"Column '{location_column}' not found in dataframe. "
            f"Available columns: {list(jobs_df.columns)}"
        )

    filtered = jobs_df.copy()

    if country is not None and str(country).strip():
        country_iso = country_filter_to_iso(country)

        def row_matches_country(row: pd.Series) -> bool:
            if country_iso:
                normalized_match = normalized_country_match(row, country_iso)
                if normalized_match is not None:
                    return normalized_match
            location = row.get(location_column)
            location_text = "" if pd.isna(location) else str(location)
            return location_matches_country(location_text, country)

        mask = filtered.apply(row_matches_country, axis=1)
        filtered = filtered.loc[mask].copy()

    if state is not None and str(state).strip() and country_supports_us_state_filter(country):
        state_filter = normalize_us_state_filter(state)

        def row_matches_state(row: pd.Series) -> bool:
            if state_filter and state_filter[1]:
                normalized_match = normalized_us_state_match(row, state_filter[1].upper())
                if normalized_match is not None:
                    return normalized_match
            location = row.get(location_column)
            location_text = "" if pd.isna(location) else str(location)
            return location_matches_us_state(location_text, state)

        mask = filtered.apply(row_matches_state, axis=1)
        filtered = filtered.loc[mask].copy()

    return filtered


def filter_jobs_by_country(
    input_csv: Path | str,
    output_csv: Path | str,
    country: str | None,
    state: str | None = None,
    location_column: str = "location_name",
) -> pd.DataFrame:
    """
    Read a jobs CSV, filter rows by country/state, write the result, and return it.
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    filtered_df = filter_jobs_df_by_country(
        jobs_df=df,
        country=country,
        state=state,
        location_column=location_column,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_csv, index=False)

    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Country filter: {country}")
    print(f"State filter: {state}")
    print(f"Total jobs: {len(df)}")
    print(f"Matching jobs: {len(filtered_df)}")

    return filtered_df


def main() -> None:
    """
    Standalone test only.
    """
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    data_dir = project_root / "data"

    input_csv = data_dir / "sample_combined_jobs_filtered.csv"
    output_csv = data_dir / "sample_combined_jobs_us_only.csv"

    country = "United States"   # or None
    state = None                # e.g. "CA" or "California"

    filter_jobs_by_country(
        input_csv=input_csv,
        output_csv=output_csv,
        country=country,
        state=state,
        location_column="location_name",
    )


if __name__ == "__main__":
    main()
