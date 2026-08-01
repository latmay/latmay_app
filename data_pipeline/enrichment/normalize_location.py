from __future__ import annotations

"""Normalize jobs.location_name into ISO-backed location fields."""

import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import geonamescache
import pycountry
from psycopg.types.json import Jsonb


LOCATION_NORMALIZATION_VERSION = "location-normalizer-v1"
DEFAULT_LOCATION_NORMALIZATION_BATCH_SIZE = 500


def _build_city_name_index() -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for city in geonamescache.GeonamesCache(min_city_population=1000).get_cities().values():
        raw = str(city.get("name") or "")
        nfkd = unicodedata.normalize("NFKD", raw)
        stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
        key = re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9]+", " ", stripped).strip()).lower()
        if key:
            index.setdefault(key, []).append(city)
    return index


_CITY_BY_NAME: dict[str, list[dict[str, Any]]] = _build_city_name_index()

VALID_PARSE_STATUSES = {
    "parsed",
    "country_only",
    "remote",
    "multi_location",
    "city_resolved",
    "ambiguous",
    "unparseable",
    "missing",
}

REMOTE_RE = re.compile(r"\b(remote|remotely|work\s+from\s+home|wfh|virtual|anywhere|worldwide)\b", re.IGNORECASE)
HYBRID_RE = re.compile(r"\bhybrid\b", re.IGNORECASE)
ONSITE_RE = re.compile(r"\b(on[-\s]?site|in[-\s]?office|office[-\s]?based)\b", re.IGNORECASE)
ISO_SUBDIVISION_RE = re.compile(r"\b([A-Z]{2})[-_ ]([A-Z0-9]{1,3})\b", re.IGNORECASE)
MULTI_LOCATION_RE = re.compile(r"\s*(?:\||;|\n|\t|/|\s+or\s+|\s+-\s+)\s*", re.IGNORECASE)
NOISE_RE = re.compile(
    r"\b("
    r"multiple\s+locations?|locations?|based|role|position|job|office|offices|"
    r"remote|remotely|hybrid|onsite|on\s+site|on-site|in\s+office|in-office|"
    r"work\s+from\s+home|wfh|virtual|hq|global|area|airport"
    r")\b",
    re.IGNORECASE,
)

_N_LOCATIONS_RE = re.compile(r"^\d+\s+locations?$", re.IGNORECASE)
_DASH_CODE_RE = re.compile(r"^(.+?)-([A-Za-z]{2})\s*,?\s*$")
_ZIP_SUFFIX_RE = re.compile(r"\s+\d{4,5}(?:-\d{4})?$")
_PREFIX_STATE_RE = re.compile(r"^([A-Za-z]{2})\s+(.+)$")
_SUFFIX_STATE_RE = re.compile(r"^(.+)\s+([A-Za-z]{2})$")
_COUNTRY_SUFFIX_RE = re.compile(
    r"\s+(?:u\.?\s*s\.?\s*a?\.?|united\s+states(?:\s+of\s+america)?|"
    r"united\s+kingdom|great\s+britain|u\.?\s*k\.?|canada|india|australia)\s*$",
    re.IGNORECASE,
)

COUNTRY_ALIASES = {
    "america": "US",
    "canada": "CA",
    "great britain": "GB",
    "gb": "GB",
    "u k": "GB",
    "uk": "GB",
    "united kingdom": "GB",
    "united states": "US",
    "united states america": "US",
    "united states of america": "US",
    "us": "US",
    "u s": "US",
    "usa": "US",
    "south korea": "KR",
    "hong kong": "HK",
    "the netherlands": "NL",
    "netherlands": "NL",
    "saudi arabia": "SA",
    "czech republic": "CZ",
    "brasil": "BR",
    "democratic republic of the congo": "CD",
    "democratic republic of congo": "CD",
}

COMMON_SUBDIVISION_ALIASES = {
    ("US", "DC"): ("dc", "d c", "washington dc", "washington d c", "district of columbia"),
    ("US", "MA"): ("mass", "massachusetts"),
}

COMMON_INFERABLE_COUNTRIES = ("US", "CA", "GB", "AU")


@dataclass(frozen=True)
class LocationSegment:
    country: str | None
    region: str | None
    city_resolved: bool = False

    def as_dict(self) -> dict[str, str | None]:
        return {"country": self.country, "region": self.region}


@dataclass(frozen=True)
class LocationNormalizationResult:
    location_country: str | None
    location_region: str | None
    location_segments: list[dict[str, str | None]]
    work_arrangement: str | None
    location_parse_status: str


def normalize_lookup_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def clean_location_text(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"_+", ", ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^\s*(multiple\s+locations?|locations?)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^([A-Z]{2}):\s+", r"\1, ", text)
    return text.strip(" ,-")


def derive_work_arrangement(location_name: Any) -> str | None:
    text = str(location_name or "")
    if HYBRID_RE.search(text):
        return "hybrid"
    if REMOTE_RE.search(text):
        return "remote"
    if ONSITE_RE.search(text):
        return "onsite"
    return None


def country_from_code(code: str | None) -> str | None:
    if not code:
        return None
    country = pycountry.countries.get(alpha_2=str(code).upper())
    return country.alpha_2 if country is not None else None


def lookup_country_token(token: str, *, allow_alpha2: bool = False) -> str | None:
    normalized = normalize_lookup_text(token)
    if not normalized:
        return None
    alias = COUNTRY_ALIASES.get(normalized)
    if alias:
        return alias
    compact = normalized.replace(" ", "")
    if compact in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[compact]
    if len(compact) == 2 and allow_alpha2 and is_common_subdivision_code(compact):
        return None
    if len(compact) == 2 and not allow_alpha2:
        return None
    if len(compact) < 3 and not allow_alpha2:
        return None
    try:
        country = pycountry.countries.lookup(compact if compact.isalpha() else normalized)
    except LookupError:
        return None
    return country.alpha_2


def is_common_subdivision_code(value: str) -> bool:
    normalized = normalize_lookup_text(value)
    if len(normalized) != 2:
        return False
    for country_code in COMMON_INFERABLE_COUNTRIES:
        for subdivision in subdivisions_for_country(country_code):
            if subdivision.code.rsplit("-", 1)[-1].lower() == normalized:
                return True
    return False


def subdivision_lookup_keys(subdivision: Any) -> set[str]:
    keys = {
        normalize_lookup_text(subdivision.name),
        normalize_lookup_text(subdivision.code.rsplit("-", 1)[-1]),
    }
    if getattr(subdivision, "type", None):
        keys.add(normalize_lookup_text(str(subdivision.type)))
    return {key for key in keys if key}


def subdivisions_for_country(country_code: str) -> list[Any]:
    return list(pycountry.subdivisions.get(country_code=country_code) or [])


def subdivision_for_country_and_token(country_code: str, token: str) -> str | None:
    country_code = country_code.upper()
    normalized = normalize_lookup_text(token)
    if not normalized:
        return None

    for code_country, code_region in ISO_SUBDIVISION_RE.findall(str(token or "")):
        full_code = f"{code_country.upper()}-{code_region.upper()}"
        subdivision = pycountry.subdivisions.get(code=full_code)
        if subdivision is not None and subdivision.country_code == country_code:
            return subdivision.code.rsplit("-", 1)[-1]

    for (alias_country, region), aliases in COMMON_SUBDIVISION_ALIASES.items():
        if alias_country == country_code and normalized in aliases:
            full_code = f"{country_code}-{region}"
            if pycountry.subdivisions.get(code=full_code) is not None:
                return region

    compact = normalized.replace(" ", "")
    for subdivision in subdivisions_for_country(country_code):
        region = subdivision.code.rsplit("-", 1)[-1]
        full_code = f"{country_code}-{region}"
        if pycountry.subdivisions.get(code=full_code) is None:
            continue
        keys = subdivision_lookup_keys(subdivision)
        if normalized in keys or compact == region.lower():
            return region

    return None


def infer_common_subdivision(token: str, *, has_context: bool) -> LocationSegment | None:
    normalized = normalize_lookup_text(token)
    if not normalized:
        return None

    for (country_code, region), aliases in COMMON_SUBDIVISION_ALIASES.items():
        if normalized in aliases:
            return LocationSegment(country=country_code, region=region)

    for country_code in COMMON_INFERABLE_COUNTRIES:
        for subdivision in subdivisions_for_country(country_code):
            region = subdivision.code.rsplit("-", 1)[-1]
            full_code = f"{country_code}-{region}"
            if pycountry.subdivisions.get(code=full_code) is None:
                continue
            if normalized == normalize_lookup_text(subdivision.name):
                return LocationSegment(country=country_code, region=region)
            if has_context and normalized == region.lower():
                return LocationSegment(country=country_code, region=region)

    return None


def expand_dash_separated_code(segment: str) -> str:
    """Expand 'Boston-MA' → 'Boston, MA' so the 2-letter code parses as a subdivision."""
    m = _DASH_CODE_RE.match(segment.strip())
    if m:
        return f"{m.group(1).strip()}, {m.group(2).strip()}"
    return segment


def expand_state_adjacent_code(segment: str) -> str:
    """Expand 'Austin TX' → 'Austin, TX' or 'WI Marshfield' → 'Marshfield, WI'."""
    if "," in segment:
        return segment
    m = _SUFFIX_STATE_RE.match(segment)
    if m and is_common_subdivision_code(m.group(2)):
        return f"{m.group(1).strip()}, {m.group(2).strip()}"
    m = _PREFIX_STATE_RE.match(segment)
    if m and is_common_subdivision_code(m.group(1)):
        return f"{m.group(2).strip()}, {m.group(1).strip()}"
    return segment


def segment_tokens(segment: str) -> list[str]:
    cleaned = re.sub(r"[()]", ",", segment)
    tokens = []
    for raw in cleaned.split(","):
        t = raw.strip(" .:-")
        if not t:
            continue
        t = _ZIP_SUFFIX_RE.sub("", t).strip(" .:-")
        if t:
            tokens.append(t)
    return tokens


def strip_noise(segment: str) -> str:
    text = NOISE_RE.sub(" ", segment)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,-")


def parse_segment(segment: str) -> LocationSegment:
    cleaned = strip_noise(segment)

    for country_code, region in ISO_SUBDIVISION_RE.findall(cleaned):
        country = country_from_code(country_code)
        full_code = f"{str(country_code).upper()}-{str(region).upper()}"
        subdivision = pycountry.subdivisions.get(code=full_code)
        if country and subdivision is not None:
            return LocationSegment(country=country, region=subdivision.code.rsplit("-", 1)[-1])

    cleaned = expand_dash_separated_code(cleaned)
    cleaned = expand_state_adjacent_code(cleaned)
    tokens = segment_tokens(cleaned)
    has_context = len(tokens) > 1

    country = None
    for token in tokens + ([cleaned] if cleaned else []):
        country = lookup_country_token(token, allow_alpha2=len(tokens) == 1)
        if country:
            break

    if not country:
        for token in tokens:
            if len(normalize_lookup_text(token).replace(" ", "")) == 2:
                country = lookup_country_token(token, allow_alpha2=True)
                if country:
                    break

    if country:
        for token in tokens:
            region = subdivision_for_country_and_token(country, token)
            if region:
                return LocationSegment(country=country, region=region)
        return LocationSegment(country=country, region=None)

    for token in tokens:
        inferred = infer_common_subdivision(token, has_context=has_context)
        if inferred is not None:
            return inferred
        stripped = _COUNTRY_SUFFIX_RE.sub("", token).strip()
        if stripped != token:
            inferred = infer_common_subdivision(stripped, has_context=True)
            if inferred is not None:
                return inferred

    city_resolved = resolve_city_segment(cleaned)
    if city_resolved is None:
        stripped_cleaned = _COUNTRY_SUFFIX_RE.sub("", cleaned).strip()
        if stripped_cleaned != cleaned:
            city_resolved = resolve_city_segment(stripped_cleaned)
    if city_resolved is None and "," in cleaned:
        for token in tokens:
            city_resolved = resolve_city_segment(token)
            if city_resolved is not None:
                break
    if city_resolved is not None:
        return city_resolved

    return LocationSegment(country=None, region=None)


def city_search_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def city_record_population(record: dict[str, Any]) -> int:
    try:
        return int(record.get("population") or 0)
    except (TypeError, ValueError):
        return 0


def resolve_city_segment(segment: str) -> LocationSegment | None:
    city_name = city_search_name(segment)
    if not city_name or "," in city_name:
        return None

    matches = _CITY_BY_NAME.get(normalize_lookup_text(city_name), [])
    if not matches:
        return None

    record = max(matches, key=city_record_population)
    country = country_from_code(record.get("countrycode"))
    if country is None:
        return None

    region = None
    if country == "US":
        admin1code = str(record.get("admin1code") or "").strip().upper()
        full_code = f"US-{admin1code}" if admin1code else ""
        if pycountry.subdivisions.get(code=full_code) is not None:
            region = admin1code

    return LocationSegment(country=country, region=region, city_resolved=True)


def split_location_segments(location_name: Any) -> list[str]:
    text = clean_location_text(location_name)
    if not text:
        return []
    parts = [part.strip(" ,-") for part in MULTI_LOCATION_RE.split(text) if part.strip(" ,-")]
    if not parts:
        return [text]

    expanded: list[str] = []
    for part in parts:
        if "-" in part and not ISO_SUBDIVISION_RE.search(part):
            sub_parts = [p.strip() for p in part.split("-") if p.strip()]
            if len(sub_parts) >= 2 and lookup_country_token(sub_parts[0]):
                expanded.extend(sub_parts)
                continue
        expanded.append(part)
    return expanded


def has_location_signal(text: Any) -> bool:
    cleaned = strip_noise(clean_location_text(text))
    return bool(normalize_lookup_text(cleaned))


def dedupe_segments(segments: list[LocationSegment]) -> list[LocationSegment]:
    deduped: list[LocationSegment] = []
    seen: set[tuple[str | None, str | None]] = set()
    for segment in segments:
        key = (segment.country, segment.region)
        if key in seen:
            if segment.city_resolved:
                index = next(
                    idx
                    for idx, existing in enumerate(deduped)
                    if (existing.country, existing.region) == key
                )
                existing = deduped[index]
                deduped[index] = LocationSegment(
                    country=existing.country,
                    region=existing.region,
                    city_resolved=True,
                )
            continue
        seen.add(key)
        deduped.append(segment)
    return deduped


def parse_location_name(location_name: Any) -> LocationNormalizationResult:
    work_arrangement = derive_work_arrangement(location_name)
    if location_name is None or not str(location_name).strip():
        return LocationNormalizationResult(
            location_country=None,
            location_region=None,
            location_segments=[],
            work_arrangement=work_arrangement,
            location_parse_status="missing",
        )

    if _N_LOCATIONS_RE.match(str(location_name).strip()):
        return LocationNormalizationResult(
            location_country=None,
            location_region=None,
            location_segments=[],
            work_arrangement=work_arrangement,
            location_parse_status="unparseable",
        )

    raw_segments = split_location_segments(location_name)
    parsed_segments = dedupe_segments([parse_segment(segment) for segment in raw_segments])
    resolved = [s for s in parsed_segments if s.country is not None]
    if resolved:
        parsed_segments = resolved
    if not parsed_segments:
        parsed_segments = [LocationSegment(country=None, region=None)]

    primary = parsed_segments[0]
    city_resolved = any(segment.city_resolved for segment in parsed_segments)
    if work_arrangement == "remote":
        status = "remote"
    elif city_resolved:
        status = "city_resolved"
    elif len(parsed_segments) > 1:
        status = "multi_location"
    elif primary.country and primary.region:
        status = "parsed"
    elif primary.country:
        status = "country_only"
    elif has_location_signal(location_name):
        status = "ambiguous"
    else:
        status = "unparseable"

    return LocationNormalizationResult(
        location_country=primary.country,
        location_region=primary.region,
        location_segments=[segment.as_dict() for segment in parsed_segments],
        work_arrangement=work_arrangement,
        location_parse_status=status,
    )


def get_location_normalization_batch_size() -> int:
    return max(
        1,
        int(
            os.environ.get(
                "NON_ML_BATCH_SIZE",
                os.environ.get("ENRICHMENT_BATCH_SIZE", str(DEFAULT_LOCATION_NORMALIZATION_BATCH_SIZE)),
            )
        ),
    )


def update_location_normalization(
    conn,
    *,
    only_missing: bool = True,
    batch_size: int | None = None,
) -> int:
    if batch_size is None:
        batch_size = get_location_normalization_batch_size()
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    recheck_ambiguous = os.environ.get("LOCATION_RECHECK_AMBIGUOUS", "").strip().lower() in {"true", "1", "yes", "on"}

    where_clause = "WHERE TRUE"
    params: tuple[str, ...] = ()
    if only_missing:
        if recheck_ambiguous:
            where_clause += " AND (location_normalization_version IS DISTINCT FROM %s OR location_parse_status = 'ambiguous')"
        else:
            where_clause += " AND location_normalization_version IS DISTINCT FROM %s"
        params = (LOCATION_NORMALIZATION_VERSION,)

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS candidate_count FROM jobs {where_clause}", params)
        total_candidates = int(cur.fetchone()["candidate_count"])

    print(
        f"normalize_location: {total_candidates} rows need location normalization; "
        f"committing in batches of {batch_size}",
        flush=True,
    )

    normalized_at = datetime.now(timezone.utc).replace(microsecond=0)
    status_counts = {status: 0 for status in VALID_PARSE_STATUSES}
    processed_count = 0
    last_id = 0

    if total_candidates <= 0:
        conn.commit()
        print(
            "normalize_location: completed 0 rows; "
            + ", ".join(f"{status}=0" for status in sorted(status_counts)),
            flush=True,
        )
        return 0

    while True:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, location_name
                FROM jobs
                {where_clause}
                  AND id > %s
                ORDER BY id
                LIMIT %s
                """,
                (*params, last_id, batch_size),
            )
            rows = cur.fetchall()

        if not rows:
            break

        update_rows = []
        for row in rows:
            result = parse_location_name(row.get("location_name"))
            status_counts[result.location_parse_status] += 1
            update_rows.append((
                result.location_country,
                result.location_region,
                Jsonb(result.location_segments),
                result.work_arrangement,
                result.location_parse_status,
                LOCATION_NORMALIZATION_VERSION,
                normalized_at,
                row["id"],
            ))

        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE jobs
                SET location_country = %s,
                    location_region = %s,
                    location_segments = %s,
                    work_arrangement = %s,
                    location_parse_status = %s,
                    location_normalization_version = %s,
                    location_normalized_at_utc = %s
                WHERE id = %s
                """,
                update_rows,
            )
        conn.commit()

        processed_count += len(rows)
        last_id = int(rows[-1]["id"])
        print(
            f"normalize_location: committed {processed_count}/{total_candidates}",
            flush=True,
        )

    print(
        "normalize_location: completed "
        f"{processed_count} rows; "
        + ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items())),
        flush=True,
    )
    return processed_count


def run(conn) -> int:
    return update_location_normalization(conn)


if __name__ == "__main__":
    from data_pipeline.common.db import connect

    with connect() as db_conn:
        run(db_conn)
