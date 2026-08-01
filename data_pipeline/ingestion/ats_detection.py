from __future__ import annotations

"""
ATS detection and canonical URL construction for known job-board providers.

Pure functions — no I/O, no external dependencies beyond stdlib.
"""

import json
import re
from typing import Any, Callable, Iterable
from urllib.parse import quote, urlparse


def parse_workday_site(site_url: str) -> dict[str, str] | None:
    parsed = urlparse(site_url)
    if not parsed.scheme or not parsed.netloc:
        return None
    path_parts = [part for part in parsed.path.split("/") if part]
    locale = "en-US"
    if path_parts and re.fullmatch(r"[a-z]{2}-[A-Z]{2}", path_parts[0]):
        locale = path_parts[0]
        path_parts = path_parts[1:]
    if not path_parts:
        return None
    tenant = parsed.netloc.split(".", 1)[0]
    site_slug = path_parts[-1]
    return {
        "scheme": parsed.scheme,
        "host": parsed.netloc,
        "tenant": tenant,
        "locale": locale,
        "site_slug": site_slug,
        "source_company": f"{tenant}:{site_slug}",
        "source_url": site_url,
    }


def workday_jobs_endpoint(site_info: dict[str, str]) -> str:
    return f"{site_info['scheme']}://{site_info['host']}/wday/cxs/{site_info['tenant']}/{site_info['site_slug']}/jobs"


def ats_match_from_url(url: str) -> dict[str, Any] | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    parts = [part for part in parsed.path.strip("/").split("/") if part]

    if "greenhouse.io" in host:
        token = ""
        if host in {"boards.greenhouse.io", "job-boards.greenhouse.io"} and parts:
            token = parts[0]
        elif "boards-api.greenhouse.io" in host and len(parts) >= 3 and parts[0] == "v1" and parts[1] == "boards":
            token = parts[2]
        if token:
            return {"provider": "greenhouse", "identifier": token, "matched_url": url}
        return {"provider": "greenhouse", "identifier": "", "matched_url": url}

    if "lever.co" in host:
        token = ""
        if host == "jobs.lever.co" and parts:
            token = parts[0]
        elif host == "api.lever.co" and len(parts) >= 3 and parts[0] == "v0" and parts[1] == "postings":
            token = parts[2]
        if token:
            return {"provider": "lever", "identifier": token, "matched_url": url}
        return {"provider": "lever", "identifier": "", "matched_url": url}

    if "ashbyhq.com" in host:
        token = ""
        if host == "jobs.ashbyhq.com" and parts:
            token = parts[0]
        elif host == "api.ashbyhq.com" and len(parts) >= 3 and parts[0] == "posting-api" and parts[1] == "job-board":
            token = parts[2]
        if token:
            return {"provider": "ashby", "identifier": token, "matched_url": url}
        return {"provider": "ashby", "identifier": "", "matched_url": url}

    if "jibeapply.com" in host or (parts[:2] == ["api", "jobs"]) or parsed.path.rstrip("/") == "/api/jobs":
        return {"provider": "icims", "identifier": host, "matched_url": url}

    if "myworkdayjobs.com" in host or "myworkdaysite.com" in host:
        site_info = parse_workday_site(url)
        if site_info:
            return {"provider": "workday", "identifier": site_info["source_company"], "matched_url": url, "site_info": site_info}
        return {"provider": "workday", "identifier": "", "matched_url": url}

    provider_hosts = (
        ("icims", ("icims.com",)),
        ("smartrecruiters", ("smartrecruiters.com",)),
        ("workable", ("workable.com",)),
        ("jobvite", ("jobvite.com",)),
        ("bamboohr", ("bamboohr.com",)),
        ("sap_successfactors", ("successfactors", "sapsf.com")),
        ("oracle_cloud_recruiting", ("oraclecloud.com", "oraclecloudcloud.com")),
        ("adp", ("adp.com", "workforcenow.adp.com")),
        ("rippling", ("rippling.com",)),
        ("eightfold", ("eightfold.ai",)),
        ("avature", ("avature.net",)),
    )
    for provider, host_hints in provider_hosts:
        if any(hint in host for hint in host_hints):
            return {"provider": provider, "identifier": "", "matched_url": url}
    return None


def detect_known_ats(signal_urls: Iterable[str]) -> dict[str, Any] | None:
    for url in signal_urls:
        match = ats_match_from_url(url)
        if match:
            return match
    return None


def ats_source_entry_url(ats: dict[str, Any]) -> str | None:
    """Return the canonical ATS job-listing URL suitable for ats_career_sources.json."""
    provider = ats.get("provider")
    token = str(ats.get("identifier") or "").strip()
    if provider == "greenhouse" and token:
        return f"https://boards-api.greenhouse.io/v1/boards/{quote(token)}/jobs?content=true"
    if provider == "lever" and token:
        return f"https://api.lever.co/v0/postings/{quote(token)}?mode=json"
    if provider == "ashby" and token:
        return f"https://api.ashbyhq.com/posting-api/job-board/{quote(token)}?includeCompensation=true"
    if provider == "icims" and ats.get("matched_url"):
        return str(ats["matched_url"])
    if provider == "workday" and isinstance(ats.get("site_info"), dict):
        return ats["site_info"].get("source_url")
    return None


ATS_PROBE_CONFIGS: list[dict[str, Any]] = [
    {
        "provider": "greenhouse",
        "url_template": "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true",
        "valid": lambda r: isinstance(r, dict) and isinstance(r.get("jobs"), list),
    },
    {
        "provider": "lever",
        "url_template": "https://api.lever.co/v0/postings/{slug}?mode=json",
        "valid": lambda r: isinstance(r, list),
    },
    {
        "provider": "ashby",
        "url_template": "https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true",
        "valid": lambda r: isinstance(r, dict) and isinstance(r.get("jobs"), list),
    },
]


def ats_probe_slugs(company: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    def add(s: str | None) -> None:
        if not s:
            return
        s = s.strip().lower()
        if s and s not in seen:
            seen.add(s)
            result.append(s)

    add(company.get("source_company"))

    website = str(company.get("website") or "").strip()
    if website:
        host = urlparse(website).netloc.lower().lstrip("www.")
        add(host.split(".")[0])

    name = str(company.get("company_name") or "").strip()
    if name:
        add(re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"))
        add(re.sub(r"[^a-z0-9]+", "", name.lower()))

    return result


def probe_ats_endpoints(
    company: dict[str, Any],
    fetch_fn: Callable[[str], tuple[int, str | None]],
    diagnostics: dict[str, Any],
) -> dict[str, Any] | None:
    """Second-pass ATS detection: probe known API endpoints with guessed slugs.

    fetch_fn(url) must return (status_code, response_text_or_none).
    Records every attempt under diagnostics['ats_probe_results'].
    """
    slugs = ats_probe_slugs(company)
    results: list[dict[str, Any]] = []
    for slug in slugs:
        for cfg in ATS_PROBE_CONFIGS:
            url = cfg["url_template"].format(slug=quote(slug, safe=""))
            status, text = fetch_fn(url)
            entry: dict[str, Any] = {
                "provider": cfg["provider"],
                "slug": slug,
                "url": url,
                "status": status,
                "matched": False,
            }
            if status == 200 and text:
                try:
                    payload = json.loads(text)
                    if cfg["valid"](payload):
                        entry["matched"] = True
                        results.append(entry)
                        diagnostics["ats_probe_results"] = results
                        return {"provider": cfg["provider"], "identifier": slug, "matched_url": url}
                except Exception:
                    pass
            results.append(entry)
    diagnostics["ats_probe_results"] = results
    return None
