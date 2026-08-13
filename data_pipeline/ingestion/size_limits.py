from __future__ import annotations

"""Byte limits for ingestion HTTP responses and stored job fields."""

import json
import os
from typing import Any

import requests


MIB = 1024 * 1024


def env_mib(name: str, default_mib: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default_mib * MIB
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number of MiB.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive number of MiB.")
    return int(value * MIB)


MAX_DETAIL_RESPONSE_BYTES = env_mib("INGESTION_MAX_DETAIL_RESPONSE_MIB", 8)
MAX_LISTING_RESPONSE_BYTES = env_mib("INGESTION_MAX_LISTING_RESPONSE_MIB", 32)
MAX_CONTENT_HTML_BYTES = env_mib("INGESTION_MAX_CONTENT_HTML_MIB", 4)
MAX_CONTENT_TEXT_BYTES = env_mib("INGESTION_MAX_CONTENT_TEXT_MIB", 1)
MAX_RAW_JSON_BYTES = env_mib("INGESTION_MAX_RAW_JSON_MIB", 2)


class ResponseTooLarge(requests.RequestException):
    def __init__(self, *, limit_bytes: int, url: str):
        super().__init__(f"Decompressed response exceeded {limit_bytes} bytes", request=None, response=None)
        self.limit_bytes = limit_bytes
        self.url = url


def close_response(response: requests.Response) -> None:
    try:
        response.close()
    except Exception:
        pass


def read_response_with_limit(response: requests.Response, limit_bytes: int) -> requests.Response:
    """Buffer a streamed response while enforcing a decompressed byte limit."""
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            if int(content_length) > limit_bytes and not response.headers.get("Content-Encoding"):
                close_response(response)
                raise ResponseTooLarge(limit_bytes=limit_bytes, url=response.url)
        except ValueError:
            pass

    # A small compatibility path for lightweight response doubles in unit tests.
    if not hasattr(response, "iter_content"):
        content = getattr(response, "content", None)
        if content is None:
            content = str(getattr(response, "text", "")).encode("utf-8")
        if len(content) > limit_bytes:
            raise ResponseTooLarge(limit_bytes=limit_bytes, url=getattr(response, "url", "unknown"))
        return response

    chunks: list[bytes] = []
    total = 0
    try:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > limit_bytes:
                raise ResponseTooLarge(limit_bytes=limit_bytes, url=response.url)
            chunks.append(chunk)
    except Exception:
        close_response(response)
        raise

    response._content = b"".join(chunks)
    response._content_consumed = True
    return response


def truncate_utf8(value: Any, limit_bytes: int) -> tuple[Any, bool]:
    if not isinstance(value, str):
        return value, False
    encoded = value.encode("utf-8")
    if len(encoded) <= limit_bytes:
        return value, False
    return encoded[:limit_bytes].decode("utf-8", errors="ignore"), True


def limit_job_record_storage(record: dict[str, Any]) -> dict[str, Any]:
    """Bound large stored fields and record safe truncation/omission metadata."""
    html, html_truncated = truncate_utf8(record.get("content_html"), MAX_CONTENT_HTML_BYTES)
    text, text_truncated = truncate_utf8(record.get("content_text"), MAX_CONTENT_TEXT_BYTES)
    record["content_html"] = html
    record["content_text"] = text

    raw_json = record.get("raw_json") or {}
    if html_truncated or text_truncated:
        if isinstance(raw_json, dict):
            raw_json = dict(raw_json)
        else:
            raw_json = {"original_raw_json_type": type(raw_json).__name__}
        raw_json["ingestion_size_limits"] = {
            "content_html_truncated": html_truncated,
            "content_text_truncated": text_truncated,
            "raw_json_omitted": False,
        }

    raw_json_bytes = json.dumps(raw_json, ensure_ascii=False, default=str).encode("utf-8")
    raw_json_omitted = len(raw_json_bytes) > MAX_RAW_JSON_BYTES
    if raw_json_omitted:
        fingerprint = raw_json.get("listing_fingerprint") if isinstance(raw_json, dict) else None
        raw_json = {
            "oversize_raw_json_omitted": True,
            "original_size_bytes": len(raw_json_bytes),
            "ingestion_size_limits": {
                "content_html_truncated": html_truncated,
                "content_text_truncated": text_truncated,
                "raw_json_omitted": True,
            },
        }
        if fingerprint:
            raw_json["listing_fingerprint"] = fingerprint
    record["raw_json"] = raw_json
    return record
