from __future__ import annotations

import unittest

import requests

from data_pipeline.ingestion.size_limits import (
    ResponseTooLarge,
    limit_job_record_storage,
    read_response_with_limit,
    truncate_utf8,
)


class IngestionSizeLimitTests(unittest.TestCase):
    def test_stream_limit_counts_decompressed_chunks(self) -> None:
        response = requests.Response()
        response.status_code = 200
        response.url = "https://example.test/jobs"
        response.headers = {}
        response.raw = type(
            "Raw",
            (),
            {"stream": lambda _self, _chunk_size, decode_content=True: iter([b"1234", b"5678"])},
        )()

        with self.assertRaises(ResponseTooLarge):
            read_response_with_limit(response, 7)

    def test_utf8_truncation_does_not_split_character(self) -> None:
        value, truncated = truncate_utf8("abcé", 4)
        self.assertEqual(value, "abc")
        self.assertTrue(truncated)

    def test_raw_json_omission_preserves_fingerprint(self) -> None:
        record = {
            "content_html": "ok",
            "content_text": "ok",
            "raw_json": {"listing_fingerprint": "same", "large": "x" * (2 * 1024 * 1024)},
        }
        limited = limit_job_record_storage(record)
        self.assertEqual(limited["raw_json"]["listing_fingerprint"], "same")
        self.assertTrue(limited["raw_json"]["oversize_raw_json_omitted"])


if __name__ == "__main__":
    unittest.main()
