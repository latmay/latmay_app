from __future__ import annotations

import signal
import unittest
from unittest.mock import MagicMock, patch

import psycopg

from data_pipeline.common import db
from data_pipeline.enrichment import main_enrich_non_ml
from data_pipeline.ingestion import main_ingest


class FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[str, ...]]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[str, ...]) -> None:
        self.executed.append((sql, params))


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_obj = FakeCursor()
        self.commit_count = 0
        self.rollback_count = 0

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1


class DbTimeoutTests(unittest.TestCase):
    def test_env_timeout_milliseconds(self) -> None:
        with patch.dict("os.environ", {"EXAMPLE_TIMEOUT_SECONDS": "1.5"}, clear=True):
            self.assertEqual(db.env_timeout_milliseconds("EXAMPLE_TIMEOUT_SECONDS"), 1500)

        for raw_value in ("", "0", "-1", "not-a-number"):
            with patch.dict("os.environ", {"EXAMPLE_TIMEOUT_SECONDS": raw_value}, clear=True):
                self.assertIsNone(db.env_timeout_milliseconds("EXAMPLE_TIMEOUT_SECONDS"))

    def test_configure_connection_timeouts_sets_session_values(self) -> None:
        conn = FakeConnection()

        with patch.dict(
            "os.environ",
            {
                "INGESTION_DB_STATEMENT_TIMEOUT_SECONDS": "120",
                "INGESTION_DB_LOCK_TIMEOUT_SECONDS": "30",
            },
            clear=True,
        ):
            db.configure_connection_timeouts(conn)  # type: ignore[arg-type]

        self.assertEqual(
            conn.cursor_obj.executed,
            [
                ("SELECT set_config('statement_timeout', %s, false)", ("120000ms",)),
                ("SELECT set_config('lock_timeout', %s, false)", ("30000ms",)),
            ],
        )
        self.assertEqual(conn.commit_count, 1)

    def test_configure_connection_timeouts_skips_when_unset(self) -> None:
        conn = FakeConnection()

        with patch.dict("os.environ", {}, clear=True):
            db.configure_connection_timeouts(conn)  # type: ignore[arg-type]

        self.assertEqual(conn.cursor_obj.executed, [])
        self.assertEqual(conn.commit_count, 0)

    def test_connect_applies_timeouts_before_yielding_connection(self) -> None:
        conn = MagicMock()
        connection_context = MagicMock()
        connection_context.__enter__.return_value = conn

        with (
            patch.dict("os.environ", {"LOCAL_DATABASE_URL": "postgresql://test:test@localhost:5432/test"}),
            patch.object(db.psycopg, "connect", return_value=connection_context, create=True),
            patch.object(db, "configure_connection_timeouts") as configure_timeouts,
            db.connect() as yielded_conn,
        ):
            self.assertIs(yielded_conn, conn)
            configure_timeouts.assert_called_once_with(conn)

    def test_ingestion_uses_shared_configured_connection(self) -> None:
        events: list[str] = []
        conn = FakeConnection()

        class FakeConnect:
            def __enter__(self) -> FakeConnection:
                events.append("connect")
                return conn

            def __exit__(self, *args: object) -> None:
                events.append("disconnect")

        with (
            patch.object(main_ingest, "connect", return_value=FakeConnect()),
            patch.object(main_ingest, "initialize_schema", side_effect=lambda _conn: events.append("schema")),
            patch.object(main_ingest, "run_ats_ingestion_steps", side_effect=lambda _conn: events.append("ats")),
            patch.object(main_ingest, "run_or_skip", side_effect=lambda *args, **kwargs: events.append("content_fill")),
            patch.object(main_ingest, "print_posted_at_quality_summary", side_effect=lambda _conn: events.append("summary")),
        ):
            main_ingest.run_ingestion_job()

        self.assertEqual(
            events,
            ["connect", "schema", "ats", "content_fill", "summary", "disconnect"],
        )

    def test_non_ml_sigterm_cancels_rolls_back_and_closes_connection(self) -> None:
        conn = MagicMock()
        installed_handler = None

        def capture_handler(_signal_number: int, handler: object) -> None:
            nonlocal installed_handler
            installed_handler = handler

        with (
            patch.object(main_enrich_non_ml.signal, "getsignal", return_value=signal.SIG_DFL),
            patch.object(main_enrich_non_ml.signal, "signal", side_effect=capture_handler),
            main_enrich_non_ml.cancel_connection_on_sigterm(conn),
        ):
            self.assertIsNotNone(installed_handler)
            with self.assertRaisesRegex(SystemExit, "143"):
                installed_handler(signal.SIGTERM, None)  # type: ignore[operator]

        conn.cancel.assert_called_once_with()
        conn.rollback.assert_called_once_with()
        conn.close.assert_called_once_with()

    def test_run_or_skip_rolls_back_and_continues_on_optional_db_cancel(self) -> None:
        conn = FakeConnection()

        with patch.dict("os.environ", {"ENABLE_CONTENT_FILL": "true"}, clear=True):
            main_ingest.run_or_skip(
                "content_fill",
                "ENABLE_CONTENT_FILL",
                lambda: (_ for _ in ()).throw(psycopg.errors.QueryCanceled("statement timeout")),
                conn=conn,
                continue_on_db_cancel=True,
            )

        self.assertEqual(conn.rollback_count, 1)

    def test_run_or_skip_reraises_db_cancel_when_not_marked_continuable(self) -> None:
        conn = FakeConnection()

        with patch.dict("os.environ", {"ENABLE_WORKDAY_INGESTION": "true"}, clear=True):
            with self.assertRaises(psycopg.errors.QueryCanceled):
                main_ingest.run_or_skip(
                    "workday",
                    "ENABLE_WORKDAY_INGESTION",
                    lambda: (_ for _ in ()).throw(psycopg.errors.QueryCanceled("statement timeout")),
                    conn=conn,
                    continue_on_db_cancel=False,
                )

        self.assertEqual(conn.rollback_count, 0)


if __name__ == "__main__":
    unittest.main()
