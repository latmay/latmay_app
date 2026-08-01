from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
import os
import time
from typing import Any


@dataclass
class RankingTimingEvent:
    step_name: str
    elapsed_seconds: float
    metadata: dict[str, Any]


_ranking_timing_events: ContextVar[list[RankingTimingEvent] | None] = ContextVar(
    "ranking_timing_events",
    default=None,
)


def start_ranking_timing_collection() -> Token[list[RankingTimingEvent] | None]:
    return _ranking_timing_events.set([])


def record_ranking_timing(step_name: str, started_at: float, **metadata: Any) -> None:
    event = RankingTimingEvent(
        step_name=step_name,
        elapsed_seconds=time.perf_counter() - started_at,
        metadata=dict(metadata),
    )
    events = _ranking_timing_events.get()
    if events is None:
        print(_format_timing_line("Ranking timing", event), flush=True)
        return
    events.append(event)


def print_ranking_timing_summary() -> None:
    if os.environ.get("ENABLE_RANKING_TIMING_SUMMARY", "true").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    events = _ranking_timing_events.get()
    if events is None:
        return

    print(f"Ranking timing summary start: events={len(events)}", flush=True)
    for index, event in enumerate(events, start=1):
        print(_format_timing_line("Ranking timing summary", event, index=index), flush=True)
    print("Ranking timing summary end", flush=True)


def reset_ranking_timing_collection(token: Token[list[RankingTimingEvent] | None]) -> None:
    _ranking_timing_events.reset(token)


def _format_timing_line(prefix: str, event: RankingTimingEvent, *, index: int | None = None) -> str:
    fields: list[tuple[str, Any]] = []
    if index is not None:
        fields.append(("index", f"{index:03d}"))
    fields.extend(
        [
            ("step", event.step_name),
            ("elapsed_seconds", f"{event.elapsed_seconds:.3f}"),
        ]
    )
    fields.extend(event.metadata.items())
    return f"{prefix}: " + ", ".join(f"{key}={value}" for key, value in fields)
