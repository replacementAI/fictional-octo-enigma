from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import EVENT_COLUMNS


def extract_session_metadata(filename: str) -> dict[str, str]:
    stem = Path(filename).stem
    subject_id, session_month = stem.rsplit("_", 1)
    return {
        "subject_id": subject_id,
        "session_id": stem,
        "session_month": session_month,
    }


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def parse_tappy_line(line: str, session_id: str) -> dict[str, Any] | None:
    parts = line.strip().split("\t")
    if len(parts) != 8:
        return None

    subject_id, event_date, event_time, hand_code, hold_time, transition_code, latency_time, flight_time = parts
    try:
        event_timestamp = datetime.strptime(
            f"{event_date} {event_time}",
            "%y%m%d %H:%M:%S.%f",
        )
    except ValueError:
        event_timestamp = None

    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "session_month": session_id.rsplit("_", 1)[-1],
        "event_date": event_date,
        "event_time": event_time,
        "event_timestamp": event_timestamp,
        "hand_code": hand_code,
        "hold_time_ms": _parse_float(hold_time),
        "transition_code": transition_code,
        "latency_time_ms": _parse_float(latency_time),
        "flight_time_ms": _parse_float(flight_time),
    }


def parse_tappy_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    metadata = extract_session_metadata(path.name)
    rows: list[dict[str, Any]] = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = parse_tappy_line(line, metadata["session_id"])
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame.from_records(rows, columns=EVENT_COLUMNS)


def parse_all_tappy_files(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    frames = [
        parse_tappy_file(path)
        for path in sorted(data_dir.glob("*.txt"))
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.concat(frames, ignore_index=True)
