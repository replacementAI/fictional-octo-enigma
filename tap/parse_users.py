from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .schemas import SUBJECT_COLUMNS


def _parse_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _parse_int(value: str) -> int | None:
    value = value.strip()
    if not value or value.lower() == "don't know":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def normalize_user_record(record: dict[str, str]) -> dict[str, Any]:
    subject_id = record.get("subject_id", "").strip()
    return {
        "subject_id": subject_id,
        "birth_year": _parse_int(record.get("BirthYear", "")),
        "gender": record.get("Gender", "").strip() or None,
        "parkinsons": _parse_bool(record.get("Parkinsons", "")),
        "tremors": _parse_bool(record.get("Tremors", "")),
        "diagnosis_year": _parse_int(record.get("DiagnosisYear", "")),
        "sided": record.get("Sided", "").strip() or None,
        "updrs": record.get("UPDRS", "").strip() or None,
        "impact": record.get("Impact", "").strip() or None,
        "levadopa": _parse_bool(record.get("Levadopa", "")),
        "da": _parse_bool(record.get("DA", "")),
        "maob": _parse_bool(record.get("MAOB", "")),
        "other_med": _parse_bool(record.get("Other", "")),
    }


def parse_user_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    record: dict[str, str] = {
        "subject_id": path.stem.replace("User_", "", 1),
    }

    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        record[key.strip()] = value.strip()

    return normalize_user_record(record)


def parse_all_user_files(users_dir: str | Path) -> pd.DataFrame:
    users_dir = Path(users_dir)
    records = [
        parse_user_file(path)
        for path in sorted(users_dir.glob("User_*.txt"))
    ]
    if not records:
        return pd.DataFrame(columns=SUBJECT_COLUMNS)
    return pd.DataFrame.from_records(records, columns=SUBJECT_COLUMNS)
