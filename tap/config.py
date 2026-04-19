from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    os.getenv(
        "PARKSENSE_TAP_DATA_ROOT",
        "/Users/salmansolanki/Downloads/archive-2",
    )
)


@dataclass(frozen=True)
class TapPipelinePaths:
    tappy_data_dir: Path
    users_dir: Path
    artifacts_dir: Path


RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
MIN_EVENTS_PER_SESSION = 25


def get_default_paths() -> TapPipelinePaths:
    return TapPipelinePaths(
        tappy_data_dir=DEFAULT_DATA_ROOT / "Archived-Data" / "Tappy Data",
        users_dir=DEFAULT_DATA_ROOT / "Archived-users" / "Archived users",
        artifacts_dir=BASE_DIR / "artifacts",
    )
