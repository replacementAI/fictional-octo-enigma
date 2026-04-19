from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import MIN_EVENTS_PER_SESSION, get_default_paths
from .feature_engineering import build_session_feature_table
from .parse_tappy import parse_all_tappy_files
from .parse_users import parse_all_user_files


def merge_sessions_with_subjects(
    session_features: pd.DataFrame,
    subjects_df: pd.DataFrame,
) -> pd.DataFrame:
    return session_features.merge(subjects_df, on="subject_id", how="left")


def filter_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df[df["parkinsons"].notna()].copy()
    filtered = filtered[filtered["n_events"] >= MIN_EVENTS_PER_SESSION].copy()
    filtered["parkinsons"] = filtered["parkinsons"].astype(int)
    return filtered.reset_index(drop=True)


def build_training_dataset(
    data_dir: str | Path | None = None,
    users_dir: str | Path | None = None,
) -> pd.DataFrame:
    paths = get_default_paths()
    tappy_dir = Path(data_dir) if data_dir is not None else paths.tappy_data_dir
    user_dir = Path(users_dir) if users_dir is not None else paths.users_dir

    events_df = parse_all_tappy_files(tappy_dir)
    subjects_df = parse_all_user_files(user_dir)
    session_features = build_session_feature_table(events_df)
    merged = merge_sessions_with_subjects(session_features, subjects_df)
    return filter_training_rows(merged)
