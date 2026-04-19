from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE


def make_group_split(
    df: pd.DataFrame,
    group_col: str = "subject_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    groups = df[group_col]
    train_val_idx, test_idx = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        ).split(df, groups=groups)
    )
    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    val_fraction = VALIDATION_SIZE / (1.0 - TEST_SIZE)
    train_idx, val_idx = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=val_fraction,
            random_state=RANDOM_STATE,
        ).split(train_val, groups=train_val[group_col])
    )
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)
    return train, val, test
