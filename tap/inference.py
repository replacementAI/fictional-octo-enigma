from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import get_default_paths
from .feature_engineering import compute_session_features
from .model_contract import build_modality_result


def load_tap_model(artifacts_dir: str | Path | None = None) -> tuple[Any, list[str]]:
    paths = get_default_paths()
    artifacts_path = Path(artifacts_dir) if artifacts_dir is not None else paths.artifacts_dir
    model = joblib.load(artifacts_path / "tap_model.pkl")
    feature_columns = json.loads((artifacts_path / "tap_feature_columns.json").read_text())
    return model, feature_columns


def transform_session_to_features(session_df: pd.DataFrame) -> pd.DataFrame:
    features = compute_session_features(session_df)
    return pd.DataFrame([features])


def predict_tap_risk(session_df: pd.DataFrame, artifacts_dir: str | Path | None = None) -> dict[str, Any]:
    model, feature_columns = load_tap_model(artifacts_dir)
    feature_frame = transform_session_to_features(session_df)
    probability = float(model.predict_proba(feature_frame[feature_columns])[:, 1][0])
    warnings: list[str] = []
    sample_count = int(feature_frame.iloc[0].get("n_events", 0))
    if sample_count < 25:
        warnings.append("too_few_tap_events")

    return build_modality_result(
        modality="tap",
        model_version="tap_v1",
        risk_score=probability,
        raw_features=feature_frame.iloc[0].to_dict(),
        confidence=None,
        input_quality=1.0 if sample_count >= 25 else 0.5,
        sample_count=sample_count,
        warnings=warnings,
        feature_schema_version="v1",
        primary_signal="Tap timing variability",
    )
