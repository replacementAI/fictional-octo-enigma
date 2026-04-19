from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def _safe_cv(values: pd.Series) -> float:
    mean = float(values.mean()) if len(values) else 0.0
    if not values.size or np.isclose(mean, 0.0):
        return 0.0
    return float(values.std(ddof=0) / mean)


def _summary_stats(values: pd.Series, prefix: str) -> dict[str, float]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_cv": 0.0,
            f"{prefix}_skew": 0.0,
            f"{prefix}_kurtosis": 0.0,
        }

    return {
        f"{prefix}_mean": float(clean.mean()),
        f"{prefix}_std": float(clean.std(ddof=0)),
        f"{prefix}_median": float(clean.median()),
        f"{prefix}_p10": float(clean.quantile(0.10)),
        f"{prefix}_p90": float(clean.quantile(0.90)),
        f"{prefix}_cv": _safe_cv(clean),
        f"{prefix}_skew": float(skew(clean, bias=False)) if len(clean) > 2 else 0.0,
        f"{prefix}_kurtosis": float(kurtosis(clean, fisher=True, bias=False)) if len(clean) > 3 else 0.0,
    }


def compute_basic_timing_features(session_df: pd.DataFrame) -> dict[str, float]:
    features: dict[str, float] = {
        "n_events": float(len(session_df)),
    }
    if "event_timestamp" in session_df and session_df["event_timestamp"].notna().any():
        ordered = session_df.sort_values("event_timestamp")
        start = ordered["event_timestamp"].iloc[0]
        end = ordered["event_timestamp"].iloc[-1]
        duration = (end - start).total_seconds() if start and end else 0.0
        features["session_duration_s"] = float(max(duration, 0.0))
    else:
        features["session_duration_s"] = 0.0

    features.update(_summary_stats(session_df["hold_time_ms"], "hold"))
    features.update(_summary_stats(session_df["latency_time_ms"], "latency"))
    features.update(_summary_stats(session_df["flight_time_ms"], "flight"))
    return features


def compute_transition_features(session_df: pd.DataFrame) -> dict[str, float]:
    transitions = session_df["transition_code"].fillna("")
    total = max(len(transitions), 1)
    counts = transitions.value_counts()

    ll_prop = float(counts.get("LL", 0) / total)
    lr_prop = float(counts.get("LR", 0) / total)
    rl_prop = float(counts.get("RL", 0) / total)
    rr_prop = float(counts.get("RR", 0) / total)
    ls_prop = float(counts.get("LS", 0) / total)
    sl_prop = float(counts.get("SL", 0) / total)

    return {
        "ll_prop": ll_prop,
        "lr_prop": lr_prop,
        "rl_prop": rl_prop,
        "rr_prop": rr_prop,
        "ls_prop": ls_prop,
        "sl_prop": sl_prop,
        "lr_rl_diff": abs(lr_prop - rl_prop),
        "ll_rr_diff": abs(ll_prop - rr_prop),
    }


def compute_asymmetry_features(session_df: pd.DataFrame) -> dict[str, float]:
    left = session_df[session_df["hand_code"] == "L"]
    right = session_df[session_df["hand_code"] == "R"]
    special = session_df[session_df["hand_code"] == "S"]
    total = max(len(session_df), 1)

    left_hold_mean = float(left["hold_time_ms"].mean()) if not left.empty else 0.0
    right_hold_mean = float(right["hold_time_ms"].mean()) if not right.empty else 0.0

    return {
        "left_hold_mean": left_hold_mean,
        "right_hold_mean": right_hold_mean,
        "left_hold_std": float(left["hold_time_ms"].std(ddof=0)) if len(left) > 1 else 0.0,
        "right_hold_std": float(right["hold_time_ms"].std(ddof=0)) if len(right) > 1 else 0.0,
        "left_right_hold_diff": abs(left_hold_mean - right_hold_mean),
        "left_prop": float(len(left) / total),
        "right_prop": float(len(right) / total),
        "special_key_prop": float(len(special) / total),
    }


def _half_delta(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    if len(clean) < 4:
        return 0.0
    midpoint = len(clean) // 2
    first = clean.iloc[:midpoint]
    second = clean.iloc[midpoint:]
    if first.empty or second.empty:
        return 0.0
    return float(second.mean() - first.mean())


def compute_fatigue_features(session_df: pd.DataFrame) -> dict[str, float]:
    ordered = session_df.sort_values("event_timestamp") if "event_timestamp" in session_df else session_df
    return {
        "hold_mean_delta_second_half": _half_delta(ordered["hold_time_ms"]),
        "latency_mean_delta_second_half": _half_delta(ordered["latency_time_ms"]),
        "flight_mean_delta_second_half": _half_delta(ordered["flight_time_ms"]),
    }


def compute_session_features(session_df: pd.DataFrame) -> dict[str, Any]:
    session_df = session_df.copy()
    features: dict[str, Any] = {
        "subject_id": session_df["subject_id"].iloc[0],
        "session_id": session_df["session_id"].iloc[0],
        "session_month": session_df["session_month"].iloc[0],
    }
    features.update(compute_basic_timing_features(session_df))
    features.update(compute_transition_features(session_df))
    features.update(compute_asymmetry_features(session_df))
    features.update(compute_fatigue_features(session_df))
    return features


def build_session_feature_table(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    grouped = events_df.groupby("session_id", sort=True)
    records = [compute_session_features(group) for _, group in grouped]
    return pd.DataFrame.from_records(records)
