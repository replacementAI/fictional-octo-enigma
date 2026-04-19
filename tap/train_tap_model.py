from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .build_dataset import build_training_dataset
from .config import RANDOM_STATE, get_default_paths
from .evaluate_tap_model import evaluate_model
from .split_data import make_group_split


NON_FEATURE_COLUMNS = {
    "subject_id",
    "session_id",
    "session_month",
    "parkinsons",
    "gender",
    "sided",
    "impact",
    "updrs",
}


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]
    X = df[feature_columns].copy()
    y = df["parkinsons"].astype(int).copy()
    return X, y, feature_columns


def train_logistic_regression(X_train, y_train):
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def fit_candidate_models(X_train, y_train) -> dict[str, Any]:
    return {
        "logistic_regression": train_logistic_regression(X_train, y_train),
        "random_forest": train_random_forest(X_train, y_train),
        "gradient_boosting": train_gradient_boosting(X_train, y_train),
    }


def select_best_model(models: dict[str, Any], X_val, y_val) -> tuple[str, Any, dict[str, dict[str, float]]]:
    metrics_by_model: dict[str, dict[str, float]] = {}
    best_name = ""
    best_model = None
    best_score = float("-inf")

    for name, model in models.items():
        metrics = evaluate_model(model, X_val, y_val)
        metrics_by_model[name] = metrics
        if metrics["roc_auc"] > best_score:
            best_name = name
            best_model = model
            best_score = metrics["roc_auc"]

    if best_model is None:
        raise RuntimeError("No candidate tap models were trained.")
    return best_name, best_model, metrics_by_model


def save_model_artifacts(
    model: Any,
    feature_columns: list[str],
    metrics: dict[str, Any],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / "tap_model.pkl")
    (output_dir / "tap_feature_columns.json").write_text(json.dumps(feature_columns, indent=2))
    (output_dir / "tap_metrics.json").write_text(json.dumps(metrics, indent=2))


def main() -> None:
    paths = get_default_paths()
    dataset = build_training_dataset(paths.tappy_data_dir, paths.users_dir)
    train_df, val_df, test_df = make_group_split(dataset)

    X_train, y_train, feature_columns = prepare_features_and_target(train_df)
    X_val = val_df[feature_columns]
    y_val = val_df["parkinsons"].astype(int)
    X_test = test_df[feature_columns]
    y_test = test_df["parkinsons"].astype(int)

    models = fit_candidate_models(X_train, y_train)
    best_name, best_model, val_metrics = select_best_model(models, X_val, y_val)
    test_metrics = evaluate_model(best_model, X_test, y_test)

    save_model_artifacts(
        model=best_model,
        feature_columns=feature_columns,
        metrics={
            "selected_model": best_name,
            "validation": val_metrics,
            "test": test_metrics,
            "dataset_rows": int(len(dataset)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        output_dir=paths.artifacts_dir,
    )


if __name__ == "__main__":
    main()
