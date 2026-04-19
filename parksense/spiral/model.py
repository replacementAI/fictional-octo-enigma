from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "Spiral_HandPD.csv"
ARTIFACT_PATH = ROOT_DIR / "artifacts" / "spiral_pipeline.joblib"

FEATURES = [
    "RMS",
    "MAX_BETWEEN_ET_HT",
    "MIN_BETWEEN_ET_HT",
    "STD_DEVIATION_ET_HT",
    "MRT",
    "MAX_HT",
    "MIN_HT",
    "STD_HT",
    "CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT",
]

TARGET_COLUMN = "CLASS_TYPE"
PATIENT_ID_COLUMN = "ID_PATIENT"
CLASS_LABELS = {
    1: "Parkinson",
    2: "Healthy",
}


def load_spiral_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    dataset_path = Path(path).expanduser().resolve()
    dataframe = pd.read_csv(dataset_path)
    dataframe.columns = dataframe.columns.str.strip()
    return dataframe


def build_training_frame(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return dataframe[FEATURES].copy(), dataframe[TARGET_COLUMN].copy()


def build_group_labels(dataframe: pd.DataFrame) -> pd.Series:
    # HandPD reuses ID_PATIENT values across classes, so the group must include class.
    return (
        dataframe[TARGET_COLUMN].astype(str).str.strip()
        + "_"
        + dataframe[PATIENT_ID_COLUMN].astype(str).str.strip()
    )


def build_prediction_frame(records: pd.DataFrame | dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    elif isinstance(records, dict):
        frame = pd.DataFrame([records])
    else:
        frame = pd.DataFrame(records)
    return frame.reindex(columns=FEATURES)


def build_spiral_pipeline(random_state: int = 42) -> Pipeline:
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def summarize_dataset(dataframe: pd.DataFrame, path: str | Path = DATA_PATH) -> dict[str, Any]:
    groups = build_group_labels(dataframe)
    class_counts = {
        str(int(label)): int(count)
        for label, count in dataframe[TARGET_COLUMN].value_counts().sort_index().items()
    }
    return {
        "data_path": str(Path(path).expanduser().resolve()),
        "num_samples": int(len(dataframe)),
        "num_features": len(FEATURES),
        "num_groups": int(groups.nunique()),
        "class_counts": class_counts,
        "missing_feature_values": int(dataframe[FEATURES].isna().sum().sum()),
        "evaluation_strategy": "GroupKFold by CLASS_TYPE + ID_PATIENT",
    }


def evaluate_spiral_model(dataframe: pd.DataFrame) -> dict[str, float]:
    X, y = build_training_frame(dataframe)
    groups = build_group_labels(dataframe)
    cv = GroupKFold(n_splits=min(5, int(groups.nunique())))
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }
    results = cross_validate(
        build_spiral_pipeline(),
        X,
        y,
        cv=cv,
        groups=groups,
        scoring=scoring,
    )
    return {
        metric: float(results[f"test_{metric}"].mean())
        for metric in scoring
    }


def train_spiral_model(dataframe: pd.DataFrame) -> Pipeline:
    X, y = build_training_frame(dataframe)
    model = build_spiral_pipeline()
    model.fit(X, y)
    return model


def feature_importances(dataframe: pd.DataFrame) -> dict[str, float]:
    model = train_spiral_model(dataframe)
    classifier = model.named_steps["classifier"]
    pairs = zip(FEATURES, classifier.feature_importances_)
    ranked = sorted(pairs, key=lambda item: item[1], reverse=True)
    return {feature: float(importance) for feature, importance in ranked}


def save_model(model: Pipeline, path: str | Path = ARTIFACT_PATH) -> Path:
    artifact_path = Path(path).expanduser().resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
    return artifact_path


def load_model(path: str | Path = ARTIFACT_PATH) -> Pipeline:
    return joblib.load(Path(path).expanduser().resolve())


def predict_records(
    records: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    model: Pipeline | None = None,
    model_path: str | Path = ARTIFACT_PATH,
) -> list[dict[str, Any]]:
    prediction_model = model or load_model(model_path)
    frame = build_prediction_frame(records)
    predicted_classes = prediction_model.predict(frame)
    probabilities = prediction_model.predict_proba(frame)

    results: list[dict[str, Any]] = []
    classes = [int(label) for label in prediction_model.classes_]
    for row, predicted_class, probability_row in zip(frame.to_dict(orient="records"), predicted_classes, probabilities):
        probability_map = {
            CLASS_LABELS.get(label, str(label)).lower(): float(probability)
            for label, probability in zip(classes, probability_row)
        }
        results.append(
            {
                "prediction_code": int(predicted_class),
                "prediction_label": CLASS_LABELS.get(int(predicted_class), str(predicted_class)),
                "probabilities": probability_map,
                "features": row,
            }
        )
    return results
