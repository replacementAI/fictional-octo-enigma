from __future__ import annotations

import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from parksense.handwriting.model import load_clean_manifest, train_handwriting_image_model
from parksense.spiral.model import FEATURES, build_group_labels, load_spiral_dataset


def family_metrics(labels: pd.Series, probabilities: pd.Series) -> dict[str, float]:
    y_true = labels.to_numpy()
    y_score = probabilities.to_numpy()
    y_pred = pd.Series((y_score >= 0.5).astype(int)).map({1: 1, 0: 2}).to_numpy()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score((y_true == 1).astype(int), y_score)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def evaluate_spiral_meander() -> dict[str, float]:
    spiral = load_spiral_dataset()
    meander = pd.read_csv("/Users/user/Desktop/parksense/data/Meander_HandPD.csv")
    meander.columns = meander.columns.str.strip()

    combined = spiral.copy()
    combined["GROUP"] = build_group_labels(combined)
    combined["label_code"] = combined["CLASS_TYPE"].astype(int)

    meander_features = meander[
        [
            "RMS",
            "MAX_BETWEEN_ST_HT",
            "MIN_BETWEEN_ST_HT",
            "STD_DEVIATION_ST_HT",
            "MRT",
            "MAX_HT",
            "MIN_HT",
            "STD_HT",
            "CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ST_HT",
        ]
    ].copy()
    meander_features.columns = [f"{column}_meander" for column in meander_features.columns]
    combined = pd.concat([combined.reset_index(drop=True), meander_features.reset_index(drop=True)], axis=1)

    train = combined[combined["GROUP"].isin(sorted(combined["GROUP"].unique())[:38])].copy()
    holdout = combined[~combined["GROUP"].isin(train["GROUP"])].copy()

    # Use the clean handwriting family list to avoid tuning on the future image test split conceptually.
    X_train = train[FEATURES + meander_features.columns.tolist()]
    y_train = train["label_code"]
    X_holdout = holdout[FEATURES + meander_features.columns.tolist()]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=4000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_holdout)[:, 0]
    holdout_eval = holdout.copy()
    holdout_eval["probability_parkinson"] = probabilities
    grouped = holdout_eval.groupby("GROUP").agg(
        label_code=("label_code", "first"),
        probability_parkinson=("probability_parkinson", "mean"),
    )
    return family_metrics(grouped["label_code"], grouped["probability_parkinson"])


def main() -> None:
    manifest = load_clean_manifest()
    image_results = {
        modality: train_handwriting_image_model(manifest, modality=modality)
        for modality in ("spiral", "wave", "all")
    }

    payload = {
        "handwriting_image_models": {
            modality: {
                "test_metrics": result["test_metrics"],
            }
            for modality, result in image_results.items()
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
