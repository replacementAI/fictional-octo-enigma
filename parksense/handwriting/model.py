from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from parksense.handwriting.images import CLEAN_MANIFEST_PATH

HANDWRITING_ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "handwriting_image_pipeline.joblib"
SPIRAL_IMAGE_ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "spiral_image_pipeline_final.joblib"
LABEL_TO_CODE = {
    "parkinson": 1,
    "healthy": 2,
}
CODE_TO_LABEL = {value: key for key, value in LABEL_TO_CODE.items()}
DEFAULT_QUALITY_THRESHOLDS = {
    "min_width": 128,
    "min_height": 128,
    "min_contrast_std": 0.12,
    "min_ink_mean": 0.08,
    "min_foreground_fraction": 0.07,
    "max_foreground_fraction": 0.85,
}
DEFAULT_SPIRAL_THRESHOLDS = {
    "low_risk_max": 0.40,
    "high_risk_min": 0.70,
}


def load_clean_manifest(path: str | Path = CLEAN_MANIFEST_PATH) -> pd.DataFrame:
    manifest = pd.read_csv(Path(path).expanduser().resolve())
    required = {"path", "modality", "label", "clean_split", "subject_family_key"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return manifest


def _load_grayscale_image(path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        grayscale = ImageOps.autocontrast(image.convert("L"))
        grayscale = ImageOps.pad(grayscale, image_size, color=255)
        array = np.asarray(grayscale, dtype=np.float32) / 255.0
    return 1.0 - array


def extract_image_matrix(paths: Iterable[str | Path], image_size: tuple[int, int] = (48, 48)) -> np.ndarray:
    vectors = [_load_grayscale_image(path, image_size).reshape(-1) for path in paths]
    if not vectors:
        raise ValueError("No images were provided for feature extraction.")
    return np.vstack(vectors)


def build_image_model(n_components: int = 64) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, whiten=False, random_state=42)),
            ("classifier", LogisticRegression(max_iter=4000, class_weight="balanced")),
        ]
    )


def prepare_split_frame(manifest: pd.DataFrame, modality: str) -> pd.DataFrame:
    if modality == "all":
        frame = manifest.copy()
    else:
        frame = manifest[manifest["modality"] == modality].copy()
    if frame.empty:
        raise ValueError(f"No rows available for modality '{modality}'.")

    frame["label_code"] = frame["label"].str.lower().map(LABEL_TO_CODE)
    frame["modality_wave"] = (frame["modality"] == "wave").astype(int)
    frame["modality_spiral"] = (frame["modality"] == "spiral").astype(int)
    return frame


def split_manifest(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = manifest[manifest["clean_split"] == "train"].copy()
    val = manifest[manifest["clean_split"] == "val"].copy()
    test = manifest[manifest["clean_split"] == "test"].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("Expected non-empty train/val/test splits in the clean manifest.")
    return train, val, test


def _augment_with_modality_columns(matrix: np.ndarray, frame: pd.DataFrame, use_modality_flags: bool) -> np.ndarray:
    if not use_modality_flags:
        return matrix
    modality_features = frame[["modality_spiral", "modality_wave"]].to_numpy(dtype=np.float32)
    return np.hstack([matrix, modality_features])


def _family_level_metrics(frame: pd.DataFrame) -> dict[str, float]:
    grouped = (
        frame.groupby("subject_family_key")
        .agg(
            label_code=("label_code", "first"),
            probability_parkinson=("probability_parkinson", "mean"),
        )
        .reset_index(drop=True)
    )
    y_true = grouped["label_code"].to_numpy()
    y_score = grouped["probability_parkinson"].to_numpy()
    y_pred = np.where(y_score >= 0.5, 1, 2)
    return {
        "families": float(len(grouped)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score((y_true == 1).astype(int), y_score)),
        "brier_score": float(brier_score_loss((y_true == 1).astype(int), y_score)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def _fit_calibrated_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_calibration: np.ndarray,
    y_calibration: pd.Series,
) -> CalibratedClassifierCV:
    n_components = min(64, X_train.shape[0] - 1, X_train.shape[1])
    base_model = build_image_model(n_components=max(8, n_components))
    base_model.fit(X_train, y_train)
    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method="sigmoid",
        cv=None,
    )
    calibrated_model.fit(X_calibration, y_calibration)
    return calibrated_model


def train_handwriting_image_model(
    manifest: pd.DataFrame,
    modality: str = "all",
    image_size: tuple[int, int] = (48, 48),
) -> dict[str, object]:
    frame = prepare_split_frame(manifest, modality)
    train_frame, val_frame, test_frame = split_manifest(frame)
    use_modality_flags = modality == "all"

    X_train = extract_image_matrix(train_frame["path"], image_size=image_size)
    X_val = extract_image_matrix(val_frame["path"], image_size=image_size)
    X_test = extract_image_matrix(test_frame["path"], image_size=image_size)

    X_train = _augment_with_modality_columns(X_train, train_frame, use_modality_flags)
    X_val = _augment_with_modality_columns(X_val, val_frame, use_modality_flags)
    X_test = _augment_with_modality_columns(X_test, test_frame, use_modality_flags)

    model = _fit_calibrated_model(
        X_train,
        train_frame["label_code"],
        X_val,
        val_frame["label_code"],
    )

    test_probabilities = model.predict_proba(X_test)[:, 0]

    test_eval = test_frame.copy()
    test_eval["probability_parkinson"] = test_probabilities

    return {
        "modality": modality,
        "image_size": image_size,
        "n_components": int(model.calibrated_classifiers_[0].estimator.named_steps["pca"].n_components),
        "model": model,
        "test_metrics": _family_level_metrics(test_eval),
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
        "test_rows": int(len(test_frame)),
    }


def save_handwriting_model(model: Pipeline, path: str | Path = HANDWRITING_ARTIFACT_PATH) -> Path:
    artifact_path = Path(path).expanduser().resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
    return artifact_path


def save_handwriting_bundle(bundle: dict[str, object], path: str | Path = HANDWRITING_ARTIFACT_PATH) -> Path:
    artifact_path = Path(path).expanduser().resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    return artifact_path


def load_handwriting_bundle(path: str | Path = SPIRAL_IMAGE_ARTIFACT_PATH) -> dict[str, object]:
    loaded = joblib.load(Path(path).expanduser().resolve())
    if isinstance(loaded, Pipeline):
        return {
            "model": loaded,
            "modality": "spiral",
            "image_size": (48, 48),
            "uses_modality_flags": False,
        }
    return loaded


def train_final_handwriting_model(
    manifest: pd.DataFrame,
    modality: str = "spiral",
    image_size: tuple[int, int] = (48, 48),
) -> dict[str, object]:
    frame = prepare_split_frame(manifest, modality)
    train_frame, val_frame, test_frame = split_manifest(frame)
    use_modality_flags = modality == "all"

    X_train = extract_image_matrix(train_frame["path"], image_size=image_size)
    X_val = extract_image_matrix(val_frame["path"], image_size=image_size)
    X_test = extract_image_matrix(test_frame["path"], image_size=image_size)
    X_train = _augment_with_modality_columns(X_train, train_frame, use_modality_flags)
    X_val = _augment_with_modality_columns(X_val, val_frame, use_modality_flags)
    X_test = _augment_with_modality_columns(X_test, test_frame, use_modality_flags)

    model = _fit_calibrated_model(
        X_train,
        train_frame["label_code"],
        X_val,
        val_frame["label_code"],
    )
    test_probabilities = model.predict_proba(X_test)[:, 0]
    test_eval = test_frame.copy()
    test_eval["probability_parkinson"] = test_probabilities

    return {
        "model": model,
        "modality": modality,
        "image_size": image_size,
        "n_components": int(model.calibrated_classifiers_[0].estimator.named_steps["pca"].n_components),
        "uses_modality_flags": use_modality_flags,
        "trained_rows": int(len(train_frame) + len(val_frame)),
        "subject_families": int(frame["subject_family_key"].nunique()),
        "quality_thresholds": DEFAULT_QUALITY_THRESHOLDS,
        "decision_thresholds": DEFAULT_SPIRAL_THRESHOLDS,
        "test_metrics": _family_level_metrics(test_eval),
    }


def assess_image_quality(
    image_path: str | Path,
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    selected = dict(DEFAULT_QUALITY_THRESHOLDS)
    if thresholds:
        selected.update(thresholds)

    path = Path(image_path).expanduser().resolve()
    with Image.open(path) as image:
        grayscale = ImageOps.autocontrast(image.convert("L"))
        array = np.asarray(grayscale, dtype=np.float32) / 255.0
        ink = 1.0 - array
        width, height = image.size

    metrics = {
        "width": int(width),
        "height": int(height),
        "contrast_std": float(array.std()),
        "ink_mean": float(ink.mean()),
        "foreground_fraction": float((ink > 0.25).mean()),
    }

    reasons: list[str] = []
    if metrics["width"] < selected["min_width"] or metrics["height"] < selected["min_height"]:
        reasons.append("image_resolution_too_low")
    if metrics["contrast_std"] < selected["min_contrast_std"]:
        reasons.append("low_contrast")
    if metrics["ink_mean"] < selected["min_ink_mean"]:
        reasons.append("too_little_visible_stroke")
    if metrics["foreground_fraction"] < selected["min_foreground_fraction"]:
        reasons.append("drawing_too_sparse")
    if metrics["foreground_fraction"] > selected["max_foreground_fraction"]:
        reasons.append("drawing_or_background_too_dense")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "metrics": metrics,
        "thresholds": selected,
    }


def predict_handwriting_images(
    image_paths: Iterable[str | Path],
    model_path: str | Path = SPIRAL_IMAGE_ARTIFACT_PATH,
    modality: str | None = None,
) -> list[dict[str, object]]:
    bundle = load_handwriting_bundle(model_path)
    model = bundle["model"]
    image_size = tuple(bundle.get("image_size", (48, 48)))
    resolved_modality = modality or str(bundle.get("modality", "spiral"))
    uses_modality_flags = bool(bundle.get("uses_modality_flags", False))
    quality_thresholds = bundle.get("quality_thresholds", DEFAULT_QUALITY_THRESHOLDS)
    decision_thresholds = bundle.get("decision_thresholds", DEFAULT_SPIRAL_THRESHOLDS)

    paths = [Path(path).expanduser().resolve() for path in image_paths]
    matrix = extract_image_matrix(paths, image_size=image_size)
    if uses_modality_flags:
        if resolved_modality not in {"spiral", "wave"}:
            raise ValueError("Modality must be provided as 'spiral' or 'wave' for this model.")
        frame = pd.DataFrame(
            {
                "modality_spiral": [1 if resolved_modality == "spiral" else 0] * len(paths),
                "modality_wave": [1 if resolved_modality == "wave" else 0] * len(paths),
            }
        )
        matrix = _augment_with_modality_columns(matrix, frame, use_modality_flags=True)

    probabilities = model.predict_proba(matrix)
    classes = [int(value) for value in model.classes_]

    results: list[dict[str, object]] = []
    for path, probability_row in zip(paths, probabilities):
        quality = assess_image_quality(path, thresholds=quality_thresholds)
        score_map = {
            CODE_TO_LABEL.get(label, str(label)): float(probability)
            for label, probability in zip(classes, probability_row)
        }
        parkinson_risk = score_map.get("parkinson", 0.0)
        if not quality["passed"]:
            signal_label = "insufficient_quality"
            recommended_score = None
            prediction_code = None
            prediction_label = "unscorable"
        elif parkinson_risk >= decision_thresholds["high_risk_min"]:
            signal_label = "high_spiral_risk"
            recommended_score = parkinson_risk
            prediction_code = 1
            prediction_label = CODE_TO_LABEL[prediction_code]
        elif parkinson_risk <= decision_thresholds["low_risk_max"]:
            signal_label = "low_spiral_risk"
            recommended_score = parkinson_risk
            prediction_code = 2
            prediction_label = CODE_TO_LABEL[prediction_code]
        else:
            signal_label = "uncertain_repeat_test"
            recommended_score = parkinson_risk
            prediction_code = 1 if parkinson_risk >= 0.5 else 2
            prediction_label = CODE_TO_LABEL[prediction_code]

        results.append(
            {
                "image_path": str(path),
                "modality": resolved_modality,
                "prediction_code": prediction_code,
                "prediction_label": prediction_label,
                "parkinson_risk": parkinson_risk,
                "healthy_probability": score_map.get("healthy", 0.0),
                "recommended_multimodal_score": recommended_score,
                "multimodal_ready": quality["passed"],
                "signal_decision_label": signal_label,
                "quality_check": quality,
            }
        )
    return results
