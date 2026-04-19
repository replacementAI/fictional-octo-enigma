from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "spiral_image_pipeline_final.joblib"

LABEL_TO_CODE = {"parkinson": 1, "healthy": 2}
CODE_TO_LABEL = {value: key for key, value in LABEL_TO_CODE.items()}
DEFAULT_QUALITY_THRESHOLDS = {
    "min_width": 128,
    "min_height": 128,
    "min_contrast_std": 0.12,
    "min_ink_mean": 0.08,
    "min_foreground_fraction": 0.07,
    "max_foreground_fraction": 0.85,
}
DEFAULT_DECISION_THRESHOLDS = {
    "low_risk_max": 0.40,
    "high_risk_min": 0.70,
}
DEFAULT_MODALITY_WEIGHTS = {
    "spiral": 0.25,
    "voice": 0.40,
    "tapping": 0.35,
}


@dataclass(frozen=True)
class SplitData:
    train_paths: list[Path]
    train_labels: list[int]
    val_paths: list[Path]
    val_labels: list[int]
    test_paths: list[Path]
    test_labels: list[int]


def _load_grayscale_image(path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        grayscale = ImageOps.autocontrast(image.convert("L"))
        grayscale = ImageOps.pad(grayscale, image_size, color=255)
        array = np.asarray(grayscale, dtype=np.float32) / 255.0
    return 1.0 - array


def extract_image_matrix(paths: Iterable[str | Path], image_size: tuple[int, int]) -> np.ndarray:
    vectors = [_load_grayscale_image(path, image_size).reshape(-1) for path in paths]
    if not vectors:
        raise ValueError("No images were found for feature extraction.")
    return np.vstack(vectors)


def build_image_model(n_components: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, whiten=False, random_state=42)),
            ("classifier", LogisticRegression(max_iter=4000, class_weight="balanced")),
        ]
    )


def _gather_labelled_images(base_dir: Path) -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    labels: list[int] = []
    for label_name, label_code in LABEL_TO_CODE.items():
        label_dir = base_dir / label_name
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.rglob("*.png")):
            paths.append(path)
            labels.append(label_code)
    return paths, labels


def _split_examples(paths: list[Path], labels: list[int], seed: int) -> SplitData:
    grouped: dict[int, list[Path]] = {}
    for path, label in zip(paths, labels):
        grouped.setdefault(label, []).append(path)

    rng = random.Random(seed)
    train_paths: list[Path] = []
    train_labels: list[int] = []
    val_paths: list[Path] = []
    val_labels: list[int] = []
    test_paths: list[Path] = []
    test_labels: list[int] = []

    for label, label_paths in grouped.items():
        label_paths = list(label_paths)
        rng.shuffle(label_paths)
        total = len(label_paths)
        test_count = max(1, round(total * 0.15)) if total >= 6 else 1
        val_count = max(1, round(total * 0.15)) if total >= 6 else 1
        if test_count + val_count >= total:
            test_count = 1
            val_count = 1
        train_count = total - test_count - val_count

        train_chunk = label_paths[:train_count]
        val_chunk = label_paths[train_count:train_count + val_count]
        test_chunk = label_paths[train_count + val_count:]

        train_paths.extend(train_chunk)
        train_labels.extend([label] * len(train_chunk))
        val_paths.extend(val_chunk)
        val_labels.extend([label] * len(val_chunk))
        test_paths.extend(test_chunk)
        test_labels.extend([label] * len(test_chunk))

    return SplitData(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)


def build_training_splits(data_dir: str | Path, seed: int = 42) -> SplitData:
    root = Path(data_dir).expanduser().resolve()
    if (root / "train").exists():
        train_paths, train_labels = _gather_labelled_images(root / "train")
        val_paths, val_labels = _gather_labelled_images(root / "val")
        if not val_paths and (root / "validation").exists():
            val_paths, val_labels = _gather_labelled_images(root / "validation")
        test_paths, test_labels = _gather_labelled_images(root / "test")
        if not test_paths and (root / "testing").exists():
            test_paths, test_labels = _gather_labelled_images(root / "testing")
        if not train_paths or not val_paths:
            raise ValueError("Expected train/val splits with healthy/ and parkinson/ subfolders.")
        return SplitData(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)

    paths, labels = _gather_labelled_images(root)
    if not paths:
        raise ValueError("No training images found. Expected healthy/ and parkinson/ folders.")
    return _split_examples(paths, labels, seed=seed)


def train_model(data_dir: str | Path, image_size: int = 48, seed: int = 42) -> dict[str, object]:
    splits = build_training_splits(data_dir, seed=seed)
    image_shape = (image_size, image_size)

    X_train = extract_image_matrix(splits.train_paths, image_shape)
    X_val = extract_image_matrix(splits.val_paths, image_shape)
    n_components = min(64, X_train.shape[0] - 1, X_train.shape[1])
    base_model = build_image_model(max(8, n_components))
    base_model.fit(X_train, np.asarray(splits.train_labels))

    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method="sigmoid",
        cv=None,
    )
    calibrated.fit(X_val, np.asarray(splits.val_labels))

    bundle = {
        "model": calibrated,
        "image_size": image_shape,
        "quality_thresholds": DEFAULT_QUALITY_THRESHOLDS,
        "decision_thresholds": DEFAULT_DECISION_THRESHOLDS,
        "trained_rows": len(splits.train_paths) + len(splits.val_paths),
        "test_rows": len(splits.test_paths),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    return {
        "model_path": str(MODEL_PATH),
        "trained_rows": bundle["trained_rows"],
        "validation_rows": len(splits.val_paths),
        "test_rows": len(splits.test_paths),
        "image_size": list(image_shape),
    }


def load_model_bundle(model_path: str | Path = MODEL_PATH) -> dict[str, object]:
    loaded = joblib.load(Path(model_path).expanduser().resolve())
    if isinstance(loaded, Pipeline):
        return {
            "model": loaded,
            "image_size": (48, 48),
            "quality_thresholds": DEFAULT_QUALITY_THRESHOLDS,
            "decision_thresholds": DEFAULT_DECISION_THRESHOLDS,
        }
    return loaded


def assess_image_quality(image_path: str | Path, thresholds: dict[str, float] | None = None) -> dict[str, object]:
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
    }


def predict_spiral_image(image_path: str | Path, model_path: str | Path = MODEL_PATH) -> dict[str, object]:
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    image_size = tuple(bundle.get("image_size", (48, 48)))
    quality_thresholds = dict(bundle.get("quality_thresholds", DEFAULT_QUALITY_THRESHOLDS))
    decision_thresholds = dict(bundle.get("decision_thresholds", DEFAULT_DECISION_THRESHOLDS))

    quality = assess_image_quality(image_path, thresholds=quality_thresholds)
    vector = extract_image_matrix([image_path], image_size)
    probabilities = model.predict_proba(vector)[0]
    classes = [int(value) for value in model.classes_]
    probability_map = {
        CODE_TO_LABEL.get(label, str(label)): float(probability)
        for label, probability in zip(classes, probabilities)
    }

    parkinson_risk = probability_map.get("parkinson", 0.0)
    if not quality["passed"]:
        signal_label = "insufficient_quality"
        recommended_score = None
        prediction_label = "unscorable"
    elif parkinson_risk >= decision_thresholds["high_risk_min"]:
        signal_label = "high_spiral_risk"
        recommended_score = parkinson_risk
        prediction_label = "parkinson"
    elif parkinson_risk <= decision_thresholds["low_risk_max"]:
        signal_label = "low_spiral_risk"
        recommended_score = parkinson_risk
        prediction_label = "healthy"
    else:
        signal_label = "uncertain_repeat_test"
        recommended_score = parkinson_risk
        prediction_label = "parkinson" if parkinson_risk >= 0.5 else "healthy"

    return {
        "image_path": str(Path(image_path).expanduser().resolve()),
        "prediction_label": prediction_label,
        "parkinson_risk": parkinson_risk,
        "healthy_probability": probability_map.get("healthy", 0.0),
        "recommended_multimodal_score": recommended_score,
        "multimodal_ready": quality["passed"],
        "signal_decision_label": signal_label,
        "quality_check": quality,
    }


def assess_multimodal_risk(
    spiral_score: float | None = None,
    voice_score: float | None = None,
    tapping_score: float | None = None,
) -> dict[str, object]:
    provided = {
        "spiral": spiral_score,
        "voice": voice_score,
        "tapping": tapping_score,
    }
    active = {
        name: max(0.0, min(1.0, float(value)))
        for name, value in provided.items()
        if value is not None
    }
    if not active:
        return {
            "decision_label": "unscorable_input",
            "summary": "No valid modality score is available yet. Repeat the spiral image or provide voice/tapping scores.",
            "disclaimer": "Research screening prototype only. Not a medical diagnosis.",
        }

    weight_total = sum(DEFAULT_MODALITY_WEIGHTS[name] for name in active)
    composite = sum(active[name] * DEFAULT_MODALITY_WEIGHTS[name] for name in active) / weight_total
    positive_votes = sum(score >= 0.60 for score in active.values())
    moderate_votes = sum(score >= 0.45 for score in active.values())

    if composite >= 0.65 and positive_votes >= 2:
        decision_label = "seek_neurology_evaluation"
        summary = "Motor signals across modalities support follow-up with a neurologist."
    elif composite >= 0.50 and moderate_votes >= 2:
        decision_label = "monitor_and_repeat_test"
        summary = "Signals are mixed but elevated enough to justify repeat screening and clinical caution."
    else:
        decision_label = "low_detected_risk"
        summary = "Current combined motor signal is not strongly suggestive of Parkinsonian impairment."

    return {
        "decision_label": decision_label,
        "composite_risk": round(composite, 4),
        "modalities_used": sorted(active),
        "modality_scores": active,
        "summary": summary,
        "disclaimer": "Research screening prototype only. Not a medical diagnosis.",
    }


def screen(spiral_image_path: str | None, voice_score: float | None, tapping_score: float | None) -> dict[str, object]:
    spiral_prediction = None
    spiral_score = None
    if spiral_image_path:
        spiral_prediction = predict_spiral_image(spiral_image_path)
        spiral_score = spiral_prediction["recommended_multimodal_score"]

    decision = assess_multimodal_risk(
        spiral_score=spiral_score,
        voice_score=voice_score,
        tapping_score=tapping_score,
    )

    payload = {
        "decision": decision,
        "terminal_scores": {
            "spiral_score_used": spiral_score,
            "voice_score_used": voice_score,
            "tapping_score_used": tapping_score,
            "composite_risk": decision.get("composite_risk"),
        },
    }
    if spiral_prediction is not None:
        payload["spiral_prediction"] = spiral_prediction
    return payload


class AppHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {"status": "ok", "model_ready": MODEL_PATH.exists(), "model_path": str(MODEL_PATH)},
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Route not found"})

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(content_length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Request body must be valid JSON"})
            return

        if self.path == "/predict":
            try:
                response = screen(
                    spiral_image_path=payload.get("spiral_image_path"),
                    voice_score=payload.get("voice_score"),
                    tapping_score=payload.get("tapping_score"),
                )
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, response)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Route not found"})

    def log_message(self, format: str, *args: object) -> None:
        return


def serve(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Serving drawingRFmodel on http://{host}:{port}")
    print("Routes: GET /health, POST /predict")
    server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="drawingRFmodel single-file app")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the spiral model from image folders.")
    train_parser.add_argument("--data-dir", required=True, help="Folder with train/val/test or healthy/parkinson subfolders.")
    train_parser.add_argument("--image-size", type=int, default=48)
    train_parser.add_argument("--seed", type=int, default=42)

    screen_parser = subparsers.add_parser("screen", help="Score a spiral image and combine with optional modality scores.")
    screen_parser.add_argument("--spiral-image-path", default=None)
    screen_parser.add_argument("--voice-score", type=float, default=None)
    screen_parser.add_argument("--tapping-score", type=float, default=None)

    serve_parser = subparsers.add_parser("serve", help="Serve a tiny HTTP API for UI integration.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        payload = train_model(args.data_dir, image_size=args.image_size, seed=args.seed)
    elif args.command == "screen":
        payload = screen(args.spiral_image_path, args.voice_score, args.tapping_score)
    else:
        serve(args.host, args.port)
        return

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
