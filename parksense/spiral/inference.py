from __future__ import annotations

from pathlib import Path
from typing import Any

from parksense.handwriting.model import SPIRAL_IMAGE_ARTIFACT_PATH, predict_handwriting_images


def predict_spiral_image(
    image_path: str | Path,
    model_path: str | Path = SPIRAL_IMAGE_ARTIFACT_PATH,
) -> dict[str, Any]:
    prediction = predict_handwriting_images([image_path], model_path=model_path, modality="spiral")[0]
    parkinson_risk = float(prediction["parkinson_risk"])

    if prediction["signal_decision_label"] == "insufficient_quality":
        severity = "unusable"
    elif prediction["signal_decision_label"] == "high_spiral_risk":
        severity = "high"
    elif prediction["signal_decision_label"] == "uncertain_repeat_test":
        severity = "moderate"
    else:
        severity = "low"

    prediction["severity_band"] = severity
    if prediction["signal_decision_label"] == "insufficient_quality":
        recommendation = "Uploaded spiral image quality is not sufficient. Ask the user to redraw and retake the image."
    elif prediction["signal_decision_label"] == "high_spiral_risk":
        recommendation = "Spiral drawing shows a strong Parkinsonian motor irregularity signal."
    elif prediction["signal_decision_label"] == "uncertain_repeat_test":
        recommendation = "Spiral signal is borderline. Repeat the spiral task and rely on voice and tapping for the combined decision."
    else:
        recommendation = "Spiral drawing alone does not show a strong Parkinsonian signal."

    prediction["screening_recommendation"] = recommendation
    prediction["integration_summary"] = {
        "recommended_multimodal_score": prediction["recommended_multimodal_score"],
        "multimodal_ready": prediction["multimodal_ready"],
        "signal_decision_label": prediction["signal_decision_label"],
    }
    return prediction
