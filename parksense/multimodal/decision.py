from __future__ import annotations

from typing import Mapping

DEFAULT_MODALITY_WEIGHTS = {
    "spiral": 0.25,
    "voice": 0.40,
    "tapping": 0.35,
}


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def assess_multimodal_risk(
    *,
    spiral_score: float | None = None,
    voice_score: float | None = None,
    tapping_score: float | None = None,
    weights: Mapping[str, float] | None = None,
) -> dict[str, object]:
    provided_scores = {
        "spiral": spiral_score,
        "voice": voice_score,
        "tapping": tapping_score,
    }
    active = {
        name: _clamp_score(value)
        for name, value in provided_scores.items()
        if value is not None
    }
    if not active:
        raise ValueError("At least one modality score is required.")

    selected_weights = dict(weights or DEFAULT_MODALITY_WEIGHTS)
    weight_total = sum(selected_weights.get(name, 0.0) for name in active)
    if weight_total <= 0:
        raise ValueError("Provided modality weights must sum to a positive value.")

    composite = sum(active[name] * selected_weights.get(name, 0.0) for name in active) / weight_total
    positive_votes = sum(score >= 0.60 for score in active.values())
    moderate_votes = sum(score >= 0.45 for score in active.values())
    modality_count = len(active)

    if modality_count == 1:
        confidence = "low"
    elif modality_count == 2:
        confidence = "medium"
    else:
        confidence = "high"

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
        "confidence": confidence,
        "modalities_expected": ["spiral", "voice", "tapping"],
        "modalities_used": sorted(active),
        "modality_scores": active,
        "summary": summary,
        "disclaimer": "Research screening prototype only. Not a medical diagnosis.",
    }
