from __future__ import annotations

import argparse
import json
from pathlib import Path

from parksense.multimodal.decision import assess_multimodal_risk
from parksense.spiral.inference import predict_spiral_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single runtime entry point for the drawingRFmodel spiral model."
    )
    parser.add_argument("--spiral-image-path", default=None)
    parser.add_argument("--voice-score", type=float, default=None)
    parser.add_argument("--tapping-score", type=float, default=None)
    return parser


def run_model(
    spiral_image_path: str | None,
    voice_score: float | None,
    tapping_score: float | None,
) -> dict[str, object]:
    spiral_prediction = None
    spiral_score = None

    if spiral_image_path:
        spiral_prediction = predict_spiral_image(Path(spiral_image_path))
        spiral_score = spiral_prediction["recommended_multimodal_score"]

    if spiral_score is None and voice_score is None and tapping_score is None:
        payload = {
            "decision": {
                "decision_label": "unscorable_input",
                "summary": "No valid modality score is available yet. Repeat the spiral image or provide voice/tapping scores.",
                "disclaimer": "Research screening prototype only. Not a medical diagnosis.",
            },
            "terminal_scores": {
                "spiral_score_used": None,
                "voice_score_used": None,
                "tapping_score_used": None,
                "composite_risk": None,
            },
        }
        if spiral_prediction is not None:
            payload["spiral_prediction"] = spiral_prediction
        return payload

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
            "composite_risk": decision["composite_risk"],
        },
    }
    if spiral_prediction is not None:
        payload["spiral_prediction"] = spiral_prediction
    return payload


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    payload = run_model(
        spiral_image_path=args.spiral_image_path,
        voice_score=args.voice_score,
        tapping_score=args.tapping_score,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
