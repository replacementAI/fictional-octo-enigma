from __future__ import annotations

import argparse
import json

from parksense.multimodal.decision import assess_multimodal_risk
from parksense.spiral.inference import predict_spiral_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ParkSense screening flow with a user spiral image and optional voice/tapping scores."
    )
    parser.add_argument("--spiral-image-path", default=None, help="Path to a user-provided spiral image.")
    parser.add_argument("--voice-score", type=float, default=None, help="Voice model Parkinson risk score (0-1).")
    parser.add_argument("--tapping-score", type=float, default=None, help="Tapping model Parkinson risk score (0-1).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spiral_prediction = None
    spiral_score = None

    if args.spiral_image_path:
        spiral_prediction = predict_spiral_image(args.spiral_image_path)
        spiral_score = spiral_prediction["recommended_multimodal_score"]

    if spiral_score is None and args.voice_score is None and args.tapping_score is None:
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
        print(json.dumps(payload, indent=2))
        return

    decision = assess_multimodal_risk(
        spiral_score=spiral_score,
        voice_score=args.voice_score,
        tapping_score=args.tapping_score,
    )

    payload = {
        "decision": decision,
        "terminal_scores": {
            "spiral_score_used": spiral_score,
            "voice_score_used": args.voice_score,
            "tapping_score_used": args.tapping_score,
            "composite_risk": decision["composite_risk"],
        },
    }
    if spiral_prediction is not None:
        payload["spiral_prediction"] = spiral_prediction
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
