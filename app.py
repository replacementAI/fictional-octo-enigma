from __future__ import annotations

import argparse
import json
from pathlib import Path

from parksense.multimodal.decision import assess_multimodal_risk
from parksense.spiral.inference import predict_spiral_image
from parksense.spiral.model import (
    ARTIFACT_PATH,
    evaluate_spiral_model,
    feature_importances,
    load_spiral_dataset,
    save_model,
    summarize_dataset,
    train_spiral_model,
)
from parksense.handwriting.model import (
    SPIRAL_IMAGE_ARTIFACT_PATH,
    load_clean_manifest,
    save_handwriting_bundle,
    train_final_handwriting_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single entry point for the ParkSense spiral model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    finalize = subparsers.add_parser(
        "finalize",
        help="Train and save the spiral tabular and spiral image artifacts.",
    )
    finalize.add_argument("--image-size", type=int, default=48)

    screen = subparsers.add_parser(
        "screen",
        help="Score a user spiral image and optionally combine it with voice/tapping scores.",
    )
    screen.add_argument("--spiral-image-path", default=None)
    screen.add_argument("--voice-score", type=float, default=None)
    screen.add_argument("--tapping-score", type=float, default=None)

    return parser


def run_finalize(image_size: int) -> dict[str, object]:
    spiral_dataset = load_spiral_dataset()
    spiral_model = train_spiral_model(spiral_dataset)
    spiral_tabular_path = save_model(spiral_model, ARTIFACT_PATH)

    manifest = load_clean_manifest()
    spiral_image_bundle = train_final_handwriting_model(
        manifest,
        modality="spiral",
        image_size=(image_size, image_size),
    )
    spiral_image_path = save_handwriting_bundle(
        spiral_image_bundle,
        SPIRAL_IMAGE_ARTIFACT_PATH,
    )

    return {
        "spiral_tabular": {
            "artifact_path": str(spiral_tabular_path),
            "dataset": summarize_dataset(spiral_dataset),
            "cross_validation": evaluate_spiral_model(spiral_dataset),
            "feature_importance": feature_importances(spiral_dataset),
        },
        "spiral_image": {
            "artifact_path": str(spiral_image_path),
            "trained_rows": spiral_image_bundle["trained_rows"],
            "subject_families": spiral_image_bundle["subject_families"],
            "image_size": list(spiral_image_bundle["image_size"]),
            "decision_thresholds": spiral_image_bundle["decision_thresholds"],
            "quality_thresholds": spiral_image_bundle["quality_thresholds"],
            "test_metrics": spiral_image_bundle["test_metrics"],
        },
        "integration_contract": {
            "spiral_image_output_score": "recommended_multimodal_score",
            "spiral_image_gate": "multimodal_ready",
            "combined_modalities": ["spiral", "voice", "tapping"],
        },
    }


def run_screen(
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

    if args.command == "finalize":
        payload = run_finalize(args.image_size)
    else:
        payload = run_screen(
            spiral_image_path=args.spiral_image_path,
            voice_score=args.voice_score,
            tapping_score=args.tapping_score,
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
