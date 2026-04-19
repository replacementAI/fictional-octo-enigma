from __future__ import annotations

import json

from parksense.handwriting.model import (
    SPIRAL_IMAGE_ARTIFACT_PATH,
    load_clean_manifest,
    save_handwriting_bundle,
    train_final_handwriting_model,
)
from parksense.spiral.model import (
    ARTIFACT_PATH,
    evaluate_spiral_model,
    feature_importances,
    load_spiral_dataset,
    save_model,
    summarize_dataset,
    train_spiral_model,
)


def main() -> None:
    spiral_dataset = load_spiral_dataset()
    spiral_model = train_spiral_model(spiral_dataset)
    spiral_tabular_path = save_model(spiral_model, ARTIFACT_PATH)

    manifest = load_clean_manifest()
    spiral_image_bundle = train_final_handwriting_model(manifest, modality="spiral", image_size=(48, 48))
    spiral_image_path = save_handwriting_bundle(spiral_image_bundle, SPIRAL_IMAGE_ARTIFACT_PATH)

    payload = {
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
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
