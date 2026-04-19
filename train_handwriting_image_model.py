from __future__ import annotations

import argparse
import json
from pathlib import Path

from parksense.handwriting.model import (
    HANDWRITING_ARTIFACT_PATH,
    load_clean_manifest,
    save_handwriting_model,
    train_handwriting_image_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a leakage-safe handwriting image model on the clean ParkSense manifest."
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional override for the clean handwriting manifest CSV.",
    )
    parser.add_argument(
        "--modality",
        choices=("spiral", "wave", "all"),
        default="all",
        help="Which handwriting modality to train on.",
    )
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit output path for the trained joblib artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_clean_manifest(args.manifest_path) if args.manifest_path else load_clean_manifest()
    model_path = (
        Path(args.model_path).expanduser().resolve()
        if args.model_path
        else HANDWRITING_ARTIFACT_PATH.with_name(
            f"{HANDWRITING_ARTIFACT_PATH.stem}_{args.modality}{HANDWRITING_ARTIFACT_PATH.suffix}"
        )
    )
    result = train_handwriting_image_model(
        manifest,
        modality=args.modality,
        image_size=(args.image_size, args.image_size),
    )
    artifact_path = save_handwriting_model(result["model"], model_path)

    payload = {
        "modality": result["modality"],
        "image_size": list(result["image_size"]),
        "n_components": result["n_components"],
        "train_rows": result["train_rows"],
        "val_rows": result["val_rows"],
        "test_rows": result["test_rows"],
        "test_metrics": result["test_metrics"],
        "artifact_path": str(artifact_path),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
