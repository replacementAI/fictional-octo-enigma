from __future__ import annotations

import argparse
import json

from parksense.spiral.model import (
    ARTIFACT_PATH,
    DATA_PATH,
    build_prediction_frame,
    evaluate_spiral_model,
    feature_importances,
    load_spiral_dataset,
    save_model,
    summarize_dataset,
    train_spiral_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and save the ParkSense spiral screening model."
    )
    parser.add_argument(
        "--data-path",
        default=str(DATA_PATH),
        help="CSV path for the HandPD spiral dataset.",
    )
    parser.add_argument(
        "--model-path",
        default=str(ARTIFACT_PATH),
        help="Where to save the trained model artifact.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the training summary as JSON instead of formatted text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_spiral_dataset(args.data_path)
    summary = summarize_dataset(dataset)
    metrics = evaluate_spiral_model(dataset)
    importances = feature_importances(dataset)

    model = train_spiral_model(dataset)
    save_model(model, args.model_path)

    sample_predictions = model.predict_proba(build_prediction_frame(dataset.head(3)))
    classes = [int(label) for label in model.classes_]

    payload = {
        "dataset": summary,
        "cross_validation": metrics,
        "feature_importance": importances,
        "artifact_path": args.model_path,
        "sample_probabilities": [
            {
                "row_index": index,
                "probabilities": {
                    str(label): round(float(probability), 4)
                    for label, probability in zip(classes, probabilities)
                },
            }
            for index, probabilities in zip(dataset.head(3).index.tolist(), sample_predictions)
        ],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print("=" * 60)
    print("PARKSENSE SPIRAL TRAINING")
    print("=" * 60)
    print(f"Data path          : {summary['data_path']}")
    print(f"Samples            : {summary['num_samples']}")
    print(f"Group count        : {summary['num_groups']}")
    print(f"Parkinson samples  : {summary['class_counts']['1']}")
    print(f"Healthy samples    : {summary['class_counts']['2']}")
    print(f"Missing values     : {summary['missing_feature_values']}")
    print(f"Validation         : {summary['evaluation_strategy']}")

    print("\nCross-validation")
    for name, value in metrics.items():
        print(f"  {name:<18} {value:.3f}")

    print("\nTop features")
    for feature, importance in importances.items():
        print(f"  {feature:<48} {importance:.4f}")

    print(f"\nSaved model        : {args.model_path}")
    print("Preview probabilities")
    for item in payload["sample_probabilities"]:
        print(f"  Row {item['row_index']}: {item['probabilities']}")


if __name__ == "__main__":
    main()
