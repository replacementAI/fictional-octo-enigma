from __future__ import annotations

import argparse
import json

from parksense.handwriting.images import (
    AUGMENTED_DATASET_PATH,
    CLEAN_MANIFEST_PATH,
    CLEAN_SPLIT_ROOT,
    MANIFEST_PATH,
    audit_split_leakage,
    build_image_index,
    materialize_clean_split_tree,
    save_manifest,
    with_clean_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and rebuild the augmented handwriting image dataset without split leakage."
    )
    parser.add_argument("--data-root", default=str(AUGMENTED_DATASET_PATH))
    parser.add_argument("--manifest-path", default=str(MANIFEST_PATH))
    parser.add_argument("--clean-manifest-path", default=str(CLEAN_MANIFEST_PATH))
    parser.add_argument("--output-root", default=str(CLEAN_SPLIT_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-materialize",
        action="store_true",
        help="Only write CSV manifests; do not create the clean symlink tree.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = build_image_index(args.data_root)
    raw_report = audit_split_leakage(index)

    clean_index = with_clean_splits(index, seed=args.seed)
    clean_report = audit_split_leakage(clean_index, split_column="clean_split")

    manifest_path = save_manifest(index, args.manifest_path)
    clean_manifest_path = save_manifest(clean_index, args.clean_manifest_path)

    output_root = None
    if not args.skip_materialize:
        output_root = materialize_clean_split_tree(clean_index, args.output_root)

    payload = {
        "raw_leakage": raw_report,
        "clean_leakage": clean_report,
        "manifest_path": str(manifest_path),
        "clean_manifest_path": str(clean_manifest_path),
        "clean_split_root": str(output_root) if output_root else None,
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
