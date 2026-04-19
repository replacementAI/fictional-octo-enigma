from __future__ import annotations

import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from random import Random

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
AUGMENTED_DATASET_PATH = ROOT_DIR / "augmented_hw_dataset"
MANIFEST_PATH = ROOT_DIR / "artifacts" / "handwriting_image_manifest.csv"
CLEAN_MANIFEST_PATH = ROOT_DIR / "artifacts" / "handwriting_image_clean_manifest.csv"
CLEAN_SPLIT_ROOT = ROOT_DIR / "artifacts" / "augmented_hw_dataset_clean"
CLEAN_SPLIT_NAMES = ("train", "val", "test")

FILENAME_PATTERN = re.compile(
    r"^aug_(?P<augmentation_index>\d+)_(?P<subject_task_key>V\d+(?P<class_letter>[HP])[EO])(?P<sample_index>\d+)\.png$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SplitPlan:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def normalized(self) -> tuple[float, float, float]:
        ratios = (self.train_ratio, self.val_ratio, self.test_ratio)
        total = sum(ratios)
        if total <= 0:
            raise ValueError("Split ratios must sum to a positive number.")
        return tuple(ratio / total for ratio in ratios)


def parse_image_record(path: Path, root: Path) -> dict[str, str | int]:
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Unexpected filename format: {path.name}")

    relative = path.relative_to(root)
    if len(relative.parts) != 4:
        raise ValueError(f"Unexpected dataset structure: {relative}")

    modality, source_split, label, _ = relative.parts
    subject_task_key = match.group("subject_task_key").upper()
    class_letter = match.group("class_letter").upper()
    subject_family_key = f"{subject_task_key[:-2]}{class_letter}"
    sample_index = match.group("sample_index")
    sample_key = f"{subject_task_key}{sample_index}"

    return {
        "path": str(path.resolve()),
        "relative_path": str(relative),
        "filename": path.name,
        "modality": modality,
        "source_split": source_split,
        "label": label,
        "augmentation_index": int(match.group("augmentation_index")),
        "subject_task_key": subject_task_key,
        "subject_family_key": subject_family_key,
        "sample_index": sample_index,
        "sample_key": sample_key,
    }


def build_image_index(root: str | Path = AUGMENTED_DATASET_PATH) -> pd.DataFrame:
    dataset_root = Path(root).expanduser().resolve()
    records = [
        parse_image_record(path, dataset_root)
        for path in sorted(dataset_root.rglob("*.png"))
    ]
    if not records:
        raise FileNotFoundError(f"No PNG files found under {dataset_root}")
    return pd.DataFrame.from_records(records)


def audit_split_leakage(index: pd.DataFrame, split_column: str = "source_split") -> dict[str, object]:
    report: dict[str, object] = {
        "num_images": int(len(index)),
        "modalities": sorted(index["modality"].unique().tolist()),
        "source_split_counts": {
            key: int(value)
            for key, value in index[split_column].value_counts().sort_index().items()
        },
        "sample_key_leaks": {},
        "subject_task_key_leaks": {},
        "subject_family_key_leaks": {},
    }

    for modality, subset in index.groupby("modality"):
        for key in ("sample_key", "subject_task_key", "subject_family_key"):
            spans = subset.groupby(key)[split_column].nunique()
            report[f"{key}_leaks"][modality] = int((spans > 1).sum())
    return report


def _allocation_counts(total_groups: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = [ratio * total_groups for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remainder = total_groups - sum(counts)

    order = sorted(
        range(len(raw)),
        key=lambda index: raw[index] - counts[index],
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1

    for idx in range(len(counts)):
        if counts[idx] == 0 and total_groups >= len(counts):
            donor = max(range(len(counts)), key=lambda donor_idx: counts[donor_idx])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[idx] += 1
    return tuple(counts)


def with_clean_splits(
    index: pd.DataFrame,
    seed: int = 42,
    split_plan: SplitPlan | None = None,
) -> pd.DataFrame:
    plan = split_plan or SplitPlan()
    ratios = plan.normalized()
    output = index.copy()

    group_table = (
        output[["subject_family_key", "label"]]
        .drop_duplicates()
        .sort_values(["label", "subject_family_key"])
        .reset_index(drop=True)
    )

    assignments: dict[str, str] = {}
    rng = Random(seed)

    for label, subset in group_table.groupby("label"):
        groups = subset["subject_family_key"].tolist()
        rng.shuffle(groups)
        train_count, val_count, test_count = _allocation_counts(len(groups), ratios)

        train_groups = groups[:train_count]
        val_groups = groups[train_count:train_count + val_count]
        test_groups = groups[train_count + val_count:train_count + val_count + test_count]

        for group in train_groups:
            assignments[group] = CLEAN_SPLIT_NAMES[0]
        for group in val_groups:
            assignments[group] = CLEAN_SPLIT_NAMES[1]
        for group in test_groups:
            assignments[group] = CLEAN_SPLIT_NAMES[2]

    output["clean_split"] = output["subject_family_key"].map(assignments)
    if output["clean_split"].isna().any():
        raise ValueError("Failed to assign all handwriting groups to a clean split.")
    return output


def save_manifest(index: pd.DataFrame, path: str | Path) -> Path:
    manifest_path = Path(path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(manifest_path, index=False)
    return manifest_path


def materialize_clean_split_tree(index: pd.DataFrame, output_root: str | Path = CLEAN_SPLIT_ROOT) -> Path:
    root = Path(output_root).expanduser().resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    for row in index.itertuples(index=False):
        destination = root / row.modality / row.clean_split / row.label / row.filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = Path(row.path)

        if destination.exists() or destination.is_symlink():
            destination.unlink()
        os.symlink(source, destination)
    return root
