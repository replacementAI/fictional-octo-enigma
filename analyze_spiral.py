from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from parksense.spiral.model import CLASS_LABELS, FEATURES, DATA_PATH, feature_importances, load_spiral_dataset

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).resolve().parent
COLORS = {
    1: "#E24B4A",
    2: "#1D9E75",
}


def main() -> None:
    dataframe = load_spiral_dataset(DATA_PATH)
    importances = pd.Series(feature_importances(dataframe)).sort_values(ascending=False)

    print("=" * 55)
    print("DATASET OVERVIEW")
    print("=" * 55)
    print(f"Total rows:     {len(dataframe)}")
    print("\nClass distribution:")
    print(dataframe["CLASS_TYPE"].value_counts().to_string())
    print(
        f"\nAge min: {dataframe['AGE'].min()}  max: {dataframe['AGE'].max()}  "
        f"mean: {dataframe['AGE'].mean():.1f}"
    )

    print("\n" + "=" * 55)
    print("MEAN FEATURE VALUES BY CLASS")
    print("=" * 55)
    print(dataframe.groupby("CLASS_TYPE")[FEATURES].mean().round(3).T.to_string())

    print("\n" + "=" * 55)
    print("FEATURE IMPORTANCE")
    print("=" * 55)
    for feature, importance in importances.items():
        bar = "X" * int(importance * 60)
        print(f"  {feature:<50}  {bar}  {importance:.4f}")

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle("Spiral Features: Healthy vs Parkinson", fontsize=13)
    axes = axes.flatten()
    for index, feature in enumerate(FEATURES):
        axis = axes[index]
        for class_code in sorted(dataframe["CLASS_TYPE"].unique()):
            values = dataframe[dataframe["CLASS_TYPE"] == class_code][feature].dropna()
            axis.hist(
                values,
                bins=14,
                alpha=0.55,
                label=CLASS_LABELS.get(int(class_code), str(class_code)),
                color=COLORS.get(int(class_code), "#6B7280"),
            )
        axis.set_title(feature.replace("_", " "), fontsize=8)
        axis.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=150)
    print(f"\nSaved -> {OUTPUT_DIR / 'feature_distributions.png'}")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#0F6E56" if i < 3 else "#9FE1CB" for i in range(len(importances))]
    importances.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Feature Importance", fontsize=12)
    ax.set_ylabel("Importance")
    ax.set_xticklabels(
        [feature.replace("_", "\n") for feature in importances.index],
        fontsize=7,
        rotation=45,
        ha="right",
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    print(f"Saved -> {OUTPUT_DIR / 'feature_importance.png'}")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        dataframe[FEATURES].corr(),
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation.png", dpi=150)
    print(f"Saved -> {OUTPUT_DIR / 'correlation.png'}")

    print("\nDone! Open the project folder to review the charts.")


if __name__ == "__main__":
    main()
