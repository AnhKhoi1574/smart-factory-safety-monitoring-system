"""Small plotting utilities for factory sign detection reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_class_distribution(class_distribution: pd.DataFrame, output_path: Path) -> Path:
    """Save a compact object-count bar chart for Notebook 01."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(class_distribution["class_name"], class_distribution["object_count"], color="#2f6f73")
    ax.set_title("Input Object Count by Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Object count")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
