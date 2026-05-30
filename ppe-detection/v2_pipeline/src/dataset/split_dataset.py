"""Dataset splitting helpers for train/val/test generation."""

from __future__ import annotations

from pathlib import Path


def split_dataset(
    master_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    random_seed: int = 42,
) -> dict[str, int]:
    """Split the merged original dataset into train, val, and test subsets."""
    master_dir = Path(master_dir)
    output_dir = Path(output_dir)
    # TODO: stratify where appropriate and copy matched image/label pairs into split folders.
    return {"train": 0, "val": 0, "test": 0}
