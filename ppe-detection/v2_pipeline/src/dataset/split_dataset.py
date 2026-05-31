"""Dataset splitting helpers for train/validation/test generation."""

from __future__ import annotations

from pathlib import Path


def split_dataset(
    merged_dataset_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.20,
    test_ratio: float = 0.00,
    random_seed: int = 42,
    use_external_test_set: bool = True,
) -> dict[str, int]:
    """Split the merged dataset without leaking external test data.

    Args:
        merged_dataset_dir: Directory containing merged image-label pairs.
        output_dir: Directory where split folders will be created.
        train_ratio: Fraction of ``merged_dataset_dir`` assigned to training.
        val_ratio: Fraction of ``merged_dataset_dir`` assigned to validation.
        test_ratio: Fraction assigned to test when no external test set is used.
        random_seed: Seed used for deterministic shuffling.
        use_external_test_set: If ``True``, test samples are expected to already
            live under ``splits_original/test`` from ``data/test_sources``.

    Returns:
        Counts for each split. This placeholder returns zeros until notebook 03
        implements the actual copy operation.
    """
    merged_dataset_dir = Path(merged_dataset_dir)
    output_dir = Path(output_dir)
    if use_external_test_set:
        test_ratio = 0.00

    # TODO: stratify where appropriate and copy matched image/label pairs into
    # split folders. If use_external_test_set is true, create train/val only.
    return {"train": 0, "val": 0, "test": 0}
