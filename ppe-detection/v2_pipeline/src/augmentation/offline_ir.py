"""Generate infrared-like offline augmentations for training data."""

from __future__ import annotations

from pathlib import Path


def generate_ir_augmentation(source_dir: Path, destination_dir: Path, ratio: float = 0.25) -> int:
    """Create IR-style augmented samples from training images only."""
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    # TODO: implement image transforms and label copying for selected samples.
    return 0
