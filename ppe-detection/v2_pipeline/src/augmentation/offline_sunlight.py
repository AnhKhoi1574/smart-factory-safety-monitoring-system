"""Generate harsh sunlight offline augmentations for training data."""

from __future__ import annotations

from pathlib import Path


def generate_sunlight_augmentation(source_dir: Path, destination_dir: Path, ratio: float = 0.15) -> int:
    """Create bright sunlight variants from training images only."""
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    # TODO: implement exposure and contrast transforms with label preservation.
    return 0
