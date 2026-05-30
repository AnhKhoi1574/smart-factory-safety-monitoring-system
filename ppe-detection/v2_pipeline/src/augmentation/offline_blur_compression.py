"""Generate blur and compression offline augmentations for training data."""

from __future__ import annotations

from pathlib import Path


def generate_blur_compression_augmentation(
    source_dir: Path,
    destination_dir: Path,
    ratio: float = 0.10,
) -> int:
    """Create blur/compression variants from training images only."""
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    # TODO: implement blur and JPEG compression artifacts while preserving labels.
    return 0
