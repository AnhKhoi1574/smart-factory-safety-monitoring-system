"""Dataset merge utilities for canonicalizing validated samples."""

from __future__ import annotations

from pathlib import Path


def merge_valid_samples(source_dirs: list[Path], destination_dir: Path) -> list[Path]:
    """Merge validated source folders into the master original dataset."""
    _ = [Path(path) for path in source_dirs]
    destination_dir = Path(destination_dir)
    # TODO: rename files deterministically, prevent collisions, and copy image/label pairs.
    return []
