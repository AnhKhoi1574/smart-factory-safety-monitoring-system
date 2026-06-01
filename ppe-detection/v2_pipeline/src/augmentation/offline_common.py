"""Shared helpers for deterministic offline training augmentations."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
REPORT_COLUMNS = [
    "augmentation_type",
    "original_image_path",
    "original_label_path",
    "augmented_image_path",
    "augmented_label_path",
    "status",
    "notes",
]

ImageTransform = Callable[[Image.Image, random.Random], Image.Image]


def generate_offline_augmentation(
    augmentation_type: str,
    image_prefix: str,
    images_dir: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    ratio: float,
    seed: int,
    overwrite: bool,
    transform: ImageTransform,
) -> pd.DataFrame:
    """Generate one deterministic offline augmentation report."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    if ratio < 0:
        raise ValueError("Augmentation ratio must be non-negative")
    if not images_dir.exists():
        return _single_warning_report(
            augmentation_type,
            notes=f"Training images directory not found: {images_dir}",
        )
    if not labels_dir.exists():
        return _single_warning_report(
            augmentation_type,
            notes=f"Training labels directory not found: {labels_dir}",
        )

    image_paths = _collect_images(images_dir)
    if not image_paths:
        return _single_warning_report(
            augmentation_type,
            notes=f"No training images found in {images_dir}",
        )

    selected_images = _select_images(image_paths, ratio, seed)
    rows: list[dict[str, str]] = []
    for image_path in selected_images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        augmented_image_path = output_images_dir / (
            f"{image_prefix}{image_path.stem}{image_path.suffix.lower()}"
        )
        augmented_label_path = output_labels_dir / f"{image_prefix}{image_path.stem}.txt"

        if not label_path.exists():
            rows.append(
                _report_row(
                    augmentation_type=augmentation_type,
                    image_path=image_path,
                    label_path=label_path,
                    augmented_image_path=augmented_image_path,
                    augmented_label_path=augmented_label_path,
                    status="skipped",
                    notes="matching label file not found",
                )
            )
            continue

        if (
            (augmented_image_path.exists() or augmented_label_path.exists())
            and not overwrite
        ):
            rows.append(
                _report_row(
                    augmentation_type=augmentation_type,
                    image_path=image_path,
                    label_path=label_path,
                    augmented_image_path=augmented_image_path,
                    augmented_label_path=augmented_label_path,
                    status="skipped",
                    notes="augmented output already exists",
                )
            )
            continue

        try:
            rng = random.Random(f"{seed}:{augmentation_type}:{image_path.name}")
            with Image.open(image_path) as image:
                augmented_image = transform(image.convert("RGB"), rng)
            _save_image(augmented_image, augmented_image_path)
            shutil.copy2(label_path, augmented_label_path)
            status = "generated"
            notes = ""
        except Exception as exc:
            status = "failed"
            notes = str(exc)

        rows.append(
            _report_row(
                augmentation_type=augmentation_type,
                image_path=image_path,
                label_path=label_path,
                augmented_image_path=augmented_image_path,
                augmented_label_path=augmented_label_path,
                status=status,
                notes=notes,
            )
        )

    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def image_to_uint8_array(image: Image.Image) -> np.ndarray:
    """Convert a Pillow image to a uint8 RGB array."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def uint8_array_to_image(array: np.ndarray) -> Image.Image:
    """Convert an array back to a Pillow RGB image."""
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def _select_images(image_paths: list[Path], ratio: float, seed: int) -> list[Path]:
    if ratio == 0:
        return []
    sample_count = int(round(len(image_paths) * ratio))
    sample_count = max(1, sample_count)
    sample_count = min(sample_count, len(image_paths))
    return sorted(random.Random(seed).sample(image_paths, sample_count))


def _collect_images(images_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]


def _save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(output_path, quality=92, optimize=True)
    else:
        image.save(output_path)


def _single_warning_report(augmentation_type: str, notes: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "augmentation_type": augmentation_type,
                "original_image_path": "",
                "original_label_path": "",
                "augmented_image_path": "",
                "augmented_label_path": "",
                "status": "warning",
                "notes": notes,
            }
        ],
        columns=REPORT_COLUMNS,
    )


def _report_row(
    augmentation_type: str,
    image_path: Path,
    label_path: Path,
    augmented_image_path: Path,
    augmented_label_path: Path,
    status: str,
    notes: str,
) -> dict[str, str]:
    return {
        "augmentation_type": augmentation_type,
        "original_image_path": str(image_path),
        "original_label_path": str(label_path),
        "augmented_image_path": str(augmented_image_path),
        "augmented_label_path": str(augmented_label_path),
        "status": status,
        "notes": notes,
    }
