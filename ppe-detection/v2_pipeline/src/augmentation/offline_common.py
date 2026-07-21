"""Shared helpers for deterministic offline PPE training augmentations.

All current PPE offline augmentations are photometric or quality transforms:
IR/grayscale, sunlight/glare, and blur/compression. They do not move pixels in a
way that changes object geometry, so YOLO labels are copied unchanged. This is
safe for all four classes, including `cleaning_coverall`.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any, Callable

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
    "num_objects",
    "num_person",
    "num_helmet",
    "num_vest",
    "num_cleaning_coverall",
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
    """Generate one deterministic offline augmentation report.

    Args:
        augmentation_type: Human-readable augmentation label, such as `ir`.
        image_prefix: Prefix added to augmented image and label filenames.
        images_dir: Training split image directory.
        labels_dir: Training split label directory.
        output_images_dir: Generated augmented image directory.
        output_labels_dir: Generated augmented label directory.
        ratio: Fraction of training images to sample. Non-zero ratios select at
            least one image when training images exist.
        seed: Deterministic sampling seed.
        overwrite: If `False`, existing augmented outputs are skipped.
        transform: Function that changes image appearance without moving object
            geometry.

    Returns:
        Row-level report for selected images. Label class counts are copied into
        the report so Notebook 04 can confirm class `3 = cleaning_coverall` is
        preserved in augmented labels.
    """
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
    rows: list[dict[str, Any]] = []
    for image_path in selected_images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        augmented_image_path = output_images_dir / (
            f"{image_prefix}{image_path.stem}{image_path.suffix.lower()}"
        )
        augmented_label_path = output_labels_dir / f"{image_prefix}{image_path.stem}.txt"

        if not label_path.exists():
            class_counts = _empty_class_counts()
            rows.append(
                _report_row(
                    augmentation_type=augmentation_type,
                    image_path=image_path,
                    label_path=label_path,
                    augmented_image_path=augmented_image_path,
                    augmented_label_path=augmented_label_path,
                    status="skipped",
                    notes="matching label file not found",
                    class_counts=class_counts,
                )
            )
            continue

        class_counts = _count_label_classes(label_path)

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
                    class_counts=class_counts,
                )
            )
            continue

        try:
            rng = random.Random(f"{seed}:{augmentation_type}:{image_path.name}")
            with Image.open(image_path) as image:
                augmented_image = transform(image.convert("RGB"), rng)
            _save_image(augmented_image, augmented_image_path)
            # Labels are copied unchanged because these transforms do not alter
            # object position, scale, or orientation. If geometric transforms are
            # added later, this copy step must be replaced with bbox updates.
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
                class_counts=class_counts,
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
    """Select a deterministic subset of images for one augmentation type."""
    if ratio == 0:
        return []
    sample_count = int(round(len(image_paths) * ratio))
    sample_count = max(1, sample_count)
    sample_count = min(sample_count, len(image_paths))
    return sorted(random.Random(seed).sample(image_paths, sample_count))


def _collect_images(images_dir: Path) -> list[Path]:
    """Collect supported image files while ignoring placeholders and metadata."""
    return [
        path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]


def _save_image(image: Image.Image, output_path: Path) -> None:
    """Save an augmented image with sensible JPEG settings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(output_path, quality=92, optimize=True)
    else:
        image.save(output_path)


def _single_warning_report(augmentation_type: str, notes: str) -> pd.DataFrame:
    class_counts = _empty_class_counts()
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
                **class_counts,
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
    class_counts: dict[str, int],
) -> dict[str, Any]:
    """Build a normalized report row for one selected source image."""
    return {
        "augmentation_type": augmentation_type,
        "original_image_path": str(image_path),
        "original_label_path": str(label_path),
        "augmented_image_path": str(augmented_image_path),
        "augmented_label_path": str(augmented_label_path),
        "status": status,
        "notes": notes,
        **class_counts,
    }


def _empty_class_counts() -> dict[str, int]:
    """Return zero counts for the four-class PPE schema."""
    return {
        "num_objects": 0,
        "num_person": 0,
        "num_helmet": 0,
        "num_vest": 0,
        "num_cleaning_coverall": 0,
    }


def _count_label_classes(label_path: Path) -> dict[str, int]:
    """Count PPE class IDs in one YOLO label file.

    The parser is intentionally tolerant here because Notebook 01 already owns
    strict validation. Notebook 04 only needs counts for reporting and should not
    crash if a stale or partially edited label sneaks in.
    """
    counts = _empty_class_counts()
    try:
        raw_text = Path(label_path).read_text(encoding="utf-8").strip()
    except OSError:
        return counts

    if not raw_text:
        return counts

    for line in raw_text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if class_id == 0:
            counts["num_person"] += 1
        elif class_id == 1:
            counts["num_helmet"] += 1
        elif class_id == 2:
            counts["num_vest"] += 1
        elif class_id == 3:
            counts["num_cleaning_coverall"] += 1
        else:
            continue
        counts["num_objects"] += 1
    return counts
