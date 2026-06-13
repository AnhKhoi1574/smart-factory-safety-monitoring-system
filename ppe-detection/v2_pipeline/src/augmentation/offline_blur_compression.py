"""Generate blur and compression offline augmentations for training data."""

from __future__ import annotations

import random
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFilter

from src.augmentation.offline_common import generate_offline_augmentation


def generate_blur_compression_augmentation(
    images_dir: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    ratio: float,
    seed: int = 42,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Create deterministic CCTV blur/compression variants from training images."""
    return generate_offline_augmentation(
        augmentation_type="blur_compression",
        image_prefix="blur_",
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_images_dir=output_images_dir,
        output_labels_dir=output_labels_dir,
        ratio=ratio,
        seed=seed,
        overwrite=overwrite,
        transform=_transform_blur_compression,
    )


def _transform_blur_compression(image: Image.Image, rng: random.Random) -> Image.Image:
    """Apply mild blur followed by JPEG artifact simulation."""
    if rng.random() < 0.55:
        blurred = _apply_motion_blur(image, kernel_size=rng.choice([3, 5]))
    else:
        blurred = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.55, 1.15)))

    buffer = BytesIO()
    blurred.save(
        buffer,
        format="JPEG",
        quality=rng.randint(42, 68),
        optimize=False,
        subsampling=2,
    )
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def _apply_motion_blur(image: Image.Image, kernel_size: int) -> Image.Image:
    weights = [0.0] * (kernel_size * kernel_size)
    middle = kernel_size // 2
    for index in range(kernel_size):
        weights[middle * kernel_size + index] = 1.0 / kernel_size
    return image.filter(ImageFilter.Kernel((kernel_size, kernel_size), weights, scale=1.0))
