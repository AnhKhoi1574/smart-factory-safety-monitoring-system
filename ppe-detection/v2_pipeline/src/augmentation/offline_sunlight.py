"""Generate harsh sunlight offline augmentations for training data."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from src.augmentation.offline_common import (
    generate_offline_augmentation,
    image_to_uint8_array,
    uint8_array_to_image,
)


def generate_sunlight_augmentation(
    images_dir: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    ratio: float,
    seed: int = 42,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Create deterministic sunlight/glare variants from training images only."""
    return generate_offline_augmentation(
        augmentation_type="sunlight",
        image_prefix="sun_",
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_images_dir=output_images_dir,
        output_labels_dir=output_labels_dir,
        ratio=ratio,
        seed=seed,
        overwrite=overwrite,
        transform=_transform_sunlight,
    )


def _transform_sunlight(image: Image.Image, rng: random.Random) -> Image.Image:
    """Add moderate exposure, contrast, and localized glare."""
    brightened = ImageEnhance.Brightness(image).enhance(rng.uniform(1.12, 1.28))
    contrasted = ImageEnhance.Contrast(brightened).enhance(rng.uniform(1.05, 1.18))
    color_boosted = ImageEnhance.Color(contrasted).enhance(rng.uniform(0.95, 1.08))

    base = image_to_uint8_array(color_boosted).astype(np.float32)
    height, width = base.shape[:2]
    glare_mask = _make_glare_mask(width=width, height=height, rng=rng)
    warm_glare = np.array([255.0, 244.0, 205.0], dtype=np.float32)
    strength = rng.uniform(0.18, 0.34)
    base = base * (1.0 - glare_mask[..., None] * strength) + (
        warm_glare * glare_mask[..., None] * strength
    )

    # Prevent the whole frame from becoming pure white while retaining highlights.
    base = np.minimum(base, 245.0)
    return uint8_array_to_image(base)


def _make_glare_mask(width: int, height: int, rng: random.Random) -> np.ndarray:
    center_x = rng.uniform(0.15 * width, 0.85 * width)
    center_y = rng.uniform(0.05 * height, 0.65 * height)
    radius_x = rng.uniform(0.22 * width, 0.45 * width)
    radius_y = rng.uniform(0.16 * height, 0.36 * height)

    y_grid, x_grid = np.ogrid[:height, :width]
    distance = ((x_grid - center_x) / radius_x) ** 2 + (
        (y_grid - center_y) / radius_y
    ) ** 2
    mask = np.clip(1.0 - distance, 0.0, 1.0)
    mask = mask ** rng.uniform(1.6, 2.3)

    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    blur_radius = max(8, int(min(width, height) * 0.045))
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.asarray(mask_image, dtype=np.float32) / 255.0
