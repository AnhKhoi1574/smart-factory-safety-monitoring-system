"""Generate infrared-like offline augmentations for training data."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

from src.augmentation.offline_common import (
    generate_offline_augmentation,
    uint8_array_to_image,
)


def generate_ir_augmentation(
    images_dir: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    ratio: float,
    seed: int = 42,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Create deterministic IR-style variants from training images only."""
    return generate_offline_augmentation(
        augmentation_type="ir",
        image_prefix="ir_",
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_images_dir=output_images_dir,
        output_labels_dir=output_labels_dir,
        ratio=ratio,
        seed=seed,
        overwrite=overwrite,
        transform=_transform_ir,
    )


def _transform_ir(image: Image.Image, rng: random.Random) -> Image.Image:
    """Simulate light grayscale CCTV/IR appearance without moving objects."""
    original_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    grayscale = image.convert("L")
    grayscale = _apply_clahe_if_available(grayscale)

    contrast_factor = rng.uniform(1.04, 1.16)
    brightness_factor = rng.uniform(0.92, 1.04)
    grayscale = ImageEnhance.Contrast(grayscale).enhance(contrast_factor)
    grayscale = ImageEnhance.Brightness(grayscale).enhance(brightness_factor)

    gray_array = np.asarray(grayscale, dtype=np.float32)
    noise = np.random.default_rng(rng.randrange(0, 2**32)).normal(
        loc=0.0,
        scale=rng.uniform(1.0, 3.0),
        size=gray_array.shape,
    )
    gray_array = np.clip(gray_array + noise, 0, 255)

    # Keep a subtle cool/green cast while preserving some original color cues.
    ir_tint = np.stack(
        [
            gray_array * 0.90,
            gray_array * 0.98,
            gray_array * 0.94,
        ],
        axis=-1,
    )
    rgb = (ir_tint * 0.72) + (original_array * 0.28)
    return uint8_array_to_image(rgb)


def _apply_clahe_if_available(grayscale: Image.Image) -> Image.Image:
    try:
        import cv2
    except ImportError:
        return grayscale

    gray_array = np.asarray(grayscale, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.15, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_array)
    return Image.fromarray(enhanced, mode="L")
