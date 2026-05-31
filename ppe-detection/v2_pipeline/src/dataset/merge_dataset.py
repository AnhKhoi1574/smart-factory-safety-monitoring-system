"""Merge validated PPE samples into normalized image-label folders."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd


MERGE_REPORT_COLUMNS = [
    "source",
    "base_name",
    "original_image_path",
    "original_label_path",
    "new_image_path",
    "new_label_path",
    "merge_status",
    "notes",
]


def merge_valid_samples(
    validation_df: pd.DataFrame,
    output_images_dir: Path,
    output_labels_dir: Path,
    prefix: str = "ppe",
) -> pd.DataFrame:
    """Copy valid image-label pairs into the master original dataset.

    Args:
        validation_df: Report produced by ``validate_dataset``.
        output_images_dir: Destination folder for merged images.
        output_labels_dir: Destination folder for merged labels.
        prefix: Prefix used in generated filenames, such as ``ppe`` or ``test``.

    Returns:
        A DataFrame describing copied files and merge status.
    """
    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    if validation_df.empty:
        return _empty_merge_report()

    valid_samples = validation_df.loc[
        validation_df["status"].eq("valid")
    ].copy()
    if valid_samples.empty:
        return _empty_merge_report()

    valid_samples = valid_samples.sort_values(
        by=["source", "base_name", "image_path"],
        kind="stable",
    ).reset_index(drop=True)

    merge_rows: list[dict[str, Any]] = []
    for sample_index, sample_row in valid_samples.iterrows():
        source = str(sample_row["source"])
        image_path = Path(str(sample_row["image_path"]))
        label_path = Path(str(sample_row["label_path"]))
        base_filename = f"{prefix}_{sample_index + 1:05d}"
        destination_image_path = output_images_dir / (
            f"{base_filename}{image_path.suffix.lower()}"
        )
        destination_label_path = output_labels_dir / f"{base_filename}.txt"
        destination_image_path, destination_label_path = _avoid_overwrite_pair(
            destination_image_path,
            destination_label_path,
        )

        try:
            shutil.copy2(image_path, destination_image_path)
            shutil.copy2(label_path, destination_label_path)
            merge_status = "merged"
            notes = ""
        except OSError as exc:
            merge_status = "failed"
            notes = str(exc)

        merge_rows.append(
            {
                "source": source,
                "base_name": str(sample_row["base_name"]),
                "original_image_path": str(image_path),
                "original_label_path": str(label_path),
                "new_image_path": str(destination_image_path),
                "new_label_path": str(destination_label_path),
                "merge_status": merge_status,
                "notes": notes,
            }
        )

    return pd.DataFrame(merge_rows, columns=MERGE_REPORT_COLUMNS)


def _avoid_overwrite_pair(
    image_path: Path,
    label_path: Path,
) -> tuple[Path, Path]:
    """Generate a matching image/label path pair that does not overwrite files."""
    if not image_path.exists() and not label_path.exists():
        return image_path, label_path

    image_stem = image_path.stem
    image_suffix = image_path.suffix
    label_suffix = label_path.suffix
    for suffix_index in range(2, 10_000):
        candidate_image_path = image_path.with_name(
            f"{image_stem}_{suffix_index}{image_suffix}"
        )
        candidate_label_path = label_path.with_name(
            f"{image_stem}_{suffix_index}{label_suffix}"
        )
        if not candidate_image_path.exists() and not candidate_label_path.exists():
            return candidate_image_path, candidate_label_path

    raise FileExistsError(f"Could not find safe filename for {image_path.name}")


def _empty_merge_report() -> pd.DataFrame:
    """Return an empty merge report with the expected schema."""
    return pd.DataFrame(columns=MERGE_REPORT_COLUMNS)
