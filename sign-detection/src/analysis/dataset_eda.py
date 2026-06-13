"""Minimal pre-split profiling utilities for factory sign detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


CLASS_COLUMNS: dict[int, str] = {
    0: "M014_Helmet",
    1: "M015_Vest",
    2: "P004_NoThoroughfare",
    3: "W011_Slippery",
}


def _split_group_key(row: pd.Series) -> str:
    return (
        f"M014={int(bool(row['has_M014_Helmet']))}|"
        f"M015={int(bool(row['has_M015_Vest']))}|"
        f"P004={int(bool(row['has_P004_NoThoroughfare']))}|"
        f"W011={int(bool(row['has_W011_Slippery']))}|"
        f"NONE={int(bool(row['is_no_sign']))}"
    )


def build_image_level_profile(validation_df: pd.DataFrame) -> pd.DataFrame:
    """Create one valid-image row for Notebook 02 stratified splitting."""
    valid = validation_df[validation_df["status"] == "valid"].copy()
    if valid.empty:
        columns = [
            "image_name",
            "base_name",
            "num_objects",
            "num_M014_Helmet",
            "num_M015_Vest",
            "num_P004_NoThoroughfare",
            "num_W011_Slippery",
            "is_no_sign",
            "has_M014_Helmet",
            "has_M015_Vest",
            "has_P004_NoThoroughfare",
            "has_W011_Slippery",
            "image_width",
            "image_height",
            "aspect_ratio",
            "split_group_key",
        ]
        return pd.DataFrame(columns=columns)

    valid["aspect_ratio"] = valid["image_width"].astype(float) / valid["image_height"].astype(float)
    valid["split_group_key"] = valid.apply(_split_group_key, axis=1)
    columns = [
        "image_name",
        "base_name",
        "num_objects",
        "num_M014_Helmet",
        "num_M015_Vest",
        "num_P004_NoThoroughfare",
        "num_W011_Slippery",
        "is_no_sign",
        "has_M014_Helmet",
        "has_M015_Vest",
        "has_P004_NoThoroughfare",
        "has_W011_Slippery",
        "image_width",
        "image_height",
        "aspect_ratio",
        "split_group_key",
    ]
    return valid[columns].reset_index(drop=True)


def _parse_valid_label_rows(row: pd.Series) -> list[dict[str, Any]]:
    label_path = Path(str(row["label_path"]))
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    records: list[dict[str, Any]] = []
    image_width = int(row["image_width"])
    image_height = int(row["image_height"])
    for line in text.splitlines():
        class_text, x_text, y_text, width_text, height_text = line.split()
        class_id = int(float(class_text))
        x_center = float(x_text)
        y_center = float(y_text)
        width_norm = float(width_text)
        height_norm = float(height_text)
        box_width_px = width_norm * image_width
        box_height_px = height_norm * image_height
        records.append(
            {
                "image_name": row["image_name"],
                "class_id": class_id,
                "class_name": CLASS_COLUMNS.get(class_id, str(class_id)),
                "x_center_norm": x_center,
                "y_center_norm": y_center,
                "width_norm": width_norm,
                "height_norm": height_norm,
                "box_area_norm": width_norm * height_norm,
                "image_width": image_width,
                "image_height": image_height,
                "box_width_px": box_width_px,
                "box_height_px": box_height_px,
                "box_area_px": box_width_px * box_height_px,
            }
        )
    return records


def build_bbox_records(validation_df: pd.DataFrame, class_names: dict[int, str]) -> pd.DataFrame:
    """Build one row per object box from valid, non-empty labels."""
    records: list[dict[str, Any]] = []
    valid_labeled = validation_df[
        (validation_df["status"] == "valid") & (validation_df["num_objects"].astype(int) > 0)
    ]
    for _, row in valid_labeled.iterrows():
        for record in _parse_valid_label_rows(row):
            record["class_name"] = class_names.get(record["class_id"], record["class_name"])
            records.append(record)

    columns = [
        "image_name",
        "class_id",
        "class_name",
        "x_center_norm",
        "y_center_norm",
        "width_norm",
        "height_norm",
        "box_area_norm",
        "image_width",
        "image_height",
        "box_width_px",
        "box_height_px",
        "box_area_px",
    ]
    return pd.DataFrame(records, columns=columns)


def build_input_summary(validation_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact one-row summary for Notebook 01."""
    valid = validation_df[validation_df["status"] == "valid"]
    summary = {
        "total_images": int((validation_df["image_name"].fillna("") != "").sum()),
        "valid_images": int(len(valid)),
        "invalid_rows": int((validation_df["status"] == "invalid").sum()),
        "no_sign_images": int(valid["is_no_sign"].fillna(False).astype(bool).sum()),
        "labeled_images": int((valid["num_objects"].fillna(0).astype(int) > 0).sum()),
        "total_objects": int(valid["num_objects"].fillna(0).astype(int).sum()),
    }
    for class_name in CLASS_COLUMNS.values():
        summary[f"num_{class_name}"] = int(valid[f"num_{class_name}"].fillna(0).astype(int).sum())
    return pd.DataFrame([summary])


def build_class_distribution(validation_df: pd.DataFrame, class_names: dict[int, str]) -> pd.DataFrame:
    """Summarize object counts by class from valid rows."""
    valid = validation_df[validation_df["status"] == "valid"]
    rows = []
    for class_id, fallback_name in CLASS_COLUMNS.items():
        class_name = class_names.get(class_id, fallback_name)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "object_count": int(valid[f"num_{fallback_name}"].fillna(0).astype(int).sum()),
                "image_count": int(valid[f"has_{fallback_name}"].fillna(False).astype(bool).sum()),
            }
        )
    return pd.DataFrame(rows)
