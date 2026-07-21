"""Read-only exploratory analysis helpers for PPE YOLO datasets.

These helpers are intentionally lightweight. Notebook 02 uses them to inspect
the three input source lanes before splitting:

``open_source``
    Train-only public/external perspective data.
``factory_source``
    Target-domain CCTV data for train/validation.
``test_source``
    Final untouched test data.

The functions only read image and label files. They do not split, copy,
augment, or modify dataset contents.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_images(images_dir: Path) -> list[Path]:
    """Return image paths in deterministic order."""
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _image_size(image_path: Path) -> tuple[int, int]:
    """Read image dimensions without modifying the file."""
    with Image.open(image_path) as image:
        return image.size


def collect_image_records(images_dir: Path) -> pd.DataFrame:
    """Collect one row per readable image in a YOLO image directory.

    Hidden files such as `.gitkeep` are ignored by extension filtering. If an
    image cannot be opened, it is skipped here because Notebook 01 is already
    responsible for validation failures.
    """
    rows: list[dict[str, Any]] = []
    for image_path in _iter_images(images_dir):
        try:
            image_width, image_height = _image_size(image_path)
        except OSError:
            continue

        rows.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "image_width": image_width,
                "image_height": image_height,
                "aspect_ratio": image_width / image_height if image_height else 0.0,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["image_name", "image_path", "image_width", "image_height", "aspect_ratio"],
    )


def collect_bbox_records(
    images_dir: Path,
    labels_dir: Path,
    class_names: dict[int, str],
) -> pd.DataFrame:
    """Collect YOLO bounding boxes with normalized and pixel dimensions.

    This function trusts that Notebook 01 has already validated YOLO syntax.
    Any malformed rows are skipped defensively so EDA can still run and show the
    valid portion of the dataset.
    """
    image_df = collect_image_records(images_dir)
    image_lookup = {
        row.image_name: (int(row.image_width), int(row.image_height))
        for row in image_df.itertuples(index=False)
    }

    rows: list[dict[str, Any]] = []
    for image_name, (image_width, image_height) in image_lookup.items():
        image_stem = Path(image_name).stem
        label_path = Path(labels_dir) / f"{image_stem}.txt"
        if not label_path.exists():
            continue

        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) != 5:
                continue

            try:
                class_id = int(float(parts[0]))
                x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])
            except ValueError:
                continue

            box_width_px = width_norm * image_width
            box_height_px = height_norm * image_height
            rows.append(
                {
                    "image_name": image_name,
                    "label_name": label_path.name,
                    "line_number": line_number,
                    "class_id": class_id,
                    "class_name": class_names.get(class_id, f"class_{class_id}"),
                    "x_center_norm": x_center_norm,
                    "y_center_norm": y_center_norm,
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

    return pd.DataFrame(
        rows,
        columns=[
            "image_name",
            "label_name",
            "line_number",
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
        ],
    )


def compute_class_distribution(bbox_df: pd.DataFrame) -> pd.DataFrame:
    """Compute object counts and percentages by class."""
    columns = ["class_id", "class_name", "object_count", "percentage"]
    if bbox_df.empty:
        return pd.DataFrame(columns=columns)

    distribution = (
        bbox_df.groupby(["class_id", "class_name"], as_index=False)
        .size()
        .rename(columns={"size": "object_count"})
        .sort_values("class_id")
    )
    total = int(distribution["object_count"].sum())
    distribution["percentage"] = (
        distribution["object_count"] / total * 100 if total else 0.0
    )
    return distribution[columns]


def compute_objects_per_image(
    bbox_df: pd.DataFrame,
    image_df: pd.DataFrame,
    class_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Compute total and per-class object counts for every image.

    Args:
        bbox_df: Box-level records from :func:`collect_bbox_records`.
        image_df: Image-level records from :func:`collect_image_records`.
        class_names: Optional class map. If omitted, classes are inferred from
            `bbox_df`. Passing the config class map keeps missing classes visible
            as zero-count columns.

    Returns:
        One row per image with `object_count` and `<class_name>_count` columns.
    """
    normalized_class_names = _normalize_class_names(class_names, bbox_df)
    count_columns = [f"{class_name}_count" for class_name in normalized_class_names.values()]
    base_columns = ["image_name", "object_count", *count_columns]
    if image_df.empty:
        return pd.DataFrame(columns=base_columns)

    output = image_df[["image_name"]].copy()
    counts = bbox_df.groupby("image_name").size().rename("object_count") if not bbox_df.empty else pd.Series(dtype=int)
    output = output.merge(counts, on="image_name", how="left")
    output["object_count"] = output["object_count"].fillna(0).astype(int)

    for class_name in normalized_class_names.values():
        if bbox_df.empty:
            output[f"{class_name}_count"] = 0
            continue
        class_counts = (
            bbox_df.loc[bbox_df["class_name"] == class_name]
            .groupby("image_name")
            .size()
            .rename(f"{class_name}_count")
        )
        output = output.merge(class_counts, on="image_name", how="left")
        output[f"{class_name}_count"] = output[f"{class_name}_count"].fillna(0).astype(int)

    return output[base_columns]


def compute_eda_warnings(
    bbox_df: pd.DataFrame,
    objects_per_image_df: pd.DataFrame,
    tiny_area_threshold: float = 0.0005,
    large_area_threshold: float = 0.50,
    many_objects_quantile: float = 0.95,
    many_objects_minimum: int = 20,
) -> pd.DataFrame:
    """Create image-level warning rows for review, without invalidating samples.

    Warnings are deliberately practical rather than exhaustive. They point to
    samples worth reviewing before splitting, especially target-domain factory
    images with people but missing PPE or role-uniform boxes.
    """
    rows: list[dict[str, str]] = []

    for row in objects_per_image_df.itertuples(index=False):
        person_count = int(getattr(row, "person_count", 0))
        helmet_count = int(getattr(row, "helmet_count", 0))
        vest_count = int(getattr(row, "vest_count", 0))
        coverall_count = int(getattr(row, "cleaning_coverall_count", 0))

        if person_count > 0 and helmet_count == 0:
            rows.append(
                {
                    "image_name": row.image_name,
                    "warning_type": "person_without_helmet",
                    "details": f"{person_count} person boxes, 0 helmet boxes",
                }
            )
        if person_count > 0 and vest_count == 0 and coverall_count == 0:
            rows.append(
                {
                    "image_name": row.image_name,
                    "warning_type": "person_without_role_uniform",
                    "details": (
                        f"{person_count} person boxes, 0 vest boxes, "
                        "0 cleaning_coverall boxes"
                    ),
                }
            )
        if person_count == 0 and (helmet_count > 0 or vest_count > 0 or coverall_count > 0):
            rows.append(
                {
                    "image_name": row.image_name,
                    "warning_type": "ppe_without_person",
                    "details": (
                        f"{helmet_count} helmet boxes, {vest_count} vest boxes, "
                        f"{coverall_count} cleaning_coverall boxes, 0 person boxes"
                    ),
                }
            )

    if not objects_per_image_df.empty:
        quantile_value = objects_per_image_df["object_count"].quantile(many_objects_quantile)
        many_objects_threshold = max(many_objects_minimum, int(quantile_value))
        for row in objects_per_image_df.loc[
            objects_per_image_df["object_count"] >= many_objects_threshold
        ].itertuples(index=False):
            if row.object_count > 0:
                rows.append(
                    {
                        "image_name": row.image_name,
                        "warning_type": "many_objects",
                        "details": f"{row.object_count} total boxes; threshold={many_objects_threshold}",
                    }
                )

    if not bbox_df.empty:
        for row in bbox_df.loc[bbox_df["box_area_norm"] <= tiny_area_threshold].itertuples(index=False):
            rows.append(
                {
                    "image_name": row.image_name,
                    "warning_type": "very_tiny_box",
                    "details": f"{row.class_name} area={row.box_area_norm:.6f}",
                }
            )
        for row in bbox_df.loc[bbox_df["box_area_norm"] >= large_area_threshold].itertuples(index=False):
            rows.append(
                {
                    "image_name": row.image_name,
                    "warning_type": "very_large_box",
                    "details": f"{row.class_name} area={row.box_area_norm:.3f}",
                }
            )

    return pd.DataFrame(rows, columns=["image_name", "warning_type", "details"])


def summarize_yolo_dataset(
    images_dir: Path,
    labels_dir: Path,
    class_names: dict[int, str],
) -> dict[str, Any]:
    """Return compact dataset-level EDA metrics for a YOLO image/label pair."""
    image_df = collect_image_records(images_dir)
    bbox_df = collect_bbox_records(images_dir, labels_dir, class_names)
    objects_per_image_df = compute_objects_per_image(bbox_df, image_df, class_names)
    class_distribution_df = compute_class_distribution(bbox_df)

    class_counts = {
        str(row.class_name): int(row.object_count)
        for row in class_distribution_df.itertuples(index=False)
    }
    small_object_counts = Counter()
    if not bbox_df.empty:
        small_boxes = bbox_df.loc[bbox_df["box_area_norm"] <= 0.0005]
        small_object_counts.update(small_boxes["class_name"].tolist())

    return {
        "total_images": int(len(image_df)),
        "total_objects": int(len(bbox_df)),
        "class_counts": class_counts,
        "average_objects_per_image": float(objects_per_image_df["object_count"].mean())
        if not objects_per_image_df.empty
        else 0.0,
        "unique_resolutions": int(
            image_df[["image_width", "image_height"]].drop_duplicates().shape[0]
        )
        if not image_df.empty
        else 0,
        "small_object_counts": dict(small_object_counts),
        "images_with_person_no_helmet": int(
            (
                (objects_per_image_df["person_count"] > 0)
                & (objects_per_image_df["helmet_count"] == 0)
            ).sum()
        )
        if not objects_per_image_df.empty
        else 0,
        "images_with_person_no_role_uniform": int(
            (
                (objects_per_image_df["person_count"] > 0)
                & (objects_per_image_df["vest_count"] == 0)
                & (objects_per_image_df.get("cleaning_coverall_count", 0) == 0)
            ).sum()
        )
        if not objects_per_image_df.empty
        else 0,
        "images_with_ppe_no_person": int(
            (
                (objects_per_image_df["person_count"] == 0)
                & (
                    (objects_per_image_df["helmet_count"] > 0)
                    | (objects_per_image_df["vest_count"] > 0)
                    | (objects_per_image_df.get("cleaning_coverall_count", 0) > 0)
                )
            ).sum()
        )
        if not objects_per_image_df.empty
        else 0,
    }


def summarize_source_eda(
    source_name: str,
    images_dir: Path,
    labels_dir: Path,
    class_names: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the minimal EDA tables for one PPE source lane.

    Args:
        source_name: Source lane name, such as `factory_source`.
        images_dir: Directory containing images.
        labels_dir: Directory containing YOLO labels.
        class_names: Config class map.

    Returns:
        A tuple of `(source_summary, class_distribution, bbox_size_summary,
        warnings)`. Each DataFrame includes a `source` column so results from
        multiple source lanes can be concatenated directly.
    """
    image_df = collect_image_records(images_dir)
    bbox_df = collect_bbox_records(images_dir, labels_dir, class_names)
    objects_per_image_df = compute_objects_per_image(bbox_df, image_df, class_names)
    warnings_df = compute_eda_warnings(bbox_df, objects_per_image_df)
    class_distribution_df = compute_class_distribution(bbox_df)
    bbox_size_df = summarize_bbox_sizes_by_class(bbox_df, class_names)

    if not warnings_df.empty:
        warnings_df.insert(0, "source", source_name)
    else:
        warnings_df = pd.DataFrame(columns=["source", "image_name", "warning_type", "details"])

    summary_row: dict[str, Any] = {
        "source": source_name,
        "num_images": int(len(image_df)),
        "num_labeled_images": int((objects_per_image_df["object_count"] > 0).sum())
        if not objects_per_image_df.empty
        else 0,
        "num_empty_images": int((objects_per_image_df["object_count"] == 0).sum())
        if not objects_per_image_df.empty
        else 0,
        "total_objects": int(len(bbox_df)),
        "mean_objects_per_image": float(objects_per_image_df["object_count"].mean())
        if not objects_per_image_df.empty
        else 0.0,
        "unique_resolutions": int(
            image_df[["image_width", "image_height"]].drop_duplicates().shape[0]
        )
        if not image_df.empty
        else 0,
    }
    for class_id, class_name in sorted(class_names.items()):
        count = 0
        if not bbox_df.empty:
            count = int((bbox_df["class_id"] == int(class_id)).sum())
        summary_row[f"num_{class_name}"] = count

    source_summary_df = pd.DataFrame([summary_row])

    for frame in (class_distribution_df, bbox_size_df):
        if not frame.empty:
            frame.insert(0, "source", source_name)

    return source_summary_df, class_distribution_df, bbox_size_df, warnings_df


def summarize_bbox_sizes_by_class(
    bbox_df: pd.DataFrame,
    class_names: dict[int, str],
    tiny_area_threshold: float = 0.0005,
) -> pd.DataFrame:
    """Summarize bounding-box size patterns by class.

    Small PPE and role-uniform objects are common in elevated CCTV. This summary
    keeps the useful signal without saving a large per-box CSV from Notebook 02.
    """
    columns = [
        "class_id",
        "class_name",
        "num_boxes",
        "mean_box_area_norm",
        "median_box_area_norm",
        "mean_box_width_px",
        "mean_box_height_px",
        "tiny_box_count",
    ]
    if bbox_df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for class_id, class_name in sorted(class_names.items()):
        class_boxes = bbox_df.loc[bbox_df["class_id"] == int(class_id)]
        if class_boxes.empty:
            rows.append(
                {
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "num_boxes": 0,
                    "mean_box_area_norm": 0.0,
                    "median_box_area_norm": 0.0,
                    "mean_box_width_px": 0.0,
                    "mean_box_height_px": 0.0,
                    "tiny_box_count": 0,
                }
            )
            continue

        rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "num_boxes": int(len(class_boxes)),
                "mean_box_area_norm": float(class_boxes["box_area_norm"].mean()),
                "median_box_area_norm": float(class_boxes["box_area_norm"].median()),
                "mean_box_width_px": float(class_boxes["box_width_px"].mean()),
                "mean_box_height_px": float(class_boxes["box_height_px"].mean()),
                "tiny_box_count": int(
                    (class_boxes["box_area_norm"] <= tiny_area_threshold).sum()
                ),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _normalize_class_names(
    class_names: dict[int, str] | None,
    bbox_df: pd.DataFrame,
) -> dict[int, str]:
    """Return a stable class map, falling back to classes seen in bbox_df."""
    if class_names:
        return {int(class_id): str(class_name) for class_id, class_name in class_names.items()}
    if bbox_df.empty:
        return {}
    observed = bbox_df[["class_id", "class_name"]].drop_duplicates().sort_values("class_id")
    return {
        int(row.class_id): str(row.class_name)
        for row in observed.itertuples(index=False)
    }


def summarize_dataset(dataset_dir: Path) -> dict[str, int]:
    """Backward-compatible summary for a dataset with images/ and labels/ folders."""
    dataset_dir = Path(dataset_dir)
    summary = summarize_yolo_dataset(
        dataset_dir / "images",
        dataset_dir / "labels",
        {0: "person", 1: "helmet", 2: "vest", 3: "cleaning_coverall"},
    )
    return {
        "images": int(summary["total_images"]),
        "labels": len(list((dataset_dir / "labels").glob("*.txt")))
        if (dataset_dir / "labels").exists()
        else 0,
        "objects": int(summary["total_objects"]),
    }
