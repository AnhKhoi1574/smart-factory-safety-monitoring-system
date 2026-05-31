"""Validate teammate-provided YOLO datasets before merging originals.

The public entry point returns a pandas DataFrame because notebook users need a
report they can display, filter, and save directly to CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd


VALIDATION_COLUMNS = [
    "source",
    "base_name",
    "image_path",
    "label_path",
    "status",
    "errors",
    "warnings",
    "image_width",
    "image_height",
    "num_objects",
    "num_person",
    "num_helmet",
    "num_vest",
]

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_CLASS_IDS = {0, 1, 2}
BOUNDARY_TOLERANCE = 1e-4


@dataclass(slots=True)
class LabelStats:
    """Parsed object counts from one YOLO label file."""

    num_objects: int = 0
    num_person: int = 0
    num_helmet: int = 0
    num_vest: int = 0


def validate_dataset(
    raw_sources_dir: Path,
    class_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Validate all teammate source folders under a raw sources directory.

    Args:
        raw_sources_dir: Directory containing source folders such as
            ``member_01/images`` and ``member_01/labels``.
        class_ids: Optional allowed YOLO class IDs. Defaults to ``{0, 1, 2}``.

    Returns:
        A row-level validation report. Rows with ``status == "valid"`` are safe
        for the merge step.
    """
    raw_sources_dir = Path(raw_sources_dir)
    allowed_class_ids = class_ids or DEFAULT_CLASS_IDS
    validation_rows: list[dict[str, Any]] = []

    if not raw_sources_dir.exists():
        return _empty_report()

    source_directories = sorted(
        path for path in raw_sources_dir.iterdir() if path.is_dir()
    )
    if not source_directories:
        return _empty_report()

    for source_directory in source_directories:
        validation_rows.extend(
            _validate_source_directory(source_directory, allowed_class_ids)
        )

    report = pd.DataFrame(validation_rows, columns=VALIDATION_COLUMNS)
    return _annotate_cross_source_findings(report)


def _validate_source_directory(
    source_directory: Path,
    allowed_class_ids: set[int],
) -> list[dict[str, Any]]:
    """Validate one teammate source folder."""
    source_name = source_directory.name
    images_dir = source_directory / "images"
    labels_dir = source_directory / "labels"
    source_rows: list[dict[str, Any]] = []

    if not images_dir.exists() or not labels_dir.exists():
        missing_parts = []
        if not images_dir.exists():
            missing_parts.append("images/")
        if not labels_dir.exists():
            missing_parts.append("labels/")
        source_rows.append(
            _build_report_row(
                source=source_name,
                base_name="",
                image_path=None,
                label_path=None,
                errors=[f"Missing required folder(s): {', '.join(missing_parts)}"],
            )
        )

    image_paths = _collect_image_paths(images_dir) if images_dir.exists() else {}
    unsupported_images = (
        _collect_unsupported_image_paths(images_dir) if images_dir.exists() else []
    )
    label_paths = _collect_label_paths(labels_dir) if labels_dir.exists() else {}
    duplicate_base_names = {
        base_name
        for base_name, paths in image_paths.items()
        if len(paths) > 1
    }

    all_base_names = sorted(set(image_paths) | set(label_paths))
    for base_name in all_base_names:
        images_for_base_name = image_paths.get(base_name, [])
        label_path = label_paths.get(base_name)

        if images_for_base_name:
            for image_path in images_for_base_name:
                source_rows.append(
                    _validate_sample(
                        source=source_name,
                        base_name=base_name,
                        image_path=image_path,
                        label_path=label_path,
                        allowed_class_ids=allowed_class_ids,
                        duplicate_in_source=base_name in duplicate_base_names,
                    )
                )
        else:
            source_rows.append(
                _build_report_row(
                    source=source_name,
                    base_name=base_name,
                    image_path=None,
                    label_path=label_path,
                    errors=["Label file has no matching image"],
                )
            )

    for image_path in unsupported_images:
        source_rows.append(
            _build_report_row(
                source=source_name,
                base_name=image_path.stem,
                image_path=image_path,
                label_path=None,
                errors=[f"Unsupported image extension: {image_path.suffix}"],
            )
        )

    return source_rows


def _validate_sample(
    source: str,
    base_name: str,
    image_path: Path,
    label_path: Path | None,
    allowed_class_ids: set[int],
    duplicate_in_source: bool,
) -> dict[str, Any]:
    """Validate one image and its paired YOLO label file."""
    errors: list[str] = []
    warnings: list[str] = []
    image_width: int | None = None
    image_height: int | None = None
    label_stats = LabelStats()

    if duplicate_in_source:
        errors.append("Duplicate image base name inside the same source folder")

    try:
        image_width, image_height = _read_image_size(image_path)
    except ValueError as exc:
        errors.append(str(exc))

    if label_path is None:
        errors.append("Image has no matching label file")
    elif image_width is not None and image_height is not None:
        label_stats, label_errors = _validate_label_file(
            label_path=label_path,
            image_width=image_width,
            image_height=image_height,
            allowed_class_ids=allowed_class_ids,
        )
        errors.extend(label_errors)

    return _build_report_row(
        source=source,
        base_name=base_name,
        image_path=image_path,
        label_path=label_path,
        errors=errors,
        warnings=warnings,
        image_width=image_width,
        image_height=image_height,
        label_stats=label_stats,
    )


def _validate_label_file(
    label_path: Path,
    image_width: int,
    image_height: int,
    allowed_class_ids: set[int],
) -> tuple[LabelStats, list[str]]:
    """Validate YOLO label rows and return object counts."""
    errors: list[str] = []
    label_stats = LabelStats()

    try:
        raw_text = label_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return label_stats, [f"Could not read label file: {exc}"]

    if not raw_text:
        return label_stats, ["Label file is empty"]

    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        values = line.split()
        if len(values) != 5:
            errors.append(
                f"Line {line_number}: expected 5 values, found {len(values)}"
            )
            continue

        parsed_values = _parse_label_values(values, line_number)
        if isinstance(parsed_values, str):
            errors.append(parsed_values)
            continue

        class_id, x_center, y_center, box_width, box_height = parsed_values
        if class_id not in allowed_class_ids:
            errors.append(
                f"Line {line_number}: class_id {class_id} is not allowed"
            )

        geometry_errors = _validate_yolo_geometry(
            line_number=line_number,
            x_center=x_center,
            y_center=y_center,
            box_width=box_width,
            box_height=box_height,
            image_width=image_width,
            image_height=image_height,
        )
        errors.extend(geometry_errors)

        if not geometry_errors and class_id in allowed_class_ids:
            _add_class_count(label_stats, class_id)

    return label_stats, errors


def _parse_label_values(
    values: list[str],
    line_number: int,
) -> tuple[int, float, float, float, float] | str:
    """Parse one YOLO label row."""
    try:
        class_id_float = float(values[0])
        coordinates = [float(value) for value in values[1:]]
    except ValueError:
        return f"Line {line_number}: class_id and coordinates must be numeric"

    if not class_id_float.is_integer():
        return f"Line {line_number}: class_id must be an integer"

    class_id = int(class_id_float)
    x_center, y_center, box_width, box_height = coordinates
    return class_id, x_center, y_center, box_width, box_height


def _validate_yolo_geometry(
    line_number: int,
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> list[str]:
    """Validate normalized YOLO coordinates and derived image boundaries."""
    errors: list[str] = []
    coordinate_values = {
        "x_center": x_center,
        "y_center": y_center,
        "width": box_width,
        "height": box_height,
    }

    for coordinate_name, coordinate_value in coordinate_values.items():
        if not 0 <= coordinate_value <= 1:
            errors.append(
                f"Line {line_number}: {coordinate_name} must be in [0, 1]"
            )

    if box_width <= 0:
        errors.append(f"Line {line_number}: width must be greater than 0")
    if box_height <= 0:
        errors.append(f"Line {line_number}: height must be greater than 0")

    # YOLO labels can be normalized correctly but still imply boxes slightly
    # outside the image due to rounding. The tolerance keeps that check practical.
    left = (x_center - box_width / 2) * image_width
    right = (x_center + box_width / 2) * image_width
    top = (y_center - box_height / 2) * image_height
    bottom = (y_center + box_height / 2) * image_height
    tolerance_x = BOUNDARY_TOLERANCE * image_width
    tolerance_y = BOUNDARY_TOLERANCE * image_height

    if left < -tolerance_x or right > image_width + tolerance_x:
        errors.append(f"Line {line_number}: box x-boundaries exceed image")
    if top < -tolerance_y or bottom > image_height + tolerance_y:
        errors.append(f"Line {line_number}: box y-boundaries exceed image")

    return errors


def _annotate_cross_source_findings(report: pd.DataFrame) -> pd.DataFrame:
    """Add warnings for allowed cross-source duplicates and exact duplicates."""
    if report.empty:
        return report

    report = report.copy()
    duplicate_base_mask = report.duplicated(
        subset=["base_name"],
        keep=False,
    ) & report["base_name"].ne("")
    duplicate_base_names = report.loc[duplicate_base_mask, "base_name"].unique()

    for base_name in duplicate_base_names:
        sources = set(report.loc[report["base_name"] == base_name, "source"])
        if len(sources) > 1:
            _append_warning(
                report,
                report["base_name"] == base_name,
                "Duplicate base name appears across sources; merge naming handles this",
            )

    valid_image_rows = report["image_path"].fillna("").ne("")
    hash_to_indexes: dict[str, list[int]] = {}
    image_path_items = report.loc[valid_image_rows, "image_path"].items()
    for row_index, image_path_text in image_path_items:
        image_path = Path(image_path_text)
        try:
            file_hash = _hash_file(image_path)
        except OSError:
            continue
        hash_to_indexes.setdefault(file_hash, []).append(row_index)

    for row_indexes in hash_to_indexes.values():
        if len(row_indexes) > 1:
            _append_warning(
                report,
                report.index.isin(row_indexes),
                "Exact duplicate image content detected",
            )

    return report


def _append_warning(
    report: pd.DataFrame,
    row_mask: pd.Series,
    message: str,
) -> None:
    """Append a warning message to matching rows in-place."""
    for row_index in report.index[row_mask]:
        current_warning = str(report.at[row_index, "warnings"] or "")
        if message in current_warning:
            continue
        report.at[row_index, "warnings"] = (
            message if not current_warning else f"{current_warning}; {message}"
        )


def _collect_image_paths(images_dir: Path) -> dict[str, list[Path]]:
    """Collect supported image files grouped by base name."""
    image_paths: dict[str, list[Path]] = {}
    for image_path in sorted(images_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
            image_paths.setdefault(image_path.stem, []).append(image_path)
    return image_paths


def _collect_unsupported_image_paths(images_dir: Path) -> list[Path]:
    """Collect files in images/ that look like unsupported image inputs."""
    return [
        path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() not in VALID_IMAGE_EXTENSIONS
    ]


def _collect_label_paths(labels_dir: Path) -> dict[str, Path]:
    """Collect YOLO label files by base name."""
    return {
        label_path.stem: label_path
        for label_path in sorted(labels_dir.iterdir())
        if label_path.is_file() and label_path.suffix.lower() == ".txt"
    }


def _read_image_size(image_path: Path) -> tuple[int, int]:
    """Read image dimensions with Pillow."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("Pillow is required to read image dimensions") from exc

    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception as exc:
        raise ValueError(f"Image is unreadable or corrupted: {exc}") from exc

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    return width, height


def _add_class_count(label_stats: LabelStats, class_id: int) -> None:
    """Increment class-specific counts for a valid YOLO row."""
    label_stats.num_objects += 1
    if class_id == 0:
        label_stats.num_person += 1
    elif class_id == 1:
        label_stats.num_helmet += 1
    elif class_id == 2:
        label_stats.num_vest += 1


def _hash_file(path: Path) -> str:
    """Compute a SHA-256 hash for duplicate-content warnings."""
    digest = sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_report_row(
    source: str,
    base_name: str,
    image_path: Path | None,
    label_path: Path | None,
    errors: list[str],
    warnings: list[str] | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    label_stats: LabelStats | None = None,
) -> dict[str, Any]:
    """Build a normalized validation report row."""
    label_stats = label_stats or LabelStats()
    warnings = warnings or []
    return {
        "source": source,
        "base_name": base_name,
        "image_path": str(image_path) if image_path else "",
        "label_path": str(label_path) if label_path else "",
        "status": "invalid" if errors else "valid",
        "errors": "; ".join(errors),
        "warnings": "; ".join(warnings),
        "image_width": image_width,
        "image_height": image_height,
        "num_objects": label_stats.num_objects,
        "num_person": label_stats.num_person,
        "num_helmet": label_stats.num_helmet,
        "num_vest": label_stats.num_vest,
    }


def _empty_report() -> pd.DataFrame:
    """Return an empty validation report with the expected schema."""
    return pd.DataFrame(columns=VALIDATION_COLUMNS)
