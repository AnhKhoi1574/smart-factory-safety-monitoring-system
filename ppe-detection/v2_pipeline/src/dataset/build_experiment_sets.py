"""Build YOLO ablation datasets for the PPE v2 pipeline.

Notebook 05 uses this module after Notebook 03 has created source-aware
train/val/test splits and Notebook 04 has generated train-only offline
augmentation. The builder only copies files into experiment folders; it never
modifies the original split or augmented data.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "exp_A_original_only": {
        "include_offline_augmented": False,
        "notes": "train uses original split only; online augmentation off later",
    },
    "exp_B_online_aug": {
        "include_offline_augmented": False,
        "notes": "train uses original split only; online augmentation on later",
    },
    "exp_C_offline_aug": {
        "include_offline_augmented": True,
        "notes": "train uses original plus offline augmented samples; online augmentation off/minimal later",
    },
    "exp_D_full_pipeline": {
        "include_offline_augmented": True,
        "notes": "train uses original plus offline augmented samples; online augmentation on later",
    },
}

COPY_REPORT_COLUMNS = [
    "experiment",
    "split",
    "source_type",
    "original_image_path",
    "original_label_path",
    "copied_image_path",
    "copied_label_path",
    "status",
    "notes",
]
SUMMARY_COLUMNS = [
    "experiment",
    "split",
    "num_images",
    "num_labels",
    "num_no_object_images",
    "num_objects",
    "num_person",
    "num_helmet",
    "num_vest",
    "num_cleaning_coverall",
    "notes",
]
WARNING_COLUMNS = ["experiment", "split", "warning_type", "details"]


def build_ablation_datasets(
    splits_dir: Path,
    augmented_dir: Path,
    experiments_dir: Path,
    class_names: dict[int, str],
    output_yaml_dir: Path | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the four ablation dataset folders and dataset YAML files.

    Args:
        splits_dir: Root produced by Notebook 03, containing
            ``train/``, ``val/``, and ``test/`` YOLO splits.
        augmented_dir: Flat train-only augmentation root produced by Notebook
            04, containing ``images/`` and ``labels/``.
        experiments_dir: Destination root for generated experiment datasets.
        class_names: Class ID to class name mapping. The PPE pipeline currently
            uses four classes, including ``3 = cleaning_coverall``.
        output_yaml_dir: Folder where ``data_exp_*.yaml`` files are written.
            Defaults to the inferred ``v2_pipeline`` root.
        overwrite: If ``True``, remove files inside existing experiment folders
            before copying. If ``False``, fail safely when outputs already
            contain files.

    Returns:
        ``(copy_report, summary, warnings)`` DataFrames.

    Raises:
        FileNotFoundError: If any required split folder is missing.
        ValueError: If a required original split has no valid image-label pairs.
        FileExistsError: If experiment folders already contain files and
            ``overwrite`` is ``False``.
    """
    splits_dir = Path(splits_dir)
    augmented_dir = Path(augmented_dir)
    experiments_dir = Path(experiments_dir)
    v2_root = _infer_v2_root(experiments_dir)
    output_yaml_dir = Path(output_yaml_dir) if output_yaml_dir else v2_root
    normalized_class_names = _normalize_class_names(class_names)

    _validate_original_splits(splits_dir)
    _prepare_experiments_dir(experiments_dir, overwrite=overwrite)

    report_rows: list[dict[str, str]] = []
    offline_pairs = collect_yolo_pairs(augmented_dir / "images", augmented_dir / "labels")

    if not offline_pairs:
        # C and D are still built from original train data, but the warning
        # makes it clear that the offline-augmentation factor is absent.
        for experiment in ("exp_C_offline_aug", "exp_D_full_pipeline"):
            report_rows.append(
                _copy_report_row(
                    experiment=experiment,
                    split="train",
                    source_type="offline_augmented",
                    status="warning",
                    notes="no offline augmented pairs found; train contains original split only",
                )
            )

    for experiment, settings in EXPERIMENTS.items():
        experiment_dir = experiments_dir / experiment
        for split in SPLITS:
            _ensure_yolo_split_dirs(experiment_dir / split)

        for split in SPLITS:
            report_rows.extend(
                copy_yolo_pairs(
                    pairs=collect_yolo_pairs(
                        splits_dir / split / "images",
                        splits_dir / split / "labels",
                    ),
                    destination_split_dir=experiment_dir / split,
                    experiment=experiment,
                    split=split,
                    source_type=f"original_{split}",
                )
            )

        if settings["include_offline_augmented"] and offline_pairs:
            report_rows.extend(
                copy_yolo_pairs(
                    pairs=offline_pairs,
                    destination_split_dir=experiment_dir / "train",
                    experiment=experiment,
                    split="train",
                    source_type="offline_augmented",
                )
            )

        write_yolo_dataset_yaml(
            yaml_dir=output_yaml_dir,
            experiment_name=experiment,
            experiments_dir=experiments_dir,
            class_names=normalized_class_names,
        )

    copy_report = pd.DataFrame(report_rows, columns=COPY_REPORT_COLUMNS)
    summary = summarize_experiments(experiments_dir, normalized_class_names)
    warnings = verify_experiment_integrity(
        experiments_dir=experiments_dir,
        class_names=normalized_class_names,
        copy_report=copy_report,
    )
    return copy_report, summary, warnings


def collect_yolo_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Return sorted image-label pairs with matching stems."""
    image_paths = _collect_image_paths(images_dir)
    label_by_stem = {path.stem: path for path in _collect_label_paths(labels_dir)}
    pairs = [
        (image_path, label_by_stem[image_path.stem])
        for image_path in image_paths
        if image_path.stem in label_by_stem
    ]
    return sorted(pairs, key=lambda pair: pair[0].name)


def copy_yolo_pairs(
    pairs: list[tuple[Path, Path]],
    destination_split_dir: Path,
    experiment: str,
    split: str,
    source_type: str,
) -> list[dict[str, str]]:
    """Copy YOLO image-label pairs into one destination split.

    Filename conflicts are resolved with a deterministic ``_dupNNN`` suffix.
    This protects original train filenames when offline augmented files happen
    to share a name with an existing image.
    """
    destination_images_dir = Path(destination_split_dir) / "images"
    destination_labels_dir = Path(destination_split_dir) / "labels"
    destination_images_dir.mkdir(parents=True, exist_ok=True)
    destination_labels_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for image_path, label_path in pairs:
        copied_image_path = destination_images_dir / image_path.name
        copied_label_path = destination_labels_dir / label_path.name
        notes = ""

        if copied_image_path.exists() or copied_label_path.exists():
            copied_image_path, copied_label_path = _safe_conflict_paths(
                destination_images_dir=destination_images_dir,
                destination_labels_dir=destination_labels_dir,
                image_name=image_path.name,
            )
            notes = "filename_conflict resolved with safe suffix"

        shutil.copy2(image_path, copied_image_path)
        shutil.copy2(label_path, copied_label_path)
        rows.append(
            _copy_report_row(
                experiment=experiment,
                split=split,
                source_type=source_type,
                original_image_path=str(image_path),
                original_label_path=str(label_path),
                copied_image_path=str(copied_image_path),
                copied_label_path=str(copied_label_path),
                status="copied",
                notes=notes,
            )
        )
    return rows


def write_yolo_dataset_yaml(
    yaml_dir: Path,
    experiment_name: str,
    experiments_dir: Path,
    class_names: dict[int, str],
) -> Path:
    """Write one Ultralytics dataset YAML for an experiment."""
    yaml_dir = Path(yaml_dir)
    yaml_dir.mkdir(parents=True, exist_ok=True)
    v2_root = _infer_v2_root(experiments_dir)
    experiment_dir = Path(experiments_dir) / experiment_name
    relative_experiment_dir = experiment_dir.resolve().relative_to(v2_root.resolve())
    payload: dict[str, Any] = {
        "path": relative_experiment_dir.as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": _normalize_class_names(class_names),
    }
    yaml_path = yaml_dir / f"data_{experiment_name}.yaml"
    with yaml_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(payload, file_handle, sort_keys=False)
    return yaml_path


def count_yolo_split(split_dir: Path, class_names: dict[int, str]) -> dict[str, int]:
    """Count image, label, empty-label, and object totals for one split."""
    split_dir = Path(split_dir)
    image_paths = _collect_image_paths(split_dir / "images")
    label_paths = _collect_label_paths(split_dir / "labels")
    class_counts = {class_id: 0 for class_id in _normalize_class_names(class_names)}
    num_no_object_images = 0

    for label_path in label_paths:
        class_ids = _read_label_class_ids(label_path)
        if not class_ids:
            num_no_object_images += 1
        for class_id in class_ids:
            if class_id in class_counts:
                class_counts[class_id] += 1

    return {
        "num_images": len(image_paths),
        "num_labels": len(label_paths),
        "num_no_object_images": num_no_object_images,
        "num_objects": sum(class_counts.values()),
        "num_person": class_counts.get(0, 0),
        "num_helmet": class_counts.get(1, 0),
        "num_vest": class_counts.get(2, 0),
        "num_cleaning_coverall": class_counts.get(3, 0),
    }


def summarize_experiments(
    experiments_dir: Path,
    class_names: dict[int, str],
) -> pd.DataFrame:
    """Build a compact summary for every experiment split."""
    rows: list[dict[str, Any]] = []
    for experiment, settings in EXPERIMENTS.items():
        experiment_dir = Path(experiments_dir) / experiment
        for split in SPLITS:
            rows.append(
                {
                    "experiment": experiment,
                    "split": split,
                    **count_yolo_split(experiment_dir / split, class_names),
                    "notes": settings["notes"] if split == "train" else "fixed split",
                }
            )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def verify_experiment_integrity(
    experiments_dir: Path,
    class_names: dict[int, str],
    copy_report: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return integrity warnings for the generated ablation datasets."""
    normalized_class_names = _normalize_class_names(class_names)
    rows: list[dict[str, str]] = []
    reference_val_signature: str | None = None
    reference_test_signature: str | None = None

    if copy_report is not None and not copy_report.empty:
        conflict_rows = copy_report[
            copy_report["notes"].fillna("").str.contains("filename_conflict")
        ]
        for _, row in conflict_rows.iterrows():
            rows.append(
                _warning_row(
                    experiment=row["experiment"],
                    split=row["split"],
                    warning_type="filename_conflict",
                    details=str(row["copied_image_path"]),
                )
            )

    for experiment in EXPERIMENTS:
        experiment_dir = Path(experiments_dir) / experiment
        for split in SPLITS:
            split_dir = experiment_dir / split
            image_paths = _collect_image_paths(split_dir / "images")
            label_paths = _collect_label_paths(split_dir / "labels")
            image_stems = {path.stem for path in image_paths}
            label_stems = {path.stem for path in label_paths}

            if not image_paths and not label_paths:
                rows.append(_warning_row(experiment, split, "empty_split", str(split_dir)))
            if len(image_paths) != len(label_paths):
                rows.append(
                    _warning_row(
                        experiment,
                        split,
                        "image_label_count_mismatch",
                        f"{len(image_paths)} images vs {len(label_paths)} labels",
                    )
                )
            for stem in sorted(image_stems - label_stems):
                rows.append(_warning_row(experiment, split, "missing_label", stem))
            for stem in sorted(label_stems - image_stems):
                rows.append(_warning_row(experiment, split, "missing_image", stem))

            class_counts = count_yolo_split(split_dir, normalized_class_names)
            for class_id, class_name in normalized_class_names.items():
                if _class_count_from_summary(class_counts, class_id) == 0:
                    rows.append(
                        _warning_row(
                            experiment,
                            split,
                            "class_missing_from_split",
                            f"{class_id}: {class_name}",
                        )
                    )

            if split == "val":
                signature = _split_signature(split_dir)
                if reference_val_signature is None:
                    reference_val_signature = signature
                elif signature != reference_val_signature:
                    rows.append(
                        _warning_row(
                            experiment,
                            split,
                            "val_set_differs_across_experiments",
                            str(split_dir),
                        )
                    )
            if split == "test":
                signature = _split_signature(split_dir)
                if reference_test_signature is None:
                    reference_test_signature = signature
                elif signature != reference_test_signature:
                    rows.append(
                        _warning_row(
                            experiment,
                            split,
                            "test_set_differs_across_experiments",
                            str(split_dir),
                        )
                    )

    return pd.DataFrame(rows, columns=WARNING_COLUMNS)


def build_class_distribution(
    experiments_dir: Path,
    class_names: dict[int, str],
) -> pd.DataFrame:
    """Return object counts by class for callers that need detailed analysis.

    Notebook 05 intentionally does not save this as a separate artifact anymore;
    the compact summary already contains the class counts needed for a quick
    ablation dataset sanity check.
    """
    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        for split in SPLITS:
            counts = count_yolo_split(Path(experiments_dir) / experiment / split, class_names)
            for class_id, class_name in _normalize_class_names(class_names).items():
                rows.append(
                    {
                        "experiment": experiment,
                        "split": split,
                        "class_id": class_id,
                        "class_name": class_name,
                        "object_count": _class_count_from_summary(counts, class_id),
                    }
                )
    return pd.DataFrame(rows)


def _validate_original_splits(splits_dir: Path) -> None:
    for split in SPLITS:
        images_dir = Path(splits_dir) / split / "images"
        labels_dir = Path(splits_dir) / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError(f"Missing required split folders: {images_dir} / {labels_dir}")
        if not collect_yolo_pairs(images_dir, labels_dir):
            raise ValueError(f"Split has no matched YOLO image-label pairs: {images_dir.parent}")


def _prepare_experiments_dir(experiments_dir: Path, overwrite: bool) -> None:
    occupied = [
        experiment
        for experiment in EXPERIMENTS
        if _experiment_has_files(Path(experiments_dir) / experiment)
    ]
    if occupied and not overwrite:
        raise FileExistsError(
            "Experiment folder(s) already contain files: "
            f"{', '.join(occupied)}. Set overwrite=True to regenerate."
        )
    if overwrite:
        for experiment in EXPERIMENTS:
            _clear_files(Path(experiments_dir) / experiment)


def _collect_image_paths(images_dir: Path) -> list[Path]:
    if not Path(images_dir).exists():
        return []
    return [
        path
        for path in sorted(Path(images_dir).iterdir())
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]


def _collect_label_paths(labels_dir: Path) -> list[Path]:
    if not Path(labels_dir).exists():
        return []
    return [
        path
        for path in sorted(Path(labels_dir).iterdir())
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() == ".txt"
    ]


def _read_label_class_ids(label_path: Path) -> list[int]:
    class_ids: list[int] = []
    try:
        lines = Path(label_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return class_ids

    for line in lines:
        values = line.split()
        if not values:
            continue
        try:
            # YOLO class IDs are integers, but float conversion tolerates labels
            # that were accidentally serialized as "3.0".
            class_ids.append(int(float(values[0])))
        except ValueError:
            continue
    return class_ids


def _ensure_yolo_split_dirs(split_dir: Path) -> None:
    (Path(split_dir) / "images").mkdir(parents=True, exist_ok=True)
    (Path(split_dir) / "labels").mkdir(parents=True, exist_ok=True)


def _safe_conflict_paths(
    destination_images_dir: Path,
    destination_labels_dir: Path,
    image_name: str,
) -> tuple[Path, Path]:
    image_path = Path(image_name)
    for suffix_index in range(1, 10_000):
        candidate_stem = f"{image_path.stem}_dup{suffix_index:03d}"
        candidate_image = destination_images_dir / f"{candidate_stem}{image_path.suffix}"
        candidate_label = destination_labels_dir / f"{candidate_stem}.txt"
        if not candidate_image.exists() and not candidate_label.exists():
            return candidate_image, candidate_label
    raise FileExistsError(f"Could not resolve filename conflict for {image_name}")


def _experiment_has_files(experiment_dir: Path) -> bool:
    return Path(experiment_dir).exists() and any(
        path.is_file() for path in Path(experiment_dir).rglob("*")
    )


def _clear_files(experiment_dir: Path) -> None:
    if not Path(experiment_dir).exists():
        return
    for path in Path(experiment_dir).rglob("*"):
        if path.is_file():
            path.unlink()


def _split_signature(split_dir: Path) -> str:
    digest = hashlib.sha256()
    for folder_name in ("images", "labels"):
        folder = Path(split_dir) / folder_name
        for path in sorted(folder.iterdir()) if folder.exists() else []:
            if not path.is_file():
                continue
            digest.update(folder_name.encode("utf-8"))
            digest.update(path.name.encode("utf-8"))
            digest.update(_file_hash(path).encode("utf-8"))
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_v2_root(path: Path) -> Path:
    resolved = Path(path).resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name == "v2_pipeline":
            return candidate
    raise RuntimeError(f"Could not infer v2_pipeline root from {path}")


def _normalize_class_names(class_names: dict[int, str]) -> dict[int, str]:
    return {int(class_id): str(class_name) for class_id, class_name in class_names.items()}


def _class_count_from_summary(counts: dict[str, int], class_id: int) -> int:
    column_by_class = {
        0: "num_person",
        1: "num_helmet",
        2: "num_vest",
        3: "num_cleaning_coverall",
    }
    return counts.get(column_by_class.get(class_id, ""), 0)


def _copy_report_row(
    experiment: str,
    split: str,
    source_type: str,
    status: str,
    notes: str,
    original_image_path: str = "",
    original_label_path: str = "",
    copied_image_path: str = "",
    copied_label_path: str = "",
) -> dict[str, str]:
    return {
        "experiment": experiment,
        "split": split,
        "source_type": source_type,
        "original_image_path": original_image_path,
        "original_label_path": original_label_path,
        "copied_image_path": copied_image_path,
        "copied_label_path": copied_label_path,
        "status": status,
        "notes": notes,
    }


def _warning_row(
    experiment: str,
    split: str,
    warning_type: str,
    details: str,
) -> dict[str, str]:
    return {
        "experiment": experiment,
        "split": split,
        "warning_type": warning_type,
        "details": details,
    }
