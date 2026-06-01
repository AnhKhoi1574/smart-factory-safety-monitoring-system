"""Build experiment-ready dataset variants for ablation studies."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
OFFLINE_AUGMENTATION_SOURCES = ("ir", "sunlight", "blur_compression")
EXPERIMENTS = {
    "exp_A_original_only": {
        "include_offline_augmented": False,
        "notes": "original train only; online augmentation off during training",
    },
    "exp_B_online_aug": {
        "include_offline_augmented": False,
        "notes": "original train only; online augmentation on during training",
    },
    "exp_C_offline_aug": {
        "include_offline_augmented": True,
        "notes": "original plus offline augmented train; online augmentation off/minimal",
    },
    "exp_D_full_pipeline": {
        "include_offline_augmented": True,
        "notes": "original plus offline augmented train; online augmentation on",
    },
}
REPORT_COLUMNS = [
    "experiment",
    "split",
    "source_type",
    "original_path",
    "copied_path",
    "status",
    "notes",
]
SUMMARY_COLUMNS = [
    "experiment",
    "split",
    "num_images",
    "num_labels",
    "num_objects",
    "num_person",
    "num_helmet",
    "num_vest",
    "notes",
]
CLASS_DISTRIBUTION_COLUMNS = [
    "experiment",
    "split",
    "class_id",
    "class_name",
    "object_count",
]
WARNING_COLUMNS = ["experiment", "split", "warning_type", "details"]


def build_ablation_datasets(
    splits_original_dir: Path,
    augmented_train_dir: Path,
    experiments_dir: Path,
    class_names: dict[int, str],
    overwrite: bool = False,
) -> pd.DataFrame:
    """Create YOLO folders for the four ablation experiments.

    Args:
        splits_original_dir: Existing original ``train``, ``val``, and ``test``
            split folders produced by Notebook 03.
        augmented_train_dir: Offline augmentation root produced by Notebook 04.
        experiments_dir: Destination root for generated experiment datasets.
        class_names: Mapping from class ID to class name.
        overwrite: If ``True``, clear existing experiment files first.

    Returns:
        A row-level copy report for all copied, skipped, or warning records.

    Raises:
        FileNotFoundError: If required original split folders are missing.
        ValueError: If required original splits are empty or mismatched.
        FileExistsError: If destination folders already contain files and
            ``overwrite`` is ``False``.
    """
    splits_original_dir = Path(splits_original_dir)
    augmented_train_dir = Path(augmented_train_dir)
    experiments_dir = Path(experiments_dir)
    normalized_class_names = _normalize_class_names(class_names)

    _validate_original_splits(splits_original_dir)
    _prepare_experiments_dir(experiments_dir, overwrite=overwrite)

    report_rows: list[dict[str, str]] = []
    offline_pairs = _collect_offline_augmented_pairs(augmented_train_dir)
    if not offline_pairs:
        for experiment in ("exp_C_offline_aug", "exp_D_full_pipeline"):
            report_rows.append(
                _report_row(
                    experiment=experiment,
                    split="train",
                    source_type="offline_augmented",
                    original_path="",
                    copied_path="",
                    status="warning",
                    notes=(
                        "offline augmented data missing; experiment contains "
                        "original train samples only"
                    ),
                )
            )

    for experiment, settings in EXPERIMENTS.items():
        experiment_dir = experiments_dir / experiment
        for split in SPLITS:
            _ensure_yolo_split_dirs(experiment_dir, split)

        for split in SPLITS:
            report_rows.extend(
                copy_yolo_split(
                    source_split_dir=splits_original_dir / split,
                    destination_split_dir=experiment_dir / split,
                    experiment=experiment,
                    split=split,
                    source_type="original",
                )
            )

        if settings["include_offline_augmented"] and offline_pairs:
            report_rows.extend(
                copy_augmented_training_data(
                    augmented_pairs=offline_pairs,
                    destination_train_dir=experiment_dir / "train",
                    experiment=experiment,
                )
            )

        write_dataset_yaml(
            v2_root=_infer_v2_root(experiments_dir),
            experiment_name=experiment,
            class_names=normalized_class_names,
        )

    return pd.DataFrame(report_rows, columns=REPORT_COLUMNS)


def build_experiment_sets(base_dir: Path, experiments_dir: Path) -> dict[str, Path]:
    """Backward-compatible wrapper returning expected experiment paths."""
    base_dir = Path(base_dir)
    experiments_dir = Path(experiments_dir)
    return {
        experiment: experiments_dir / experiment
        for experiment in EXPERIMENTS
    }


def copy_yolo_split(
    source_split_dir: Path,
    destination_split_dir: Path,
    experiment: str,
    split: str,
    source_type: str,
) -> list[dict[str, str]]:
    """Copy one YOLO split while preserving original filenames."""
    source_images_dir = Path(source_split_dir) / "images"
    source_labels_dir = Path(source_split_dir) / "labels"
    destination_images_dir = Path(destination_split_dir) / "images"
    destination_labels_dir = Path(destination_split_dir) / "labels"
    destination_images_dir.mkdir(parents=True, exist_ok=True)
    destination_labels_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for image_path, label_path in _find_yolo_pairs(source_images_dir, source_labels_dir):
        copied_image_path = destination_images_dir / image_path.name
        copied_label_path = destination_labels_dir / label_path.name
        shutil.copy2(image_path, copied_image_path)
        shutil.copy2(label_path, copied_label_path)
        rows.extend(
            [
                _report_row(
                    experiment=experiment,
                    split=split,
                    source_type=source_type,
                    original_path=str(image_path),
                    copied_path=str(copied_image_path),
                    status="copied",
                    notes="",
                ),
                _report_row(
                    experiment=experiment,
                    split=split,
                    source_type=source_type,
                    original_path=str(label_path),
                    copied_path=str(copied_label_path),
                    status="copied",
                    notes="",
                ),
            ]
        )
    return rows


def copy_augmented_training_data(
    augmented_pairs: list[tuple[Path, Path]],
    destination_train_dir: Path,
    experiment: str,
) -> list[dict[str, str]]:
    """Copy offline augmented pairs into an experiment train split."""
    destination_images_dir = Path(destination_train_dir) / "images"
    destination_labels_dir = Path(destination_train_dir) / "labels"
    destination_images_dir.mkdir(parents=True, exist_ok=True)
    destination_labels_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for image_path, label_path in augmented_pairs:
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
        rows.extend(
            [
                _report_row(
                    experiment=experiment,
                    split="train",
                    source_type="offline_augmented",
                    original_path=str(image_path),
                    copied_path=str(copied_image_path),
                    status="copied",
                    notes=notes,
                ),
                _report_row(
                    experiment=experiment,
                    split="train",
                    source_type="offline_augmented",
                    original_path=str(label_path),
                    copied_path=str(copied_label_path),
                    status="copied",
                    notes=notes,
                ),
            ]
        )
    return rows


def write_dataset_yaml(
    v2_root: Path,
    experiment_name: str,
    class_names: dict[int, str],
) -> Path:
    """Write one Ultralytics dataset YAML file relative to ``v2_pipeline``."""
    v2_root = Path(v2_root)
    payload: dict[str, Any] = {
        "path": f"data/experiments/{experiment_name}",
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": _normalize_class_names(class_names),
    }
    yaml_path = v2_root / f"data_{experiment_name}.yaml"
    with yaml_path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(payload, file_handle, sort_keys=False)
    return yaml_path


def count_yolo_split(
    split_dir: Path,
    class_names: dict[int, str],
) -> dict[str, int]:
    """Count images, labels, and class objects for one YOLO split."""
    split_dir = Path(split_dir)
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    class_counts = {class_id: 0 for class_id in _normalize_class_names(class_names)}

    label_paths = _collect_label_paths(labels_dir)
    for label_path in label_paths:
        for class_id in _read_label_class_ids(label_path):
            if class_id in class_counts:
                class_counts[class_id] += 1

    return {
        "num_images": len(_collect_image_paths(images_dir)),
        "num_labels": len(label_paths),
        "num_objects": sum(class_counts.values()),
        "num_person": class_counts.get(0, 0),
        "num_helmet": class_counts.get(1, 0),
        "num_vest": class_counts.get(2, 0),
    }


def summarize_experiments(
    experiments_dir: Path,
    class_names: dict[int, str],
) -> pd.DataFrame:
    """Build the ablation dataset summary table."""
    rows: list[dict[str, Any]] = []
    for experiment, settings in EXPERIMENTS.items():
        experiment_dir = Path(experiments_dir) / experiment
        for split in SPLITS:
            counts = count_yolo_split(experiment_dir / split, class_names)
            rows.append(
                {
                    "experiment": experiment,
                    "split": split,
                    **counts,
                    "notes": settings["notes"] if split == "train" else "fixed split",
                }
            )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def build_class_distribution(
    experiments_dir: Path,
    class_names: dict[int, str],
) -> pd.DataFrame:
    """Count YOLO objects by class for every experiment split."""
    normalized_class_names = _normalize_class_names(class_names)
    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        for split in SPLITS:
            labels_dir = Path(experiments_dir) / experiment / split / "labels"
            class_counts = {class_id: 0 for class_id in normalized_class_names}
            for label_path in _collect_label_paths(labels_dir):
                for class_id in _read_label_class_ids(label_path):
                    if class_id in class_counts:
                        class_counts[class_id] += 1
            for class_id, class_name in normalized_class_names.items():
                rows.append(
                    {
                        "experiment": experiment,
                        "split": split,
                        "class_id": class_id,
                        "class_name": class_name,
                        "object_count": class_counts[class_id],
                    }
                )
    return pd.DataFrame(rows, columns=CLASS_DISTRIBUTION_COLUMNS)


def verify_experiment_integrity(
    experiments_dir: Path,
    class_names: dict[int, str],
    copy_report: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return integrity warnings for generated ablation datasets."""
    normalized_class_names = _normalize_class_names(class_names)
    rows: list[dict[str, str]] = []
    reference_val_signature: str | None = None
    reference_test_signature: str | None = None

    if copy_report is not None and not copy_report.empty:
        conflicts = copy_report[
            copy_report["notes"].fillna("").str.contains("filename_conflict")
        ]
        for _, row in conflicts.iterrows():
            rows.append(
                _warning_row(
                    row["experiment"],
                    row["split"],
                    "filename_conflict",
                    f"Resolved conflict for {row['copied_path']}",
                )
            )

    for experiment in EXPERIMENTS:
        experiment_dir = Path(experiments_dir) / experiment
        for split in SPLITS:
            split_dir = experiment_dir / split
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"
            image_paths = _collect_image_paths(images_dir)
            label_paths = _collect_label_paths(labels_dir)
            image_stems = {path.stem for path in image_paths}
            label_stems = {path.stem for path in label_paths}

            if not image_paths and not label_paths:
                rows.append(
                    _warning_row(experiment, split, "empty_split", str(split_dir))
                )
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
                rows.append(
                    _warning_row(experiment, split, "missing_label", stem)
                )
            for stem in sorted(label_stems - image_stems):
                rows.append(
                    _warning_row(experiment, split, "missing_image", stem)
                )

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


def _validate_original_splits(splits_original_dir: Path) -> None:
    for split in SPLITS:
        split_dir = Path(splits_original_dir) / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError(
                f"Required split folder missing: {split_dir}/images or labels"
            )
        pairs = _find_yolo_pairs(images_dir, labels_dir)
        if not pairs:
            raise ValueError(f"Required split is empty or mismatched: {split_dir}")


def _prepare_experiments_dir(experiments_dir: Path, overwrite: bool) -> None:
    experiments_dir = Path(experiments_dir)
    occupied_experiments = [
        experiment
        for experiment in EXPERIMENTS
        if _experiment_has_files(experiments_dir / experiment)
    ]
    if occupied_experiments and not overwrite:
        raise FileExistsError(
            "Experiment folder(s) already contain files: "
            f"{', '.join(occupied_experiments)}. "
            "Set overwrite=True to regenerate experiment datasets."
        )
    if overwrite:
        for experiment in EXPERIMENTS:
            _clear_files(experiments_dir / experiment)


def _collect_offline_augmented_pairs(
    augmented_train_dir: Path,
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for source_name in OFFLINE_AUGMENTATION_SOURCES:
        source_dir = Path(augmented_train_dir) / source_name
        images_dir = source_dir / "images"
        labels_dir = source_dir / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue
        pairs.extend(_find_yolo_pairs(images_dir, labels_dir))
    return sorted(pairs, key=lambda item: (item[0].parent.parent.name, item[0].name))


def _find_yolo_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    image_paths = _collect_image_paths(images_dir)
    label_paths = {path.stem: path for path in _collect_label_paths(labels_dir)}
    pairs: list[tuple[Path, Path]] = []
    for image_path in image_paths:
        label_path = label_paths.get(image_path.stem)
        if label_path is not None:
            pairs.append((image_path, label_path))
    return pairs


def _collect_image_paths(images_dir: Path) -> list[Path]:
    if not Path(images_dir).exists():
        return []
    return [
        path
        for path in sorted(Path(images_dir).iterdir())
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]


def _collect_label_paths(labels_dir: Path) -> list[Path]:
    if not Path(labels_dir).exists():
        return []
    return [
        path
        for path in sorted(Path(labels_dir).iterdir())
        if path.is_file() and path.suffix.lower() == ".txt"
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
            class_ids.append(int(float(values[0])))
        except ValueError:
            continue
    return class_ids


def _ensure_yolo_split_dirs(experiment_dir: Path, split: str) -> None:
    (Path(experiment_dir) / split / "images").mkdir(parents=True, exist_ok=True)
    (Path(experiment_dir) / split / "labels").mkdir(parents=True, exist_ok=True)


def _safe_conflict_paths(
    destination_images_dir: Path,
    destination_labels_dir: Path,
    image_name: str,
) -> tuple[Path, Path]:
    image_path = Path(image_name)
    image_stem = image_path.stem
    image_suffix = image_path.suffix
    for suffix_index in range(2, 10_000):
        candidate_stem = f"{image_stem}_{suffix_index}"
        candidate_image = destination_images_dir / f"{candidate_stem}{image_suffix}"
        candidate_label = destination_labels_dir / f"{candidate_stem}.txt"
        if not candidate_image.exists() and not candidate_label.exists():
            return candidate_image, candidate_label
    raise FileExistsError(f"Could not resolve filename conflict for {image_name}")


def _experiment_has_files(experiment_dir: Path) -> bool:
    if not Path(experiment_dir).exists():
        return False
    return any(path.is_file() for path in Path(experiment_dir).rglob("*"))


def _clear_files(experiment_dir: Path) -> None:
    if not Path(experiment_dir).exists():
        return
    for path in Path(experiment_dir).rglob("*"):
        if path.is_file():
            path.unlink()


def _split_signature(split_dir: Path) -> str:
    digest = hashlib.sha256()
    for child_name in ("images", "labels"):
        child_dir = Path(split_dir) / child_name
        for path in sorted(child_dir.iterdir()) if child_dir.exists() else []:
            if not path.is_file():
                continue
            digest.update(child_name.encode("utf-8"))
            digest.update(path.name.encode("utf-8"))
            digest.update(_file_hash(path).encode("utf-8"))
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_v2_root(experiments_dir: Path) -> Path:
    experiments_dir = Path(experiments_dir).resolve()
    if experiments_dir.name == "experiments" and experiments_dir.parent.name == "data":
        return experiments_dir.parent.parent
    for candidate in (experiments_dir, *experiments_dir.parents):
        if candidate.name == "v2_pipeline":
            return candidate
    raise RuntimeError(f"Could not infer v2_pipeline root from {experiments_dir}")


def _normalize_class_names(class_names: dict[int, str]) -> dict[int, str]:
    return {int(class_id): str(class_name) for class_id, class_name in class_names.items()}


def _class_count_from_summary(counts: dict[str, int], class_id: int) -> int:
    if class_id == 0:
        return counts.get("num_person", 0)
    if class_id == 1:
        return counts.get("num_helmet", 0)
    if class_id == 2:
        return counts.get("num_vest", 0)
    return 0


def _report_row(
    experiment: str,
    split: str,
    source_type: str,
    original_path: str,
    copied_path: str,
    status: str,
    notes: str,
) -> dict[str, str]:
    return {
        "experiment": experiment,
        "split": split,
        "source_type": source_type,
        "original_path": original_path,
        "copied_path": copied_path,
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
