# Dataset Preparation

Purpose: document how prepared YOLO sign data is placed into `data/input/images` and `data/input/labels` before validation.

## Input Contract

Place the prepared YOLO dataset directly into:

```text
data/input/images/
data/input/labels/
```

Each image should have a matching `.txt` label file with the same base name. A label file may be empty; an empty label is a valid no-sign image and is counted separately for stratified splitting.

Notebook 01 validates and profiles this input dataset only. It does not split, augment, train, or modify input files.

## No-Sign Images

No-sign images are useful because they help the detector avoid false safety-sign activations. Keep them as empty `.txt` label files instead of deleting their labels.

## Validation Outputs

Notebook 01 writes:

```text
reports/validation/validation_report.csv
reports/profile/input_image_profile.csv
reports/profile/input_bbox_records.csv
reports/profile/input_summary.csv
```

Notebook 02 uses `reports/profile/input_image_profile.csv` to create a balanced train/val/test split by class presence and no-sign status.
