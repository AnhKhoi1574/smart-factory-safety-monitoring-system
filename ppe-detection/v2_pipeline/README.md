# v2 Pipeline

This is the clean, reproducible PPE detection pipeline for the next development cycle. It is notebook-first for collaboration, with reusable helper modules under `src/`.

## Goals

- validate teammate dataset drops,
- merge valid samples into a canonical original dataset,
- split before augmentation,
- generate offline augmented training-only variants,
- build experiment-ready datasets for ablations,
- train and compare candidate YOLO architectures,
- run the ablation study in a controlled order,
- train and evaluate the final model exactly once on the untouched test set,
- demonstrate inference plus tracking and PPE association logic.

## Core Rules

1. Every teammate source folder must contain `images/` and `labels/`.
2. Every image must have a matching YOLO `.txt` label file.
3. The split is created before any augmentation.
4. Offline augmentation is applied only to training images.
5. By default, `data/test_sources/` provides the held-out test set.
6. With an external test set, split `master_original` into train/val only.
7. The final test split stays untouched until final evaluation.

## Main Areas

- `configs/`: class map, dataset settings, augmentation settings, and training settings.
- `data/`: local-only data workspace with `.gitkeep` placeholders.
- `notebooks/`: ordered execution flow for the full pipeline.
- `src/`: reusable Python helpers shared by the notebooks.
- `docs/`: concise operating notes for the workflow.
- `runs/` and `weights/`: local artifact targets, kept out of Git.

## Notebook Order

1. `01_validate_and_merge_dataset.ipynb`
2. `02_dataset_eda.ipynb`
3. `03_split_dataset.ipynb`
4. `04_offline_augmentation.ipynb`
5. `05_build_ablation_datasets.ipynb`
6. `06_candidate_model_training.ipynb`
7. `07_ablation_study.ipynb`
8. `08_final_training_and_evaluation.ipynb`
9. `09_inference_tracking_demo.ipynb`
