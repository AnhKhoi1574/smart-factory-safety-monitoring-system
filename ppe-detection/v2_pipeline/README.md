# v2 Pipeline

This is the clean, reproducible PPE detection pipeline for the Smart Factory
Safety Monitoring System.

The detector predicts four object classes:

```text
0 = person
1 = helmet
2 = vest
3 = cleaning_coverall
```

Backend violation logic is handled after detection. Do not train direct
violation classes.

## Current Data Contract

Input data is placed manually under:

```text
data/input/open_source/images/
data/input/open_source/labels/

data/input/factory_source/images/
data/input/factory_source/labels/

data/input/test_source/images/
data/input/test_source/labels/
```

Source policy:

- `open_source`: train only.
- `factory_source`: train and validation.
- `test_source`: final untouched test set.

Generated data belongs under:

```text
data/generated/splits/
data/generated/augmented/
data/generated/experiments/
```

The old `data/master_original`, `data/raw_sources`, and
`data/open_source_train` workflow is no longer the active v2 design.

## Core Rules

1. Validate input sources before splitting.
2. Split before any augmentation.
3. Keep validation and test original and untouched.
4. Use validation metrics for model and ablation decisions.
5. Use the test set only once for final reporting.
6. Keep raw input data, generated datasets, reports, runs, and weights out of
   Git.

## Notebook Order

1. `01_validate_and_merge_dataset.ipynb` - validates source lanes only.
2. `02_dataset_eda.ipynb`
3. `03_split_dataset.ipynb`
4. `04_offline_augmentation.ipynb`
5. `05_build_ablation_datasets.ipynb`
6. `06_candidate_model_training.ipynb`
7. `07_ablation_study.ipynb`
8. `08_final_training_and_evaluation.ipynb`
9. `09_inference_tracking_demo.ipynb`

Reusable code lives under `src/`; notebooks should stay as readable
orchestration.
