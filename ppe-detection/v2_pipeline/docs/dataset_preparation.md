# Dataset Preparation

The v2 intake contract expects each teammate to deliver a folder with:

- `images/`
- `labels/`

Each image must have a matching YOLO `.txt` file with normalized coordinates and class IDs in the three-class PPE schema.

## Validation and Merge Flow

1. Drop teammate folders into `data/raw_sources/` locally.
2. Run `01_validate_and_merge_dataset.ipynb`.
3. Validate folder shape, image-label pairing, class IDs, and YOLO row format.
4. Merge only valid samples into the merged dataset folder, `data/master_original/`.
5. Record findings under `reports/validation/`.

The merge stage should use deterministic renaming so samples from different teammates can coexist without collisions.

Merged original samples are renamed with one global sequence across all sources:

- `ppe_00001.jpg`
- `ppe_00001.txt`
- `ppe_00002.jpg`
- `ppe_00002.txt`

If you have a separate held-out test set, place it under `data/test_sources/` with the same `images/` and `labels/` structure. Notebook `01_validate_and_merge_dataset.ipynb` validates it separately and copies valid rows into `data/splits_original/test/` with names such as:

- `test_00001.jpg`
- `test_00001.txt`

The default config uses this external-test workflow:

- `use_external_test_set: true`
- `split.train: 0.80`
- `split.val: 0.20`
- `split.test: 0.00`

With that setting, notebook `03_split_dataset.ipynb` should split the merged dataset in `data/master_original/` into train and validation only. It must not reserve another 10% test split from the merged dataset, because the final test set already comes from `data/test_sources/`.

## Validation Report and Merge Outputs

Notebook `01_validate_and_merge_dataset.ipynb` produces CSV files under `reports/validation/`:

- `validation_report.csv`: one row per discovered sample or unmatched file, including status, errors, warnings, image dimensions, and class counts.
- `invalid_samples.csv`: rows from the validation report where `status != "valid"`.
- `dataset_summary.csv`: total valid, invalid, warning, duplicate-name, and object-count summaries.
- `merge_report.csv`: the original and renamed file paths for copied valid training/validation candidate samples.
- `test_validation_report.csv`: validation results for optional separate test sources.
- `test_invalid_samples.csv`: invalid rows from optional separate test sources.
- `test_merge_report.csv`: copied valid separate test rows and their `test_00001`-style filenames.

Cross-source duplicate base names are warnings because merged filenames use a global sequence. Empty labels, unreadable images, invalid YOLO rows, missing pairs, and duplicate image base names inside one source remain invalid.

This step only validates and merges original samples. Train/validation/test splitting and augmentation happen in later notebooks.

## Notebook 02 Dataset EDA

Notebook `02_dataset_eda.ipynb` analyzes the cleaned merged dataset under `data/master_original/` and, when present, the separate held-out test set under `data/splits_original/test/`.

This step does not modify images, labels, raw sources, test data, or split data. It only writes ignored CSV and PNG reports under `reports/eda/`.

The EDA checks class balance, image resolutions, object density, bounding box sizes, small-object patterns, and annotation review warnings such as people without helmet or vest boxes. These warnings are prompts for human review, not validation failures.

Use the EDA outputs to decide whether the cleaned originals are ready for Notebook `03_split_dataset.ipynb`. Splitting still happens after EDA, and augmentation still happens only after splitting.
