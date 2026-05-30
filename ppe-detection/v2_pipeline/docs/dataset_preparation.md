# Dataset Preparation

The v2 intake contract expects each teammate to deliver a folder with:

- `images/`
- `labels/`

Each image must have a matching YOLO `.txt` file with normalized coordinates and class IDs in the three-class PPE schema.

## Validation and Merge Flow

1. Drop teammate folders into `data/raw_sources/` locally.
2. Run `01_validate_and_merge_dataset.ipynb`.
3. Validate folder shape, image-label pairing, class IDs, and YOLO row format.
4. Merge only valid samples into `data/master_original/`.
5. Record findings under `reports/validation/`.

The merge stage should use deterministic renaming so samples from different teammates can coexist without collisions.
