# PPE Detection Module

This module contains the PPE object-detection workstream for the Smart Factory
Safety Monitoring System.

The active development area is:

```text
ppe-detection/v2_pipeline/
```

The legacy area is preserved for traceability:

```text
ppe-detection/v1_legacy/
```

## Current v2 Goal

Train a lightweight YOLO detector for fixed top-down/elevated factory CCTV
views. The detector predicts objects only; backend logic handles role and
violation decisions after detection and tracking.

Fixed v2 class map:

- `0`: person
- `1`: helmet
- `2`: vest
- `3`: cleaning_coverall

`cleaning_coverall` separates cleaning staff uniforms from normal high-visibility
vests, so the backend can apply role-specific PPE rules.

## Active Data Layout

Use this input layout for new PPE v2 work:

```text
v2_pipeline/data/input/open_source/images/
v2_pipeline/data/input/open_source/labels/

v2_pipeline/data/input/factory_source/images/
v2_pipeline/data/input/factory_source/labels/

v2_pipeline/data/input/test_source/images/
v2_pipeline/data/input/test_source/labels/
```

Source policy:

- `open_source`: train-only external/public data.
- `factory_source`: target-domain CCTV data for train and validation.
- `test_source`: final untouched test data.

Generated outputs are created by the pipeline when needed under:

```text
v2_pipeline/data/generated/
```

Do not create new work using the old `raw_sources`, `master_original`,
`splits_original`, or `open_source_train` folders.

## Pipeline Shape

The v2 notebooks should follow this order:

1. Validate input source lanes.
2. Split open-source/factory/test sources into the new generated split layout.
3. Run EDA and split verification.
4. Generate offline augmentation from training data only.
5. Build ablation datasets.
6. Train candidate YOLO architectures.
7. Run ablation with the selected architecture.
8. Train the final model and evaluate once on the untouched test set.
9. Demonstrate inference/tracking and backend rule association.

## Module Areas

- `v1_legacy/`: archived notebooks, reports, local artifacts, and early training
  work. Do not modify for new v2 development unless explicitly requested.
- `v2_pipeline/configs/`: class names, dataset paths, augmentation settings, and
  training settings.
- `v2_pipeline/notebooks/`: ordered workflow notebooks.
- `v2_pipeline/src/`: reusable helpers used by notebooks.
- `v2_pipeline/docs/`: concise operating notes.

## Git Hygiene

Do not commit raw data, generated datasets, reports, YOLO runs, weights, or model
exports. Keep configs, docs, notebooks, source code, and intentional `.gitkeep`
files tracked.
