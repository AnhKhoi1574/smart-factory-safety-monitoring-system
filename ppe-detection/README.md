# PPE Detection Module

This module contains both the preserved first training cycle and the new reproducible YOLO pipeline for PPE compliance detection.

## Structure

- `v1_legacy/`: archived notebooks, reports, YOLO runs, result plots, candidate comparison CSVs, and downloaded pretrained checkpoints from the original training effort.
- `v2_pipeline/`: the new notebook-first workflow for teammate dataset intake, validation, merging, splitting, offline augmentation, training experiments, and inference/tracking demos.
- `v1_legacy/data/`: retained local legacy dataset workspace from the earlier phase. It remains a local artifact area and is not part of the new v2 pipeline contract.

## Class Convention

All v2 work uses the same three-class label map:

- `0`: person
- `1`: helmet
- `2`: vest

## Working Rule

New development should happen in `v2_pipeline/`. The `v1_legacy/` area is preserved for traceability and comparison, not for continued feature work.
