# v1 Legacy PPE Work

This folder preserves the first PPE detector development cycle exactly as a historical baseline.

## Contents

- `notebooks/`: the original EDA, preprocessing, and modeling notebooks.
- `reports/`: generated analysis artifacts from the legacy workflow.
- `results/`: candidate comparison outputs and visual verification figures.
- `runs/`: old YOLO training and validation outputs.
- `weights/`: downloaded pretrained checkpoints that were used during the legacy experiments.
- `preprocessing_report.md`: the original preprocessing write-up.

## Scope

This area is read-only from a workflow perspective. It is here so the team can:

- inspect what was done in the first iteration,
- compare future v2 results against the original experiments,
- keep old outputs without mixing them into the new pipeline.

## Notes

- The legacy dataset workspace now lives inside this archive at `ppe-detection/v1_legacy/data/`.
- The archived notebooks resolve data through relative paths such as `../data/CHV_dataset` and `../data/CHV_yolo`.
- New dataset preparation, training, ablations, and demos should be built in `../v2_pipeline/`.
