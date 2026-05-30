# Training Pipeline

The training workflow is designed to keep experiments comparable and to protect the final test set from leakage.

## Sequence

1. Validate and merge raw teammate inputs.
2. Run EDA on the merged original dataset.
3. Split into train, val, and test.
4. Generate offline augmentation from the training split only.
5. Build ablation dataset variants.
6. Train candidate YOLO architectures with shared settings.
7. Select the best architecture.
8. Run augmentation ablations on that architecture.
9. Train the final model.
10. Evaluate once on the untouched test split.

## Candidate Model Triage

Candidate architectures are compared using the priority order in `configs/training_config.yaml`:

1. recall
2. map50
3. latency
4. model_size
