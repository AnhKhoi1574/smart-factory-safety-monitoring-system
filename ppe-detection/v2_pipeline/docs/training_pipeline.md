# Training Pipeline

The training workflow is designed to keep experiments comparable and to protect the final test set from leakage.

## Sequence

1. Validate the source-lane input folders.
2. Run EDA on the input source distribution.
3. Split into train, val, and test from `open_source`, `factory_source`, and `test_source`.
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
3. map50-95
4. FPS / latency
5. model_size

Notebook 06 performs architecture triage before the ablation study. Every
candidate model is trained on the same four-class
`data_exp_D_full_pipeline.yaml` dataset YAML with the same training and online
augmentation settings. The YAML must declare:

```text
0 = person
1 = helmet
2 = vest
3 = cleaning_coverall
```

This makes the comparison about model architecture rather than dataset
composition.

Candidate selection uses validation metrics only. The untouched test split is
not used for choosing the architecture, tuning augmentation settings, or ranking
candidate runs. After the best lightweight architecture is selected, Notebook 07
can run the ablation study on that locked architecture.

Candidate runs are saved under `runs/candidate_models/`, and compact CSV
reports are saved under `reports/training/`:

- `candidate_model_results.csv`
- `candidate_model_failures.csv`
- `candidate_model_ranking.csv`

Notebook 06 keeps Ultralytics plot generation off during candidate triage to
avoid creating many per-candidate PNG artifacts. The CSV ranking is the source
of truth for architecture selection.

On Windows/Jupyter, tune `batch`, `workers`, and `amp` conservatively. PyTorch
dataloader workers can survive a kernel crash as orphan
`--multiprocessing-fork` processes if the kernel dies. The training helpers set
OpenMP environment variables before loading Ultralytics to avoid the duplicate
`libomp.dll` / `libiomp5md.dll` runtime crash.

## Final Training and Test Evaluation

Notebook 08 performs the final training run after architecture triage and
ablation decisions are already locked. It uses the selected architecture from
Notebook 06 and the selected training configuration from Notebook 07 or
`configs/training_config.yaml`.

This is the first notebook allowed to evaluate on the untouched test split.
The resulting test metrics are for final reporting only; they should not be
used to switch architectures, rerun ablations, or tune augmentation settings.

Final runs are saved under `runs/final_model/`, reports under
`reports/training/`, and deployment exports under `weights/final/`. ONNX export
is supported by default, while optional formats such as TensorRT should be
treated as machine-dependent and may fail gracefully.
