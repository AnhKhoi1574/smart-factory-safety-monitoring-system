# Smart Factory Safety Monitoring System

This repository contains model-training workstreams for a smart factory safety
monitoring system. The active PPE workstream trains a YOLO object detector for
fixed elevated factory CCTV views.

The active PPE v2 detector uses four object classes:

- `0`: person
- `1`: helmet
- `2`: vest
- `3`: cleaning_coverall

The detector does not predict violation classes directly. It emits objects that
the backend can associate over time, then role/PPE logic decides whether a
person is compliant, missing a helmet, missing a role uniform, or wearing a
cleaning coverall.

## Repository Layout

- `environment.yml`: Anaconda environment definition for notebook work.
- `requirements.txt`: pip-oriented dependency mirror for managed environments.
- `ANACONDA_SETUP_GUIDE.md`: setup notes, including CUDA-enabled PyTorch.
- `show_structure.py`: lightweight tree printer for repository checks.
- `ppe-detection/`: PPE detection module with legacy work and the active v2
  pipeline.
- `sign-detection/`: factory safety sign detection pipeline.

## PPE v2 Data Contract

The active PPE pipeline is:

```text
ppe-detection/v2_pipeline/
```

Input data is placed locally under:

```text
ppe-detection/v2_pipeline/data/input/open_source/images/
ppe-detection/v2_pipeline/data/input/open_source/labels/

ppe-detection/v2_pipeline/data/input/factory_source/images/
ppe-detection/v2_pipeline/data/input/factory_source/labels/

ppe-detection/v2_pipeline/data/input/test_source/images/
ppe-detection/v2_pipeline/data/input/test_source/labels/
```

Source policy:

- `open_source`: external/public data used for training only.
- `factory_source`: target-domain factory CCTV data used for train/validation.
- `test_source`: final untouched test data used only for final evaluation.

Generated folders are created by notebooks or helper code when needed under:

```text
ppe-detection/v2_pipeline/data/generated/
```

Do not use the old `raw_sources`, `master_original`, or `open_source_train`
workflow for new PPE v2 work.

## Recommended PPE Workflow

1. Create the conda environment from `environment.yml`.
2. If training on NVIDIA hardware, install CUDA-enabled PyTorch using
   [ANACONDA_SETUP_GUIDE.md](ANACONDA_SETUP_GUIDE.md).
3. Place YOLO image/label pairs into the PPE v2 `data/input` source lanes.
4. Run the notebooks in `ppe-detection/v2_pipeline/notebooks/` in order.
5. Keep validation and test data original and untouched after splitting.
6. Use validation metrics for model/ablation decisions; use the test set only
   once for final reporting.

## Data and Artifact Policy

Raw input data, generated splits, augmented datasets, experiment datasets,
reports, YOLO runs, model weights, and exported models are intentionally ignored
by Git.

Track source code, configs, docs, notebooks, and intentional `.gitkeep`
placeholders only.

## Main Entry Points

- PPE module overview: [ppe-detection/README.md](ppe-detection/README.md)
- PPE legacy area: [ppe-detection/v1_legacy/README.md](ppe-detection/v1_legacy/README.md)
- PPE v2 pipeline: [ppe-detection/v2_pipeline/README.md](ppe-detection/v2_pipeline/README.md)
- Sign detection module: [sign-detection/README.md](sign-detection/README.md)
