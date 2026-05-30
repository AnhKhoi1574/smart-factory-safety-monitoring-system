# Smart Factory Safety Monitoring System

This repository contains the PPE detection workstream for a smart factory safety monitoring system. The detector is built for top-down CCTV views and uses three YOLO classes:

- `0`: person
- `1`: helmet
- `2`: vest

The repository is now organized into two tracks:

- `ppe-detection/v1_legacy/`: preserved first-pass work, including the original notebooks, reports, YOLO runs, candidate results, and downloaded checkpoint files.
- `ppe-detection/v2_pipeline/`: the clean, reproducible notebook-first pipeline for dataset intake, validation, merging, splitting, offline augmentation, candidate training, ablations, final evaluation, and inference/tracking demos.

## Repository Layout

- `environment.yml`: the main Anaconda environment definition for notebook work.
- `requirements.txt`: a pip-oriented dependency mirror for teams that prefer pip installs inside a managed environment.
- `ANACONDA_SETUP_GUIDE.md`: setup instructions, including NVIDIA CUDA-enabled PyTorch installation.
- `show_structure.py`: lightweight tree printer for repository checks.
- `ppe-detection/`: PPE-specific module documentation and both v1/v2 work areas.

## Recommended Workflow

1. Create the conda environment from `environment.yml` with Python 3.14.
2. If you train on NVIDIA hardware, install CUDA-enabled PyTorch using the commands in [ANACONDA_SETUP_GUIDE.md](ANACONDA_SETUP_GUIDE.md).
3. Use the notebooks in `ppe-detection/v2_pipeline/notebooks/` in order.
4. Keep teammate source drops under `ppe-detection/v2_pipeline/data/raw_sources/` only on your local machine.

## Data and Artifact Policy

- Raw datasets, merged datasets, split datasets, augmented datasets, YOLO runs, and model weights are intentionally ignored by Git.
- `.gitkeep` files are retained so the v2 folder structure stays visible on GitHub.
- The legacy local dataset now lives under `ppe-detection/v1_legacy/data/` and is treated as a local artifact area rather than part of the new reproducible pipeline.

## Main Entry Points

- Module overview: [ppe-detection/README.md](ppe-detection/README.md)
- Legacy area: [ppe-detection/v1_legacy/README.md](ppe-detection/v1_legacy/README.md)
- New pipeline: [ppe-detection/v2_pipeline/README.md](ppe-detection/v2_pipeline/README.md)
