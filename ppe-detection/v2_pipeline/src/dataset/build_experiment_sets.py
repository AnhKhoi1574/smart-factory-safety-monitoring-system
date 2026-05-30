"""Build experiment-ready dataset variants for ablation studies."""

from __future__ import annotations

from pathlib import Path


def build_experiment_sets(base_dir: Path, experiments_dir: Path) -> dict[str, Path]:
    """Create experiment dataset folders for ablation and final training stages."""
    base_dir = Path(base_dir)
    experiments_dir = Path(experiments_dir)
    # TODO: assemble exp_A through exp_D from original splits and offline augmentations.
    return {
        "exp_A_original_only": experiments_dir / "exp_A_original_only",
        "exp_B_online_aug": experiments_dir / "exp_B_online_aug",
        "exp_C_offline_aug": experiments_dir / "exp_C_offline_aug",
        "exp_D_full_pipeline": experiments_dir / "exp_D_full_pipeline",
    }
