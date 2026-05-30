"""YOLO training entry points used by the v2 notebooks."""

from __future__ import annotations

from pathlib import Path


def train_yolo_model(data_config: Path, model_name: str, run_dir: Path, epochs: int = 100) -> dict[str, str]:
    """Train a YOLO model with notebook-supplied settings."""
    data_config = Path(data_config)
    run_dir = Path(run_dir)
    # TODO: call Ultralytics training with shared config and structured logging.
    return {"model": model_name, "status": "not_implemented", "run_dir": str(run_dir)}
