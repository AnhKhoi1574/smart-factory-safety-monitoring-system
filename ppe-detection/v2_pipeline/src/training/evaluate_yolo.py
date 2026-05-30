"""Evaluation helpers for YOLO candidate and final models."""

from __future__ import annotations

from pathlib import Path


def evaluate_yolo_model(model_path: Path, data_config: Path) -> dict[str, float]:
    """Evaluate a trained YOLO model against a dataset definition."""
    model_path = Path(model_path)
    data_config = Path(data_config)
    # TODO: run validation and return comparable metrics for model triage.
    return {"map50": 0.0, "recall": 0.0}
