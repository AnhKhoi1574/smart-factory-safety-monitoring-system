"""Model export helpers for downstream deployment targets."""

from __future__ import annotations

from pathlib import Path


def export_model(model_path: Path, export_format: str = "onnx") -> Path:
    """Export a final YOLO model to a deployment-friendly format."""
    model_path = Path(model_path)
    # TODO: wrap Ultralytics export calls and capture output artifact paths.
    return model_path
