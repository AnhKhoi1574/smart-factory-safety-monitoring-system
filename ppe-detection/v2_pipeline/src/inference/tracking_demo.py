"""Tracking demo helpers for PPE detector inference notebooks."""

from __future__ import annotations

from pathlib import Path


def run_tracking_demo(model_path: Path, video_path: Path, tracker: str = "bytetrack") -> dict[str, str]:
    """Run a placeholder tracked inference pass over a video source."""
    model_path = Path(model_path)
    video_path = Path(video_path)
    # TODO: integrate YOLO predict/track flow with ByteTrack or BoT-SORT options.
    return {"tracker": tracker, "status": "not_implemented"}
