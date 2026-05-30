"""Plotting helpers shared by v2 analysis notebooks."""

from __future__ import annotations

from pathlib import Path


def save_placeholder_figure(output_path: Path, title: str) -> Path:
    """Reserve a clear API for figure export helpers used by notebooks."""
    output_path = Path(output_path)
    # TODO: centralize Matplotlib styling and file export behavior.
    return output_path
