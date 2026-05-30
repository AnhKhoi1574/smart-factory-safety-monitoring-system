"""Validation helpers for teammate-provided YOLO datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ValidationIssue:
    """Represents a single dataset validation finding."""

    sample_id: str
    message: str
    severity: str = "error"


@dataclass(slots=True)
class ValidationReport:
    """Summary of dataset validation results."""

    total_images: int
    total_labels: int
    valid_pairs: int
    issues: list[ValidationIssue] = field(default_factory=list)


def validate_dataset(source_dir: Path) -> ValidationReport:
    """Validate a teammate dataset folder containing `images/` and `labels/`."""
    source_dir = Path(source_dir)
    # TODO: verify folder layout, image/label matching, class ids, and YOLO bbox format.
    return ValidationReport(total_images=0, total_labels=0, valid_pairs=0, issues=[])
