"""Reusable plotting helpers for PPE dataset EDA notebooks."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

CLASS_COLORS = {
    "person": "#9BB8D3",
    "helmet": "#DF82A8",
    "vest": "#D4C2A5",
}


def _load_plotting() -> tuple[Any, Any, Any]:
    """Import plotting dependencies only when a figure is requested."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Rectangle
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Matplotlib and Seaborn are required for EDA plotting. "
            "Install the v2_pipeline plotting dependencies before running figure cells."
        ) from exc

    sns.set_theme(style="whitegrid", context="notebook")
    return plt, sns, Rectangle


def _save_figure(fig: Any, output_path: Path | None) -> Path | None:
    if output_path is None:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return output_path


def _empty_plot(title: str, output_path: Path | None = None) -> Any:
    plt, _, _ = _load_plotting()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, "No records available", ha="center", va="center", fontsize=13)
    ax.set_axis_off()
    ax.set_title(title)
    _save_figure(fig, output_path)
    return fig


def plot_class_distribution(
    class_distribution_df: pd.DataFrame,
    output_path: Path | None = None,
    title: str = "Class distribution",
) -> plt.Figure:
    """Plot object counts by class."""
    if class_distribution_df.empty:
        return _empty_plot(title, output_path)

    plt, sns, _ = _load_plotting()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=class_distribution_df,
        x="class_name",
        y="object_count",
        hue="class_name",
        palette=CLASS_COLORS,
        legend=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Object count")
    for patch, percentage in zip(
        ax.patches, class_distribution_df["percentage"], strict=False
    ):
        height = patch.get_height()
        ax.annotate(
            f"{int(height)}\n{percentage:.1f}%",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    _save_figure(fig, output_path)
    return fig


def plot_bbox_area_distribution(
    bbox_df: pd.DataFrame,
    output_path: Path | None = None,
    title: str = "Bounding box area distribution",
) -> plt.Figure:
    """Plot normalized bounding box area by class."""
    if bbox_df.empty:
        return _empty_plot(title, output_path)

    plt, sns, _ = _load_plotting()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=bbox_df,
        x="box_area_norm",
        hue="class_name",
        bins=40,
        palette=CLASS_COLORS,
        element="step",
        stat="count",
        common_norm=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Normalized box area")
    ax.set_ylabel("Object count")
    ax.set_xlim(left=0)
    _save_figure(fig, output_path)
    return fig


def plot_bbox_width_height_scatter(
    bbox_df: pd.DataFrame,
    output_path: Path | None = None,
    title: str = "Bounding box width vs height",
) -> plt.Figure:
    """Plot normalized width and height for every box."""
    if bbox_df.empty:
        return _empty_plot(title, output_path)

    plt, sns, _ = _load_plotting()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=bbox_df,
        x="width_norm",
        y="height_norm",
        hue="class_name",
        palette=CLASS_COLORS,
        alpha=0.65,
        s=34,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Normalized width")
    ax.set_ylabel("Normalized height")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(title="Class")
    _save_figure(fig, output_path)
    return fig


def plot_objects_per_image(
    objects_per_image_df: pd.DataFrame,
    output_path: Path | None = None,
    title: str = "Objects per image",
) -> plt.Figure:
    """Plot the distribution of total object counts per image."""
    if objects_per_image_df.empty:
        return _empty_plot(title, output_path)

    plt, sns, _ = _load_plotting()
    fig, ax = plt.subplots(figsize=(8, 5))
    max_count = int(objects_per_image_df["object_count"].max())
    bins = range(0, max_count + 2)
    sns.histplot(
        data=objects_per_image_df, x="object_count", bins=bins, color="#264653", ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Object count per image")
    ax.set_ylabel("Image count")
    _save_figure(fig, output_path)
    return fig


def _draw_box(ax: Any, row: pd.Series, image_width: int, image_height: int) -> None:
    _, _, Rectangle = _load_plotting()
    class_name = str(row["class_name"])
    color = CLASS_COLORS.get(class_name, "#457B9D")
    box_width = float(row["width_norm"]) * image_width
    box_height = float(row["height_norm"]) * image_height
    left = (float(row["x_center_norm"]) * image_width) - box_width / 2
    top = (float(row["y_center_norm"]) * image_height) - box_height / 2
    ax.add_patch(
        Rectangle(
            (left, top),
            box_width,
            box_height,
            fill=False,
            edgecolor=color,
            linewidth=2.2,
        )
    )

    # Scale label size with the object while keeping tiny PPE labels readable.
    object_scale = min(box_width, box_height)
    label_font_size = max(3.0, min(7.0, object_scale * 0.10))
    label_padding = max(0.2, min(0.8, label_font_size * 0.08))
    if class_name in {"person", "vest"}:
        label_x = left + 1
        label_y = top + box_height - 2
        vertical_alignment = "bottom"
    else:
        label_x = left
        label_y = max(top - 2, 0)
        vertical_alignment = "bottom"

    ax.text(
        label_x,
        label_y,
        class_name,
        color="black",
        fontsize=label_font_size,
        va=vertical_alignment,
        bbox={
            "facecolor": color,
            "alpha": 0.85,
            "edgecolor": "none",
            "pad": label_padding,
        },
    )


def visualize_sample_annotations(
    image_df: pd.DataFrame,
    bbox_df: pd.DataFrame,
    output_path: Path | None = None,
    sample_count: int = 6,
    random_state: int = 42,
    title: str = "Sample annotations",
) -> plt.Figure:
    """Create a grid of sample images with YOLO boxes drawn on top."""
    if image_df.empty:
        return _empty_plot(title, output_path)

    annotated_names = (
        bbox_df["image_name"].drop_duplicates().tolist() if not bbox_df.empty else []
    )
    candidates = (
        image_df[image_df["image_name"].isin(annotated_names)]
        if annotated_names
        else image_df
    )
    if candidates.empty:
        candidates = image_df

    sample_size = min(sample_count, len(candidates))
    sample_df = (
        candidates.sample(n=sample_size, random_state=random_state)
        if sample_size
        else candidates
    )

    plt, _, _ = _load_plotting()
    cols = min(3, max(1, sample_size))
    rows = ceil(sample_size / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, image_row in zip(
        axes_list, sample_df.itertuples(index=False), strict=False
    ):
        image_path = Path(image_row.image_path)
        with Image.open(image_path) as image:
            ax.imshow(image.convert("RGB"))
        image_boxes = bbox_df.loc[bbox_df["image_name"] == image_row.image_name]
        for _, box_row in image_boxes.iterrows():
            _draw_box(
                ax, box_row, int(image_row.image_width), int(image_row.image_height)
            )
        ax.set_title(image_row.image_name, fontsize=10)
        ax.set_axis_off()

    for ax in axes_list[sample_size:]:
        ax.set_axis_off()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def save_placeholder_figure(output_path: Path, title: str) -> Path:
    """Backward-compatible placeholder export helper."""
    plt, _, _ = _load_plotting()
    fig = _empty_plot(title, output_path)
    plt.close(fig)
    return Path(output_path)
