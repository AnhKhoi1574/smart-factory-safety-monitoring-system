# Augmentation Strategy

The v2 pipeline separates augmentation into two layers:

- online augmentation inside YOLO training,
- offline augmentation generated as additional training samples.

## Split-Before-Augmentation Rule

The merged original dataset must be split into train, val, and test before any augmentation happens. This prevents augmented siblings of an image from leaking across evaluation boundaries.

## Offline Augmentation Targets

Offline augmentation is created only from training images:

- infrared-like variants,
- harsh sunlight variants,
- blur/compression variants.

These are stored under `data/augmented_train/` and later assembled into the experiment folders.
