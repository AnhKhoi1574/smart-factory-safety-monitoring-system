# Augmentation Strategy

The v2 pipeline separates augmentation into two layers:

- online augmentation inside YOLO training,
- offline augmentation generated as additional training samples.

## Split-Before-Augmentation Rule

The merged dataset must be split into train, val, and test before any augmentation happens. This prevents augmented siblings of an image from leaking across evaluation boundaries.

## Offline Augmentation Targets

Offline augmentation is created only from training images:

- infrared-like variants,
- harsh sunlight variants,
- blur/compression variants.

These are stored under `data/augmented_train/` and later assembled into the experiment folders.

## Notebook 04 Offline Augmentation

Notebook 04 reads only from `data/splits_original/train/images` and
`data/splits_original/train/labels`. It does not read from, write to, or
regenerate validation or test samples. This keeps evaluation data original and
prevents augmented siblings from leaking into model selection or final testing.

The offline augmentations are intentionally mild:

- IR/night simulation converts RGB images toward monochrome low-light CCTV,
  adds light sensor noise, and adjusts contrast for night/IR robustness.
- Harsh sunlight simulation moderately increases brightness and contrast, then
  adds localized glare or overexposed regions to improve robustness to outdoor
  factory lighting and reflections.
- Blur/compression simulation applies mild motion or Gaussian blur plus JPEG
  artifacts to mimic CCTV video compression, motion smear, and low-bandwidth
  image quality.

YOLO label files are copied unchanged because these transforms do not move,
crop, resize, flip, or warp objects. Augmented image and label filenames use
matching prefixes such as `ir_`, `sun_`, and `blur_`.

Validation and test sets remain untouched throughout this step.
