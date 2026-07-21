# Augmentation Strategy

The v2 PPE pipeline uses two augmentation layers:

- offline augmentation generated as additional training samples,
- online augmentation inside Ultralytics training.

## Split-Before-Augmentation Rule

Always split original data before augmentation:

```text
validate input sources
-> create data/generated/splits
-> augment training split only
-> build experiments
-> train
```

Validation and test images must stay original and untouched.

## Notebook 04 Offline Augmentation

Notebook `04_offline_augmentation.ipynb` reads only:

```text
data/generated/splits/train/images/
data/generated/splits/train/labels/
```

It writes all offline augmented samples into one generated train-only pool:

```text
data/generated/augmented/images/
data/generated/augmented/labels/
```

Current offline transforms:

- IR / grayscale CCTV simulation
- harsh sunlight / glare simulation
- blur / JPEG compression simulation

These transforms are intentionally mild and do not move objects. Because object
geometry is unchanged, labels are copied unchanged. This preserves all four
classes:

```text
0 = person
1 = helmet
2 = vest
3 = cleaning_coverall
```

If geometric augmentation is added later, bounding boxes must be updated instead
of copied unchanged.

## Report Policy

Notebook 04 keeps artifacts small. It saves only:

```text
reports/augmentation/offline_augmentation_report.csv
reports/augmentation/offline_augmentation_summary.csv
```

It does not save separate CSVs for each augmentation type or default example
figures. Visual checks can be added later if a specific augmentation setting
needs review.
