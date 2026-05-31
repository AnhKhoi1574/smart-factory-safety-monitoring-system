# Ablation Plan

The ablation study should answer one question at a time and should only begin after the best candidate architecture has been selected.

## Recommended Order

1. `exp_A_original_only`: no extra augmentation beyond the original split.
2. `exp_B_online_aug`: original split plus online YOLO augmentation.
3. `exp_C_offline_aug`: original split plus offline augmented training data.
4. `exp_D_full_pipeline`: online augmentation plus offline augmentation together.

## Final Test-Set Rule

The test split is reserved for the final chosen configuration only. By default, it comes from `data/test_sources/` and is copied into `data/splits_original/test/` during validation and merge. Do not also reserve 10% of the merged dataset for testing when `use_external_test_set: true`; split the merged dataset in `data/master_original/` into train/validation only. During candidate comparison and ablation work, rely on training/validation metrics and keep the test set untouched.
