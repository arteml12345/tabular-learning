"""Tests for hybrid_embed.data.splits: CV and single-split generation."""

import numpy as np

from hybrid_embed.data.splits import generate_cv_splits, generate_single_split


def test_cv_splits_correct_count():
    """5-fold CV should produce exactly 5 FoldSplit objects."""
    rng = np.random.RandomState(42)
    y = rng.choice([0, 1], size=200)
    print(f"Dataset size: {len(y)}")

    splits = generate_cv_splits(y, n_folds=5, task="binary")
    print(f"Number of splits: {len(splits)}")

    assert len(splits) == 5
    for i, s in enumerate(splits):
        assert s.fold_index == i
        print(f"Fold {i}: train={len(s.train_indices)}, "
              f"val={len(s.val_indices)}, test={len(s.test_indices)}")
    print("Test passed: correct number of folds")


def test_cv_splits_no_test_overlap():
    """Test indices across folds should be disjoint and cover all samples."""
    rng = np.random.RandomState(42)
    y = rng.normal(0, 1, size=200)

    splits = generate_cv_splits(y, n_folds=5, task="regression")
    all_test = np.concatenate([s.test_indices for s in splits])
    print(f"Total test indices: {len(all_test)}, unique: {len(np.unique(all_test))}")

    assert len(all_test) == len(np.unique(all_test)), "Test indices should not overlap"
    assert set(all_test) == set(range(200)), "Test indices should cover all samples"
    print("Test passed: test sets are disjoint and cover all samples")


def test_cv_splits_train_val_disjoint():
    """Within each fold, train and val indices should be disjoint."""
    rng = np.random.RandomState(42)
    y = rng.choice([0, 1], size=200)

    splits = generate_cv_splits(y, n_folds=5, task="binary")
    for s in splits:
        overlap = set(s.train_indices) & set(s.val_indices)
        print(f"Fold {s.fold_index}: train/val overlap = {len(overlap)}")
        assert len(overlap) == 0, f"Fold {s.fold_index} has train/val overlap"
    print("Test passed: train and val are disjoint in all folds")


def test_cv_splits_no_leak_to_test():
    """Within each fold, neither train nor val should overlap with test."""
    rng = np.random.RandomState(42)
    y = rng.choice([0, 1], size=200)

    splits = generate_cv_splits(y, n_folds=5, task="binary")
    for s in splits:
        train_test_overlap = set(s.train_indices) & set(s.test_indices)
        val_test_overlap = set(s.val_indices) & set(s.test_indices)
        print(f"Fold {s.fold_index}: train/test overlap = {len(train_test_overlap)}, "
              f"val/test overlap = {len(val_test_overlap)}")
        assert len(train_test_overlap) == 0
        assert len(val_test_overlap) == 0
    print("Test passed: no data leakage into test set")


def test_cv_splits_stratified_classification():
    """Stratified splits should preserve class ratio in test sets."""
    rng = np.random.RandomState(42)
    y = np.array([0] * 140 + [1] * 60)
    rng.shuffle(y)
    overall_ratio = y.mean()
    print(f"Overall class-1 ratio: {overall_ratio:.2f}")

    splits = generate_cv_splits(y, n_folds=5, task="binary")
    for s in splits:
        test_ratio = y[s.test_indices].mean()
        print(f"Fold {s.fold_index}: test class-1 ratio = {test_ratio:.2f}")
        assert abs(test_ratio - overall_ratio) < 0.1, (
            f"Fold {s.fold_index} test ratio {test_ratio:.2f} deviates too "
            f"much from overall {overall_ratio:.2f}"
        )
    print("Test passed: stratification preserves class ratios")


def test_cv_splits_reproducible():
    """Same seed should produce identical splits."""
    rng = np.random.RandomState(42)
    y = rng.choice([0, 1], size=200)

    splits1 = generate_cv_splits(y, n_folds=5, task="binary", master_seed=42)
    splits2 = generate_cv_splits(y, n_folds=5, task="binary", master_seed=42)

    for s1, s2 in zip(splits1, splits2):
        assert np.array_equal(s1.train_indices, s2.train_indices)
        assert np.array_equal(s1.val_indices, s2.val_indices)
        assert np.array_equal(s1.test_indices, s2.test_indices)
        print(f"Fold {s1.fold_index}: identical across both runs")
    print("Test passed: splits are reproducible with same seed")


def test_cv_splits_different_seed():
    """Different seeds should produce different splits."""
    rng = np.random.RandomState(42)
    y = rng.choice([0, 1], size=200)

    splits1 = generate_cv_splits(y, n_folds=5, task="binary", master_seed=42)
    splits2 = generate_cv_splits(y, n_folds=5, task="binary", master_seed=99)

    any_different = False
    for s1, s2 in zip(splits1, splits2):
        if not np.array_equal(s1.test_indices, s2.test_indices):
            any_different = True
            break
    print(f"Splits differ with different seeds: {any_different}")
    assert any_different, "Different seeds should produce different splits"
    print("Test passed: different seeds produce different splits")


def test_single_split_fractions():
    """Single split should produce approximately correct set sizes."""
    rng = np.random.RandomState(42)
    n = 200
    y = rng.normal(0, 1, size=n)

    split = generate_single_split(
        y, task="regression",
        train_fraction=0.7, val_fraction=0.15, master_seed=42,
    )
    print(f"Train: {len(split.train_indices)}, "
          f"Val: {len(split.val_indices)}, "
          f"Test: {len(split.test_indices)}")

    assert abs(len(split.train_indices) / n - 0.7) < 0.05
    assert abs(len(split.val_indices) / n - 0.15) < 0.05
    assert abs(len(split.test_indices) / n - 0.15) < 0.05
    assert split.fold_index == 0
    print("Test passed: single split has correct approximate fractions")


def test_single_split_no_overlap():
    """Single split train, val, test should be disjoint."""
    rng = np.random.RandomState(42)
    y = rng.choice([0, 1], size=200)

    split = generate_single_split(y, task="binary")

    train_set = set(split.train_indices)
    val_set = set(split.val_indices)
    test_set = set(split.test_indices)

    assert len(train_set & val_set) == 0, "Train/val overlap"
    assert len(train_set & test_set) == 0, "Train/test overlap"
    assert len(val_set & test_set) == 0, "Val/test overlap"

    total = len(train_set) + len(val_set) + len(test_set)
    print(f"Total indices: {total}, expected: {len(y)}")
    assert total == len(y), "All samples should be assigned"
    print("Test passed: single split sets are disjoint and complete")
