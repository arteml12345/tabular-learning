"""Split generation: K-fold CV and single train/val/test splits.

Provides reproducible data splitting with stratification for
classification tasks. Fold 0 is conventionally used as the HPO
split; the remaining folds are used for final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


@dataclass
class FoldSplit:
    """Index arrays defining a single train/val/test split.

    Used by both K-fold CV and single-split modes.
    All indices refer to positions in the original dataset.

    Parameters
    ----------
    fold_index : int
        Which fold this split belongs to (0-based). Fold 0 is
        used as the HPO split in K-fold mode.
    train_indices : np.ndarray
        Row indices for the training set.
    val_indices : np.ndarray
        Row indices for the validation set (used for early
        stopping during training).
    test_indices : np.ndarray
        Row indices for the test set (used for evaluation only).
    """

    fold_index: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


def generate_cv_splits(
    y: np.ndarray,
    n_folds: int = 5,
    task: str = "regression",
    master_seed: int = 42,
    val_fraction: float = 0.15,
) -> list[FoldSplit]:
    """Generate K-fold cross-validation splits.

    For each fold the held-out portion becomes the test set, and
    the remaining samples are further split into train and val
    (val is used for early stopping during training).

    For classification tasks, ``StratifiedKFold`` preserves class
    distribution in each fold. For regression, ``KFold`` is used.

    Parameters
    ----------
    y : np.ndarray
        Target variable (used for stratification in classification).
    n_folds : int, default 5
        Number of cross-validation folds.
    task : str, default "regression"
        ``"binary"`` or ``"multiclass"`` for stratified splits,
        ``"regression"`` for standard splits.
    master_seed : int, default 42
        Random seed for reproducibility.
    val_fraction : float, default 0.15
        Fraction of the non-test data to hold out as validation
        within each fold.

    Returns
    -------
    list[FoldSplit]
        One ``FoldSplit`` per fold, length ``n_folds``.
    """
    if task in ("binary", "multiclass"):
        splitter = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=master_seed,
        )
        split_iter = splitter.split(np.zeros(len(y)), y)
    else:
        splitter = KFold(
            n_splits=n_folds, shuffle=True, random_state=master_seed,
        )
        split_iter = splitter.split(np.zeros(len(y)))

    fold_splits = []
    for fold_index, (non_test_indices, test_indices) in enumerate(split_iter):
        if task in ("binary", "multiclass"):
            train_idx, val_idx = train_test_split(
                non_test_indices,
                test_size=val_fraction,
                random_state=master_seed + fold_index,
                stratify=y[non_test_indices],
            )
        else:
            train_idx, val_idx = train_test_split(
                non_test_indices,
                test_size=val_fraction,
                random_state=master_seed + fold_index,
            )

        fold_splits.append(FoldSplit(
            fold_index=fold_index,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_indices,
        ))

    return fold_splits


def generate_single_split(
    y: np.ndarray,
    task: str = "regression",
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    master_seed: int = 42,
) -> FoldSplit:
    """Generate a single train/val/test split.

    Intended for quick debugging, not for final evaluation.
    The test fraction is ``1 - train_fraction - val_fraction``.

    Parameters
    ----------
    y : np.ndarray
        Target variable.
    task : str, default "regression"
        Task type (used for stratification).
    train_fraction : float, default 0.7
        Fraction of data for training.
    val_fraction : float, default 0.15
        Fraction of data for validation.
    master_seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    FoldSplit
        A single split with ``fold_index=0``.
    """
    test_fraction = 1.0 - train_fraction - val_fraction
    all_indices = np.arange(len(y))

    stratify_y = y if task in ("binary", "multiclass") else None

    trainval_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_fraction,
        random_state=master_seed,
        stratify=stratify_y,
    )

    val_relative = val_fraction / (train_fraction + val_fraction)
    stratify_tv = y[trainval_idx] if stratify_y is not None else None

    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_relative,
        random_state=master_seed,
        stratify=stratify_tv,
    )

    return FoldSplit(
        fold_index=0,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
    )
