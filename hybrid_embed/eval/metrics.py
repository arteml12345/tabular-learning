"""Metric computation per task type.

Provides functions to compute primary and secondary metrics for
binary classification, multiclass classification, and regression
tasks, plus helpers for metric aggregation across CV folds.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


_HIGHER_IS_BETTER = {
    "roc_auc": True,
    "pr_auc": True,
    "accuracy": True,
    "macro_f1": True,
    "r_squared": True,
    "log_loss": False,
    "rmse": False,
    "mae": False,
}

_PRIMARY_METRIC = {
    "binary": "roc_auc",
    "multiclass": "accuracy",
    "regression": "rmse",
}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None,
    task: str,
) -> dict[str, float]:
    """Compute all relevant metrics for a given task.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels/values.
    y_pred : np.ndarray
        Predicted labels (classification) or values (regression).
    y_pred_proba : np.ndarray or None
        Predicted probabilities. Shape ``(n,)`` for binary
        (probability of positive class) or ``(n, K)`` for
        multiclass. None for regression.
    task : str
        ``"binary"``, ``"multiclass"``, or ``"regression"``.

    Returns
    -------
    dict[str, float]
        Keys depend on task:

        - binary: ``roc_auc``, ``log_loss``, ``pr_auc``
        - multiclass: ``accuracy``, ``macro_f1``, ``log_loss``
        - regression: ``rmse``, ``mae``, ``r_squared``
    """
    if task == "binary":
        return _binary_metrics(y_true, y_pred, y_pred_proba)
    elif task == "multiclass":
        return _multiclass_metrics(y_true, y_pred, y_pred_proba)
    elif task == "regression":
        return _regression_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task: {task}")


def get_primary_metric_name(task: str) -> str:
    """Return the primary metric name for a task.

    Parameters
    ----------
    task : str
        ``"binary"``, ``"multiclass"``, or ``"regression"``.

    Returns
    -------
    str
        ``"roc_auc"`` for binary, ``"accuracy"`` for multiclass,
        ``"rmse"`` for regression.
    """
    if task not in _PRIMARY_METRIC:
        raise ValueError(f"Unknown task: {task}")
    return _PRIMARY_METRIC[task]


def is_higher_better(metric_name: str) -> bool:
    """Return True if higher values are better for this metric.

    Parameters
    ----------
    metric_name : str
        One of the known metric names.

    Returns
    -------
    bool
        True for roc_auc, pr_auc, accuracy, macro_f1, r_squared.
        False for log_loss, rmse, mae.
    """
    if metric_name not in _HIGHER_IS_BETTER:
        raise ValueError(f"Unknown metric: {metric_name}")
    return _HIGHER_IS_BETTER[metric_name]


def aggregate_fold_metrics(
    fold_metrics: list[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute mean and std of each metric across folds.

    Parameters
    ----------
    fold_metrics : list[dict[str, float]]
        One dict per fold, all with the same keys.

    Returns
    -------
    mean_dict : dict[str, float]
        Mean of each metric across folds.
    std_dict : dict[str, float]
        Standard deviation of each metric across folds.
    """
    keys = fold_metrics[0].keys()
    mean_dict = {}
    std_dict = {}
    for key in keys:
        values = [m[key] for m in fold_metrics]
        mean_dict[key] = float(np.mean(values))
        std_dict[key] = float(np.std(values))
    return mean_dict, std_dict


def _binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None,
) -> dict[str, float]:
    """Compute binary classification metrics."""
    metrics = {}
    if y_pred_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
        metrics["log_loss"] = float(log_loss(y_true, y_pred_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_pred_proba))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["log_loss"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def _multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None,
) -> dict[str, float]:
    """Compute multiclass classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    if y_pred_proba is not None:
        metrics["log_loss"] = float(log_loss(y_true, y_pred_proba))
    else:
        metrics["log_loss"] = float("nan")
    return metrics


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r_squared": float(r2_score(y_true, y_pred)),
    }
