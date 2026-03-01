"""Tests for hybrid_embed.eval.metrics: metric computation and aggregation."""

import numpy as np

from hybrid_embed.eval.metrics import (
    compute_metrics,
    get_primary_metric_name,
    is_higher_better,
    aggregate_fold_metrics,
)


def test_binary_metrics_perfect():
    """Perfect binary predictions should yield roc_auc=1 and low log_loss."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])

    metrics = compute_metrics(y_true, y_pred, y_proba, task="binary")
    print(f"Perfect binary metrics: {metrics}")

    assert metrics["roc_auc"] == 1.0
    assert metrics["log_loss"] < 0.1
    assert metrics["pr_auc"] == 1.0
    print("Test passed: perfect binary predictions yield ideal metrics")


def test_binary_metrics_random():
    """Random binary predictions should yield roc_auc near 0.5."""
    rng = np.random.RandomState(42)
    n = 1000
    y_true = rng.choice([0, 1], size=n)
    y_proba = rng.uniform(0, 1, size=n)
    y_pred = (y_proba > 0.5).astype(int)

    metrics = compute_metrics(y_true, y_pred, y_proba, task="binary")
    print(f"Random binary metrics: {metrics}")

    assert abs(metrics["roc_auc"] - 0.5) < 0.1
    print("Test passed: random predictions yield roc_auc near 0.5")


def test_multiclass_metrics():
    """Multiclass metrics with known predictions."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_proba = np.zeros((10, 3))
    for i in range(10):
        y_proba[i, y_pred[i]] = 0.9
        y_proba[i] += 0.05
        y_proba[i] /= y_proba[i].sum()

    metrics = compute_metrics(y_true, y_pred, y_proba, task="multiclass")
    print(f"Multiclass metrics: {metrics}")

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["log_loss"] < 0.2
    print("Test passed: perfect multiclass predictions yield ideal metrics")


def test_regression_metrics_perfect():
    """Perfect regression predictions should yield rmse=0 and r_squared=1."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    metrics = compute_metrics(y_true, y_pred, None, task="regression")
    print(f"Perfect regression metrics: {metrics}")

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r_squared"] == 1.0
    print("Test passed: perfect regression predictions yield ideal metrics")


def test_regression_metrics_known():
    """Hand-calculated regression metrics."""
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    metrics = compute_metrics(y_true, y_pred, None, task="regression")
    print(f"Known regression metrics: {metrics}")

    expected_mse = np.mean((y_true - y_pred) ** 2)
    expected_rmse = np.sqrt(expected_mse)
    expected_mae = np.mean(np.abs(y_true - y_pred))
    print(f"Expected RMSE: {expected_rmse:.6f}, MAE: {expected_mae:.6f}")

    assert abs(metrics["rmse"] - expected_rmse) < 1e-6
    assert abs(metrics["mae"] - expected_mae) < 1e-6
    print("Test passed: regression metrics match hand calculations")


def test_primary_metric_names():
    """Primary metric names should match spec."""
    assert get_primary_metric_name("binary") == "roc_auc"
    assert get_primary_metric_name("multiclass") == "accuracy"
    assert get_primary_metric_name("regression") == "rmse"
    print("Test passed: primary metric names are correct")


def test_is_higher_better():
    """is_higher_better should return correct values for all metrics."""
    assert is_higher_better("roc_auc") is True
    assert is_higher_better("pr_auc") is True
    assert is_higher_better("accuracy") is True
    assert is_higher_better("macro_f1") is True
    assert is_higher_better("r_squared") is True
    assert is_higher_better("log_loss") is False
    assert is_higher_better("rmse") is False
    assert is_higher_better("mae") is False
    print("Test passed: is_higher_better correct for all metrics")


def test_aggregate_fold_metrics():
    """Aggregation should compute correct mean and std across folds."""
    fold_metrics = [
        {"roc_auc": 0.90, "log_loss": 0.30},
        {"roc_auc": 0.92, "log_loss": 0.28},
        {"roc_auc": 0.88, "log_loss": 0.32},
    ]

    mean_dict, std_dict = aggregate_fold_metrics(fold_metrics)
    print(f"Mean: {mean_dict}")
    print(f"Std:  {std_dict}")

    expected_mean_auc = np.mean([0.90, 0.92, 0.88])
    expected_std_auc = np.std([0.90, 0.92, 0.88])
    expected_mean_ll = np.mean([0.30, 0.28, 0.32])
    expected_std_ll = np.std([0.30, 0.28, 0.32])

    assert abs(mean_dict["roc_auc"] - expected_mean_auc) < 1e-10
    assert abs(std_dict["roc_auc"] - expected_std_auc) < 1e-10
    assert abs(mean_dict["log_loss"] - expected_mean_ll) < 1e-10
    assert abs(std_dict["log_loss"] - expected_std_ll) < 1e-10
    print("Test passed: fold aggregation computes correct mean and std")
