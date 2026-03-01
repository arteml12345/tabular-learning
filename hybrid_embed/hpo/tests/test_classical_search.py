"""Tests for hybrid_embed.hpo.classical_search."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
from hyperopt import hp
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from hybrid_embed.config import ClassicalStepConfig
from hybrid_embed.hpo.classical_search import hyperopt_search_classical


# =====================================================================
# Helpers
# =====================================================================

def _make_binary_embeddings(n_train=300, n_val=80, n_features=64, seed=42):
    """Synthetic embedding data with moderate binary signal.

    Adds noise to the decision boundary so defaults land around
    0.80-0.85, leaving room for HPO to show improvement.
    """
    rng = np.random.RandomState(seed)
    E = rng.randn(n_train + n_val, n_features).astype(np.float64)
    logit = 0.6 * E[:, 0] + 0.4 * E[:, 1] - 0.3 * E[:, 2] + rng.randn(n_train + n_val) * 1.2
    y = (logit > 0).astype(int)
    return (
        E[:n_train], y[:n_train],
        E[n_train:], y[n_train:],
    )


def _make_regression_embeddings(n_train=300, n_val=80, n_features=64, seed=42):
    """Synthetic embedding data with learnable regression signal."""
    rng = np.random.RandomState(seed)
    E = rng.randn(n_train + n_val, n_features).astype(np.float64)
    y = (
        3.0 * E[:, 0] + 1.5 * E[:, 1] - 2.0 * E[:, 2]
        + rng.randn(n_train + n_val) * 0.3
    )
    return (
        E[:n_train], y[:n_train],
        E[n_train:], y[n_train:],
    )


def _make_random_binary_embeddings(n_train=300, n_val=80, n_features=64, seed=99):
    """Embedding data with purely random binary labels."""
    rng = np.random.RandomState(seed)
    E = rng.randn(n_train + n_val, n_features).astype(np.float64)
    y = rng.choice([0, 1], size=n_train + n_val).astype(int)
    return (
        E[:n_train], y[:n_train],
        E[n_train:], y[n_train:],
    )


def _train_with_defaults(E_train, y_train, E_val, y_val, task, model_type):
    """Train a classical model with default params, return primary metric."""
    from hybrid_embed.classical.model_zoo import build_model, train_classical_model
    from hybrid_embed.eval.metrics import compute_metrics, get_primary_metric_name

    step = ClassicalStepConfig(model_type=model_type)
    from hybrid_embed.classical.model_zoo import resolve_classical_step
    resolved = resolve_classical_step(step, task)

    model = build_model(resolved["class"], resolved["params"], task)
    trained = train_classical_model(
        model, E_train, y_train,
        E_val=E_val, y_val=y_val,
        supports_early_stopping=resolved["supports_early_stopping"],
    )

    metric_name = get_primary_metric_name(task)
    if task == "binary":
        proba = trained.predict_proba(E_val)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics = compute_metrics(y_val, preds, proba, task)
    elif task == "regression":
        preds = trained.predict(E_val)
        metrics = compute_metrics(y_val, preds, None, task)
    else:
        proba = trained.predict_proba(E_val)
        preds = trained.predict(E_val)
        metrics = compute_metrics(y_val, preds, proba, task)
    return metrics[metric_name]


# =====================================================================
# Structural tests
# =====================================================================

def test_classical_search_xgboost_binary():
    """XGBoost binary search runs and returns dict with required keys."""
    E_tr, y_tr, E_v, y_v = _make_binary_embeddings(n_train=200, n_val=50)
    step = ClassicalStepConfig(model_type="xgboost")
    result = hyperopt_search_classical(
        E_tr, y_tr, E_v, y_v,
        task="binary",
        classical_step=step,
        max_trials=5,
    )
    assert isinstance(result, dict)
    for key in ("best_params", "best_score", "n_trials_completed", "total_time_seconds"):
        assert key in result, f"Missing key: {key}"
    assert result["n_trials_completed"] == 5
    assert np.isfinite(result["best_score"])
    print(f"XGBoost binary: best_score={result['best_score']:.4f}, "
          f"trials={result['n_trials_completed']}")
    print("Test passed: XGBoost binary search structure correct")


def test_classical_search_ridge_regression():
    """Ridge regression search returns valid output."""
    E_tr, y_tr, E_v, y_v = _make_regression_embeddings(n_train=200, n_val=50)
    step = ClassicalStepConfig(model_type="ridge")
    result = hyperopt_search_classical(
        E_tr, y_tr, E_v, y_v,
        task="regression",
        classical_step=step,
        max_trials=5,
    )
    assert result["n_trials_completed"] == 5
    assert np.isfinite(result["best_score"])
    print(f"Ridge regression: best_score(RMSE)={result['best_score']:.4f}")
    print("Test passed: Ridge regression search valid")


def test_classical_search_custom_space():
    """Custom narrow space restricts the search."""
    custom_space = {
        "n_estimators": hp.quniform("n_estimators", 50, 100, 50),
        "max_depth": hp.quniform("max_depth", 3, 5, 1),
    }
    E_tr, y_tr, E_v, y_v = _make_binary_embeddings(n_train=200, n_val=50)
    step = ClassicalStepConfig(model_type="xgboost", search_space=custom_space)
    result = hyperopt_search_classical(
        E_tr, y_tr, E_v, y_v,
        task="binary",
        classical_step=step,
        max_trials=5,
    )
    bp = result["best_params"]
    assert bp["n_estimators"] in (50, 100)
    assert bp["max_depth"] in (3, 4, 5)
    print(f"Custom space: n_estimators={bp['n_estimators']}, max_depth={bp['max_depth']}")
    print("Test passed: custom space constrains search")


def test_classical_search_custom_model():
    """Custom model (KNN) search produces valid output."""
    custom_space = {
        "n_neighbors": hp.choice("n_neighbors", [3, 5, 7, 11, 15]),
    }
    E_tr, y_tr, E_v, y_v = _make_binary_embeddings(n_train=200, n_val=50)
    step = ClassicalStepConfig(
        model_type="custom",
        model_class=KNeighborsClassifier,
        search_space=custom_space,
        fixed_params={"weights": "distance"},
    )
    result = hyperopt_search_classical(
        E_tr, y_tr, E_v, y_v,
        task="binary",
        classical_step=step,
        max_trials=5,
    )
    assert result["n_trials_completed"] == 5
    assert np.isfinite(result["best_score"])
    assert result["best_params"]["weights"] == "distance"
    print(f"KNN: best_score={result['best_score']:.4f}")
    print("Test passed: custom model search valid")


def test_classical_search_logs_trials():
    """Logger receives trial records in JSONL."""
    from hybrid_embed.log.local_logger import LocalLogger

    E_tr, y_tr, E_v, y_v = _make_binary_embeddings(n_train=200, n_val=50)
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LocalLogger(output_dir=tmpdir, dataset_id="test")
        step = ClassicalStepConfig(model_type="xgboost")
        hyperopt_search_classical(
            E_tr, y_tr, E_v, y_v,
            task="binary",
            classical_step=step,
            max_trials=5,
            logger=logger,
        )
        log_path = os.path.join(logger.get_hpo_dir(), "classical_hpo_trials.jsonl")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 5
        for line in lines:
            record = json.loads(line)
            assert "trial_id" in record
            assert "params" in record
            assert "val_metric" in record
        print(f"Logged {len(lines)} trial records")
        print("Test passed: logger captures all trials")


def test_classical_search_time_budget():
    """Time budget limits the number of trials."""
    E_tr, y_tr, E_v, y_v = _make_binary_embeddings(n_train=200, n_val=50)
    step = ClassicalStepConfig(model_type="xgboost")
    result = hyperopt_search_classical(
        E_tr, y_tr, E_v, y_v,
        task="binary",
        classical_step=step,
        max_trials=1000,
        time_budget_seconds=5,
    )
    assert result["n_trials_completed"] < 1000
    assert result["total_time_seconds"] < 30
    print(f"Time budget: {result['n_trials_completed']} trials in "
          f"{result['total_time_seconds']:.1f}s")
    print("Test passed: time budget limits trials")


# =====================================================================
# HPO progress tests: signal data should show improvement
# =====================================================================

def test_hpo_improves_over_defaults_xgboost_binary():
    """HPO should find XGBoost params better than defaults on learnable data."""
    E_tr, y_tr, E_v, y_v = _make_binary_embeddings(seed=42)

    default_score = _train_with_defaults(
        E_tr, y_tr, E_v, y_v, task="binary", model_type="xgboost",
    )
    print(f"Default XGBoost ROC-AUC: {default_score:.4f}")

    step = ClassicalStepConfig(model_type="xgboost")
    result = hyperopt_search_classical(
        E_tr, y_tr, E_v, y_v,
        task="binary",
        classical_step=step,
        max_trials=15,
        master_seed=42,
    )
    hpo_score = result["best_score"]
    print(f"HPO best ROC-AUC: {hpo_score:.4f}")
    print(f"HPO params: {result['best_params']}")

    assert hpo_score > 0.70, (
        f"HPO XGBoost should achieve decent ROC-AUC on learnable data; got {hpo_score:.4f}"
    )
    tolerance = 0.05
    assert hpo_score >= default_score + tolerance, (
        f"HPO should improve over default by at least {tolerance}: "
        f"HPO={hpo_score:.4f} vs Default={default_score:.4f}"
    )
    print("Test passed: HPO improves over defaults on binary XGBoost")


def test_hpo_best_trial_beats_median_regression():
    """HPO best trial should beat the median trial, proving optimization works."""
    from hybrid_embed.log.local_logger import LocalLogger

    E_tr, y_tr, E_v, y_v = _make_regression_embeddings(seed=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LocalLogger(output_dir=tmpdir, dataset_id="test")
        step = ClassicalStepConfig(model_type="xgboost")
        result = hyperopt_search_classical(
            E_tr, y_tr, E_v, y_v,
            task="regression",
            classical_step=step,
            max_trials=20,
            master_seed=42,
            logger=logger,
        )

        log_path = os.path.join(logger.get_hpo_dir(), "classical_hpo_trials.jsonl")
        with open(log_path) as f:
            records = [json.loads(line) for line in f]

        scores = [r["val_metric"]["rmse"] for r in records]
        median_score = float(np.median(scores))
        best_score = result["best_score"]
        worst_score = max(scores)

        print(f"Trial RMSEs: best={best_score:.4f}, median={median_score:.4f}, "
              f"worst={worst_score:.4f}")

        assert best_score < median_score, (
            f"Best trial should beat median: best={best_score:.4f} vs "
            f"median={median_score:.4f}"
        )
        assert best_score < worst_score, (
            f"Best trial should beat worst: best={best_score:.4f} vs "
            f"worst={worst_score:.4f}"
        )
        assert best_score < 2.0, (
            f"Best RMSE should be reasonable on learnable data; got {best_score:.4f}"
        )
        print("Test passed: HPO best trial beats median on regression")


def test_hpo_explores_diverse_configs():
    """HPO trial logs should show diverse configurations."""
    from hybrid_embed.log.local_logger import LocalLogger

    E_tr, y_tr, E_v, y_v = _make_binary_embeddings(n_train=200, n_val=50)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LocalLogger(output_dir=tmpdir, dataset_id="test")
        step = ClassicalStepConfig(model_type="xgboost")
        hyperopt_search_classical(
            E_tr, y_tr, E_v, y_v,
            task="binary",
            classical_step=step,
            max_trials=8,
            logger=logger,
        )
        log_path = os.path.join(logger.get_hpo_dir(), "classical_hpo_trials.jsonl")
        with open(log_path) as f:
            records = [json.loads(line) for line in f]

        scores = [r["val_metric"]["roc_auc"] for r in records]
        lrs = [r["params"].get("learning_rate", 0) for r in records]
        unique_lrs = len(set(f"{lr:.8f}" for lr in lrs))
        print(f"Scores across trials: {[f'{s:.4f}' for s in scores]}")
        print(f"Unique LR values: {unique_lrs}/{len(lrs)}")

        assert unique_lrs > 1, "HPO should explore multiple learning_rate values"
        print("Test passed: HPO explores diverse configurations")


# =====================================================================
# Random data: HPO should NOT produce a strong model
# =====================================================================

def test_hpo_random_data_binary_no_leakage():
    """On random labels, HPO best score should stay near chance."""
    E_tr, y_tr, E_v, y_v = _make_random_binary_embeddings(seed=99)

    step = ClassicalStepConfig(model_type="xgboost")
    result = hyperopt_search_classical(
        E_tr, y_tr, E_v, y_v,
        task="binary",
        classical_step=step,
        max_trials=10,
        master_seed=99,
    )
    hpo_score = result["best_score"]
    print(f"Random-data HPO best ROC-AUC: {hpo_score:.4f}")
    assert hpo_score < 0.65, (
        f"HPO should not find signal in random labels; ROC-AUC={hpo_score:.4f}"
    )
    print("Test passed: no leakage with HPO on random binary labels")


def test_hpo_no_significant_improvement_on_random_data():
    """HPO on random data: best trial should not beat default significantly."""
    E_tr, y_tr, E_v, y_v = _make_random_binary_embeddings(seed=99)

    default_score = _train_with_defaults(
        E_tr, y_tr, E_v, y_v, task="binary", model_type="xgboost",
    )
    print(f"Random-data default ROC-AUC: {default_score:.4f}")

    step = ClassicalStepConfig(model_type="xgboost")
    result = hyperopt_search_classical(
        E_tr, y_tr, E_v, y_v,
        task="binary",
        classical_step=step,
        max_trials=10,
        master_seed=99,
    )
    hpo_score = result["best_score"]
    improvement = hpo_score - default_score
    print(f"Random-data HPO best ROC-AUC: {hpo_score:.4f}")
    print(f"Improvement over default: {improvement:+.4f}")

    assert improvement < 0.10, (
        f"HPO should not significantly improve on random data; "
        f"improvement={improvement:+.4f}"
    )
    print("Test passed: no significant HPO improvement on random data")


def test_hpo_random_data_regression_no_leakage():
    """LightGBM on random regression targets: RMSE should not be good."""
    rng = np.random.RandomState(99)
    n_train, n_val = 300, 80
    E = rng.randn(n_train + n_val, 64).astype(np.float64)
    y = rng.randn(n_train + n_val).astype(np.float64)

    step = ClassicalStepConfig(model_type="lightgbm")
    result = hyperopt_search_classical(
        E[:n_train], y[:n_train], E[n_train:], y[n_train:],
        task="regression",
        classical_step=step,
        max_trials=10,
        master_seed=99,
    )
    from sklearn.metrics import r2_score as r2
    from hybrid_embed.classical.model_zoo import build_model, train_classical_model

    resolved = resolve_from_result(result, step, "regression")
    model = build_model(resolved["class"], result["best_params"], "regression")
    trained = train_classical_model(
        model, E[:n_train], y[:n_train],
        E_val=E[n_train:], y_val=y[n_train:],
        supports_early_stopping=resolved["supports_early_stopping"],
    )
    preds = trained.predict(E[n_train:])
    r2_val = r2(y[n_train:], preds)
    print(f"Random-data LightGBM R^2: {r2_val:.4f}")
    assert r2_val < 0.15, (
        f"LightGBM should not learn from random targets; R^2={r2_val:.4f}"
    )
    print("Test passed: no leakage with HPO on random regression targets")


def resolve_from_result(result, step, task):
    """Helper: resolve step config for validation."""
    from hybrid_embed.classical.model_zoo import resolve_classical_step
    return resolve_classical_step(step, task)
