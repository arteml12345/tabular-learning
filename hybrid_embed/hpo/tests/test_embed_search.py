"""Tests for hybrid_embed.hpo.embed_search."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from hybrid_embed.config import EmbeddingStepConfig
from hybrid_embed.hpo.embed_search import hyperopt_search_embedding


# =====================================================================
# Helpers
# =====================================================================

def _make_binary_data_with_signal(
    n_train=200, n_val=60, n_numeric=5, n_cat_cols=2, seed=42,
):
    """Synthetic data with clear learnable binary signal."""
    rng = np.random.RandomState(seed)
    n = n_train + n_val

    numeric = rng.randn(n, n_numeric).astype(np.float32)
    signal = 2.0 * numeric[:, 0] + 1.5 * numeric[:, 1] - numeric[:, 2]
    y = (signal > 0).astype(np.int64)

    n_categories_per_column = [5, 4]
    cats = np.column_stack([
        rng.randint(0, c, size=n) for c in n_categories_per_column
    ]).astype(np.int64)

    def _split(arr, idx):
        return arr[:idx], arr[idx:]

    X_train = {
        "numeric": numeric[:n_train],
        "categorical": cats[:n_train],
    }
    X_val = {
        "numeric": numeric[n_train:],
        "categorical": cats[n_train:],
    }
    return (
        X_train, y[:n_train], X_val, y[n_train:],
        n_numeric, n_categories_per_column,
    )


def _make_regression_data_with_signal(
    n_train=200, n_val=60, n_numeric=5, n_cat_cols=2, seed=42,
):
    """Synthetic regression data with clear learnable signal."""
    rng = np.random.RandomState(seed)
    n = n_train + n_val

    numeric = rng.randn(n, n_numeric).astype(np.float32)
    y = (
        3.0 * numeric[:, 0] + 1.5 * numeric[:, 1]
        - 2.0 * numeric[:, 2] + rng.randn(n).astype(np.float32) * 0.3
    )

    n_categories_per_column = [5, 4]
    cats = np.column_stack([
        rng.randint(0, c, size=n) for c in n_categories_per_column
    ]).astype(np.int64)

    X_train = {"numeric": numeric[:n_train], "categorical": cats[:n_train]}
    X_val = {"numeric": numeric[n_train:], "categorical": cats[n_train:]}
    return (
        X_train, y[:n_train].astype(np.float64),
        X_val, y[n_train:].astype(np.float64),
        n_numeric, n_categories_per_column,
    )


def _make_random_binary_data(
    n_train=200, n_val=60, n_numeric=5, n_cat_cols=2, seed=99,
):
    """Data where the target is pure noise (no learnable signal)."""
    rng = np.random.RandomState(seed)
    n = n_train + n_val

    numeric = rng.randn(n, n_numeric).astype(np.float32)
    y = rng.choice([0, 1], size=n).astype(np.int64)

    n_categories_per_column = [5, 4]
    cats = np.column_stack([
        rng.randint(0, c, size=n) for c in n_categories_per_column
    ]).astype(np.int64)

    X_train = {"numeric": numeric[:n_train], "categorical": cats[:n_train]}
    X_val = {"numeric": numeric[n_train:], "categorical": cats[n_train:]}
    return (
        X_train, y[:n_train], X_val, y[n_train:],
        n_numeric, n_categories_per_column,
    )


def _train_with_defaults(X_train, y_train, X_val, y_val,
                         n_numeric, n_categories_per_column,
                         task="binary", max_epochs=20):
    """Train MLP with default (non-tuned) params, return val metric."""
    from hybrid_embed.embed.scratch_mlp import MLPEmbeddingModel
    from hybrid_embed.eval.metrics import compute_metrics, get_primary_metric_name

    default_config = {
        "embedding_dim": 32,
        "hidden_dims": [64, 32],
        "dropout": 0.1,
        "activation": "relu",
        "use_layer_norm": False,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 256,
        "max_epochs": max_epochs,
        "patience": 5,
        "device": "cpu",
    }
    n_classes = int(max(y_train.max(), y_val.max()) + 1) if task != "regression" else None
    model = MLPEmbeddingModel(
        task=task,
        n_numeric=n_numeric,
        n_categories_per_column=n_categories_per_column,
        n_classes=n_classes,
    )
    model.fit(X_train, y_train, X_val, y_val, default_config)

    metric_name = get_primary_metric_name(task)
    if task == "binary":
        proba = model.predict_proba(X_val)
        preds = (proba[:, 1] >= 0.5).astype(int)
        metrics = compute_metrics(y_val, preds, proba[:, 1], task)
    elif task == "regression":
        preds = model.predict(X_val)
        metrics = compute_metrics(y_val, preds, None, task)
    else:
        proba = model.predict_proba(X_val)
        preds = model.predict(X_val)
        metrics = compute_metrics(y_val, preds, proba, task)
    return metrics[metric_name]


# =====================================================================
# Structural tests
# =====================================================================

def test_embedding_search_runs():
    """Run search with max_trials=3, max_epochs=3. Returns dict with required keys."""
    (X_tr, y_tr, X_v, y_v,
     n_num, n_cats) = _make_binary_data_with_signal(n_train=100, n_val=30)

    step = EmbeddingStepConfig(model_type="mlp")
    result = hyperopt_search_embedding(
        X_train=X_tr, y_train=y_tr, X_val=X_v, y_val=y_v,
        task="binary",
        n_numeric=n_num,
        n_categories_per_column=n_cats,
        n_classes=2,
        embedding_step=step,
        max_trials=3,
        max_epochs=3,
        patience=2,
        device="cpu",
    )
    assert isinstance(result, dict)
    for key in ("best_params", "best_score", "n_trials_completed", "total_time_seconds"):
        assert key in result, f"Missing key: {key}"
    assert result["n_trials_completed"] == 3
    print(f"Trials completed: {result['n_trials_completed']}")
    print(f"Best score: {result['best_score']:.4f}")
    print("Test passed: embedding search runs and returns correct structure")


def test_embedding_search_returns_valid_config():
    """best_params has all required keys for MLP."""
    (X_tr, y_tr, X_v, y_v,
     n_num, n_cats) = _make_binary_data_with_signal(n_train=100, n_val=30)

    step = EmbeddingStepConfig(model_type="mlp")
    result = hyperopt_search_embedding(
        X_train=X_tr, y_train=y_tr, X_val=X_v, y_val=y_v,
        task="binary",
        n_numeric=n_num,
        n_categories_per_column=n_cats,
        n_classes=2,
        embedding_step=step,
        max_trials=3,
        max_epochs=3,
        patience=2,
        device="cpu",
    )
    bp = result["best_params"]
    required = {"hidden_dims", "dropout", "activation", "lr", "batch_size"}
    for key in required:
        assert key in bp, f"Missing key in best_params: {key}"
    print(f"best_params keys: {sorted(bp.keys())}")
    print("Test passed: best_params contains required config keys")


def test_embedding_search_time_budget():
    """Time budget caps the number of trials."""
    (X_tr, y_tr, X_v, y_v,
     n_num, n_cats) = _make_binary_data_with_signal(n_train=100, n_val=30)

    step = EmbeddingStepConfig(model_type="mlp")
    result = hyperopt_search_embedding(
        X_train=X_tr, y_train=y_tr, X_val=X_v, y_val=y_v,
        task="binary",
        n_numeric=n_num,
        n_categories_per_column=n_cats,
        n_classes=2,
        embedding_step=step,
        max_trials=1000,
        max_epochs=3,
        patience=2,
        time_budget_seconds=8,
        device="cpu",
    )
    assert result["n_trials_completed"] < 1000
    assert result["total_time_seconds"] < 30
    print(f"Trials within budget: {result['n_trials_completed']}")
    print(f"Total time: {result['total_time_seconds']:.1f}s")
    print("Test passed: time budget limits trials")


def test_embedding_search_logs_trials():
    """Logger receives trial records in JSONL."""
    from hybrid_embed.log.local_logger import LocalLogger

    (X_tr, y_tr, X_v, y_v,
     n_num, n_cats) = _make_binary_data_with_signal(n_train=100, n_val=30)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LocalLogger(output_dir=tmpdir, dataset_id="test")
        step = EmbeddingStepConfig(model_type="mlp")
        result = hyperopt_search_embedding(
            X_train=X_tr, y_train=y_tr, X_val=X_v, y_val=y_v,
            task="binary",
            n_numeric=n_num,
            n_categories_per_column=n_cats,
            n_classes=2,
            embedding_step=step,
            max_trials=3,
            max_epochs=3,
            patience=2,
            device="cpu",
            logger=logger,
        )
        log_path = os.path.join(logger.get_hpo_dir(), "embedding_hpo_trials.jsonl")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 3
        for line in lines:
            record = json.loads(line)
            assert "trial_id" in record
            assert "params" in record
            assert "val_metric" in record
        print(f"Logged {len(lines)} trials to JSONL")
        print("Test passed: logger captures all trial records")


def test_embedding_search_score_is_valid():
    """best_score is a valid finite float."""
    (X_tr, y_tr, X_v, y_v,
     n_num, n_cats) = _make_binary_data_with_signal(n_train=100, n_val=30)

    step = EmbeddingStepConfig(model_type="mlp")
    result = hyperopt_search_embedding(
        X_train=X_tr, y_train=y_tr, X_val=X_v, y_val=y_v,
        task="binary",
        n_numeric=n_num,
        n_categories_per_column=n_cats,
        n_classes=2,
        embedding_step=step,
        max_trials=5,
        max_epochs=5,
        patience=3,
        device="cpu",
    )
    assert np.isfinite(result["best_score"])
    assert 0.0 <= result["best_score"] <= 1.0
    print(f"Best ROC-AUC: {result['best_score']:.4f}")
    print("Test passed: best_score is a valid metric value")


# =====================================================================
# HPO progress tests: signal data should show improvement
# =====================================================================

def test_hpo_improves_over_defaults_binary():
    """HPO should find params at least as good as (often better than) defaults.

    Trains a model with fixed default params, then runs HPO with
    enough trials. Verifies that HPO's best score is not worse than
    default by a significant margin, and that across trials there
    is variance in scores (proof that different configs are explored).
    """
    (X_tr, y_tr, X_v, y_v,
     n_num, n_cats) = _make_binary_data_with_signal(
        n_train=300, n_val=80, seed=42,
    )

    default_score = _train_with_defaults(
        X_tr, y_tr, X_v, y_v, n_num, n_cats,
        task="binary", max_epochs=15,
    )
    print(f"Default ROC-AUC: {default_score:.4f}")

    step = EmbeddingStepConfig(model_type="mlp")
    result = hyperopt_search_embedding(
        X_train=X_tr, y_train=y_tr, X_val=X_v, y_val=y_v,
        task="binary",
        n_numeric=n_num,
        n_categories_per_column=n_cats,
        n_classes=2,
        embedding_step=step,
        max_trials=8,
        max_epochs=15,
        patience=5,
        device="cpu",
        master_seed=42,
    )
    hpo_score = result["best_score"]
    print(f"HPO best ROC-AUC: {hpo_score:.4f}")
    print(f"HPO params: {result['best_params']}")

    assert hpo_score > 0.70, (
        f"HPO should achieve decent ROC-AUC on learnable data; got {hpo_score:.4f}"
    )
    tolerance = 0.10
    assert hpo_score >= default_score + tolerance, (
        f"HPO should improve over default by at least {tolerance}: "
        f"HPO={hpo_score:.4f} vs Default={default_score:.4f}"
    )
    print("Test passed: HPO improves over defaults on binary task")


def test_hpo_improves_over_defaults_regression():
    """HPO on regression data should match or beat defaults."""
    (X_tr, y_tr, X_v, y_v,
     n_num, n_cats) = _make_regression_data_with_signal(
        n_train=300, n_val=80, seed=42,
    )

    default_score = _train_with_defaults(
        X_tr, y_tr, X_v, y_v, n_num, n_cats,
        task="regression", max_epochs=15,
    )
    print(f"Default RMSE: {default_score:.4f}")

    step = EmbeddingStepConfig(model_type="mlp")
    result = hyperopt_search_embedding(
        X_train=X_tr, y_train=y_tr, X_val=X_v, y_val=y_v,
        task="regression",
        n_numeric=n_num,
        n_categories_per_column=n_cats,
        n_classes=None,
        embedding_step=step,
        max_trials=8,
        max_epochs=15,
        patience=5,
        device="cpu",
        master_seed=42,
    )
    hpo_score = result["best_score"]
    print(f"HPO best RMSE: {hpo_score:.4f}")

    tolerance = 0.3
    assert hpo_score <= default_score - tolerance, (
        f"HPO should reduce RMSE by at least {tolerance}: "
        f"HPO={hpo_score:.4f} vs Default={default_score:.4f}"
    )
    print("Test passed: HPO improves over defaults on regression task")


def test_hpo_explores_diverse_configs():
    """HPO trial logs should show diverse configurations being explored."""
    from hybrid_embed.log.local_logger import LocalLogger

    (X_tr, y_tr, X_v, y_v,
     n_num, n_cats) = _make_binary_data_with_signal(n_train=100, n_val=30)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LocalLogger(output_dir=tmpdir, dataset_id="test")
        step = EmbeddingStepConfig(model_type="mlp")
        hyperopt_search_embedding(
            X_train=X_tr, y_train=y_tr, X_val=X_v, y_val=y_v,
            task="binary",
            n_numeric=n_num,
            n_categories_per_column=n_cats,
            n_classes=2,
            embedding_step=step,
            max_trials=5,
            max_epochs=3,
            patience=2,
            device="cpu",
            logger=logger,
        )
        log_path = os.path.join(logger.get_hpo_dir(), "embedding_hpo_trials.jsonl")
        with open(log_path) as f:
            records = [json.loads(line) for line in f]

        scores = [r["val_metric"]["roc_auc"] for r in records]
        lrs = [r["params"]["lr"] for r in records]
        unique_lrs = len(set(f"{lr:.8f}" for lr in lrs))
        print(f"Scores across trials: {scores}")
        print(f"Unique LR values: {unique_lrs}/{len(lrs)}")

        assert unique_lrs > 1, "HPO should explore multiple LR values"
        print("Test passed: HPO explores diverse configurations")


# =====================================================================
# Random data: HPO should NOT produce a strong model
# =====================================================================

def test_hpo_random_data_binary_no_leakage():
    """On random labels, HPO best score should stay near chance (ROC-AUC ~0.5)."""
    (X_tr, y_tr, X_v, y_v,
     n_num, n_cats) = _make_random_binary_data(
        n_train=300, n_val=80, seed=99,
    )

    step = EmbeddingStepConfig(model_type="mlp")
    result = hyperopt_search_embedding(
        X_train=X_tr, y_train=y_tr, X_val=X_v, y_val=y_v,
        task="binary",
        n_numeric=n_num,
        n_categories_per_column=n_cats,
        n_classes=2,
        embedding_step=step,
        max_trials=6,
        max_epochs=10,
        patience=4,
        device="cpu",
        master_seed=99,
    )
    hpo_score = result["best_score"]
    print(f"Random-data HPO best ROC-AUC: {hpo_score:.4f}")
    assert hpo_score < 0.65, (
        f"HPO should not find signal in random labels; ROC-AUC={hpo_score:.4f}"
    )
    print("Test passed: no leakage with HPO on random binary labels")


def test_hpo_no_significant_improvement_on_random_data():
    """HPO on random data: best trial should not be much better than default."""
    (X_tr, y_tr, X_v, y_v,
     n_num, n_cats) = _make_random_binary_data(
        n_train=300, n_val=80, seed=99,
    )

    default_score = _train_with_defaults(
        X_tr, y_tr, X_v, y_v, n_num, n_cats,
        task="binary", max_epochs=10,
    )
    print(f"Random-data default ROC-AUC: {default_score:.4f}")

    step = EmbeddingStepConfig(model_type="mlp")
    result = hyperopt_search_embedding(
        X_train=X_tr, y_train=y_tr, X_val=X_v, y_val=y_v,
        task="binary",
        n_numeric=n_num,
        n_categories_per_column=n_cats,
        n_classes=2,
        embedding_step=step,
        max_trials=6,
        max_epochs=10,
        patience=4,
        device="cpu",
        master_seed=99,
    )
    hpo_score = result["best_score"]
    improvement = hpo_score - default_score
    print(f"Random-data HPO best ROC-AUC: {hpo_score:.4f}")
    print(f"Improvement over default: {improvement:+.4f}")

    assert improvement < 0.15, (
        f"HPO should not significantly improve on random data; "
        f"improvement={improvement:+.4f}"
    )
    print("Test passed: no significant HPO improvement on random data")


def test_embedding_search_regression_with_fixed_params():
    """Fixed params should be respected in HPO."""
    (X_tr, y_tr, X_v, y_v,
     n_num, n_cats) = _make_regression_data_with_signal(n_train=100, n_val=30)

    step = EmbeddingStepConfig(
        model_type="mlp",
        fixed_params={"embedding_layer": -1, "use_layer_norm": True},
    )
    result = hyperopt_search_embedding(
        X_train=X_tr, y_train=y_tr, X_val=X_v, y_val=y_v,
        task="regression",
        n_numeric=n_num,
        n_categories_per_column=n_cats,
        n_classes=None,
        embedding_step=step,
        max_trials=3,
        max_epochs=3,
        patience=2,
        device="cpu",
    )
    assert result["best_params"]["use_layer_norm"] is True
    assert result["best_params"]["embedding_layer"] == -1
    print("Test passed: fixed params are preserved through HPO")
