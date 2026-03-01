"""End-to-end integration tests (T19).

These tests run the full pipeline on synthetic datasets with
minimal budgets. They are slow (minutes) and are marked with
``@pytest.mark.slow`` so they can be excluded via
``pytest -m 'not slow'``.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from hybrid_embed.config import (
    BudgetConfig,
    ClassicalStepConfig,
    EmbeddingStepConfig,
    RunConfig,
    RunResult,
    TaskConfig,
)
from hybrid_embed.eval.runner import HybridTabularModel, run_experiment


# -----------------------------------------------------------------------
# Synthetic data generators
# -----------------------------------------------------------------------

def _make_binary_data(n=300, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "num_0": rng.randn(n),
        "num_1": rng.randn(n),
        "num_2": rng.randn(n),
        "num_3": rng.randn(n),
        "cat_0": rng.choice(["A", "B", "C"], size=n),
        "cat_1": rng.choice(["X", "Y"], size=n),
    })
    y = (df["num_0"] + 0.8 * df["num_1"] > 0).astype(int).values
    return df, y


def _make_multiclass_data(n=400, n_classes=4, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "num_0": rng.randn(n),
        "num_1": rng.randn(n),
        "num_2": rng.randn(n),
        "num_3": rng.randn(n),
        "cat_0": rng.choice(["A", "B", "C", "D"], size=n),
    })
    score = df["num_0"] + 0.5 * df["num_1"]
    boundaries = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
    y = np.digitize(score, boundaries).astype(int)
    return df, y


def _make_regression_data(n=300, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "num_0": rng.randn(n),
        "num_1": rng.randn(n),
        "num_2": rng.randn(n),
        "num_3": rng.randn(n),
        "cat_0": rng.choice(["A", "B", "C"], size=n),
    })
    y = (3.0 * df["num_0"].values
         + 1.5 * df["num_1"].values
         + rng.randn(n) * 0.5)
    return df, y


# -----------------------------------------------------------------------
# Shared config helpers
# -----------------------------------------------------------------------

_EMBED_TRIALS = 3
_EMBED_EPOCHS = 5
_EMBED_PATIENCE = 3
_CLASSICAL_TRIALS = 5
_N_FOLDS = 2


def _budget():
    return BudgetConfig(
        embed_hpo_max_trials=_EMBED_TRIALS,
        embed_max_epochs=_EMBED_EPOCHS,
        embed_patience=_EMBED_PATIENCE,
        classical_hpo_max_trials=_CLASSICAL_TRIALS,
    )


def _run_cfg(output_dir, dataset_id="integration"):
    return RunConfig(
        n_folds=_N_FOLDS,
        master_seed=42,
        device="cpu",
        output_dir=output_dir,
        dataset_id=dataset_id,
        save_predictions=True,
        save_embeddings=True,
    )


def _count_jsonl_lines(path: str) -> int:
    with open(path) as fh:
        return sum(1 for _ in fh)


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

@pytest.mark.slow
def test_end_to_end_binary_classification():
    """Full pipeline on synthetic binary classification data."""
    tmpdir = tempfile.mkdtemp()
    df, y = _make_binary_data()

    result = run_experiment(
        df, y,
        TaskConfig(task="binary"),
        EmbeddingStepConfig(model_type="mlp"),
        ClassicalStepConfig(model_type="xgboost"),
        _budget(),
        _run_cfg(tmpdir, "binary_integ"),
    )

    assert isinstance(result, RunResult)
    assert result.task == "binary"
    assert result.n_folds == _N_FOLDS

    assert "roc_auc" in result.mean_metrics
    roc_auc = result.mean_metrics["roc_auc"]
    print(f"Binary ROC-AUC: {roc_auc:.4f}")
    assert roc_auc > 0.5, f"Expected ROC-AUC > 0.5, got {roc_auc:.4f}"

    # Artifact structure
    ds_dir = os.path.join(tmpdir, "binary_integ")
    run_dirs = [d for d in os.listdir(ds_dir)
                if d.startswith("run_") and os.path.isdir(os.path.join(ds_dir, d))]
    assert len(run_dirs) >= 1
    run_dir = os.path.join(ds_dir, run_dirs[0])

    hpo_dir = os.path.join(run_dir, "hpo")
    embed_trials_path = os.path.join(hpo_dir, "embedding_hpo_trials.jsonl")
    classical_trials_path = os.path.join(hpo_dir, "classical_hpo_trials.jsonl")
    assert os.path.isfile(embed_trials_path)
    assert os.path.isfile(classical_trials_path)
    assert _count_jsonl_lines(embed_trials_path) == _EMBED_TRIALS
    assert _count_jsonl_lines(classical_trials_path) == _CLASSICAL_TRIALS

    for fold_idx in range(_N_FOLDS):
        fold_dir = os.path.join(run_dir, "folds", f"fold_{fold_idx}")
        assert os.path.isdir(fold_dir), f"Missing fold dir: {fold_dir}"

    summary_path = os.path.join(run_dir, "summary_metrics.json")
    assert os.path.isfile(summary_path)
    with open(summary_path) as fh:
        summary = json.load(fh)
    assert "mean" in summary or "roc_auc" in str(summary)
    print("Binary integration test passed")


@pytest.mark.slow
def test_end_to_end_multiclass_classification():
    """Full pipeline on synthetic multiclass (4-class) data."""
    tmpdir = tempfile.mkdtemp()
    df, y = _make_multiclass_data(n_classes=4)

    result = run_experiment(
        df, y,
        TaskConfig(task="multiclass"),
        EmbeddingStepConfig(model_type="mlp"),
        ClassicalStepConfig(model_type="xgboost"),
        _budget(),
        _run_cfg(tmpdir, "multi_integ"),
    )

    assert isinstance(result, RunResult)
    assert "accuracy" in result.mean_metrics
    accuracy = result.mean_metrics["accuracy"]
    print(f"Multiclass accuracy: {accuracy:.4f}")
    assert accuracy > 0.25, f"Expected accuracy > 0.25 (4 classes), got {accuracy:.4f}"
    print("Multiclass integration test passed")


@pytest.mark.slow
def test_end_to_end_regression():
    """Full pipeline on synthetic regression data."""
    tmpdir = tempfile.mkdtemp()
    df, y = _make_regression_data()

    result = run_experiment(
        df, y,
        TaskConfig(task="regression"),
        EmbeddingStepConfig(model_type="mlp"),
        ClassicalStepConfig(model_type="xgboost"),
        _budget(),
        _run_cfg(tmpdir, "reg_integ"),
    )

    assert isinstance(result, RunResult)
    assert "rmse" in result.mean_metrics
    assert "r_squared" in result.mean_metrics
    r2 = result.mean_metrics["r_squared"]
    print(f"Regression R-squared: {r2:.4f}")
    assert r2 > 0.0, f"Expected R-squared > 0 (better than mean), got {r2:.4f}"
    print("Regression integration test passed")


@pytest.mark.slow
def test_end_to_end_tab_transformer():
    """Full pipeline using TabTransformer (pytorch-tabular adapter)."""
    tmpdir = tempfile.mkdtemp()
    df, y = _make_binary_data()

    result = run_experiment(
        df, y,
        TaskConfig(task="binary"),
        EmbeddingStepConfig(model_type="tab_transformer"),
        ClassicalStepConfig(model_type="xgboost"),
        BudgetConfig(
            embed_hpo_max_trials=2,
            embed_max_epochs=3,
            embed_patience=2,
            classical_hpo_max_trials=3,
        ),
        _run_cfg(tmpdir, "tabtrans_integ"),
    )

    assert isinstance(result, RunResult)
    assert "roc_auc" in result.mean_metrics
    roc_auc = result.mean_metrics["roc_auc"]
    print(f"TabTransformer ROC-AUC: {roc_auc:.4f}")
    assert np.isfinite(roc_auc)
    print("TabTransformer integration test passed")


@pytest.mark.slow
def test_reproducibility():
    """Two runs with the same seed produce identical fold metrics."""
    results = []
    for run_idx in range(2):
        tmpdir = tempfile.mkdtemp()
        df, y = _make_binary_data()
        result = run_experiment(
            df, y,
            TaskConfig(task="binary"),
            EmbeddingStepConfig(model_type="mlp"),
            ClassicalStepConfig(model_type="xgboost"),
            _budget(),
            _run_cfg(tmpdir, f"repro_{run_idx}"),
        )
        results.append(result)

    for key in results[0].mean_metrics:
        v0 = results[0].mean_metrics[key]
        v1 = results[1].mean_metrics[key]
        print(f"  {key}: run0={v0:.6f}  run1={v1:.6f}")
        assert np.isclose(v0, v1, atol=1e-6), (
            f"Metric '{key}' differs between runs: {v0} vs {v1}"
        )
    print("Reproducibility test passed")


def test_embeddings_only_constraint():
    """Classical models receive embeddings, not raw features.

    Runs with small data and verifies the embedding shape
    matches the classical model's input dimensionality.
    """
    tmpdir = tempfile.mkdtemp()
    df, y = _make_binary_data(n=150)

    model = HybridTabularModel(
        task="binary",
        embedding=EmbeddingStepConfig(model_type="mlp"),
        classical=ClassicalStepConfig(model_type="xgboost"),
        budget_config=BudgetConfig(
            embed_hpo_max_trials=2,
            embed_max_epochs=3,
            embed_patience=2,
            classical_hpo_max_trials=3,
        ),
        run_config=RunConfig(
            master_seed=42,
            device="cpu",
            output_dir=tmpdir,
            dataset_id="constraint_test",
        ),
    )
    model.fit(df, y)

    embeddings = model._get_embeddings(df)
    n_raw_features = df.shape[1]
    embed_dim = embeddings.shape[1]

    print(f"Raw feature count: {n_raw_features}")
    print(f"Embedding dim: {embed_dim}")
    assert embeddings.shape[0] == len(df)

    preds = model._classical_model.predict(embeddings)
    assert preds.shape == (len(df),), (
        f"Classical model should accept embeddings; got shape {preds.shape}"
    )
    print("Embeddings-only constraint test passed")
