"""Tests for hybrid_embed.eval.runner: end-to-end smoke tests.

These verify the pipeline runs on small synthetic data with minimal
budgets, not that the models are accurate.
"""

import os
import tempfile

import numpy as np
import pandas as pd

from hybrid_embed.config import (
    BudgetConfig,
    ClassicalStepConfig,
    EmbeddingStepConfig,
    FoldResult,
    RunConfig,
    RunResult,
    TaskConfig,
)
from hybrid_embed.eval.runner import HybridTabularModel, run_experiment


def _make_synthetic_data(n=150, seed=42):
    """Generate a small synthetic dataset with 4 numeric + 2 categorical columns."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "num_0": rng.randn(n),
        "num_1": rng.randn(n),
        "num_2": rng.randn(n),
        "num_3": rng.randn(n),
        "cat_0": rng.choice(["A", "B", "C"], size=n),
        "cat_1": rng.choice(["X", "Y"], size=n),
    })
    y = (df["num_0"] + df["num_1"] > 0).astype(int).values
    return df, y


def _minimal_configs(output_dir):
    """Return configs with minimal budgets for fast smoke testing."""
    task_config = TaskConfig(task="binary")
    embedding_step = EmbeddingStepConfig(model_type="mlp")
    classical_step = ClassicalStepConfig(model_type="xgboost")
    budget_config = BudgetConfig(
        embed_hpo_max_trials=2,
        embed_max_epochs=3,
        embed_patience=2,
        classical_hpo_max_trials=5,
    )
    run_config = RunConfig(
        n_folds=2,
        master_seed=42,
        device="cpu",
        output_dir=output_dir,
        dataset_id="smoke_test",
        save_predictions=True,
        save_embeddings=True,
    )
    return task_config, embedding_step, classical_step, budget_config, run_config


def test_run_experiment_completes():
    """run_experiment should complete on synthetic data and return RunResult."""
    tmpdir = tempfile.mkdtemp()
    df, y = _make_synthetic_data()
    task_config, embed, classical, budget, run_cfg = _minimal_configs(tmpdir)

    result = run_experiment(df, y, task_config, embed, classical, budget, run_cfg)

    print(f"Result type: {type(result).__name__}")
    print(f"n_folds: {result.n_folds}")
    print(f"task: {result.task}")
    print(f"total_time: {result.total_time_seconds:.2f}s")

    assert isinstance(result, RunResult)
    assert result.n_folds == 2
    assert result.task == "binary"
    assert result.total_time_seconds > 0
    print("Test passed: run_experiment completes successfully")


def test_run_experiment_metrics_populated():
    """RunResult should have mean_metrics and std_metrics with expected keys."""
    tmpdir = tempfile.mkdtemp()
    df, y = _make_synthetic_data()
    task_config, embed, classical, budget, run_cfg = _minimal_configs(tmpdir)

    result = run_experiment(df, y, task_config, embed, classical, budget, run_cfg)

    print(f"mean_metrics keys: {sorted(result.mean_metrics.keys())}")
    print(f"std_metrics keys: {sorted(result.std_metrics.keys())}")
    for k, v in result.mean_metrics.items():
        print(f"  mean {k}: {v:.4f}")
    for k, v in result.std_metrics.items():
        print(f"  std  {k}: {v:.4f}")

    expected_keys = {"roc_auc", "log_loss", "pr_auc"}
    assert expected_keys.issubset(result.mean_metrics.keys()), (
        f"Missing keys: {expected_keys - result.mean_metrics.keys()}"
    )
    assert expected_keys.issubset(result.std_metrics.keys()), (
        f"Missing keys: {expected_keys - result.std_metrics.keys()}"
    )

    for key in expected_keys:
        assert np.isfinite(result.mean_metrics[key]), f"mean {key} is not finite"
        assert np.isfinite(result.std_metrics[key]), f"std {key} is not finite"

    print("Test passed: metrics are populated with expected keys")


def test_run_experiment_fold_results():
    """fold_results should have correct length and each FoldResult has metrics."""
    tmpdir = tempfile.mkdtemp()
    df, y = _make_synthetic_data()
    task_config, embed, classical, budget, run_cfg = _minimal_configs(tmpdir)

    result = run_experiment(df, y, task_config, embed, classical, budget, run_cfg)

    print(f"Number of fold_results: {len(result.fold_results)}")
    assert len(result.fold_results) == 2

    for fr in result.fold_results:
        print(f"  Fold {fr.fold_index}: metrics={fr.metrics}, "
              f"embedding_dim={fr.embedding_dim}")
        assert isinstance(fr, FoldResult)
        assert isinstance(fr.metrics, dict)
        assert len(fr.metrics) > 0
        assert fr.embedding_dim > 0

    fold_indices = [fr.fold_index for fr in result.fold_results]
    assert sorted(fold_indices) == [0, 1]
    print("Test passed: fold_results have correct structure")


def test_run_experiment_artifacts_created():
    """Run directory should contain expected subdirectories and files."""
    tmpdir = tempfile.mkdtemp()
    df, y = _make_synthetic_data()
    task_config, embed, classical, budget, run_cfg = _minimal_configs(tmpdir)

    result = run_experiment(df, y, task_config, embed, classical, budget, run_cfg)

    smoke_dir = os.path.join(tmpdir, "smoke_test")
    assert os.path.isdir(smoke_dir), f"Dataset dir missing: {smoke_dir}"

    run_dirs = [d for d in os.listdir(smoke_dir)
                if d.startswith("run_") and os.path.isdir(os.path.join(smoke_dir, d))]
    assert len(run_dirs) >= 1, "No run directories found"
    run_dir = os.path.join(smoke_dir, run_dirs[0])
    print(f"Run directory: {run_dir}")

    assert os.path.isdir(os.path.join(run_dir, "hpo"))
    assert os.path.isdir(os.path.join(run_dir, "folds"))

    hpo_dir = os.path.join(run_dir, "hpo")
    hpo_files = os.listdir(hpo_dir)
    print(f"HPO files: {hpo_files}")
    assert "best_embedding_config.json" in hpo_files
    assert "best_classical_config.json" in hpo_files

    assert os.path.isfile(os.path.join(run_dir, "config.yaml"))
    assert os.path.isfile(os.path.join(run_dir, "schema.json"))
    assert os.path.isfile(os.path.join(run_dir, "validation_report.json"))
    assert os.path.isfile(os.path.join(run_dir, "summary_metrics.json"))

    for fold_idx in range(2):
        fold_dir = os.path.join(run_dir, "folds", f"fold_{fold_idx}")
        assert os.path.isdir(fold_dir), f"Fold dir missing: {fold_dir}"
        fold_files = os.listdir(fold_dir)
        print(f"  Fold {fold_idx} files: {fold_files}")
        assert "metrics.json" in fold_files
        assert "embedding_ckpt.pt" in fold_files
        assert "classical_model.pkl" in fold_files

    print("Test passed: all expected artifacts created")


def test_hybrid_tabular_model_fit_predict():
    """HybridTabularModel should fit and produce predictions with correct shape."""
    tmpdir = tempfile.mkdtemp()
    df, y = _make_synthetic_data()

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
            dataset_id="wrapper_test",
        ),
    )

    model.fit(df, y)
    preds = model.predict(df)

    print(f"Predictions shape: {preds.shape}")
    print(f"Unique predictions: {np.unique(preds)}")
    assert preds.shape == (len(df),)
    print("Test passed: HybridTabularModel fit and predict work correctly")


def test_hybrid_tabular_model_predict_proba():
    """HybridTabularModel.predict_proba should return correct shape for binary."""
    tmpdir = tempfile.mkdtemp()
    df, y = _make_synthetic_data()

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
            dataset_id="wrapper_test",
        ),
    )

    model.fit(df, y)
    proba = model.predict_proba(df)

    print(f"Probabilities shape: {proba.shape}")
    print(f"Sample probabilities: {proba[:5]}")

    assert proba.shape == (len(df), 2), f"Expected ({len(df)}, 2), got {proba.shape}"

    assert np.all(proba >= 0) and np.all(proba <= 1), "Probabilities out of range"
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "Probabilities don't sum to 1"

    print("Test passed: predict_proba returns valid probabilities")


def _make_regression_data(n=200, seed=42):
    """Generate regression data with a large-scale target (mean ~100)."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "num_0": rng.randn(n),
        "num_1": rng.randn(n),
        "num_2": rng.randn(n),
        "num_3": rng.randn(n),
        "cat_0": rng.choice(["A", "B", "C"], size=n),
    })
    y = 100.0 + 5.0 * df["num_0"].values + 3.0 * df["num_1"].values + rng.randn(n) * 0.5
    return df, y


def _make_string_label_data(n=150, seed=42):
    """Generate binary data with string labels instead of 0/1."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "num_0": rng.randn(n),
        "num_1": rng.randn(n),
        "num_2": rng.randn(n),
        "num_3": rng.randn(n),
        "cat_0": rng.choice(["A", "B", "C"], size=n),
    })
    raw = (df["num_0"] + df["num_1"] > 0).astype(int)
    y = np.where(raw == 1, "positive", "negative")
    return df, y


def test_predict_regression_original_scale():
    """predict() on regression must return values in the original target scale."""
    tmpdir = tempfile.mkdtemp()
    df, y = _make_regression_data()

    model = HybridTabularModel(
        task="regression",
        embedding=EmbeddingStepConfig(model_type="mlp"),
        classical=ClassicalStepConfig(model_type="xgboost"),
        budget_config=BudgetConfig(
            embed_hpo_max_trials=2,
            embed_max_epochs=3,
            embed_patience=2,
            classical_hpo_max_trials=3,
        ),
        run_config=RunConfig(
            master_seed=42, device="cpu",
            output_dir=tmpdir, dataset_id="reg_scale_test",
        ),
    )
    model.fit(df, y)
    preds = model.predict(df)

    print(f"Original y mean: {y.mean():.2f}, std: {y.std():.2f}")
    print(f"Predicted  mean: {preds.mean():.2f}, std: {preds.std():.2f}")

    assert preds.mean() > 50, (
        f"Predictions should be near original scale (~100), got mean={preds.mean():.2f}"
    )
    assert abs(preds.mean() - y.mean()) < 20, (
        f"Predicted mean too far from original: {preds.mean():.2f} vs {y.mean():.2f}"
    )
    print("Test passed: regression predictions are in original scale")


def test_predict_classification_original_labels():
    """predict() on classification with string labels returns original label values."""
    tmpdir = tempfile.mkdtemp()
    df, y = _make_string_label_data()

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
            master_seed=42, device="cpu",
            output_dir=tmpdir, dataset_id="label_test",
        ),
    )
    model.fit(df, y)
    preds = model.predict(df)

    unique_preds = set(preds)
    print(f"Original labels: {set(y)}")
    print(f"Predicted labels: {unique_preds}")

    assert unique_preds.issubset({"positive", "negative"}), (
        f"Predictions should be original string labels, got {unique_preds}"
    )
    print("Test passed: classification predictions use original labels")
