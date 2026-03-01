"""Tests for hybrid_embed.embed.scratch_mlp: TabularMLP and MLPEmbeddingModel."""

import os
import tempfile

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score

from hybrid_embed.embed.scratch_mlp import TabularMLP, MLPEmbeddingModel
from hybrid_embed.utils import seed_everything


def _make_synthetic_data(n_samples=200, seed=42):
    """Generate synthetic preprocessed data for testing.

    5 numeric features, 3 categorical features with vocab sizes [10, 5, 8].
    Binary classification target.
    """
    rng = np.random.RandomState(seed)
    X = {
        "numeric": rng.randn(n_samples, 5).astype(np.float64),
        "categorical": np.column_stack([
            rng.randint(0, 10, size=n_samples),
            rng.randint(0, 5, size=n_samples),
            rng.randint(0, 8, size=n_samples),
        ]).astype(np.int64),
    }
    y = rng.choice([0, 1], size=n_samples).astype(np.float32)
    return X, y


def _small_config(**overrides):
    """Return a minimal config dict for fast testing."""
    config = {
        "embedding_dim": 16,
        "hidden_dims": [64, 32],
        "dropout": 0.1,
        "activation": "relu",
        "use_layer_norm": False,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 64,
        "max_epochs": 3,
        "patience": 2,
        "device": "cpu",
    }
    config.update(overrides)
    return config


def test_tabular_mlp_forward_shape():
    """Forward pass should return correct output shape."""
    seed_everything(42)
    model = TabularMLP(
        n_numeric=5,
        n_categories_per_column=[10, 5, 8],
        embedding_dim=16,
        hidden_dims=[64, 32],
        task="binary",
        n_classes=2,
    )
    x_num = torch.randn(16, 5)
    x_cat = torch.randint(0, 5, (16, 3))

    output = model(x_num, x_cat)
    print(f"Forward output shape: {output.shape}")

    assert output.shape == (16, 1)
    print("Test passed: binary forward shape correct")


def test_tabular_mlp_forward_multiclass():
    """Multiclass forward pass should return (batch, n_classes)."""
    seed_everything(42)
    model = TabularMLP(
        n_numeric=5,
        n_categories_per_column=[10, 5, 8],
        embedding_dim=16,
        hidden_dims=[64, 32],
        task="multiclass",
        n_classes=4,
    )
    x_num = torch.randn(16, 5)
    x_cat = torch.randint(0, 5, (16, 3))

    output = model(x_num, x_cat)
    print(f"Multiclass forward shape: {output.shape}")

    assert output.shape == (16, 4)
    print("Test passed: multiclass forward shape correct")


def test_tabular_mlp_encode_shape():
    """Encode should return shape (batch, last_hidden_dim)."""
    seed_everything(42)
    model = TabularMLP(
        n_numeric=5,
        n_categories_per_column=[10, 5, 8],
        embedding_dim=16,
        hidden_dims=[64, 32],
        task="binary",
        n_classes=2,
    )
    x_num = torch.randn(16, 5)
    x_cat = torch.randint(0, 5, (16, 3))

    emb = model.encode(x_num, x_cat)
    print(f"Encode output shape: {emb.shape}")

    assert emb.shape == (16, 32)
    print("Test passed: encode shape matches last hidden dim")


def test_mlp_embedding_model_fit():
    """Fitting MLPEmbeddingModel should complete without errors."""
    seed_everything(42)
    X, y = _make_synthetic_data()
    n_train = 150
    X_train = {k: v[:n_train] for k, v in X.items()}
    X_val = {k: v[n_train:] for k, v in X.items()}
    y_train, y_val = y[:n_train], y[n_train:]

    model = MLPEmbeddingModel(
        task="binary", n_numeric=5,
        n_categories_per_column=[10, 5, 8], n_classes=2,
    )
    config = _small_config()
    model.fit(X_train, y_train, X_val, y_val, config)

    print(f"embedding_dim: {model.embedding_dim}")
    assert model.embedding_dim == 32
    print("Test passed: MLPEmbeddingModel fit completes, embedding_dim set")


def test_mlp_embedding_model_encode():
    """After fit, encode should return correct shape numpy array with no NaN."""
    seed_everything(42)
    X, y = _make_synthetic_data()
    n_train = 150
    X_train = {k: v[:n_train] for k, v in X.items()}
    X_val = {k: v[n_train:] for k, v in X.items()}
    y_train, y_val = y[:n_train], y[n_train:]

    model = MLPEmbeddingModel(
        task="binary", n_numeric=5,
        n_categories_per_column=[10, 5, 8], n_classes=2,
    )
    model.fit(X_train, y_train, X_val, y_val, _small_config())

    embeddings = model.encode(X_train)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"NaN count: {np.isnan(embeddings).sum()}")

    assert embeddings.shape == (n_train, 32)
    assert isinstance(embeddings, np.ndarray)
    assert np.isnan(embeddings).sum() == 0
    print("Test passed: encode returns valid numpy array")


def test_mlp_embedding_model_encode_deterministic():
    """Encoding the same data twice should yield identical results."""
    seed_everything(42)
    X, y = _make_synthetic_data()
    n_train = 150
    X_train = {k: v[:n_train] for k, v in X.items()}
    X_val = {k: v[n_train:] for k, v in X.items()}
    y_train, y_val = y[:n_train], y[n_train:]

    model = MLPEmbeddingModel(
        task="binary", n_numeric=5,
        n_categories_per_column=[10, 5, 8], n_classes=2,
    )
    model.fit(X_train, y_train, X_val, y_val, _small_config())

    emb1 = model.encode(X_train)
    emb2 = model.encode(X_train)
    print(f"Max diff between two encodes: {np.abs(emb1 - emb2).max()}")

    np.testing.assert_array_equal(emb1, emb2)
    print("Test passed: encode is deterministic")


def test_mlp_save_load_roundtrip():
    """Save and load should produce identical embeddings."""
    seed_everything(42)
    X, y = _make_synthetic_data()
    n_train = 150
    X_train = {k: v[:n_train] for k, v in X.items()}
    X_val = {k: v[n_train:] for k, v in X.items()}
    y_train, y_val = y[:n_train], y[n_train:]

    model = MLPEmbeddingModel(
        task="binary", n_numeric=5,
        n_categories_per_column=[10, 5, 8], n_classes=2,
    )
    model.fit(X_train, y_train, X_val, y_val, _small_config())
    emb_before = model.encode(X_train)

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "mlp_ckpt.pt")
    model.save(path)
    print(f"Saved to {path}")

    loaded = MLPEmbeddingModel.load(path)
    emb_after = loaded.encode(X_train)
    print(f"Max diff after load: {np.abs(emb_before - emb_after).max()}")

    np.testing.assert_allclose(emb_before, emb_after, atol=1e-6)
    assert loaded.embedding_dim == model.embedding_dim
    print("Test passed: save/load roundtrip preserves embeddings")


def test_mlp_early_stopping():
    """Training should stop before max_epochs with small patience."""
    seed_everything(42)
    X, y = _make_synthetic_data(n_samples=100)
    n_train = 70
    X_train = {k: v[:n_train] for k, v in X.items()}
    X_val = {k: v[n_train:] for k, v in X.items()}
    y_train, y_val = y[:n_train], y[n_train:]

    model = MLPEmbeddingModel(
        task="binary", n_numeric=5,
        n_categories_per_column=[10, 5, 8], n_classes=2,
    )
    config = _small_config(max_epochs=100, patience=2)
    model.fit(X_train, y_train, X_val, y_val, config)

    history = model.get_training_history()
    n_epochs = len(history["train_loss"])
    print(f"Trained for {n_epochs} epochs (max was 100, patience=2)")

    assert n_epochs < 100, "Early stopping should have kicked in"
    print("Test passed: early stopping works")


def test_mlp_different_activations():
    """All supported activations should produce valid outputs."""
    seed_everything(42)
    X, y = _make_synthetic_data(n_samples=100)
    n_train = 70
    X_train = {k: v[:n_train] for k, v in X.items()}
    X_val = {k: v[n_train:] for k, v in X.items()}
    y_train, y_val = y[:n_train], y[n_train:]

    for act in ("relu", "gelu", "silu"):
        seed_everything(42)
        model = MLPEmbeddingModel(
            task="binary", n_numeric=5,
            n_categories_per_column=[10, 5, 8], n_classes=2,
        )
        config = _small_config(activation=act, max_epochs=2)
        model.fit(X_train, y_train, X_val, y_val, config)
        emb = model.encode(X_train)
        print(f"Activation '{act}': embedding shape={emb.shape}, "
              f"NaN={np.isnan(emb).sum()}")
        assert emb.shape == (n_train, 32)
        assert np.isnan(emb).sum() == 0
    print("Test passed: all activations produce valid outputs")


def test_mlp_regression_task():
    """Fitting on a regression task should produce valid embeddings."""
    seed_everything(42)
    rng = np.random.RandomState(42)
    n_samples = 200
    X = {
        "numeric": rng.randn(n_samples, 5).astype(np.float64),
        "categorical": rng.randint(0, 5, size=(n_samples, 3)).astype(np.int64),
    }
    y = rng.normal(0, 1, size=n_samples).astype(np.float32)

    n_train = 150
    X_train = {k: v[:n_train] for k, v in X.items()}
    X_val = {k: v[n_train:] for k, v in X.items()}
    y_train, y_val = y[:n_train], y[n_train:]

    model = MLPEmbeddingModel(
        task="regression", n_numeric=5,
        n_categories_per_column=[5, 5, 5], n_classes=None,
    )
    config = _small_config(max_epochs=3)
    model.fit(X_train, y_train, X_val, y_val, config)

    emb = model.encode(X_train)
    print(f"Regression embeddings shape: {emb.shape}")
    print(f"NaN count: {np.isnan(emb).sum()}")

    assert emb.shape == (n_train, 32)
    assert np.isnan(emb).sum() == 0
    print("Test passed: regression task produces valid embeddings")


# ---------------------------------------------------------------------------
# Quality tests: verify the model actually learns signal from data
# ---------------------------------------------------------------------------


def _make_learnable_binary(n_samples=1000, seed=42):
    """Binary classification with a learnable decision boundary.

    y = 1 if 2*x0 + x1 - 1.5*x2 + cat_effect > 0 else 0
    Categorical column 0 adds +1 for values >= 5 (out of 10).
    """
    rng = np.random.RandomState(seed)
    x_num = rng.randn(n_samples, 5).astype(np.float64)
    x_cat = np.column_stack([
        rng.randint(0, 10, size=n_samples),
        rng.randint(0, 5, size=n_samples),
        rng.randint(0, 8, size=n_samples),
    ]).astype(np.int64)

    logit = 2.0 * x_num[:, 0] + x_num[:, 1] - 1.5 * x_num[:, 2]
    logit += (x_cat[:, 0] >= 5).astype(float) * 1.5
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.rand(n_samples) < prob).astype(np.float32)

    X = {"numeric": x_num, "categorical": x_cat}
    return X, y


def _make_learnable_multiclass(n_samples=1200, n_classes=3, seed=42):
    """Multiclass classification with cluster-based ground truth.

    Each class is centered at a different point in feature space.
    """
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_classes, 5) * 2.0
    labels = []
    x_num_parts = []
    for c in range(n_classes):
        n_c = n_samples // n_classes
        x_num_parts.append(centers[c] + rng.randn(n_c, 5) * 0.8)
        labels.extend([c] * n_c)

    x_num = np.vstack(x_num_parts).astype(np.float64)
    y = np.array(labels, dtype=np.float32)

    perm = rng.permutation(len(y))
    x_num = x_num[perm]
    y = y[perm]

    x_cat = rng.randint(0, 5, size=(len(y), 2)).astype(np.int64)

    X = {"numeric": x_num, "categorical": x_cat}
    return X, y, n_classes


def _make_learnable_regression(n_samples=1000, seed=42):
    """Regression with a learnable non-linear relationship.

    y = 3*x0 + sin(2*x1) - 0.5*x2^2 + noise
    """
    rng = np.random.RandomState(seed)
    x_num = rng.randn(n_samples, 5).astype(np.float64)
    x_cat = rng.randint(0, 5, size=(n_samples, 2)).astype(np.int64)

    y = (
        3.0 * x_num[:, 0]
        + np.sin(2.0 * x_num[:, 1])
        - 0.5 * x_num[:, 2] ** 2
        + rng.randn(n_samples) * 0.3
    ).astype(np.float32)

    X = {"numeric": x_num, "categorical": x_cat}
    return X, y


def _quality_config(**overrides):
    """Config with enough capacity and epochs to actually learn."""
    config = {
        "embedding_dim": 32,
        "hidden_dims": [128, 64],
        "dropout": 0.05,
        "activation": "relu",
        "use_layer_norm": False,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 128,
        "max_epochs": 60,
        "patience": 15,
        "device": "cpu",
    }
    config.update(overrides)
    return config


def test_mlp_binary_quality():
    """MLP should achieve ROC-AUC > 0.85 on a learnable binary task."""
    seed_everything(42)
    X, y = _make_learnable_binary(n_samples=1000)
    n_train, n_val = 700, 150
    X_train = {k: v[:n_train] for k, v in X.items()}
    X_val = {k: v[n_train:n_train + n_val] for k, v in X.items()}
    X_test = {k: v[n_train + n_val:] for k, v in X.items()}
    y_train = y[:n_train]
    y_val = y[n_train:n_train + n_val]
    y_test = y[n_train + n_val:]

    model = MLPEmbeddingModel(
        task="binary", n_numeric=5,
        n_categories_per_column=[10, 5, 8], n_classes=2,
    )
    model.fit(X_train, y_train, X_val, y_val, _quality_config())

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"Binary test ROC-AUC: {auc:.4f}")
    assert auc > 0.85, f"Expected ROC-AUC > 0.85, got {auc:.4f}"
    print("Test passed: MLP achieves strong binary classification")


def test_mlp_multiclass_quality():
    """MLP should achieve accuracy > 0.85 on a learnable 3-class task."""
    seed_everything(42)
    X, y, n_classes = _make_learnable_multiclass(n_samples=1200)
    n_train, n_val = 800, 200
    X_train = {k: v[:n_train] for k, v in X.items()}
    X_val = {k: v[n_train:n_train + n_val] for k, v in X.items()}
    X_test = {k: v[n_train + n_val:] for k, v in X.items()}
    y_train = y[:n_train]
    y_val = y[n_train:n_train + n_val]
    y_test = y[n_train + n_val:]

    model = MLPEmbeddingModel(
        task="multiclass", n_numeric=5,
        n_categories_per_column=[5, 5], n_classes=n_classes,
    )
    model.fit(X_train, y_train, X_val, y_val, _quality_config())

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Multiclass test accuracy: {acc:.4f}")
    assert acc > 0.85, f"Expected accuracy > 0.85, got {acc:.4f}"
    print("Test passed: MLP achieves strong multiclass classification")


def test_mlp_regression_quality():
    """MLP should achieve R^2 > 0.80 on a learnable regression task."""
    seed_everything(42)
    X, y = _make_learnable_regression(n_samples=1000)
    n_train, n_val = 700, 150
    X_train = {k: v[:n_train] for k, v in X.items()}
    X_val = {k: v[n_train:n_train + n_val] for k, v in X.items()}
    X_test = {k: v[n_train + n_val:] for k, v in X.items()}
    y_train = y[:n_train]
    y_val = y[n_train:n_train + n_val]
    y_test = y[n_train + n_val:]

    model = MLPEmbeddingModel(
        task="regression", n_numeric=5,
        n_categories_per_column=[5, 5], n_classes=None,
    )
    model.fit(X_train, y_train, X_val, y_val, _quality_config())

    preds = model.predict(X_test).squeeze()
    r2 = r2_score(y_test, preds)
    print(f"Regression test R^2: {r2:.4f}")
    assert r2 > 0.80, f"Expected R^2 > 0.80, got {r2:.4f}"
    print("Test passed: MLP achieves strong regression performance")


# ---------------------------------------------------------------------------
# Sanity tests: random labels should NOT produce predictive models
# ---------------------------------------------------------------------------


def test_mlp_random_labels_binary_no_leakage():
    """MLP on shuffled binary labels should stay near chance (ROC-AUC ~ 0.5)."""
    seed_everything(42)
    X, _ = _make_learnable_binary(n_samples=800)
    rng = np.random.RandomState(99)
    y_random = rng.choice([0, 1], size=800).astype(np.float32)

    n_train, n_val = 500, 150
    X_train = {k: v[:n_train] for k, v in X.items()}
    X_val = {k: v[n_train:n_train + n_val] for k, v in X.items()}
    X_test = {k: v[n_train + n_val:] for k, v in X.items()}
    y_train = y_random[:n_train]
    y_val = y_random[n_train:n_train + n_val]
    y_test = y_random[n_train + n_val:]

    model = MLPEmbeddingModel(
        task="binary", n_numeric=5,
        n_categories_per_column=[10, 5, 8], n_classes=2,
    )
    model.fit(X_train, y_train, X_val, y_val, _quality_config())

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"Random-label binary ROC-AUC: {auc:.4f}")
    assert auc < 0.65, (
        f"Model should not learn from random labels; ROC-AUC={auc:.4f}"
    )
    print("Test passed: no leakage on random binary labels")


def test_mlp_random_labels_regression_no_leakage():
    """MLP on random regression targets should have R^2 near 0."""
    seed_everything(42)
    X, _ = _make_learnable_regression(n_samples=800)
    rng = np.random.RandomState(99)
    y_random = rng.randn(800).astype(np.float32)

    n_train, n_val = 500, 150
    X_train = {k: v[:n_train] for k, v in X.items()}
    X_val = {k: v[n_train:n_train + n_val] for k, v in X.items()}
    X_test = {k: v[n_train + n_val:] for k, v in X.items()}
    y_train = y_random[:n_train]
    y_val = y_random[n_train:n_train + n_val]
    y_test = y_random[n_train + n_val:]

    model = MLPEmbeddingModel(
        task="regression", n_numeric=5,
        n_categories_per_column=[5, 5], n_classes=None,
    )
    model.fit(X_train, y_train, X_val, y_val, _quality_config())

    preds = model.predict(X_test).squeeze()
    r2 = r2_score(y_test, preds)
    print(f"Random-label regression R^2: {r2:.4f}")
    assert r2 < 0.15, (
        f"Model should not learn from random targets; R^2={r2:.4f}"
    )
    print("Test passed: no leakage on random regression labels")
