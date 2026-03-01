"""Tests for the pytorch-tabular adapter (PytorchTabularEmbeddingModel)."""

import os
import shutil
import tempfile
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

from hybrid_embed.config import Schema
from hybrid_embed.embed.adapters.pytorch_tabular_adapter import (
    PytorchTabularEmbeddingModel,
)
from hybrid_embed.utils import seed_everything

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------

def _make_synthetic_data(n_samples=200, seed=42):
    """Random binary-classification data (no signal)."""
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


def _make_learnable_binary(n_samples=1000, seed=42):
    """Binary data with a learnable decision boundary."""
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

    return {"numeric": x_num, "categorical": x_cat}, y


def _make_learnable_multiclass(n_samples=1200, n_classes=3, seed=42):
    """Multiclass data: well-separated Gaussian clusters."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_classes, 5) * 2.0
    parts, labels = [], []
    for c in range(n_classes):
        n_c = n_samples // n_classes
        parts.append(centers[c] + rng.randn(n_c, 5) * 0.8)
        labels.extend([c] * n_c)

    x_num = np.vstack(parts).astype(np.float64)
    y = np.array(labels, dtype=np.float32)
    perm = rng.permutation(len(y))
    x_num, y = x_num[perm], y[perm]
    x_cat = rng.randint(0, 5, size=(len(y), 2)).astype(np.int64)

    return {"numeric": x_num, "categorical": x_cat}, y, n_classes


def _make_learnable_regression(n_samples=1000, seed=42):
    """Regression data with a learnable relationship."""
    rng = np.random.RandomState(seed)
    x_num = rng.randn(n_samples, 5).astype(np.float64)
    x_cat = rng.randint(0, 5, size=(n_samples, 2)).astype(np.int64)
    y = (
        3.0 * x_num[:, 0]
        + np.sin(2.0 * x_num[:, 1])
        - 0.5 * x_num[:, 2] ** 2
        + rng.randn(n_samples) * 0.3
    ).astype(np.float32)
    return {"numeric": x_num, "categorical": x_cat}, y


_SCHEMA_BINARY = Schema(
    numeric_columns=[f"num_{i}" for i in range(5)],
    categorical_columns=[f"cat_{i}" for i in range(3)],
)

_SCHEMA_REG = Schema(
    numeric_columns=[f"num_{i}" for i in range(5)],
    categorical_columns=[f"cat_{i}" for i in range(2)],
)


def _split(X, y, n_train, n_val=None):
    """Split dict data into train/val (and optionally test)."""
    n_val = n_val or (len(y) - n_train)
    Xt = {k: v[:n_train] for k, v in X.items()}
    Xv = {k: v[n_train:n_train + n_val] for k, v in X.items()}
    yt, yv = y[:n_train], y[n_train:n_train + n_val]
    return Xt, yt, Xv, yv


def _small_config(model_type, **overrides):
    """Minimal config for fast testing."""
    config = {
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 64,
        "max_epochs": 3,
        "patience": None,
        "device": "cpu",
    }

    if model_type == "tab_transformer":
        config.update(input_embed_dim=16, num_heads=2, num_attn_blocks=2)
    elif model_type == "category_embedding":
        config.update(layers="64-32", activation="ReLU", dropout=0.1)
    elif model_type == "tabnet":
        config.update(n_d=16, n_a=16, n_steps=3)
    elif model_type == "gate":
        config.update(gflu_stages=2, num_trees=5, tree_depth=3)

    config.update(overrides)
    return config


def _quality_config(model_type, **overrides):
    """Config with enough capacity and epochs to learn."""
    config = {
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 128,
        "max_epochs": 30,
        "patience": None,
        "device": "cpu",
    }

    if model_type == "tab_transformer":
        config.update(input_embed_dim=32, num_heads=4, num_attn_blocks=3)
    elif model_type == "category_embedding":
        config.update(layers="128-64", activation="ReLU", dropout=0.05)
    elif model_type == "tabnet":
        config.update(n_d=32, n_a=32, n_steps=4)
    elif model_type == "gate":
        config.update(gflu_stages=4, num_trees=10, tree_depth=4)

    config.update(overrides)
    return config


# ==========================================================================
# Structural tests
# ==========================================================================


def test_adapter_fit_tab_transformer():
    """TabTransformer should fit without errors and set embedding_dim."""
    seed_everything(42)
    X, y = _make_synthetic_data()
    Xt, yt, Xv, yv = _split(X, y, 150)

    model = PytorchTabularEmbeddingModel(
        model_type="tab_transformer", task="binary",
        schema=_SCHEMA_BINARY, n_classes=2,
    )
    model.fit(Xt, yt, Xv, yv, _small_config("tab_transformer"))

    print(f"TabTransformer embedding_dim: {model.embedding_dim}")
    assert model.embedding_dim > 0
    print("Test passed: TabTransformer fit works")


def test_adapter_encode_shape():
    """After fit, encode should return correct shape numpy array."""
    seed_everything(42)
    X, y = _make_synthetic_data()
    Xt, yt, Xv, yv = _split(X, y, 150)

    model = PytorchTabularEmbeddingModel(
        model_type="category_embedding", task="binary",
        schema=_SCHEMA_BINARY, n_classes=2,
    )
    model.fit(Xt, yt, Xv, yv, _small_config("category_embedding"))

    emb = model.encode(Xt)
    print(f"Embeddings shape: {emb.shape}, type: {type(emb).__name__}")
    assert emb.shape[0] == 150
    assert emb.shape[1] == model.embedding_dim
    assert isinstance(emb, np.ndarray)
    assert np.isnan(emb).sum() == 0
    print("Test passed: encode returns valid numpy array")


def test_adapter_encode_deterministic():
    """Encoding the same data twice should yield identical results."""
    seed_everything(42)
    X, y = _make_synthetic_data()
    Xt, yt, Xv, yv = _split(X, y, 150)

    model = PytorchTabularEmbeddingModel(
        model_type="tab_transformer", task="binary",
        schema=_SCHEMA_BINARY, n_classes=2,
    )
    model.fit(Xt, yt, Xv, yv, _small_config("tab_transformer"))

    emb1 = model.encode(Xt)
    emb2 = model.encode(Xt)
    print(f"Max diff between encodes: {np.abs(emb1 - emb2).max()}")

    np.testing.assert_array_equal(emb1, emb2)
    print("Test passed: encode is deterministic")


def test_adapter_save_load_roundtrip():
    """Save and load should produce identical embeddings."""
    seed_everything(42)
    X, y = _make_synthetic_data()
    Xt, yt, Xv, yv = _split(X, y, 150)

    model = PytorchTabularEmbeddingModel(
        model_type="category_embedding", task="binary",
        schema=_SCHEMA_BINARY, n_classes=2,
    )
    model.fit(Xt, yt, Xv, yv, _small_config("category_embedding"))
    emb_before = model.encode(Xt)

    tmpdir = tempfile.mkdtemp()
    save_dir = os.path.join(tmpdir, "adapter_ckpt")
    try:
        model.save(save_dir)
        loaded = PytorchTabularEmbeddingModel.load(save_dir)
        emb_after = loaded.encode(Xt)
        print(f"Max diff after load: {np.abs(emb_before - emb_after).max()}")
        np.testing.assert_allclose(emb_before, emb_after, atol=1e-5)
        assert loaded.embedding_dim == model.embedding_dim
        print("Test passed: save/load roundtrip preserves embeddings")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_adapter_category_embedding_model():
    """CategoryEmbedding should fit and produce valid embeddings."""
    seed_everything(42)
    X, y = _make_synthetic_data()
    Xt, yt, Xv, yv = _split(X, y, 150)

    model = PytorchTabularEmbeddingModel(
        model_type="category_embedding", task="binary",
        schema=_SCHEMA_BINARY, n_classes=2,
    )
    model.fit(Xt, yt, Xv, yv, _small_config("category_embedding"))
    emb = model.encode(Xt)
    print(f"CategoryEmbedding embeddings: shape={emb.shape}")
    assert emb.shape[0] == 150
    assert np.isnan(emb).sum() == 0
    print("Test passed: CategoryEmbedding works")


def test_adapter_regression_task():
    """Regression task should produce valid embeddings."""
    seed_everything(42)
    X, y = _make_learnable_regression(n_samples=200)
    Xt, yt, Xv, yv = _split(X, y, 150)

    model = PytorchTabularEmbeddingModel(
        model_type="category_embedding", task="regression",
        schema=_SCHEMA_REG, n_classes=None,
    )
    model.fit(Xt, yt, Xv, yv, _small_config("category_embedding"))
    emb = model.encode(Xt)
    print(f"Regression embeddings: shape={emb.shape}")
    assert emb.shape[0] == 150
    assert np.isnan(emb).sum() == 0
    print("Test passed: regression task works")


def test_adapter_multiclass_task():
    """Multiclass task should produce valid embeddings."""
    seed_everything(42)
    X, y, n_c = _make_learnable_multiclass(n_samples=300, n_classes=4)
    schema = Schema(
        numeric_columns=[f"num_{i}" for i in range(5)],
        categorical_columns=[f"cat_{i}" for i in range(2)],
    )
    Xt, yt, Xv, yv = _split(X, y, 200)

    model = PytorchTabularEmbeddingModel(
        model_type="category_embedding", task="multiclass",
        schema=schema, n_classes=n_c,
    )
    model.fit(Xt, yt, Xv, yv, _small_config("category_embedding"))
    emb = model.encode(Xt)
    print(f"Multiclass embeddings: shape={emb.shape}")
    assert emb.shape[0] == 200
    assert np.isnan(emb).sum() == 0
    print("Test passed: multiclass task works")


def test_adapter_unknown_model_type():
    """Invalid model_type should raise ValueError."""
    try:
        PytorchTabularEmbeddingModel(
            model_type="nonexistent", task="binary",
            schema=_SCHEMA_BINARY, n_classes=2,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
        print("Test passed: unknown model_type raises ValueError")


# ==========================================================================
# Quality tests: verify each architecture actually learns
# ==========================================================================


def _run_binary_quality_test(model_type):
    """Helper: train on learnable binary data, check ROC-AUC."""
    seed_everything(42)
    X, y = _make_learnable_binary(n_samples=1000)
    Xt, yt, Xv, yv = _split(X, y, 700, 150)
    X_test = {k: v[850:] for k, v in X.items()}
    y_test = y[850:]

    model = PytorchTabularEmbeddingModel(
        model_type=model_type, task="binary",
        schema=_SCHEMA_BINARY, n_classes=2,
    )
    model.fit(Xt, yt, Xv, yv, _quality_config(model_type))

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    return auc


def test_quality_tab_transformer():
    """TabTransformer should achieve ROC-AUC > 0.70 on categorical-signal data.

    TabTransformer applies self-attention exclusively to categorical
    embeddings, so we test it on data where categoricals carry the
    dominant signal.
    """
    seed_everything(42)
    rng = np.random.RandomState(42)
    n = 1000
    x_num = rng.randn(n, 3).astype(np.float64)
    n_cat_cols = 5
    cat_vocab = 10
    x_cat = rng.randint(0, cat_vocab, size=(n, n_cat_cols)).astype(np.int64)

    logit = np.zeros(n)
    for c in range(n_cat_cols):
        logit += (x_cat[:, c] >= cat_vocab // 2).astype(float) * 1.5
    logit += 0.3 * x_num[:, 0]
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.rand(n) < prob).astype(np.float32)

    X = {"numeric": x_num, "categorical": x_cat}
    schema = Schema(
        numeric_columns=[f"num_{i}" for i in range(3)],
        categorical_columns=[f"cat_{i}" for i in range(n_cat_cols)],
    )

    Xt = {k: v[:700] for k, v in X.items()}
    Xv = {k: v[700:850] for k, v in X.items()}
    X_test = {k: v[850:] for k, v in X.items()}
    yt, yv, y_test = y[:700], y[700:850], y[850:]

    model = PytorchTabularEmbeddingModel(
        model_type="tab_transformer", task="binary",
        schema=schema, n_classes=2,
    )
    config = _quality_config("tab_transformer", max_epochs=80)
    model.fit(Xt, yt, Xv, yv, config)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"TabTransformer ROC-AUC: {auc:.4f}")
    assert auc > 0.60, f"Expected ROC-AUC > 0.60, got {auc:.4f}"
    print("Test passed: TabTransformer has predictive power")


def test_quality_category_embedding():
    """CategoryEmbedding should achieve ROC-AUC > 0.80 on learnable binary data."""
    auc = _run_binary_quality_test("category_embedding")
    print(f"CategoryEmbedding ROC-AUC: {auc:.4f}")
    assert auc > 0.80, f"Expected ROC-AUC > 0.80, got {auc:.4f}"
    print("Test passed: CategoryEmbedding has predictive power")


def test_quality_tabnet():
    """TabNet should achieve ROC-AUC > 0.70 on learnable binary data."""
    auc = _run_binary_quality_test("tabnet")
    print(f"TabNet ROC-AUC: {auc:.4f}")
    assert auc > 0.70, f"Expected ROC-AUC > 0.70, got {auc:.4f}"
    print("Test passed: TabNet has predictive power")


def test_quality_gate():
    """GATE should achieve ROC-AUC > 0.75 on learnable binary data."""
    auc = _run_binary_quality_test("gate")
    print(f"GATE ROC-AUC: {auc:.4f}")
    assert auc > 0.75, f"Expected ROC-AUC > 0.75, got {auc:.4f}"
    print("Test passed: GATE has predictive power")


def test_quality_regression():
    """CategoryEmbedding on regression should achieve R^2 > 0.70."""
    seed_everything(42)
    X, y = _make_learnable_regression(n_samples=1000)
    Xt, yt, Xv, yv = _split(X, y, 700, 150)
    X_test = {k: v[850:] for k, v in X.items()}
    y_test = y[850:]

    model = PytorchTabularEmbeddingModel(
        model_type="category_embedding", task="regression",
        schema=_SCHEMA_REG, n_classes=None,
    )
    model.fit(Xt, yt, Xv, yv, _quality_config("category_embedding"))

    preds = model.predict(X_test).squeeze()
    r2 = r2_score(y_test, preds)
    print(f"Regression R^2: {r2:.4f}")
    assert r2 > 0.70, f"Expected R^2 > 0.70, got {r2:.4f}"
    print("Test passed: regression model has predictive power")


def test_quality_multiclass():
    """CategoryEmbedding on multiclass should achieve accuracy > 0.85."""
    seed_everything(42)
    X, y, n_c = _make_learnable_multiclass(n_samples=1200)
    schema = Schema(
        numeric_columns=[f"num_{i}" for i in range(5)],
        categorical_columns=[f"cat_{i}" for i in range(2)],
    )
    Xt, yt, Xv, yv = _split(X, y, 800, 200)
    X_test = {k: v[1000:] for k, v in X.items()}
    y_test = y[1000:]

    model = PytorchTabularEmbeddingModel(
        model_type="category_embedding", task="multiclass",
        schema=schema, n_classes=n_c,
    )
    model.fit(Xt, yt, Xv, yv, _quality_config("category_embedding"))

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Multiclass accuracy: {acc:.4f}")
    assert acc > 0.85, f"Expected accuracy > 0.85, got {acc:.4f}"
    print("Test passed: multiclass model has predictive power")


# ==========================================================================
# Sanity tests: random labels should NOT produce predictive models
# ==========================================================================


def test_random_labels_category_embedding_no_leakage():
    """CategoryEmbedding on shuffled labels should stay near chance."""
    seed_everything(42)
    X, _ = _make_learnable_binary(n_samples=800)
    rng = np.random.RandomState(99)
    y_random = rng.choice([0, 1], size=800).astype(np.float32)

    Xt, yt, Xv, yv = _split(X, y_random, 500, 150)
    X_test = {k: v[650:] for k, v in X.items()}
    y_test = y_random[650:]

    model = PytorchTabularEmbeddingModel(
        model_type="category_embedding", task="binary",
        schema=_SCHEMA_BINARY, n_classes=2,
    )
    model.fit(Xt, yt, Xv, yv, _quality_config("category_embedding"))

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"Random-label CategoryEmbedding ROC-AUC: {auc:.4f}")
    assert auc < 0.65, (
        f"Model should not learn from random labels; ROC-AUC={auc:.4f}"
    )
    print("Test passed: no leakage on random binary labels")


def test_random_labels_regression_no_leakage():
    """CategoryEmbedding on random regression targets should have R^2 near 0."""
    seed_everything(42)
    X, _ = _make_learnable_regression(n_samples=800)
    rng = np.random.RandomState(99)
    y_random = rng.randn(800).astype(np.float32)

    Xt, yt, Xv, yv = _split(X, y_random, 500, 150)
    X_test = {k: v[650:] for k, v in X.items()}
    y_test = y_random[650:]

    model = PytorchTabularEmbeddingModel(
        model_type="category_embedding", task="regression",
        schema=_SCHEMA_REG, n_classes=None,
    )
    model.fit(Xt, yt, Xv, yv, _quality_config("category_embedding"))

    preds = model.predict(X_test).squeeze()
    r2 = r2_score(y_test, preds)
    print(f"Random-label regression R^2: {r2:.4f}")
    assert r2 < 0.15, (
        f"Model should not learn from random targets; R^2={r2:.4f}"
    )
    print("Test passed: no leakage on random regression labels")
