"""Tests for hybrid_embed.embed.base: EmbeddingModel ABC and TabularDataset."""

import numpy as np
import torch

from hybrid_embed.embed.base import EmbeddingModel, TabularDataset


class DummyEmbedding(EmbeddingModel):
    """Minimal concrete subclass for testing the ABC contract."""

    def fit(self, X_train, y_train, X_val, y_val, config):
        self.embedding_dim = config.get("embed_dim", 16)
        self._n_features = X_train["numeric"].shape[1]
        return self

    def predict(self, X):
        return np.zeros(len(X["numeric"]))

    def encode(self, X):
        n = len(X["numeric"])
        return np.random.randn(n, self.embedding_dim)

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        return cls()


def _make_preprocessed_data(n_samples=100, n_numeric=5, n_categorical=3, seed=42):
    """Generate synthetic preprocessed data dict."""
    rng = np.random.RandomState(seed)
    X = {
        "numeric": rng.randn(n_samples, n_numeric).astype(np.float64),
        "categorical": rng.randint(0, 10, size=(n_samples, n_categorical)).astype(np.int64),
    }
    y = rng.choice([0, 1], size=n_samples).astype(np.float32)
    return X, y


def test_cannot_instantiate_abc():
    """EmbeddingModel cannot be instantiated directly."""
    try:
        EmbeddingModel()
        assert False, "Should have raised TypeError"
    except TypeError as e:
        print(f"Correctly raised TypeError: {e}")
    print("Test passed: ABC cannot be instantiated")


def test_concrete_subclass_works():
    """A concrete subclass should implement fit/encode correctly."""
    X, y = _make_preprocessed_data()
    model = DummyEmbedding()
    config = {"embed_dim": 32}

    model.fit(X, y, X, y, config)
    print(f"embedding_dim after fit: {model.embedding_dim}")
    assert model.embedding_dim == 32

    embeddings = model.encode(X)
    print(f"Encode output shape: {embeddings.shape}")
    assert embeddings.shape == (100, 32)
    assert isinstance(embeddings, np.ndarray)
    print("Test passed: concrete subclass fit/encode work correctly")


def test_tabular_dataset_length():
    """TabularDataset should report correct length."""
    X, y = _make_preprocessed_data(n_samples=100)
    dataset = TabularDataset(X, y)
    print(f"Dataset length: {len(dataset)}")

    assert len(dataset) == 100
    print("Test passed: dataset length is correct")


def test_tabular_dataset_getitem():
    """Getting an item should return a tuple of tensors with correct shapes."""
    X, y = _make_preprocessed_data(n_samples=100, n_numeric=5, n_categorical=3)
    dataset = TabularDataset(X, y)

    numeric, categorical, target = dataset[5]
    print(f"Numeric shape: {numeric.shape}")
    print(f"Categorical shape: {categorical.shape}")
    print(f"Target shape: {target.shape}")

    assert numeric.shape == (5,)
    assert categorical.shape == (3,)
    assert target.shape == ()
    print("Test passed: getitem returns correct shapes")


def test_tabular_dataset_dtypes():
    """Tensor dtypes should be float32 for numeric/target, long for categorical."""
    X, y = _make_preprocessed_data()
    dataset = TabularDataset(X, y)

    numeric, categorical, target = dataset[0]
    print(f"Numeric dtype: {numeric.dtype}")
    print(f"Categorical dtype: {categorical.dtype}")
    print(f"Target dtype: {target.dtype}")

    assert numeric.dtype == torch.float32
    assert categorical.dtype == torch.long
    assert target.dtype == torch.float32
    print("Test passed: tensor dtypes are correct")
