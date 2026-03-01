"""Tests for hybrid_embed.data.preprocess: TabularPreprocessor."""

import numpy as np
import pandas as pd

from hybrid_embed.config import Schema
from hybrid_embed.data.preprocess import TabularPreprocessor


def _make_schema():
    """Schema for the synthetic test data."""
    return Schema(
        numeric_columns=["num_a", "num_b", "num_c"],
        categorical_columns=["cat_x", "cat_y"],
    )


def _make_train_test(seed=42):
    """Generate synthetic train/test DataFrames with known properties.

    - num_a: clean float
    - num_b: float with some NaN
    - num_c: clean int-like float
    - cat_x: string with NaN in some rows
    - cat_y: string; test set has an unseen value
    """
    rng = np.random.RandomState(seed)
    n_train, n_test = 100, 30

    train = pd.DataFrame({
        "num_a": rng.normal(10, 2, size=n_train),
        "num_b": rng.normal(50, 10, size=n_train),
        "num_c": rng.uniform(0, 100, size=n_train),
        "cat_x": rng.choice(["A", "B", "C"], size=n_train),
        "cat_y": rng.choice(["red", "green", "blue"], size=n_train),
    })
    train.loc[5, "num_b"] = np.nan
    train.loc[10, "num_b"] = np.nan
    train.loc[15, "cat_x"] = None

    test = pd.DataFrame({
        "num_a": rng.normal(10, 2, size=n_test),
        "num_b": rng.normal(50, 10, size=n_test),
        "num_c": rng.uniform(0, 100, size=n_test),
        "cat_x": rng.choice(["A", "B", "C"], size=n_test),
        "cat_y": rng.choice(["red", "green", "blue", "purple"], size=n_test),
    })

    return train, test


def test_fit_transform_numeric_shape():
    """Numeric output should have shape (n_samples, n_numeric)."""
    schema = _make_schema()
    train, _ = _make_train_test()
    pp = TabularPreprocessor(schema, task="regression")
    pp.fit(train)

    out = pp.transform(train)
    print(f"Numeric shape: {out['numeric'].shape}")

    assert out["numeric"].shape == (100, 3)
    print("Test passed: numeric output has correct shape")


def test_fit_transform_categorical_shape():
    """Categorical output should have shape (n_samples, n_cat) and dtype int64."""
    schema = _make_schema()
    train, _ = _make_train_test()
    pp = TabularPreprocessor(schema, task="regression")
    pp.fit(train)

    out = pp.transform(train)
    print(f"Categorical shape: {out['categorical'].shape}")
    print(f"Categorical dtype: {out['categorical'].dtype}")

    assert out["categorical"].shape == (100, 2)
    assert out["categorical"].dtype == np.int64
    print("Test passed: categorical output has correct shape and dtype")


def test_numeric_nan_imputed():
    """No NaN values should remain in numeric output after imputation."""
    schema = _make_schema()
    train, _ = _make_train_test()
    pp = TabularPreprocessor(schema, task="regression")
    pp.fit(train)

    out = pp.transform(train)
    n_nan = np.isnan(out["numeric"]).sum()
    print(f"NaN count in numeric output: {n_nan}")

    assert n_nan == 0, "Numeric output should have no NaN after imputation"
    print("Test passed: no NaN in numeric output")


def test_numeric_scaled_zero_mean():
    """Scaled train numeric features should have approximately zero mean."""
    schema = _make_schema()
    train, _ = _make_train_test()
    pp = TabularPreprocessor(schema, task="regression")
    pp.fit(train)

    out = pp.transform(train)
    col_means = out["numeric"].mean(axis=0)
    print(f"Column means after scaling: {col_means}")

    for i, m in enumerate(col_means):
        assert abs(m) < 0.1, f"Column {i} mean {m} is not close to 0"
    print("Test passed: scaled train data has ~zero mean")


def test_categorical_oov_handling():
    """Values not in train vocabulary should map to OOV index (0)."""
    schema = _make_schema()
    train, test = _make_train_test()
    pp = TabularPreprocessor(schema, task="regression")
    pp.fit(train)

    out = pp.transform(test)
    cat_y_col_idx = schema.categorical_columns.index("cat_y")
    cat_y_encoded = out["categorical"][:, cat_y_col_idx]

    purple_mask = test["cat_y"] == "purple"
    purple_encoded = cat_y_encoded[purple_mask.values]
    print(f"'purple' rows encoded as: {purple_encoded}")

    if len(purple_encoded) > 0:
        assert np.all(purple_encoded == 0), "Unseen values should map to OOV (0)"
    print("Test passed: unseen categorical values map to OOV index 0")


def test_categorical_missing_handling():
    """NaN categorical values should map to MISSING index (1)."""
    schema = _make_schema()
    train, _ = _make_train_test()
    pp = TabularPreprocessor(schema, task="regression")
    pp.fit(train)

    out = pp.transform(train)
    cat_x_col_idx = schema.categorical_columns.index("cat_x")
    cat_x_encoded = out["categorical"][:, cat_x_col_idx]

    missing_idx = 15
    print(f"Row {missing_idx} cat_x original: {train['cat_x'].iloc[missing_idx]}")
    print(f"Row {missing_idx} cat_x encoded: {cat_x_encoded[missing_idx]}")

    assert cat_x_encoded[missing_idx] == 1, "NaN should map to MISSING index (1)"
    print("Test passed: NaN categorical maps to MISSING index 1")


def test_cardinality_cap():
    """High-cardinality column should be capped to max + special tokens."""
    rng = np.random.RandomState(42)
    n = 500
    schema = Schema(
        numeric_columns=[],
        categorical_columns=["high_card"],
    )
    train = pd.DataFrame({
        "high_card": [f"val_{i}" for i in rng.randint(0, 2000, size=n)],
    })
    pp = TabularPreprocessor(schema, task="regression", max_categorical_cardinality=100)
    pp.fit(train)

    sizes = pp.get_n_categories_per_column()
    print(f"Vocabulary size (including special tokens): {sizes[0]}")

    assert sizes[0] <= 102, f"Vocab size {sizes[0]} exceeds cap (100 + 2 special)"
    print("Test passed: cardinality cap enforced")


def test_target_transform_classification():
    """String classification labels should be mapped to contiguous ints."""
    schema = _make_schema()
    train, _ = _make_train_test()
    y_train = np.array(["cat", "dog", "bird"] * 33 + ["cat"])

    pp = TabularPreprocessor(schema, task="multiclass")
    pp.fit(train, y_train)

    y_transformed = pp.transform_target(y_train)
    print(f"Original labels: {np.unique(y_train)}")
    print(f"Transformed labels: {np.unique(y_transformed)}")

    assert set(y_transformed) == {0, 1, 2}
    assert pp.n_classes == 3
    print("Test passed: classification labels mapped to 0..K-1")


def test_target_transform_regression():
    """Regression target should be scaled to ~zero mean and ~unit std."""
    schema = _make_schema()
    train, _ = _make_train_test()
    y_train = np.random.RandomState(42).normal(100, 25, size=100)

    pp = TabularPreprocessor(schema, task="regression")
    pp.fit(train, y_train)

    y_transformed = pp.transform_target(y_train)
    print(f"Transformed mean: {y_transformed.mean():.4f}")
    print(f"Transformed std: {y_transformed.std():.4f}")

    assert abs(y_transformed.mean()) < 0.1
    assert abs(y_transformed.std() - 1.0) < 0.1
    assert pp.n_classes is None
    print("Test passed: regression target scaled to ~N(0,1)")


def test_inverse_transform_target_regression():
    """Transform then inverse should recover original regression values."""
    schema = _make_schema()
    train, _ = _make_train_test()
    y_train = np.random.RandomState(42).normal(100, 25, size=100)

    pp = TabularPreprocessor(schema, task="regression")
    pp.fit(train, y_train)

    y_transformed = pp.transform_target(y_train)
    y_recovered = pp.inverse_transform_target(y_transformed)
    print(f"Original[0:5]: {y_train[:5]}")
    print(f"Recovered[0:5]: {y_recovered[:5]}")

    np.testing.assert_allclose(y_train, y_recovered, atol=1e-10)
    print("Test passed: inverse transform recovers original values")


def test_get_n_categories():
    """get_n_categories_per_column should include special tokens."""
    schema = _make_schema()
    train, _ = _make_train_test()
    pp = TabularPreprocessor(schema, task="regression")
    pp.fit(train)

    sizes = pp.get_n_categories_per_column()
    print(f"Category sizes: {sizes}")
    print(f"cat_x unique (train): {train['cat_x'].dropna().nunique()}")
    print(f"cat_y unique (train): {train['cat_y'].nunique()}")

    assert sizes[0] == train["cat_x"].dropna().nunique() + 2
    assert sizes[1] == train["cat_y"].nunique() + 2
    print("Test passed: category sizes include OOV + MISSING tokens")


def test_preprocessor_fit_on_train_only():
    """Scaler parameters should come from train data, not test data."""
    rng = np.random.RandomState(42)
    schema = Schema(numeric_columns=["x"], categorical_columns=[])

    train = pd.DataFrame({"x": rng.normal(0, 1, size=100)})
    test = pd.DataFrame({"x": rng.normal(10, 5, size=50)})

    pp = TabularPreprocessor(schema, task="regression")
    pp.fit(train)

    train_out = pp.transform(train)
    test_out = pp.transform(test)
    print(f"Train output mean: {train_out['numeric'].mean():.4f}")
    print(f"Test output mean: {test_out['numeric'].mean():.4f}")

    assert abs(train_out["numeric"].mean()) < 0.1
    assert abs(test_out["numeric"].mean()) > 1.0, (
        "Test mean should NOT be ~0 since scaler was fit on train only"
    )
    print("Test passed: scaler is fit on train only")


def test_missing_indicator_toggle():
    """When add_missing_indicator=True, extra columns should be added."""
    schema = _make_schema()
    train, _ = _make_train_test()

    pp_off = TabularPreprocessor(schema, task="regression", add_missing_indicator=False)
    pp_off.fit(train)
    out_off = pp_off.transform(train)

    pp_on = TabularPreprocessor(schema, task="regression", add_missing_indicator=True)
    pp_on.fit(train)
    out_on = pp_on.transform(train)

    print(f"Without indicator: numeric shape = {out_off['numeric'].shape}")
    print(f"With indicator: numeric shape = {out_on['numeric'].shape}")
    print(f"get_n_numeric_features (off): {pp_off.get_n_numeric_features()}")
    print(f"get_n_numeric_features (on): {pp_on.get_n_numeric_features()}")

    assert out_on["numeric"].shape[1] > out_off["numeric"].shape[1]
    assert pp_on.get_n_numeric_features() == out_on["numeric"].shape[1]
    assert pp_off.get_n_numeric_features() == out_off["numeric"].shape[1]
    print("Test passed: missing indicator adds extra columns")
