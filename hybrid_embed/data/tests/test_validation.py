"""Tests for hybrid_embed.data.validation: defensive data checks."""

import json
import os
import tempfile

import numpy as np
import pandas as pd

from hybrid_embed.data.validation import (
    validate_dataframe,
    save_validation_report,
    ValidationReport,
)


def _make_clean_dataframe(n_rows=50, seed=42):
    """Generate a clean DataFrame with 3 numeric + 1 categorical column."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "age": rng.randint(18, 80, size=n_rows),
        "income": rng.normal(50000, 15000, size=n_rows),
        "score": rng.uniform(0, 100, size=n_rows),
        "city": rng.choice(["NYC", "LA", "CHI", "HOU"], size=n_rows),
    })


def test_valid_data_passes():
    """Clean data should pass validation unchanged."""
    df = _make_clean_dataframe()
    y = np.random.RandomState(42).choice([0, 1], size=len(df))
    print(f"Input shape: {df.shape}")
    print(f"Target unique values: {np.unique(y)}")

    X_clean, report = validate_dataframe(df, y, task="binary")
    print(f"Output shape: {X_clean.shape}")
    print(f"Dropped constant: {report.dropped_constant_columns}")
    print(f"Dropped all-NaN: {report.dropped_allnan_columns}")
    print(f"Warnings: {report.warnings}")

    assert X_clean.shape == df.shape
    assert list(X_clean.columns) == list(df.columns)
    assert len(report.dropped_constant_columns) == 0
    assert len(report.dropped_allnan_columns) == 0
    assert len(report.warnings) == 0
    assert report.n_rows == 50
    assert report.n_cols_original == 4
    assert report.n_cols_after == 4
    print("Test passed: clean data passes validation unchanged")


def test_drops_constant_column():
    """Columns with zero variance should be dropped with a warning."""
    df = _make_clean_dataframe()
    df["const_col"] = 999
    y = np.random.RandomState(42).choice([0, 1], size=len(df))
    print(f"Input columns: {list(df.columns)}")

    X_clean, report = validate_dataframe(df, y, task="binary")
    print(f"Output columns: {list(X_clean.columns)}")
    print(f"Dropped constant: {report.dropped_constant_columns}")

    assert "const_col" not in X_clean.columns
    assert "const_col" in report.dropped_constant_columns
    assert report.n_cols_original == 5
    assert report.n_cols_after == 4
    assert any("const_col" in w for w in report.warnings)
    print("Test passed: constant column dropped correctly")


def test_drops_allnan_column():
    """Columns with all NaN should be dropped with a warning."""
    df = _make_clean_dataframe()
    df["empty_col"] = np.nan
    y = np.random.RandomState(42).choice([0, 1], size=len(df))
    print(f"Input columns: {list(df.columns)}")

    X_clean, report = validate_dataframe(df, y, task="binary")
    print(f"Output columns: {list(X_clean.columns)}")
    print(f"Dropped all-NaN: {report.dropped_allnan_columns}")

    assert "empty_col" not in X_clean.columns
    assert "empty_col" in report.dropped_allnan_columns
    assert any("empty_col" in w for w in report.warnings)
    print("Test passed: all-NaN column dropped correctly")


def test_raises_on_nan_target():
    """NaN values in target should raise ValueError."""
    df = _make_clean_dataframe()
    y = np.array([0.0, 1.0, np.nan] + [0.0] * 47)
    print(f"Target has NaN at index 2: {y[2]}")

    try:
        validate_dataframe(df, y, task="binary")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
    print("Test passed: NaN target rejected")


def test_raises_on_empty_dataframe():
    """Empty DataFrame should raise ValueError."""
    df = pd.DataFrame()
    y = np.array([])
    print(f"Empty DataFrame shape: {df.shape}")

    try:
        validate_dataframe(df, y, task="regression")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
    print("Test passed: empty DataFrame rejected")


def test_raises_on_length_mismatch():
    """Mismatched X and y lengths should raise ValueError."""
    df = _make_clean_dataframe(n_rows=50)
    y = np.zeros(30)
    print(f"X rows: {len(df)}, y length: {len(y)}")

    try:
        validate_dataframe(df, y, task="regression")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
    print("Test passed: length mismatch rejected")


def test_raises_on_single_class_binary():
    """Binary task with only 1 class should raise ValueError."""
    df = _make_clean_dataframe()
    y = np.ones(len(df))
    print(f"Unique classes: {np.unique(y)}")

    try:
        validate_dataframe(df, y, task="binary")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
    print("Test passed: single-class binary rejected")


def test_raises_on_wrong_class_count_binary():
    """Binary task with 3 classes should raise ValueError."""
    df = _make_clean_dataframe()
    y = np.array([0, 1, 2] * 16 + [0, 1])
    print(f"Unique classes: {np.unique(y)}, count: {len(np.unique(y))}")

    try:
        validate_dataframe(df, y, task="binary")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
    print("Test passed: 3-class binary rejected")


def test_multiclass_allows_many_classes():
    """Multiclass task should accept 5 classes without error."""
    df = _make_clean_dataframe()
    y = np.array([0, 1, 2, 3, 4] * 10)
    print(f"Unique classes: {np.unique(y)}")

    X_clean, report = validate_dataframe(df, y, task="multiclass")
    print(f"Validation passed, output shape: {X_clean.shape}")

    assert X_clean.shape == df.shape
    print("Test passed: multiclass accepts 5 classes")


def test_regression_allows_any_y():
    """Regression task should accept continuous y without error."""
    df = _make_clean_dataframe()
    y = np.random.RandomState(42).normal(0, 1, size=len(df))
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")

    X_clean, report = validate_dataframe(df, y, task="regression")
    print(f"Validation passed, output shape: {X_clean.shape}")

    assert X_clean.shape == df.shape
    print("Test passed: regression accepts continuous y")


def test_save_validation_report():
    """Verify validation report saves to JSON correctly."""
    report = ValidationReport(
        dropped_constant_columns=["const"],
        dropped_allnan_columns=["empty"],
        warnings=["Dropped constant column: 'const'"],
        n_rows=100,
        n_cols_original=5,
        n_cols_after=3,
    )
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "validation_report.json")

    save_validation_report(report, path)
    print(f"Saved report to {path}")

    with open(path, "r") as fh:
        loaded = json.load(fh)
    print(f"Loaded report: {loaded}")

    assert loaded["dropped_constant_columns"] == ["const"]
    assert loaded["dropped_allnan_columns"] == ["empty"]
    assert loaded["n_rows"] == 100
    assert loaded["n_cols_original"] == 5
    assert loaded["n_cols_after"] == 3
    print("Test passed: validation report JSON roundtrip correct")
