"""Tests for hybrid_embed.data.schema: automatic column type detection."""

import numpy as np
import pandas as pd

from hybrid_embed.data.schema import infer_schema


def test_string_columns_are_categorical():
    """Object-dtype columns should be classified as categorical."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "name": rng.choice(["Alice", "Bob", "Carol"], size=50),
        "city": rng.choice(["NYC", "LA"], size=50),
    })
    print(f"Dtypes:\n{df.dtypes}")

    schema = infer_schema(df)
    print(f"Categorical: {schema.categorical_columns}")
    print(f"Numeric: {schema.numeric_columns}")

    assert "name" in schema.categorical_columns
    assert "city" in schema.categorical_columns
    assert len(schema.numeric_columns) == 0
    print("Test passed: string columns are categorical")


def test_float_columns_are_numeric():
    """Float-dtype columns should be classified as numeric."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "income": rng.normal(50000, 15000, size=50),
        "temperature": rng.uniform(-10, 40, size=50),
    })
    print(f"Dtypes:\n{df.dtypes}")

    schema = infer_schema(df)
    print(f"Numeric: {schema.numeric_columns}")
    print(f"Categorical: {schema.categorical_columns}")

    assert "income" in schema.numeric_columns
    assert "temperature" in schema.numeric_columns
    assert len(schema.categorical_columns) == 0
    print("Test passed: float columns are numeric")


def test_low_cardinality_int_is_categorical():
    """Integer column with few unique values should be categorical."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "rating": rng.choice([1, 2, 3, 4, 5], size=50),
    })
    print(f"Dtype: {df['rating'].dtype}, unique: {df['rating'].nunique()}")

    schema = infer_schema(df)
    print(f"Categorical: {schema.categorical_columns}")

    assert "rating" in schema.categorical_columns
    print("Test passed: low-cardinality int is categorical")


def test_high_cardinality_int_is_numeric():
    """Integer column with many unique values should be numeric."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "user_id": rng.randint(0, 1000, size=200),
    })
    n_unique = df["user_id"].nunique()
    print(f"Dtype: {df['user_id'].dtype}, unique values: {n_unique}")

    schema = infer_schema(df)
    print(f"Numeric: {schema.numeric_columns}")

    assert "user_id" in schema.numeric_columns
    assert n_unique > 20
    print("Test passed: high-cardinality int is numeric")


def test_bool_columns_are_categorical():
    """Boolean columns should be classified as categorical."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "is_active": rng.choice([True, False], size=50),
    })
    df["is_active"] = df["is_active"].astype(bool)
    print(f"Dtype: {df['is_active'].dtype}")

    schema = infer_schema(df)
    print(f"Categorical: {schema.categorical_columns}")

    assert "is_active" in schema.categorical_columns
    print("Test passed: bool columns are categorical")


def test_mixed_dataframe():
    """Mixed DataFrame should split columns correctly."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "age": rng.randint(18, 80, size=100),
        "income": rng.normal(50000, 15000, size=100),
        "city": rng.choice(["NYC", "LA", "CHI"], size=100),
        "rating": rng.choice([1, 2, 3, 4, 5], size=100),
        "is_member": rng.choice([True, False], size=100),
    })
    df["is_member"] = df["is_member"].astype(bool)
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")

    schema = infer_schema(df)
    print(f"Numeric: {schema.numeric_columns}")
    print(f"Categorical: {schema.categorical_columns}")

    assert "income" in schema.numeric_columns
    assert "age" in schema.numeric_columns
    assert "city" in schema.categorical_columns
    assert "rating" in schema.categorical_columns
    assert "is_member" in schema.categorical_columns
    print("Test passed: mixed DataFrame split correctly")


def test_categorical_threshold_parameter():
    """Different thresholds should change classification of int columns."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "code": rng.choice(range(15), size=100),
    })
    n_unique = df["code"].nunique()
    print(f"Column 'code' has {n_unique} unique values")

    schema_low = infer_schema(df, categorical_threshold=10)
    print(f"Threshold=10 -> numeric: {schema_low.numeric_columns}, "
          f"categorical: {schema_low.categorical_columns}")

    schema_high = infer_schema(df, categorical_threshold=20)
    print(f"Threshold=20 -> numeric: {schema_high.numeric_columns}, "
          f"categorical: {schema_high.categorical_columns}")

    assert "code" in schema_low.numeric_columns
    assert "code" in schema_high.categorical_columns
    print("Test passed: threshold changes classification")
