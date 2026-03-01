"""Schema inference: automatic column type detection.

Determines which columns in a DataFrame are numeric and which are
categorical, using dtype inspection and a cardinality heuristic
for integer columns.
"""

from __future__ import annotations

import pandas as pd

from hybrid_embed.config import Schema


def infer_schema(
    X: pd.DataFrame,
    categorical_threshold: int = 20,
) -> Schema:
    """Infer column types from a DataFrame.

    Classifies each column as either numeric or categorical based
    on its dtype and, for integer columns, the number of unique
    values. This is a generic, non-expert heuristic.

    Rules (applied in order):

    1. Columns with dtype ``object`` or ``category`` -> categorical.
    2. Columns with dtype ``bool`` -> categorical.
    3. Integer columns with <= ``categorical_threshold`` unique
       values -> categorical (heuristic for encoded categoricals).
    4. All remaining numeric columns (int, float) -> numeric.

    Parameters
    ----------
    X : pd.DataFrame
        Already validated (no constant or all-NaN columns).
    categorical_threshold : int, default 20
        Integer columns with <= this many unique values are treated
        as categorical.

    Returns
    -------
    Schema
        With ``numeric_columns`` and ``categorical_columns`` populated.
    """
    numeric_columns = []
    categorical_columns = []

    for col in X.columns:
        dtype = X[col].dtype

        if dtype == object or isinstance(dtype, pd.CategoricalDtype):
            categorical_columns.append(col)
        elif dtype == bool or dtype == "bool":
            categorical_columns.append(col)
        elif pd.api.types.is_integer_dtype(dtype):
            n_unique = X[col].nunique()
            if n_unique <= categorical_threshold:
                categorical_columns.append(col)
            else:
                numeric_columns.append(col)
        elif pd.api.types.is_float_dtype(dtype):
            numeric_columns.append(col)
        else:
            numeric_columns.append(col)

    return Schema(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
