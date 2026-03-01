"""Data validation: defensive checks before any processing.

Implements generic, non-expert checks that catch degenerate inputs
early (empty data, all-NaN columns, constant columns, invalid
targets) and produce a validation report for logging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    """Results from the data validation step (spec Section 4).

    Records which columns were dropped and why, along with
    dataset size information. Saved to ``validation_report.json``
    for reproducibility.

    Parameters
    ----------
    dropped_constant_columns : list[str]
        Columns dropped because they had zero variance.
    dropped_allnan_columns : list[str]
        Columns dropped because all values were NaN.
    warnings : list[str]
        Human-readable warning messages generated during validation.
    n_rows : int
        Number of rows in the dataset.
    n_cols_original : int
        Number of columns before validation drops.
    n_cols_after : int
        Number of columns after validation drops.
    """

    dropped_constant_columns: list[str] = field(default_factory=list)
    dropped_allnan_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_cols_original: int = 0
    n_cols_after: int = 0


def validate_dataframe(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    task: str,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate input data and return cleaned DataFrame + report.

    Performs generic defensive checks and removes degenerate columns
    (constant or all-NaN). Hard failures raise ``ValueError``; soft
    issues (dropped columns) are recorded in the report with warnings.

    Checks performed (in order):

    1. X must have > 0 rows and > 0 columns.
    2. y must have no NaN values.
    3. ``len(X)`` must equal ``len(y)``.
    4. For classification tasks: y must have >= 2 unique classes.
    5. For binary task: y must have exactly 2 unique classes.
    6. Drop constant columns (zero variance) with warning.
    7. Drop all-NaN columns with warning.
    8. After drops, X must still have > 0 columns.

    Parameters
    ----------
    X : pd.DataFrame
        Raw feature matrix.
    y : pd.Series or np.ndarray
        Target variable.
    task : str
        One of ``"binary"``, ``"multiclass"``, ``"regression"``.

    Returns
    -------
    X_clean : pd.DataFrame
        With problematic columns removed.
    report : ValidationReport

    Raises
    ------
    ValueError
        If any hard check fails (empty data, NaN in y, wrong class
        count, etc.).
    """
    report = ValidationReport()
    report.n_rows = len(X)
    report.n_cols_original = len(X.columns)

    if len(X) == 0 or len(X.columns) == 0:
        raise ValueError(
            "Input DataFrame is empty "
            f"(rows={len(X)}, columns={len(X.columns)})."
        )

    y_arr = np.asarray(y)

    if len(X) != len(y_arr):
        raise ValueError(
            f"Length mismatch: X has {len(X)} rows but y has {len(y_arr)} elements."
        )

    if np.issubdtype(y_arr.dtype, np.floating):
        has_nan = np.any(np.isnan(y_arr))
    else:
        has_nan = pd.Series(y_arr).isna().any()

    if has_nan:
        raise ValueError("Target variable y contains NaN values.")

    n_unique = len(np.unique(y_arr))

    if task in ("binary", "multiclass") and n_unique < 2:
        raise ValueError(
            f"Classification task '{task}' requires at least 2 classes "
            f"in y, but found {n_unique}."
        )

    if task == "binary" and n_unique != 2:
        raise ValueError(
            f"Binary task requires exactly 2 classes in y, "
            f"but found {n_unique}."
        )

    allnan_cols = [col for col in X.columns if X[col].isna().all()]
    if allnan_cols:
        report.dropped_allnan_columns = allnan_cols
        for col in allnan_cols:
            msg = f"Dropped all-NaN column: '{col}'"
            report.warnings.append(msg)
        X = X.drop(columns=allnan_cols)

    constant_cols = []
    for col in X.columns:
        non_null = X[col].dropna()
        if len(non_null) == 0:
            continue
        if non_null.nunique() <= 1:
            constant_cols.append(col)

    if constant_cols:
        report.dropped_constant_columns = constant_cols
        for col in constant_cols:
            msg = f"Dropped constant column: '{col}'"
            report.warnings.append(msg)
        X = X.drop(columns=constant_cols)

    report.n_cols_after = len(X.columns)

    if len(X.columns) == 0:
        raise ValueError(
            "No columns remain after dropping constant and all-NaN columns."
        )

    return X, report


def save_validation_report(report: ValidationReport, path: str) -> None:
    """Save a validation report to a JSON file.

    Parameters
    ----------
    report : ValidationReport
        The report to save.
    path : str
        Output file path.
    """
    with open(path, "w") as fh:
        json.dump(asdict(report), fh, indent=2)
