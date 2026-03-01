"""Preprocessing: scaling, imputation, categorical encoding, target handling.

Implements the minimal, non-expert preprocessing allowed by the spec:
numeric imputation (median), numeric scaling (Standard/Robust),
categorical vocabulary mapping with special tokens, and target
variable encoding/scaling.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

from hybrid_embed.config import Schema


OOV_INDEX = 0
MISSING_INDEX = 1
VOCAB_OFFSET = 2


class TabularPreprocessor:
    """Preprocess tabular data for embedding model consumption.

    Fitted on training data only. Transforms train/val/test
    consistently. Implements the sklearn fit/transform pattern.

    Parameters
    ----------
    schema : Schema
        Inferred column types.
    task : str
        ``"binary"``, ``"multiclass"``, or ``"regression"``.
    scaler_type : str, default "standard"
        ``"standard"`` or ``"robust"``.
    max_categorical_cardinality : int, default 1000
        Cap vocabulary for high-cardinality categoricals. Values
        beyond the top-N most frequent are mapped to ``__OOV__``.
    add_missing_indicator : bool, default False
        If True, add binary indicator columns for each numeric
        feature that had missing values in the training data.
    """

    def __init__(
        self,
        schema: Schema,
        task: str,
        scaler_type: str = "standard",
        max_categorical_cardinality: int = 1000,
        add_missing_indicator: bool = False,
    ):
        self.schema = schema
        self.task = task
        self.scaler_type = scaler_type
        self.max_categorical_cardinality = max_categorical_cardinality
        self.add_missing_indicator = add_missing_indicator

        self._numeric_medians: dict[str, float] = {}
        self._scaler = None
        self._cat_vocabs: dict[str, dict] = {}
        self._target_encoder = None
        self._target_scaler = None
        self._n_classes: int | None = None
        self._missing_indicator_cols: list[str] = []

    def fit(
        self, X_train: pd.DataFrame, y_train: np.ndarray | None = None,
    ) -> "TabularPreprocessor":
        """Fit preprocessor on training data.

        Learns median per numeric column (for imputation), scaler
        parameters, vocabulary per categorical column (with
        cardinality cap), and target encoder/scaler.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : np.ndarray or None
            Training target. Required for target encoding/scaling.

        Returns
        -------
        self
        """
        self._fit_numeric(X_train)
        self._fit_categorical(X_train)
        if y_train is not None:
            self._fit_target(y_train)
        return self

    def transform(self, X: pd.DataFrame) -> dict:
        """Transform a DataFrame into model-ready format.

        Parameters
        ----------
        X : pd.DataFrame
            Raw features (train, val, or test).

        Returns
        -------
        dict
            ``"numeric"``: ``np.ndarray`` of shape
            ``(n_samples, n_numeric_features)``, scaled and imputed.
            ``"categorical"``: ``np.ndarray`` of shape
            ``(n_samples, n_categorical)``, dtype ``int64``,
            integer-encoded.
        """
        result = {}

        if self.schema.numeric_columns:
            numeric = self._transform_numeric(X)
            result["numeric"] = numeric
        else:
            result["numeric"] = np.zeros((len(X), 0), dtype=np.float64)

        if self.schema.categorical_columns:
            categorical = self._transform_categorical(X)
            result["categorical"] = categorical
        else:
            result["categorical"] = np.zeros((len(X), 0), dtype=np.int64)

        return result

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        """Transform target variable.

        For classification, maps labels to contiguous integers
        ``0..K-1`` via ``LabelEncoder``. For regression, applies
        ``StandardScaler`` (zero mean, unit variance).

        Parameters
        ----------
        y : np.ndarray
            Raw target values.

        Returns
        -------
        np.ndarray
            Transformed target.
        """
        if self.task in ("binary", "multiclass"):
            return self._target_encoder.transform(y)
        else:
            y_2d = np.asarray(y, dtype=np.float64).reshape(-1, 1)
            return self._target_scaler.transform(y_2d).ravel()

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Inverse-transform target (for regression predictions).

        Parameters
        ----------
        y : np.ndarray
            Transformed target values.

        Returns
        -------
        np.ndarray
            Original-scale values.
        """
        if self.task in ("binary", "multiclass"):
            return self._target_encoder.inverse_transform(y)
        else:
            y_2d = np.asarray(y, dtype=np.float64).reshape(-1, 1)
            return self._target_scaler.inverse_transform(y_2d).ravel()

    def get_n_categories_per_column(self) -> list[int]:
        """Return vocabulary size per categorical column.

        Each size includes the ``__OOV__`` (index 0) and
        ``__MISSING__`` (index 1) special tokens, plus the
        real vocabulary entries.

        Returns
        -------
        list[int]
            One entry per categorical column in schema order.
        """
        sizes = []
        for col in self.schema.categorical_columns:
            n_real = len(self._cat_vocabs[col])
            sizes.append(n_real + VOCAB_OFFSET)
        return sizes

    def get_n_numeric_features(self) -> int:
        """Return number of numeric features after preprocessing.

        Includes missing indicator columns if enabled.

        Returns
        -------
        int
        """
        n = len(self.schema.numeric_columns)
        if self.add_missing_indicator:
            n += len(self._missing_indicator_cols)
        return n

    @property
    def n_classes(self) -> int | None:
        """Number of classes (classification) or None (regression)."""
        return self._n_classes

    def _fit_numeric(self, X_train: pd.DataFrame) -> None:
        """Learn medians and scaler from training numeric columns."""
        if not self.schema.numeric_columns:
            return

        self._missing_indicator_cols = []
        for col in self.schema.numeric_columns:
            median_val = X_train[col].median()
            self._numeric_medians[col] = median_val
            if self.add_missing_indicator and X_train[col].isna().any():
                self._missing_indicator_cols.append(col)

        imputed = X_train[self.schema.numeric_columns].copy()
        for col in self.schema.numeric_columns:
            imputed[col] = imputed[col].fillna(self._numeric_medians[col])

        if self.scaler_type == "robust":
            self._scaler = RobustScaler()
        else:
            self._scaler = StandardScaler()

        self._scaler.fit(imputed.values)

    def _transform_numeric(self, X: pd.DataFrame) -> np.ndarray:
        """Impute, scale, and optionally add missing indicators."""
        imputed = X[self.schema.numeric_columns].copy()

        indicator_arrays = []
        if self.add_missing_indicator:
            for col in self._missing_indicator_cols:
                indicator_arrays.append(imputed[col].isna().astype(np.float64).values)

        for col in self.schema.numeric_columns:
            imputed[col] = imputed[col].fillna(self._numeric_medians[col])

        scaled = self._scaler.transform(imputed.values)

        if indicator_arrays:
            indicators = np.column_stack(indicator_arrays)
            scaled = np.hstack([scaled, indicators])

        return scaled

    def _fit_categorical(self, X_train: pd.DataFrame) -> None:
        """Build vocabulary per categorical column from training data."""
        for col in self.schema.categorical_columns:
            series = X_train[col].dropna()
            counts = Counter(series)

            if len(counts) > self.max_categorical_cardinality:
                top_values = [
                    val for val, _ in counts.most_common(self.max_categorical_cardinality)
                ]
            else:
                top_values = list(counts.keys())

            vocab = {val: idx + VOCAB_OFFSET for idx, val in enumerate(top_values)}
            self._cat_vocabs[col] = vocab

    def _transform_categorical(self, X: pd.DataFrame) -> np.ndarray:
        """Map categorical values to integer indices."""
        n_samples = len(X)
        n_cats = len(self.schema.categorical_columns)
        encoded = np.zeros((n_samples, n_cats), dtype=np.int64)

        for j, col in enumerate(self.schema.categorical_columns):
            vocab = self._cat_vocabs[col]
            for i in range(n_samples):
                val = X[col].iloc[i]
                if pd.isna(val):
                    encoded[i, j] = MISSING_INDEX
                elif val in vocab:
                    encoded[i, j] = vocab[val]
                else:
                    encoded[i, j] = OOV_INDEX

        return encoded

    def _fit_target(self, y_train: np.ndarray) -> None:
        """Fit target encoder (classification) or scaler (regression)."""
        if self.task in ("binary", "multiclass"):
            self._target_encoder = LabelEncoder()
            self._target_encoder.fit(y_train)
            self._n_classes = len(self._target_encoder.classes_)
        else:
            self._target_scaler = StandardScaler()
            y_2d = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
            self._target_scaler.fit(y_2d)
            self._n_classes = None
