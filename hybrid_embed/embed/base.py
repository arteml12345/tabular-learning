"""Embedding model base class and tabular dataset helper.

Defines the abstract interface that all embedding models (both
from-scratch and library-backed) must implement, plus a PyTorch
Dataset wrapper for preprocessed tabular data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingModel(ABC):
    """Abstract base class for tabular embedding models.

    All embedding models must implement ``fit``, ``predict``,
    ``encode``, ``save``, and ``load``. The ``encode`` method
    extracts a fixed-length dense vector per sample that will
    be used as the only input to classical models.

    Subclasses must set ``self.embedding_dim`` after ``fit()``.

    Attributes
    ----------
    embedding_dim : int
        Dimensionality of the embedding vectors produced by
        ``encode()``. Set during ``fit()``.
    """

    embedding_dim: int

    @abstractmethod
    def fit(
        self,
        X_train: dict,
        y_train: np.ndarray,
        X_val: dict,
        y_val: np.ndarray,
        config: dict,
    ) -> "EmbeddingModel":
        """Train the embedding model as a standalone predictor.

        Parameters
        ----------
        X_train : dict
            Keys: ``"numeric"`` (np.ndarray, shape ``(n, n_num)``),
            ``"categorical"`` (np.ndarray, shape ``(n, n_cat)``,
            dtype int64).
        y_train : np.ndarray
            Training target values.
        X_val : dict
            Same format as ``X_train``.
        y_val : np.ndarray
            Validation target values.
        config : dict
            Model hyperparameters.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, X: dict) -> np.ndarray:
        """Predict using the full model (for HPO evaluation).

        Parameters
        ----------
        X : dict
            Same format as ``fit`` inputs.

        Returns
        -------
        np.ndarray
            Predictions (class labels or continuous values).
        """

    def predict_proba(self, X: dict) -> np.ndarray:
        """Predict probabilities (classification only).

        Default raises ``NotImplementedError``. Subclasses for
        classification tasks should override.

        Parameters
        ----------
        X : dict
            Same format as ``fit`` inputs.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_classes)``.
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, X: dict) -> np.ndarray:
        """Extract embeddings from an intermediate layer.

        The model is trained as a standalone predictor. This method
        returns the learned internal representation from a specific
        layer (see spec Section 7.4).

        Parameters
        ----------
        X : dict
            Same format as ``fit`` inputs.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, self.embedding_dim)``.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model state to disk.

        Parameters
        ----------
        path : str
            File path for the checkpoint.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "EmbeddingModel":
        """Load model state from disk.

        Parameters
        ----------
        path : str
            File path to the saved checkpoint.

        Returns
        -------
        EmbeddingModel
            The restored model instance.
        """

    def get_training_history(self) -> dict:
        """Return training history (losses, metrics per epoch).

        Default implementation returns an empty dict. Subclasses
        should override to provide epoch-level training logs.

        Returns
        -------
        dict
        """
        return {}


class TabularDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed tabular data.

    Converts the preprocessed dict format (numpy arrays) into
    PyTorch tensors suitable for DataLoader consumption.

    Parameters
    ----------
    X : dict
        ``"numeric"``: np.ndarray of shape ``(n, n_num)``.
        ``"categorical"``: np.ndarray of shape ``(n, n_cat)``,
        dtype int64.
    y : np.ndarray
        Target values.
    """

    def __init__(self, X: dict, y: np.ndarray):
        self.numeric = torch.tensor(
            X["numeric"], dtype=torch.float32,
        )
        self.categorical = torch.tensor(
            X["categorical"], dtype=torch.long,
        )
        self.targets = torch.tensor(
            np.asarray(y, dtype=np.float32), dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (numeric, categorical, target) tensors for one sample.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        return self.numeric[idx], self.categorical[idx], self.targets[idx]
