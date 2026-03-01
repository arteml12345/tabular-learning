"""From-scratch MLP embedding model (Mode A).

A tabular MLP with learned per-column categorical embeddings,
trained end-to-end as a standalone predictor. After training,
embeddings are extracted from a configurable intermediate hidden
layer for consumption by classical models.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hybrid_embed.embed.base import EmbeddingModel, TabularDataset
from hybrid_embed.eval.metrics import compute_metrics, get_primary_metric_name, is_higher_better


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}

_ENCODE_BATCH_SIZE = 4096


class TabularMLP(nn.Module):
    """PyTorch module: MLP with categorical embedding tables.

    Trained as a standalone predictor for the target task. After
    training, embeddings are extracted from a configurable
    intermediate hidden layer.

    Architecture:

    1. Per-categorical-column embedding table (``nn.Embedding``)
    2. Numeric input projected through a linear layer
    3. Concatenate all embeddings + numeric projection
    4. MLP trunk: ``[Linear -> Norm? -> Activation -> Dropout]``
       (hidden layers indexed ``0..n-1``)
    5. Output head: ``Linear`` -> task output

    Parameters
    ----------
    n_numeric : int
        Number of numeric input features.
    n_categories_per_column : list[int]
        Vocabulary size per categorical column (including special
        tokens).
    embedding_dim : int
        Dimension for each categorical embedding table.
    hidden_dims : list[int]
        Width of each hidden layer in the MLP trunk.
    dropout : float
        Dropout rate applied after activation in each hidden layer.
    activation : str
        ``"relu"``, ``"gelu"``, or ``"silu"``.
    use_layer_norm : bool
        Whether to apply ``LayerNorm`` before activation.
    task : str
        ``"binary"``, ``"multiclass"``, or ``"regression"``.
    n_classes : int or None
        Number of classes for classification output head.
    embedding_layer : int
        Index of hidden layer to extract embeddings from.
        Default ``-1`` (last hidden layer before prediction head).
    """

    def __init__(
        self,
        n_numeric: int,
        n_categories_per_column: list[int],
        embedding_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
        activation: str = "relu",
        use_layer_norm: bool = False,
        task: str = "binary",
        n_classes: int | None = None,
        embedding_layer: int = -1,
    ):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.task = task
        self.hidden_dims = hidden_dims

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(n_cat, embedding_dim)
            for n_cat in n_categories_per_column
        ])

        cat_total_dim = len(n_categories_per_column) * embedding_dim
        input_dim = n_numeric + cat_total_dim

        act_class = _ACTIVATIONS.get(activation, nn.ReLU)

        trunk_layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            trunk_layers.append(nn.Linear(in_dim, h_dim))
            if use_layer_norm:
                trunk_layers.append(nn.LayerNorm(h_dim))
            trunk_layers.append(act_class())
            trunk_layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.trunk = nn.ModuleList(trunk_layers)
        self._layers_per_block = 4 if use_layer_norm else 3

        if task == "binary":
            self.head = nn.Linear(hidden_dims[-1], 1)
        elif task == "multiclass":
            self.head = nn.Linear(hidden_dims[-1], n_classes)
        else:
            self.head = nn.Linear(hidden_dims[-1], 1)

    def _input_layer(self, x_numeric: torch.Tensor, x_categorical: torch.Tensor) -> torch.Tensor:
        """Combine numeric and categorical inputs into a single vector."""
        parts = []
        if x_numeric.shape[1] > 0:
            parts.append(x_numeric)
        for i, emb in enumerate(self.cat_embeddings):
            parts.append(emb(x_categorical[:, i]))
        return torch.cat(parts, dim=1)

    def _run_trunk_up_to(self, x: torch.Tensor, up_to_block: int) -> torch.Tensor:
        """Run through trunk blocks up to and including the given block index."""
        n_blocks = len(self.hidden_dims)
        if up_to_block < 0:
            up_to_block = n_blocks + up_to_block

        end_layer = (up_to_block + 1) * self._layers_per_block
        for layer in list(self.trunk)[:end_layer]:
            x = layer(x)
        return x

    def forward(self, x_numeric: torch.Tensor, x_categorical: torch.Tensor) -> torch.Tensor:
        """Full forward pass for training.

        Parameters
        ----------
        x_numeric : torch.Tensor
            Shape ``(batch, n_numeric)``, float32.
        x_categorical : torch.Tensor
            Shape ``(batch, n_cat)``, long.

        Returns
        -------
        torch.Tensor
            Task predictions (logits or values).
        """
        x = self._input_layer(x_numeric, x_categorical)
        for layer in self.trunk:
            x = layer(x)
        return self.head(x)

    def encode(self, x_numeric: torch.Tensor, x_categorical: torch.Tensor) -> torch.Tensor:
        """Forward pass up to ``embedding_layer``.

        Returns the activated output of the specified hidden layer
        block, before the prediction head.

        Parameters
        ----------
        x_numeric : torch.Tensor
        x_categorical : torch.Tensor

        Returns
        -------
        torch.Tensor
            Shape ``(batch, hidden_dims[embedding_layer])``.
        """
        x = self._input_layer(x_numeric, x_categorical)
        return self._run_trunk_up_to(x, self.embedding_layer)


class MLPEmbeddingModel(EmbeddingModel):
    """EmbeddingModel wrapper around TabularMLP.

    Handles the training loop, early stopping, device management,
    and implements the full ``EmbeddingModel`` interface.

    The model is trained as a standalone predictor (with a
    prediction head for the target task). After training,
    embeddings are extracted from a configurable intermediate
    hidden layer.

    Parameters
    ----------
    task : str
        ``"binary"``, ``"multiclass"``, or ``"regression"``.
    n_numeric : int
        Number of numeric features.
    n_categories_per_column : list[int]
        Vocabulary sizes per categorical column.
    n_classes : int or None
        Number of classes (classification only).
    """

    def __init__(
        self,
        task: str,
        n_numeric: int,
        n_categories_per_column: list[int],
        n_classes: int | None = None,
    ):
        self.task = task
        self.n_numeric = n_numeric
        self.n_categories_per_column = n_categories_per_column
        self.n_classes = n_classes
        self._model: TabularMLP | None = None
        self._device: torch.device = torch.device("cpu")
        self._history: dict[str, list] = {}
        self.embedding_dim: int = 0

    def fit(
        self,
        X_train: dict,
        y_train: np.ndarray,
        X_val: dict,
        y_val: np.ndarray,
        config: dict,
    ) -> "MLPEmbeddingModel":
        """Train the MLP as a standalone predictor.

        Parameters
        ----------
        X_train : dict
            Preprocessed training data.
        y_train : np.ndarray
        X_val : dict
            Preprocessed validation data.
        y_val : np.ndarray
        config : dict
            Hyperparameters. Expected keys: ``embedding_dim``,
            ``hidden_dims``, ``dropout``, ``activation``,
            ``use_layer_norm``, ``lr``, ``weight_decay``,
            ``batch_size``, ``max_epochs``, ``patience``,
            ``device``. Optional: ``embedding_layer``.

        Returns
        -------
        self
        """
        device_str = config.get("device", "cpu")
        self._device = torch.device(device_str)

        embedding_dim = int(config.get("embedding_dim", 64))
        hidden_dims = config.get("hidden_dims", [128, 64])
        dropout = float(config.get("dropout", 0.1))
        activation = config.get("activation", "relu")
        use_layer_norm = bool(config.get("use_layer_norm", False))
        embedding_layer = int(config.get("embedding_layer", -1))
        lr = float(config.get("lr", 1e-3))
        weight_decay = float(config.get("weight_decay", 1e-5))
        batch_size = int(config.get("batch_size", 256))
        max_epochs = int(config.get("max_epochs", 50))
        patience = int(config.get("patience", 8))

        self._model = TabularMLP(
            n_numeric=self.n_numeric,
            n_categories_per_column=self.n_categories_per_column,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            use_layer_norm=use_layer_norm,
            task=self.task,
            n_classes=self.n_classes,
            embedding_layer=embedding_layer,
        ).to(self._device)

        resolved_layer = embedding_layer
        if resolved_layer < 0:
            resolved_layer = len(hidden_dims) + resolved_layer
        self.embedding_dim = hidden_dims[resolved_layer]

        loss_fn = self._get_loss_fn()
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=lr, weight_decay=weight_decay,
        )

        train_dataset = TabularDataset(X_train, y_train)
        val_dataset = TabularDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        primary_metric = get_primary_metric_name(self.task)
        higher_better = is_higher_better(primary_metric)

        best_score = float("-inf") if higher_better else float("inf")
        best_weights = None
        epochs_no_improve = 0

        self._history = {"train_loss": [], "val_loss": [], "val_metric": []}

        for epoch in range(max_epochs):
            train_loss = self._train_one_epoch(train_loader, loss_fn, optimizer)
            val_loss, val_metrics = self._evaluate(val_loader, loss_fn, y_val)

            val_score = val_metrics[primary_metric]
            self._history["train_loss"].append(train_loss)
            self._history["val_loss"].append(val_loss)
            self._history["val_metric"].append(val_score)

            improved = (
                (higher_better and val_score > best_score)
                or (not higher_better and val_score < best_score)
            )
            if improved:
                best_score = val_score
                best_weights = copy.deepcopy(self._model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        if best_weights is not None:
            self._model.load_state_dict(best_weights)

        return self

    def predict(self, X: dict) -> np.ndarray:
        """Predict using the full model.

        Parameters
        ----------
        X : dict
            Preprocessed data.

        Returns
        -------
        np.ndarray
            Class labels (classification) or values (regression).
        """
        raw = self._predict_raw(X)
        if self.task == "binary":
            proba = torch.sigmoid(torch.tensor(raw)).numpy()
            return (proba >= 0.5).astype(int)
        elif self.task == "multiclass":
            return np.argmax(raw, axis=1)
        else:
            return raw

    def predict_proba(self, X: dict) -> np.ndarray:
        """Predict probabilities (classification only).

        Parameters
        ----------
        X : dict
            Preprocessed data.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_classes)`` for multiclass or
            ``(n_samples, 2)`` for binary.
        """
        raw = self._predict_raw(X)
        if self.task == "binary":
            p1 = torch.sigmoid(torch.tensor(raw)).numpy()
            return np.column_stack([1.0 - p1, p1])
        elif self.task == "multiclass":
            logits = torch.tensor(raw)
            return torch.softmax(logits, dim=1).numpy()
        else:
            raise NotImplementedError("predict_proba not available for regression")

    def encode(self, X: dict) -> np.ndarray:
        """Extract embeddings using the trained model.

        Runs ``model.encode()`` in batches with gradients disabled,
        returns a CPU numpy array.

        Parameters
        ----------
        X : dict
            Preprocessed data.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, self.embedding_dim)``.
        """
        self._model.eval()
        numeric = torch.tensor(X["numeric"], dtype=torch.float32)
        categorical = torch.tensor(X["categorical"], dtype=torch.long)

        parts = []
        n = len(numeric)
        with torch.no_grad():
            for start in range(0, n, _ENCODE_BATCH_SIZE):
                end = min(start + _ENCODE_BATCH_SIZE, n)
                x_num = numeric[start:end].to(self._device)
                x_cat = categorical[start:end].to(self._device)
                emb = self._model.encode(x_num, x_cat)
                parts.append(emb.cpu())

        return torch.cat(parts, dim=0).numpy()

    def save(self, path: str) -> None:
        """Save model checkpoint to disk.

        Parameters
        ----------
        path : str
            File path for the checkpoint.
        """
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "task": self.task,
            "n_numeric": self.n_numeric,
            "n_categories_per_column": self.n_categories_per_column,
            "n_classes": self.n_classes,
            "embedding_dim": self.embedding_dim,
            "model_config": {
                "embedding_dim": self._model.cat_embeddings[0].embedding_dim if self._model.cat_embeddings else 0,
                "hidden_dims": self._model.hidden_dims,
                "embedding_layer": self._model.embedding_layer,
                "task": self._model.task,
            },
        }, path)

    @classmethod
    def load(cls, path: str) -> "MLPEmbeddingModel":
        """Load model checkpoint from disk.

        Parameters
        ----------
        path : str
            File path to the saved checkpoint.

        Returns
        -------
        MLPEmbeddingModel
        """
        checkpoint = torch.load(path, weights_only=False)
        instance = cls(
            task=checkpoint["task"],
            n_numeric=checkpoint["n_numeric"],
            n_categories_per_column=checkpoint["n_categories_per_column"],
            n_classes=checkpoint["n_classes"],
        )
        instance.embedding_dim = checkpoint["embedding_dim"]

        mc = checkpoint["model_config"]
        instance._model = TabularMLP(
            n_numeric=instance.n_numeric,
            n_categories_per_column=instance.n_categories_per_column,
            embedding_dim=mc["embedding_dim"],
            hidden_dims=mc["hidden_dims"],
            embedding_layer=mc["embedding_layer"],
            task=mc["task"],
            n_classes=instance.n_classes,
        )
        instance._model.load_state_dict(checkpoint["model_state_dict"])
        instance._device = torch.device("cpu")
        return instance

    def get_training_history(self) -> dict:
        """Return per-epoch training history.

        Returns
        -------
        dict
            Keys: ``"train_loss"``, ``"val_loss"``, ``"val_metric"``.
        """
        return self._history

    def _get_loss_fn(self) -> nn.Module:
        """Return the appropriate loss function for the task."""
        if self.task == "binary":
            return nn.BCEWithLogitsLoss()
        elif self.task == "multiclass":
            return nn.CrossEntropyLoss()
        else:
            return nn.MSELoss()

    def _train_one_epoch(
        self,
        loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one training epoch, return average loss."""
        self._model.train()
        total_loss = 0.0
        n_batches = 0

        for x_num, x_cat, y_batch in loader:
            x_num = x_num.to(self._device)
            x_cat = x_cat.to(self._device)
            y_batch = y_batch.to(self._device)

            optimizer.zero_grad()
            output = self._model(x_num, x_cat)

            if self.task == "binary":
                loss = loss_fn(output.squeeze(-1), y_batch)
            elif self.task == "multiclass":
                loss = loss_fn(output, y_batch.long())
            else:
                loss = loss_fn(output.squeeze(-1), y_batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _evaluate(
        self,
        loader: DataLoader,
        loss_fn: nn.Module,
        y_val_full: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Evaluate on validation set, return (avg_loss, metrics_dict)."""
        self._model.eval()
        total_loss = 0.0
        n_batches = 0
        all_outputs = []

        with torch.no_grad():
            for x_num, x_cat, y_batch in loader:
                x_num = x_num.to(self._device)
                x_cat = x_cat.to(self._device)
                y_batch = y_batch.to(self._device)

                output = self._model(x_num, x_cat)

                if self.task == "binary":
                    loss = loss_fn(output.squeeze(-1), y_batch)
                elif self.task == "multiclass":
                    loss = loss_fn(output, y_batch.long())
                else:
                    loss = loss_fn(output.squeeze(-1), y_batch)

                total_loss += loss.item()
                n_batches += 1
                all_outputs.append(output.cpu())

        avg_loss = total_loss / max(n_batches, 1)
        raw_output = torch.cat(all_outputs, dim=0).numpy()

        if self.task == "binary":
            proba = torch.sigmoid(torch.tensor(raw_output.squeeze(-1))).numpy()
            y_pred = (proba >= 0.5).astype(int)
            metrics = compute_metrics(y_val_full, y_pred, proba, task="binary")
        elif self.task == "multiclass":
            proba = torch.softmax(torch.tensor(raw_output), dim=1).numpy()
            y_pred = np.argmax(raw_output, axis=1)
            metrics = compute_metrics(y_val_full, y_pred, proba, task="multiclass")
        else:
            y_pred = raw_output.squeeze(-1)
            metrics = compute_metrics(y_val_full, y_pred, None, task="regression")

        return avg_loss, metrics

    def _predict_raw(self, X: dict) -> np.ndarray:
        """Run forward pass in batches, return raw model output."""
        self._model.eval()
        numeric = torch.tensor(X["numeric"], dtype=torch.float32)
        categorical = torch.tensor(X["categorical"], dtype=torch.long)

        parts = []
        n = len(numeric)
        with torch.no_grad():
            for start in range(0, n, _ENCODE_BATCH_SIZE):
                end = min(start + _ENCODE_BATCH_SIZE, n)
                x_num = numeric[start:end].to(self._device)
                x_cat = categorical[start:end].to(self._device)
                out = self._model(x_num, x_cat)
                parts.append(out.cpu())

        return torch.cat(parts, dim=0).numpy()
