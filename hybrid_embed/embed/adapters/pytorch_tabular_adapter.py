"""pytorch-tabular adapter (Mode B).

Wraps pytorch-tabular library models behind the framework's
``EmbeddingModel`` interface.  Supports TabTransformer,
CategoryEmbedding, TabNet, and GATE architectures.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any

import numpy as np
import pandas as pd
import torch

from hybrid_embed.config import Schema
from hybrid_embed.embed.base import EmbeddingModel

logger = logging.getLogger(__name__)

_PT_TASK_MAP = {
    "binary": "classification",
    "multiclass": "classification",
    "regression": "regression",
}


def _get_model_config_map() -> dict:
    """Lazy-import pytorch-tabular model configs to avoid top-level import overhead."""
    from pytorch_tabular.models import (
        CategoryEmbeddingModelConfig,
        GatedAdditiveTreeEnsembleConfig,
        TabNetModelConfig,
        TabTransformerConfig,
    )
    return {
        "tab_transformer": TabTransformerConfig,
        "category_embedding": CategoryEmbeddingModelConfig,
        "tabnet": TabNetModelConfig,
        "gate": GatedAdditiveTreeEnsembleConfig,
    }


def _build_model_config(model_type: str, task: str, config: dict) -> Any:
    """Build the pytorch-tabular model config from HPO params."""
    config_map = _get_model_config_map()
    if model_type not in config_map:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Supported: {list(config_map.keys())}"
        )

    pt_task = _PT_TASK_MAP[task]
    cfg_class = config_map[model_type]

    lr = config.get("lr", 1e-3)

    if model_type == "tab_transformer":
        return cfg_class(
            task=pt_task,
            input_embed_dim=config.get("input_embed_dim", 32),
            num_heads=config.get("num_heads", 4),
            num_attn_blocks=config.get("num_attn_blocks", 2),
            attn_dropout=config.get("attn_dropout", 0.1),
            ff_dropout=config.get("ff_dropout", 0.1),
            share_embedding=config.get("add_shared_embedding", False),
            shared_embedding_fraction=config.get("shared_embedding_fraction", 0.25),
            learning_rate=lr,
        )

    if model_type == "category_embedding":
        layers = config.get("layers", "128-64")
        activation = config.get("activation", "ReLU")
        dropout = config.get("dropout", 0.1)
        return cfg_class(
            task=pt_task,
            layers=layers,
            activation=activation,
            dropout=dropout,
            learning_rate=lr,
        )

    if model_type == "tabnet":
        return cfg_class(
            task=pt_task,
            n_d=config.get("n_d", 16),
            n_a=config.get("n_a", 16),
            n_steps=config.get("n_steps", 3),
            gamma=config.get("gamma", 1.3),
            learning_rate=lr,
        )

    if model_type == "gate":
        return cfg_class(
            task=pt_task,
            gflu_stages=config.get("gflu_stages", 6),
            num_trees=config.get("num_trees", 20),
            tree_depth=config.get("tree_depth", 5),
            learning_rate=lr,
        )

    raise ValueError(f"Unhandled model_type: {model_type}")


class PytorchTabularEmbeddingModel(EmbeddingModel):
    """Adapter wrapping pytorch-tabular models as EmbeddingModel.

    Translates between the hybrid_embed interface (preprocessed dict
    inputs) and pytorch-tabular's DataFrame + config interface.

    Parameters
    ----------
    model_type : str
        Which pytorch-tabular architecture to use.  Supported:
        ``"tab_transformer"``, ``"category_embedding"``, ``"tabnet"``,
        ``"gate"``.
    task : str
        ``"binary"``, ``"multiclass"``, or ``"regression"``.
    schema : Schema
        Column type information (used for DataFrame reconstruction).
    n_classes : int or None
        Number of classes (classification only).
    """

    def __init__(
        self,
        model_type: str,
        task: str,
        schema: Schema,
        n_classes: int | None = None,
    ):
        if model_type not in _get_model_config_map():
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Supported: {list(_get_model_config_map().keys())}"
            )
        self.model_type = model_type
        self.task = task
        self.schema = schema
        self.n_classes = n_classes
        self._tabular_model = None
        self.embedding_dim: int = 0
        self._num_col_names: list[str] = []
        self._cat_col_names: list[str] = []
        self._target_col = "__target__"

    def fit(
        self,
        X_train: dict,
        y_train: np.ndarray,
        X_val: dict,
        y_val: np.ndarray,
        config: dict,
    ) -> "PytorchTabularEmbeddingModel":
        """Train via pytorch-tabular.

        Parameters
        ----------
        X_train : dict
            Preprocessed training data.
        y_train : np.ndarray
        X_val : dict
            Preprocessed validation data.
        y_val : np.ndarray
        config : dict
            Hyperparameters.

        Returns
        -------
        self
        """
        from pytorch_tabular import TabularModel
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

        self._num_col_names = [
            f"num_{i}" for i in range(X_train["numeric"].shape[1])
        ]
        self._cat_col_names = [
            f"cat_{i}" for i in range(X_train["categorical"].shape[1])
        ]

        train_df = self._to_dataframe(X_train, y_train)
        val_df = self._to_dataframe(X_val, y_val)

        data_config = DataConfig(
            target=[self._target_col],
            continuous_cols=self._num_col_names if self._num_col_names else None,
            categorical_cols=self._cat_col_names if self._cat_col_names else None,
        )

        model_config = _build_model_config(self.model_type, self.task, config)

        max_epochs = int(config.get("max_epochs", 50))
        patience = config.get("patience", 8)
        batch_size = int(config.get("batch_size", 256))
        device_str = config.get("device", "cpu")

        accelerator = "cpu"
        if device_str.startswith("cuda"):
            accelerator = "gpu"
        elif device_str == "mps":
            accelerator = "mps"

        if patience is not None:
            early_stopping = "valid_loss"
            es_patience = int(patience)
        else:
            early_stopping = None
            es_patience = 3

        trainer_config = TrainerConfig(
            max_epochs=max_epochs,
            batch_size=batch_size,
            early_stopping=early_stopping,
            early_stopping_patience=es_patience,
            checkpoints=None,
            progress_bar="none",
            accelerator=accelerator,
        )

        optimizer_config = OptimizerConfig(
            optimizer="AdamW",
            optimizer_params={
                "weight_decay": config.get("weight_decay", 1e-5),
            },
        )

        logging.getLogger("pytorch_tabular").setLevel(logging.WARNING)
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        logging.getLogger("lightning").setLevel(logging.WARNING)

        self._tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )

        self._tabular_model.fit(train=train_df, validation=val_df)

        self.embedding_dim = self._determine_embedding_dim(val_df)

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
        df = self._to_dataframe(X)
        result = self._tabular_model.predict(df, progress_bar="none")
        if self.task in ("binary", "multiclass"):
            return result[f"{self._target_col}_prediction"].values.astype(int)
        else:
            return result[f"{self._target_col}_prediction"].values.astype(float)

    def predict_proba(self, X: dict) -> np.ndarray:
        """Predict probabilities (classification only).

        Parameters
        ----------
        X : dict
            Preprocessed data.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_classes)``.
        """
        if self.task == "regression":
            raise NotImplementedError("predict_proba not available for regression")
        df = self._to_dataframe(X)
        result = self._tabular_model.predict(df, progress_bar="none")
        prob_cols = [c for c in result.columns if "_probability" in c]
        prob_cols.sort()
        return result[prob_cols].values.astype(float)

    def encode(self, X: dict) -> np.ndarray:
        """Extract backbone embeddings.

        Uses a forward hook on the model's backbone to capture
        the intermediate representation before the prediction head.

        Parameters
        ----------
        X : dict
            Preprocessed data.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, self.embedding_dim)``.
        """
        df = self._to_dataframe(X)
        embeddings = self._extract_backbone_output(df)
        return embeddings

    def save(self, path: str) -> None:
        """Save model checkpoint to disk.

        Parameters
        ----------
        path : str
            Directory path for the checkpoint.
        """
        os.makedirs(path, exist_ok=True)
        self._tabular_model.save_model(path)
        meta = {
            "model_type": self.model_type,
            "task": self.task,
            "schema": {
                "numeric_columns": self.schema.numeric_columns,
                "categorical_columns": self.schema.categorical_columns,
                "dropped_columns": self.schema.dropped_columns,
            },
            "n_classes": self.n_classes,
            "embedding_dim": self.embedding_dim,
            "num_col_names": self._num_col_names,
            "cat_col_names": self._cat_col_names,
        }
        with open(os.path.join(path, "adapter_meta.json"), "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "PytorchTabularEmbeddingModel":
        """Load model checkpoint from disk.

        Parameters
        ----------
        path : str
            Directory path to the saved checkpoint.

        Returns
        -------
        PytorchTabularEmbeddingModel
        """
        from pytorch_tabular import TabularModel

        with open(os.path.join(path, "adapter_meta.json")) as f:
            meta = json.load(f)

        schema = Schema(
            numeric_columns=meta["schema"]["numeric_columns"],
            categorical_columns=meta["schema"]["categorical_columns"],
            dropped_columns=meta["schema"]["dropped_columns"],
        )
        instance = cls(
            model_type=meta["model_type"],
            task=meta["task"],
            schema=schema,
            n_classes=meta["n_classes"],
        )
        instance.embedding_dim = meta["embedding_dim"]
        instance._num_col_names = meta["num_col_names"]
        instance._cat_col_names = meta["cat_col_names"]
        instance._tabular_model = TabularModel.load_model(path)
        return instance

    def get_training_history(self) -> dict:
        """Return training history from pytorch-tabular.

        Returns
        -------
        dict
        """
        if self._tabular_model is None:
            return {}
        try:
            trainer = self._tabular_model.trainer
            if trainer and trainer.callback_metrics:
                return {k: v.item() if hasattr(v, "item") else v
                        for k, v in trainer.callback_metrics.items()}
        except Exception:
            pass
        return {}

    def _to_dataframe(self, X: dict, y: np.ndarray | None = None) -> pd.DataFrame:
        """Convert preprocessed dict format to DataFrame."""
        data = {}
        for i, col_name in enumerate(self._num_col_names):
            data[col_name] = X["numeric"][:, i].astype(np.float64)
        for i, col_name in enumerate(self._cat_col_names):
            data[col_name] = X["categorical"][:, i].astype(str)

        df = pd.DataFrame(data)
        if y is not None:
            if self.task in ("binary", "multiclass"):
                df[self._target_col] = y.astype(int)
            else:
                df[self._target_col] = y.astype(float)
        return df

    def _extract_backbone_output(self, df: pd.DataFrame) -> np.ndarray:
        """Extract backbone representation using a forward hook."""
        model = self._tabular_model.model
        model.eval()

        captured = []

        if self.model_type == "tabnet":
            hook_target = model._backbone.tabnet.tabnet.final_mapping
            def hook_fn(module, inp, out):
                captured.append(inp[0].detach().cpu())
        else:
            hook_target = model._backbone
            def hook_fn(module, inp, out):
                t = out
                if isinstance(t, tuple):
                    t = t[0]
                if t.dim() > 2:
                    t = t.reshape(t.size(0), -1)
                captured.append(t.detach().cpu())

        handle = hook_target.register_forward_hook(hook_fn)
        try:
            _ = self._tabular_model.predict(df, progress_bar="none")
        finally:
            handle.remove()

        result = torch.cat(captured, dim=0).numpy()
        return result

    def _determine_embedding_dim(self, sample_df: pd.DataFrame) -> int:
        """Run a small forward pass to determine the backbone output dim."""
        small_df = sample_df.head(min(8, len(sample_df))).copy()
        emb = self._extract_backbone_output(small_df)
        return emb.shape[1]
