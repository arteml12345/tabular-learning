"""Configuration dataclasses and serialization helpers.

Defines all configuration, schema, and result dataclasses used
throughout the hybrid_embed framework. Also provides helpers for
validation, dict conversion, and YAML serialization.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml


VALID_TASKS = ("binary", "multiclass", "regression")


@dataclass
class Schema:
    """Inferred column type information for a dataset.

    Populated by the schema inference step (spec Section 5).
    Used downstream by preprocessing, embedding models, and
    the pytorch-tabular adapter.

    Parameters
    ----------
    numeric_columns : list[str]
        Column names identified as numeric (int/float).
    categorical_columns : list[str]
        Column names identified as categorical (object/string/
        low-cardinality int).
    dropped_columns : list[str]
        Columns removed during data validation (constant columns,
        all-NaN columns). Tracked for reproducibility and logging.
    """

    numeric_columns: list[str]
    categorical_columns: list[str]
    dropped_columns: list[str] = field(default_factory=list)


@dataclass
class TaskConfig:
    """Specifies the prediction task type.

    Parameters
    ----------
    task : str
        One of ``"binary"``, ``"multiclass"``, ``"regression"``.
    target_column : str or None
        Name of the target column when loading from CSV/DataFrame.
        None when target is passed separately as y.
    positive_class : Any or None
        For binary classification: override which label is treated
        as the positive class. If None, the framework infers it
        as the higher-valued (or alphabetically second) label.
    """

    task: str
    target_column: str | None = None
    positive_class: Any = None


@dataclass
class BudgetConfig:
    """HPO budget limits for both embedding and classical stages.

    Controls how much compute Hyperopt is allowed to spend.
    Time budgets are wall-clock limits; if exceeded, Hyperopt
    stops launching new trials and returns the best result so far.

    Parameters
    ----------
    embed_hpo_max_trials : int
        Maximum number of Hyperopt trials for the embedding model.
    embed_max_epochs : int
        Maximum training epochs per embedding trial (early stopping
        may stop sooner).
    embed_patience : int
        Early stopping patience (epochs without improvement).
    embed_time_budget_seconds : float or None
        Wall-clock time limit for embedding HPO. None = no limit.
    classical_hpo_max_trials : int
        Maximum number of Hyperopt trials for the classical model.
    classical_time_budget_seconds : float or None
        Wall-clock time limit for classical HPO. None = no limit.
    """

    embed_hpo_max_trials: int = 50
    embed_max_epochs: int = 50
    embed_patience: int = 8
    embed_time_budget_seconds: float | None = None
    classical_hpo_max_trials: int = 150
    classical_time_budget_seconds: float | None = None


@dataclass
class EmbeddingStepConfig:
    """Configuration for the embedding model step.

    The user selects ONE embedding model per experiment and
    optionally provides a custom HP search space and/or fixed
    parameters. If search_space is None, the framework uses the
    built-in default space for the selected model_type.

    Parameters
    ----------
    model_type : str
        Mode A (from-scratch): ``"mlp"``
        Mode B (pytorch-tabular): ``"tab_transformer"``, ``"tabnet"``,
        ``"gate"``, ``"category_embedding"``
    search_space : dict or None
        Hyperopt search space for this model's hyperparameters.
        If None, uses the built-in default space for model_type.
    fixed_params : dict or None
        Parameters always passed to the model (not tuned).
        These override defaults and are merged with HPO-sampled
        params. For MLP, can include ``"embedding_layer"`` to control
        which hidden layer embeddings are extracted from.
    """

    model_type: str = "mlp"
    search_space: dict | None = None
    fixed_params: dict | None = None


@dataclass
class ClassicalStepConfig:
    """Configuration for the classical model step.

    The user selects ONE classical model per experiment. The
    framework ships with default HP spaces for common models
    (spec Section 9.2). The user can also provide any
    sklearn-compatible class via ``model_type="custom"``.

    Parameters
    ----------
    model_type : str
        Built-in models (each has a default HP space):
        ``"logistic_regression"``, ``"ridge"``, ``"elastic_net"``,
        ``"random_forest"``, ``"extra_trees"``, ``"xgboost"``,
        ``"lightgbm"``, ``"catboost"``.
        Custom: ``"custom"`` (must also set model_class and
        search_space).
    model_class : type or None
        Required when ``model_type="custom"``. Must be
        sklearn-compatible (fit/predict/predict_proba).
    search_space : dict or None
        Hyperopt search space. If None, uses the built-in
        default space for model_type.
    fixed_params : dict or None
        Parameters always passed to the constructor (not tuned).
    supports_early_stopping : bool or None
        If True, framework passes eval_set and
        early_stopping_rounds to fit(). None = auto-detect
        (True for xgboost/lightgbm/catboost, False otherwise).
    """

    model_type: str = "xgboost"
    model_class: type | None = None
    search_space: dict | None = None
    fixed_params: dict | None = None
    supports_early_stopping: bool | None = None


@dataclass
class RunConfig:
    """General experiment settings (not model-specific).

    Parameters
    ----------
    n_folds : int
        Number of cross-validation folds. HPO runs on fold-0,
        then the best config is evaluated across all folds.
    val_fraction : float
        Fraction of training data used as validation when running
        in single-split mode (HybridTabularModel). Ignored in
        K-fold mode where folds define the splits.
    master_seed : int
        Single seed from which all other seeds are derived
        (per-fold, per-trial). Same value = reproducible results.
    device : str
        Where PyTorch models run. ``"auto"`` picks the best available
        (CUDA GPU > Apple MPS > CPU). Can be forced to a specific
        device: ``"cuda"``, ``"mps"``, or ``"cpu"``.
    deterministic : bool
        When True, enables PyTorch deterministic algorithms.
        Bit-for-bit reproducible at the cost of slower training.
    save_embeddings : bool
        Whether to save extracted embedding vectors (numpy arrays)
        to disk per fold. Off by default (can be large).
    save_predictions : bool
        Whether to save per-fold predictions to disk. Useful for
        post-hoc analysis (confusion matrices, error analysis).
    output_dir : str
        Root directory for run artifacts. Each experiment creates
        a subfolder with logs, metrics, and saved models.
    dataset_id : str
        Label used in artifact folder naming and logs. Purely
        organizational (e.g. ``"adult"``, ``"california_housing"``).
    scaler : str
        Scaler for numeric features. ``"standard"`` = zero mean,
        unit variance. ``"robust"`` = median-centered, IQR-scaled
        (better for data with outliers).
    max_categorical_cardinality : int
        Categories with more unique values than this cap get
        rare values collapsed into __OOV__. Prevents huge
        embedding tables for high-cardinality columns.
    add_missing_indicator : bool
        When True, adds a binary column per feature indicating
        whether the original value was missing (before imputation).
    """

    n_folds: int = 5
    val_fraction: float = 0.2
    master_seed: int = 42
    device: str = "auto"
    deterministic: bool = False
    save_embeddings: bool = False
    save_predictions: bool = True
    output_dir: str = "runs"
    dataset_id: str = "default"
    scaler: str = "standard"
    max_categorical_cardinality: int = 1000
    add_missing_indicator: bool = False


@dataclass
class FoldResult:
    """Results from a single CV fold evaluation.

    Parameters
    ----------
    fold_index : int
        Which fold this result corresponds to (0-based).
    metrics : dict[str, float]
        Metric name -> value for this fold (e.g., ``{"roc_auc": 0.93}``).
    predictions : np.ndarray or None
        Per-sample predictions for this fold's test set.
        Populated only if RunConfig.save_predictions is True.
    embedding_dim : int
        Dimensionality of the embedding vectors produced by the
        embedding model for this fold.
    """

    fold_index: int
    metrics: dict[str, float]
    predictions: np.ndarray | None = None
    embedding_dim: int = 0


@dataclass
class RunResult:
    """Complete results from a run_experiment() call.

    Contains per-fold results, aggregated metrics (mean +/- std
    across folds), and the best HP configs found during HPO.
    This is the primary output used for thesis reporting.

    Parameters
    ----------
    task : str
        The task type (``"binary"``, ``"multiclass"``, ``"regression"``).
    dataset_id : str
        Label identifying the dataset used.
    n_folds : int
        Number of CV folds used.
    fold_results : list[FoldResult]
        Per-fold evaluation results.
    mean_metrics : dict[str, float]
        Mean of each metric across all folds.
    std_metrics : dict[str, float]
        Standard deviation of each metric across all folds.
    best_embedding_config : dict
        Best embedding HP config found during HPO on fold-0.
    best_classical_config : dict
        Best classical HP config found during HPO on fold-0.
    total_time_seconds : float
        Total wall-clock time for the entire experiment.
    """

    task: str
    dataset_id: str
    n_folds: int
    fold_results: list[FoldResult]
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    best_embedding_config: dict
    best_classical_config: dict
    total_time_seconds: float


def validate_task_config(task_config: TaskConfig) -> None:
    """Validate that the task type is one of the allowed values.

    Parameters
    ----------
    task_config : TaskConfig
        The task configuration to validate.

    Raises
    ------
    ValueError
        If ``task_config.task`` is not ``"binary"``, ``"multiclass"``,
        or ``"regression"``.
    """
    if task_config.task not in VALID_TASKS:
        raise ValueError(
            f"Invalid task '{task_config.task}'. "
            f"Must be one of {VALID_TASKS}."
        )


def config_to_dict(config) -> dict:
    """Convert any config dataclass to a JSON-serializable dict.

    Recursively converts dataclass fields. Handles numpy arrays
    by converting them to lists.

    Parameters
    ----------
    config : dataclass instance
        Any of the configuration or result dataclasses.

    Returns
    -------
    dict
        A JSON-serializable dictionary representation.
    """
    if not dataclasses.is_dataclass(config):
        raise TypeError(f"Expected a dataclass instance, got {type(config)}")

    result = {}
    for f in dataclasses.fields(config):
        value = getattr(config, f.name)
        if dataclasses.is_dataclass(value):
            value = config_to_dict(value)
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, list):
            value = [
                config_to_dict(v) if dataclasses.is_dataclass(v) else v
                for v in value
            ]
        elif isinstance(value, type):
            value = f"{value.__module__}.{value.__qualname__}"
        result[f.name] = value
    return result


def save_config_yaml(config, path: str) -> None:
    """Save a config dataclass to a YAML file.

    Parameters
    ----------
    config : dataclass instance
        Any of the configuration dataclasses.
    path : str
        File path to write the YAML output.
    """
    d = config_to_dict(config)
    with open(path, "w") as fh:
        yaml.dump(d, fh, default_flow_style=False, sort_keys=False)
