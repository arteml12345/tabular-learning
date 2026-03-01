"""Experiment runner: full K-fold CV pipeline and sklearn-style wrapper.

Orchestrates all pipeline stages -- data validation, schema inference,
split generation, preprocessing, embedding HPO, classical HPO,
per-fold evaluation, metric aggregation, and artifact logging.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from hybrid_embed.config import (
    BudgetConfig,
    ClassicalStepConfig,
    EmbeddingStepConfig,
    FoldResult,
    RunConfig,
    RunResult,
    TaskConfig,
    config_to_dict,
    validate_task_config,
)
from hybrid_embed.data.preprocess import TabularPreprocessor
from hybrid_embed.data.schema import infer_schema
from hybrid_embed.data.splits import FoldSplit, generate_cv_splits, generate_single_split
from hybrid_embed.data.validation import validate_dataframe
from hybrid_embed.eval.metrics import (
    aggregate_fold_metrics,
    compute_metrics,
    get_primary_metric_name,
)
from hybrid_embed.hpo.classical_search import hyperopt_search_classical
from hybrid_embed.hpo.embed_search import hyperopt_search_embedding
from hybrid_embed.classical.model_zoo import (
    build_model,
    resolve_classical_step,
    train_classical_model,
)
from hybrid_embed.log.local_logger import LocalLogger
from hybrid_embed.utils import Timer, seed_everything

logger = logging.getLogger(__name__)


def run_experiment(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    task_config: TaskConfig,
    embedding_step: EmbeddingStepConfig,
    classical_step: ClassicalStepConfig,
    budget_config: BudgetConfig,
    run_config: RunConfig,
) -> RunResult:
    """Run the full experiment pipeline.

    The user has already chosen ONE embedding model and ONE
    classical model. This function tunes their hyperparameters
    and evaluates via K-fold CV.

    Steps:
    1. Seed everything for reproducibility.
    2. Validate data.
    3. Infer schema.
    4. Generate CV splits.
    5. HPO phase (on fold 0): tune embedding, then classical.
    6. Per-fold evaluation with best configs.
    7. Aggregate metrics across folds.
    8. Save summary and return RunResult.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series or np.ndarray
    task_config : TaskConfig
    embedding_step : EmbeddingStepConfig
    classical_step : ClassicalStepConfig
    budget_config : BudgetConfig
    run_config : RunConfig

    Returns
    -------
    RunResult
    """
    with Timer() as total_timer:
        validate_task_config(task_config)
        task = task_config.task

        seed_everything(run_config.master_seed)

        exp_logger = LocalLogger(
            output_dir=run_config.output_dir,
            dataset_id=run_config.dataset_id,
        )

        y_arr = np.asarray(y)

        # -- 1. Data validation -------------------------------------------
        X_clean, validation_report = validate_dataframe(X, y_arr, task)
        exp_logger.save_validation_report(validation_report)
        logger.info("Data validation complete: %d rows, %d cols",
                     len(X_clean), X_clean.shape[1])

        # -- 2. Schema inference ------------------------------------------
        schema = infer_schema(X_clean)
        exp_logger.save_schema(schema)
        logger.info("Schema: %d numeric, %d categorical",
                     len(schema.numeric_columns),
                     len(schema.categorical_columns))

        # -- 3. CV splits -------------------------------------------------
        splits = generate_cv_splits(
            y_arr,
            n_folds=run_config.n_folds,
            task=task,
            master_seed=run_config.master_seed,
            val_fraction=run_config.val_fraction,
        )
        logger.info("Generated %d CV splits", len(splits))

        # -- 4. HPO on fold 0 --------------------------------------------
        fold0 = splits[0]
        preprocessor_hpo = _fit_preprocessor(
            X_clean, y_arr, fold0.train_indices, schema, task, run_config,
        )

        X_train_hpo = preprocessor_hpo.transform(X_clean.iloc[fold0.train_indices])
        X_val_hpo = preprocessor_hpo.transform(X_clean.iloc[fold0.val_indices])
        y_train_hpo = preprocessor_hpo.transform_target(y_arr[fold0.train_indices])
        y_val_hpo = preprocessor_hpo.transform_target(y_arr[fold0.val_indices])

        n_numeric = preprocessor_hpo.get_n_numeric_features()
        n_cats_per_col = preprocessor_hpo.get_n_categories_per_column()
        n_classes = preprocessor_hpo.n_classes

        device_str = run_config.device

        # 4a. Embedding HPO
        logger.info("Starting embedding HPO (max %d trials)",
                     budget_config.embed_hpo_max_trials)
        embed_hpo_result = hyperopt_search_embedding(
            X_train=X_train_hpo,
            y_train=y_train_hpo,
            X_val=X_val_hpo,
            y_val=y_val_hpo,
            task=task,
            n_numeric=n_numeric,
            n_categories_per_column=n_cats_per_col,
            n_classes=n_classes,
            embedding_step=embedding_step,
            max_trials=budget_config.embed_hpo_max_trials,
            max_epochs=budget_config.embed_max_epochs,
            patience=budget_config.embed_patience,
            time_budget_seconds=budget_config.embed_time_budget_seconds,
            master_seed=run_config.master_seed,
            device=device_str,
            logger=exp_logger,
            schema=schema,
        )
        best_embed_params = embed_hpo_result["best_params"]
        exp_logger.save_best_embedding_config(
            _make_serializable(best_embed_params)
        )
        logger.info("Embedding HPO done: %d trials, best score=%.4f",
                     embed_hpo_result["n_trials_completed"],
                     embed_hpo_result["best_score"])

        # 4b. Train best embedding model on fold-0 to get embeddings for classical HPO
        embed_model_hpo = _build_and_train_embedding(
            embedding_step.model_type, task, n_numeric, n_cats_per_col,
            n_classes, X_train_hpo, y_train_hpo, X_val_hpo, y_val_hpo,
            best_embed_params, schema,
        )
        E_train_hpo = embed_model_hpo.encode(X_train_hpo)
        E_val_hpo = embed_model_hpo.encode(X_val_hpo)

        # 4c. Classical HPO
        logger.info("Starting classical HPO (max %d trials)",
                     budget_config.classical_hpo_max_trials)
        classical_hpo_result = hyperopt_search_classical(
            E_train=E_train_hpo,
            y_train=y_train_hpo,
            E_val=E_val_hpo,
            y_val=y_val_hpo,
            task=task,
            classical_step=classical_step,
            max_trials=budget_config.classical_hpo_max_trials,
            time_budget_seconds=budget_config.classical_time_budget_seconds,
            master_seed=run_config.master_seed,
            logger=exp_logger,
        )
        best_classical_params = classical_hpo_result["best_params"]
        exp_logger.save_best_classical_config(
            _make_serializable(best_classical_params)
        )
        logger.info("Classical HPO done: %d trials, best score=%.4f",
                     classical_hpo_result["n_trials_completed"],
                     classical_hpo_result["best_score"])

        # Save full config
        exp_logger.save_config({
            "task_config": config_to_dict(task_config),
            "embedding_step": config_to_dict(embedding_step),
            "classical_step": config_to_dict(classical_step),
            "budget_config": config_to_dict(budget_config),
            "run_config": config_to_dict(run_config),
        })

        # -- 5. Per-fold evaluation ---------------------------------------
        classical_spec = resolve_classical_step(classical_step, task)
        fold_results: list[FoldResult] = []

        for fold_split in splits:
            fold_result = _evaluate_fold(
                fold_split=fold_split,
                X_clean=X_clean,
                y_arr=y_arr,
                task=task,
                schema=schema,
                run_config=run_config,
                embedding_step=embedding_step,
                best_embed_params=best_embed_params,
                classical_spec=classical_spec,
                best_classical_params=best_classical_params,
                exp_logger=exp_logger,
            )
            fold_results.append(fold_result)
            logger.info("Fold %d: %s", fold_split.fold_index,
                         fold_result.metrics)

        # -- 6. Aggregate -------------------------------------------------
        fold_metrics = [fr.metrics for fr in fold_results]
        mean_metrics, std_metrics = aggregate_fold_metrics(fold_metrics)
        exp_logger.save_summary_metrics(mean_metrics, std_metrics)

        logger.info("Mean metrics: %s", mean_metrics)
        logger.info("Std  metrics: %s", std_metrics)

    return RunResult(
        task=task,
        dataset_id=run_config.dataset_id,
        n_folds=run_config.n_folds,
        fold_results=fold_results,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        best_embedding_config=best_embed_params,
        best_classical_config=best_classical_params,
        total_time_seconds=total_timer.elapsed_seconds,
    )


class HybridTabularModel:
    """Sklearn-style interface for the hybrid embedding framework.

    Uses a single train/val split (not full CV) for HPO and
    training. After ``fit()``, the model can be used for
    ``predict()`` and ``predict_proba()`` on new data.

    Parameters
    ----------
    task : str
        ``"binary"``, ``"multiclass"``, or ``"regression"``.
    embedding : EmbeddingStepConfig
        Which embedding model to use.
    classical : ClassicalStepConfig
        Which classical model to use.
    budget_config : BudgetConfig or None
        HPO budget limits. Defaults to small budgets if None.
    run_config : RunConfig or None
        Experiment settings. Defaults if None.
    """

    def __init__(
        self,
        task: str,
        embedding: EmbeddingStepConfig,
        classical: ClassicalStepConfig,
        budget_config: BudgetConfig | None = None,
        run_config: RunConfig | None = None,
    ):
        self.task = task
        self.embedding = embedding
        self.classical = classical
        self.budget_config = budget_config or BudgetConfig()
        self.run_config = run_config or RunConfig()

        self._preprocessor: TabularPreprocessor | None = None
        self._embed_model: Any = None
        self._classical_model: Any = None
        self._schema = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "HybridTabularModel":
        """Run HPO and train best models on a single train/val split.

        Parameters
        ----------
        X : pd.DataFrame
            Raw features.
        y : pd.Series or np.ndarray
            Target variable.

        Returns
        -------
        self
        """
        task = self.task
        rc = self.run_config
        bc = self.budget_config

        seed_everything(rc.master_seed)
        y_arr = np.asarray(y)

        X_clean, _ = validate_dataframe(X, y_arr, task)
        self._schema = infer_schema(X_clean)

        split = generate_single_split(
            y_arr,
            task=task,
            val_fraction=rc.val_fraction,
            master_seed=rc.master_seed,
        )

        self._preprocessor = _fit_preprocessor(
            X_clean, y_arr, split.train_indices, self._schema, task, rc,
        )
        X_tr = self._preprocessor.transform(X_clean.iloc[split.train_indices])
        X_vl = self._preprocessor.transform(X_clean.iloc[split.val_indices])
        y_tr = self._preprocessor.transform_target(y_arr[split.train_indices])
        y_vl = self._preprocessor.transform_target(y_arr[split.val_indices])

        n_numeric = self._preprocessor.get_n_numeric_features()
        n_cats = self._preprocessor.get_n_categories_per_column()
        n_classes = self._preprocessor.n_classes
        device_str = rc.device

        # Embedding HPO
        embed_hpo = hyperopt_search_embedding(
            X_train=X_tr, y_train=y_tr, X_val=X_vl, y_val=y_vl,
            task=task, n_numeric=n_numeric,
            n_categories_per_column=n_cats, n_classes=n_classes,
            embedding_step=self.embedding,
            max_trials=bc.embed_hpo_max_trials,
            max_epochs=bc.embed_max_epochs,
            patience=bc.embed_patience,
            time_budget_seconds=bc.embed_time_budget_seconds,
            master_seed=rc.master_seed,
            device=device_str,
            schema=self._schema,
        )
        best_embed_params = embed_hpo["best_params"]

        # Train best embedding model
        self._embed_model = _build_and_train_embedding(
            self.embedding.model_type, task, n_numeric, n_cats,
            n_classes, X_tr, y_tr, X_vl, y_vl,
            best_embed_params, self._schema,
        )

        # Classical HPO
        E_tr = self._embed_model.encode(X_tr)
        E_vl = self._embed_model.encode(X_vl)

        classical_hpo = hyperopt_search_classical(
            E_train=E_tr, y_train=y_tr, E_val=E_vl, y_val=y_vl,
            task=task, classical_step=self.classical,
            max_trials=bc.classical_hpo_max_trials,
            time_budget_seconds=bc.classical_time_budget_seconds,
            master_seed=rc.master_seed,
        )
        best_classical_params = classical_hpo["best_params"]

        # Train best classical model
        spec = resolve_classical_step(self.classical, task)
        merged_params = dict(spec["params"])
        merged_params.update(best_classical_params)
        self._classical_model = build_model(
            spec["class"], merged_params, task,
            random_state=rc.master_seed,
        )
        self._classical_model = train_classical_model(
            self._classical_model, E_tr, y_tr,
            E_val=E_vl, y_val=y_vl,
            supports_early_stopping=spec["supports_early_stopping"],
        )

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using fitted models.

        Parameters
        ----------
        X : pd.DataFrame
            Raw features (same format as training data).

        Returns
        -------
        np.ndarray
            Predicted labels (classification) or values (regression),
            in the same scale / label space as the original ``y``
            passed to ``fit()``.
        """
        self._check_fitted()
        E = self._get_embeddings(X)
        preds = self._classical_model.predict(E)
        return self._preprocessor.inverse_transform_target(preds)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (classification only).

        Parameters
        ----------
        X : pd.DataFrame
            Raw features.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_classes)``.

        Raises
        ------
        AttributeError
            If the classical model does not support ``predict_proba``.
        """
        self._check_fitted()
        E = self._get_embeddings(X)
        return self._classical_model.predict_proba(E)

    def _get_embeddings(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess and extract embeddings for new data."""
        X_transformed = self._preprocessor.transform(X)
        return self._embed_model.encode(X_transformed)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "HybridTabularModel has not been fitted. Call fit() first."
            )


# =====================================================================
# Internal helpers
# =====================================================================

def _fit_preprocessor(
    X_clean: pd.DataFrame,
    y_arr: np.ndarray,
    train_indices: np.ndarray,
    schema,
    task: str,
    run_config: RunConfig,
) -> TabularPreprocessor:
    """Fit a TabularPreprocessor on the training subset."""
    preprocessor = TabularPreprocessor(
        schema=schema,
        task=task,
        scaler_type=run_config.scaler,
        max_categorical_cardinality=run_config.max_categorical_cardinality,
        add_missing_indicator=run_config.add_missing_indicator,
    )
    preprocessor.fit(
        X_clean.iloc[train_indices],
        y_arr[train_indices],
    )
    return preprocessor


def _build_and_train_embedding(
    model_type, task, n_numeric, n_categories_per_column,
    n_classes, X_train, y_train, X_val, y_val, params, schema,
):
    """Instantiate and train an embedding model with the given config."""
    if model_type == "mlp":
        from hybrid_embed.embed.scratch_mlp import MLPEmbeddingModel
        model = MLPEmbeddingModel(
            task=task,
            n_numeric=n_numeric,
            n_categories_per_column=n_categories_per_column,
            n_classes=n_classes,
        )
        model.fit(X_train, y_train, X_val, y_val, params)
        return model

    from hybrid_embed.embed.adapters.pytorch_tabular_adapter import (
        PytorchTabularEmbeddingModel,
    )
    model = PytorchTabularEmbeddingModel(
        model_type=model_type,
        task=task,
        schema=schema,
        n_classes=n_classes,
    )
    model.fit(X_train, y_train, X_val, y_val, params)
    return model


def _evaluate_fold(
    fold_split: FoldSplit,
    X_clean: pd.DataFrame,
    y_arr: np.ndarray,
    task: str,
    schema,
    run_config: RunConfig,
    embedding_step: EmbeddingStepConfig,
    best_embed_params: dict,
    classical_spec: dict,
    best_classical_params: dict,
    exp_logger: LocalLogger,
) -> FoldResult:
    """Train and evaluate on a single CV fold."""
    fi = fold_split.fold_index
    logger.info("Evaluating fold %d", fi)

    # Fresh preprocessor for this fold
    preprocessor = _fit_preprocessor(
        X_clean, y_arr, fold_split.train_indices, schema, task, run_config,
    )

    X_tr = preprocessor.transform(X_clean.iloc[fold_split.train_indices])
    X_vl = preprocessor.transform(X_clean.iloc[fold_split.val_indices])
    X_te = preprocessor.transform(X_clean.iloc[fold_split.test_indices])
    y_tr = preprocessor.transform_target(y_arr[fold_split.train_indices])
    y_vl = preprocessor.transform_target(y_arr[fold_split.val_indices])
    y_te = preprocessor.transform_target(y_arr[fold_split.test_indices])

    n_numeric = preprocessor.get_n_numeric_features()
    n_cats = preprocessor.get_n_categories_per_column()
    n_classes = preprocessor.n_classes

    # Train embedding model with best config
    embed_model = _build_and_train_embedding(
        embedding_step.model_type, task, n_numeric, n_cats,
        n_classes, X_tr, y_tr, X_vl, y_vl,
        best_embed_params, schema,
    )

    # Extract embeddings
    E_tr = embed_model.encode(X_tr)
    E_vl = embed_model.encode(X_vl)
    E_te = embed_model.encode(X_te)

    # Train classical model with best config
    merged_params = dict(classical_spec["params"])
    merged_params.update(best_classical_params)
    classical_model = build_model(
        classical_spec["class"], merged_params, task,
        random_state=run_config.master_seed + fi,
    )
    classical_model = train_classical_model(
        classical_model, E_tr, y_tr,
        E_val=E_vl, y_val=y_vl,
        supports_early_stopping=classical_spec["supports_early_stopping"],
    )

    # Predict on test embeddings
    preds = classical_model.predict(E_te)

    proba = None
    if task in ("binary", "multiclass"):
        try:
            proba_raw = classical_model.predict_proba(E_te)
            if task == "binary":
                proba = proba_raw[:, 1]
            else:
                proba = proba_raw
        except AttributeError:
            pass

    metrics = compute_metrics(y_te, preds, proba, task)

    # Save fold artifacts
    save_preds = preds if run_config.save_predictions else None
    save_embeds = E_te if run_config.save_embeddings else None
    exp_logger.save_fold_artifacts(
        fold_index=fi,
        metrics=metrics,
        embedding_model=embed_model,
        classical_model=classical_model,
        predictions=save_preds,
        embeddings=save_embeds,
    )

    return FoldResult(
        fold_index=fi,
        metrics=metrics,
        predictions=save_preds,
        embedding_dim=E_te.shape[1] if E_te.shape[0] > 0 else 0,
    )


def _make_serializable(params: dict) -> dict:
    """Convert param dict values to JSON-safe types."""
    out = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        elif isinstance(v, list):
            out[k] = v
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = str(v)
    return out
