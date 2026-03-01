"""Embedding HPO search via Hyperopt TPE.

Runs a Hyperopt search loop to find the best hyperparameters for
the selected embedding model. The model is trained as a standalone
predictor; the HPO objective is the model's own validation metric.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, tpe

from hybrid_embed.config import EmbeddingStepConfig
from hybrid_embed.eval.metrics import get_primary_metric_name, is_higher_better
from hybrid_embed.hpo.spaces import (
    postprocess_mlp_params,
    postprocess_pytorch_tabular_params,
    resolve_embedding_search_space,
)


def hyperopt_search_embedding(
    X_train: dict,
    y_train: np.ndarray,
    X_val: dict,
    y_val: np.ndarray,
    task: str,
    n_numeric: int,
    n_categories_per_column: list[int],
    n_classes: int | None,
    embedding_step: EmbeddingStepConfig,
    max_trials: int = 50,
    max_epochs: int = 50,
    patience: int = 8,
    time_budget_seconds: float | None = None,
    master_seed: int = 42,
    device: str = "cpu",
    logger: Any | None = None,
    schema: Any | None = None,
) -> dict:
    """Run Hyperopt TPE search for the embedding model's hyperparameters.

    Parameters
    ----------
    X_train, X_val : dict
        Preprocessed training/validation data.
    y_train, y_val : np.ndarray
        Target arrays.
    task : str
        ``"binary"``, ``"multiclass"``, or ``"regression"``.
    n_numeric : int
        Number of numeric features.
    n_categories_per_column : list[int]
        Vocabulary size per categorical column.
    n_classes : int or None
        Number of classes (classification) or None (regression).
    embedding_step : EmbeddingStepConfig
        User configuration for the embedding model.
    max_trials : int
        Maximum number of Hyperopt trials.
    max_epochs : int
        Max training epochs per trial.
    patience : int
        Early stopping patience per trial.
    time_budget_seconds : float or None
        Wall-clock budget; stops launching new trials after this.
    master_seed : int
        Random seed.
    device : str
        PyTorch device.
    logger : LocalLogger or None
        If provided, each trial is logged.
    schema : Schema or None
        Required for pytorch-tabular model types.

    Returns
    -------
    dict
        ``"best_params"``, ``"best_score"``, ``"n_trials_completed"``,
        ``"total_time_seconds"``.
    """
    search_space = resolve_embedding_search_space(embedding_step)
    model_type = embedding_step.model_type
    fixed_params = dict(embedding_step.fixed_params or {})

    primary_metric = get_primary_metric_name(task)
    higher_better = is_higher_better(primary_metric)

    start_time = time.time()
    trial_counter = [0]
    best_tracker = {"score": None, "params": None}

    def _objective(sampled_params: dict) -> dict:
        elapsed = time.time() - start_time
        if time_budget_seconds is not None and elapsed > time_budget_seconds:
            return {"loss": 0.0, "status": STATUS_OK}

        trial_id = trial_counter[0]
        trial_counter[0] += 1

        from hybrid_embed.utils import seed_everything
        seed_everything(master_seed + trial_id)

        params = _postprocess(sampled_params, model_type)
        params.update(fixed_params)
        params["max_epochs"] = max_epochs
        params["patience"] = patience
        params["device"] = device

        try:
            model = _build_and_train(
                model_type, task, n_numeric,
                n_categories_per_column, n_classes,
                X_train, y_train, X_val, y_val,
                params, schema,
            )
            val_score = _evaluate_model(model, X_val, y_val, task, primary_metric)
        except Exception:
            val_score = _worst_score(higher_better)

        loss = -val_score if higher_better else val_score

        if best_tracker["score"] is None or (
            (higher_better and val_score > best_tracker["score"])
            or (not higher_better and val_score < best_tracker["score"])
        ):
            best_tracker["score"] = val_score
            best_tracker["params"] = params

        if logger is not None:
            logger.log_embedding_trial({
                "trial_id": trial_id,
                "params": _make_serializable(params),
                "val_metric": {primary_metric: val_score},
                "loss": loss,
            })

        return {"loss": loss, "status": STATUS_OK}

    trials = Trials()

    timeout_fn = None
    if time_budget_seconds is not None:
        timeout_fn = partial(_check_timeout, start_time=start_time, budget=time_budget_seconds)

    rng = np.random.default_rng(master_seed)

    fmin(
        fn=_objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_trials,
        trials=trials,
        rstate=rng,
        show_progressbar=False,
        early_stop_fn=timeout_fn,
    )

    total_time = time.time() - start_time

    return {
        "best_params": best_tracker["params"] or {},
        "best_score": best_tracker["score"] if best_tracker["score"] is not None else float("nan"),
        "n_trials_completed": trial_counter[0],
        "total_time_seconds": total_time,
    }


# =====================================================================
# Internal helpers
# =====================================================================

def _postprocess(params: dict, model_type: str) -> dict:
    if model_type == "mlp":
        return postprocess_mlp_params(params)
    return postprocess_pytorch_tabular_params(params, model_type)


def _build_and_train(
    model_type, task, n_numeric, n_categories_per_column,
    n_classes, X_train, y_train, X_val, y_val, params, schema,
):
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


def _evaluate_model(model, X_val, y_val, task, primary_metric):
    from hybrid_embed.eval.metrics import compute_metrics

    if task == "binary":
        proba = model.predict_proba(X_val)
        preds = (proba[:, 1] >= 0.5).astype(int)
        metrics = compute_metrics(y_val, preds, proba[:, 1], task="binary")
    elif task == "multiclass":
        proba = model.predict_proba(X_val)
        preds = model.predict(X_val)
        metrics = compute_metrics(y_val, preds, proba, task="multiclass")
    else:
        preds = model.predict(X_val)
        metrics = compute_metrics(y_val, preds, None, task="regression")

    return metrics[primary_metric]


def _worst_score(higher_better: bool) -> float:
    return float("-inf") if higher_better else float("inf")


def _check_timeout(trials, *, start_time: float, budget: float):
    elapsed = time.time() - start_time
    if elapsed > budget:
        return True, []
    return False, []


def _make_serializable(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        elif isinstance(v, list):
            out[k] = v
        else:
            out[k] = str(v)
    return out
