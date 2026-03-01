"""Classical HPO search via Hyperopt TPE.

Runs a Hyperopt search loop to find the best hyperparameters for
the selected classical model. The model is trained on embedding
vectors produced by an already-trained embedding model.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, tpe

from hybrid_embed.classical.model_zoo import (
    build_model,
    resolve_classical_step,
    train_classical_model,
)
from hybrid_embed.config import ClassicalStepConfig
from hybrid_embed.eval.metrics import (
    compute_metrics,
    get_primary_metric_name,
    is_higher_better,
)
from hybrid_embed.hpo.spaces import postprocess_classical_params


def hyperopt_search_classical(
    E_train: np.ndarray,
    y_train: np.ndarray,
    E_val: np.ndarray,
    y_val: np.ndarray,
    task: str,
    classical_step: ClassicalStepConfig,
    max_trials: int = 150,
    time_budget_seconds: float | None = None,
    master_seed: int = 42,
    logger: Any | None = None,
) -> dict:
    """Run Hyperopt TPE search for the selected classical model's HPs.

    Tunes the ONE model specified in ``classical_step`` within its
    search space (user-provided or default).

    Per trial:
    1. Sample hyperparams from the search space.
    2. Merge with ``classical_step.fixed_params``.
    3. Postprocess params.
    4. Build model.
    5. Train model (with early stopping if supported and val
       data provided).
    6. Compute validation metric.
    7. Log trial.
    8. Return negative score for Hyperopt (Hyperopt minimizes).

    Parameters
    ----------
    E_train : np.ndarray
        Training embeddings, shape ``(n_train, embed_dim)``.
    y_train : np.ndarray
        Training targets.
    E_val : np.ndarray
        Validation embeddings, shape ``(n_val, embed_dim)``.
    y_val : np.ndarray
        Validation targets.
    task : str
        ``"binary"``, ``"multiclass"``, or ``"regression"``.
    classical_step : ClassicalStepConfig
        User configuration for the classical model.
    max_trials : int
        Maximum number of Hyperopt trials.
    time_budget_seconds : float or None
        Wall-clock budget; stops launching new trials after this.
    master_seed : int
        Random seed.
    logger : LocalLogger or None
        If provided, each trial is logged.

    Returns
    -------
    dict
        ``"best_params"``, ``"best_score"``, ``"n_trials_completed"``,
        ``"total_time_seconds"``.
    """
    resolved = resolve_classical_step(classical_step, task)
    model_class = resolved["class"]
    base_params = resolved["params"]
    search_space = resolved["search_space"]
    supports_es = resolved["supports_early_stopping"]

    fixed_params = dict(classical_step.fixed_params or {})

    primary_metric = get_primary_metric_name(task)
    higher_better = is_higher_better(primary_metric)

    start_time = time.time()
    trial_counter = [0]
    best_tracker: dict[str, Any] = {"score": None, "params": None}

    def _objective(sampled_params: dict) -> dict:
        elapsed = time.time() - start_time
        if time_budget_seconds is not None and elapsed > time_budget_seconds:
            return {"loss": 0.0, "status": STATUS_OK}

        trial_id = trial_counter[0]
        trial_counter[0] += 1

        params = postprocess_classical_params(sampled_params)
        merged = dict(base_params)
        merged.update(params)
        merged.update(fixed_params)

        try:
            model = build_model(
                model_class, merged, task, random_state=master_seed,
            )
            trained = train_classical_model(
                model, E_train, y_train,
                E_val=E_val, y_val=y_val,
                supports_early_stopping=supports_es,
            )
            val_score = _evaluate(trained, E_val, y_val, task, primary_metric)
        except Exception:
            val_score = _worst_score(higher_better)

        loss = -val_score if higher_better else val_score

        if best_tracker["score"] is None or (
            (higher_better and val_score > best_tracker["score"])
            or (not higher_better and val_score < best_tracker["score"])
        ):
            best_tracker["score"] = val_score
            best_tracker["params"] = merged

        if logger is not None:
            logger.log_classical_trial({
                "trial_id": trial_id,
                "params": _make_serializable(merged),
                "val_metric": {primary_metric: val_score},
                "loss": loss,
            })

        return {"loss": loss, "status": STATUS_OK}

    trials = Trials()

    timeout_fn = None
    if time_budget_seconds is not None:
        timeout_fn = partial(
            _check_timeout, start_time=start_time, budget=time_budget_seconds,
        )

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
        "best_score": (
            best_tracker["score"]
            if best_tracker["score"] is not None
            else float("nan")
        ),
        "n_trials_completed": trial_counter[0],
        "total_time_seconds": total_time,
    }


# =====================================================================
# Internal helpers
# =====================================================================

def _evaluate(model, E_val, y_val, task, primary_metric):
    if task == "binary":
        proba = model.predict_proba(E_val)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics = compute_metrics(y_val, preds, proba, task)
    elif task == "multiclass":
        proba = model.predict_proba(E_val)
        preds = model.predict(E_val)
        metrics = compute_metrics(y_val, preds, proba, task)
    else:
        preds = model.predict(E_val)
        metrics = compute_metrics(y_val, preds, None, task)
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
