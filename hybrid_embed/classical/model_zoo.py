"""Classical model registry and utilities.

Provides a registry of built-in classical models (each with a
default Hyperopt search space), resolution of user-provided configs
into concrete model specs, and a training helper that handles
early stopping for boosted tree models.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from hyperopt import hp

from hybrid_embed.config import ClassicalStepConfig


# -----------------------------------------------------------------------
# Search spaces (Hyperopt)
# -----------------------------------------------------------------------

def _logistic_regression_space() -> dict:
    return {
        "C": hp.loguniform("C", np.log(1e-4), np.log(1e2)),
    }


def _ridge_clf_space() -> dict:
    return {
        "C": hp.loguniform("C", np.log(1e-4), np.log(1e2)),
    }


def _ridge_reg_space() -> dict:
    return {
        "alpha": hp.loguniform("alpha", np.log(1e-4), np.log(1e4)),
    }


def _elastic_net_clf_space() -> dict:
    return {
        "alpha": hp.loguniform("alpha", np.log(1e-4), np.log(1e2)),
        "l1_ratio": hp.uniform("l1_ratio", 0.01, 0.99),
    }


def _elastic_net_reg_space() -> dict:
    return {
        "alpha": hp.loguniform("alpha", np.log(1e-4), np.log(1e2)),
        "l1_ratio": hp.uniform("l1_ratio", 0.0, 1.0),
    }


def _random_forest_space() -> dict:
    return {
        "n_estimators": hp.quniform("n_estimators", 50, 500, 50),
        "max_depth": hp.quniform("max_depth", 3, 15, 1),
        "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 20, 1),
        "max_features": hp.choice("max_features", ["sqrt", "log2", 0.5, 0.8]),
    }


def _extra_trees_space() -> dict:
    return {
        "n_estimators": hp.quniform("n_estimators", 50, 500, 50),
        "max_depth": hp.quniform("max_depth", 3, 15, 1),
        "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 20, 1),
        "max_features": hp.choice("max_features", ["sqrt", "log2", 0.5, 0.8]),
    }


def _xgboost_space() -> dict:
    return {
        "n_estimators": hp.quniform("n_estimators", 50, 1000, 50),
        "max_depth": hp.quniform("max_depth", 3, 10, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(1e-3), np.log(0.3)),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
        "gamma": hp.loguniform("gamma", np.log(1e-8), np.log(1.0)),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-8), np.log(10.0)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-8), np.log(10.0)),
    }


def _lightgbm_space() -> dict:
    return {
        "n_estimators": hp.quniform("n_estimators", 50, 1000, 50),
        "max_depth": hp.quniform("max_depth", 3, 10, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(1e-3), np.log(0.3)),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "min_child_samples": hp.quniform("min_child_samples", 5, 50, 5),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-8), np.log(10.0)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-8), np.log(10.0)),
        "num_leaves": hp.quniform("num_leaves", 15, 127, 1),
    }


def _catboost_space() -> dict:
    return {
        "iterations": hp.quniform("iterations", 50, 1000, 50),
        "depth": hp.quniform("depth", 3, 10, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(1e-3), np.log(0.3)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1e-2), np.log(10.0)),
    }


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------

def _build_registry() -> dict:
    """Build the model registry lazily to avoid import overhead."""
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import (
        ElasticNet,
        LogisticRegression,
        Ridge,
        SGDClassifier,
    )

    import catboost
    import lightgbm
    import xgboost

    return {
        "logistic_regression": {
            "class_clf": LogisticRegression,
            "class_reg": Ridge,
            "default_params_clf": {"max_iter": 1000},
            "default_params_reg": {},
            "default_search_space_fn_clf": _logistic_regression_space,
            "default_search_space_fn_reg": _ridge_reg_space,
            "supports_early_stopping": False,
        },
        "ridge": {
            "class_clf": LogisticRegression,
            "class_reg": Ridge,
            "default_params_clf": {"max_iter": 1000},
            "default_params_reg": {},
            "default_search_space_fn_clf": _ridge_clf_space,
            "default_search_space_fn_reg": _ridge_reg_space,
            "supports_early_stopping": False,
        },
        "elastic_net": {
            "class_clf": SGDClassifier,
            "class_reg": ElasticNet,
            "default_params_clf": {
                "loss": "log_loss",
                "penalty": "elasticnet",
                "max_iter": 2000,
            },
            "default_params_reg": {"max_iter": 1000},
            "default_search_space_fn_clf": _elastic_net_clf_space,
            "default_search_space_fn_reg": _elastic_net_reg_space,
            "supports_early_stopping": False,
        },
        "random_forest": {
            "class_clf": RandomForestClassifier,
            "class_reg": RandomForestRegressor,
            "default_params_clf": {},
            "default_params_reg": {},
            "default_search_space_fn_clf": _random_forest_space,
            "default_search_space_fn_reg": _random_forest_space,
            "supports_early_stopping": False,
        },
        "extra_trees": {
            "class_clf": ExtraTreesClassifier,
            "class_reg": ExtraTreesRegressor,
            "default_params_clf": {},
            "default_params_reg": {},
            "default_search_space_fn_clf": _extra_trees_space,
            "default_search_space_fn_reg": _extra_trees_space,
            "supports_early_stopping": False,
        },
        "xgboost": {
            "class_clf": xgboost.XGBClassifier,
            "class_reg": xgboost.XGBRegressor,
            "default_params_clf": {"verbosity": 0},
            "default_params_reg": {"verbosity": 0},
            "default_search_space_fn_clf": _xgboost_space,
            "default_search_space_fn_reg": _xgboost_space,
            "supports_early_stopping": True,
        },
        "lightgbm": {
            "class_clf": lightgbm.LGBMClassifier,
            "class_reg": lightgbm.LGBMRegressor,
            "default_params_clf": {"verbose": -1},
            "default_params_reg": {"verbose": -1},
            "default_search_space_fn_clf": _lightgbm_space,
            "default_search_space_fn_reg": _lightgbm_space,
            "supports_early_stopping": True,
        },
        "catboost": {
            "class_clf": catboost.CatBoostClassifier,
            "class_reg": catboost.CatBoostRegressor,
            "default_params_clf": {"verbose": 0},
            "default_params_reg": {"verbose": 0},
            "default_search_space_fn_clf": _catboost_space,
            "default_search_space_fn_reg": _catboost_space,
            "supports_early_stopping": True,
        },
    }


_REGISTRY_CACHE: dict | None = None


def _get_registry() -> dict:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _build_registry()
    return _REGISTRY_CACHE


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def resolve_classical_step(
    step_config: ClassicalStepConfig,
    task: str,
) -> dict:
    """Resolve a ClassicalStepConfig into a concrete model spec.

    Parameters
    ----------
    step_config : ClassicalStepConfig
        User-provided classical model configuration.
    task : str
        ``"binary"``, ``"multiclass"``, or ``"regression"``.

    Returns
    -------
    dict
        Keys: ``"class"`` (type), ``"params"`` (dict),
        ``"search_space"`` (dict), ``"supports_early_stopping"``
        (bool).

    Raises
    ------
    ValueError
        If ``model_type="custom"`` but ``model_class`` or
        ``search_space`` is missing, or if ``model_type`` is
        unknown.
    """
    if step_config.model_type == "custom":
        if step_config.model_class is None:
            raise ValueError(
                "model_type='custom' requires model_class to be set"
            )
        if step_config.search_space is None:
            raise ValueError(
                "model_type='custom' requires search_space to be set"
            )
        return {
            "class": step_config.model_class,
            "params": dict(step_config.fixed_params or {}),
            "search_space": step_config.search_space,
            "supports_early_stopping": bool(
                step_config.supports_early_stopping
            ),
        }

    registry = _get_registry()
    if step_config.model_type not in registry:
        raise ValueError(
            f"Unknown model_type '{step_config.model_type}'. "
            f"Supported: {list(registry.keys()) + ['custom']}"
        )

    entry = registry[step_config.model_type]

    is_reg = task == "regression"
    suffix = "_reg" if is_reg else "_clf"
    model_class = entry["class_reg"] if is_reg else entry["class_clf"]

    params = dict(entry[f"default_params{suffix}"])
    if step_config.fixed_params:
        params.update(step_config.fixed_params)

    search_space = (
        step_config.search_space
        if step_config.search_space is not None
        else entry[f"default_search_space_fn{suffix}"]()
    )

    supports_es = (
        step_config.supports_early_stopping
        if step_config.supports_early_stopping is not None
        else entry["supports_early_stopping"]
    )

    return {
        "class": model_class,
        "params": params,
        "search_space": search_space,
        "supports_early_stopping": supports_es,
    }


def build_model(
    model_class: type,
    params: dict,
    task: str,
    random_state: int = 42,
) -> Any:
    """Instantiate a classical model with given parameters.

    Handles task-specific configuration for known model families
    (XGBoost, LightGBM, CatBoost) and injects ``random_state``
    where supported.

    Parameters
    ----------
    model_class : type
        sklearn-compatible model class.
    params : dict
        Constructor parameters.
    task : str
        ``"binary"``, ``"multiclass"``, or ``"regression"``.
    random_state : int
        Random seed.

    Returns
    -------
    object
        Instantiated sklearn-compatible model.
    """
    import catboost
    import lightgbm
    import xgboost

    params = dict(params)

    _cast_int_params(model_class, params)

    if issubclass(model_class, (xgboost.XGBClassifier, xgboost.XGBRegressor)):
        if task == "binary":
            params.setdefault("objective", "binary:logistic")
            params.setdefault("eval_metric", "logloss")
        elif task == "multiclass":
            params.setdefault("objective", "multi:softprob")
            params.setdefault("eval_metric", "mlogloss")
        else:
            params.setdefault("objective", "reg:squarederror")
            params.setdefault("eval_metric", "rmse")
        params.setdefault("verbosity", 0)
        params.setdefault("random_state", random_state)
        params.setdefault("n_jobs", 1)

    elif issubclass(model_class, (lightgbm.LGBMClassifier, lightgbm.LGBMRegressor)):
        if task == "binary":
            params.setdefault("objective", "binary")
        elif task == "multiclass":
            params.setdefault("objective", "multiclass")
        else:
            params.setdefault("objective", "regression")
        params.setdefault("verbose", -1)
        params.setdefault("random_state", random_state)
        params.setdefault("n_jobs", 1)

    elif issubclass(model_class, (catboost.CatBoostClassifier, catboost.CatBoostRegressor)):
        if task == "binary":
            params.setdefault("loss_function", "Logloss")
        elif task == "multiclass":
            params.setdefault("loss_function", "MultiClass")
        else:
            params.setdefault("loss_function", "RMSE")
        params.setdefault("verbose", 0)
        params.setdefault("random_seed", random_state)

    else:
        try:
            import inspect
            sig = inspect.signature(model_class.__init__)
            if "random_state" in sig.parameters:
                params.setdefault("random_state", random_state)
            elif "seed" in sig.parameters:
                params.setdefault("seed", random_state)
        except (ValueError, TypeError):
            pass

    return model_class(**params)


def train_classical_model(
    model: Any,
    E_train: np.ndarray,
    y_train: np.ndarray,
    E_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    supports_early_stopping: bool = False,
    early_stopping_rounds: int = 50,
) -> Any:
    """Train a classical model, optionally with early stopping.

    For XGBoost, LightGBM, and CatBoost: uses ``eval_set`` and
    early stopping callbacks when validation data is provided.

    For all other models: calls ``model.fit(E_train, y_train)``
    directly.

    Parameters
    ----------
    model : object
        sklearn-compatible model (already instantiated).
    E_train : np.ndarray
        Training embeddings, shape ``(n_train, embed_dim)``.
    y_train : np.ndarray
        Training targets.
    E_val : np.ndarray or None
        Validation embeddings.
    y_val : np.ndarray or None
        Validation targets.
    supports_early_stopping : bool
        Whether the model supports early stopping.
    early_stopping_rounds : int
        Patience for early stopping.

    Returns
    -------
    object
        The trained model.
    """
    import catboost
    import lightgbm
    import xgboost

    if supports_early_stopping and E_val is not None and y_val is not None:
        if isinstance(model, (xgboost.XGBClassifier, xgboost.XGBRegressor)):
            model.set_params(early_stopping_rounds=early_stopping_rounds)
            model.fit(
                E_train, y_train,
                eval_set=[(E_val, y_val)],
                verbose=False,
            )
            return model

        if isinstance(model, (lightgbm.LGBMClassifier, lightgbm.LGBMRegressor)):
            callbacks = [
                lightgbm.early_stopping(early_stopping_rounds, verbose=False),
                lightgbm.log_evaluation(period=0),
            ]
            model.fit(
                E_train, y_train,
                eval_set=[(E_val, y_val)],
                callbacks=callbacks,
            )
            return model

        if isinstance(model, (catboost.CatBoostClassifier, catboost.CatBoostRegressor)):
            model.fit(
                E_train, y_train,
                eval_set=(E_val, y_val),
                early_stopping_rounds=early_stopping_rounds,
                verbose=0,
            )
            return model

    model.fit(E_train, y_train)
    return model


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

_INT_PARAMS = {
    "n_estimators", "max_depth", "min_samples_leaf", "min_child_weight",
    "min_child_samples", "num_leaves", "iterations", "depth", "n_steps",
}


def _cast_int_params(model_class: type, params: dict) -> None:
    """Cast known integer hyperparameters from float to int.

    Hyperopt's ``quniform`` returns floats; boosted tree libraries
    expect ints for parameters like ``n_estimators``.
    """
    for key in _INT_PARAMS:
        if key in params and isinstance(params[key], float):
            params[key] = int(params[key])
