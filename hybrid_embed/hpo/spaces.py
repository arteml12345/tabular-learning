"""HPO search space definitions and postprocessing utilities.

Defines default Hyperopt search spaces for all embedding and
classical model types, resolution helpers that respect user
overrides, and postprocessors that convert raw Hyperopt samples
into clean config dicts.
"""

from __future__ import annotations

import numpy as np
from hyperopt import hp
from hyperopt.pyll.stochastic import sample as hp_sample

from hybrid_embed.config import ClassicalStepConfig, EmbeddingStepConfig


# =====================================================================
# MLP search space
# =====================================================================

def get_mlp_search_space() -> dict:
    """Return Hyperopt search space for the MLP embedding model.

    Returns
    -------
    dict
        Hyperopt space with keys: embedding_dim, depth, width,
        dropout, activation, use_layer_norm, lr, weight_decay,
        batch_size.
    """
    return {
        "embedding_dim": hp.quniform("embedding_dim", 32, 256, 16),
        "depth": hp.quniform("depth", 2, 8, 1),
        "width": hp.quniform("width", 128, 2048, 64),
        "dropout": hp.uniform("dropout", 0, 0.5),
        "activation": hp.choice("activation", ["relu", "gelu", "silu"]),
        "use_layer_norm": hp.choice("use_layer_norm", [True, False]),
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-2)),
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-6), np.log(1e-2)),
        "batch_size": hp.choice("batch_size", [256, 512, 1024, 2048]),
    }


# =====================================================================
# pytorch-tabular search spaces
# =====================================================================

def _tab_transformer_space() -> dict:
    return {
        "input_embed_dim": hp.choice("input_embed_dim", [32, 64, 128]),
        "num_heads": hp.choice("num_heads", [2, 4, 8]),
        "num_attn_blocks": hp.quniform("num_attn_blocks", 2, 6, 1),
        "attn_dropout": hp.uniform("attn_dropout", 0, 0.3),
        "add_shared_embedding": hp.choice("add_shared_embedding", [True, False]),
        "shared_embedding_fraction": hp.uniform("shared_embedding_fraction", 0.25, 0.75),
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-3)),
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-6), np.log(1e-2)),
        "batch_size": hp.choice("batch_size", [128, 256, 512]),
    }


def _category_embedding_space() -> dict:
    return {
        "depth": hp.quniform("depth", 2, 6, 1),
        "width": hp.quniform("width", 64, 512, 64),
        "dropout": hp.uniform("dropout", 0, 0.5),
        "activation": hp.choice("activation", ["ReLU", "GELU", "SiLU"]),
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-3)),
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-6), np.log(1e-2)),
        "batch_size": hp.choice("batch_size", [128, 256, 512]),
    }


def _tabnet_space() -> dict:
    return {
        "n_d": hp.choice("n_d", [8, 16, 32, 64]),
        "n_a": hp.choice("n_a", [8, 16, 32, 64]),
        "n_steps": hp.quniform("n_steps", 3, 10, 1),
        "gamma": hp.uniform("gamma", 1.0, 2.0),
        "relaxation_factor": hp.uniform("relaxation_factor", 1.0, 2.0),
        "lr": hp.loguniform("lr", np.log(1e-4), np.log(1e-2)),
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-6), np.log(1e-2)),
        "batch_size": hp.choice("batch_size", [256, 512, 1024]),
    }


def _gate_space() -> dict:
    return {
        "gflu_stages": hp.quniform("gflu_stages", 2, 10, 1),
        "tree_depth": hp.quniform("tree_depth", 3, 6, 1),
        "num_trees": hp.quniform("num_trees", 5, 30, 5),
        "binning_activation": hp.choice("binning_activation", ["entmoid", "sparsemoid"]),
        "feature_mask_function": hp.choice("feature_mask_function", ["softmax", "entmax"]),
        "lr": hp.loguniform("lr", np.log(1e-4), np.log(1e-2)),
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-6), np.log(1e-2)),
        "batch_size": hp.choice("batch_size", [128, 256, 512]),
    }


_PT_SPACE_MAP = {
    "tab_transformer": _tab_transformer_space,
    "category_embedding": _category_embedding_space,
    "tabnet": _tabnet_space,
    "gate": _gate_space,
}


def get_pytorch_tabular_search_space(model_type: str) -> dict:
    """Return Hyperopt search space for a pytorch-tabular model.

    Parameters
    ----------
    model_type : str
        One of ``"tab_transformer"``, ``"category_embedding"``,
        ``"tabnet"``, ``"gate"``.

    Returns
    -------
    dict
        Hyperopt space dict with model-specific keys.

    Raises
    ------
    ValueError
        If ``model_type`` is not supported.
    """
    if model_type not in _PT_SPACE_MAP:
        raise ValueError(
            f"Unknown pytorch-tabular model_type '{model_type}'. "
            f"Supported: {list(_PT_SPACE_MAP.keys())}"
        )
    return _PT_SPACE_MAP[model_type]()


# =====================================================================
# Resolution helpers
# =====================================================================

_EMBEDDING_SPACE_MAP = {
    "mlp": get_mlp_search_space,
}


def resolve_embedding_search_space(step_config: EmbeddingStepConfig) -> dict:
    """Return the effective search space for the embedding step.

    If ``step_config.search_space`` is provided, returns it directly.
    Otherwise returns the default space for ``step_config.model_type``.

    Parameters
    ----------
    step_config : EmbeddingStepConfig
        Embedding model configuration.

    Returns
    -------
    dict
        Hyperopt search space.
    """
    if step_config.search_space is not None:
        return step_config.search_space

    if step_config.model_type in _EMBEDDING_SPACE_MAP:
        return _EMBEDDING_SPACE_MAP[step_config.model_type]()

    if step_config.model_type in _PT_SPACE_MAP:
        return _PT_SPACE_MAP[step_config.model_type]()

    raise ValueError(
        f"No default search space for model_type '{step_config.model_type}'"
    )


def resolve_classical_search_space(
    step_config: ClassicalStepConfig,
    task: str,
) -> dict:
    """Return the effective search space for the classical step.

    If ``step_config.search_space`` is provided, returns it directly.
    Otherwise looks up the default space from the model registry.
    For ``model_type="custom"``, ``search_space`` is required.

    Parameters
    ----------
    step_config : ClassicalStepConfig
        Classical model configuration.
    task : str
        ``"binary"``, ``"multiclass"``, or ``"regression"``.

    Returns
    -------
    dict
        Hyperopt search space.
    """
    if step_config.search_space is not None:
        return step_config.search_space

    if step_config.model_type == "custom":
        raise ValueError(
            "model_type='custom' requires search_space to be set"
        )

    from hybrid_embed.classical.model_zoo import resolve_classical_step

    resolved = resolve_classical_step(step_config, task)
    return resolved["search_space"]


# =====================================================================
# Sampling utility
# =====================================================================

def sample_from_space(space: dict, seed: int | None = None) -> dict:
    """Sample one configuration from a Hyperopt space.

    Parameters
    ----------
    space : dict
        Hyperopt search space.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        A single sampled configuration.
    """
    rng = np.random.default_rng(seed)
    return hp_sample(space, rng=rng)


# =====================================================================
# Postprocessing
# =====================================================================

def _resolve_choice(value, choices: list):
    """Resolve an hp.choice value that may be an index or already resolved.

    Hyperopt's ``fmin`` returns integer indices for ``hp.choice``,
    while ``pyll.stochastic.sample`` returns the actual values.
    This handles both cases idempotently.
    """
    if value in choices:
        return value
    if isinstance(value, int) and 0 <= value < len(choices):
        return choices[value]
    return value


_CHOICE_VALUES_MLP = {
    "activation": ["relu", "gelu", "silu"],
    "use_layer_norm": [True, False],
    "batch_size": [256, 512, 1024, 2048],
}

_QUNIFORM_INT_KEYS_MLP = {"embedding_dim", "depth", "width"}


def postprocess_mlp_params(params: dict) -> dict:
    """Convert raw Hyperopt sample to clean MLP config dict.

    - Casts ``quniform`` floats to int
    - Derives ``hidden_dims`` list from ``depth`` and ``width``
    - Resolves ``hp.choice`` indices to actual values

    Parameters
    ----------
    params : dict
        Raw sample from ``sample_from_space`` or Hyperopt ``fmin``.

    Returns
    -------
    dict
        Clean config dict ready for model construction.
    """
    out = dict(params)

    for key, choices in _CHOICE_VALUES_MLP.items():
        if key in out:
            out[key] = _resolve_choice(out[key], choices)

    for key in _QUNIFORM_INT_KEYS_MLP:
        if key in out:
            out[key] = int(out[key])

    if "depth" in out and "width" in out:
        out["hidden_dims"] = [out["width"]] * out["depth"]

    return out


_CHOICE_VALUES_PT = {
    "tab_transformer": {
        "input_embed_dim": [32, 64, 128],
        "num_heads": [2, 4, 8],
        "add_shared_embedding": [True, False],
        "batch_size": [128, 256, 512],
    },
    "category_embedding": {
        "activation": ["ReLU", "GELU", "SiLU"],
        "batch_size": [128, 256, 512],
    },
    "tabnet": {
        "n_d": [8, 16, 32, 64],
        "n_a": [8, 16, 32, 64],
        "batch_size": [256, 512, 1024],
    },
    "gate": {
        "binning_activation": ["entmoid", "sparsemoid"],
        "feature_mask_function": ["softmax", "entmax"],
        "batch_size": [128, 256, 512],
    },
}

_QUNIFORM_INT_KEYS_PT = {
    "tab_transformer": {"num_attn_blocks"},
    "category_embedding": {"depth", "width"},
    "tabnet": {"n_steps"},
    "gate": {"gflu_stages", "tree_depth", "num_trees"},
}


def postprocess_pytorch_tabular_params(
    params: dict, model_type: str
) -> dict:
    """Convert raw Hyperopt sample to clean pytorch-tabular config.

    - Casts values to correct types
    - Resolves ``hp.choice`` indices to actual values
    - Derives ``layers`` from ``depth`` and ``width`` for
      ``category_embedding``
    - Validates ``input_embed_dim % num_heads == 0`` for
      ``tab_transformer`` (adjusts ``num_heads`` if needed)
    - Adds ``model_type`` to the returned dict

    Parameters
    ----------
    params : dict
        Raw sample from Hyperopt.
    model_type : str
        pytorch-tabular model type.

    Returns
    -------
    dict
        Clean config dict.
    """
    out = dict(params)

    choice_map = _CHOICE_VALUES_PT.get(model_type, {})
    for key, choices in choice_map.items():
        if key in out:
            out[key] = _resolve_choice(out[key], choices)

    int_keys = _QUNIFORM_INT_KEYS_PT.get(model_type, set())
    for key in int_keys:
        if key in out:
            out[key] = int(out[key])

    if model_type == "category_embedding" and "depth" in out and "width" in out:
        out["layers"] = f"{out['width']}-" * out["depth"]
        out["layers"] = out["layers"].rstrip("-")

    if model_type == "tab_transformer":
        embed_dim = out.get("input_embed_dim", 32)
        num_heads = out.get("num_heads", 2)
        if embed_dim % num_heads != 0:
            for candidate in [2, 4, 8]:
                if embed_dim % candidate == 0:
                    out["num_heads"] = candidate
                    break

    out["model_type"] = model_type
    return out


_QUNIFORM_INT_KEYS_CLASSICAL = {
    "n_estimators", "max_depth", "min_samples_leaf",
    "min_child_weight", "min_child_samples", "num_leaves",
    "iterations", "depth",
}

_CHOICE_VALUES_CLASSICAL = {
    "max_features": ["sqrt", "log2", 0.5, 0.8],
}


def postprocess_classical_params(params: dict) -> dict:
    """Convert raw Hyperopt sample to clean classical config dict.

    - Casts ``quniform`` floats to int
    - Resolves ``hp.choice`` indices to actual values

    Parameters
    ----------
    params : dict
        Raw sample from Hyperopt.

    Returns
    -------
    dict
        Clean config dict.
    """
    out = dict(params)

    for key, choices in _CHOICE_VALUES_CLASSICAL.items():
        if key in out:
            out[key] = _resolve_choice(out[key], choices)

    for key in _QUNIFORM_INT_KEYS_CLASSICAL:
        if key in out and isinstance(out[key], float):
            out[key] = int(out[key])

    return out
