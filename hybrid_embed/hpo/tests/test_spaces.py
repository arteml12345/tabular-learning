"""Tests for hybrid_embed.hpo.spaces."""

import pytest

from hyperopt import hp

from hybrid_embed.config import ClassicalStepConfig, EmbeddingStepConfig
from hybrid_embed.hpo.spaces import (
    get_mlp_search_space,
    get_pytorch_tabular_search_space,
    postprocess_classical_params,
    postprocess_mlp_params,
    postprocess_pytorch_tabular_params,
    resolve_classical_search_space,
    resolve_embedding_search_space,
    sample_from_space,
)


# ==================================================================
# MLP
# ==================================================================

def test_mlp_default_space_sample():
    """Sample 10 configs from default MLP space; all must be valid."""
    space = get_mlp_search_space()
    required_keys = {
        "embedding_dim", "depth", "width", "dropout",
        "activation", "use_layer_norm", "lr", "weight_decay",
        "batch_size",
    }
    for i in range(10):
        s = sample_from_space(space, seed=i)
        assert required_keys.issubset(s.keys()), (
            f"Sample {i} missing keys: {required_keys - s.keys()}"
        )
        assert 32 <= s["embedding_dim"] <= 256
        assert 2 <= s["depth"] <= 8
        assert 128 <= s["width"] <= 2048
        assert 0 <= s["dropout"] <= 0.5
        assert 1e-5 <= s["lr"] <= 1e-2
        assert 1e-6 <= s["weight_decay"] <= 1e-2
    print("Test passed: 10 MLP samples all valid")


def test_postprocess_mlp_params():
    """Postprocessed MLP params should have hidden_dims as list[int]."""
    space = get_mlp_search_space()
    for i in range(5):
        raw = sample_from_space(space, seed=i)
        pp = postprocess_mlp_params(raw)
        assert isinstance(pp["hidden_dims"], list)
        assert len(pp["hidden_dims"]) == pp["depth"]
        assert all(isinstance(d, int) for d in pp["hidden_dims"])
        assert isinstance(pp["embedding_dim"], int)
        assert pp["activation"] in ("relu", "gelu", "silu")
        assert isinstance(pp["use_layer_norm"], bool)
        assert pp["batch_size"] in (256, 512, 1024, 2048)
    print("Test passed: MLP postprocessing produces clean dicts")


# ==================================================================
# pytorch-tabular
# ==================================================================

def test_pytorch_tabular_default_space_tab_transformer():
    """Sample 10 configs for tab_transformer; all must be valid."""
    space = get_pytorch_tabular_search_space("tab_transformer")
    required_keys = {
        "input_embed_dim", "num_heads", "num_attn_blocks",
        "attn_dropout", "add_shared_embedding",
        "shared_embedding_fraction", "lr", "weight_decay",
        "batch_size",
    }
    for i in range(10):
        s = sample_from_space(space, seed=i)
        assert required_keys.issubset(s.keys())
        assert 0 <= s["attn_dropout"] <= 0.3
        assert 0.25 <= s["shared_embedding_fraction"] <= 0.75
    print("Test passed: 10 TabTransformer samples all valid")


def test_pytorch_tabular_default_space_category_embedding():
    """Sample configs for category_embedding; verify keys."""
    space = get_pytorch_tabular_search_space("category_embedding")
    for i in range(5):
        s = sample_from_space(space, seed=i)
        assert "depth" in s
        assert "width" in s
        assert "activation" in s
    print("Test passed: CategoryEmbedding samples valid")


def test_pytorch_tabular_default_space_tabnet():
    """Sample configs for tabnet; verify keys."""
    space = get_pytorch_tabular_search_space("tabnet")
    for i in range(5):
        s = sample_from_space(space, seed=i)
        assert "n_d" in s
        assert "n_a" in s
        assert "n_steps" in s
    print("Test passed: TabNet samples valid")


def test_pytorch_tabular_default_space_gate():
    """Sample configs for gate; verify keys."""
    space = get_pytorch_tabular_search_space("gate")
    for i in range(5):
        s = sample_from_space(space, seed=i)
        assert "gflu_stages" in s
        assert "tree_depth" in s
        assert "num_trees" in s
    print("Test passed: GATE samples valid")


def test_pytorch_tabular_unknown_model():
    """Unknown model type should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown pytorch-tabular"):
        get_pytorch_tabular_search_space("nonexistent_model")
    print("Test passed: unknown model type raises ValueError")


def test_postprocess_pytorch_tabular_tab_transformer():
    """Postprocessed tab_transformer params should be clean."""
    space = get_pytorch_tabular_search_space("tab_transformer")
    for i in range(10):
        raw = sample_from_space(space, seed=i)
        pp = postprocess_pytorch_tabular_params(raw, "tab_transformer")
        assert pp["model_type"] == "tab_transformer"
        assert isinstance(pp["num_attn_blocks"], int)
        assert pp["input_embed_dim"] in (32, 64, 128)
        assert pp["num_heads"] in (2, 4, 8)
        assert pp["input_embed_dim"] % pp["num_heads"] == 0, (
            f"embed_dim={pp['input_embed_dim']} not divisible "
            f"by num_heads={pp['num_heads']}"
        )
    print("Test passed: TabTransformer postprocessing valid with "
          "embed_dim/num_heads constraint")


def test_postprocess_pytorch_tabular_category_embedding():
    """Postprocessed category_embedding should have layers string."""
    space = get_pytorch_tabular_search_space("category_embedding")
    for i in range(5):
        raw = sample_from_space(space, seed=i)
        pp = postprocess_pytorch_tabular_params(raw, "category_embedding")
        assert pp["model_type"] == "category_embedding"
        assert "layers" in pp
        parts = pp["layers"].split("-")
        assert len(parts) == pp["depth"]
    print("Test passed: CategoryEmbedding postprocessing produces layers")


# ==================================================================
# Resolution
# ==================================================================

def test_resolve_embedding_space_default():
    """EmbeddingStepConfig with search_space=None returns default."""
    config = EmbeddingStepConfig(model_type="mlp")
    space = resolve_embedding_search_space(config)
    assert "embedding_dim" in space
    assert "depth" in space
    print("Test passed: default MLP space resolved")


def test_resolve_embedding_space_default_pt():
    """Resolving a pytorch-tabular model type returns its default."""
    config = EmbeddingStepConfig(model_type="tab_transformer")
    space = resolve_embedding_search_space(config)
    assert "input_embed_dim" in space
    assert "num_heads" in space
    print("Test passed: default TabTransformer space resolved")


def test_resolve_embedding_space_override():
    """User-provided search_space overrides defaults."""
    custom = {"my_param": hp.uniform("my_param", 0, 1)}
    config = EmbeddingStepConfig(model_type="mlp", search_space=custom)
    space = resolve_embedding_search_space(config)
    assert space is custom
    print("Test passed: user override used for embedding space")


def test_resolve_classical_space_default():
    """ClassicalStepConfig with search_space=None returns default."""
    config = ClassicalStepConfig(model_type="xgboost")
    space = resolve_classical_search_space(config, task="binary")
    assert "n_estimators" in space
    assert "learning_rate" in space
    print("Test passed: default xgboost space resolved")


def test_resolve_classical_space_override():
    """User-provided search_space overrides defaults."""
    custom = {"n_estimators": hp.quniform("n_estimators", 10, 50, 10)}
    config = ClassicalStepConfig(model_type="xgboost", search_space=custom)
    space = resolve_classical_search_space(config, task="binary")
    assert space is custom
    print("Test passed: user override used for classical space")


def test_resolve_classical_space_custom_missing():
    """Custom model without search_space raises ValueError."""
    config = ClassicalStepConfig(model_type="custom")
    with pytest.raises(ValueError, match="requires search_space"):
        resolve_classical_search_space(config, task="binary")
    print("Test passed: custom without space raises ValueError")


# ==================================================================
# Classical postprocessing
# ==================================================================

def test_postprocess_classical_params():
    """Classical postprocessing casts types correctly."""
    raw = {
        "n_estimators": 200.0,
        "max_depth": 5.0,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "max_features": 2,
    }
    pp = postprocess_classical_params(raw)
    assert isinstance(pp["n_estimators"], int)
    assert pp["n_estimators"] == 200
    assert isinstance(pp["max_depth"], int)
    assert pp["max_depth"] == 5
    assert isinstance(pp["learning_rate"], float)
    assert pp["max_features"] == 0.5
    print("Test passed: classical postprocessing produces correct types")
