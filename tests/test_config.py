"""Tests for hybrid_embed.config: dataclasses, validation, serialization."""

import os
import tempfile

import yaml
from sklearn.svm import SVC

from hybrid_embed.config import (
    Schema,
    TaskConfig,
    BudgetConfig,
    EmbeddingStepConfig,
    ClassicalStepConfig,
    RunConfig,
    FoldResult,
    RunResult,
    validate_task_config,
    config_to_dict,
    save_config_yaml,
)


def test_schema_creation():
    """Verify Schema stores column type information correctly."""
    schema = Schema(
        numeric_columns=["age", "income"],
        categorical_columns=["city"],
        dropped_columns=["const_col"],
    )
    print(f"Schema numeric: {schema.numeric_columns}")
    print(f"Schema categorical: {schema.categorical_columns}")
    print(f"Schema dropped: {schema.dropped_columns}")

    assert schema.numeric_columns == ["age", "income"]
    assert schema.categorical_columns == ["city"]
    assert schema.dropped_columns == ["const_col"]
    print("Test passed: Schema fields are correct")


def test_budget_config_defaults():
    """Verify BudgetConfig defaults match spec Section 16."""
    bc = BudgetConfig()
    print(f"embed_hpo_max_trials: {bc.embed_hpo_max_trials}")
    print(f"embed_max_epochs: {bc.embed_max_epochs}")
    print(f"embed_patience: {bc.embed_patience}")
    print(f"classical_hpo_max_trials: {bc.classical_hpo_max_trials}")

    assert bc.embed_hpo_max_trials == 50
    assert bc.embed_max_epochs == 50
    assert bc.embed_patience == 8
    assert bc.embed_time_budget_seconds is None
    assert bc.classical_hpo_max_trials == 150
    assert bc.classical_time_budget_seconds is None
    print("Test passed: BudgetConfig defaults match spec")


def test_run_config_defaults():
    """Verify RunConfig defaults."""
    rc = RunConfig()
    print(f"n_folds: {rc.n_folds}")
    print(f"master_seed: {rc.master_seed}")
    print(f"device: {rc.device}")
    print(f"scaler: {rc.scaler}")

    assert rc.n_folds == 5
    assert rc.val_fraction == 0.2
    assert rc.master_seed == 42
    assert rc.device == "auto"
    assert rc.deterministic is False
    assert rc.save_embeddings is False
    assert rc.save_predictions is True
    assert rc.output_dir == "runs"
    assert rc.dataset_id == "default"
    assert rc.scaler == "standard"
    assert rc.max_categorical_cardinality == 1000
    assert rc.add_missing_indicator is False
    print("Test passed: RunConfig defaults are correct")


def test_embedding_step_config_defaults():
    """Verify EmbeddingStepConfig defaults."""
    ec = EmbeddingStepConfig()
    print(f"model_type: {ec.model_type}")
    print(f"search_space: {ec.search_space}")
    print(f"fixed_params: {ec.fixed_params}")

    assert ec.model_type == "mlp"
    assert ec.search_space is None
    assert ec.fixed_params is None
    print("Test passed: EmbeddingStepConfig defaults are correct")


def test_embedding_step_config_custom():
    """Verify EmbeddingStepConfig with custom values."""
    space = {"input_embed_dim": 64, "num_heads": 4}
    ec = EmbeddingStepConfig(
        model_type="tab_transformer",
        search_space=space,
    )
    print(f"model_type: {ec.model_type}")
    print(f"search_space: {ec.search_space}")

    assert ec.model_type == "tab_transformer"
    assert ec.search_space == space
    assert ec.fixed_params is None
    print("Test passed: EmbeddingStepConfig custom values set correctly")


def test_classical_step_config_defaults():
    """Verify ClassicalStepConfig defaults."""
    cc = ClassicalStepConfig()
    print(f"model_type: {cc.model_type}")
    print(f"supports_early_stopping: {cc.supports_early_stopping}")

    assert cc.model_type == "xgboost"
    assert cc.model_class is None
    assert cc.search_space is None
    assert cc.fixed_params is None
    assert cc.supports_early_stopping is None
    print("Test passed: ClassicalStepConfig defaults are correct")


def test_classical_step_config_custom_class():
    """Verify ClassicalStepConfig with a custom sklearn class."""
    space = {"C": [0.01, 0.1, 1.0], "kernel": ["rbf", "linear"]}
    cc = ClassicalStepConfig(
        model_type="custom",
        model_class=SVC,
        search_space=space,
        fixed_params={"probability": True},
    )
    print(f"model_type: {cc.model_type}")
    print(f"model_class: {cc.model_class}")
    print(f"search_space: {cc.search_space}")
    print(f"fixed_params: {cc.fixed_params}")

    assert cc.model_type == "custom"
    assert cc.model_class is SVC
    assert cc.search_space == space
    assert cc.fixed_params == {"probability": True}
    print("Test passed: ClassicalStepConfig custom class set correctly")


def test_validate_task_config_valid():
    """Verify valid task types pass validation."""
    for task in ("binary", "multiclass", "regression"):
        tc = TaskConfig(task=task)
        validate_task_config(tc)
        print(f"Task '{task}' validated OK")
    print("Test passed: all valid tasks accepted")


def test_validate_task_config_invalid():
    """Verify invalid task type raises ValueError."""
    tc = TaskConfig(task="ranking")
    try:
        validate_task_config(tc)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
    print("Test passed: invalid task rejected")


def test_config_to_dict_roundtrip():
    """Verify config_to_dict produces correct keys and values."""
    rc = RunConfig(n_folds=3, master_seed=123, dataset_id="test")
    d = config_to_dict(rc)
    print(f"Dict keys: {list(d.keys())}")
    print(f"Dict values: {d}")

    assert d["n_folds"] == 3
    assert d["master_seed"] == 123
    assert d["dataset_id"] == "test"
    assert d["device"] == "auto"
    assert isinstance(d, dict)
    print("Test passed: config_to_dict roundtrip correct")


def test_config_to_dict_custom_class():
    """Verify config_to_dict handles model_class (type) correctly."""
    cc = ClassicalStepConfig(model_type="custom", model_class=SVC)
    d = config_to_dict(cc)
    print(f"model_class serialized as: {d['model_class']}")

    assert isinstance(d["model_class"], str)
    assert "SVC" in d["model_class"]
    print("Test passed: model_class serialized as string")


def test_save_config_yaml():
    """Verify save_config_yaml writes readable YAML."""
    rc = RunConfig(n_folds=3, master_seed=99, dataset_id="yaml_test")
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "config.yaml")

    save_config_yaml(rc, path)
    print(f"Saved config to {path}")

    with open(path, "r") as fh:
        loaded = yaml.safe_load(fh)
    print(f"Loaded YAML: {loaded}")

    assert loaded["n_folds"] == 3
    assert loaded["master_seed"] == 99
    assert loaded["dataset_id"] == "yaml_test"
    print("Test passed: YAML save/load roundtrip correct")
