"""Tests for hybrid_embed.classical.model_zoo."""

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import hp

from hybrid_embed.config import ClassicalStepConfig
from hybrid_embed.classical.model_zoo import (
    build_model,
    resolve_classical_step,
    train_classical_model,
)


def _make_embedding_data(n_samples=200, n_features=64, seed=42):
    """Synthetic embedding-like data for testing classical models."""
    rng = np.random.RandomState(seed)
    E = rng.randn(n_samples, n_features).astype(np.float64)
    y_binary = (E[:, 0] + E[:, 1] > 0).astype(int)
    y_multi = (np.digitize(E[:, 0], bins=[-1, -0.3, 0.3, 1.0])).clip(0, 4)
    y_reg = (3.0 * E[:, 0] + E[:, 1] + rng.randn(n_samples) * 0.5).astype(np.float64)
    return E, y_binary, y_multi, y_reg


# ==========================================================================
# Resolution tests
# ==========================================================================

def test_resolve_builtin_model_defaults():
    """Resolving xgboost with no overrides returns correct defaults."""
    config = ClassicalStepConfig(model_type="xgboost")
    result = resolve_classical_step(config, task="binary")

    import xgboost
    print(f"Resolved class: {result['class'].__name__}")
    print(f"supports_early_stopping: {result['supports_early_stopping']}")
    print(f"search_space keys: {sorted(result['search_space'].keys())}")

    assert result["class"] is xgboost.XGBClassifier
    assert result["supports_early_stopping"] is True
    assert "n_estimators" in result["search_space"]
    assert "learning_rate" in result["search_space"]
    print("Test passed: xgboost defaults resolved correctly")


def test_resolve_builtin_model_regression():
    """Resolving xgboost for regression returns the regressor class."""
    config = ClassicalStepConfig(model_type="xgboost")
    result = resolve_classical_step(config, task="regression")

    import xgboost
    print(f"Resolved class: {result['class'].__name__}")
    assert result["class"] is xgboost.XGBRegressor
    print("Test passed: xgboost regression class resolved correctly")


def test_resolve_builtin_model_custom_space():
    """User-provided search_space overrides the default."""
    custom_space = {"n_estimators": hp.quniform("n_estimators", 10, 50, 10)}
    config = ClassicalStepConfig(
        model_type="xgboost", search_space=custom_space,
    )
    result = resolve_classical_step(config, task="binary")

    assert result["search_space"] is custom_space
    assert len(result["search_space"]) == 1
    print("Test passed: custom search_space used instead of default")


def test_resolve_builtin_model_fixed_params():
    """fixed_params are merged with defaults."""
    config = ClassicalStepConfig(
        model_type="ridge", fixed_params={"alpha": 1.0},
    )
    result = resolve_classical_step(config, task="regression")

    print(f"Resolved params: {result['params']}")
    assert result["params"]["alpha"] == 1.0
    print("Test passed: fixed_params merged correctly")


def test_resolve_custom_model():
    """Custom model resolution returns all user-provided fields."""
    custom_space = {"n_neighbors": hp.quniform("n_neighbors", 1, 20, 1)}
    config = ClassicalStepConfig(
        model_type="custom",
        model_class=KNeighborsClassifier,
        search_space=custom_space,
        fixed_params={"weights": "distance"},
    )
    result = resolve_classical_step(config, task="binary")

    print(f"Resolved class: {result['class'].__name__}")
    assert result["class"] is KNeighborsClassifier
    assert result["search_space"] is custom_space
    assert result["params"]["weights"] == "distance"
    assert result["supports_early_stopping"] is False
    print("Test passed: custom model resolved correctly")


def test_resolve_custom_model_missing_class():
    """Custom model without model_class should raise ValueError."""
    config = ClassicalStepConfig(
        model_type="custom",
        search_space={"k": hp.quniform("k", 1, 10, 1)},
    )
    try:
        resolve_classical_step(config, task="binary")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
        print("Test passed: missing model_class raises ValueError")


def test_resolve_custom_model_missing_space():
    """Custom model without search_space should raise ValueError."""
    config = ClassicalStepConfig(
        model_type="custom",
        model_class=KNeighborsClassifier,
    )
    try:
        resolve_classical_step(config, task="binary")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly raised ValueError: {e}")
        print("Test passed: missing search_space raises ValueError")


def test_resolve_all_builtin_models():
    """All built-in model types should resolve without errors."""
    model_types = [
        "logistic_regression", "ridge", "elastic_net",
        "random_forest", "extra_trees",
        "xgboost", "lightgbm", "catboost",
    ]
    for mt in model_types:
        config = ClassicalStepConfig(model_type=mt)
        result = resolve_classical_step(config, task="binary")
        assert result["class"] is not None
        assert isinstance(result["search_space"], dict)
        print(f"  {mt}: class={result['class'].__name__}, "
              f"es={result['supports_early_stopping']}")
    print("Test passed: all built-in models resolve correctly")


# ==========================================================================
# Build tests
# ==========================================================================

def test_build_model_xgboost():
    """XGBoost model should be instantiated with correct params."""
    import xgboost
    model = build_model(
        xgboost.XGBClassifier,
        {"n_estimators": 50, "max_depth": 3},
        task="binary",
    )
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    print(f"Built model: {type(model).__name__}")
    print("Test passed: XGBoost model built correctly")


def test_build_model_custom():
    """Custom model class should be instantiated."""
    model = build_model(
        KNeighborsClassifier,
        {"n_neighbors": 5, "weights": "distance"},
        task="binary",
    )
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    print(f"Built model: {type(model).__name__}")
    print("Test passed: custom model built correctly")


# ==========================================================================
# Training tests
# ==========================================================================

def test_train_xgboost_with_early_stopping():
    """XGBoost should train with early stopping on validation data."""
    import xgboost
    E, y_bin, _, _ = _make_embedding_data()
    model = build_model(
        xgboost.XGBClassifier,
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
        task="binary",
    )
    trained = train_classical_model(
        model, E[:150], y_bin[:150],
        E_val=E[150:], y_val=y_bin[150:],
        supports_early_stopping=True,
    )
    preds = trained.predict(E[150:])
    print(f"XGBoost predictions shape: {preds.shape}")
    assert preds.shape == (50,)
    print("Test passed: XGBoost trains with early stopping")


def test_train_lightgbm_with_early_stopping():
    """LightGBM should train with early stopping."""
    import lightgbm
    E, y_bin, _, _ = _make_embedding_data()
    model = build_model(
        lightgbm.LGBMClassifier,
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
        task="binary",
    )
    trained = train_classical_model(
        model, E[:150], y_bin[:150],
        E_val=E[150:], y_val=y_bin[150:],
        supports_early_stopping=True,
    )
    preds = trained.predict(E[150:])
    print(f"LightGBM predictions shape: {preds.shape}")
    assert preds.shape == (50,)
    print("Test passed: LightGBM trains with early stopping")


def test_train_catboost_with_early_stopping():
    """CatBoost should train with early stopping."""
    import catboost
    E, y_bin, _, _ = _make_embedding_data()
    model = build_model(
        catboost.CatBoostClassifier,
        {"iterations": 200, "depth": 3, "learning_rate": 0.1},
        task="binary",
    )
    trained = train_classical_model(
        model, E[:150], y_bin[:150],
        E_val=E[150:], y_val=y_bin[150:],
        supports_early_stopping=True,
    )
    preds = trained.predict(E[150:])
    print(f"CatBoost predictions shape: {preds.shape}")
    assert preds.shape == (50,)
    print("Test passed: CatBoost trains with early stopping")


def test_train_ridge():
    """Ridge should train without validation set."""
    from sklearn.linear_model import Ridge
    E, _, _, y_reg = _make_embedding_data()
    model = build_model(Ridge, {"alpha": 1.0}, task="regression")
    trained = train_classical_model(model, E[:150], y_reg[:150])
    preds = trained.predict(E[150:])
    print(f"Ridge predictions shape: {preds.shape}")
    assert preds.shape == (50,)
    print("Test passed: Ridge trains and predicts correctly")


def test_train_custom_model():
    """Custom model should train and predict correctly."""
    E, y_bin, _, _ = _make_embedding_data()
    model = build_model(
        KNeighborsClassifier,
        {"n_neighbors": 5},
        task="binary",
    )
    trained = train_classical_model(model, E[:150], y_bin[:150])
    preds = trained.predict(E[150:])
    print(f"KNN predictions shape: {preds.shape}")
    assert preds.shape == (50,)
    print("Test passed: custom model trains and predicts correctly")


# ==========================================================================
# Sanity tests: random labels should NOT produce predictive models
# ==========================================================================

def test_xgboost_random_labels_no_leakage():
    """XGBoost on random binary labels should stay near chance."""
    import xgboost
    rng = np.random.RandomState(42)
    E = rng.randn(400, 64).astype(np.float64)
    y_random = rng.choice([0, 1], size=400).astype(int)

    model = build_model(
        xgboost.XGBClassifier,
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1},
        task="binary",
    )
    trained = train_classical_model(
        model, E[:300], y_random[:300],
        E_val=E[300:], y_val=y_random[300:],
        supports_early_stopping=True,
    )
    proba = trained.predict_proba(E[300:])[:, 1]
    auc = roc_auc_score(y_random[300:], proba)
    print(f"XGBoost random-label ROC-AUC: {auc:.4f}")
    assert auc < 0.65, (
        f"XGBoost should not learn from random labels; ROC-AUC={auc:.4f}"
    )
    print("Test passed: no leakage with XGBoost on random labels")


def test_lightgbm_random_labels_no_leakage():
    """LightGBM on random binary labels should stay near chance."""
    import lightgbm
    rng = np.random.RandomState(42)
    E = rng.randn(400, 64).astype(np.float64)
    y_random = rng.choice([0, 1], size=400).astype(int)

    model = build_model(
        lightgbm.LGBMClassifier,
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1},
        task="binary",
    )
    trained = train_classical_model(
        model, E[:300], y_random[:300],
        E_val=E[300:], y_val=y_random[300:],
        supports_early_stopping=True,
    )
    proba = trained.predict_proba(E[300:])[:, 1]
    auc = roc_auc_score(y_random[300:], proba)
    print(f"LightGBM random-label ROC-AUC: {auc:.4f}")
    assert auc < 0.65, (
        f"LightGBM should not learn from random labels; ROC-AUC={auc:.4f}"
    )
    print("Test passed: no leakage with LightGBM on random labels")


def test_random_forest_random_labels_no_leakage():
    """RandomForest on random regression targets should have R^2 near 0."""
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.RandomState(42)
    E = rng.randn(400, 64).astype(np.float64)
    y_random = rng.randn(400).astype(np.float64)

    model = build_model(
        RandomForestRegressor,
        {"n_estimators": 100, "max_depth": 5},
        task="regression",
    )
    trained = train_classical_model(model, E[:300], y_random[:300])
    preds = trained.predict(E[300:])
    r2 = r2_score(y_random[300:], preds)
    print(f"RandomForest random-label R^2: {r2:.4f}")
    assert r2 < 0.15, (
        f"RandomForest should not learn from random targets; R^2={r2:.4f}"
    )
    print("Test passed: no leakage with RandomForest on random labels")
