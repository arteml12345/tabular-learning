"""Example 05: 5-fold CV, MLP (defaults), XGBoost with CUSTOM HP space, binary.

Demonstrates overriding the default XGBoost search space with a
narrower, researcher-defined space.
"""
import numpy as np
from hyperopt import hp
from sklearn.datasets import fetch_openml

from hybrid_embed.config import (
    TaskConfig, BudgetConfig, RunConfig,
    EmbeddingStepConfig, ClassicalStepConfig,
)
from hybrid_embed.eval.runner import run_experiment

X, y = fetch_openml("adult", version=2, return_X_y=True, as_frame=True)

result = run_experiment(
    X=X,
    y=y,
    task_config=TaskConfig(task="binary"),
    embedding_step=EmbeddingStepConfig(model_type="mlp"),
    classical_step=ClassicalStepConfig(
        model_type="xgboost",
        search_space={
            "max_depth": hp.quniform("max_depth", 3, 8, 1),
            "learning_rate": hp.loguniform(
                "lr", np.log(0.01), np.log(0.3)
            ),
            "n_estimators": hp.quniform("n_est", 100, 500, 50),
            "subsample": hp.uniform("subsample", 0.6, 1.0),
        },
        fixed_params={
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
    ),
    budget_config=BudgetConfig(classical_hpo_max_trials=80),
    run_config=RunConfig(dataset_id="adult"),
)

print(f"ROC-AUC: {result.mean_metrics['roc_auc']:.4f}")
print(f"Best XGBoost config: {result.best_classical_config}")
