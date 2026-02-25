"""Example 04: 5-fold CV, MLP with CUSTOM HP space, XGBoost, binary.

Demonstrates overriding the default MLP search space with a narrower,
user-defined space. Parameters not in the search space are set via
fixed_params.
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
    embedding_step=EmbeddingStepConfig(
        model_type="mlp",
        search_space={
            "depth": hp.quniform("depth", 2, 4, 1),
            "width": hp.choice("width", [128, 256]),
            "dropout": hp.uniform("dropout", 0.0, 0.3),
            "lr": hp.loguniform("lr", np.log(1e-4), np.log(1e-2)),
        },
        fixed_params={
            "activation": "gelu",
            "use_layer_norm": True,
            "embedding_dim": 64,
            "weight_decay": 1e-5,
            "batch_size": 512,
        },
    ),
    classical_step=ClassicalStepConfig(model_type="xgboost"),
    budget_config=BudgetConfig(embed_hpo_max_trials=30),
    run_config=RunConfig(dataset_id="adult"),
)

print(f"ROC-AUC: {result.mean_metrics['roc_auc']:.4f}")
print(f"Best MLP config: {result.best_embedding_config}")
