"""Example 10: 5-fold CV, TabTransformer with CUSTOM HP space, XGBoost, binary.

Demonstrates overriding the default TabTransformer search space.
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
        model_type="tab_transformer",
        search_space={
            "input_embed_dim": hp.choice("embed_dim", [32, 64]),
            "num_heads": hp.choice("heads", [2, 4]),
            "num_attn_blocks": hp.quniform("blocks", 2, 4, 1),
            "lr": hp.loguniform("lr", np.log(1e-4), np.log(5e-3)),
        },
        fixed_params={
            "attn_dropout": 0.1,
            "batch_size": 256,
        },
    ),
    classical_step=ClassicalStepConfig(model_type="xgboost"),
    budget_config=BudgetConfig(embed_hpo_max_trials=25),
    run_config=RunConfig(dataset_id="adult"),
)

print(f"ROC-AUC: {result.mean_metrics['roc_auc']:.4f}")
print(f"Best TabTransformer config: {result.best_embedding_config}")
