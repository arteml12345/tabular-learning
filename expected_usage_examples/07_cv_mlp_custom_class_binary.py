"""Example 07: 5-fold CV, MLP, CUSTOM classical model class, binary.

Demonstrates passing your own sklearn-compatible model class with
its own HP search space to the framework.
"""
import numpy as np
from hyperopt import hp
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC

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
        model_type="custom",
        model_class=SVC,
        search_space={
            "C": hp.loguniform("C", np.log(1e-3), np.log(1e3)),
            "kernel": hp.choice("kernel", ["rbf", "linear"]),
            "gamma": hp.choice("gamma", ["scale", "auto"]),
        },
        fixed_params={"probability": True},
    ),
    budget_config=BudgetConfig(classical_hpo_max_trials=40),
    run_config=RunConfig(dataset_id="adult"),
)

print(f"ROC-AUC: {result.mean_metrics['roc_auc']:.4f}")
print(f"Best SVC config: {result.best_classical_config}")
