"""Example 09: 5-fold CV, TabTransformer (pytorch-tabular), XGBoost, binary.

Uses a pre-built TabTransformer model via the pytorch-tabular
adapter (Mode B) with default HP search space.
"""
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
    embedding_step=EmbeddingStepConfig(model_type="tab_transformer"),
    classical_step=ClassicalStepConfig(model_type="xgboost"),
    budget_config=BudgetConfig(
        embed_hpo_max_trials=30,
        classical_hpo_max_trials=100,
    ),
    run_config=RunConfig(dataset_id="adult"),
)

print(f"ROC-AUC: {result.mean_metrics['roc_auc']:.4f}")
