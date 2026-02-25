"""Example 01: 5-fold CV, from-scratch MLP, XGBoost, binary classification.

Simplest full example. All defaults -- no custom HP spaces.
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
    embedding_step=EmbeddingStepConfig(model_type="mlp"),
    classical_step=ClassicalStepConfig(model_type="xgboost"),
    budget_config=BudgetConfig(
        embed_hpo_max_trials=50,
        classical_hpo_max_trials=150,
    ),
    run_config=RunConfig(
        n_folds=5,
        dataset_id="adult",
    ),
)

print(f"ROC-AUC: {result.mean_metrics['roc_auc']:.4f} "
      f"+/- {result.std_metrics['roc_auc']:.4f}")
