"""Example 06: 5-fold CV, MLP, Ridge regression.

Using a simple sklearn linear model as the classical step.
"""
from sklearn.datasets import fetch_california_housing
import pandas as pd

from hybrid_embed.config import (
    TaskConfig, BudgetConfig, RunConfig,
    EmbeddingStepConfig, ClassicalStepConfig,
)
from hybrid_embed.eval.runner import run_experiment

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

result = run_experiment(
    X=X,
    y=y,
    task_config=TaskConfig(task="regression"),
    embedding_step=EmbeddingStepConfig(model_type="mlp"),
    classical_step=ClassicalStepConfig(model_type="ridge"),
    budget_config=BudgetConfig(
        embed_hpo_max_trials=50,
        classical_hpo_max_trials=100,
    ),
    run_config=RunConfig(dataset_id="california_housing"),
)

print(f"RMSE: {result.mean_metrics['rmse']:.4f}")
