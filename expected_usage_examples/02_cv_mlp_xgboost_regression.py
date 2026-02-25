"""Example 02: 5-fold CV, from-scratch MLP, XGBoost, regression.

Same as example 01 but for a regression task.
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
    classical_step=ClassicalStepConfig(model_type="xgboost"),
    budget_config=BudgetConfig(
        embed_hpo_max_trials=50,
        classical_hpo_max_trials=150,
    ),
    run_config=RunConfig(
        n_folds=5,
        dataset_id="california_housing",
    ),
)

print(f"RMSE: {result.mean_metrics['rmse']:.4f} "
      f"+/- {result.std_metrics['rmse']:.4f}")
print(f"R2:   {result.mean_metrics['r_squared']:.4f} "
      f"+/- {result.std_metrics['r_squared']:.4f}")
