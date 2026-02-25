"""Example 03: 5-fold CV, from-scratch MLP, XGBoost, multiclass.

Same as example 01 but for a multiclass classification task.
"""
from sklearn.datasets import fetch_covtype
import pandas as pd

from hybrid_embed.config import (
    TaskConfig, BudgetConfig, RunConfig,
    EmbeddingStepConfig, ClassicalStepConfig,
)
from hybrid_embed.eval.runner import run_experiment

data = fetch_covtype()
X = pd.DataFrame(data.data)
y = data.target

result = run_experiment(
    X=X,
    y=y,
    task_config=TaskConfig(task="multiclass"),
    embedding_step=EmbeddingStepConfig(model_type="mlp"),
    classical_step=ClassicalStepConfig(model_type="xgboost"),
    budget_config=BudgetConfig(
        embed_hpo_max_trials=30,
        classical_hpo_max_trials=100,
    ),
    run_config=RunConfig(
        n_folds=5,
        dataset_id="covertype",
    ),
)

print(f"Accuracy: {result.mean_metrics['accuracy']:.4f} "
      f"+/- {result.std_metrics['accuracy']:.4f}")
