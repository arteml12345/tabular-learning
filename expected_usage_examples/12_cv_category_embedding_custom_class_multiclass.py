"""Example 12: 5-fold CV, CategoryEmbedding, custom model, multiclass.

Demonstrates a pytorch-tabular model combined with a user-provided
classical model for multiclass classification.
"""
import numpy as np
from hyperopt import hp
from sklearn.datasets import fetch_covtype
from sklearn.neighbors import KNeighborsClassifier
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
    embedding_step=EmbeddingStepConfig(model_type="category_embedding"),
    classical_step=ClassicalStepConfig(
        model_type="custom",
        model_class=KNeighborsClassifier,
        search_space={
            "n_neighbors": hp.quniform("k", 3, 25, 2),
            "weights": hp.choice("w", ["uniform", "distance"]),
            "metric": hp.choice("m", ["euclidean", "manhattan"]),
        },
    ),
    budget_config=BudgetConfig(
        embed_hpo_max_trials=25,
        classical_hpo_max_trials=40,
    ),
    run_config=RunConfig(dataset_id="covertype"),
)

print(f"Accuracy: {result.mean_metrics['accuracy']:.4f}")
print(f"Best KNN config: {result.best_classical_config}")
