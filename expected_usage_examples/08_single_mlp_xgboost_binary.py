"""Example 08: Single train/test split, MLP, XGBoost, binary.

Uses the sklearn-style HybridTabularModel API instead of
run_experiment. No cross-validation -- just a single split
for fast iteration.
"""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from hybrid_embed.config import (
    BudgetConfig, RunConfig,
    EmbeddingStepConfig, ClassicalStepConfig,
)
from hybrid_embed.eval.runner import HybridTabularModel

X, y = fetch_openml("adult", version=2, return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

model = HybridTabularModel(
    task="binary",
    embedding=EmbeddingStepConfig(model_type="mlp"),
    classical=ClassicalStepConfig(model_type="xgboost"),
    budget_config=BudgetConfig(
        embed_hpo_max_trials=20,
        classical_hpo_max_trials=50,
    ),
    run_config=RunConfig(),
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

from sklearn.metrics import roc_auc_score, accuracy_score

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba[:, 1]):.4f}")
