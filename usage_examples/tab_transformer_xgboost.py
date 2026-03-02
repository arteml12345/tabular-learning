"""Usage example: TabTransformer embedding + XGBoost classical model.

Demonstrates the full hybrid_embed pipeline for three task types:
  1. Binary classification
  2. Multiclass classification
  3. Regression

Every configuration parameter is set explicitly (nothing left to
default) and annotated with a comment explaining its role.

For each task the data is split 50/50 into framework data and
out-of-sample (OOS) data. The framework trains on the first half
via HybridTabularModel. Then both the NN-only head and the full
hybrid (NN + XGBoost) are evaluated on the held-out OOS half, so
one can compare whether the classical model on top of embeddings
improves over the raw NN predictions.
"""


import json
import os

import numpy as np
import pandas as pd
from hyperopt import hp
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from hybrid_embed.config import (
    BudgetConfig,
    ClassicalStepConfig,
    EmbeddingStepConfig,
    RunConfig,
    TaskConfig,
)
from hybrid_embed.eval.runner import HybridTabularModel, run_experiment


# =====================================================================
# Synthetic data generators
# =====================================================================

def make_binary_data(n=1000, seed=42):
    """Synthetic binary classification dataset.

    4 numeric features, 2 categorical features, and a binary target
    derived from a learnable linear combination of the numerics.
    """
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "age": rng.normal(40, 12, n).round(1),
        "income": rng.randn(n),
        "score_a": rng.randn(n),
        "score_b": rng.randn(n),
        "department": rng.choice(["engineering", "sales", "marketing", "hr"], n),
        "region": rng.choice(["north", "south", "east", "west"], n),
    })
    y = (df["score_a"] + 0.5 * df["score_b"] > 0).astype(int).values
    return df, y


def make_multiclass_data(n=1200, n_classes=4, seed=42):
    """Synthetic multiclass dataset with 4 classes.

    Target is derived from quantiles of a linear combination,
    ensuring all classes are roughly balanced.
    """
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "feat_0": rng.randn(n),
        "feat_1": rng.randn(n),
        "feat_2": rng.randn(n),
        "feat_3": rng.randn(n),
        "color": rng.choice(["red", "green", "blue"], n),
        "size": rng.choice(["S", "M", "L", "XL"], n),
    })
    score = df["feat_0"] + 0.5 * df["feat_1"] - 0.3 * df["feat_2"]
    boundaries = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
    y = np.digitize(score, boundaries).astype(int)
    return df, y


def make_regression_data(n=1000, seed=42):
    """Synthetic regression dataset with continuous target.

    Target is a noisy linear combination of the numeric features.
    """
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "x0": rng.randn(n),
        "x1": rng.randn(n),
        "x2": rng.randn(n),
        "x3": rng.randn(n),
        "material": rng.choice(["wood", "steel", "concrete"], n),
        "grade": rng.choice(["A", "B", "C"], n),
    })
    y = (3.0 * df["x0"].values
         + 1.5 * df["x1"].values
         - 0.8 * df["x2"].values
         + rng.randn(n) * 0.5)
    return df, y


# =====================================================================
# Embedding model search space: TabTransformer
# =====================================================================

# Custom TabTransformer embedding search space.
tab_transformer_search_space = {
    # Dimensionality of categorical embeddings fed into the transformer.
    "input_embed_dim": hp.choice("input_embed_dim", [32, 64, 128]),

    # Number of attention heads per transformer block.
    "num_heads": hp.choice("num_heads", [2, 4, 8]),

    # Number of stacked transformer attention blocks.
    "num_attn_blocks": hp.quniform("num_attn_blocks", 2, 6, 1),

    # Dropout rate within the attention mechanism.
    "attn_dropout": hp.uniform("attn_dropout", 0, 0.3),

    # Dropout rate within the feed-forward sublayers.
    "ff_dropout": hp.uniform("ff_dropout", 0, 0.3),

    # Whether to add a shared embedding layer across all categoricals.
    "add_shared_embedding": hp.choice("add_shared_embedding", [True, False]),

    # Fraction of embedding dim allocated to the shared embedding.
    "shared_embedding_fraction": hp.uniform("shared_embedding_fraction", 0.25, 0.75),

    # Learning rate for the AdamW optimizer.
    "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-3)),

    # L2 weight decay for the AdamW optimizer.
    "weight_decay": hp.loguniform("weight_decay", np.log(1e-6), np.log(1e-2)),

    # Mini-batch size for training.
    "batch_size": hp.choice("batch_size", [128, 256, 512]),
}

# No fixed params needed for TabTransformer.
tab_transformer_fixed_params = None


# =====================================================================
# Classical model search space: XGBoost
# =====================================================================

# Custom XGBoost search space (same for classification and regression).
xgboost_search_space = {
    # Total number of boosting rounds.
    "n_estimators": hp.quniform("n_estimators", 50, 500, 50),

    # Maximum tree depth per booster.
    "max_depth": hp.quniform("max_depth", 3, 8, 1),

    # Step size shrinkage (learning rate) to prevent overfitting.
    "learning_rate": hp.loguniform("learning_rate", np.log(1e-3), np.log(0.3)),

    # Fraction of training rows sampled per tree (row subsampling).
    "subsample": hp.uniform("subsample", 0.5, 1.0),

    # Fraction of features sampled per tree (column subsampling).
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),

    # Minimum sum of instance weight (hessian) needed in a child node.
    "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),

    # Minimum loss reduction required to make a further partition on a leaf.
    "gamma": hp.loguniform("gamma", np.log(1e-8), np.log(1.0)),

    # L1 regularization on leaf weights.
    "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-8), np.log(10.0)),

    # L2 regularization on leaf weights.
    "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-8), np.log(10.0)),
}

# Fixed XGBoost params that are NOT tuned during HPO:
xgboost_fixed_params = {
    # Silence XGBoost's internal logging.
    "verbosity": 0,
}


# =====================================================================
# Helper: inspect logs after a run
# =====================================================================

def inspect_logs(result, run_config):
    """Print the contents of every artifact written during the run.

    The framework writes all artifacts into a structured folder:

        <output_dir>/<dataset_id>/run_YYYYMMDD_HHMMSS/
            config.yaml            -- full experiment configuration
            schema.json            -- inferred column types
            validation_report.json -- data quality checks
            summary_metrics.json   -- mean/std metrics across folds
            hpo/
                embedding_hpo_trials.jsonl  -- one line per embedding HPO trial
                classical_hpo_trials.jsonl  -- one line per classical HPO trial
                best_embedding_config.json  -- best embedding hyperparameters
                best_classical_config.json  -- best classical hyperparameters
            folds/
                fold_0/
                    metrics.json       -- per-fold test metrics
                    embedding_ckpt.pt  -- saved embedding model weights
                    classical_model.pkl -- saved classical model (joblib)
                    predictions.csv    -- per-sample predictions (if save_predictions)
                    embeddings.npy     -- embedding vectors (if save_embeddings)
                fold_1/
                    ...
    """
    # Locate the run directory.
    # The logger creates: <output_dir>/<dataset_id>/run_<timestamp>/
    ds_dir = os.path.join(run_config.output_dir, run_config.dataset_id)
    run_dirs = sorted([
        d for d in os.listdir(ds_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(ds_dir, d))
    ])
    run_dir = os.path.join(ds_dir, run_dirs[-1])
    print(f"\n--- Artifact directory: {run_dir} ---\n")

    # 1. config.yaml -- the full configuration used for this run
    config_path = os.path.join(run_dir, "config.yaml")
    print("[config.yaml] Full experiment config:")
    with open(config_path) as f:
        print(f.read())

    # 2. schema.json -- which columns were detected as numeric vs categorical
    schema_path = os.path.join(run_dir, "schema.json")
    print("[schema.json] Inferred column types:")
    with open(schema_path) as f:
        schema = json.load(f)
    print(f"  Numeric columns:     {schema['numeric_columns']}")
    print(f"  Categorical columns: {schema['categorical_columns']}")
    print(f"  Dropped columns:     {schema['dropped_columns']}")

    # 3. validation_report.json -- data quality checks
    report_path = os.path.join(run_dir, "validation_report.json")
    print("\n[validation_report.json] Data validation:")
    with open(report_path) as f:
        report = json.load(f)
    print(f"  {json.dumps(report, indent=2)}")

    # 4. HPO trial logs (JSONL -- one JSON object per line per trial)
    hpo_dir = os.path.join(run_dir, "hpo")

    embed_trials_path = os.path.join(hpo_dir, "embedding_hpo_trials.jsonl")
    print("\n[embedding_hpo_trials.jsonl] Embedding HPO trials:")
    with open(embed_trials_path) as f:
        for line in f:
            trial = json.loads(line)
            # val_metric is a dict like {"roc_auc": 0.85}
            print(f"  Trial {trial.get('trial_id', '?')}: "
                  f"val_metric={trial.get('val_metric')}  "
                  f"loss={trial.get('loss')}  "
                  f"params={trial.get('params', {})}")

    classical_trials_path = os.path.join(hpo_dir, "classical_hpo_trials.jsonl")
    print("\n[classical_hpo_trials.jsonl] Classical HPO trials:")
    with open(classical_trials_path) as f:
        for line in f:
            trial = json.loads(line)
            print(f"  Trial {trial.get('trial_id', '?')}: "
                  f"val_metric={trial.get('val_metric')}  "
                  f"loss={trial.get('loss')}")

    # 5. Best HP configs found
    best_embed_path = os.path.join(hpo_dir, "best_embedding_config.json")
    print("\n[best_embedding_config.json]:")
    with open(best_embed_path) as f:
        print(f"  {json.dumps(json.load(f), indent=2)}")

    best_classical_path = os.path.join(hpo_dir, "best_classical_config.json")
    print("\n[best_classical_config.json]:")
    with open(best_classical_path) as f:
        print(f"  {json.dumps(json.load(f), indent=2)}")

    # 6. summary_metrics.json -- aggregated across all folds
    summary_path = os.path.join(run_dir, "summary_metrics.json")
    print("\n[summary_metrics.json] Aggregated CV metrics:")
    with open(summary_path) as f:
        summary = json.load(f)
    print(f"  {json.dumps(summary, indent=2)}")

    # 7. Per-fold artifacts
    folds_dir = os.path.join(run_dir, "folds")
    for fold_name in sorted(os.listdir(folds_dir)):
        fold_dir = os.path.join(folds_dir, fold_name)
        if not os.path.isdir(fold_dir):
            continue
        print(f"\n[{fold_name}] contents: {os.listdir(fold_dir)}")

        metrics_path = os.path.join(fold_dir, "metrics.json")
        with open(metrics_path) as f:
            fold_metrics = json.load(f)
        print(f"  Metrics: {fold_metrics}")

        # Embeddings saved as .npy (numpy array)
        emb_path = os.path.join(fold_dir, "embeddings.npy")
        if os.path.isfile(emb_path):
            emb = np.load(emb_path)
            print(f"  Embeddings shape: {emb.shape}")

        # Predictions saved as .csv
        pred_path = os.path.join(fold_dir, "predictions.csv")
        if os.path.isfile(pred_path):
            preds = np.loadtxt(pred_path, delimiter=",")
            print(f"  Predictions shape: {preds.shape}, "
                  f"first 5: {preds[:5]}")

    # 8. RunResult object fields (returned in memory, not on disk)
    print("\n--- RunResult object ---")
    print(f"  task:           {result.task}")
    print(f"  dataset_id:     {result.dataset_id}")
    print(f"  n_folds:        {result.n_folds}")
    print(f"  total_time:     {result.total_time_seconds:.2f}s")
    print(f"  mean_metrics:   {result.mean_metrics}")
    print(f"  std_metrics:    {result.std_metrics}")
    print(f"  best_embedding_config: {result.best_embedding_config}")
    print(f"  best_classical_config: {result.best_classical_config}")
    for fr in result.fold_results:
        print(f"  Fold {fr.fold_index}: metrics={fr.metrics}, "
              f"embedding_dim={fr.embedding_dim}")


# =====================================================================
# Helper: OOS evaluation  (NN-only vs Hybrid)
# =====================================================================

def evaluate_oos_binary(model, X_oos, y_oos):
    """Evaluate both NN-only and hybrid models on OOS binary data."""
    print("\n--- Out-of-Sample Evaluation (Binary) ---")

    # Preprocess OOS data using the fitted preprocessor
    X_oos_prep = model._preprocessor.transform(X_oos)

    # NN-only: the embedding model has its own classification head
    nn_preds = model._embed_model.predict(X_oos_prep)
    nn_proba = model._embed_model.predict_proba(X_oos_prep)
    nn_proba_pos = nn_proba[:, 1]

    nn_auc = roc_auc_score(y_oos, nn_proba_pos)
    nn_acc = accuracy_score(y_oos, nn_preds)
    print(f"  NN-only   ROC-AUC: {nn_auc:.4f}   Accuracy: {nn_acc:.4f}")

    # Hybrid: TabTransformer + XGBoost
    hybrid_preds = model.predict(X_oos)
    hybrid_proba = model.predict_proba(X_oos)
    hybrid_proba_pos = hybrid_proba[:, 1]

    hybrid_auc = roc_auc_score(y_oos, hybrid_proba_pos)
    hybrid_acc = accuracy_score(y_oos, hybrid_preds)
    print(f"  Hybrid    ROC-AUC: {hybrid_auc:.4f}   Accuracy: {hybrid_acc:.4f}")

    delta_auc = hybrid_auc - nn_auc
    print(f"  Delta     ROC-AUC: {delta_auc:+.4f}   "
          f"({'hybrid wins' if delta_auc > 0 else 'NN wins'})")


def evaluate_oos_multiclass(model, X_oos, y_oos):
    """Evaluate both NN-only and hybrid models on OOS multiclass data."""
    print("\n--- Out-of-Sample Evaluation (Multiclass) ---")

    X_oos_prep = model._preprocessor.transform(X_oos)

    # NN-only
    nn_preds = model._embed_model.predict(X_oos_prep)
    nn_acc = accuracy_score(y_oos, nn_preds)
    nn_f1 = f1_score(y_oos, nn_preds, average="macro")
    print(f"  NN-only   Accuracy: {nn_acc:.4f}   Macro-F1: {nn_f1:.4f}")

    # Hybrid
    hybrid_preds = model.predict(X_oos)
    hybrid_acc = accuracy_score(y_oos, hybrid_preds)
    hybrid_f1 = f1_score(y_oos, hybrid_preds, average="macro")
    print(f"  Hybrid    Accuracy: {hybrid_acc:.4f}   Macro-F1: {hybrid_f1:.4f}")

    delta_acc = hybrid_acc - nn_acc
    print(f"  Delta     Accuracy: {delta_acc:+.4f}   "
          f"({'hybrid wins' if delta_acc > 0 else 'NN wins'})")


def evaluate_oos_regression(model, X_oos, y_oos):
    """Evaluate both NN-only and hybrid models on OOS regression data."""
    print("\n--- Out-of-Sample Evaluation (Regression) ---")

    X_oos_prep = model._preprocessor.transform(X_oos)

    # NN-only: predictions are in scaled target space, so we
    # inverse-transform them back to the original scale.
    nn_preds_scaled = model._embed_model.predict(X_oos_prep)
    nn_preds = model._preprocessor.inverse_transform_target(nn_preds_scaled)

    nn_rmse = float(np.sqrt(mean_squared_error(y_oos, nn_preds)))
    nn_mae = float(mean_absolute_error(y_oos, nn_preds))
    nn_r2 = float(r2_score(y_oos, nn_preds))
    print(f"  NN-only   RMSE: {nn_rmse:.4f}   MAE: {nn_mae:.4f}   "
          f"R2: {nn_r2:.4f}")

    # Hybrid: predict() returns predictions in the original target scale.
    hybrid_preds = model.predict(X_oos)

    hybrid_rmse = float(np.sqrt(mean_squared_error(y_oos, hybrid_preds)))
    hybrid_mae = float(mean_absolute_error(y_oos, hybrid_preds))
    hybrid_r2 = float(r2_score(y_oos, hybrid_preds))
    print(f"  Hybrid    RMSE: {hybrid_rmse:.4f}   MAE: {hybrid_mae:.4f}   "
          f"R2: {hybrid_r2:.4f}")

    delta_r2 = hybrid_r2 - nn_r2
    print(f"  Delta     R2: {delta_r2:+.4f}   "
          f"({'hybrid wins' if delta_r2 > 0 else 'NN wins'})")


# =====================================================================
# Shared config builders
# =====================================================================

def build_embedding_step():
    """Build the EmbeddingStepConfig for TabTransformer."""
    return EmbeddingStepConfig(
        # "tab_transformer" = pytorch-tabular TabTransformer (Mode B).
        model_type="tab_transformer",

        search_space=tab_transformer_search_space,
        fixed_params=tab_transformer_fixed_params,
    )


def build_classical_step(task="binary"):
    """Build the ClassicalStepConfig for XGBoost."""
    return ClassicalStepConfig(
        model_type="xgboost",
        model_class=None,
        search_space=xgboost_search_space,
        fixed_params=xgboost_fixed_params,
        supports_early_stopping=True,
    )


def build_budget():
    """Build the BudgetConfig with all parameters explicit."""
    return BudgetConfig(
        # Max Hyperopt trials for the embedding model HPO.
        embed_hpo_max_trials=5,

        # Max training epochs per embedding HPO trial.
        # Early stopping may terminate sooner.
        embed_max_epochs=10,

        # Early stopping patience: stop if validation metric does not
        # improve for this many consecutive epochs.
        embed_patience=3,

        # Wall-clock time limit (seconds) for embedding HPO.
        # None = no limit; set to e.g. 300.0 for a 5-minute cap.
        embed_time_budget_seconds=None,

        # Max Hyperopt trials for the classical model HPO.
        classical_hpo_max_trials=10,

        # Wall-clock time limit (seconds) for classical HPO.
        # None = no limit.
        classical_time_budget_seconds=None,
    )


def build_run_config(output_dir, dataset_id, **overrides):
    """Build the RunConfig with all parameters explicit."""
    defaults = dict(
        # Number of cross-validation folds.
        # HPO runs on fold 0; best config is then evaluated across all folds.
        n_folds=2,

        # Fraction of training data used for validation in single-split
        # mode (HybridTabularModel). Ignored in K-fold mode.
        val_fraction=0.2,

        # Master random seed. All per-fold and per-trial seeds are
        # derived from this, so the same value = reproducible results.
        master_seed=42,

        # Device for PyTorch training.
        # "auto" = best available (CUDA > MPS > CPU).
        # "cpu" forces CPU; "cuda" or "mps" for specific accelerators.
        device="cpu",

        # Enable PyTorch deterministic mode for bit-for-bit reproducibility.
        # Slightly slower but guarantees identical results across runs.
        deterministic=True,

        # Save per-fold embedding vectors (numpy .npy files) to disk.
        # Can be large; useful for post-hoc analysis of learned representations.
        save_embeddings=True,

        # Save per-fold predictions to disk as .csv files.
        # Useful for error analysis, confusion matrices, etc.
        save_predictions=True,

        # Root directory for all experiment artifacts.
        # Each experiment creates: <output_dir>/<dataset_id>/run_<timestamp>/
        output_dir=output_dir,

        # Human-readable label for this dataset, used in folder naming.
        dataset_id=dataset_id,

        # Scaler for numeric features before feeding to the embedding model.
        # "standard" = zero mean, unit variance (StandardScaler).
        # "robust" = median-centered, IQR-scaled (better with outliers).
        scaler="standard",

        # Maximum number of unique values a categorical column can have.
        # Columns exceeding this limit get rare values collapsed into __OOV__.
        # Prevents huge embedding lookup tables for high-cardinality columns.
        max_categorical_cardinality=100,

        # Add a binary indicator column per feature flagging whether the
        # original value was missing (before imputation).
        # Useful when missingness itself carries signal.
        add_missing_indicator=False,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


# =====================================================================
# Example 1: Binary classification
# =====================================================================

def run_binary_example():
    print("=" * 70)
    print("EXAMPLE 1: Binary classification -- TabTransformer + XGBoost")
    print("=" * 70)

    df, y = make_binary_data(n=1000, seed=42)

    # Split 50/50: framework data vs OOS hold-out
    df_train, df_oos, y_train, y_oos = train_test_split(
        df, y, test_size=0.5, random_state=42, stratify=y,
    )
    print(f"Framework data: {len(df_train)} rows")
    print(f"OOS hold-out:   {len(df_oos)} rows")
    print(f"Target distribution (train): {np.bincount(y_train)}")

    # -- TaskConfig --
    task_config = TaskConfig(
        # "binary" for binary classification. Also: "multiclass", "regression"
        task="binary",

        # Name of the target column when target is inside the DataFrame.
        # Set to None when y is passed separately (as we do here).
        target_column=None,

        # Which label is the positive class (for ROC-AUC computation).
        # None = auto-detect (uses the higher-valued label).
        # Set explicitly to 1 to be clear.
        positive_class=1,
    )

    embedding_step = build_embedding_step()
    classical_step = build_classical_step(task="binary")
    budget_config = build_budget()
    run_config = build_run_config("runs", "binary_example")

    # -----------------------------------------------------------------
    # Part A: Full CV pipeline (run_experiment) on framework data
    # -----------------------------------------------------------------
    print("\n--- Part A: run_experiment (K-fold CV) ---")
    result = run_experiment(
        X=df_train,
        y=y_train,
        task_config=task_config,
        embedding_step=embedding_step,
        classical_step=classical_step,
        budget_config=budget_config,
        run_config=run_config,
    )

    print(f"\nCV ROC-AUC: {result.mean_metrics['roc_auc']:.4f} "
          f"+/- {result.std_metrics['roc_auc']:.4f}")
    print(f"Total time: {result.total_time_seconds:.1f}s")

    # Inspect all saved artifacts
    inspect_logs(result, run_config)

    # -----------------------------------------------------------------
    # Part B: HybridTabularModel -- fit on framework data, evaluate OOS
    # -----------------------------------------------------------------
    print("\n--- Part B: HybridTabularModel (single split + OOS) ---")
    model = HybridTabularModel(
        task="binary",
        embedding=embedding_step,
        classical=classical_step,
        budget_config=budget_config,
        run_config=build_run_config("runs", "binary_wrapper"),
    )
    model.fit(df_train, y_train)

    # OOS evaluation: NN-only vs Hybrid
    evaluate_oos_binary(model, df_oos, y_oos)

    return result, model


# =====================================================================
# Example 2: Multiclass classification
# =====================================================================

def run_multiclass_example():
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Multiclass classification (4 classes) -- TabTransformer + XGBoost")
    print("=" * 70)

    df, y = make_multiclass_data(n=1200, n_classes=4, seed=42)

    df_train, df_oos, y_train, y_oos = train_test_split(
        df, y, test_size=0.5, random_state=42, stratify=y,
    )
    print(f"Framework data: {len(df_train)} rows")
    print(f"OOS hold-out:   {len(df_oos)} rows")
    print(f"Classes: {np.unique(y_train)}, counts: {np.bincount(y_train)}")

    task_config = TaskConfig(
        task="multiclass",
        target_column=None,
        # positive_class is not used for multiclass; set to None.
        positive_class=None,
    )

    embedding_step = build_embedding_step()
    classical_step = build_classical_step(task="multiclass")
    budget_config = build_budget()
    run_config = build_run_config("runs", "multiclass_example",
                                  scaler="robust")

    # Part A: Full CV pipeline
    print("\n--- Part A: run_experiment (K-fold CV) ---")
    result = run_experiment(
        X=df_train, y=y_train,
        task_config=task_config,
        embedding_step=embedding_step,
        classical_step=classical_step,
        budget_config=budget_config,
        run_config=run_config,
    )

    print(f"\nCV Accuracy: {result.mean_metrics['accuracy']:.4f} "
          f"+/- {result.std_metrics['accuracy']:.4f}")
    print(f"Macro F1: {result.mean_metrics['macro_f1']:.4f} "
          f"+/- {result.std_metrics['macro_f1']:.4f}")
    print(f"Total time: {result.total_time_seconds:.1f}s")

    inspect_logs(result, run_config)

    # Part B: HybridTabularModel -- fit + OOS
    print("\n--- Part B: HybridTabularModel (single split + OOS) ---")
    model = HybridTabularModel(
        task="multiclass",
        embedding=embedding_step,
        classical=classical_step,
        budget_config=budget_config,
        run_config=build_run_config("runs", "multiclass_wrapper",
                                    scaler="robust"),
    )
    model.fit(df_train, y_train)

    evaluate_oos_multiclass(model, df_oos, y_oos)

    return result, model


# =====================================================================
# Example 3: Regression
# =====================================================================

def run_regression_example():
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Regression -- TabTransformer + XGBoost")
    print("=" * 70)

    df, y = make_regression_data(n=1000, seed=42)

    df_train, df_oos, y_train, y_oos = train_test_split(
        df, y, test_size=0.5, random_state=42,
    )
    print(f"Framework data: {len(df_train)} rows")
    print(f"OOS hold-out:   {len(df_oos)} rows")
    print(f"Target range (train): [{y_train.min():.2f}, {y_train.max():.2f}]")

    task_config = TaskConfig(
        task="regression",
        target_column=None,
        # positive_class is not used for regression; set to None.
        positive_class=None,
    )

    embedding_step = build_embedding_step()
    classical_step = build_classical_step(task="regression")
    budget_config = build_budget()
    run_config = build_run_config("runs", "regression_example",
                                  add_missing_indicator=True)

    # Part A: Full CV pipeline
    print("\n--- Part A: run_experiment (K-fold CV) ---")
    result = run_experiment(
        X=df_train, y=y_train,
        task_config=task_config,
        embedding_step=embedding_step,
        classical_step=classical_step,
        budget_config=budget_config,
        run_config=run_config,
    )

    print(f"\nCV RMSE: {result.mean_metrics['rmse']:.4f} "
          f"+/- {result.std_metrics['rmse']:.4f}")
    print(f"MAE: {result.mean_metrics['mae']:.4f} "
          f"+/- {result.std_metrics['mae']:.4f}")
    print(f"R-squared: {result.mean_metrics['r_squared']:.4f} "
          f"+/- {result.std_metrics['r_squared']:.4f}")
    print(f"Total time: {result.total_time_seconds:.1f}s")

    inspect_logs(result, run_config)

    # Part B: HybridTabularModel -- fit + OOS
    print("\n--- Part B: HybridTabularModel (single split + OOS) ---")
    model = HybridTabularModel(
        task="regression",
        embedding=embedding_step,
        classical=classical_step,
        budget_config=budget_config,
        run_config=build_run_config("runs", "regression_wrapper",
                                    add_missing_indicator=True),
    )
    model.fit(df_train, y_train)

    evaluate_oos_regression(model, df_oos, y_oos)

    return result, model


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    run_binary_example()
    run_multiclass_example()
    run_regression_example()
