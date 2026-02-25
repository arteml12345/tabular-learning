Expected Usage Examples
======================

These files are API usage examples showing how the framework is
intended to be used. They are NOT runnable code yet -- they will
become runnable once the framework is implemented.

Each file demonstrates one specific combination of:
- Split strategy: single train/test split vs K-fold CV
- Embedding model: from-scratch MLP (Mode A) vs pytorch-tabular (Mode B)
- Classical model: built-in with defaults, built-in with custom HP, custom class
- Task type: binary classification, multiclass, regression

Naming convention:
  {split}_{embedding}_{classical}_{task}.py

Files:
  Mode A (from-scratch MLP):
    01_cv_mlp_xgboost_binary.py               - simplest full example
    02_cv_mlp_xgboost_regression.py            - regression variant
    03_cv_mlp_xgboost_multiclass.py            - multiclass variant
    04_cv_mlp_custom_hp_xgboost_binary.py      - custom MLP search space
    05_cv_mlp_xgboost_custom_hp_binary.py      - custom XGBoost search space
    06_cv_mlp_ridge_regression.py              - sklearn classical model
    07_cv_mlp_custom_class_binary.py           - user-provided classical class
    08_single_mlp_xgboost_binary.py            - single train/test split

  Mode B (pytorch-tabular):
    09_cv_tabtransformer_xgboost_binary.py     - TabTransformer default
    10_cv_tabtransformer_custom_hp_xgb_binary.py - custom TabTransformer HP
    11_cv_tabnet_ridge_regression.py           - TabNet + Ridge
    12_cv_category_embedding_custom_class_multiclass.py
