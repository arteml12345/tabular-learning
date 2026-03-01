"""Tests for hybrid_embed.log.local_logger: artifact folder management."""

import json
import os
import tempfile

import yaml

from hybrid_embed.config import Schema
from hybrid_embed.log.local_logger import LocalLogger


def test_logger_creates_directory_structure():
    """Initializing the logger should create run_dir with hpo/ and folds/."""
    tmpdir = tempfile.mkdtemp()
    logger = LocalLogger(output_dir=tmpdir, dataset_id="test_ds")
    print(f"Run dir: {logger.run_dir}")

    assert os.path.isdir(logger.run_dir)
    assert os.path.isdir(logger.get_hpo_dir())
    assert os.path.isdir(os.path.join(logger.run_dir, "folds"))
    assert "test_ds" in logger.run_dir
    assert "run_" in os.path.basename(logger.run_dir)
    print("Test passed: directory structure created correctly")


def test_save_config():
    """save_config should write a readable YAML file."""
    tmpdir = tempfile.mkdtemp()
    logger = LocalLogger(output_dir=tmpdir, dataset_id="test_ds")

    config = {"n_folds": 5, "seed": 42, "model": "mlp"}
    logger.save_config(config)

    path = os.path.join(logger.run_dir, "config.yaml")
    assert os.path.exists(path)

    with open(path, "r") as fh:
        loaded = yaml.safe_load(fh)
    print(f"Loaded config: {loaded}")

    assert loaded["n_folds"] == 5
    assert loaded["seed"] == 42
    assert loaded["model"] == "mlp"
    print("Test passed: config saved and loaded correctly")


def test_log_embedding_trial():
    """Logging 3 embedding trials should produce 3 JSONL lines."""
    tmpdir = tempfile.mkdtemp()
    logger = LocalLogger(output_dir=tmpdir, dataset_id="test_ds")

    for i in range(3):
        logger.log_embedding_trial({
            "trial_id": i,
            "params": {"lr": 0.001 * (i + 1)},
            "val_metric": 0.8 + i * 0.01,
        })

    path = os.path.join(logger.get_hpo_dir(), "embedding_hpo_trials.jsonl")
    assert os.path.exists(path)

    with open(path, "r") as fh:
        lines = fh.readlines()
    print(f"Number of JSONL lines: {len(lines)}")

    assert len(lines) == 3
    for i, line in enumerate(lines):
        record = json.loads(line)
        assert record["trial_id"] == i
        print(f"Trial {i}: {record}")
    print("Test passed: 3 embedding trials logged correctly")


def test_log_classical_trial():
    """Logging 2 classical trials should produce 2 JSONL lines."""
    tmpdir = tempfile.mkdtemp()
    logger = LocalLogger(output_dir=tmpdir, dataset_id="test_ds")

    for i in range(2):
        logger.log_classical_trial({
            "trial_id": i,
            "params": {"max_depth": 3 + i},
            "val_metric": 0.85 + i * 0.02,
        })

    path = os.path.join(logger.get_hpo_dir(), "classical_hpo_trials.jsonl")
    with open(path, "r") as fh:
        lines = fh.readlines()
    print(f"Number of JSONL lines: {len(lines)}")

    assert len(lines) == 2
    for i, line in enumerate(lines):
        record = json.loads(line)
        assert record["trial_id"] == i
    print("Test passed: 2 classical trials logged correctly")


def test_save_fold_artifacts_metrics():
    """save_fold_artifacts should write metrics.json for a fold."""
    tmpdir = tempfile.mkdtemp()
    logger = LocalLogger(output_dir=tmpdir, dataset_id="test_ds")

    metrics = {"roc_auc": 0.92, "log_loss": 0.28}
    logger.save_fold_artifacts(fold_index=0, metrics=metrics)

    fold_dir = logger.get_fold_dir(0)
    assert os.path.isdir(fold_dir)
    print(f"Fold dir: {fold_dir}")

    metrics_path = os.path.join(fold_dir, "metrics.json")
    assert os.path.exists(metrics_path)

    with open(metrics_path, "r") as fh:
        loaded = json.load(fh)
    print(f"Loaded metrics: {loaded}")

    assert loaded["roc_auc"] == 0.92
    assert loaded["log_loss"] == 0.28
    print("Test passed: fold metrics saved and loaded correctly")


def test_save_summary_metrics():
    """save_summary_metrics should write a valid JSON with mean and std."""
    tmpdir = tempfile.mkdtemp()
    logger = LocalLogger(output_dir=tmpdir, dataset_id="test_ds")

    mean = {"roc_auc": 0.90, "log_loss": 0.30}
    std = {"roc_auc": 0.02, "log_loss": 0.03}
    logger.save_summary_metrics(mean, std)

    path = os.path.join(logger.run_dir, "summary_metrics.json")
    assert os.path.exists(path)

    with open(path, "r") as fh:
        loaded = json.load(fh)
    print(f"Loaded summary: {loaded}")

    assert loaded["mean"]["roc_auc"] == 0.90
    assert loaded["std"]["roc_auc"] == 0.02
    assert loaded["mean"]["log_loss"] == 0.30
    assert loaded["std"]["log_loss"] == 0.03
    print("Test passed: summary metrics saved correctly")


def test_multiple_folds():
    """Saving artifacts for folds 0-4 should create all fold directories."""
    tmpdir = tempfile.mkdtemp()
    logger = LocalLogger(output_dir=tmpdir, dataset_id="test_ds")

    for i in range(5):
        metrics = {"roc_auc": 0.88 + i * 0.01}
        logger.save_fold_artifacts(fold_index=i, metrics=metrics)

    for i in range(5):
        fold_dir = logger.get_fold_dir(i)
        assert os.path.isdir(fold_dir), f"Fold {i} directory missing"
        metrics_path = os.path.join(fold_dir, "metrics.json")
        assert os.path.exists(metrics_path), f"Fold {i} metrics.json missing"
        print(f"Fold {i}: directory and metrics.json exist")

    print("Test passed: all 5 fold directories created with metrics")
