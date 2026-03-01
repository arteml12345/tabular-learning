"""Local artifact logging for experiment runs.

Manages the folder structure and file I/O for a single experiment
run: config, schema, validation report, HPO trial logs, per-fold
artifacts, and summary metrics.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime

import joblib
import numpy as np
import yaml

from hybrid_embed.config import Schema


class LocalLogger:
    """Manages the artifact folder for a single experiment run.

    Creates the folder structure on initialization::

        <output_dir>/<dataset_id>/run_YYYYMMDD_HHMMSS/
            hpo/
            folds/

    Parameters
    ----------
    output_dir : str
        Root directory for all experiment runs.
    dataset_id : str
        Label identifying the dataset (used in folder naming).
    """

    def __init__(self, output_dir: str, dataset_id: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_dir = os.path.join(output_dir, dataset_id, f"run_{timestamp}")
        os.makedirs(self._run_dir, exist_ok=True)
        os.makedirs(self.get_hpo_dir(), exist_ok=True)
        os.makedirs(os.path.join(self._run_dir, "folds"), exist_ok=True)

    @property
    def run_dir(self) -> str:
        """Return the full path to this run's directory.

        Returns
        -------
        str
        """
        return self._run_dir

    def save_config(self, config: dict) -> None:
        """Save experiment configuration to ``config.yaml``.

        Parameters
        ----------
        config : dict
            Full run configuration (JSON-serializable).
        """
        path = os.path.join(self._run_dir, "config.yaml")
        with open(path, "w") as fh:
            yaml.dump(config, fh, default_flow_style=False, sort_keys=False)

    def save_schema(self, schema: Schema) -> None:
        """Save inferred schema to ``schema.json``.

        Parameters
        ----------
        schema : Schema
            Inferred column type information.
        """
        path = os.path.join(self._run_dir, "schema.json")
        with open(path, "w") as fh:
            json.dump(asdict(schema), fh, indent=2)

    def save_validation_report(self, report) -> None:
        """Save data validation report to ``validation_report.json``.

        Parameters
        ----------
        report : ValidationReport
            Results from data validation.
        """
        path = os.path.join(self._run_dir, "validation_report.json")
        with open(path, "w") as fh:
            json.dump(asdict(report), fh, indent=2)

    def log_embedding_trial(self, trial_record: dict) -> None:
        """Append one embedding HPO trial record to JSONL log.

        Parameters
        ----------
        trial_record : dict
            Trial metadata (trial_id, params, val_metric, etc.).
        """
        path = os.path.join(self.get_hpo_dir(), "embedding_hpo_trials.jsonl")
        with open(path, "a") as fh:
            fh.write(json.dumps(trial_record) + "\n")

    def log_classical_trial(self, trial_record: dict) -> None:
        """Append one classical HPO trial record to JSONL log.

        Parameters
        ----------
        trial_record : dict
            Trial metadata (trial_id, params, val_metric, etc.).
        """
        path = os.path.join(self.get_hpo_dir(), "classical_hpo_trials.jsonl")
        with open(path, "a") as fh:
            fh.write(json.dumps(trial_record) + "\n")

    def save_best_embedding_config(self, config: dict) -> None:
        """Save best embedding HP configuration to JSON.

        Parameters
        ----------
        config : dict
            Best hyperparameter configuration found during HPO.
        """
        path = os.path.join(self.get_hpo_dir(), "best_embedding_config.json")
        with open(path, "w") as fh:
            json.dump(config, fh, indent=2)

    def save_best_classical_config(self, config: dict) -> None:
        """Save best classical HP configuration to JSON.

        Parameters
        ----------
        config : dict
            Best hyperparameter configuration found during HPO.
        """
        path = os.path.join(self.get_hpo_dir(), "best_classical_config.json")
        with open(path, "w") as fh:
            json.dump(config, fh, indent=2)

    def save_fold_artifacts(
        self,
        fold_index: int,
        metrics: dict,
        embedding_model=None,
        classical_model=None,
        predictions: np.ndarray | None = None,
        embeddings: np.ndarray | None = None,
    ) -> None:
        """Save all artifacts for a single fold.

        Creates ``folds/fold_{fold_index}/`` and writes metrics,
        optionally saves model checkpoints, predictions, and
        embeddings.

        Parameters
        ----------
        fold_index : int
            Fold number (0-based).
        metrics : dict
            Metric name -> value for this fold.
        embedding_model : object or None
            If provided, must have a ``.save(path)`` method.
        classical_model : object or None
            If provided, saved with ``joblib.dump``.
        predictions : np.ndarray or None
            Per-sample predictions; saved as ``predictions.csv``.
        embeddings : np.ndarray or None
            Embedding vectors; saved as ``embeddings.npy``.
        """
        fold_dir = self.get_fold_dir(fold_index)
        os.makedirs(fold_dir, exist_ok=True)

        with open(os.path.join(fold_dir, "metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)

        if embedding_model is not None:
            embedding_model.save(os.path.join(fold_dir, "embedding_ckpt.pt"))

        if classical_model is not None:
            joblib.dump(classical_model, os.path.join(fold_dir, "classical_model.pkl"))

        if predictions is not None:
            np.savetxt(
                os.path.join(fold_dir, "predictions.csv"),
                predictions, delimiter=",",
            )

        if embeddings is not None:
            np.save(os.path.join(fold_dir, "embeddings.npy"), embeddings)

    def save_summary_metrics(
        self,
        mean_metrics: dict,
        std_metrics: dict,
    ) -> None:
        """Save aggregated CV metrics to ``summary_metrics.json``.

        Parameters
        ----------
        mean_metrics : dict
            Mean of each metric across folds.
        std_metrics : dict
            Standard deviation of each metric across folds.
        """
        path = os.path.join(self._run_dir, "summary_metrics.json")
        summary = {"mean": mean_metrics, "std": std_metrics}
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=2)

    def get_fold_dir(self, fold_index: int) -> str:
        """Return the path to a fold's artifact directory.

        Parameters
        ----------
        fold_index : int
            Fold number (0-based).

        Returns
        -------
        str
        """
        return os.path.join(self._run_dir, "folds", f"fold_{fold_index}")

    def get_hpo_dir(self) -> str:
        """Return the path to the HPO subdirectory.

        Returns
        -------
        str
        """
        return os.path.join(self._run_dir, "hpo")
