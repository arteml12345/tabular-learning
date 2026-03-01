"""Utility functions for seed management, device detection, and timing.

Provides reproducibility helpers, automatic device selection for
PyTorch, and a simple wall-clock timer context manager.
"""

import random
import time
import warnings

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set random seed for all sources of randomness.

    Sets seeds for Python's ``random`` module, NumPy, and PyTorch
    (CPU and CUDA). Call this at the start of every experiment to
    ensure reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to use across all random generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_seed(master_seed: int, offset: int) -> int:
    """Derive a deterministic sub-seed from a master seed and offset.

    Used to generate per-fold and per-trial seeds from a single
    master seed, ensuring different components get different but
    reproducible random states.

    Parameters
    ----------
    master_seed : int
        The base seed for the experiment.
    offset : int
        An integer offset (e.g., fold index or trial id).

    Returns
    -------
    int
        A derived seed in the range [0, 2**31).
    """
    return (master_seed + offset) % (2**31)


def get_device(preference: str = "auto") -> torch.device:
    """Select the best available PyTorch device.

    When ``preference`` is ``"auto"``, selects the best device in
    priority order: CUDA > MPS > CPU. If a specific device is
    requested but unavailable, falls back to CPU with a warning.

    Parameters
    ----------
    preference : str, default "auto"
        One of ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Returns
    -------
    torch.device
        The selected device.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn("CUDA requested but not available, falling back to CPU.")
        return torch.device("cpu")

    if preference == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        warnings.warn("MPS requested but not available, falling back to CPU.")
        return torch.device("cpu")

    return torch.device("cpu")


class Timer:
    """Context manager for wall-clock timing.

    Records the elapsed wall-clock time between entry and exit.

    Attributes
    ----------
    elapsed_seconds : float
        Wall-clock seconds elapsed. Set when the context exits.

    Examples
    --------
    >>> with Timer() as t:
    ...     time.sleep(0.1)
    >>> print(t.elapsed_seconds)  # ~0.1
    """

    def __init__(self):
        self.elapsed_seconds: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start
