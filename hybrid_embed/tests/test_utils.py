"""Tests for hybrid_embed.utils: seed management, device detection, timing."""

import time

import numpy as np
import torch

from hybrid_embed.utils import seed_everything, derive_seed, get_device, Timer


def test_seed_everything_reproducibility():
    """Verify that seeding produces identical random outputs."""
    seed_everything(42)
    t1 = torch.randn(5)
    a1 = np.random.rand(5)
    print(f"Seed 42 - torch: {t1.tolist()}")
    print(f"Seed 42 - numpy: {a1.tolist()}")

    seed_everything(42)
    t2 = torch.randn(5)
    a2 = np.random.rand(5)
    print(f"Seed 42 (repeat) - torch: {t2.tolist()}")
    print(f"Seed 42 (repeat) - numpy: {a2.tolist()}")

    assert torch.equal(t1, t2), "Torch tensors should be identical with same seed"
    assert np.array_equal(a1, a2), "Numpy arrays should be identical with same seed"

    seed_everything(99)
    t3 = torch.randn(5)
    a3 = np.random.rand(5)
    print(f"Seed 99 - torch: {t3.tolist()}")
    print(f"Seed 99 - numpy: {a3.tolist()}")

    assert not torch.equal(t1, t3), "Different seeds should produce different torch outputs"
    assert not np.array_equal(a1, a3), "Different seeds should produce different numpy outputs"
    print("Test passed: seed_everything is reproducible")


def test_derive_seed_deterministic():
    """Verify derive_seed is deterministic and offset-sensitive."""
    s1 = derive_seed(42, 0)
    s2 = derive_seed(42, 0)
    print(f"derive_seed(42, 0) = {s1}")
    print(f"derive_seed(42, 0) = {s2}")
    assert s1 == s2, "Same inputs should give same output"

    s3 = derive_seed(42, 1)
    s4 = derive_seed(42, 2)
    print(f"derive_seed(42, 1) = {s3}")
    print(f"derive_seed(42, 2) = {s4}")
    assert s3 != s4, "Different offsets should give different outputs"

    s5 = derive_seed(100, 5)
    print(f"derive_seed(100, 5) = {s5}")
    assert 0 <= s5 < 2**31, "Derived seed should be in [0, 2**31)"
    print("Test passed: derive_seed is deterministic")


def test_get_device_cpu():
    """Verify that requesting CPU returns a CPU device."""
    device = get_device("cpu")
    print(f"get_device('cpu') = {device}")
    assert device == torch.device("cpu")
    print("Test passed: get_device('cpu') returns cpu")


def test_get_device_auto():
    """Verify that auto-detection returns a valid device without error."""
    device = get_device("auto")
    print(f"get_device('auto') = {device}")
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda", "mps")
    print("Test passed: get_device('auto') returns a valid device")


def test_timer():
    """Verify Timer measures elapsed time correctly."""
    with Timer() as t:
        time.sleep(0.1)
    print(f"Timer elapsed: {t.elapsed_seconds:.4f}s")
    assert t.elapsed_seconds >= 0.1, "Should measure at least 0.1s"
    assert t.elapsed_seconds < 0.5, "Should not take more than 0.5s"
    print("Test passed: Timer measures wall-clock time correctly")
