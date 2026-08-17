"""Tests for the Phase 0.5 synthetic instance generator."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import beta as scipy_beta
from scipy.stats import chisquare

from instances.synthetic_generator import (
    DEFAULT_CONFIG,
    SyntheticInstance,
    check_seed_non_overlap,
    generate_instance,
)


def test_determinism():
    a = generate_instance(seed=42)
    b = generate_instance(seed=42)
    assert a == b


def test_shapes_default_config():
    inst = generate_instance(seed=0)
    assert inst.initial_positions.shape == (10, 2)
    assert len(inst.tasks) == 60
    required = {"id", "release_epoch", "pickup", "drop", "duration"}
    for t in inst.tasks:
        assert required.issubset(t.keys())
        assert isinstance(t["pickup"], np.ndarray) and t["pickup"].shape == (2,)
        assert isinstance(t["drop"], np.ndarray) and t["drop"].shape == (2,)


def test_bounds():
    inst = generate_instance(seed=1)
    W = DEFAULT_CONFIG["warehouse_size"]
    assert np.all(inst.initial_positions >= 0.0)
    assert np.all(inst.initial_positions <= W)
    H = DEFAULT_CONFIG["H"]
    for t in inst.tasks:
        assert np.all(t["pickup"] >= 0.0) and np.all(t["pickup"] <= W)
        assert np.all(t["drop"] >= 0.0) and np.all(t["drop"] <= W)
        assert DEFAULT_CONFIG["d_min"] <= t["duration"] <= DEFAULT_CONFIG["d_max"]
        assert isinstance(t["release_epoch"], int)
        assert 0 <= t["release_epoch"] <= H - 1


def test_task_ids_sequential():
    inst = generate_instance(seed=2)
    assert [t["id"] for t in inst.tasks] == list(range(60))


def test_cluster_structure():
    inst = generate_instance(seed=7, config={"T": 5000})
    sigma_pick = DEFAULT_CONFIG["sigma_pick"]
    pickups = np.stack([t["pickup"] for t in inst.tasks])  # (5000, 2)
    for k in range(DEFAULT_CONFIG["K_pick"]):
        mask = inst.pickup_cluster_indices == k
        if mask.sum() < 2:
            continue
        cluster_pickups = pickups[mask]
        var_xy = cluster_pickups.var(axis=0)
        assert np.all(var_xy <= 3.0 * sigma_pick ** 2), (
            f"cluster {k} var {var_xy} exceeds 3 * sigma_pick^2 = {3.0 * sigma_pick ** 2}"
        )


def test_release_distribution_chi_square():
    inst = generate_instance(seed=12345, config={"T": 5000})
    H = DEFAULT_CONFIG["H"]
    releases = np.array([t["release_epoch"] for t in inst.tasks])
    observed = np.bincount(releases, minlength=H)
    edges = np.arange(H + 1) / H
    cdf = scipy_beta.cdf(edges, 2.0, 2.0)
    probs = np.diff(cdf)
    expected = probs * 5000.0
    _, p = chisquare(observed, expected)
    assert p > 0.01, f"chi-square p-value {p} <= 0.01"


def test_seed_non_overlap():
    check_seed_non_overlap(range(900, 920), range(1000, 1050))
    with pytest.raises(ValueError, match="3"):
        check_seed_non_overlap([1, 2, 3], [3, 4, 5])


def test_config_override():
    inst = generate_instance(seed=0, config={"R": 5})
    assert inst.R == 5
    assert inst.T == 60
    assert inst.H == 20
    assert inst.config["R"] == 5
    assert inst.config["T"] == 60
    assert inst.config["warehouse_size"] == 50.0
