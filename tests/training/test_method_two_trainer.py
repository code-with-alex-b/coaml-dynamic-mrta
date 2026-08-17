"""Smoke tests for the method-two REINFORCE trainer on tiny configs."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from instances.synthetic_generator import generate_instance
from scoring.gnn_scorer import GNNScorer
from training.expert_dataset_generator import _serialize_instance
import training.method_two_trainer as m2
from training.method_two_trainer import (
    BASELINE_GRPO,
    BASELINE_RLOO,
    DEFAULT_WEIGHTS,
    InstanceDataset,
    MethodTwoConfig,
    RolloutResult,
    annealed_epsilon,
    compute_greedy_baseline,
    grpo_advantages,
    reinforce_gradient_step,
    rloo_advantages,
    rollout_one_instance,
    train,
    update_baseline,
)


SMALL_CONFIG = {"R": 3, "T": 5, "H": 5}


def _write_cache(cache_dir, seeds, config=SMALL_CONFIG):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        inst = generate_instance(seed, config=config)
        rec = {"seed": int(seed), "instance": _serialize_instance(inst)}
        with (cache_dir / f"seed{seed}.json").open("w") as f:
            json.dump(rec, f)


def test_instance_dataset_loads_expected_count(tmp_path):
    cache = tmp_path / "train"
    _write_cache(cache, [80000, 80001, 80002])
    ds = InstanceDataset(cache)
    assert len(ds) == 3
    inst = ds.get(0)
    assert inst.R == 3 and inst.T == 5


def test_rollout_one_instance_records_trajectory():
    inst = generate_instance(80010, config=SMALL_CONFIG)
    torch.manual_seed(0)
    scorer = GNNScorer()
    result = rollout_one_instance(
        scorer, inst, DEFAULT_WEIGHTS, epsilon=1.0, K_sink=10
    )
    assert isinstance(result, RolloutResult)
    assert len(result.Theta_list) > 0
    assert len(result.Theta_list) == len(result.P_list) == len(result.y_hat_list)
    assert np.isfinite(result.combined_cost)
    N = inst.R + inst.T
    assert result.Theta_list[0].shape == (N, N)
    assert result.y_hat_list[0].shape == (N, N)


def test_reinforce_gradient_step_updates_parameters():
    inst = generate_instance(80020, config=SMALL_CONFIG)
    torch.manual_seed(0)
    scorer = GNNScorer()
    optimiser = torch.optim.Adam(scorer.parameters(), lr=1e-3)

    rollout = rollout_one_instance(
        scorer, inst, DEFAULT_WEIGHTS, epsilon=1.0, K_sink=10
    )
    before = [p.detach().clone() for p in scorer.parameters()]

    # Pass an explicit baseline so the advantage is non-zero (baseline=None would centre a single rollout to exactly zero -> no update).
    metrics = reinforce_gradient_step(
        scorer, optimiser, [rollout], baseline=0.0,
        epsilon=1.0, gradient_clip_norm=10.0,
    )
    assert np.isfinite(metrics["mean_cost"])
    assert np.isfinite(metrics["grad_norm"])
    assert metrics["grad_norm"] > 0.0

    after = list(scorer.parameters())
    changed = any(
        not torch.equal(b, a.detach()) for b, a in zip(before, after)
    )
    assert changed


def test_update_baseline_ema():
    # First call: None -> batch mean.
    b0 = update_baseline(None, [10.0, 20.0], alpha=0.05)
    assert b0 == pytest.approx(15.0)
    # Subsequent call: EMA smoothing.
    b1 = update_baseline(b0, [25.0], alpha=0.05)
    assert b1 == pytest.approx(0.95 * 15.0 + 0.05 * 25.0)
    # Larger alpha tracks the batch faster.
    b2 = update_baseline(10.0, [20.0], alpha=0.5)
    assert b2 == pytest.approx(15.0)


def test_train_runs_and_saves_checkpoint(tmp_path):
    cache = tmp_path / "train"
    _write_cache(cache, [80030, 80031, 80032])
    ckpt = tmp_path / "checkpoints" / "method_two.pt"
    config = MethodTwoConfig(
        cache_dir=str(cache),
        checkpoint_path=str(ckpt),
        batch_size=2,
        epsilon_initial=1.0,
        epsilon_terminal=1.0,  # constant epsilon for the smoke run
        K_sink=10,
        log_every_steps=1,
    )
    torch.manual_seed(0)
    scorer = train(config, num_steps=5)

    assert isinstance(scorer, GNNScorer)
    assert ckpt.exists()
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model_state_dict" in loaded
    assert np.isfinite(loaded["cost_history"]).all()


def test_train_end_to_end_checkpoint_loads_into_fresh_scorer(tmp_path):
    cache = tmp_path / "train"
    _write_cache(cache, [80040, 80041])
    ckpt = tmp_path / "checkpoints" / "method_two.pt"
    config = MethodTwoConfig(
        cache_dir=str(cache),
        checkpoint_path=str(ckpt),
        batch_size=2,
        epsilon_initial=1.0,
        epsilon_terminal=1.0,  # constant epsilon for the smoke run
        K_sink=10,
        log_every_steps=0,
    )
    torch.manual_seed(0)
    train(config, num_steps=3)

    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    fresh = GNNScorer()
    fresh.load_state_dict(loaded["model_state_dict"])  # must not raise
    fresh.eval()

    inst = generate_instance(80042, config=SMALL_CONFIG)
    from simulator.dynamic_simulator import DynamicSimulator

    sim = DynamicSimulator(inst)
    with torch.no_grad():
        theta = fresh(sim.state, inst)
    assert theta.shape == (inst.R + inst.T, inst.R + inst.T)


# Enhancement tests: per-instance greedy baseline and annealed epsilon.


def _scorer_and_rollouts(seed, instances, epsilon=1.0, K_sink=10):
    """Build an identically-initialised scorer and identical rollouts.

    Seeding before scorer construction makes both the parameters and the
    subsequent Gumbel draws reproducible, so two calls with the same seed yield
    the same Theta/P/y_hat values (on independent autograd graphs)."""
    torch.manual_seed(seed)
    scorer = GNNScorer()
    rollouts = [
        rollout_one_instance(scorer, inst, DEFAULT_WEIGHTS, epsilon, K_sink)
        for inst in instances
    ]
    return scorer, rollouts


def _grad_vector(scorer):
    return torch.cat(
        [
            p.grad.reshape(-1)
            for p in scorer.parameters()
            if p.grad is not None
        ]
    )


def test_per_instance_greedy_differs_from_running_mean():
    instA = generate_instance(80050, config=SMALL_CONFIG)
    instB = generate_instance(80051, config=SMALL_CONFIG)
    instances = [instA, instB]

    # Running-mean baseline (None -> batch mean).
    sA, rA = _scorer_and_rollouts(7, instances)
    optA = torch.optim.Adam(sA.parameters(), lr=1e-3)
    reinforce_gradient_step(
        sA, optA, rA, baseline=None, epsilon=1.0, gradient_clip_norm=10.0
    )
    gA = _grad_vector(sA)

    # Per-instance greedy baselines on the identical scorer + rollouts.
    sB, rB = _scorer_and_rollouts(7, instances)
    optB = torch.optim.Adam(sB.parameters(), lr=1e-3)
    greedy_baselines = [
        compute_greedy_baseline(instA, DEFAULT_WEIGHTS),
        compute_greedy_baseline(instB, DEFAULT_WEIGHTS),
    ]
    reinforce_gradient_step(
        sB, optB, rB, baseline=greedy_baselines,
        epsilon=1.0, gradient_clip_norm=10.0,
    )
    gB = _grad_vector(sB)

    # Same trajectories, different advantage weighting -> different gradient.
    assert gA.shape == gB.shape
    assert not torch.allclose(gA, gB)


def test_annealed_epsilon_log_linear_decay():
    num_steps = 100
    eps_i, eps_t = 1.0, 0.01
    vals = [
        annealed_epsilon(s, num_steps, eps_i, eps_t) for s in range(num_steps)
    ]
    assert vals[0] == pytest.approx(eps_i)
    assert vals[-1] == pytest.approx(eps_t)
    assert all(vals[k] > vals[k + 1] for k in range(num_steps - 1))
    log_diffs = np.diff(np.log(vals))
    assert np.allclose(log_diffs, log_diffs[0])
    # Explicit short anneal window holds at terminal afterwards.
    assert annealed_epsilon(80, 100, eps_i, eps_t, anneal_steps=50) == pytest.approx(eps_t)


# GRPO mode (Component 10, DEC-06).


def test_grpo_does_k_rollouts_per_instance(tmp_path, monkeypatch):
    cache = tmp_path / "train"
    _write_cache(cache, [80080, 80081, 80082])
    ckpt = tmp_path / "checkpoints" / "grpo.pt"

    calls = {"n": 0}
    original = m2.rollout_one_instance

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(m2, "rollout_one_instance", counting)

    config = MethodTwoConfig(
        cache_dir=str(cache),
        checkpoint_path=str(ckpt),
        batch_size=2,
        grpo_k=3,
        epsilon_initial=1.0,
        epsilon_terminal=1.0,
        K_sink=10,
        log_every_steps=0,
        baseline_mode=BASELINE_GRPO,
    )
    torch.manual_seed(0)
    train(config, num_steps=1)
    # 2 instances x 3 rollouts = 6 in one step.
    assert calls["n"] == 2 * 3


def test_grpo_advantage_standardisation():
    costs = [10.0, 20.0, 30.0, 40.0, 50.0]
    adv = grpo_advantages(costs)

    arr = np.asarray(costs)
    expected = (arr - arr.mean()) / (arr.std() + 1e-8)
    assert np.allclose(adv, expected)
    # Standardised: mean ~0, std ~1.
    assert abs(float(np.mean(adv))) < 1e-6
    assert abs(float(np.std(adv)) - 1.0) < 1e-6
    # Below-mean costs get negative advantage, above-mean positive.
    assert adv[0] < 0 and adv[-1] > 0

    # Degenerate group (all equal) -> zero advantages, no divide-by-zero.
    assert np.allclose(grpo_advantages([7.0, 7.0, 7.0]), [0.0, 0.0, 0.0])


def test_grpo_train_end_to_end(tmp_path):
    cache = tmp_path / "train"
    _write_cache(cache, [80090, 80091, 80092])
    ckpt = tmp_path / "checkpoints" / "grpo_e2e.pt"
    config = MethodTwoConfig(
        cache_dir=str(cache),
        checkpoint_path=str(ckpt),
        batch_size=2,
        grpo_k=3,
        epsilon_initial=1.0,
        epsilon_terminal=1.0,
        K_sink=10,
        log_every_steps=1,
        baseline_mode=BASELINE_GRPO,
    )
    torch.manual_seed(0)
    scorer = train(config, num_steps=3)

    assert isinstance(scorer, GNNScorer)
    assert ckpt.exists()
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model_state_dict" in loaded
    assert np.isfinite(loaded["cost_history"]).all()
    GNNScorer().load_state_dict(loaded["model_state_dict"])  # reloads cleanly


# RLOO mode (Component 10 modification, Ahmadian et al. 2024).


def test_rloo_advantage_leave_one_out():
    costs = [10.0, 20.0, 30.0, 40.0, 50.0]
    adv = rloo_advantages(costs)
    # baseline_i = mean of the other 4 costs; advantage_i = c_i - baseline_i.
    expected = [-25.0, -12.5, 0.0, 12.5, 25.0]
    assert np.allclose(adv, expected)
    # No std normalisation: this equals (N/(N-1)) * (c_i - group_mean).
    arr = np.asarray(costs)
    assert np.allclose(adv, (len(arr) / (len(arr) - 1)) * (arr - arr.mean()))
    # Below-mean costs negative, above-mean positive; advantages sum to ~0.
    assert adv[0] < 0 and adv[-1] > 0
    assert abs(sum(adv)) < 1e-6


def test_rloo_train_end_to_end(tmp_path):
    cache = tmp_path / "train"
    _write_cache(cache, [80100, 80101, 80102])
    ckpt = tmp_path / "checkpoints" / "rloo_e2e.pt"
    config = MethodTwoConfig(
        cache_dir=str(cache),
        checkpoint_path=str(ckpt),
        batch_size=2,
        rloo_k=3,
        epsilon_initial=1.0,
        epsilon_terminal=1.0,
        K_sink=10,
        log_every_steps=1,
        baseline_mode=BASELINE_RLOO,
    )
    torch.manual_seed(0)
    scorer = train(config, num_steps=3)

    assert isinstance(scorer, GNNScorer)
    assert ckpt.exists()
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model_state_dict" in loaded
    assert np.isfinite(loaded["cost_history"]).all()
    GNNScorer().load_state_dict(loaded["model_state_dict"])  # reloads cleanly
