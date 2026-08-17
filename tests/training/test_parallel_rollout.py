"""Failure-mode tests for the parallel rollout path (parallelisation plan B').

Covers the eight verification points from Step 4i of the plan.
    1. Single-rollout exactness vs a sequential reference under a matched seed.
    2. Gradient accumulation: summed per-task grads equal the single-process grad.
    3. Worker-count invariance: real spawn pools of 2 and 3 give identical grads.
    4. Group atomicity: a group's advantages use all K costs (leave-one-out).
    5. RNG independence: distinct k_idx draw distinct Gumbel, same seed repeats.
    6. Clip location: workers never clip; the main process clips the summed grad.
    7. Statistical equivalence: matched-seed parallel costs equal the reference.
    8. num_workers=1 byte-identity: the sequential path never builds a pool and
       is deterministic.

All rollouts run on CPU with single-thread BLAS so the main-process reference
matches the worker arithmetic to within floating-point tolerance.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

torch.set_num_threads(1)

from instances.synthetic_generator import generate_instance
from scoring.gnn_scorer import GNNScorer
from training import method_two_trainer as M
from training.method_two_trainer import (
    MethodTwoConfig,
    rollout_one_instance,
    rloo_advantages,
    grpo_advantages,
)
from training import rollout_worker as W
from training.rollout_worker import (
    build_payloads,
    derive_seed,
    make_pool,
    reduce_and_apply,
    run_group_task,
    sum_gradients,
)


WEIGHTS = dict(M.DEFAULT_WEIGHTS)
EPS = 1.0
K_SINK = 10
BASE_SEED = 0
STEP = 1


def make_scorer(seed: int = 0) -> GNNScorer:
    torch.manual_seed(seed)
    return GNNScorer().cpu()


def copy_scorer(src: GNNScorer) -> GNNScorer:
    dst = GNNScorer().cpu()
    dst.load_state_dict({k: v.clone() for k, v in src.state_dict().items()})
    return dst


def reference_group(
    scorer: GNNScorer,
    inst,
    inst_seed: int,
    mode: str,
    k: int,
    normalizer: float,
    baseline: float = 0.0,
    eps: float = EPS,
):
    """Sequential reference mirroring ``run_group_task`` arithmetic exactly."""
    scorer.zero_grad(set_to_none=True)
    rollouts = []
    for k_idx in range(k):
        torch.manual_seed(derive_seed(BASE_SEED, STEP, inst_seed, k_idx))
        rollouts.append(rollout_one_instance(scorer, inst, WEIGHTS, eps, K_SINK))
    costs = [float(r.combined_cost) for r in rollouts]
    if mode == M.BASELINE_RLOO:
        advs = rloo_advantages(costs)
    elif mode == M.BASELINE_GRPO:
        advs = grpo_advantages(costs)
    else:
        advs = [c - baseline for c in costs]
    for r, adv in zip(rollouts, advs):
        for Theta, P, y_hat in zip(r.Theta_list, r.P_list, r.y_hat_list):
            Theta.backward(((adv / eps) * (P - y_hat) / normalizer).detach())
    grad = {
        n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        for n, p in scorer.named_parameters()
    }
    return grad, costs


def assert_grads_close(g1, g2, atol=1e-4, rtol=1e-4):
    for name in g1:
        assert torch.allclose(
            g1[name], g2[name], atol=atol, rtol=rtol
        ), f"gradient mismatch on {name}: max diff {(g1[name]-g2[name]).abs().max()}"


def global_norm(grad_dict) -> float:
    return float(
        torch.sqrt(sum((g.detach() ** 2).sum() for g in grad_dict.values()))
    )


# 1. Single-rollout exactness
def test_single_rollout_exactness():
    scorer = make_scorer(7)
    inst = generate_instance(123)
    inst_seed = inst.seed

    config = MethodTwoConfig(
        baseline_mode=M.BASELINE_RUNNING_MEAN, K_sink=K_SINK, rng_base_seed=BASE_SEED
    )
    batch = [(inst_seed, inst, 0.0)]  # baseline scalar 0
    payloads = build_payloads(scorer, batch, config, EPS, STEP)
    result = run_group_task(payloads[0])

    ref_grad, ref_costs = reference_group(
        copy_scorer(scorer), inst, inst_seed, M.BASELINE_RUNNING_MEAN, 1, 1.0, 0.0
    )

    assert result["costs"] == pytest.approx(ref_costs, abs=1e-9)
    assert_grads_close(result["grad"], ref_grad)


# 2. Gradient accumulation
def test_gradient_accumulation_equals_single_process():
    scorer = make_scorer(11)
    seeds = [201, 202, 203]
    instances = [generate_instance(s) for s in seeds]
    inst_seeds = [i.seed for i in instances]
    k = 2
    config = MethodTwoConfig(
        baseline_mode=M.BASELINE_RLOO, rloo_k=k, K_sink=K_SINK, rng_base_seed=BASE_SEED
    )
    total = len(instances) * k

    batch = [(s, inst, None) for s, inst in zip(inst_seeds, instances)]
    payloads = build_payloads(scorer, batch, config, EPS, STEP)
    results = [run_group_task(p) for p in payloads]
    summed = sum_gradients(scorer, results)

    # Single-process reference: accumulate ALL groups into one grad (no per-group zeroing), which the summed worker grads must equal.
    ref_scorer = copy_scorer(scorer)
    ref_scorer.zero_grad(set_to_none=True)
    for s, inst in zip(inst_seeds, instances):
        rollouts = []
        for k_idx in range(k):
            torch.manual_seed(derive_seed(BASE_SEED, STEP, s, k_idx))
            rollouts.append(rollout_one_instance(ref_scorer, inst, WEIGHTS, EPS, K_SINK))
        costs = [float(r.combined_cost) for r in rollouts]
        advs = rloo_advantages(costs)
        for r, adv in zip(rollouts, advs):
            for Theta, P, y_hat in zip(r.Theta_list, r.P_list, r.y_hat_list):
                Theta.backward(((adv / EPS) * (P - y_hat) / total).detach())
    ref = {n: p.grad.detach().clone() for n, p in ref_scorer.named_parameters()}

    assert_grads_close(summed, ref)


# 3. Worker-count invariance (real spawn pools)
def test_worker_count_invariance():
    scorer = make_scorer(13)
    seeds = [301, 302, 303]
    instances = [generate_instance(s) for s in seeds]
    inst_seeds = [i.seed for i in instances]
    k = 2
    config = MethodTwoConfig(
        baseline_mode=M.BASELINE_RLOO, rloo_k=k, K_sink=K_SINK, rng_base_seed=BASE_SEED
    )
    batch = [(s, inst, None) for s, inst in zip(inst_seeds, instances)]
    payloads = build_payloads(scorer, batch, config, EPS, STEP)

    serial = sum_gradients(scorer, [run_group_task(p) for p in payloads])

    init_kwargs = scorer.get_init_kwargs()
    sums = {}
    for nw in (2, 3):
        pool = make_pool(nw, init_kwargs)
        try:
            results = list(pool.map(run_group_task, payloads))
        finally:
            pool.shutdown(wait=True)
        sums[nw] = sum_gradients(scorer, results)

    assert_grads_close(sums[2], sums[3])
    assert_grads_close(sums[2], serial)


# 4. Group atomicity / leave-one-out advantages over all K costs
def test_group_atomic_advantages():
    scorer = make_scorer(17)
    inst = generate_instance(404)
    inst_seed = inst.seed
    k = 4
    config = MethodTwoConfig(
        baseline_mode=M.BASELINE_RLOO, rloo_k=k, K_sink=K_SINK, rng_base_seed=BASE_SEED
    )
    batch = [(inst_seed, inst, None)]
    payloads = build_payloads(scorer, batch, config, EPS, STEP)
    result = run_group_task(payloads[0])

    # The advantages must be the leave-one-out advantages over ALL k costs.
    expected_advs = rloo_advantages(result["costs"])
    assert len(result["costs"]) == k

    # Reconstruct the grad from those advantages over the same-seed rollouts.
    ref = copy_scorer(scorer)
    ref.zero_grad(set_to_none=True)
    rollouts = []
    for k_idx in range(k):
        torch.manual_seed(derive_seed(BASE_SEED, STEP, inst_seed, k_idx))
        rollouts.append(rollout_one_instance(ref, inst, WEIGHTS, EPS, K_SINK))
    for r, adv in zip(rollouts, expected_advs):
        for Theta, P, y_hat in zip(r.Theta_list, r.P_list, r.y_hat_list):
            Theta.backward(((adv / EPS) * (P - y_hat) / k).detach())
    ref_grad = {n: p.grad.detach().clone() for n, p in ref.named_parameters()}

    assert_grads_close(result["grad"], ref_grad)


# 5. RNG independence within a group, reproducibility under a fixed seed
def test_rng_independence_within_group():
    scorer = make_scorer(19)
    inst = generate_instance(505)
    inst_seed = inst.seed

    seed_a = derive_seed(BASE_SEED, STEP, inst_seed, 0)
    seed_b = derive_seed(BASE_SEED, STEP, inst_seed, 1)
    assert seed_a != seed_b

    torch.manual_seed(seed_a)
    ra = rollout_one_instance(scorer, inst, WEIGHTS, EPS, K_SINK)
    torch.manual_seed(seed_b)
    rb = rollout_one_instance(scorer, inst, WEIGHTS, EPS, K_SINK)

    def differs(la, lb):
        if len(la) != len(lb):
            return True
        return any(not torch.equal(a, b) for a, b in zip(la, lb))

    # Distinct k_idx must produce distinct Gumbel draws, hence distinct samples.
    assert differs(ra.P_list, rb.P_list), "two group members drew identical samples"

    # Same seed must reproduce the same draws exactly.
    torch.manual_seed(seed_a)
    ra2 = rollout_one_instance(scorer, inst, WEIGHTS, EPS, K_SINK)
    assert len(ra.P_list) == len(ra2.P_list)
    assert all(torch.equal(a, b) for a, b in zip(ra.P_list, ra2.P_list))


# 6. Clip happens in the main process on the summed grad, never in workers
def test_clip_in_main_not_in_worker():
    scorer = make_scorer(23)
    inst = generate_instance(606)
    inst_seed = inst.seed
    clip = 1.0

    config = MethodTwoConfig(
        baseline_mode=M.BASELINE_RUNNING_MEAN,
        K_sink=K_SINK,
        rng_base_seed=BASE_SEED,
        gradient_clip_norm=clip,
    )
    # A hugely negative baseline blows up the advantage so the raw grad norm vastly exceeds the clip; if the worker clipped, this would not hold.
    batch = [(inst_seed, inst, -1.0e6)]
    payloads = build_payloads(scorer, batch, config, EPS, STEP)
    result = run_group_task(payloads[0])

    worker_norm = global_norm(result["grad"])
    assert worker_norm > clip, "worker appears to have clipped its gradient"

    optimiser = torch.optim.Adam(scorer.parameters(), lr=1e-4)
    reduce_and_apply(scorer, optimiser, [result], clip)
    applied = {n: p.grad.detach().clone() for n, p in scorer.named_parameters()}
    assert global_norm(applied) <= clip + 1e-4


# 7. Statistical equivalence: matched-seed parallel costs equal the reference
def test_matched_seed_cost_equivalence():
    scorer = make_scorer(29)
    seeds = [701, 702, 703, 704]
    instances = [generate_instance(s) for s in seeds]
    inst_seeds = [i.seed for i in instances]
    k = 3
    config = MethodTwoConfig(
        baseline_mode=M.BASELINE_RLOO, rloo_k=k, K_sink=K_SINK, rng_base_seed=BASE_SEED
    )
    batch = [(s, inst, None) for s, inst in zip(inst_seeds, instances)]
    payloads = build_payloads(scorer, batch, config, EPS, STEP)
    results = [run_group_task(p) for p in payloads]

    ref_scorer = copy_scorer(scorer)
    for s, inst, res in zip(inst_seeds, instances, results):
        ref_costs = []
        for k_idx in range(k):
            torch.manual_seed(derive_seed(BASE_SEED, STEP, s, k_idx))
            r = rollout_one_instance(ref_scorer, inst, WEIGHTS, EPS, K_SINK)
            ref_costs.append(float(r.combined_cost))
        assert res["costs"] == pytest.approx(ref_costs, abs=1e-9)


# 8. num_workers=1 takes the sequential path verbatim and is deterministic
def _write_tiny_cache(cache_dir: Path, seeds):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        inst = generate_instance(s)
        rec = {"seed": int(s), "instance": inst.to_dict()}
        with (cache_dir / f"seed{int(s):06d}.json").open("w") as f:
            json.dump(rec, f)


def _final_weights(config, num_steps):
    random.seed(0)
    torch.manual_seed(0)
    scorer = M.train(config, num_steps)
    return {n: p.detach().clone() for n, p in scorer.named_parameters()}


def test_num_workers_one_no_pool_and_deterministic(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    _write_tiny_cache(cache_dir, [810, 811, 812, 813])

    # num_workers=1 must never construct a worker pool.
    def _boom(*a, **k):
        raise AssertionError("make_pool called on the num_workers=1 path")

    monkeypatch.setattr(W, "make_pool", _boom)

    # Pin the sequential path to CPU: MPS RNG is non-deterministic run-to-run, so the determinism claim is a CPU claim, testing our code not the MPS backend.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    def cfg():
        return MethodTwoConfig(
            cache_dir=str(cache_dir),
            checkpoint_path=str(tmp_path / "ckpt.pt"),
            baseline_mode=M.BASELINE_RLOO,
            rloo_k=2,
            batch_size=2,
            K_sink=K_SINK,
            epsilon_initial=EPS,
            epsilon_terminal=EPS,
            log_every_steps=0,
            checkpoint_every_steps=100,
            num_workers=1,
        )

    w1 = _final_weights(cfg(), 2)
    w2 = _final_weights(cfg(), 2)
    for n in w1:
        assert torch.equal(w1[n], w2[n]), f"sequential path not deterministic at {n}"
