"""Data-parallel rollout workers for method two (parallelisation plan B').

Each gradient step fans whole rollouts out to CPU worker processes. A worker
runs its assigned rollouts under autograd on CPU, computes the local REINFORCE
gradient contribution for its baseline mode, and returns only the accumulated
gradient tensors plus the rollout costs. The main process sums the per-worker
gradients, clips the global norm once, and steps the optimiser.

The per-rollout gradient is fully separable, so summing the workers' partial
gradients reproduces the single-process gradient exactly (the gradient is
linear in the per-rollout terms). Workers never touch MPS or CUDA, never step
the optimiser, and never clip, so all device and clipping concerns stay in the
main process.

Determinism. Every rollout is seeded from ``(base_seed, step, inst_seed,
k_idx)`` and nothing else, so results are independent of how many workers run
or how tasks are scheduled. The only randomness inside a rollout is the Gumbel
draw in the perturbed-Hungarian sampler, so seeding torch's global RNG once per
rollout fully determines it.

This module is imported lazily by ``method_two_trainer`` to avoid a circular
import. It imports the low-level rollout and advantage helpers from the trainer
at module load, which is safe because the trainer does not import this module at
its own top level.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import torch

from instances.synthetic_generator import SyntheticInstance
from scoring.gnn_scorer import GNNScorer
from training.method_two_trainer import (
    BASELINE_GRPO,
    BASELINE_RLOO,
    GROUP_BASELINE_MODES,
    grpo_advantages,
    rloo_advantages,
    rollout_one_instance,
)


# Module-global scorer, built once per worker process and reused across tasks.
_WORKER_SCORER: Optional[GNNScorer] = None
_WORKER_KW: Optional[dict] = None


def derive_seed(base_seed: int, step: int, inst_seed: int, k_idx: int) -> int:
    """Deterministic per-rollout seed keyed on the rollout identity.

    Mixes ``(base_seed, step, inst_seed, k_idx)`` with a simple multiplicative
    hash into a 32-bit value. The key deliberately excludes worker identity so
    the same rollout produces the same Gumbel draw regardless of worker count
    or scheduling order.
    """
    h = int(base_seed) & 0xFFFFFFFF
    for v in (int(step), int(inst_seed), int(k_idx)):
        h = (h * 1000003 + (v & 0xFFFFFFFF)) & 0xFFFFFFFF
    return h


def _ensure_scorer(init_kwargs: dict) -> GNNScorer:
    """Return a process-local CPU scorer, building it once and caching it.

    Rebuilds only if the requested architecture differs from the cached one.
    The scorer stays on CPU; weights are loaded fresh per task.
    """
    global _WORKER_SCORER, _WORKER_KW
    if _WORKER_SCORER is None or _WORKER_KW != init_kwargs:
        _WORKER_SCORER = GNNScorer(**init_kwargs)
        _WORKER_KW = dict(init_kwargs)
    return _WORKER_SCORER


def _worker_init(init_kwargs: dict) -> None:
    """Pool initialiser. Pins worker threading and pre-builds the scorer.

    Single-thread BLAS avoids oversubscription across workers and removes a
    source of run-to-run floating-point nondeterminism. The environment
    assignments here are a backstop only: BLAS backends read these at
    import time, and torch is already imported by the time this runs in a
    spawned child, so the values that actually bind are the ones the parent
    process exported before importing torch. ``torch.set_num_threads(1)``
    does take effect at any point, and the confirmation line below prints
    what the worker really has, per worker, once at spawn.
    """
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ[var] = "1"
    torch.set_num_threads(1)
    print(
        f"  [worker {os.getpid()}] torch.get_num_threads()="
        f"{torch.get_num_threads()} "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
        f"VECLIB_MAXIMUM_THREADS={os.environ.get('VECLIB_MAXIMUM_THREADS')}",
        flush=True,
    )
    _ensure_scorer(init_kwargs)


def run_group_task(payload: dict) -> dict:
    """Run one unit of work (one rollout, or one atomic K-rollout group).

    Loads the broadcast weights into the process-local CPU scorer, runs the
    rollouts with autograd, computes the local REINFORCE gradient for the
    payload's baseline mode, and returns the accumulated gradient tensors, the
    rollout costs, and small logging aggregates. No clipping and no optimiser
    step happen here.
    """
    torch.set_num_threads(1)
    scorer = _ensure_scorer(payload["init_kwargs"])
    scorer.load_state_dict(payload["state_dict"])
    scorer.zero_grad(set_to_none=True)

    inst = SyntheticInstance.from_dict(payload["inst_dict"])
    cost_weights = payload["cost_weights"]
    epsilon = float(payload["epsilon"])
    K_sink = int(payload["K_sink"])
    mode = payload["mode"]
    normalizer = float(payload["normalizer"])
    k = int(payload["k"])

    rollouts = []
    for k_idx in range(k):
        seed = derive_seed(
            payload["base_seed"], payload["step"], payload["inst_seed"], k_idx
        )
        torch.manual_seed(seed)
        rollouts.append(
            rollout_one_instance(scorer, inst, cost_weights, epsilon, K_sink)
        )

    costs = [float(r.combined_cost) for r in rollouts]

    if mode == BASELINE_RLOO:
        advantages = rloo_advantages(costs)
        group_std = float(np.std(costs))
    elif mode == BASELINE_GRPO:
        advantages = grpo_advantages(costs)
        group_std = float(np.std(costs) + 1e-8)
    else:
        # Non-group modes: one rollout per task, centred on the scalar baseline from the main process (running-mean EMA or per-instance greedy cost).
        b = float(payload["baseline_scalar"])
        advantages = [c - b for c in costs]
        group_std = 0.0

    for rollout, adv in zip(rollouts, advantages):
        for Theta, P, y_hat in zip(
            rollout.Theta_list, rollout.P_list, rollout.y_hat_list
        ):
            grad_on_theta = (adv / epsilon) * (P - y_hat) / normalizer
            Theta.backward(grad_on_theta.detach(), retain_graph=False)

    grad = {
        name: (
            p.grad.detach().cpu().clone()
            if p.grad is not None
            else torch.zeros_like(p, device="cpu")
        )
        for name, p in scorer.named_parameters()
    }

    return {
        "grad": grad,
        "costs": costs,
        "group_std": group_std,
        "group_mean": float(np.mean(costs)),
    }


def build_payloads(
    scorer: GNNScorer,
    batch: List[Tuple[int, SyntheticInstance, Optional[float]]],
    config,
    epsilon: float,
    step: int,
) -> List[dict]:
    """Build one task payload per batch entry.

    ``batch`` is a list of ``(inst_seed, instance, baseline_scalar)``. For the
    group modes ``baseline_scalar`` is ignored (each group self-baselines); for
    the non-group modes it is the per-rollout centring constant.
    """
    mode = config.baseline_mode
    is_group = mode in GROUP_BASELINE_MODES
    if is_group:
        k = config.rloo_k if mode == BASELINE_RLOO else config.grpo_k
    else:
        k = 1
    B = len(batch)
    normalizer = (B * k) if is_group else B

    state_dict = {
        name: t.detach().cpu().clone() for name, t in scorer.state_dict().items()
    }
    init_kwargs = scorer.get_init_kwargs()

    payloads = []
    for inst_seed, inst, baseline_scalar in batch:
        payloads.append(
            {
                "init_kwargs": init_kwargs,
                "state_dict": state_dict,
                "inst_dict": inst.to_dict(),
                "cost_weights": dict(config.weights),
                "epsilon": float(epsilon),
                "K_sink": int(config.K_sink),
                "mode": mode,
                "normalizer": normalizer,
                "k": int(k),
                "base_seed": int(config.rng_base_seed),
                "step": int(step),
                "inst_seed": int(inst_seed),
                "baseline_scalar": (
                    0.0 if baseline_scalar is None else float(baseline_scalar)
                ),
            }
        )
    return payloads


def sum_gradients(scorer: GNNScorer, results: List[dict]) -> dict:
    """Sum the per-task gradient dicts into one dict keyed by parameter name."""
    names = [n for n, _ in scorer.named_parameters()]
    summed = {n: None for n in names}
    for res in results:
        for n in names:
            g = res["grad"][n]
            summed[n] = g.clone() if summed[n] is None else summed[n] + g
    return summed


def reduce_and_apply(
    scorer: GNNScorer,
    optimiser: torch.optim.Optimizer,
    results: List[dict],
    gradient_clip_norm: float,
) -> float:
    """Sum worker gradients into ``scorer.grad``, clip once, and step.

    The global-norm clip is applied here in the main process on the summed
    gradient, never inside a worker.
    """
    summed = sum_gradients(scorer, results)
    scorer.zero_grad(set_to_none=True)
    for name, p in scorer.named_parameters():
        p.grad = summed[name].to(p.device)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        scorer.parameters(), gradient_clip_norm
    ).item()
    optimiser.step()
    return float(grad_norm)


def make_pool(num_workers: int, init_kwargs: dict) -> ProcessPoolExecutor:
    """Create a persistent spawn-based CPU worker pool.

    Spawn is required for MPS and CUDA safety; the workers stay CPU-only so no
    GPU context is ever created in a child. The pool is reused across all
    gradient steps to amortise the spawn cost.
    """
    ctx = multiprocessing.get_context("spawn")
    return ProcessPoolExecutor(
        max_workers=int(num_workers),
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(init_kwargs,),
    )


def parallel_rollout_step(
    pool: Optional[ProcessPoolExecutor],
    scorer: GNNScorer,
    optimiser: torch.optim.Optimizer,
    batch: List[Tuple[int, SyntheticInstance, Optional[float]]],
    config,
    epsilon: float,
    step: int,
) -> Tuple[dict, List[float]]:
    """Run one parallel gradient step and return ``(metrics, batch_costs)``.

    With ``pool`` set, tasks run across the worker processes; with ``pool`` None
    they run serially in-process (used for tests and debugging of the parallel
    logic, not the verbatim sequential production path). Metrics mirror the
    sequential trainer's dict keys so logging is unchanged.
    """
    payloads = build_payloads(scorer, batch, config, epsilon, step)
    if pool is None:
        results = [run_group_task(p) for p in payloads]
    else:
        results = list(pool.map(run_group_task, payloads))

    grad_norm = reduce_and_apply(
        scorer, optimiser, results, config.gradient_clip_norm
    )

    batch_costs = [c for res in results for c in res["costs"]]
    group_means = [res["group_mean"] for res in results]
    mode = config.baseline_mode

    metrics = {
        "mean_cost": float(np.mean(batch_costs)),
        "grad_norm": float(grad_norm),
        "baseline_used": float(np.mean(group_means)),
    }
    if mode in GROUP_BASELINE_MODES:
        metrics["mean_group_std"] = float(
            np.mean([res["group_std"] for res in results])
        )
    return metrics, batch_costs
