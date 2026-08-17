"""Method two rollout and REINFORCE trainer (methodology section 3.4).

For each gradient step a minibatch of training instances is rolled out under
the perturbed-Hungarian policy (section 3.4.1). Each trajectory's combined
cost is scored, and the per-epoch score-function gradient (section 3.4.4)

    grad_w log p_eps(a_t | theta_t, s_t) = (1 / eps) * (a_t - y_hat_eps)^T grad_w theta_t

is weighted by the centred return ``(C(traj) - baseline)`` and backpropagated
through the scorer via ``Theta_t.backward(...)``. The baseline is a running
EMA of batch mean cost (section 3.4.5), gradients are clipped by global norm
nu = 10 and applied with Adam at lr 1e-4 (section 3.4.7).

Because Adam performs gradient descent and the score-function gradient is
weighted by ``+advantage``, the update reduces the probability of
higher-than-baseline-cost trajectories, i.e. it minimises expected cost.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch

from instances.synthetic_generator import SyntheticInstance
from simulator.dynamic_simulator import DynamicSimulator
from scoring.gnn_scorer import GNNScorer
from sampling.perturbed_hungarian import perturbed_hungarian_sample
from baselines.bipartite_policies import run_greedy_policy


DEFAULT_WEIGHTS = {"w_dist": 0.0637, "w_make": 0.2398, "w_bal": 0.6965}

BASELINE_RUNNING_MEAN = "running_mean"
BASELINE_PER_INSTANCE_GREEDY = "per_instance_greedy"
BASELINE_GRPO = "grpo"
BASELINE_RLOO = "rloo"
BASELINE_MODES = (
    BASELINE_RUNNING_MEAN,
    BASELINE_PER_INSTANCE_GREEDY,
    BASELINE_GRPO,
    BASELINE_RLOO,
)
GROUP_BASELINE_MODES = (BASELINE_GRPO, BASELINE_RLOO)


@dataclass
class MethodTwoConfig:
    learning_rate: float = 1e-4
    batch_size: int = 64
    baseline_alpha: float = 0.05
    gradient_clip_norm: float = 10.0
    log_every_steps: int = 50
    cache_dir: str = "cache/training_set_il_v3/train"
    checkpoint_path: str = "checkpoints/method_two.pt"
    weights: dict = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    K_sink: int = 20
    epsilon_initial: float = 1.0
    epsilon_terminal: float = 0.01
    # None anneals across all num_steps; an int anneals over that many steps then holds at epsilon_terminal.
    epsilon_anneal_steps: Optional[int] = None
    # "running_mean" is the EMA of batch mean cost; "per_instance_greedy" centres each trajectory on its own instance's greedy cost.
    baseline_mode: str = BASELINE_RUNNING_MEAN
    warm_start_from: Optional[str] = None
    # GRPO (Shao et al. 2024, DEC-06): rollouts per instance whose costs form the group for the standardised advantage.
    grpo_k: int = 5
    # RLOO (Ahmadian et al. 2024): rollouts per instance for the leave-one-out baseline (raw group mean, no std normalisation).
    rloo_k: int = 5
    resume_from: Optional[str] = None
    start_step: int = 1
    checkpoint_every_steps: int = 25
    # num_workers <= 1 is bit-identical to the sequential path; >= 2 fans rollouts out to spawn-started CPU worker processes keyed by rng_base_seed.
    num_workers: int = 1
    worker_start_method: str = "spawn"
    rng_base_seed: int = 0
    # eval_every_steps > 0 with val_cache_dir set evaluates on the val cache under hard decode and saves the best gap-closure checkpoint separately, without touching the periodic one; 0 disables both.
    eval_every_steps: int = 0
    val_cache_dir: Optional[str] = None
    num_val_instances: int = 200
    # Floor on hard-decode served_all_rate to qualify as best; without it a low-coverage policy could post a high gap on the surviving subset (mirrors the guard in il_trainer).
    best_serve_all_floor: float = 0.95


@dataclass
class RolloutResult:
    """Trajectory information needed for the REINFORCE gradient."""

    Theta_list: list  # differentiable in scorer params
    P_list: list
    y_hat_list: list
    combined_cost: float


class InstanceDataset:
    """Lazy index of training instances from the cache.

    Reads only the instance spec from each record, ignoring MILP solutions and
    expert decisions (method two is reward-driven, not imitation).
    """

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.index = []  # list of (seed, instance_dict)
        self._greedy_cost = {}  # seed -> greedy combined cost (lazy)
        for f in sorted(self.cache_dir.glob("seed*.json")):
            with f.open("r") as fp:
                rec = json.load(fp)
            self.index.append((rec["seed"], rec["instance"]))
        print(
            f"InstanceDataset loaded {len(self.index)} instances "
            f"from {self.cache_dir}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.index)

    def get(self, idx: int) -> SyntheticInstance:
        _, inst_dict = self.index[idx]
        return SyntheticInstance.from_dict(inst_dict)

    def greedy_cost(self, idx: int, weights: dict) -> float:
        """Greedy combined cost for instance ``idx``, cached per seed."""
        seed, inst_dict = self.index[idx]
        if seed not in self._greedy_cost:
            instance = SyntheticInstance.from_dict(inst_dict)
            self._greedy_cost[seed] = compute_greedy_baseline(instance, weights)
        return self._greedy_cost[seed]


def compute_greedy_baseline(
    instance: SyntheticInstance, weights: dict
) -> float:
    """Greedy combined cost for an instance (methodology section 3.4.5).

    Runs the myopic greedy policy to termination and returns its combined
    trajectory cost, used as a per-instance REINFORCE baseline.
    """
    sim = run_greedy_policy(instance)
    return float(sim.compute_cost(weights).combined)


def annealed_epsilon(
    step: int,
    num_steps: int,
    epsilon_initial: float,
    epsilon_terminal: float,
    anneal_steps: Optional[int] = None,
) -> float:
    """Log-linear (exponential) interpolation of the perturbation epsilon.

    ``step`` is 0-indexed. Progress runs from 0 at the first step to 1 at the
    end of the anneal window (``anneal_steps`` if given, else ``num_steps``),
    then holds at ``epsilon_terminal``. Mirrors method one's schedule
    (methodology section 3.4.6).
    """
    log_initial = math.log(epsilon_initial)
    log_terminal = math.log(epsilon_terminal)
    span = anneal_steps if anneal_steps else num_steps
    progress = step / max(1, span - 1)
    progress = min(progress, 1.0)
    return math.exp(log_initial + progress * (log_terminal - log_initial))


def rollout_one_instance(
    scorer: GNNScorer,
    instance: SyntheticInstance,
    weights: dict,
    epsilon: float,
    K_sink: int,
) -> RolloutResult:
    """Roll out the policy on one instance with perturbed-Hungarian sampling.

    Records the per-epoch ``Theta``, ``P``, ``y_hat`` tensors needed for the
    REINFORCE gradient and returns the final combined cost.
    """
    sim = DynamicSimulator(instance)
    R, T = instance.R, instance.T

    Theta_list = []
    P_list = []
    y_hat_list = []

    while not sim.is_terminal:
        state = sim.state
        Theta = scorer(state, instance)  # requires grad
        P, dispatcher_action, y_hat = perturbed_hungarian_sample(
            Theta, epsilon, R, T, K_sink=K_sink
        )

        Theta_list.append(Theta)
        P_list.append(P)
        y_hat_list.append(y_hat)

        # Guard against any masked picks slipping through Hungarian.
        valid_commits = [
            (r, c)
            for r, c in dispatcher_action
            if r in state.available_robots and c in state.pending_tasks
        ]
        sim.step(valid_commits)

    combined_cost = sim.compute_cost(weights).combined

    return RolloutResult(
        Theta_list=Theta_list,
        P_list=P_list,
        y_hat_list=y_hat_list,
        combined_cost=combined_cost,
    )


def reinforce_gradient_step(
    scorer: GNNScorer,
    optimiser: torch.optim.Optimizer,
    rollouts: List[RolloutResult],
    baseline: Union[float, None, List[float]],
    epsilon: float,
    gradient_clip_norm: float,
) -> dict:
    """Apply one REINFORCE gradient step over a batch of rollouts.

    ``baseline`` is a scalar/None (running-mean) or a per-rollout list (greedy).
    Returns a dict with the batch mean cost, the scorer gradient norm and the
    baseline value used.
    """
    costs = [r.combined_cost for r in rollouts]
    mean_cost = float(np.mean(costs))
    B = len(rollouts)

    optimiser.zero_grad()
    if isinstance(baseline, (list, tuple)):
        if len(baseline) != B:
            raise ValueError(
                f"Per-instance baseline length {len(baseline)} != batch size {B}"
            )
        baselines = [float(b) for b in baseline]
    else:
        scalar = mean_cost if baseline is None else float(baseline)
        baselines = [scalar] * B

    for rollout, b in zip(rollouts, baselines):
        advantage = rollout.combined_cost - b
        for Theta, P, y_hat in zip(
            rollout.Theta_list, rollout.P_list, rollout.y_hat_list
        ):
            # Detached so it acts as the upstream gradient seed; autograd chains through Theta = GNNScorer(state, instance).
            grad_on_theta = (advantage / epsilon) * (P - y_hat) / B
            Theta.backward(grad_on_theta.detach(), retain_graph=False)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        scorer.parameters(), gradient_clip_norm
    ).item()
    optimiser.step()

    return {
        "mean_cost": mean_cost,
        "grad_norm": float(grad_norm),
        "baseline_used": float(np.mean(baselines)),
    }


def grpo_advantages(costs: List[float]) -> List[float]:
    """Group-relative standardised advantages (Shao et al. 2024, DEC-06).

    ``advantage_k = (C_k - mean(C)) / (std(C) + 1e-8)`` over the K costs of one
    instance's group. Population std (ddof=0). Below-mean trajectories get
    negative advantage, which under the cost-minimising sign convention raises
    the probability of their actions.
    """
    arr = np.asarray(costs, dtype=np.float64)
    mean = arr.mean()
    std = arr.std() + 1e-8
    return ((arr - mean) / std).tolist()


def reinforce_grpo_gradient_step(
    scorer: GNNScorer,
    optimiser: torch.optim.Optimizer,
    grouped_rollouts: List[List[RolloutResult]],
    epsilon: float,
    gradient_clip_norm: float,
) -> dict:
    """One GRPO gradient step over groups of K rollouts per instance.

    Each group's K trajectory costs are standardised to advantages; every
    per-epoch ``(P_t - y_hat_t)`` term in a rollout is weighted by that
    rollout's standardised advantage, scaled by ``1 / total_rollouts``.
    """
    optimiser.zero_grad()

    total_rollouts = sum(len(g) for g in grouped_rollouts)
    all_costs = []
    group_means = []
    group_stds = []

    for group in grouped_rollouts:
        costs = [r.combined_cost for r in group]
        all_costs.extend(costs)
        advantages = grpo_advantages(costs)
        group_means.append(float(np.mean(costs)))
        group_stds.append(float(np.std(costs) + 1e-8))
        for rollout, adv in zip(group, advantages):
            for Theta, P, y_hat in zip(
                rollout.Theta_list, rollout.P_list, rollout.y_hat_list
            ):
                grad_on_theta = (adv / epsilon) * (P - y_hat) / total_rollouts
                Theta.backward(grad_on_theta.detach(), retain_graph=False)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        scorer.parameters(), gradient_clip_norm
    ).item()
    optimiser.step()

    return {
        "mean_cost": float(np.mean(all_costs)),
        "grad_norm": float(grad_norm),
        "mean_group_std": float(np.mean(group_stds)),
        "baseline_used": float(np.mean(group_means)),
    }


def rloo_advantages(costs: List[float]) -> List[float]:
    """Leave-one-out advantages (Ahmadian et al. 2024).

    For each rollout ``i``, the baseline is the mean of the *other* K-1 group
    costs, so ``advantage_i = C_i - mean(C_{j != i})``. This is an unbiased
    leave-one-out baseline with NO std normalisation, which avoids the
    positive-feedback blow-up GRPO suffers as the group variance collapses (the
    1/std factor amplifies advantages when the policy becomes more
    deterministic).
    """
    N = len(costs)
    if N < 2:
        raise ValueError(
            f"RLOO needs at least 2 rollouts per group, got {N}; the "
            f"leave-one-out baseline is undefined for a single rollout."
        )
    costs_arr = np.asarray(costs, dtype=np.float64)
    total = costs_arr.sum()
    advantages = []
    for c_i in costs_arr:
        mean_others = (total - c_i) / (N - 1)
        advantages.append(float(c_i - mean_others))
    return advantages


def reinforce_rloo_gradient_step(
    scorer: GNNScorer,
    optimiser: torch.optim.Optimizer,
    grouped_rollouts: List[List[RolloutResult]],
    epsilon: float,
    gradient_clip_norm: float,
) -> dict:
    """One RLOO gradient step over groups of K rollouts per instance.

    Mirrors ``reinforce_grpo_gradient_step`` but weights each rollout by its
    leave-one-out advantage (raw, un-normalised), scaled by
    ``1 / total_rollouts``.
    """
    optimiser.zero_grad()

    total_rollouts = sum(len(g) for g in grouped_rollouts)
    all_costs = []
    group_means = []
    group_stds = []

    for group in grouped_rollouts:
        costs = [r.combined_cost for r in group]
        all_costs.extend(costs)
        advantages = rloo_advantages(costs)
        group_means.append(float(np.mean(costs)))
        group_stds.append(float(np.std(costs)))
        for rollout, adv in zip(group, advantages):
            for Theta, P, y_hat in zip(
                rollout.Theta_list, rollout.P_list, rollout.y_hat_list
            ):
                grad_on_theta = (adv / epsilon) * (P - y_hat) / total_rollouts
                Theta.backward(grad_on_theta.detach(), retain_graph=False)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        scorer.parameters(), gradient_clip_norm
    ).item()
    optimiser.step()

    return {
        "mean_cost": float(np.mean(all_costs)),
        "grad_norm": float(grad_norm),
        "mean_group_std": float(np.mean(group_stds)),
        "baseline_used": float(np.mean(group_means)),
    }


def update_baseline(
    baseline: Optional[float], batch_costs: List[float], alpha: float
) -> float:
    """EMA update of the cost baseline (section 3.4.5)."""
    batch_mean = float(np.mean(batch_costs))
    if baseline is None:
        return batch_mean
    return (1 - alpha) * baseline + alpha * batch_mean


def best_checkpoint_path(checkpoint_path: str) -> str:
    """Path for the separate best checkpoint, derived from ``checkpoint_path``.

    Inserts ``_best`` before the ``.pt`` suffix (e.g. ``foo.pt`` becomes
    ``foo_best.pt``) so the best checkpoint never collides with the periodic
    one. A path without a ``.pt`` suffix just gets ``_best.pt`` appended.
    """
    p = str(checkpoint_path)
    if p.endswith(".pt"):
        return p[:-3] + "_best.pt"
    return p + "_best.pt"


def evaluate_val_gap_closure(scorer: GNNScorer, val_records: list, weights: dict):
    """Mean serving gap closure versus the MILP bound on the validation cache.

    Evaluates every record under hard (epsilon = 0) decode, which is fully
    deterministic and draws no random numbers, so it does not perturb the
    training RNG stream. Returns ``(gap, served_all_rate, summary)`` where
    ``gap`` is ``summary["modes"]["hard"]["serving_gap_vs_milp"]["mean"]`` and
    is ``None`` when no instance served all tasks. The scorer is returned to
    train mode afterwards (``rollout_policy`` leaves it in eval mode).
    """
    from evaluation.method_two_evaluator import evaluate_one_instance, summarize

    was_training = scorer.training
    records = [
        evaluate_one_instance(
            scorer, rec, modes=["hard"], weights=weights, epsilon=0.0
        )
        for rec in val_records
    ]
    if was_training:
        scorer.train()

    summary = summarize(records, modes=["hard"])
    hard = summary["modes"]["hard"]
    return (
        hard["serving_gap_vs_milp"]["mean"],
        hard["served_all_rate"],
        summary,
    )


def train(
    config: MethodTwoConfig,
    num_steps: int,
    step_callback: Optional[Callable[[int], None]] = None,
) -> GNNScorer:
    """Top-level training entry for method two.

    ``step_callback``, if given, is called once per completed step with the
    step number, at the very end of the step, after that step's gradient
    update, metrics, in-loop validation and periodic checkpoint save -- e.g.
    for external RSS logging, thermal monitoring, or a resource-budget
    abort. It may raise; the exception propagates out of this function
    (through the worker-pool shutdown in ``finally``) to the caller, which
    decides how to report and stop. Firing it last is deliberate: a
    callback that aborts on a checkpoint step leaves that step's checkpoint
    already written, so the run stays resumable from the step it aborted
    on. Optional and defaults to None, so existing callers are
    unaffected."""
    from training.rollout_worker import make_pool, parallel_rollout_step

    use_parallel = (
        config.num_workers >= 2
        and config.baseline_mode in BASELINE_MODES
    )

    if use_parallel:
        # Canonical model/optimiser stay on CPU to match workers, avoiding MPS/CUDA subprocess pitfalls and making weight broadcast a no-op transfer.
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(
        f"Training on device: {device}"
        + (f" (parallel, {config.num_workers} workers)" if use_parallel else ""),
        flush=True,
    )
    dataset = InstanceDataset(Path(config.cache_dir))

    # Method two inherits its architecture from resume_from/warm_start_from's init_kwargs (resume wins, since its state dict is what loads); missing key falls back to module defaults.
    init_kwargs = {}
    shape_source = None
    for candidate in (config.resume_from, config.warm_start_from):
        if not candidate:
            continue
        peek = torch.load(candidate, map_location="cpu", weights_only=False)
        if isinstance(peek.get("init_kwargs"), dict):
            init_kwargs = dict(peek["init_kwargs"])
            shape_source = candidate
        break
    if init_kwargs:
        print(
            f"Building scorer to match {shape_source}: {init_kwargs}",
            flush=True,
        )
    scorer = GNNScorer(**init_kwargs)

    if config.warm_start_from:
        print(f"Warm-starting from {config.warm_start_from}", flush=True)
        ckpt = torch.load(
            config.warm_start_from, map_location=device, weights_only=False
        )
        scorer.load_state_dict(ckpt["model_state_dict"])

    scorer = scorer.to(device)
    optimiser = torch.optim.Adam(scorer.parameters(), lr=config.learning_rate)

    Path(config.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    if config.baseline_mode not in BASELINE_MODES:
        raise ValueError(f"Unknown baseline_mode: {config.baseline_mode}")

    print(
        f"Training method two for {num_steps} steps. "
        f"Dataset size: {len(dataset)}",
        flush=True,
    )
    print(
        f"Epsilon anneal: {config.epsilon_initial} -> {config.epsilon_terminal}"
        f" over {config.epsilon_anneal_steps or num_steps} steps (log scale). "
        f"Baseline: {config.baseline_mode}. "
        f"Batch size: {config.batch_size}. lr: {config.learning_rate}.",
        flush=True,
    )

    baseline: Optional[float] = None  # running-mean EMA state
    costs_history = []
    grad_norms = []
    grad_norm_history = []  # parallel to costs_history, persisted in checkpoints
    t0 = time.time()

    track_best = bool(config.eval_every_steps and config.val_cache_dir)
    best_gap: Optional[float] = None
    best_step: Optional[int] = None
    # Persisted in the periodic checkpoint so the trajectory survives without the stdout log.
    eval_history: list = []
    best_ckpt_path = best_checkpoint_path(config.checkpoint_path)
    val_records: list = []
    if track_best:
        from evaluation.method_two_evaluator import load_records

        val_records = load_records(
            config.val_cache_dir, config.num_val_instances
        )
        print(
            f"Best-checkpoint tracking on: evaluating {len(val_records)} val "
            f"instances every {config.eval_every_steps} steps (hard gap vs "
            f"MILP), best saved to {best_ckpt_path}.",
            flush=True,
        )

    if config.resume_from is not None:
        resume_ckpt = torch.load(
            config.resume_from, map_location=device, weights_only=False
        )
        scorer.load_state_dict(resume_ckpt["model_state_dict"])
        optimiser.load_state_dict(resume_ckpt["optimizer_state_dict"])
        baseline = resume_ckpt.get("baseline", None)
        costs_history = resume_ckpt.get("cost_history", []) or []
        grad_norm_history = resume_ckpt.get("grad_norm_history", []) or []
        # Carried across a resume so an overnight restart mid-sweep doesn't lose the best-so-far.
        best_gap = resume_ckpt.get("best_gap", None)
        best_step = resume_ckpt.get("best_step", None)
        # Absent in checkpoints written before eval_history existed, hence the default.
        eval_history = resume_ckpt.get("eval_history", []) or []
        # MethodTwoConfig is a mutable dataclass, not a NamedTuple, so assign directly rather than _replace.
        config.start_step = resume_ckpt["step"] + 1
        print(
            f"Resumed from step {resume_ckpt['step']} at {config.resume_from}"
            + (
                f" (best gap {best_gap:.4f} at step {best_step})"
                if best_gap is not None
                else ""
            ),
            flush=True,
        )

    pool = None
    if use_parallel:
        pool = make_pool(config.num_workers, scorer.get_init_kwargs())
        print(
            f"Started spawn worker pool with {config.num_workers} workers.",
            flush=True,
        )

    try:
        for step in range(config.start_step, num_steps + 1):
            epsilon = annealed_epsilon(
                step - 1,
                num_steps,
                config.epsilon_initial,
                config.epsilon_terminal,
                config.epsilon_anneal_steps,
            )
            batch_indices = random.sample(range(len(dataset)), config.batch_size)
            instances = [dataset.get(i) for i in batch_indices]

            # First running-mean step (baseline is None) runs sequentially to establish the EMA, since the parallel path can't reproduce it without a forward-only pre-pass.
            run_parallel_this_step = use_parallel and not (
                config.baseline_mode == BASELINE_RUNNING_MEAN and baseline is None
            )

            if run_parallel_this_step:
                inst_seeds = [dataset.index[i][0] for i in batch_indices]
                if config.baseline_mode == BASELINE_PER_INSTANCE_GREEDY:
                    baseline_scalars = [
                        dataset.greedy_cost(i, config.weights) for i in batch_indices
                    ]
                elif config.baseline_mode == BASELINE_RUNNING_MEAN:
                    baseline_scalars = [baseline] * len(batch_indices)
                else:
                    baseline_scalars = [None] * len(batch_indices)
                batch = list(zip(inst_seeds, instances, baseline_scalars))
                metrics, batch_costs = parallel_rollout_step(
                    pool, scorer, optimiser, batch, config, epsilon, step
                )
                if config.baseline_mode == BASELINE_RUNNING_MEAN:
                    baseline = update_baseline(
                        baseline, batch_costs, config.baseline_alpha
                    )
            elif config.baseline_mode in GROUP_BASELINE_MODES:
                # Gumbel RNG advances between calls, so each rollout in a group is a distinct sample.
                group_k = (
                    config.grpo_k
                    if config.baseline_mode == BASELINE_GRPO
                    else config.rloo_k
                )
                grouped_rollouts = [
                    [
                        rollout_one_instance(
                            scorer, inst, config.weights, epsilon, config.K_sink
                        )
                        for _ in range(group_k)
                    ]
                    for inst in instances
                ]
                group_step = (
                    reinforce_grpo_gradient_step
                    if config.baseline_mode == BASELINE_GRPO
                    else reinforce_rloo_gradient_step
                )
                metrics = group_step(
                    scorer,
                    optimiser,
                    grouped_rollouts,
                    epsilon,
                    config.gradient_clip_norm,
                )
            else:
                rollouts = [
                    rollout_one_instance(
                        scorer, inst, config.weights, epsilon, config.K_sink
                    )
                    for inst in instances
                ]

                if config.baseline_mode == BASELINE_PER_INSTANCE_GREEDY:
                    step_baseline: Union[float, None, List[float]] = [
                        dataset.greedy_cost(i, config.weights)
                        for i in batch_indices
                    ]
                else:
                    step_baseline = baseline

                metrics = reinforce_gradient_step(
                    scorer,
                    optimiser,
                    rollouts,
                    step_baseline,
                    epsilon,
                    config.gradient_clip_norm,
                )

                # Only the EMA variant carries state across steps; greedy mode's baseline is fixed per-instance.
                if config.baseline_mode == BASELINE_RUNNING_MEAN:
                    batch_costs = [r.combined_cost for r in rollouts]
                    baseline = update_baseline(
                        baseline, batch_costs, config.baseline_alpha
                    )

            costs_history.append(metrics["mean_cost"])
            grad_norms.append(metrics["grad_norm"])
            grad_norm_history.append(metrics["grad_norm"])

            if config.log_every_steps and step % config.log_every_steps == 0:
                elapsed = time.time() - t0
                recent_cost = np.mean(costs_history[-config.log_every_steps:])
                recent_grad = np.mean(grad_norms[-config.log_every_steps:])
                if config.baseline_mode in GROUP_BASELINE_MODES:
                    print(
                        f"[step {step}/{num_steps}] cost={recent_cost:.3f} "
                        f"grad_norm={recent_grad:.4f} "
                        f"group_std={metrics['mean_group_std']:.3f} "
                        f"epsilon={epsilon:.4f} mode={config.baseline_mode} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"[step {step}/{num_steps}] cost={recent_cost:.3f} "
                        f"grad_norm={recent_grad:.4f} "
                        f"baseline={metrics['baseline_used']:.3f} "
                        f"epsilon={epsilon:.4f} mode={config.baseline_mode} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )

            # Runs before the periodic save so the periodic checkpoint records the updated best.
            if track_best and (
                step % config.eval_every_steps == 0 or step == num_steps
            ):
                gap, served_all_rate, _ = evaluate_val_gap_closure(
                    scorer, val_records, config.weights
                )
                gap_str = "n/a" if gap is None else f"{gap:.4f}"
                # Full trajectory, not just best_gap/best_step, so the curve between evaluations doesn't depend on keeping the stdout log.
                eval_history.append(
                    {
                        "step": step,
                        "gap": gap,
                        "served_all_rate": served_all_rate,
                        "n_val": len(val_records),
                    }
                )
                print(
                    f"[eval step {step}/{num_steps}] hard gap_vs_milp={gap_str} "
                    f"served_all_rate={served_all_rate:.3f} "
                    f"(n_val={len(val_records)})",
                    flush=True,
                )
                meets_floor = served_all_rate >= config.best_serve_all_floor
                improved = gap is not None and (best_gap is None or gap > best_gap)
                if improved and not meets_floor:
                    print(
                        f"[eval step {step}/{num_steps}] gap {gap:.4f} improves "
                        f"but served_all_rate {served_all_rate:.3f} is below "
                        f"the floor {config.best_serve_all_floor:.2f}; not "
                        f"saving as best.",
                        flush=True,
                    )
                if improved and meets_floor:
                    best_gap = gap
                    best_step = step
                    # Separate file; never overwrites the periodic checkpoint.
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": scorer.state_dict(),
                            "config": config.__dict__,
                            "best_gap": best_gap,
                            "best_step": best_step,
                            "served_all_rate": served_all_rate,
                            "eval_history": eval_history,
                            "init_kwargs": scorer.get_init_kwargs(),
                        },
                        best_ckpt_path,
                    )
                    print(
                        f"New best gap closure {best_gap:.4f} at step {step}; "
                        f"saved best checkpoint to {best_ckpt_path}",
                        flush=True,
                    )

            # Format matches the resume block above.
            if step % config.checkpoint_every_steps == 0 or step == num_steps:
                checkpoint = {
                    "step": step,
                    "model_state_dict": scorer.state_dict(),
                    "optimizer_state_dict": optimiser.state_dict(),
                    "config": config.__dict__,
                    "baseline": baseline,
                    "cost_history": costs_history,
                    "grad_norm_history": grad_norm_history,
                    "best_gap": best_gap,
                    "best_step": best_step,
                    "eval_history": eval_history,
                    # So a resume rebuilds correctly when this run inherited a non-default architecture from its warm start.
                    "init_kwargs": scorer.get_init_kwargs(),
                }
                torch.save(checkpoint, config.checkpoint_path)
                print(
                    f"Saved checkpoint at step {step} to {config.checkpoint_path}",
                    flush=True,
                )

            # Last thing in the step, so an abort here always leaves this step's checkpoint already on disk.
            if step_callback is not None:
                step_callback(step)

    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
            # Force-reap survivors: shutdown() can return without workers exiting when the pool broke mid-run (known ARM64 spawn crash, dm21/dm23), which let crash-and-relaunch double up live pools.
            for proc in list((getattr(pool, "_processes", None) or {}).values()):
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
            print("Shut down worker pool.", flush=True)

    if track_best:
        if best_gap is None:
            print(
                "Best-checkpoint tracking on, but no validation evaluation "
                "produced a serving gap closure; no best checkpoint saved.",
                flush=True,
            )
        else:
            print(
                f"Best gap closure: {best_gap:.4f} at step {best_step} "
                f"(best checkpoint at {best_ckpt_path}).",
                flush=True,
            )

    return scorer


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Method two REINFORCE trainer (methodology section 3.4)."
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--checkpoint-path", type=str, default="checkpoints/method_two.pt"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Constant-epsilon shortcut; sets both initial and terminal.",
    )
    parser.add_argument("--epsilon-initial", type=float, default=1.0)
    parser.add_argument("--epsilon-terminal", type=float, default=0.01)
    parser.add_argument("--epsilon-anneal-steps", type=int, default=None)
    parser.add_argument(
        "--baseline-mode",
        type=str,
        default=BASELINE_RUNNING_MEAN,
        choices=list(BASELINE_MODES),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Instances per step. Defaults to 13 in grpo mode (~64 rollouts "
        "with grpo_k=5), else 64.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--warm-start-from", type=str, default=None)
    parser.add_argument("--grpo-k", type=int, default=5)
    parser.add_argument("--rloo-k", type=int, default=5)
    parser.add_argument("--checkpoint-every-steps", type=int, default=25)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="CPU rollout workers. 1 runs the sequential path verbatim; "
        ">=2 fans rollouts out to spawn worker processes.",
    )
    parser.add_argument("--rng-base-seed", type=int, default=0)
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=0,
        help="In-loop validation cadence. 0 disables best-checkpoint tracking.",
    )
    parser.add_argument("--val-cache-dir", type=str, default=None)
    parser.add_argument("--num-val-instances", type=int, default=200)
    args = parser.parse_args()

    if args.epsilon is not None:
        eps_initial = eps_terminal = args.epsilon
    else:
        eps_initial, eps_terminal = args.epsilon_initial, args.epsilon_terminal

    if args.batch_size is not None:
        batch_size = args.batch_size
    elif args.baseline_mode in GROUP_BASELINE_MODES:
        batch_size = 13  # ~64 rollouts with group K=5
    else:
        batch_size = 64

    print(
        f"Starting method two training with {args.steps} steps, "
        f"epsilon {eps_initial}->{eps_terminal}, baseline={args.baseline_mode}",
        flush=True,
    )
    config = MethodTwoConfig(
        log_every_steps=args.log_every,
        checkpoint_path=args.checkpoint_path,
        epsilon_initial=eps_initial,
        epsilon_terminal=eps_terminal,
        epsilon_anneal_steps=args.epsilon_anneal_steps,
        baseline_mode=args.baseline_mode,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        gradient_clip_norm=args.gradient_clip_norm,
        warm_start_from=args.warm_start_from,
        grpo_k=args.grpo_k,
        rloo_k=args.rloo_k,
        checkpoint_every_steps=args.checkpoint_every_steps,
        resume_from=args.resume_from,
        num_workers=args.num_workers,
        rng_base_seed=args.rng_base_seed,
        eval_every_steps=args.eval_every_steps,
        val_cache_dir=args.val_cache_dir,
        num_val_instances=args.num_val_instances,
    )
    train(config, args.steps)


if __name__ == "__main__":
    main()
