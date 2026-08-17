"""Evaluation harness for method one.

Rolls the trained GNN scorer out against the cached split with an exact Hungarian
decode at inference, then scores the trajectory against the distance-only Hungarian
floor and the anticipative MILP objective.

    gap closure = (greedy_cost - policy_cost) / (greedy_cost - milp_objective)

The denominator is the interval between the myopic floor and the anticipative
benchmark. Closure is returned as None when it falls below 1e-9, which excludes
records whose stored bound a feasible schedule undercuts.

The decode takes a hard Hungarian assignment on the doubly augmented score matrix,
reads the action from the top-left R by T block, and keeps only commits whose robot
is available and whose task is pending.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from instances.synthetic_generator import SyntheticInstance
from simulator.dynamic_simulator import DynamicSimulator, SimulatorState
from scoring.gnn_scorer import GNNScorer
from baselines.bipartite_policies import run_greedy_policy


DEFAULT_WEIGHTS = {"w_dist": 0.0637, "w_make": 0.2398, "w_bal": 0.6965}
# Test split is reserved for the final thesis evaluation and must be requested explicitly.
DEFAULT_EVAL_CACHE_DIR = Path("cache/training_set_il_v3/val")

_SOURCES = ("policy", "greedy", "milp", "replayed_expert")
_TERMS = ("distance", "makespan", "imbalance")
_GAP_DENOM_TOL = 1e-9


def _decode_action(
    theta: torch.Tensor,
    state: SimulatorState,
    R: int,
    T: int,
    epsilon: float = 0.0,
    samples: int = 1,
) -> List[Tuple[int, int]]:
    """Decode one dispatcher action from the augmented matrix ``Theta``.

    ``epsilon == 0`` is plain hard Hungarian (the original behaviour). When
    ``epsilon > 0`` each draw adds Gumbel(0, 1) noise scaled by ``epsilon``
    before solving Hungarian, the same perturbed argmax used in method one's
    training and in method two. With ``samples > 1`` the per-draw actions are
    taken by majority vote per (r, c) commit (a reduced-variance soft policy);
    a commit is kept when it appears in more than half the draws.
    """
    def _one_draw() -> set:
        matrix = theta
        if epsilon > 0.0:
            N = theta.shape[0]
            U = torch.rand(N, N, device=theta.device).clamp(1e-10, 1 - 1e-10)
            gumbel = -torch.log(-torch.log(U))
            matrix = theta + epsilon * gumbel
        cost_matrix = (-matrix).detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        picks = set()
        for r, c in zip(row_ind, col_ind):
            r, c = int(r), int(c)
            if r < R and c < T:
                if r in state.available_robots and c in state.pending_tasks:
                    picks.add((r, c))
        return picks

    if epsilon <= 0.0 or samples <= 1:
        return sorted(_one_draw())

    counts: dict = {}
    for _ in range(samples):
        for commit in _one_draw():
            counts[commit] = counts.get(commit, 0) + 1
    threshold = samples / 2.0
    # Keep majority commits only, dropping any that would double-book a robot/task already used by a higher-vote commit.
    voted = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    used_robots: set = set()
    used_tasks: set = set()
    action: List[Tuple[int, int]] = []
    for (r, c), n in voted:
        if n <= threshold:
            continue
        if r in used_robots or c in used_tasks:
            continue
        used_robots.add(r)
        used_tasks.add(c)
        action.append((r, c))
    return sorted(action)


def rollout_policy(
    scorer: GNNScorer,
    instance: SyntheticInstance,
    weights: Optional[dict] = None,
    epsilon: float = 0.0,
    samples: int = 1,
) -> DynamicSimulator:
    """Roll the trained GNN policy out on one instance to termination.

    At each epoch the scorer produces the doubly augmented matrix ``Theta`` and
    ``_decode_action`` selects the dispatcher action (restricted to available
    robots and pending tasks). With ``epsilon == 0`` this is hard Hungarian
    (the default); ``epsilon > 0`` uses perturbed Hungarian. Returns the
    terminal simulator. ``weights`` is unused during rollout (cost is scored
    afterwards) and accepted only for signature symmetry.
    """
    sim = DynamicSimulator(instance)
    scorer.eval()
    with torch.no_grad():
        while not sim.is_terminal:
            state = sim.state
            theta = scorer(state, instance)
            commits = _decode_action(
                theta, state, instance.R, instance.T,
                epsilon=epsilon, samples=samples,
            )
            sim.step(commits)
    return sim


def rollout_failed(sim: DynamicSimulator) -> bool:
    """Whether a terminal rollout failed to serve every serviceable task.

    A rollout is a failure if the simulator hit its wall-clock cap, if any
    task is still pending at termination, or if any serviceable task went
    unserved. In all of these cases the trajectory cost is meaningless (a
    non-serving rollout scores artificially low), so the policy cost must be
    discarded rather than reported as a strong result.
    """
    return bool(
        sim.trajectory.simulator_failure
        or sim.state.pending_tasks
        or sim.trajectory.unserved_task_ids
    )


def replay_expert(
    instance: SyntheticInstance, expert_decisions: List[List[Tuple[int, int]]]
) -> DynamicSimulator:
    """Replay the cached anticipative decisions through the Path B simulator.

    Steps each epoch's commitments in order, then drains with empty actions
    until termination. Returns the terminal simulator.
    """
    sim = DynamicSimulator(instance)
    for action in expert_decisions:
        if sim.is_terminal:
            break
        sim.step([(int(r), int(j)) for (r, j) in action])
    while not sim.is_terminal:
        sim.step([])
    return sim


def gap_closure(
    greedy_cost: float, policy_cost: float, ceiling_cost: float
) -> Optional[float]:
    """Fraction of the greedy-to-ceiling gap closed by the policy.

    Returns ``None`` when the denominator ``greedy_cost - ceiling_cost`` is
    below ``1e-9`` (the floor already sits at the ceiling).
    """
    denom = greedy_cost - ceiling_cost
    if denom < _GAP_DENOM_TOL:
        return None
    return (greedy_cost - policy_cost) / denom


def evaluate_one_instance(
    scorer: GNNScorer,
    cached_record: dict,
    weights: Optional[dict] = None,
    epsilon: float = 0.0,
    samples: int = 1,
) -> dict:
    """Evaluate the policy on one cached instance against all reference points.

    Reconstructs the instance from the cache, rolls out the policy and greedy,
    reads the MILP objective from the cache, and replays the cached expert
    decisions. ``epsilon``/``samples`` control the policy decoder (see
    ``rollout_policy``). Returns a flat per-instance metrics dict.
    """
    weights = weights if weights is not None else DEFAULT_WEIGHTS
    instance = SyntheticInstance.from_dict(cached_record["instance"])

    sim_policy = rollout_policy(
        scorer, instance, weights, epsilon=epsilon, samples=samples
    )
    policy_failed = rollout_failed(sim_policy)

    sim_greedy = run_greedy_policy(instance)
    gc = sim_greedy.compute_cost(weights)

    ms = cached_record["milp_solution"]
    milp_cost = float(ms["objective_value"])

    expert_decisions = [
        [(int(r), int(j)) for (r, j) in epoch]
        for epoch in cached_record["expert_decisions"]
    ]
    sim_expert = replay_expert(instance, expert_decisions)
    ec = sim_expert.compute_cost(weights)

    result = {
        "seed": int(cached_record.get("seed", -1)),
        "greedy_cost": gc.combined,
        "greedy_distance": gc.distance,
        "greedy_makespan": gc.makespan,
        "greedy_imbalance": gc.imbalance,
        "milp_cost": milp_cost,
        "milp_distance": float(ms["distance"]),
        "milp_makespan": float(ms["makespan"]),
        "milp_imbalance": float(ms["imbalance"]),
        "replayed_expert_cost": ec.combined,
        "replayed_expert_distance": ec.distance,
        "replayed_expert_makespan": ec.makespan,
        "replayed_expert_imbalance": ec.imbalance,
        "mip_gap": float(ms.get("mip_gap", float("nan"))),
        "policy_failed": policy_failed,
        "policy_simulator_failure": bool(
            sim_policy.trajectory.simulator_failure
        ),
        "greedy_simulator_failure": bool(
            sim_greedy.trajectory.simulator_failure
        ),
    }

    if policy_failed:
        # A failed rollout serves fewer tasks and scores an artificially low cost, so discard it entirely.
        result["policy_cost"] = None
        result["policy_distance"] = None
        result["policy_makespan"] = None
        result["policy_imbalance"] = None
        result["gap_closure_vs_expert"] = None
        result["gap_closure_vs_milp"] = None
    else:
        pc = sim_policy.compute_cost(weights)
        result["policy_cost"] = pc.combined
        result["policy_distance"] = pc.distance
        result["policy_makespan"] = pc.makespan
        result["policy_imbalance"] = pc.imbalance
        # Primary: expert replay is the achievable ceiling; values above 1.0 are possible since it is not epoch-constrained-optimal.
        result["gap_closure_vs_expert"] = gap_closure(
            gc.combined, pc.combined, ec.combined
        )
        # Secondary: MILP ignores epoch-boundary idle-wait cost, so it underestimates achievable cost and is not the gap ceiling.
        result["gap_closure_vs_milp"] = gap_closure(
            gc.combined, pc.combined, milp_cost
        )
    return result


def load_cached_records(
    cache_dir=DEFAULT_EVAL_CACHE_DIR, max_instances: Optional[int] = None
) -> List[dict]:
    """Load ``seed*.json`` records from ``cache_dir`` in seed order."""
    cache_dir = Path(cache_dir)
    paths = sorted(cache_dir.glob("seed*.json"))
    if max_instances is not None:
        paths = paths[:max_instances]
    records = []
    for p in paths:
        with p.open("r") as f:
            records.append(json.load(f))
    return records


def _dist_stats(values: List[Optional[float]]) -> dict:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None, "p25": None, "p75": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "n": int(arr.size),
    }


def _gap_closure_stats(values: List[Optional[float]]) -> dict:
    arr = np.array([v for v in values if v is not None], dtype=float)
    stats = _dist_stats(values)
    stats["count_positive"] = int((arr > 0.0).sum()) if arr.size else 0
    stats["count_above_0_30"] = int((arr > 0.30).sum()) if arr.size else 0
    return stats


def summarize(per_instance: List[dict]) -> dict:
    """Aggregate per-instance metrics into per-split summary statistics.

    Failed policy rollouts (``policy_failed``) are excluded from every policy
    cost and gap-closure statistic, since their cost is meaningless. The
    greedy, MILP and replayed-expert reference costs are reported over all
    evaluated instances. The failure count is itself a primary metric.
    """
    n_evaluated = len(per_instance)
    n_failures = sum(1 for r in per_instance if r.get("policy_failed"))
    succeeded = [r for r in per_instance if not r.get("policy_failed")]

    def _rows(src):
        # Policy stats only over succeeded rollouts; references over all.
        return succeeded if src == "policy" else per_instance

    combined_cost = {
        src: _dist_stats([r[f"{src}_cost"] for r in _rows(src)])
        for src in _SOURCES
    }

    per_term = {}
    for term in _TERMS:
        block = {
            src: _dist_stats([r[f"{src}_{term}"] for r in _rows(src)])
            for src in _SOURCES
        }
        block["gap_closure_vs_milp"] = _gap_closure_stats(
            [
                gap_closure(
                    r[f"greedy_{term}"], r[f"policy_{term}"], r[f"milp_{term}"]
                )
                for r in succeeded
            ]
        )
        block["gap_closure_vs_expert"] = _gap_closure_stats(
            [
                gap_closure(
                    r[f"greedy_{term}"],
                    r[f"policy_{term}"],
                    r[f"replayed_expert_{term}"],
                )
                for r in succeeded
            ]
        )
        per_term[term] = block

    return {
        "n_instances_evaluated": n_evaluated,
        "n_policy_failures": n_failures,
        "n_instances_succeeded": n_evaluated - n_failures,
        # Backward-compatible alias for the evaluated total.
        "n_instances": n_evaluated,
        "n_greedy_failures": sum(
            1 for r in per_instance if r.get("greedy_simulator_failure")
        ),
        "combined_cost": combined_cost,
        "gap_closure_vs_milp": _gap_closure_stats(
            [r["gap_closure_vs_milp"] for r in succeeded]
        ),
        "gap_closure_vs_expert": _gap_closure_stats(
            [r["gap_closure_vs_expert"] for r in succeeded]
        ),
        "per_term": per_term,
    }


def evaluate_split(
    scorer: GNNScorer,
    cache_dir=DEFAULT_EVAL_CACHE_DIR,
    weights: Optional[dict] = None,
    max_instances: Optional[int] = None,
    verbose: bool = False,
    epsilon: float = 0.0,
    samples: int = 1,
) -> dict:
    """Evaluate every cached instance in a split and summarise the results."""
    weights = weights if weights is not None else DEFAULT_WEIGHTS
    records = load_cached_records(cache_dir, max_instances=max_instances)
    per_instance = []
    for i, record in enumerate(records):
        result = evaluate_one_instance(
            scorer, record, weights, epsilon=epsilon, samples=samples
        )
        per_instance.append(result)
        if verbose:
            gcm = result["gap_closure_vs_milp"]
            gce = result["gap_closure_vs_expert"]
            pc = result["policy_cost"]
            policy_str = "FAILED" if result["policy_failed"] else f"{pc:.3f}"
            print(
                f"[{i + 1}/{len(records)}] seed={result['seed']} "
                f"policy={policy_str} "
                f"greedy={result['greedy_cost']:.3f} "
                f"expert={result['replayed_expert_cost']:.3f} "
                f"gap_vs_expert={'n/a' if gce is None else f'{gce:.3f}'} "
                f"milp_lb={result['milp_cost']:.3f} "
                f"gap_vs_milp_lb={'n/a' if gcm is None else f'{gcm:.3f}'}",
                flush=True,
            )
    return {"per_instance": per_instance, "summary": summarize(per_instance)}


def load_scorer_from_checkpoint(checkpoint_path, **scorer_kwargs) -> GNNScorer:
    """Load a ``GNNScorer`` from a checkpoint written by the IL trainer."""
    scorer = GNNScorer(**scorer_kwargs)
    ckpt = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    scorer.load_state_dict(ckpt["model_state_dict"])
    scorer.eval()
    return scorer


def _print_summary(summary: dict) -> None:
    n_eval = summary["n_instances_evaluated"]
    n_fail = summary["n_policy_failures"]
    n_ok = summary["n_instances_succeeded"]
    print(
        f"Policy failures: {n_fail}/{n_eval} "
        f"(succeeded: {n_ok}/{n_eval})",
        flush=True,
    )

    if n_eval > 0 and n_fail == n_eval:
        print(
            "Policy failed on all instances. Training did not produce a "
            "serving policy.",
            flush=True,
        )
        return

    cc = summary["combined_cost"]
    for src in _SOURCES:
        s = cc[src]
        if s["mean"] is None:
            continue
        print(
            f"  {src:16s} combined cost (n={s['n']}) mean={s['mean']:.3f} "
            f"std={s['std']:.3f} p25={s['p25']:.3f} p75={s['p75']:.3f}",
            flush=True,
        )
    print(
        "  Note: gap closure > 1.0 is possible and expected. The expert"
        " replay ceiling is achievable but not epoch-constrained-optimal;",
        flush=True,
    )
    print(
        "  a well-timed online policy can exceed it on individual instances.",
        flush=True,
    )
    for label, key in (
        ("vs expert replay (ceiling)", "gap_closure_vs_expert"),
        ("vs MILP (lower bound)",      "gap_closure_vs_milp"),
    ):
        g = summary[key]
        mean = "n/a" if g["mean"] is None else f"{g['mean']:.3f}"
        print(
            f"  gap closure {label}: mean={mean} n={g['n']} "
            f">0={g['count_positive']} >0.30={g['count_above_0_30']}",
            flush=True,
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate method one against a cached split (default val; "
        "the test split is reserved for the final thesis)."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/il_method_one.pt"
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Overrides the cache directory derived from --split.",
    )
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument(
        "--decode-epsilon",
        type=float,
        default=0.0,
        help="Gumbel perturbation magnitude for decoding. 0.0 = hard Hungarian.",
    )
    parser.add_argument(
        "--decode-samples",
        type=int,
        default=1,
        help="Gumbel draws per state, majority-voted, when --decode-epsilon>0.",
    )
    args = parser.parse_args()

    cache_dir = (
        args.cache_dir
        if args.cache_dir
        else f"cache/training_set_il_v3/{args.split}"
    )

    # Mirrors the refusal guards in method_two_evaluator and il_trainer.
    if any(part.lower() == "test" for part in Path(cache_dir).parts):
        raise SystemExit(
            f"REFUSING to evaluate on a test split ({cache_dir}). "
            f"The test set is reserved for the final thesis."
        )

    torch.manual_seed(0)
    print(f"Loading scorer from {args.checkpoint}", flush=True)
    scorer = load_scorer_from_checkpoint(args.checkpoint)
    print(
        f"Evaluating split at {cache_dir} "
        f"(decode_epsilon={args.decode_epsilon}, "
        f"decode_samples={args.decode_samples})",
        flush=True,
    )
    out = evaluate_split(
        scorer,
        cache_dir=cache_dir,
        max_instances=args.max_instances,
        verbose=True,
        epsilon=args.decode_epsilon,
        samples=args.decode_samples,
    )
    _print_summary(out["summary"])


if __name__ == "__main__":
    main()
