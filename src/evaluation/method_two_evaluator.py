"""Evaluation harness for method two on the validation set.

Mirrors ``method_one_evaluator``, same decode and same gap-closure definition, so the
two are directly comparable, but evaluates both the hard Hungarian decode and a
perturbed decode at eps=1.0 per instance.

    gap closure = (greedy_cost - policy_cost) / (greedy_cost - milp_objective)

Validation split only. A guard aborts if the resolved cache directory points at a test
split, which is held in reserve.

Success for an inference mode means the policy committed at least one task, which is a
liveness check rather than method one's serve-all notion. A stricter served_all flag is
recorded per instance so the two are comparable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch

from instances.synthetic_generator import SyntheticInstance
from baselines.bipartite_policies import (
    run_greedy_policy,
    run_hungarian_kappa_policy,
)
from evaluation.method_one_evaluator import (
    gap_closure,
    load_scorer_from_checkpoint,
    replay_expert,
    rollout_failed,
    rollout_policy,
)


DEFAULT_VAL_CACHE_DIR = Path("cache/training_set_il_v3/val")


def load_records(cache_dir, num_instances: int) -> List[dict]:
    """Load up to ``num_instances`` ``seed*.json`` records in seed order."""
    paths = sorted(Path(cache_dir).glob("seed*.json"))[:num_instances]
    records = []
    for p in paths:
        with p.open("r") as f:
            records.append(json.load(f))
    return records


def _terms(breakdown) -> dict:
    """Raw objective terms off an already-built CostBreakdown."""
    return {
        "term_travel_time": breakdown.distance,
        "term_makespan": breakdown.makespan,
        "term_balance": breakdown.imbalance,
    }


def _sim_detail(sim, breakdown, weights=None) -> dict:
    """Read already-computed per-rollout quantities off a terminal simulator.

    Purely a reader. Nothing here re-runs a rollout, re-solves anything, or
    recomputes a cost: the objective terms come off the ``CostBreakdown`` the
    caller already built, and the rest are fields the simulator recorded at
    termination.

    ``breakdown`` is None for a non-serving rollout. The aggregate path
    deliberately discards such a rollout's cost, which means a policy that
    fails on hard instances is otherwise scored on a self-selected easier
    subset. For recording only, and only when ``weights`` is supplied, the cost
    is computed here so that instance still appears in the CSV with its cost
    and a false serve-all flag rather than vanishing. Rows so recorded carry
    ``cost_is_from_failed_rollout`` true. This never feeds an aggregate.

    ``sim_time_at_termination`` is the simulator's own wall clock at
    termination, in simulator time units, which is the quantity compared
    against ``sim.wall_clock_cap``. It is NOT real elapsed compute time, and
    anything needing measured compute time must not read it.
    """
    traj = sim.trajectory
    counts = [0] * int(sim.R)
    for c in traj.commitments:
        counts[int(c["robot_id"])] += 1
    failed_cost = None
    if breakdown is None and weights is not None:
        breakdown = sim.compute_cost(weights)
        failed_cost = float(breakdown.combined)
    terms = (
        {"term_travel_time": None, "term_makespan": None, "term_balance": None}
        if breakdown is None else _terms(breakdown)
    )
    return {
        **terms,
        "cost_including_failed": failed_cost,
        "cost_is_from_failed_rollout": failed_cost is not None,
        "n_unserved_tasks": len(traj.unserved_task_ids),
        "sim_time_at_termination": traj.wall_clock_at_termination,
        "epoch_count": traj.epochs_run,
        "per_robot_task_counts": ";".join(str(x) for x in counts),
        "simulator_failure": bool(traj.simulator_failure),
    }


def _eval_mode(scorer, instance, weights, epsilon, n_samples, detail_sink=None):
    """Roll out a mode ``n_samples`` times; return (cost, success, served_all,
    n_served_samples).

    ``cost`` is the mean cost over the samples but only when EVERY sample served
    all tasks (so the average is over robustly-serving rollouts, mirroring the
    failure-exclusion rule); otherwise ``cost`` is None. ``success`` is whether
    any sample committed at least one task. For ``n_samples == 1`` this reduces
    exactly to single-rollout behaviour.

    ``detail_sink``, when given, is a dict populated in place with the last
    sample's per-rollout detail (see ``_sim_detail``). It is a pure observer:
    it adds no rollout, draws no random number, and does not alter ``cost``.
    With the production ``n_samples == 1`` the last sample is the only sample.
    """
    costs, served, committed = [], [], []
    for _ in range(n_samples):
        sim = rollout_policy(scorer, instance, weights, epsilon=epsilon)
        served_all = not rollout_failed(sim)
        served.append(served_all)
        committed.append(len(sim.trajectory.commitments) > 0)
        # Same call/arguments/position as before the recorder was added, so the value flowing into `costs` is unchanged.
        breakdown = sim.compute_cost(weights) if served_all else None
        costs.append(
            float(breakdown.combined) if served_all else None
        )
        if detail_sink is not None:
            detail_sink.clear()
            detail_sink.update(_sim_detail(sim, breakdown, weights))
    success = any(committed)
    all_served = all(served)
    cost = float(np.mean(costs)) if all_served else None
    return cost, success, all_served, int(sum(served))


def evaluate_one_instance(
    scorer,
    cached_record: dict,
    modes: List[str],
    weights: dict,
    epsilon: float,
    num_perturbed_samples: int = 1,
    collect_detail: bool = False,
) -> dict:
    """Evaluate one instance under each requested inference mode.

    ``collect_detail`` is additive and defaults off. When on, the record also
    carries a ``{mode}_detail`` dict per mode and a ``kappa`` cost, neither of
    which is read by ``summarize`` or ``_print_report``, so every reported
    aggregate is unchanged. The extra kappa rollout is a deterministic
    numpy/scipy Hungarian policy that consumes no torch RNG, so the perturbed
    decode stream is unaffected.
    """
    instance = SyntheticInstance.from_dict(cached_record["instance"])

    # Same call/argument/position as before the recorder was added, so the value flowing into `greedy` is unchanged.
    _sim_greedy = run_greedy_policy(instance)
    _cb_greedy = _sim_greedy.compute_cost(weights)
    greedy = float(_cb_greedy.combined)

    # A transfer-scale cache has no MILP/expert reference; both are optional and resolve to None, propagating to a None gap closure rather than a fabricated number.
    _milp_sol = cached_record.get("milp_solution")
    milp = (
        float(_milp_sol["objective_value"])
        if isinstance(_milp_sol, dict) and _milp_sol.get("objective_value") is not None
        else None
    )
    if cached_record.get("expert_decisions") is not None:
        expert_decisions = [
            [(int(r), int(j)) for (r, j) in epoch]
            for epoch in cached_record["expert_decisions"]
        ]
        expert = float(
            replay_expert(instance, expert_decisions).compute_cost(weights).combined
        )
    else:
        expert = None

    record = {
        "seed": int(cached_record.get("seed", -1)),
        "greedy": greedy,
        "milp": milp,
        "expert": expert,
    }

    for mode in modes:
        # Hard inference is deterministic (one sample suffices); only perturbed inference benefits from averaging across samples.
        mode_eps = 0.0 if mode == "hard" else epsilon
        n_samples = 1 if mode == "hard" else num_perturbed_samples

        detail_sink = {} if collect_detail else None
        cost, success, served_all, n_served = _eval_mode(
            scorer, instance, weights, mode_eps, n_samples, detail_sink
        )
        if collect_detail:
            # Same loop iteration that produced the rollout, so the detail matches its seed and decode mode.
            record[f"{mode}_detail"] = detail_sink

        record[f"{mode}_success"] = success
        record[f"{mode}_served_all"] = served_all
        record[f"{mode}_n_samples"] = n_samples
        record[f"{mode}_n_served_samples"] = n_served
        # cost is None unless every sample served all tasks, excluding failed/partial rollouts from the cost and gap-closure aggregates.
        record[f"{mode}_cost"] = cost
        # Either a serving cost or a reference missing leaves this None rather than fabricating a denominator.
        record[f"{mode}_gap_vs_expert"] = (
            None if cost is None or expert is None
            else gap_closure(greedy, cost, expert)
        )
        record[f"{mode}_gap_vs_milp"] = (
            None if cost is None or milp is None
            else gap_closure(greedy, cost, milp)
        )

    if collect_detail:
        # Deferred to after the mode loop so it runs after every pre-existing operation, in the same order; deterministic and torch-RNG-free.
        _sim_kappa = run_hungarian_kappa_policy(instance, weights)
        _cb_kappa = _sim_kappa.compute_cost(weights)
        record["kappa"] = float(_cb_kappa.combined)
        # A baseline can fail the same way the policy can, so these flags travel with the costs to avoid comparing against a baseline that dropped tasks.
        record["greedy_served_all"] = not rollout_failed(_sim_greedy)
        record["greedy_n_unserved"] = len(_sim_greedy.trajectory.unserved_task_ids)
        record["kappa_served_all"] = not rollout_failed(_sim_kappa)
        record["kappa_n_unserved"] = len(_sim_kappa.trajectory.unserved_task_ids)
        record["greedy_terms"] = _terms(_cb_greedy)
        record["kappa_terms"] = _terms(_cb_kappa)

    return record


def _stats(values: List[float]) -> dict:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None, "p25": None, "p50": None,
                "p75": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "n": int(arr.size),
    }


def summarize(records: List[dict], modes: List[str]) -> dict:
    """Aggregate, excluding non-serving rollouts from cost/gap statistics.

    Cost and gap-closure stats for a mode are computed only over serving
    instances (those with a non-null cost). Success and served-all counts are
    reported as primary metrics. Reference costs (greedy/MILP/expert) are over
    all evaluated instances.
    """
    n = len(records)

    def _mean_or_none(key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "n_evaluated": n,
        "mean_greedy": float(np.mean([r["greedy"] for r in records])),
        "mean_milp": _mean_or_none("milp"),
        "mean_expert": _mean_or_none("expert"),
        "modes": {},
    }
    for mode in modes:
        n_success = int(sum(r[f"{mode}_success"] for r in records))
        n_served = int(sum(r[f"{mode}_served_all"] for r in records))
        summary["modes"][mode] = {
            "n_success": n_success,
            "success_rate": (n_success / n) if n else 0.0,
            "n_served_all": n_served,
            "served_all_rate": (n_served / n) if n else 0.0,
            # _stats drops None entries, so these are over serving instances.
            "serving_cost": _stats([r[f"{mode}_cost"] for r in records]),
            "serving_gap_vs_milp": _stats(
                [r[f"{mode}_gap_vs_milp"] for r in records]
            ),
            "serving_gap_vs_expert": _stats(
                [r[f"{mode}_gap_vs_expert"] for r in records]
            ),
        }
    return summary


def _fmt(x, prec=3):
    return "n/a" if x is None else f"{x:.{prec}f}"


def _print_report(summary: dict, records: List[dict], modes: List[str]) -> None:
    n = summary["n_evaluated"]
    print("=" * 70, flush=True)
    print("VALIDATION SET", flush=True)
    print("=" * 70, flush=True)
    print(
        f"Instances evaluated: {n}  "
        f"mean greedy={summary['mean_greedy']:.3f}  "
        f"mean expert replay (ceiling)={_fmt(summary['mean_expert'])}  "
        f"mean MILP (lower bound)={_fmt(summary['mean_milp'])}",
        flush=True,
    )
    for mode in modes:
        m = summary["modes"][mode]
        c = m["serving_cost"]
        gm = m["serving_gap_vs_milp"]
        ge = m["serving_gap_vs_expert"]
        print(f"\n{mode.capitalize()} inference:", flush=True)
        print(
            f"  Success rate (any commit): {m['n_success']}/{n} "
            f"({m['success_rate']:.3f})",
            flush=True,
        )
        print(
            f"  Served-all rate: {m['n_served_all']}/{n} "
            f"({m['served_all_rate']:.3f})",
            flush=True,
        )
        if c["n"] == 0:
            print("  Serving-only cost (n=0): no serving rollouts", flush=True)
            continue
        print(
            f"  Serving-only cost (n={c['n']}): mean={_fmt(c['mean'])}, "
            f"std={_fmt(c['std'])}, p25={_fmt(c['p25'])}, "
            f"p50={_fmt(c['p50'])}, p75={_fmt(c['p75'])}",
            flush=True,
        )
        print(
            f"  Serving-only gap closure vs expert replay (ceiling, n={ge['n']}): "
            f"mean={_fmt(ge['mean'])}, std={_fmt(ge['std'])}",
            flush=True,
        )
        print(
            "  Note: gap closure > 1.0 is possible and expected. The expert"
            " replay ceiling is achievable but not epoch-constrained-optimal.",
            flush=True,
        )
        print(
            f"  Serving-only gap closure vs MILP lower bound (n={gm['n']}): "
            f"mean={_fmt(gm['mean'])}, std={_fmt(gm['std'])}",
            flush=True,
        )

    if "hard" in modes:
        serving = [r for r in records if r.get("hard_cost") is not None]
        if serving:
            ranked = sorted(serving, key=lambda r: r["hard_cost"])
            print(
                f"\nBest 3 by hard cost (of {len(serving)} serving):",
                flush=True,
            )
            for r in ranked[:3]:
                print(
                    f"  seed={r['seed']} hard_cost={r['hard_cost']:.3f} "
                    f"greedy={r['greedy']:.3f}",
                    flush=True,
                )
            print("Worst 3 by hard cost:", flush=True)
            for r in ranked[-3:]:
                print(
                    f"  seed={r['seed']} hard_cost={r['hard_cost']:.3f} "
                    f"greedy={r['greedy']:.3f}",
                    flush=True,
                )
        else:
            print("\nNo serving hard rollouts to rank.", flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Method two evaluator (VALIDATION set only)."
    )
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument(
        "--val-cache-dir", type=str, default=str(DEFAULT_VAL_CACHE_DIR)
    )
    parser.add_argument("--num-instances", type=int, default=200)
    parser.add_argument("--inference-modes", type=str, default="hard,perturbed")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="torch.manual_seed before evaluation for reproducible perturbed "
        "inference. Hard inference is deterministic regardless.",
    )
    parser.add_argument(
        "--num-perturbed-samples",
        type=int,
        default=1,
        help="Perturbed rollouts per instance, averaged over (only when all "
        "samples serve all tasks) for lower-variance perturbed cost.",
    )
    parser.add_argument(
        "--weights", type=float, nargs=3,
        default=[0.0637, 0.2398, 0.6965],
        metavar=("W_DIST", "W_MAKE", "W_BAL"),
    )
    args = parser.parse_args()

    cache_dir = Path(args.val_cache_dir)
    # Hard guard: never touch the held-out test split.
    if any(part.lower() == "test" for part in cache_dir.parts):
        raise SystemExit(
            f"REFUSING to evaluate on a test split ({cache_dir}). The test set "
            "is reserved for the final thesis. Use the validation split."
        )

    modes = [m.strip() for m in args.inference_modes.split(",") if m.strip()]
    for m in modes:
        if m not in ("hard", "perturbed"):
            raise SystemExit(f"Unknown inference mode: {m}")
    weights = {
        "w_dist": args.weights[0],
        "w_make": args.weights[1],
        "w_bal": args.weights[2],
    }

    print(f"Loading scorer from {args.checkpoint_path}", flush=True)
    scorer = load_scorer_from_checkpoint(args.checkpoint_path)
    # Seed the Gumbel draws so perturbed inference is reproducible run-to-run.
    torch.manual_seed(args.seed)
    print(
        f"Evaluating VALIDATION split at {cache_dir} "
        f"(modes={modes}, epsilon={args.epsilon}, seed={args.seed}, "
        f"num_perturbed_samples={args.num_perturbed_samples})",
        flush=True,
    )

    records = load_records(cache_dir, args.num_instances)
    per_instance = []
    for i, rec in enumerate(records):
        result = evaluate_one_instance(
            scorer, rec, modes, weights, args.epsilon,
            num_perturbed_samples=args.num_perturbed_samples,
        )
        per_instance.append(result)
        hard = result.get("hard_cost")
        pert = result.get("perturbed_cost")
        print(
            f"[{i + 1}/{len(records)}] seed={result['seed']} "
            f"greedy={result['greedy']:.2f} "
            f"expert={_fmt(result['expert'], 2)} "
            f"milp_lb={_fmt(result['milp'], 2)} "
            f"hard={'FAIL' if hard is None else f'{hard:.2f}'} "
            f"perturbed={'FAIL' if pert is None else f'{pert:.2f}'}",
            flush=True,
        )

    summary = summarize(per_instance, modes)
    _print_report(summary, per_instance, modes)

    ckpt_name = Path(args.checkpoint_path).stem
    out_path = Path("logs") / f"method_two_evaluation_{ckpt_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint_path,
                "split": str(cache_dir),
                "modes": modes,
                "epsilon": args.epsilon,
                "seed": args.seed,
                "num_perturbed_samples": args.num_perturbed_samples,
                "weights": weights,
                "summary": summary,
                "per_instance": per_instance,
            },
            f,
            indent=2,
        )
    print(f"\nWrote per-instance records to {out_path}", flush=True)


if __name__ == "__main__":
    main()
