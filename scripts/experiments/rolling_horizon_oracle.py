"""Rolling-horizon IL cache generator (conference-paper extension).

Generates an imitation-learning training cache in the exact ``seed*.json``
format consumed by ``ILDataset`` (see ``src/training/il_trainer.py``), but
uses a rolling-horizon h-step Gurobi solve as the oracle instead of the
offline anticipative MILP used by ``expert_dataset_generator.py``.

For each instance the rolling-horizon policy from
``scripts/experiments/rolling_horizon_baseline.py`` is run to termination. The per-epoch
state and the commits executed at that epoch are recorded as
``expert_trajectory`` and ``expert_decisions`` in the same layout the
anticipative generator produces:

    expert_decisions[epoch]   list of [robot_idx, task_idx] commits at epoch
    expert_trajectory[epoch]  serialised simulator state at the START of epoch
                              (one entry longer than expert_decisions, exactly
                              as extract_expert_decisions emits it)

There is no anticipative solve, so ``milp_solution`` is set to ``None`` and a
top-level ``oracle_type`` field (``rolling_horizon_h{h}``) marks the record so
it is never confused with an anticipative cache entry.

This script imports the locked helpers from the existing modules and does not
modify any of them:
    rolling_horizon_baseline: solve_rolling_window, _window_commits
    expert_dataset_generator: _serialize_instance, _serialize_state_record
    bipartite_policies:       build_kappa_cost_matrix, hungarian_action

Run from the repo root with src/ on PYTHONPATH, in the Gurobi env, e.g.::

    PYTHONPATH="$PWD/src" python scripts/experiments/rolling_horizon_oracle.py \
        --output-dir cache/rolling_horizon_h5/train --num-instances 1000 \
        --n-robots 6 --n-tasks 18 --horizon 5 --time-limit 30 --num-workers 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np

# scripts/ is sys.path[0] when run directly, so the sibling rolling_horizon_baseline module imports without packaging.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rolling_horizon_baseline import (  # noqa: E402
    solve_rolling_window,
    _window_commits,
    _served_all,
)
from baselines.bipartite_policies import (  # noqa: E402
    build_kappa_cost_matrix,
    hungarian_action,
)
from instances.synthetic_generator import generate_instance  # noqa: E402
from simulator.dynamic_simulator import DynamicSimulator  # noqa: E402
from training.expert_dataset_generator import (  # noqa: E402
    _serialize_instance,
    _serialize_state_record,
)


LOCKED_WEIGHTS = {"w_dist": 0.0637, "w_make": 0.2398, "w_bal": 0.6965}


def run_rolling_horizon_with_trace(
    instance,
    weights: dict,
    horizon: int,
    time_limit_seconds: int,
    seed: int,
) -> Tuple[list, list, int, int, bool]:
    """Run the rolling-horizon policy, recording the per-epoch IL trace.

    Mirrors the loop in ``rolling_horizon_baseline.run_rolling_horizon_policy``
    but captures, at every epoch, the start-of-epoch state and the executed
    commits in the ``ILDataset`` layout.

    Returns ``(expert_decisions, expert_trajectory, n_solves, n_fallbacks,
    served_all)``. ``expert_trajectory`` is one entry longer than
    ``expert_decisions`` (the terminal state) and both are padded to at least
    ``H`` epochs, matching ``extract_expert_decisions``.
    """
    H = int(instance.H)
    R = int(instance.R)
    sim = DynamicSimulator(instance)
    serviceable: Set[int] = {
        int(t["id"])
        for t in instance.tasks
        if 0 <= int(t["release_epoch"]) < H
    }
    n_solves = 0
    n_fallbacks = 0

    expert_trajectory: list = [_serialize_state_record(sim)]
    expert_decisions: list = []

    while not sim.is_terminal:
        state = sim.state
        epoch = int(state.epoch)

        if not state.pending_tasks:
            action: List[Tuple[int, int]] = []
        else:
            committed_ids: Set[int] = {
                int(c["task_id"]) for c in sim.trajectory.commitments
            }
            window_ids = sorted(
                int(t["id"])
                for t in instance.tasks
                if int(t["id"]) in serviceable
                and int(t["id"]) not in committed_ids
                and int(t["release_epoch"]) < epoch + horizon
            )

            finish_times = state.wall_clock + state.busy_times
            past_busy = np.zeros(R, dtype=np.float64)
            for c in sim.trajectory.commitments:
                past_busy[int(c["robot_id"])] += float(c["completion_time"])

            n_solves += 1
            arc_solution = solve_rolling_window(
                instance,
                window_ids,
                positions=state.positions,
                finish_times=finish_times,
                wall_clock=float(state.wall_clock),
                weights=weights,
                past_busy=past_busy,
                time_limit_seconds=time_limit_seconds,
                seed=seed,
            )

            if arc_solution is None:
                # No Gurobi incumbent in the time limit: fall back to Hungarian-on-kappa, recorded as a normal label since the policy IS the oracle here.
                n_fallbacks += 1
                cost, robot_ids, task_ids = build_kappa_cost_matrix(
                    state, instance, weights, sim.trajectory.commitments
                )
                action = hungarian_action(cost, robot_ids, task_ids)
            else:
                action = _window_commits(
                    arc_solution, window_ids, state, instance
                )

        expert_decisions.append([[int(r), int(j)] for (r, j) in action])
        sim.step(action)
        expert_trajectory.append(_serialize_state_record(sim))

    # Pad to the canonical minimum of H dispatcher epochs with pure-drain (empty) epochs, exactly as extract_expert_decisions does.
    while len(expert_decisions) < H:
        expert_decisions.append([])
        expert_trajectory.append(expert_trajectory[-1])

    return (
        expert_decisions,
        expert_trajectory,
        n_solves,
        n_fallbacks,
        _served_all(sim),
    )


def build_record(
    instance,
    seed: int,
    split: str,
    horizon: int,
    time_limit_seconds: int,
    weights: dict,
    expert_decisions: list,
    expert_trajectory: list,
    n_solves: int,
    n_fallbacks: int,
    served_all: bool,
    final_cost: Optional[float],
) -> dict:
    """Assemble one cache record matching the training_set_il_v4 layout.

    ``milp_solution`` is ``None`` (no anticipative solve); ``oracle_type``
    distinguishes rolling-horizon records from anticipative ones.
    """
    return {
        "R": int(instance.R),
        "T": int(instance.T),
        "seed": int(seed),
        "split": split,
        "instance": _serialize_instance(instance),
        "milp_solution": None,
        "expert_decisions": expert_decisions,
        "expert_trajectory": expert_trajectory,
        "weights": dict(weights),
        "oracle_type": f"rolling_horizon_h{horizon}",
        "rolling_horizon": int(horizon),
        "epoch_time_limit": int(time_limit_seconds),
        "n_solves": int(n_solves),
        "n_fallbacks": int(n_fallbacks),
        "served_all": bool(served_all),
        "final_cost": (None if final_cost is None else float(final_cost)),
        "timestamp": datetime.utcnow().isoformat(),
    }


def _process_one_seed(task: dict) -> dict:
    """Worker entry point: generate, solve, write one instance. Picklable."""
    seed = int(task["seed"])
    config = {"R": int(task["n_robots"]), "T": int(task["n_tasks"])}
    horizon = int(task["horizon"])
    time_limit = int(task["time_limit"])
    output_dir = Path(task["output_dir"])
    split = task["split"]
    weights = dict(LOCKED_WEIGHTS)

    out_path = output_dir / f"seed{seed}.json"
    if out_path.exists():
        return {"seed": seed, "status": "cached", "served_all": None,
                "n_fallbacks": 0, "final_cost": None}

    t0 = time.time()
    instance = generate_instance(seed, config=config)
    (
        expert_decisions,
        expert_trajectory,
        n_solves,
        n_fallbacks,
        served_all,
    ) = run_rolling_horizon_with_trace(
        instance, weights, horizon, time_limit, seed
    )

    # Cost is only meaningful when the rollout served every serviceable task.
    final_cost = None
    if served_all:
        sim = DynamicSimulator(instance)
        for action in expert_decisions:
            if sim.is_terminal:
                break
            sim.step([(int(r), int(j)) for (r, j) in action])
        while not sim.is_terminal:
            sim.step([])
        if sim.is_terminal:
            final_cost = float(sim.compute_cost(weights).combined)

    record = build_record(
        instance, seed, split, horizon, time_limit, weights,
        expert_decisions, expert_trajectory,
        n_solves, n_fallbacks, served_all, final_cost,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(record, f, indent=2)

    return {
        "seed": seed,
        "status": "written",
        "served_all": served_all,
        "n_fallbacks": n_fallbacks,
        "final_cost": final_cost,
        "elapsed": time.time() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a rolling-horizon IL training cache."
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir",
                        type=str, required=True)
    parser.add_argument("--num-instances", "--num_instances",
                        dest="num_instances", type=int, required=True)
    parser.add_argument("--n-robots", "--n_robots", dest="n_robots",
                        type=int, required=True)
    parser.add_argument("--n-tasks", "--n_tasks", dest="n_tasks",
                        type=int, required=True)
    parser.add_argument("--horizon", type=int, default=5,
                        help="Lookahead h in epochs (default 5).")
    parser.add_argument("--time-limit", "--time_limit", dest="time_limit",
                        type=int, default=30,
                        help="Gurobi time limit per epoch solve (default 30s).")
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers",
                        type=int, default=1)
    parser.add_argument("--seed-offset", "--seed_offset", dest="seed_offset",
                        type=int, default=90000,
                        help="Base seed; instance i uses seed_offset + i "
                        "(default 90000, disjoint from existing caches).")
    parser.add_argument("--split", type=str, default=None,
                        help="Value for the record 'split' field; defaults to "
                        "the output directory's leaf name.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    split = args.split if args.split is not None else output_dir.name
    seeds = [args.seed_offset + i for i in range(args.num_instances)]

    print(
        f"Rolling-horizon oracle cache: h={args.horizon}, "
        f"time_limit={args.time_limit}s/epoch, R={args.n_robots}, "
        f"T={args.n_tasks}, {len(seeds)} instances "
        f"(seeds {seeds[0]}..{seeds[-1]}), workers={args.num_workers}",
        flush=True,
    )
    print(f"Output dir: {output_dir} (split='{split}')", flush=True)
    print(f"Weights: {LOCKED_WEIGHTS}", flush=True)

    tasks = [
        {
            "seed": s,
            "n_robots": args.n_robots,
            "n_tasks": args.n_tasks,
            "horizon": args.horizon,
            "time_limit": args.time_limit,
            "output_dir": str(output_dir),
            "split": split,
        }
        for s in seeds
    ]

    t_start = time.time()
    results: List[dict] = []
    if args.num_workers <= 1:
        for i, task in enumerate(tasks):
            res = _process_one_seed(task)
            results.append(res)
            _print_progress(i + 1, len(tasks), res)
    else:
        with mp.Pool(processes=args.num_workers) as pool:
            for i, res in enumerate(
                pool.imap_unordered(_process_one_seed, tasks)
            ):
                results.append(res)
                _print_progress(i + 1, len(tasks), res)

    _summarize(results, output_dir, time.time() - t_start)


def _print_progress(done: int, total: int, res: dict) -> None:
    if res["status"] == "cached":
        print(f"[{done}/{total}] seed={res['seed']} [cached, skipped]",
              flush=True)
        return
    cost = res.get("final_cost")
    cost_str = "n/a" if cost is None else f"{cost:.3f}"
    print(
        f"[{done}/{total}] seed={res['seed']} served_all={res['served_all']} "
        f"fallbacks={res['n_fallbacks']} cost={cost_str} "
        f"({res.get('elapsed', 0.0):.1f}s)",
        flush=True,
    )


def _summarize(results: List[dict], output_dir: Path, elapsed: float) -> None:
    n = len(results)
    written = [r for r in results if r["status"] == "written"]
    cached = [r for r in results if r["status"] == "cached"]
    served = [r for r in written if r["served_all"]]
    total_fallbacks = sum(r["n_fallbacks"] for r in written)
    print("\n" + "=" * 60, flush=True)
    print("Rolling-horizon oracle cache summary", flush=True)
    print("=" * 60, flush=True)
    print(f"  instances requested      {n}", flush=True)
    print(f"  written                  {len(written)}", flush=True)
    print(f"  already cached (skipped) {len(cached)}", flush=True)
    if written:
        print(f"  served-all               {len(served)}/{len(written)}",
              flush=True)
        print(f"  total fallback epochs    {total_fallbacks}", flush=True)
    print(f"  output dir               {output_dir}", flush=True)
    print(f"  wall time                {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
