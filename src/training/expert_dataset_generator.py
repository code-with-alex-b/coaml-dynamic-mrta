"""Expert dataset generator.

For each training seed at the locked R=6, T=18 distribution, solves the anticipative
MILP, extracts the per-epoch expert decisions, and caches the record, being the
instance, the MILP solution, the expert decisions and the replayed trajectory.

The per-epoch expert decision is the set of ``(r, j)`` pairs where robot ``r`` begins
task ``j`` in the wall-clock interval ``[t * Delta, (t+1) * Delta)`` under the MILP
schedule. The start time of ``j`` on ``r`` is ``f[r, j] - d_j - travel(pickup_j,
drop_j)``, the moment the robot begins the pickup-to-drop leg, which is
``effective_pickup_start`` in the simulator. That is the latest commit time consistent
with the MILP schedule.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from anticipative.anticipative_milp import (
    AnticipativeSolution,
    solve_anticipative,
)
from instances.synthetic_generator import (
    SyntheticInstance,
    generate_instance,
)
from simulator.dynamic_simulator import DynamicSimulator


LOCKED_CONFIG = {"R": 6, "T": 18}
LOCKED_WEIGHTS = {"w_dist": 0.0637, "w_make": 0.2398, "w_bal": 0.6965}
MIP_TIME_LIMIT = 60
MIP_GAP = 0.01

SPLITS = {
    "train": list(range(10000, 11000)),
    "val": list(range(11000, 11200)),
    "test": list(range(11200, 11400)),
}

PROTECTED_SEED_RANGES = [
    ("viability", list(range(1000, 1050))),
    ("weight_selection", list(range(900, 920))),
    ("sweep", list(range(2000, 6000))),
]

CACHE_DIR = Path("cache/training_set_il_v3")
LOG_DIR = Path("logs/training_set_il_v3")
TIEBREAK_COEFFICIENT = 1e-6


def _node_to_str(node) -> str:
    if isinstance(node, tuple):
        return f"{node[0]}{node[1]}"
    return str(node)


def _serialize_instance(inst: SyntheticInstance) -> dict:
    return {
        "R": int(inst.R),
        "T": int(inst.T),
        "H": int(inst.H),
        "seed": int(inst.seed),
        "config": inst.config,
        "initial_positions": inst.initial_positions.tolist(),
        "tasks": [
            {
                "id": int(t["id"]),
                "release_epoch": int(t["release_epoch"]),
                "pickup": np.asarray(t["pickup"], dtype=np.float64).tolist(),
                "drop": np.asarray(t["drop"], dtype=np.float64).tolist(),
                "duration": float(t["duration"]),
            }
            for t in inst.tasks
        ],
        "pickup_cluster_indices": inst.pickup_cluster_indices.tolist(),
        "drop_cluster_indices": inst.drop_cluster_indices.tolist(),
    }


def _serialize_state_record(sim: DynamicSimulator) -> dict:
    state = sim.state
    return {
        "epoch": int(state.epoch),
        "wall_clock": float(state.wall_clock),
        "positions": state.positions.tolist(),
        "finish_times": sim._robot_finish_times.tolist(),
        "pending_tasks": sorted(int(j) for j in state.pending_tasks),
        "available_robots": sorted(int(r) for r in state.available_robots),
    }


def _serialize_solution(solution: AnticipativeSolution) -> dict:
    return {
        "objective_value": float(solution.objective_value),
        "distance": float(solution.distance),
        "makespan": float(solution.makespan),
        "imbalance": float(solution.imbalance),
        "status": str(solution.status),
        "mip_gap": float(solution.mip_gap),
        "obj_bound": float(solution.obj_bound),
        "tiebreak_coefficient": float(TIEBREAK_COEFFICIENT),
        "solve_time_seconds": float(solution.solve_time_seconds),
        "n_unserved_tasks": int(solution.n_unserved_tasks),
        "completion_times": [
            {"r": int(r), "j": int(j), "f": float(v)}
            for (r, j), v in solution.completion_times.items()
        ],
        "arc_solution": [
            {"r": int(r), "i": _node_to_str(i), "j": _node_to_str(j)}
            for (r, i, j) in solution.arc_solution.keys()
        ],
    }


def extract_expert_decisions(
    instance: SyntheticInstance,
    solution: AnticipativeSolution,
) -> Tuple[list, list]:
    """Bucket each used MILP arc (r, j) into an epoch, chain-aware per robot.

    For each robot, tasks are sorted by start time and each is placed at
    ``max(floor(start_time / Delta), release_epoch)``, and from the second onward at
    least one epoch after the previous task in the chain. That keeps assignments strictly
    monotone per robot, so at most one commit per robot per epoch.

    The epoch is a lower bound. Decisions are emitted by replaying the chain through the
    epoch-quantised simulator, and a task whose epoch falls while its robot is still busy
    is deferred until the robot frees, so every label sits on a robot the scorer treats as
    available.

    The list has length at least H, since H is the arrival cutoff rather than the
    commitment cutoff and late-chain commits need drain epochs beyond it.
    chain commits, since H is the arrival cutoff rather than the commitment cutoff.
    """
    H = int(instance.H)
    Delta = float(instance.config["Delta"])
    v = float(instance.config["v"])
    tasks_by_id = {int(t["id"]): t for t in instance.tasks}

    per_robot: dict = {}
    for (r, j), f_val in solution.completion_times.items():
        task = tasks_by_id[int(j)]
        pickup = np.asarray(task["pickup"], dtype=np.float64)
        drop = np.asarray(task["drop"], dtype=np.float64)
        td = float(np.linalg.norm(pickup - drop) / v)
        duration = float(task["duration"])
        tp_rj = float(solution.origin_to_pickup_travel[(r, j)])
        start_time = float(f_val) - tp_rj - td - duration
        release_epoch_j = int(task["release_epoch"])
        per_robot.setdefault(int(r), []).append(
            (start_time, int(j), float(f_val), release_epoch_j)
        )
    for r in per_robot:
        per_robot[r].sort()

    desired_epochs: dict = {}
    max_bucket = H - 1
    for r, chain in per_robot.items():
        last_assigned_epoch = -1
        epochs_r: list = []
        for k, (st, j, finish, release_epoch_j) in enumerate(chain):
            naive_epoch = int(np.floor(st / Delta))
            if k == 0:
                e = max(naive_epoch, release_epoch_j)
            else:
                e = max(naive_epoch, release_epoch_j, last_assigned_epoch + 1)
            epochs_r.append((e, j))
            if e > max_bucket:
                max_bucket = e
            last_assigned_epoch = e
        desired_epochs[int(r)] = epochs_r

    next_idx = {r: 0 for r in desired_epochs}
    total_commits = sum(len(c) for c in desired_epochs.values())
    dispatched = 0
    # Generous safety cap against an unbounded loop; the back-dated chain guarantees every robot frees eventually.
    max_epochs = max(H, max_bucket + 1) + 4 * int(instance.T) + 4

    sim = DynamicSimulator(instance)
    expert_trajectory = [_serialize_state_record(sim)]
    expert_decisions: list = []
    t = 0
    while True:
        state = sim.state
        avail = state.available_robots
        pending = state.pending_tasks
        action: list = []
        for r, epochs_r in desired_epochs.items():
            idx = next_idx[r]
            if idx >= len(epochs_r):
                continue
            e, j = epochs_r[idx]
            if t >= e and r in avail and j in pending:
                action.append((r, j))
                next_idx[r] += 1
                dispatched += 1
        expert_decisions.append(action)
        terminal = sim.is_terminal
        if not terminal:
            _, terminal = sim.step(action)
        expert_trajectory.append(_serialize_state_record(sim))
        t += 1
        if terminal or t >= max_epochs:
            break

    # Pad to the canonical minimum length H with empty pure-drain epochs, matching the original max(H, ...) length contract.
    while len(expert_decisions) < H:
        expert_decisions.append([])
        expert_trajectory.append(expert_trajectory[-1])

    return expert_decisions, expert_trajectory


def _validate_cache_config(
    record: dict,
    weights: dict,
    mip_time_limit: int,
    mip_gap: float,
) -> None:
    """Raise ValueError when a cached record's config does not match the
    requested config.

    Weights are always checked. mip_time_limit and mip_gap_target are
    checked only when present in the record (pre-Fix-4 records lack them).
    tiebreak_coefficient defaults to 1e-6 for records without the field,
    which is the value used when the existing cache was generated.
    """
    import logging as _logging

    stored_weights = record.get("weights", {})
    for key in ("w_dist", "w_make", "w_bal"):
        expected = float(weights[key])
        found = float(stored_weights.get(key, float("nan")))
        if abs(found - expected) > 1e-12:
            raise ValueError(
                f"Cache config mismatch on seed {record.get('seed')}: "
                f"weights[{key}] expected {expected}, got {found}"
            )
    stored_tl = record.get("mip_time_limit")
    if stored_tl is not None and int(stored_tl) != int(mip_time_limit):
        raise ValueError(
            f"Cache config mismatch on seed {record.get('seed')}: "
            f"mip_time_limit expected {mip_time_limit}, got {stored_tl}"
        )
    stored_mg = record.get("mip_gap_target")
    if stored_mg is not None and abs(float(stored_mg) - float(mip_gap)) > 1e-12:
        raise ValueError(
            f"Cache config mismatch on seed {record.get('seed')}: "
            f"mip_gap_target expected {mip_gap}, got {stored_mg}"
        )
    stored_tc = record.get("milp_solution", {}).get("tiebreak_coefficient")
    if stored_tc is None:
        _logging.warning(
            "Cache record for seed %s has no tiebreak_coefficient stored; "
            "assuming 1e-6 (the value used when the existing cache was "
            "generated).",
            record.get("seed"),
        )
        stored_tc = 1e-6
    if abs(float(stored_tc) - TIEBREAK_COEFFICIENT) > 1e-10:
        raise ValueError(
            f"Cache config mismatch on seed {record.get('seed')}: "
            f"tiebreak_coefficient expected {TIEBREAK_COEFFICIENT}, "
            f"got {stored_tc}"
        )


def process_one_seed(
    split: str,
    seed: int,
    config: Optional[dict] = None,
    weights: Optional[dict] = None,
    mip_time_limit: int = MIP_TIME_LIMIT,
    mip_gap: float = MIP_GAP,
) -> dict:
    config = dict(config if config is not None else LOCKED_CONFIG)
    weights = dict(weights if weights is not None else LOCKED_WEIGHTS)
    cache_path = CACHE_DIR / split / f"seed{seed}.json"
    if cache_path.exists():
        with cache_path.open("r") as f:
            cached_record = json.load(f)
        _validate_cache_config(cached_record, weights, mip_time_limit, mip_gap)
        return cached_record

    instance = generate_instance(seed, config=config)
    solution = solve_anticipative(
        instance, weights, time_limit_seconds=mip_time_limit, mip_gap=mip_gap
    )
    expert_decisions, expert_trajectory = extract_expert_decisions(
        instance, solution
    )

    record = {
        "R": int(instance.R),
        "T": int(instance.T),
        "seed": int(seed),
        "split": split,
        "instance": _serialize_instance(instance),
        "milp_solution": _serialize_solution(solution),
        "expert_decisions": [
            [list(pair) for pair in epoch_commits]
            for epoch_commits in expert_decisions
        ],
        "expert_trajectory": expert_trajectory,
        "weights": weights,
        "mip_time_limit": int(mip_time_limit),
        "mip_gap_target": float(mip_gap),
        "timestamp": datetime.utcnow().isoformat(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(record, f, indent=2)
    return record


def generate_split(
    split: str,
    seeds: list,
    max_seeds: Optional[int] = None,
    config: Optional[dict] = None,
    weights: Optional[dict] = None,
    mip_time_limit: int = MIP_TIME_LIMIT,
    mip_gap: float = MIP_GAP,
) -> None:
    seeds = list(seeds[:max_seeds] if max_seeds is not None else seeds)
    print(f"=== Generating {split} split. {len(seeds)} seeds. ===", flush=True)

    mip_gaps: list = []
    solve_times: list = []
    n_optimal = 0
    n_time_limit = 0

    for i, seed in enumerate(seeds):
        cache_path = CACHE_DIR / split / f"seed{seed}.json"
        was_cached = cache_path.exists()
        record = process_one_seed(
            split, seed, config=config, weights=weights,
            mip_time_limit=mip_time_limit, mip_gap=mip_gap,
        )
        cached_marker = "[cached]" if was_cached else ""

        ms = record["milp_solution"]
        mip_gaps.append(float(ms["mip_gap"]))
        solve_times.append(float(ms["solve_time_seconds"]))
        if ms["status"] == "optimal":
            n_optimal += 1
        elif ms["status"] == "time_limit":
            n_time_limit += 1

        print(
            f"[{split} {i+1}/{len(seeds)}] seed={seed} "
            f"obj={ms['objective_value']:.2f} status={ms['status']} "
            f"mip_gap={ms['mip_gap']:.3f} "
            f"solve_time={ms['solve_time_seconds']:.1f}s {cached_marker}",
            flush=True,
        )

    if mip_gaps:
        print(
            f"\n[{split}] Done. {len(seeds)} instances. "
            f"optimal={n_optimal} time_limit={n_time_limit}",
            flush=True,
        )
        print(
            f"[{split}] MIP gap: mean={np.mean(mip_gaps):.3f} "
            f"p95={np.percentile(mip_gaps, 95):.3f}",
            flush=True,
        )
        print(
            f"[{split}] Solve time: mean={np.mean(solve_times):.1f}s "
            f"p95={np.percentile(solve_times, 95):.1f}s",
            flush=True,
        )


def generate_full_dataset(
    splits_to_run: Optional[list] = None,
    max_seeds_per_split: Optional[int] = None,
) -> None:
    splits_to_run = splits_to_run or list(SPLITS.keys())

    all_split_seeds: list = []
    for s in splits_to_run:
        all_split_seeds.extend(SPLITS[s])
    if len(set(all_split_seeds)) != len(all_split_seeds):
        raise ValueError("Overlapping seeds across requested splits.")
    for name, protected in PROTECTED_SEED_RANGES:
        overlap = set(all_split_seeds) & set(protected)
        if overlap:
            raise ValueError(
                f"Seed overlap with protected range '{name}': {sorted(overlap)}"
            )
    print(f"Seed disjointness verified for {splits_to_run}.", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for split in splits_to_run:
        generate_split(split, SPLITS[split], max_seeds=max_seeds_per_split)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--max-per-split", type=int, default=None)
    args = parser.parse_args()
    splits = args.splits.split(",")
    generate_full_dataset(
        splits_to_run=splits, max_seeds_per_split=args.max_per_split
    )
