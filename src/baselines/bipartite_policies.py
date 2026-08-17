"""Bipartite-matching policies for Phase 0.5.

Two baselines share a single Hungarian solver. Greedy uses a distance-only
cost matrix. Hungarian-on-kappa uses the full three-term per-epoch kappa
(distance + makespan contribution + balance term against running busy times).

The balance term ``w_bal * (b_running_r - b_running_mean)`` contributes a
constant total to any complete matching when |A_t| <= |P_t|, so it only
shapes the action when there are more available robots than pending tasks
and some robots stay idle. In that regime it pulls toward assigning the
less-busy robots.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from instances.synthetic_generator import SyntheticInstance
from simulator.dynamic_simulator import DynamicSimulator, SimulatorState


def _travel(a: np.ndarray, b: np.ndarray, v: float) -> float:
    return float(np.linalg.norm(a - b)) / v


def build_greedy_cost_matrix(
    state: SimulatorState,
    instance: SyntheticInstance,
) -> Tuple[np.ndarray, List[int], List[int]]:
    robot_ids = sorted(int(r) for r in state.available_robots)
    task_ids = sorted(int(j) for j in state.pending_tasks)
    tasks_by_id = {int(t["id"]): t for t in instance.tasks}
    v = float(instance.config["v"])

    n_r = len(robot_ids)
    n_t = len(task_ids)
    cost = np.zeros((n_r, n_t), dtype=np.float64)
    for i, r in enumerate(robot_ids):
        pos = np.asarray(state.positions[r], dtype=np.float64)
        for k, j in enumerate(task_ids):
            t = tasks_by_id[j]
            pickup = np.asarray(t["pickup"], dtype=np.float64)
            drop = np.asarray(t["drop"], dtype=np.float64)
            tp = _travel(pos, pickup, v)
            td = _travel(pickup, drop, v)
            cost[i, k] = tp + td
    return cost, robot_ids, task_ids


def build_kappa_cost_matrix(
    state: SimulatorState,
    instance: SyntheticInstance,
    weights: dict,
    trajectory_commitments: list,
) -> Tuple[np.ndarray, List[int], List[int]]:
    R = int(instance.R)
    b_running = np.zeros(R, dtype=np.float64)
    for c in trajectory_commitments:
        r = int(c["robot_id"])
        b_running[r] += float(c["completion_time"])
    b_mean = float(b_running.mean()) if R > 0 else 0.0

    robot_ids = sorted(int(r) for r in state.available_robots)
    task_ids = sorted(int(j) for j in state.pending_tasks)
    tasks_by_id = {int(t["id"]): t for t in instance.tasks}
    v = float(instance.config["v"])
    w_d = float(weights["w_dist"])
    w_m = float(weights["w_make"])
    w_b = float(weights["w_bal"])

    n_r = len(robot_ids)
    n_t = len(task_ids)
    cost = np.zeros((n_r, n_t), dtype=np.float64)
    for i, r in enumerate(robot_ids):
        pos = np.asarray(state.positions[r], dtype=np.float64)
        bal_r = b_running[r] - b_mean
        for k, j in enumerate(task_ids):
            t = tasks_by_id[j]
            pickup = np.asarray(t["pickup"], dtype=np.float64)
            drop = np.asarray(t["drop"], dtype=np.float64)
            duration = float(t["duration"])
            tp = _travel(pos, pickup, v)
            td = _travel(pickup, drop, v)
            c_j = tp + td + duration
            cost[i, k] = w_d * (tp + td) + w_m * c_j + w_b * bal_r
    return cost, robot_ids, task_ids


def hungarian_action(
    cost_matrix: np.ndarray,
    robot_ids: List[int],
    task_ids: List[int],
) -> List[Tuple[int, int]]:
    if cost_matrix.size == 0:
        return []
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return [(robot_ids[r], task_ids[c]) for r, c in zip(row_ind, col_ind)]


def _run_with_per_epoch_builder(
    sim: DynamicSimulator,
    per_epoch_builder: Callable[[SimulatorState], Tuple[np.ndarray, List[int], List[int]]],
) -> DynamicSimulator:
    while not sim.is_terminal:
        state = sim.state
        cost, robot_ids, task_ids = per_epoch_builder(state)
        action = hungarian_action(cost, robot_ids, task_ids)
        sim.step(action)
    return sim


def run_greedy_policy(instance: SyntheticInstance) -> DynamicSimulator:
    sim = DynamicSimulator(instance)
    return _run_with_per_epoch_builder(
        sim, lambda s: build_greedy_cost_matrix(s, instance)
    )


def run_hungarian_kappa_policy(
    instance: SyntheticInstance, weights: dict
) -> DynamicSimulator:
    sim = DynamicSimulator(instance)
    return _run_with_per_epoch_builder(
        sim,
        lambda s: build_kappa_cost_matrix(
            s, instance, weights, sim.trajectory.commitments
        ),
    )
