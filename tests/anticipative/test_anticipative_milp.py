"""Tests for the Phase 0.5 offline anticipative MILP."""

from __future__ import annotations

import numpy as np
import pytest

from instances.synthetic_generator import generate_instance
from simulator.dynamic_simulator import DynamicSimulator
from anticipative.anticipative_milp import (
    AnticipativeSolution,
    extract_per_epoch_decisions,
    solve_anticipative,
)


WEIGHTS_UNIT = {"w_dist": 1.0, "w_make": 1.0, "w_bal": 1.0}


def _make_trivial_single_task():
    inst = generate_instance(seed=0, config={"R": 1, "T": 1, "H": 2})
    inst.initial_positions = np.array([[0.0, 0.0]], dtype=np.float64)
    inst.tasks[0] = {
        "id": 0,
        "release_epoch": 0,
        "pickup": np.array([1.0, 0.0]),
        "drop": np.array([2.0, 0.0]),
        "duration": 1.0,
    }
    return inst


def _make_two_robot_two_task_distance_dominant():
    inst = generate_instance(seed=0, config={"R": 2, "T": 2, "H": 2})
    inst.initial_positions = np.array(
        [[0.0, 0.0], [100.0, 0.0]], dtype=np.float64
    )
    inst.tasks[0] = {
        "id": 0,
        "release_epoch": 0,
        "pickup": np.array([1.0, 0.0]),
        "drop": np.array([2.0, 0.0]),
        "duration": 1.0,
    }
    inst.tasks[1] = {
        "id": 1,
        "release_epoch": 0,
        "pickup": np.array([101.0, 0.0]),
        "drop": np.array([102.0, 0.0]),
        "duration": 1.0,
    }
    return inst


def test_trivial_single_task_instance():
    inst = _make_trivial_single_task()
    sol = solve_anticipative(inst, WEIGHTS_UNIT, time_limit_seconds=30)
    assert sol.status == "optimal"
    assert sol.objective_value > 0
    assert (0, 0) in sol.completion_times
    assert sol.n_unserved_tasks == 0


def test_distance_only_weights_pick_nearest_assignment():
    inst = _make_two_robot_two_task_distance_dominant()
    weights = {"w_dist": 1.0, "w_make": 0.0, "w_bal": 0.0}
    sol = solve_anticipative(inst, weights, time_limit_seconds=30)
    assert sol.status == "optimal"
    assert (0, 0) in sol.completion_times
    assert (1, 1) in sol.completion_times


def test_makespan_only_weights_balance_assignment():
    inst = _make_two_robot_two_task_distance_dominant()
    weights = {"w_dist": 0.0, "w_make": 1.0, "w_bal": 0.0}
    sol = solve_anticipative(inst, weights, time_limit_seconds=30)
    assert sol.status == "optimal"
    # Same instance: assigning each robot to its near task minimises makespan.
    assert (0, 0) in sol.completion_times
    assert (1, 1) in sol.completion_times


def test_status_decoding_optimal():
    inst = _make_trivial_single_task()
    sol = solve_anticipative(inst, WEIGHTS_UNIT, time_limit_seconds=30)
    assert sol.status == "optimal"


def test_all_tasks_served():
    inst = generate_instance(seed=0, config={"R": 3, "T": 5})
    sol = solve_anticipative(inst, WEIGHTS_UNIT, time_limit_seconds=60)
    assert sol.n_unserved_tasks == 0
    served_pairs = set(sol.completion_times.keys())
    served_tasks = {j for (r, j) in served_pairs}
    assert served_tasks == set(range(5))


def test_time_limit_allows_optimal_or_time_limit():
    inst = _make_trivial_single_task()
    sol = solve_anticipative(inst, WEIGHTS_UNIT, time_limit_seconds=1)
    assert sol.status in ("optimal", "time_limit")
    assert sol.objective_value is not None


def test_mip_gap_parameter_respected():
    inst = generate_instance(seed=0, config={"R": 3, "T": 5})
    sol = solve_anticipative(
        inst, WEIGHTS_UNIT, time_limit_seconds=60, mip_gap=0.5
    )
    assert sol.mip_gap <= 0.5 + 1e-6


def test_cost_decomposition_matches_objective():
    inst = generate_instance(seed=0, config={"R": 3, "T": 5})
    weights = {"w_dist": 0.5, "w_make": 1.0, "w_bal": 2.0}
    sol = solve_anticipative(inst, weights, time_limit_seconds=60)
    recomputed = (
        weights["w_dist"] * sol.distance
        + weights["w_make"] * sol.makespan
        + weights["w_bal"] * sol.imbalance
    )
    # The v3 objective adds a 1e-6 completion-time tiebreak to break schedule degeneracy, so ObjVal matches only after adding it back.
    tiebreak = 1e-6 * sum(sol.completion_times.values())
    assert abs(recomputed + tiebreak - sol.objective_value) < 1e-6


def test_extract_per_epoch_decisions_shape_and_coverage():
    inst = generate_instance(seed=0, config={"R": 3, "T": 5})
    sol = solve_anticipative(inst, WEIGHTS_UNIT, time_limit_seconds=60)
    decisions = extract_per_epoch_decisions(sol, inst)
    for epoch_action in decisions:
        for pair in epoch_action:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
    all_pairs = {pair for epoch_action in decisions for pair in epoch_action}
    expected = set(sol.completion_times.keys())
    assert all_pairs == expected


def test_extract_per_epoch_decisions_epoch_assignment_correct():
    inst = _make_trivial_single_task()
    sol = solve_anticipative(inst, WEIGHTS_UNIT, time_limit_seconds=30)
    Delta = float(inst.config["Delta"])
    tasks_by_id = {int(t["id"]): t for t in inst.tasks}
    decisions = extract_per_epoch_decisions(sol, inst)
    for t_epoch, action in enumerate(decisions):
        for (r, j) in action:
            f_rj = sol.completion_times[(r, j)]
            d_j = float(tasks_by_id[j]["duration"])
            start_time = f_rj - d_j
            assert t_epoch * Delta <= start_time < (t_epoch + 1) * Delta


def test_milp_replay_through_simulator():
    """MILP-extracted decisions execute through the simulator without rejection.
    Under Path B continuous-time semantics, the simulator accepts queue-ahead commits."""
    instance = generate_instance(seed=42, config={"R": 3, "T": 5})
    weights = {"w_dist": 1.0, "w_make": 1.0, "w_bal": 1.0}
    solution = solve_anticipative(instance, weights)
    per_epoch_actions = extract_per_epoch_decisions(solution, instance)
    sim = DynamicSimulator(instance)
    sim.reset()
    for action in per_epoch_actions:
        if sim.is_terminal:
            break
        sim.step(action)
    while not sim.is_terminal:
        sim.step([])
    assert not sim.trajectory.simulator_failure
    assert len(sim.trajectory.unserved_task_ids) == 0
    assert len(sim.trajectory.commitments) == 5


def test_milp_objective_lower_bounds_simulator_cost():
    """The MILP objective is a lower bound on the simulator-replay cost.

    Since the v3 formulation fixes (commit c75cff0) the simulator enforces
    epoch-boundary start times that the continuous-time MILP does not model,
    so the replayed schedule can only be delayed relative to the MILP's
    schedule. The MILP objective is therefore a lower bound on the replay
    cost, not an exact match, and gap closure ceilings use the expert replay
    cost with the MILP objective retained as a reference bound only."""
    instance = generate_instance(seed=42, config={"R": 3, "T": 5})
    weights = {"w_dist": 1.0, "w_make": 1.0, "w_bal": 1.0}
    solution = solve_anticipative(instance, weights)
    per_epoch_actions = extract_per_epoch_decisions(solution, instance)
    sim = DynamicSimulator(instance)
    sim.reset()
    for action in per_epoch_actions:
        if sim.is_terminal:
            break
        sim.step(action)
    while not sim.is_terminal:
        sim.step([])
    sim_cost = sim.compute_cost(weights)
    assert sim_cost.combined >= solution.objective_value - 1e-4, (
        f"Simulator replay cost {sim_cost.combined} fell below the MILP "
        f"objective {solution.objective_value}. The epoch-constrained "
        f"simulator can only delay the MILP's continuous-time schedule, so "
        f"a replay cost under the MILP objective indicates a formulation or "
        f"simulator bug."
    )
