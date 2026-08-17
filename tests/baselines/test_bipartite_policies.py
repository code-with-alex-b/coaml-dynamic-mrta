"""Tests for the Phase 0.5 bipartite-matching baselines."""

from __future__ import annotations

import numpy as np
import pytest

from instances.synthetic_generator import generate_instance
from simulator.dynamic_simulator import DynamicSimulator, SimulatorState
from baselines.bipartite_policies import (
    build_greedy_cost_matrix,
    build_kappa_cost_matrix,
    hungarian_action,
    run_greedy_policy,
    run_hungarian_kappa_policy,
)


def _make_two_robot_two_task_scenario():
    inst = generate_instance(seed=0, config={"R": 2, "T": 2})
    inst.initial_positions = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    inst.tasks[0] = {
        "id": 0,
        "release_epoch": 0,
        "pickup": np.array([3.0, 4.0], dtype=np.float64),
        "drop": np.array([8.0, 4.0], dtype=np.float64),
        "duration": 1.0,
    }
    inst.tasks[1] = {
        "id": 1,
        "release_epoch": 0,
        "pickup": np.array([15.0, 0.0], dtype=np.float64),
        "drop": np.array([20.0, 0.0], dtype=np.float64),
        "duration": 1.0,
    }
    return inst


def test_greedy_cost_matrix_shape_default():
    inst = generate_instance(seed=0)
    sim = DynamicSimulator(inst)
    state = sim.state
    expected_n_pending = sum(
        1 for t in inst.tasks if int(t["release_epoch"]) == 0
    )
    cost, robot_ids, task_ids = build_greedy_cost_matrix(state, inst)
    assert cost.shape == (10, expected_n_pending)
    assert len(robot_ids) == 10
    assert len(task_ids) == expected_n_pending


def test_greedy_cost_matrix_values():
    inst = _make_two_robot_two_task_scenario()
    sim = DynamicSimulator(inst)
    state = sim.state
    cost, robot_ids, task_ids = build_greedy_cost_matrix(state, inst)
    assert robot_ids == [0, 1]
    assert task_ids == [0, 1]
    # r0->t0 total 10, r0->t1 total 20, r1->t0 total sqrt(65)+5, r1->t1 total 10.
    assert cost[0, 0] == pytest.approx(10.0)
    assert cost[0, 1] == pytest.approx(20.0)
    assert cost[1, 0] == pytest.approx(np.sqrt(65.0) + 5.0)
    assert cost[1, 1] == pytest.approx(10.0)


def test_kappa_cost_matrix_at_epoch_zero():
    inst = _make_two_robot_two_task_scenario()
    sim = DynamicSimulator(inst)
    state = sim.state
    weights = {"w_dist": 2.0, "w_make": 3.0, "w_bal": 5.0}
    cost, robot_ids, task_ids = build_kappa_cost_matrix(
        state, inst, weights, trajectory_commitments=[]
    )
    # No prior commitments so the balance term is zero; kappa[r,j] = w_dist*travel + w_make*c_j gives r0->t0=53, r0->t1=103, r1->t0=2*(sqrt(65)+5)+3*(sqrt(65)+6), r1->t1=53.
    assert cost[0, 0] == pytest.approx(53.0)
    assert cost[0, 1] == pytest.approx(103.0)
    s65 = float(np.sqrt(65.0))
    assert cost[1, 0] == pytest.approx(2.0 * (s65 + 5.0) + 3.0 * (s65 + 6.0))
    assert cost[1, 1] == pytest.approx(53.0)


def test_kappa_balance_term_with_idle_surplus():
    # |A_t| = 3, |P_t| = 1; all robots at the same position, so only the balance term differs across robots.
    inst = generate_instance(seed=0, config={"R": 3, "T": 1})
    inst.initial_positions = np.zeros((3, 2), dtype=np.float64)
    inst.tasks[0] = {
        "id": 0,
        "release_epoch": 0,
        "pickup": np.array([1.0, 0.0]),
        "drop": np.array([2.0, 0.0]),
        "duration": 1.0,
    }
    state = SimulatorState(
        positions=np.zeros((3, 2), dtype=np.float64),
        busy_times=np.zeros(3, dtype=np.float64),
        pending_tasks={0},
        available_robots={0, 1, 2},
        epoch=1,
        wall_clock=5.0,
    )
    fake_commits = [
        {
            "epoch": 0,
            "robot_id": 0,
            "task_id": 99,
            "start_position": np.zeros(2),
            "pickup": np.zeros(2),
            "drop": np.zeros(2),
            "duration": 0.0,
            "completion_time": 10.0,
            "travel_distance": 0.0,
            "absolute_finish_time": 10.0,
        }
    ]
    weights = {"w_dist": 1.0, "w_make": 1.0, "w_bal": 1.0}
    cost, robot_ids, task_ids = build_kappa_cost_matrix(
        state, inst, weights, fake_commits
    )
    assert cost.shape == (3, 1)
    assert robot_ids == [0, 1, 2]
    # bal_0 = 10 - 10/3, bal_1 = bal_2 = -10/3, so diff = w_bal * (bal_0 - bal_1) = 10.
    assert cost[0, 0] - cost[1, 0] == pytest.approx(10.0)
    assert cost[1, 0] == pytest.approx(cost[2, 0])


def test_hungarian_rectangular_more_robots():
    cost = np.array([[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]])
    robot_ids = [10, 20, 30]
    task_ids = [100, 200]
    action = hungarian_action(cost, robot_ids, task_ids)
    assert len(action) == 2
    assigned_robots = {r for r, _ in action}
    assert 30 not in assigned_robots


def test_hungarian_rectangular_more_tasks():
    cost = np.array([[1.0, 2.0, 100.0], [1.0, 3.0, 100.0]])
    robot_ids = [10, 20]
    task_ids = [100, 200, 300]
    action = hungarian_action(cost, robot_ids, task_ids)
    assert len(action) == 2
    assigned_tasks = {j for _, j in action}
    assert 300 not in assigned_tasks


def test_hungarian_deterministic():
    cost = np.array([[5.0, 2.0, 8.0], [1.0, 9.0, 3.0], [6.0, 4.0, 7.0]])
    robot_ids = [0, 1, 2]
    task_ids = [100, 200, 300]
    a = hungarian_action(cost, robot_ids, task_ids)
    b = hungarian_action(cost, robot_ids, task_ids)
    assert a == b


def test_empty_action_no_available_robots():
    inst = _make_two_robot_two_task_scenario()
    state = SimulatorState(
        positions=np.zeros((2, 2), dtype=np.float64),
        busy_times=np.array([1.0, 1.0], dtype=np.float64),
        pending_tasks={0, 1},
        available_robots=set(),
        epoch=0,
        wall_clock=0.0,
    )
    cost, robot_ids, task_ids = build_greedy_cost_matrix(state, inst)
    assert cost.shape == (0, 2)
    action = hungarian_action(cost, robot_ids, task_ids)
    assert action == []


def test_empty_action_no_pending_tasks():
    inst = _make_two_robot_two_task_scenario()
    state = SimulatorState(
        positions=np.zeros((2, 2), dtype=np.float64),
        busy_times=np.zeros(2, dtype=np.float64),
        pending_tasks=set(),
        available_robots={0, 1},
        epoch=0,
        wall_clock=0.0,
    )
    cost, robot_ids, task_ids = build_greedy_cost_matrix(state, inst)
    assert cost.shape == (2, 0)
    action = hungarian_action(cost, robot_ids, task_ids)
    assert action == []


def test_greedy_full_run_small_instance():
    inst = generate_instance(seed=0, config={"T": 5, "R": 2})
    sim = run_greedy_policy(inst)
    assert sim.is_terminal
    assert not sim.trajectory.simulator_failure
    committed = sorted(c["task_id"] for c in sim.trajectory.commitments)
    assert committed == list(range(5))
    cb = sim.compute_cost({"w_dist": 1.0, "w_make": 1.0, "w_bal": 1.0})
    assert cb.combined >= 0.0
    assert np.isfinite(cb.combined)


def test_hungarian_kappa_full_run_small_instance():
    inst = generate_instance(seed=0, config={"T": 5, "R": 2})
    weights = {"w_dist": 1.0, "w_make": 1.0, "w_bal": 1.0}
    sim = run_hungarian_kappa_policy(inst, weights)
    assert sim.is_terminal
    assert not sim.trajectory.simulator_failure
    cb = sim.compute_cost(weights)
    assert cb.combined >= 0.0
    assert np.isfinite(cb.combined)
    assert cb.weights_used == weights


def test_hungarian_kappa_beats_or_matches_greedy_on_some_seed():
    weights = {"w_dist": 1.0, "w_make": 1.0, "w_bal": 1.0}
    wins = 0
    for seed in range(5):
        inst = generate_instance(seed=seed, config={"T": 5, "R": 2})
        greedy_cost = run_greedy_policy(inst).compute_cost(weights).combined
        kappa_cost = run_hungarian_kappa_policy(inst, weights).compute_cost(
            weights
        ).combined
        if kappa_cost <= greedy_cost + 1e-9:
            wins += 1
    assert wins >= 1, (
        "Hungarian-on-kappa did not beat or match greedy on any of seeds 0..4"
    )
