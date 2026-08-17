"""Tests for the Phase 0.5 dynamic simulator."""

from __future__ import annotations

import numpy as np
import pytest

from instances.synthetic_generator import generate_instance
from simulator.dynamic_simulator import (
    CostBreakdown,
    DynamicSimulator,
    SimulatorState,
)


def _make_two_robot_single_task_scenario():
    inst = generate_instance(seed=0, config={"R": 2, "T": 1})
    inst.initial_positions = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
    inst.tasks[0] = {
        "id": 0,
        "release_epoch": 0,
        "pickup": np.array([5.0, 5.0], dtype=np.float64),
        "drop": np.array([15.0, 15.0], dtype=np.float64),
        "duration": 2.0,
    }
    return inst


def _make_handcrafted_T4_R2():
    inst = generate_instance(seed=0, config={"R": 2, "T": 4})
    inst.initial_positions = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
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
        "pickup": np.array([11.0, 0.0]),
        "drop": np.array([12.0, 0.0]),
        "duration": 1.0,
    }
    inst.tasks[2] = {
        "id": 2,
        "release_epoch": 1,
        "pickup": np.array([3.0, 0.0]),
        "drop": np.array([4.0, 0.0]),
        "duration": 1.0,
    }
    inst.tasks[3] = {
        "id": 3,
        "release_epoch": 1,
        "pickup": np.array([13.0, 0.0]),
        "drop": np.array([14.0, 0.0]),
        "duration": 1.0,
    }
    return inst


def _greedy_dispatch_action(sim_state, tasks_by_id):
    """Each available robot assigned to its nearest pending task, one at a time."""
    action = []
    used = set()
    for r in sorted(sim_state.available_robots):
        best, best_dist = None, float("inf")
        for j in sim_state.pending_tasks:
            if j in used:
                continue
            pickup = np.asarray(tasks_by_id[j]["pickup"], dtype=np.float64)
            d = float(np.linalg.norm(sim_state.positions[r] - pickup))
            if d < best_dist:
                best_dist = d
                best = j
        if best is not None:
            action.append((r, best))
            used.add(best)
    return action


def _run_to_termination(sim, tasks_by_id, max_steps=10000):
    steps = 0
    while not sim.is_terminal and steps < max_steps:
        action = _greedy_dispatch_action(sim.state, tasks_by_id)
        sim.step(action)
        steps += 1


def test_initial_state():
    inst = generate_instance(seed=0)
    sim = DynamicSimulator(inst)
    s = sim.state
    np.testing.assert_array_equal(s.positions, inst.initial_positions)
    assert np.all(s.busy_times == 0.0)
    assert s.available_robots == set(range(inst.R))
    expected_pending = {
        int(t["id"]) for t in inst.tasks if int(t["release_epoch"]) == 0
    }
    assert s.pending_tasks == expected_pending
    assert s.epoch == 0
    assert s.wall_clock == 0.0


def test_state_is_a_copy():
    inst = generate_instance(seed=0)
    sim = DynamicSimulator(inst)
    s1 = sim.state
    original_positions = s1.positions.copy()
    original_busy = s1.busy_times.copy()
    original_pending = set(s1.pending_tasks)
    s1.positions[0, 0] = 999.0
    s1.busy_times[:] = 999.0
    s1.pending_tasks.add(99999)
    s2 = sim.state
    np.testing.assert_array_equal(s2.positions, original_positions)
    np.testing.assert_array_equal(s2.busy_times, original_busy)
    assert s2.pending_tasks == original_pending


def test_idle_step_releases_epoch_one_tasks():
    inst = generate_instance(seed=0)
    sim = DynamicSimulator(inst)
    state_before = sim.state
    next_state, terminal = sim.step([])
    expected_pending = {
        int(t["id"])
        for t in inst.tasks
        if int(t["release_epoch"]) in (0, 1)
    }
    assert next_state.epoch == 1
    assert next_state.wall_clock == pytest.approx(5.0)
    assert next_state.pending_tasks == expected_pending
    assert np.all(next_state.busy_times == 0.0)
    np.testing.assert_array_equal(next_state.positions, state_before.positions)


def test_single_commitment_transition():
    inst = _make_two_robot_single_task_scenario()
    sim = DynamicSimulator(inst)
    state, terminal = sim.step([(0, 0)])
    expected_busy_0 = np.sqrt(50.0) + np.sqrt(200.0) + 2.0 - 5.0
    assert state.pending_tasks == set()
    assert state.busy_times[0] == pytest.approx(expected_busy_0)
    assert state.busy_times[1] == 0.0
    np.testing.assert_allclose(state.positions[0], [15.0, 15.0])
    np.testing.assert_allclose(state.positions[1], [10.0, 10.0])


# Path B (b)+(i): commits to physically-busy robots are allowed; the chained task's start_time back-dates to the predecessor's finish_wall_clock.
def test_robot_queue_ahead_is_allowed():
    inst = generate_instance(seed=0, config={"R": 1, "T": 2, "H": 5})
    inst.initial_positions = np.array([[0.0, 0.0]], dtype=np.float64)
    inst.tasks[0] = {
        "id": 0,
        "release_epoch": 0,
        "pickup": np.array([10.0, 0.0]),
        "drop": np.array([20.0, 0.0]),
        "duration": 1.0,
    }
    inst.tasks[1] = {
        "id": 1,
        "release_epoch": 0,
        "pickup": np.array([21.0, 0.0]),
        "drop": np.array([22.0, 0.0]),
        "duration": 1.0,
    }
    sim = DynamicSimulator(inst)
    sim.step([(0, 0)])
    # finish_wall_clock[0] = 10 + 10 + 1 = 21; at epoch 1 (wall_clock=5) robot 0 is busy, so this commit must not raise.
    sim.step([(0, 1)])
    last = sim.trajectory.commitments[-1]
    assert last["task_id"] == 1
    assert last["start_wall_clock"] == pytest.approx(21.0)
    # From position (20, 0) -> pickup (21, 0) = 1; pickup -> drop = 1; dur = 1.
    assert last["finish_wall_clock"] == pytest.approx(24.0)


def test_illegal_task_id_raises():
    inst = _make_two_robot_single_task_scenario()
    sim = DynamicSimulator(inst)
    with pytest.raises(ValueError):
        sim.step([(0, 999)])


def test_duplicate_robot_in_action_raises():
    inst = _make_two_robot_single_task_scenario()
    sim = DynamicSimulator(inst)
    with pytest.raises(ValueError):
        sim.step([(0, 0), (0, 1)])


def test_duplicate_task_in_action_raises():
    inst = _make_two_robot_single_task_scenario()
    sim = DynamicSimulator(inst)
    with pytest.raises(ValueError):
        sim.step([(0, 0), (1, 0)])


def test_trajectory_completes_under_greedy():
    inst = generate_instance(seed=0, config={"R": 2, "T": 5})
    sim = DynamicSimulator(inst)
    tasks_by_id = {int(t["id"]): t for t in inst.tasks}
    _run_to_termination(sim, tasks_by_id)
    assert sim.is_terminal
    assert not sim.trajectory.simulator_failure
    committed = [c["task_id"] for c in sim.trajectory.commitments]
    assert sorted(committed) == list(range(5))
    assert len(committed) == 5


def test_cost_on_completed_trajectory():
    inst = generate_instance(seed=0, config={"R": 2, "T": 5})
    sim = DynamicSimulator(inst)
    tasks_by_id = {int(t["id"]): t for t in inst.tasks}
    _run_to_termination(sim, tasks_by_id)
    cb = sim.compute_cost({"w_dist": 1.0, "w_make": 1.0, "w_bal": 1.0})
    for val in (cb.distance, cb.makespan, cb.imbalance, cb.combined):
        assert val >= 0.0
        assert np.isfinite(val)
    assert cb.combined == pytest.approx(cb.distance + cb.makespan + cb.imbalance)
    expected_distance = 0.0
    for c in sim.trajectory.commitments:
        d1 = float(np.linalg.norm(c["start_position"] - c["pickup"]))
        d2 = float(np.linalg.norm(c["pickup"] - c["drop"]))
        expected_distance += d1 + d2
    assert cb.distance == pytest.approx(expected_distance, abs=1e-9)


def test_extended_horizon_continues_past_H():
    inst = generate_instance(seed=0, config={"R": 1, "T": 5, "H": 2})
    inst.initial_positions = np.array([[0.0, 0.0]], dtype=np.float64)
    inst.tasks[0] = {
        "id": 0,
        "release_epoch": 0,
        "pickup": np.array([1.0, 0.0]),
        "drop": np.array([2.0, 0.0]),
        "duration": 5.0,
    }
    inst.tasks[1] = {
        "id": 1,
        "release_epoch": 1,
        "pickup": np.array([2.0, 0.0]),
        "drop": np.array([3.0, 0.0]),
        "duration": 5.0,
    }
    for k in range(2, 5):
        inst.tasks[k] = {
            "id": k,
            "release_epoch": 5,
            "pickup": np.array([10.0, 10.0]),
            "drop": np.array([20.0, 20.0]),
            "duration": 1.0,
        }
    sim = DynamicSimulator(inst)
    tasks_by_id = {int(t["id"]): t for t in inst.tasks}
    H = inst.H

    seen_unreleased_in_pending = False
    steps = 0
    while not sim.is_terminal and steps < 100:
        s = sim.state
        if any(j in s.pending_tasks for j in (2, 3, 4)):
            seen_unreleased_in_pending = True
        action = _greedy_dispatch_action(s, tasks_by_id)
        sim.step(action)
        steps += 1
    assert sim.is_terminal
    assert not sim.trajectory.simulator_failure
    assert sim.trajectory.epochs_run > H
    assert not seen_unreleased_in_pending
    committed = sorted(c["task_id"] for c in sim.trajectory.commitments)
    assert committed == [0, 1]


def test_wall_clock_cap_triggers_failure():
    inst = generate_instance(seed=0, config={"R": 1, "T": 2, "H": 2})
    inst.initial_positions = np.array([[0.0, 0.0]], dtype=np.float64)
    inst.tasks[0] = {
        "id": 0,
        "release_epoch": 0,
        "pickup": np.array([500.0, 0.0]),
        "drop": np.array([1500.0, 0.0]),
        "duration": 5.0,
    }
    inst.tasks[1] = {
        "id": 1,
        "release_epoch": 0,
        "pickup": np.array([1.0, 0.0]),
        "drop": np.array([2.0, 0.0]),
        "duration": 1.0,
    }
    sim = DynamicSimulator(inst)
    sim.step([(0, 0)])
    steps = 0
    while not sim.is_terminal and steps < 200:
        sim.step([])
        steps += 1
    assert sim.is_terminal
    assert sim.trajectory.simulator_failure is True
    # Task 0's finish time (1505) is far past the cap (50), so it counts as unserved alongside the never-committed task 1.
    assert 0 in sim.trajectory.unserved_task_ids
    assert len(sim.trajectory.unserved_task_ids) > 0
    cb = sim.compute_cost({"w_dist": 1.0, "w_make": 1.0, "w_bal": 1.0})
    assert isinstance(cb, CostBreakdown)
