"""Smoke tests for the Phase 1 expert dataset generator on tiny configs."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from training import expert_dataset_generator as edg
from training.expert_dataset_generator import (
    LOCKED_WEIGHTS,
    PROTECTED_SEED_RANGES,
    SPLITS,
    extract_expert_decisions,
    generate_split,
    process_one_seed,
)
from instances.synthetic_generator import generate_instance
from anticipative.anticipative_milp import solve_anticipative
from simulator.dynamic_simulator import DynamicSimulator


SMALL_CONFIG = {"R": 3, "T": 5}


def test_seed_disjointness_across_splits_and_protected():
    all_split_seeds: list = []
    for split_seeds in SPLITS.values():
        all_split_seeds.extend(split_seeds)
    assert len(all_split_seeds) == len(set(all_split_seeds))
    split_set = set(all_split_seeds)
    for name, protected in PROTECTED_SEED_RANGES:
        overlap = split_set & set(protected)
        assert not overlap, f"split seeds overlap protected '{name}': {sorted(overlap)}"


def test_process_one_seed_round_trip_uses_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(edg, "CACHE_DIR", tmp_path)
    t0 = time.time()
    rec_first = process_one_seed(
        "train", 99000, config=SMALL_CONFIG, mip_time_limit=30
    )
    first_elapsed = time.time() - t0
    assert (tmp_path / "train" / "seed99000.json").exists()

    t1 = time.time()
    rec_second = process_one_seed(
        "train", 99000, config=SMALL_CONFIG, mip_time_limit=30
    )
    second_elapsed = time.time() - t1

    assert rec_second["seed"] == rec_first["seed"]
    assert (
        rec_second["milp_solution"]["objective_value"]
        == pytest.approx(rec_first["milp_solution"]["objective_value"])
    )
    # Second call should be much faster (cache hit).
    assert second_elapsed * 10 < max(first_elapsed, 0.001)


def test_expert_decisions_cover_all_tasks(monkeypatch, tmp_path):
    monkeypatch.setattr(edg, "CACHE_DIR", tmp_path)
    inst = generate_instance(99001, config=SMALL_CONFIG)
    sol = solve_anticipative(
        inst, LOCKED_WEIGHTS, time_limit_seconds=30, mip_gap=0.01
    )
    expert_decisions, expert_trajectory = extract_expert_decisions(inst, sol)
    # length is max(H, last_commit_epoch + 1); >= H always.
    assert len(expert_decisions) >= inst.H
    assert len(expert_trajectory) == len(expert_decisions) + 1
    total = sum(len(epoch_commits) for epoch_commits in expert_decisions)
    assert total == inst.T
    all_task_ids = []
    for epoch_commits in expert_decisions:
        for (r, j) in epoch_commits:
            all_task_ids.append(j)
    assert sorted(all_task_ids) == list(range(inst.T))


def test_replay_cost_matches_milp_objective(monkeypatch, tmp_path):
    """Replay cost is bounded above the MILP optimum by epoch-quantization slack.

    The availability-aware deferral fix no longer reproduces the MILP objective
    exactly. Deferring a chained commit to the epoch where the robot is free
    injects up to one Delta of idle slack per commit (the simulator back-dates
    start_time to ``max(robot_finish, epoch * Delta)``), which inflates makespan.
    The replay is therefore a feasible schedule that costs at least the offline
    optimum and, on instances from the locked distribution, at most ~10% above
    it. All serviceable tasks are still served.
    """
    monkeypatch.setattr(edg, "CACHE_DIR", tmp_path)
    inst = generate_instance(99002, config=SMALL_CONFIG)
    sol = solve_anticipative(
        inst, LOCKED_WEIGHTS, time_limit_seconds=30, mip_gap=0.01
    )
    expert_decisions, _ = extract_expert_decisions(inst, sol)
    sim = DynamicSimulator(inst)
    for action in expert_decisions:
        if sim.is_terminal:
            break
        sim.step(action)
    while not sim.is_terminal:
        sim.step([])
    cb = sim.compute_cost(LOCKED_WEIGHTS)
    assert not sim.trajectory.simulator_failure
    assert not sim.trajectory.unserved_task_ids  # all serviceable tasks served
    # Replay never beats the offline optimum (tiny negative tolerance covers the MILP objective's 1e-6 tie-break term).
    assert cb.combined >= sol.objective_value - 1e-2
    # ... and exceeds it by at most ~10% from quantization slack.
    assert cb.combined <= sol.objective_value * 1.10


def test_chained_commits_split_across_epochs(monkeypatch, tmp_path):
    """A hand-crafted instance where the MILP wants two short tasks on robot 0
    to chain within a single Delta window. The fixed extractor must defer the
    second commit to a later epoch so the robot is available (and the
    no-duplicate-robot rule holds). This is the worst case for quantization
    slack: deferring a tiny chained task by almost a full Delta inflates makespan
    by a large fraction of a tiny optimum, so the replay cost here exceeds the
    MILP objective by more than the ~10% typical of the locked distribution. The
    replay still serves every task and never beats the offline optimum."""
    import numpy as np
    monkeypatch.setattr(edg, "CACHE_DIR", tmp_path)

    inst = generate_instance(99006, config={"R": 2, "T": 3, "H": 20})
    inst.initial_positions = np.array(
        [[0.0, 0.0], [50.0, 50.0]], dtype=np.float64
    )
    inst.tasks[0] = {
        "id": 0,
        "release_epoch": 0,
        "pickup": np.array([1.0, 0.0]),
        "drop": np.array([2.0, 0.0]),
        "duration": 0.5,
    }
    inst.tasks[1] = {
        "id": 1,
        "release_epoch": 0,
        "pickup": np.array([3.0, 0.0]),
        "drop": np.array([4.0, 0.0]),
        "duration": 0.5,
    }
    inst.tasks[2] = {
        "id": 2,
        "release_epoch": 0,
        "pickup": np.array([50.0, 51.0]),
        "drop": np.array([51.0, 51.0]),
        "duration": 0.5,
    }
    sol = solve_anticipative(
        inst, LOCKED_WEIGHTS, time_limit_seconds=30, mip_gap=0.01
    )
    expert_decisions, _ = extract_expert_decisions(inst, sol)

    # At most one (r, j) per robot per epoch.
    for action in expert_decisions:
        robots_in_action = [r for (r, _) in action]
        assert len(robots_in_action) == len(set(robots_in_action))

    # Replay serves every task, never beats the optimum, and stays within the worst-case quantization slack for this chain (see docstring).
    sim = DynamicSimulator(inst)
    for action in expert_decisions:
        if sim.is_terminal:
            break
        sim.step(action)
    while not sim.is_terminal:
        sim.step([])
    cb = sim.compute_cost(LOCKED_WEIGHTS)
    assert not sim.trajectory.simulator_failure
    assert not sim.trajectory.unserved_task_ids  # all serviceable tasks served
    assert cb.combined >= sol.objective_value - 1e-2
    assert cb.combined <= sol.objective_value * 1.25


def test_generate_split_with_max_seeds_writes_files(monkeypatch, tmp_path):
    monkeypatch.setattr(edg, "CACHE_DIR", tmp_path)
    test_seeds = [99003, 99004, 99005]
    generate_split(
        "train",
        test_seeds,
        max_seeds=2,
        config=SMALL_CONFIG,
        mip_time_limit=30,
    )
    # Only the first two should have been written.
    assert (tmp_path / "train" / "seed99003.json").exists()
    assert (tmp_path / "train" / "seed99004.json").exists()
    assert not (tmp_path / "train" / "seed99005.json").exists()
