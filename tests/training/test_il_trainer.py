"""Smoke tests for the Phase 1 method-one IL training loop, tiny configs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from instances.synthetic_generator import generate_instance
from simulator.dynamic_simulator import DynamicSimulator
from scoring.gnn_scorer import GNNScorer
from training.expert_dataset_generator import (
    _serialize_instance,
    _serialize_state_record,
)
from training.il_trainer import (
    ILDataset,
    TrainingConfig,
    annealed_epsilon,
    build_expert_permutation,
    reconstruct_state,
    train,
    train_one_step,
)


# Cache builders (Gurobi-free; mirror the dataset generator's record layout).


def _build_record(inst, decisions, seed):
    """Replay ``decisions`` on a fresh simulator and serialise the record in
    the same shape ``process_one_seed`` writes (instance, expert_decisions,
    expert_trajectory). Mirrors the generator's append-after-every-iteration
    loop so ``expert_trajectory`` has length ``len(decisions) + 1``."""
    sim = DynamicSimulator(inst)
    traj = [_serialize_state_record(sim)]
    for action in decisions:
        if not sim.is_terminal:
            sim.step([(int(r), int(j)) for (r, j) in action])
        traj.append(_serialize_state_record(sim))
    return {
        "R": int(inst.R),
        "T": int(inst.T),
        "seed": int(seed),
        "split": "train",
        "instance": _serialize_instance(inst),
        "expert_decisions": [[list(p) for p in a] for a in decisions],
        "expert_trajectory": traj,
    }


def _greedy_record(seed, config):
    """Build a record by greedily assigning each pending task to a free robot
    at every epoch. Yields several commit epochs without needing the MILP."""
    inst = generate_instance(seed, config=config)
    sim = DynamicSimulator(inst)
    traj = [_serialize_state_record(sim)]
    decisions = []
    while not sim.is_terminal:
        st = sim.state
        avail = sorted(st.available_robots)
        used = set()
        action = []
        for j in sorted(st.pending_tasks):
            for r in avail:
                if r not in used:
                    action.append((r, j))
                    used.add(r)
                    break
        decisions.append([list(p) for p in action])
        sim.step(action)
        traj.append(_serialize_state_record(sim))
    record = {
        "R": int(inst.R),
        "T": int(inst.T),
        "seed": int(seed),
        "split": "train",
        "instance": _serialize_instance(inst),
        "expert_decisions": decisions,
        "expert_trajectory": traj,
    }
    return record, inst


def _two_commit_instance():
    """R=2, T=2, H=4 hand-crafted instance: task 0 releases at epoch 0,
    task 1 at epoch 2, so a greedy expert commits at epochs 0 and 2 only."""
    inst = generate_instance(50000, config={"R": 2, "T": 2, "H": 4})
    inst.initial_positions = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    inst.tasks[0] = {
        "id": 0,
        "release_epoch": 0,
        "pickup": np.array([0.5, 0.0]),
        "drop": np.array([1.0, 0.0]),
        "duration": 0.5,
    }
    inst.tasks[1] = {
        "id": 1,
        "release_epoch": 2,
        "pickup": np.array([1.0, 1.0]),
        "drop": np.array([2.0, 1.0]),
        "duration": 0.5,
    }
    return inst


def _write_records(cache_dir, records):
    import json

    split_dir = cache_dir / "train"
    split_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        with (split_dir / f"seed{rec['seed']}.json").open("w") as f:
            json.dump(rec, f)


def test_ildataset_indexes_commit_epochs_only(tmp_path):
    """Two of the four epochs are empty; the index has only the two commit
    epochs (0 and 2)."""
    inst = _two_commit_instance()
    decisions = [[(0, 0)], [], [(1, 1)], []]
    rec = _build_record(inst, decisions, seed=50000)
    assert sum(1 for a in rec["expert_decisions"] if a) == 2
    _write_records(tmp_path, [rec])

    dataset = ILDataset(tmp_path, split="train")
    assert len(dataset) == 2
    assert [ex.epoch for ex in dataset.examples] == [0, 2]
    for ex in dataset.examples:
        assert len(ex.decision) >= 1  # commit epochs are non-empty


def test_build_expert_permutation_is_valid_permutation():
    R, T = 3, 4
    decision = [(0, 2), (2, 0)]  # robot1 idle; tasks 1,3 stay pending
    P = build_expert_permutation(decision, R, T)
    N = R + T
    assert P.shape == (N, N)
    unique = set(P.unique().tolist())
    assert unique.issubset({0.0, 1.0})
    assert torch.allclose(P.sum(dim=0), torch.ones(N))
    assert torch.allclose(P.sum(dim=1), torch.ones(N))
    # Assignments land in the top-left block.
    assert P[0, 2] == 1.0
    assert P[2, 0] == 1.0
    # Idle robot 1 on the top-right diagonal.
    assert P[1, T + 1] == 1.0


def test_reconstruct_state_round_trips():
    inst = _two_commit_instance()
    sim = DynamicSimulator(inst)
    sim.step([(0, 0)])  # advance one epoch so finish_times are non-trivial
    record = _serialize_state_record(sim)

    state = reconstruct_state(record)
    truth = sim.state
    assert state.epoch == truth.epoch
    assert state.wall_clock == truth.wall_clock
    assert state.pending_tasks == truth.pending_tasks
    assert state.available_robots == truth.available_robots
    assert np.allclose(state.positions, truth.positions)
    assert np.allclose(state.busy_times, truth.busy_times)


def test_train_one_step_runs_and_produces_gradients():
    rec, _ = _greedy_record(50010, config={"R": 3, "T": 6, "H": 8})
    import tempfile
    import pathlib

    tmp = pathlib.Path(tempfile.mkdtemp())
    _write_records(tmp, [rec])
    dataset = ILDataset(tmp, split="train")
    assert len(dataset) >= 2

    torch.manual_seed(0)
    model = GNNScorer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = [dataset[0], dataset[1]]

    loss, grad_norm = train_one_step(model, optimizer, batch, M=3)
    assert np.isfinite(loss)
    assert np.isfinite(grad_norm)

    grad_total = sum(
        float(p.grad.abs().sum())
        for p in model.parameters()
        if p.grad is not None
    )
    assert grad_total > 0.0


def test_train_end_to_end_saves_checkpoint(tmp_path):
    records = []
    for seed in (50020, 50021, 50022):
        rec, _ = _greedy_record(seed, config={"R": 3, "T": 8, "H": 12})
        records.append(rec)
    _write_records(tmp_path, records)

    ckpt = tmp_path / "checkpoints" / "il_method_one.pt"
    config = TrainingConfig(
        cache_dir=tmp_path,
        checkpoint_path=ckpt,
        split="train",
        batch_size=4,
        M=3,
        seed=0,
        log_every_steps=0,
    )
    history = train(config, num_steps=10)

    assert len(history) == 10
    assert all(np.isfinite(h["loss"]) for h in history)
    assert all(np.isfinite(h["grad_norm"]) for h in history)
    assert ckpt.exists()
    eps_seq = [h["epsilon"] for h in history]
    assert eps_seq[0] == pytest.approx(config.epsilon_initial)
    assert eps_seq[-1] == pytest.approx(config.epsilon_terminal)
    assert eps_seq[0] > eps_seq[-1]


def test_annealed_epsilon_is_exponential_decay():
    num_steps = 100
    eps_i, eps_t = 1.0, 0.01
    vals = [
        annealed_epsilon(s, num_steps, eps_i, eps_t) for s in range(num_steps)
    ]

    assert vals[0] == pytest.approx(eps_i)
    assert vals[-1] == pytest.approx(eps_t)

    assert all(vals[k] > vals[k + 1] for k in range(num_steps - 1))

    # Exponential decay: equal ratios between consecutive steps, i.e. log(epsilon) is linear in the step index.
    log_diffs = np.diff(np.log(vals))
    assert np.allclose(log_diffs, log_diffs[0])

    # With an explicit anneal window shorter than num_steps, epsilon reaches the terminal value at the window's end and holds thereafter.
    assert annealed_epsilon(49, 100, eps_i, eps_t, anneal_steps=50) == pytest.approx(eps_t)
    assert annealed_epsilon(80, 100, eps_i, eps_t, anneal_steps=50) == pytest.approx(eps_t)
