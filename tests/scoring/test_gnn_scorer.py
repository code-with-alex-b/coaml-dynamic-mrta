"""Smoke tests for the Phase 1 GNN scoring head."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from instances.synthetic_generator import generate_instance
from simulator.dynamic_simulator import DynamicSimulator, SimulatorState
from scoring.gnn_scorer import (
    GNNScorer,
    HIDDEN_DIM,
    M_MASK,
    PairwiseScorer,
)


def test_pairwise_scorer_output_shape():
    scorer = PairwiseScorer(hidden_dim=HIDDEN_DIM)
    robot_emb = torch.randn(3, HIDDEN_DIM)
    task_emb = torch.randn(5, HIDDEN_DIM)
    out = scorer(robot_emb, task_emb)
    assert out.shape == (3, 5)


def test_gnn_scorer_output_shape():
    inst = generate_instance(seed=0, config={"R": 3, "T": 5})
    sim = DynamicSimulator(inst)
    scorer = GNNScorer()
    theta = scorer(sim.state, inst)
    assert theta.shape == (3 + 5, 3 + 5)


def test_augmented_matrix_block_structure():
    inst = generate_instance(seed=0, config={"R": 3, "T": 5})
    sim = DynamicSimulator(inst)
    scorer = GNNScorer()
    theta = scorer(sim.state, inst)
    R, T = 3, 5
    top_right = theta[:R, T : T + R]
    expected_top_right = -M_MASK * (1.0 - torch.eye(R))
    assert torch.allclose(top_right, expected_top_right)
    bottom_left = theta[R : R + T, :T]
    expected_bottom_left = -M_MASK * (1.0 - torch.eye(T))
    assert torch.allclose(bottom_left, expected_bottom_left)
    bottom_right = theta[R : R + T, T : T + R]
    assert torch.allclose(bottom_right, torch.zeros(T, R))


def test_masking_on_top_left_block():
    inst = generate_instance(seed=0, config={"R": 3, "T": 5})
    state = SimulatorState(
        positions=np.zeros((3, 2), dtype=np.float64),
        busy_times=np.array([10.0, 0.0, 0.0], dtype=np.float64),
        pending_tasks={0, 1, 3, 4},  # task 2 not pending
        available_robots={1, 2},  # robot 0 busy
        epoch=0,
        wall_clock=0.0,
    )
    scorer = GNNScorer()
    theta = scorer(state, inst)
    R, T = 3, 5
    top_left = theta[:R, :T]
    # Row 0 (busy robot) entirely -M_MASK
    assert torch.allclose(top_left[0], torch.full((T,), -M_MASK))
    # Column 2 (non-pending task) entirely -M_MASK
    assert torch.allclose(top_left[:, 2], torch.full((R,), -M_MASK))
    # Other (available robot, pending task) cells are finite GNN scores, well clear of -M_MASK.
    for r in (1, 2):
        for j in (0, 1, 3, 4):
            assert top_left[r, j].item() > -M_MASK + 1.0


def test_queue_ahead_mask_unmasks_busy_robot():
    """With use_queue_ahead_mask=True a busy robot's task cells keep their GNN
    scores instead of being masked to -M_MASK; non-pending task columns stay
    masked."""
    inst = generate_instance(seed=0, config={"R": 3, "T": 5})
    state = SimulatorState(
        positions=np.zeros((3, 2), dtype=np.float64),
        busy_times=np.array([10.0, 0.0, 0.0], dtype=np.float64),
        pending_tasks={0, 1, 3, 4},  # task 2 not pending
        available_robots={1, 2},  # robot 0 busy
        epoch=0,
        wall_clock=0.0,
    )
    scorer = GNNScorer(use_queue_ahead_mask=True)
    theta = scorer(state, inst)
    R, T = 3, 5
    top_left = theta[:R, :T]
    # Busy robot 0's pending-task cells are now finite GNN scores, not -M_MASK.
    for j in (0, 1, 3, 4):
        assert top_left[0, j].item() > -M_MASK + 1.0
    # The non-pending task column (task 2) stays masked for every robot.
    assert torch.allclose(top_left[:, 2], torch.full((R,), -M_MASK))
    default_scorer = GNNScorer(use_queue_ahead_mask=False)
    default_scorer.load_state_dict(scorer.state_dict())
    default_theta = default_scorer(state, inst)
    assert torch.allclose(default_theta[0, :T], torch.full((T,), -M_MASK))


def test_feature_normalisation():
    inst = generate_instance(seed=0)
    sim = DynamicSimulator(inst)
    scorer = GNNScorer()
    robot_feats, task_feats = scorer.build_features(sim.state, inst)
    # Robot positions normalised to [0, 1]
    assert (robot_feats[:, 0] >= 0.0).all() and (robot_feats[:, 0] <= 1.0).all()
    assert (robot_feats[:, 1] >= 0.0).all() and (robot_feats[:, 1] <= 1.0).all()
    # Task positions normalised to [0, 1] (zero for non-pending tasks)
    for col in range(4):
        assert (task_feats[:, col] >= 0.0).all()
        assert (task_feats[:, col] <= 1.0).all()
    pending = sorted(sim.state.pending_tasks)
    if pending:
        # Duration / mu_d for truncnorm in [1, 10] / 5.0 lies in [0.2, 2.0]
        dur_feats = task_feats[pending, 4]
        assert (dur_feats >= 0.2 - 1e-6).all()
        assert (dur_feats <= 2.0 + 1e-6).all()
    # Release epoch / H in [0, 1]
    assert (task_feats[:, 5] >= 0.0).all() and (task_feats[:, 5] <= 1.0).all()


def test_determinism():
    inst = generate_instance(seed=0, config={"R": 3, "T": 5})
    sim = DynamicSimulator(inst)
    state = sim.state
    torch.manual_seed(42)
    scorer1 = GNNScorer()
    torch.manual_seed(42)
    scorer2 = GNNScorer()
    theta1 = scorer1(state, inst)
    theta2 = scorer2(state, inst)
    assert torch.allclose(theta1, theta2)


def test_gradient_flow():
    inst = generate_instance(seed=0, config={"R": 3, "T": 5})
    state = SimulatorState(
        positions=np.zeros((3, 2), dtype=np.float64),
        busy_times=np.zeros(3, dtype=np.float64),
        pending_tasks={0, 1, 2},
        available_robots={0, 1, 2},
        epoch=0,
        wall_clock=0.0,
    )
    scorer = GNNScorer()
    theta = scorer(state, inst)
    loss = theta.sum()
    loss.backward()
    encoder_has_grad = any(
        p.grad is not None and (p.grad.abs().sum() > 0)
        for p in scorer.encoder.parameters()
    )
    scorer_has_grad = any(
        p.grad is not None and (p.grad.abs().sum() > 0)
        for p in scorer.scorer.parameters()
    )
    assert encoder_has_grad
    assert scorer_has_grad


def _assemble_theta_old(scores, state, R, T):
    """Reference implementation of the pre-commit_bias augmented assembly.

    Mirrors ``GNNScorer.build_augmented_matrix`` exactly as it stood before the
    commit_bias parameter was added, so tests can assert bit-identical output on
    the default path.
    """
    device = scores.device
    dtype = scores.dtype
    available = torch.zeros(R, dtype=torch.bool, device=device)
    for r in state.available_robots:
        available[int(r)] = True
    pending = torch.zeros(T, dtype=torch.bool, device=device)
    for j in state.pending_tasks:
        pending[int(j)] = True
    valid = available.unsqueeze(1) & pending.unsqueeze(0)
    masked = scores.masked_fill(~valid, -M_MASK)
    top_right = -M_MASK * (
        torch.ones((R, R), dtype=dtype, device=device)
        - torch.eye(R, dtype=dtype, device=device)
    )
    bottom_left = -M_MASK * (
        torch.ones((T, T), dtype=dtype, device=device)
        - torch.eye(T, dtype=dtype, device=device)
    )
    bottom_right = torch.zeros((T, R), dtype=dtype, device=device)
    top = torch.cat([masked, top_right], dim=1)
    bottom = torch.cat([bottom_left, bottom_right], dim=1)
    return torch.cat([top, bottom], dim=0)


def test_commit_bias_default_is_zero():
    scorer = GNNScorer()
    assert scorer.commit_bias == 0.0


def test_commit_bias_zero_bit_identical_to_old_code():
    """commit_bias=0.0 must be bit-identical to the pre-bias assembly."""
    inst = generate_instance(seed=0, config={"R": 3, "T": 5})
    state = SimulatorState(
        positions=np.zeros((3, 2), dtype=np.float64),
        busy_times=np.array([10.0, 0.0, 0.0], dtype=np.float64),
        pending_tasks={0, 1, 3, 4},  # task 2 not pending
        available_robots={1, 2},  # robot 0 busy
        epoch=0,
        wall_clock=0.0,
    )
    torch.manual_seed(7)
    scorer = GNNScorer(commit_bias=0.0)
    theta = scorer(state, inst)

    # Recompute raw scores the same way forward() does, then assemble via the reference old-code path and compare exactly.
    robot_feats, task_feats = scorer.build_features(state, inst)
    robot_emb, task_emb = scorer.encoder(robot_feats, task_feats)
    scores = scorer.scorer(robot_emb, task_emb)
    theta_old = _assemble_theta_old(scores, state, inst.R, inst.T)
    assert torch.equal(theta, theta_old)


def test_commit_bias_adds_to_valid_block_only():
    inst = generate_instance(seed=0, config={"R": 3, "T": 5})
    state = SimulatorState(
        positions=np.zeros((3, 2), dtype=np.float64),
        busy_times=np.array([10.0, 0.0, 0.0], dtype=np.float64),
        pending_tasks={0, 1, 3, 4},  # task 2 not pending
        available_robots={1, 2},  # robot 0 busy
        epoch=0,
        wall_clock=0.0,
    )
    bias = 2.5
    torch.manual_seed(7)
    scorer0 = GNNScorer(commit_bias=0.0)
    torch.manual_seed(7)
    scorerb = GNNScorer(commit_bias=bias)
    theta0 = scorer0(state, inst)
    thetab = scorerb(state, inst)

    R, T = inst.R, inst.T
    valid_rows = (1, 2)
    valid_cols = (0, 1, 3, 4)
    # Valid (available robot, pending task) cells shift up by exactly the bias.
    for r in valid_rows:
        for j in valid_cols:
            assert thetab[r, j].item() == pytest.approx(
                theta0[r, j].item() + bias
            )
    # Masked top-left cells (busy robot / non-pending task) are untouched.
    assert torch.equal(thetab[0, :T], theta0[0, :T])
    assert torch.equal(thetab[:R, 2], theta0[:R, 2])
    # The three augmenting blocks are untouched.
    assert torch.equal(thetab[:R, T:], theta0[:R, T:])
    assert torch.equal(thetab[R:, :], theta0[R:, :])


def test_commit_bias_not_in_state_dict():
    scorer0 = GNNScorer(commit_bias=0.0)
    scorerb = GNNScorer(commit_bias=5.0)
    # commit_bias is not a parameter or buffer, so the key set is unchanged.
    assert set(scorer0.state_dict().keys()) == set(scorerb.state_dict().keys())
    assert not any("commit_bias" in k for k in scorerb.state_dict().keys())


def test_get_init_kwargs_includes_commit_bias():
    scorer = GNNScorer(commit_bias=3.0)
    kwargs = scorer.get_init_kwargs()
    assert kwargs["commit_bias"] == 3.0
    rebuilt = GNNScorer(**kwargs)
    assert rebuilt.commit_bias == 3.0


def test_existing_checkpoint_reusable_with_commit_bias_zero(tmp_path):
    """Loading a saved checkpoint into commit_bias=0.0 reproduces old output."""
    inst = generate_instance(seed=0, config={"R": 3, "T": 5})
    sim = DynamicSimulator(inst)
    state = sim.state

    torch.manual_seed(123)
    original = GNNScorer()  # stands in for a pre-bias trained model
    theta_reference = original(state, inst)

    ckpt = tmp_path / "scorer.pt"
    torch.save(original.state_dict(), ckpt)

    # Fresh scorer with the new signature, default bias, loads the old weights.
    loaded = GNNScorer(commit_bias=0.0)
    loaded.load_state_dict(torch.load(ckpt))
    theta_loaded = loaded(state, inst)

    assert torch.equal(theta_loaded, theta_reference)


def test_integration_with_simulator():
    inst = generate_instance(seed=100)
    sim = DynamicSimulator(inst)
    sim.step([])  # one idle step releases epoch-1 tasks
    scorer = GNNScorer()
    theta = scorer(sim.state, inst)
    R, T = inst.R, inst.T
    assert theta.shape == (R + T, R + T)
    top_right = theta[:R, T : T + R]
    expected_top_right = -M_MASK * (1.0 - torch.eye(R))
    assert torch.allclose(top_right, expected_top_right)
