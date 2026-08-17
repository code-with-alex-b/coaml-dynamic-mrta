"""Smoke tests for the Phase 1 Sinkhorn operator."""

from __future__ import annotations

import numpy as np
import torch

from co_layer.sinkhorn import K_SINK_DEFAULT, sinkhorn_log, sinkhorn_log_batched
from instances.synthetic_generator import generate_instance
from scoring.gnn_scorer import GNNScorer
from simulator.dynamic_simulator import SimulatorState


def test_output_shape():
    Theta = torch.randn(5, 5)
    P = sinkhorn_log(Theta)
    assert P.shape == (5, 5)


def test_doubly_stochastic():
    torch.manual_seed(0)
    Theta = torch.randn(4, 4)
    P = sinkhorn_log(Theta)
    row_sums = P.sum(dim=1)
    col_sums = P.sum(dim=0)
    assert torch.allclose(row_sums, torch.ones(4), atol=1e-4)
    assert torch.allclose(col_sums, torch.ones(4), atol=1e-4)


def test_permutation_limit_at_low_tau():
    # Diagonal-dominant Theta: optimum is the identity permutation.
    Theta = torch.eye(3) * 10.0
    P = sinkhorn_log(Theta, tau=0.01)
    row_max = P.max(dim=1).values
    assert (row_max > 0.99).all()
    # Diagonal should be the argmax of each row.
    assert (P.argmax(dim=1) == torch.arange(3)).all()


def test_handles_mask_without_numerical_issues():
    Theta = -100.0 * (torch.ones(4, 4) - torch.eye(4))  # diag 0, off-diag -100
    P = sinkhorn_log(Theta, tau=1.0)
    assert torch.isfinite(P).all()
    assert not torch.isnan(P).any()
    row_sums = P.sum(dim=1)
    col_sums = P.sum(dim=0)
    assert torch.allclose(row_sums, torch.ones(4), atol=1e-4)
    assert torch.allclose(col_sums, torch.ones(4), atol=1e-4)


def test_gradient_flow():
    torch.manual_seed(0)
    Theta = torch.randn(5, 5, requires_grad=True)
    P = sinkhorn_log(Theta)
    loss = P.sum()
    loss.backward()
    assert Theta.grad is not None
    assert torch.isfinite(Theta.grad).all()


def test_determinism():
    torch.manual_seed(0)
    Theta = torch.randn(6, 6)
    P1 = sinkhorn_log(Theta)
    P2 = sinkhorn_log(Theta)
    assert torch.equal(P1, P2)


def test_batched_matches_unbatched():
    torch.manual_seed(0)
    Theta = torch.randn(3, 4, 4)
    P_batched = sinkhorn_log_batched(Theta)
    P_stacked = torch.stack([sinkhorn_log(Theta[b]) for b in range(3)])
    assert torch.allclose(P_batched, P_stacked, atol=1e-5)


def test_integration_with_gnn_scorer():
    inst = generate_instance(seed=0, config={"R": 3, "T": 5})
    # All tasks pending so every column has enough unmasked entries for K_SINK_DEFAULT=20 to converge tightly; a partially-masked pattern would leave single-entry columns that can't rebalance in 20 iterations.
    state = SimulatorState(
        positions=np.zeros((3, 2), dtype=np.float64),
        busy_times=np.zeros(3, dtype=np.float64),
        pending_tasks={0, 1, 2, 3, 4},
        available_robots={0, 1, 2},
        epoch=0,
        wall_clock=0.0,
    )
    scorer = GNNScorer()
    Theta = scorer(state, inst)
    assert Theta.shape == (3 + 5, 3 + 5)
    P = sinkhorn_log(Theta, num_iters=K_SINK_DEFAULT)
    assert torch.allclose(P.sum(dim=1), torch.ones(8), atol=1e-4)
    assert torch.allclose(P.sum(dim=0), torch.ones(8), atol=1e-4)
    loss = P.sum()
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
