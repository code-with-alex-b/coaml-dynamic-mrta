"""Smoke tests for the Phase 1 perturbed Fenchel-Young loss."""

from __future__ import annotations

import numpy as np
import torch

from instances.synthetic_generator import generate_instance
from losses.fenchel_young import (
    fenchel_young_loss,
    hungarian_max,
    perturbed_expected_argmax,
)
from scoring.gnn_scorer import GNNScorer
from simulator.dynamic_simulator import SimulatorState


def test_output_shape():
    torch.manual_seed(0)
    Theta = torch.randn(5, 5, requires_grad=True)
    P_star = torch.eye(5)
    loss, grad = fenchel_young_loss(Theta, P_star, M=5)
    assert grad.shape == (5, 5)
    assert loss.dim() == 0  # scalar tensor


def test_hungarian_wrapper_returns_permutation_matrix():
    torch.manual_seed(1)
    score = torch.randn(4, 4)
    P = hungarian_max(score)
    assert P.shape == (4, 4)
    unique = set(P.unique().tolist())
    assert unique.issubset({0.0, 1.0})
    assert torch.allclose(P.sum(dim=0), torch.ones(4))
    assert torch.allclose(P.sum(dim=1), torch.ones(4))


def test_perturbed_expected_argmax_is_doubly_stochastic():
    torch.manual_seed(2)
    Theta = torch.randn(5, 5)
    y_hat = perturbed_expected_argmax(Theta, epsilon=1.0, M=50)
    assert y_hat.shape == (5, 5)
    assert (y_hat >= 0.0).all() and (y_hat <= 1.0).all()
    # Each Gumbel sample produces a hard permutation, so their average is exactly doubly stochastic up to float rounding.
    assert torch.allclose(y_hat.sum(dim=0), torch.ones(5), atol=1e-6)
    assert torch.allclose(y_hat.sum(dim=1), torch.ones(5), atol=1e-6)


def test_gradient_small_when_theta_already_matches_expert():
    torch.manual_seed(3)
    N = 4
    # Diagonal-dominant Theta so the unperturbed argmax is the identity and Gumbel noise (scale ~1) cannot flip it.
    Theta = 10.0 * torch.eye(N)
    P_star = torch.eye(N)
    _loss, grad = fenchel_young_loss(Theta, P_star, epsilon=1.0, M=50)
    assert grad.abs().max().item() < 0.5


def test_gradient_pushes_theta_toward_expert():
    torch.manual_seed(4)
    N = 4
    # Diagonal is strongly negative so Hungarian avoids it, but the expert wants the identity; diagonal cells should get a strongly negative gradient so gradient descent raises Theta there.
    Theta = -100.0 * torch.eye(N)
    P_star = torch.eye(N)
    _loss, grad = fenchel_young_loss(Theta, P_star, epsilon=1.0, M=20)
    for r in range(N):
        assert grad[r, r].item() < -0.5, (
            f"diagonal grad[{r},{r}] = {grad[r, r].item()} not pushing toward expert"
        )


def test_integration_with_gnn_scorer():
    torch.manual_seed(5)
    inst = generate_instance(seed=0, config={"R": 3, "T": 5})
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
    assert Theta.shape == (8, 8)
    P_star = torch.eye(8)
    loss, grad = fenchel_young_loss(Theta, P_star, epsilon=1.0, M=5)
    assert grad.shape == (8, 8)
    Theta.backward(grad)
    encoder_has_grad = any(
        p.grad is not None and (p.grad.abs().sum() > 0)
        for p in scorer.encoder.parameters()
    )
    scorer_head_has_grad = any(
        p.grad is not None and (p.grad.abs().sum() > 0)
        for p in scorer.scorer.parameters()
    )
    assert encoder_has_grad
    assert scorer_head_has_grad
