"""Smoke tests for the Gumbel-perturbed Hungarian sampler (method two)."""

from __future__ import annotations

import torch

from instances.synthetic_generator import generate_instance
from simulator.dynamic_simulator import DynamicSimulator
from scoring.gnn_scorer import GNNScorer
from sampling.perturbed_hungarian import (
    gumbel_sample,
    perturbed_hungarian_sample,
)


def _random_theta(R: int, T: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    N = R + T
    return torch.randn(N, N)


def test_gumbel_sample_shape_and_finite():
    g = gumbel_sample((4, 6))
    assert g.shape == (4, 6)
    assert torch.isfinite(g).all()


def test_sample_returns_valid_permutation_matrix():
    R, T = 3, 5
    N = R + T
    Theta = _random_theta(R, T, seed=1)
    P, _, _ = perturbed_hungarian_sample(Theta, epsilon=1.0, R=R, T=T)

    assert P.shape == (N, N)
    assert set(P.unique().tolist()).issubset({0.0, 1.0})
    assert torch.allclose(P.sum(dim=0), torch.ones(N))
    assert torch.allclose(P.sum(dim=1), torch.ones(N))


def test_dispatcher_action_only_top_left_pairs():
    R, T = 4, 6
    Theta = _random_theta(R, T, seed=2)
    _, action, _ = perturbed_hungarian_sample(Theta, epsilon=1.0, R=R, T=T)

    assert isinstance(action, list)
    for pair in action:
        r, c = pair
        assert 0 <= r < R
        assert 0 <= c < T
    # At most min(R, T) commits, no repeated robots or tasks.
    robots = [r for r, _ in action]
    tasks = [c for _, c in action]
    assert len(robots) == len(set(robots))
    assert len(tasks) == len(set(tasks))
    assert len(action) <= min(R, T)


def test_yhat_is_doubly_stochastic():
    R, T = 3, 5
    N = R + T
    Theta = _random_theta(R, T, seed=3)
    _, _, y_hat = perturbed_hungarian_sample(
        Theta, epsilon=1.0, R=R, T=T, K_sink=50
    )

    assert y_hat.shape == (N, N)
    assert torch.allclose(y_hat.sum(dim=0), torch.ones(N), atol=1e-3)
    assert torch.allclose(y_hat.sum(dim=1), torch.ones(N), atol=1e-3)


def test_determinism_with_fixed_seed():
    R, T = 3, 5
    Theta = _random_theta(R, T, seed=4)

    torch.manual_seed(123)
    P1, action1, _ = perturbed_hungarian_sample(Theta, epsilon=1.0, R=R, T=T)
    torch.manual_seed(123)
    P2, action2, _ = perturbed_hungarian_sample(Theta, epsilon=1.0, R=R, T=T)

    assert torch.equal(P1, P2)
    assert action1 == action2


def test_integration_with_gnn_scorer():
    R, T = 3, 5
    N = R + T
    inst = generate_instance(seed=0, config={"R": R, "T": T})
    sim = DynamicSimulator(inst)
    torch.manual_seed(0)
    scorer = GNNScorer()
    with torch.no_grad():
        Theta = scorer(sim.state, inst)
    assert Theta.shape == (N, N)

    P, action, y_hat = perturbed_hungarian_sample(Theta, epsilon=1.0, R=R, T=T)
    assert P.shape == (N, N)
    assert y_hat.shape == (N, N)
    assert isinstance(action, list)
    for pair in action:
        assert isinstance(pair, tuple) and len(pair) == 2
