"""Gumbel-perturbed Hungarian sampler for method two.

The perturbed combinatorial optimisation layer. Given a doubly augmented score matrix
``Theta`` and a perturbation magnitude ``epsilon``, draws Gumbel(0, 1) noise, solves
Hungarian on the perturbed matrix, and reads the dispatcher action off the top-left
R by T subblock.

For the REINFORCE score-function gradient the per-epoch term is

    grad_w log p_eps(a_t | theta_t, s_t)
        = (1 / epsilon) * (a_t - y_hat_eps(theta_t, s_t))^T grad_w theta_t

where ``y_hat_eps = E_G[Hungarian(Theta + epsilon G)]`` is the expected perturbed
argmax, approximated here by the Sinkhorn operator at temperature ``epsilon``, which
stays differentiable in ``Theta`` so the gradient flows through the scorer.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from scipy.optimize import linear_sum_assignment

from co_layer.sinkhorn import sinkhorn_log


def gumbel_sample(shape: tuple, device: torch.device = None) -> torch.Tensor:
    """Draw a Gumbel(0, 1) tensor via the inverse-CDF trick.

    ``-log(-log(U))`` with ``U ~ Uniform(0, 1)``, clamped away from 0 and 1 to
    avoid infinities.
    """
    U = torch.rand(*shape, device=device).clamp(min=1e-10, max=1 - 1e-10)
    return -torch.log(-torch.log(U))


def perturbed_hungarian_sample(
    Theta: torch.Tensor,
    epsilon: float,
    R: int,
    T: int,
    K_sink: int = 20,
) -> Tuple[torch.Tensor, List[Tuple[int, int]], torch.Tensor]:
    """Sample a hard matching from the perturbed Hungarian distribution.

    Args:
        Theta: shape (R+T, R+T) score matrix from the GNN scorer.
        epsilon: perturbation magnitude (> 0).
        R, T: numbers of robots and tasks.
        K_sink: Sinkhorn iterations for the ``y_hat`` computation. Default 20.

    Returns:
        P: shape (R+T, R+T) hard permutation matrix sampled via perturbed
           Hungarian.
        dispatcher_action: list of (r, j) pairs from the top-left R by T
           subblock of P.
        y_hat: shape (R+T, R+T) Sinkhorn-approximated expected perturbed argmax
           at temperature epsilon, used for the REINFORCE gradient.
    """
    G = gumbel_sample(Theta.shape, device=Theta.device)

    # Detach for Hungarian: the score-function gradient flows through Theta via y_hat, not through the sampled P.
    Theta_pert = Theta.detach() + epsilon * G

    # Hungarian maximises <Theta, P>; scipy minimises cost, so negate.
    cost = (-Theta_pert).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)

    P = torch.zeros_like(Theta)
    for r, c in zip(row_ind, col_ind):
        P[int(r), int(c)] = 1.0

    dispatcher_action = [
        (int(r), int(c))
        for r, c in zip(row_ind, col_ind)
        if int(r) < R and int(c) < T
    ]

    # Sinkhorn approximates the expected perturbed argmax while staying differentiable in Theta.
    y_hat = sinkhorn_log(Theta, tau=epsilon, num_iters=K_sink)

    return P, dispatcher_action, y_hat
