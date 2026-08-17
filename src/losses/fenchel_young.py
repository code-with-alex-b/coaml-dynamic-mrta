"""Perturbed Fenchel-Young loss for structured prediction.

The target is a permutation matrix in the Birkhoff polytope B. The linear oracle is
the Hungarian algorithm, max over P in B of <Theta, P>. Following Berthet et al.
(2020), the regulariser is an implicit perturbation with iid Gumbel(0, 1) noise. The
conjugate is

    Omega^star_eps(Theta) = E_Z [ max_{P in B} <Theta + eps Z, P> ]

and the gradient of the loss with respect to Theta is

    grad_Theta L = y_hat_eps(Theta) - P_star
                 = E_Z [ argmax_{P in B} <Theta + eps Z, P> ] - P_star

approximated by Monte Carlo over M Gumbel samples. Training needs only the gradient.
The loss value is reported for monitoring and is a lower bound that can go negative;
pass ``detail_sink`` to ``fenchel_young_loss`` for the true non-negative loss.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment


EPSILON_DEFAULT = 1.0
M_SAMPLES_DEFAULT = 10


def hungarian_max(score_matrix: torch.Tensor) -> torch.Tensor:
    """Hungarian (max) on a 2D score matrix. Returns a hard permutation matrix.

    scipy.optimize.linear_sum_assignment minimises cost, so the input is
    negated. The returned permutation is non-differentiable (detached).
    """
    cost = (-score_matrix.detach()).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    P = torch.zeros_like(score_matrix)
    P[row_ind, col_ind] = 1.0
    return P


def perturbed_expected_argmax(
    Theta: torch.Tensor,
    epsilon: float = EPSILON_DEFAULT,
    M: int = M_SAMPLES_DEFAULT,
    P_ref: Optional[torch.Tensor] = None,
    detail_sink: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Monte Carlo estimate of y_hat_eps(Theta) per Berthet et al. (2020).

    Draws ``M`` Gumbel(0, 1) tensors of shape ``Theta`` by inverse CDF, solves
    Hungarian on each ``Theta + eps * Z_m``, and averages the permutations.

    ``detail_sink`` is a pure observer. When a dict is passed, per-sample scalars are
    accumulated into it. It changes neither the return value nor the number or order of
    RNG draws, so a run with a sink attached follows a bit-identical trajectory to one
    without. ``P_ref`` is read only to fill the sink.
    """
    N = int(Theta.shape[0])
    accumulator = torch.zeros_like(Theta)
    omega_star = 0.0
    ref_noise = 0.0
    for _m in range(M):
        U = torch.rand(N, N, device=Theta.device).clamp(min=1e-10, max=1 - 1e-10)
        Z = -torch.log(-torch.log(U))
        perturbed = Theta.detach() + epsilon * Z
        P_m = hungarian_max(perturbed)
        accumulator = accumulator + P_m
        if detail_sink is not None:
            # O(N^2) against the O(N^3) Hungarian solve above, reusing tensors already in hand.
            omega_star += float((perturbed * P_m).sum())
            if P_ref is not None:
                ref_noise += float((Z * P_ref).sum())
    if detail_sink is not None:
        detail_sink["omega_star"] = omega_star / float(M)
        detail_sink["omega_target"] = -epsilon * ref_noise / float(M)
    return accumulator / float(M)


def fenchel_young_loss(
    Theta: torch.Tensor,
    P_star: torch.Tensor,
    epsilon: float = EPSILON_DEFAULT,
    M: int = M_SAMPLES_DEFAULT,
    detail_sink: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the Fenchel-Young loss value and its gradient w.r.t. Theta.

    Args:
        Theta: shape (N, N), differentiable in the upstream graph.
        P_star: shape (N, N) expert permutation matrix, binary.
        epsilon: perturbation magnitude.
        M: number of Gumbel samples.
        detail_sink: optional dict filled with the diagnostics below, purely an observer.

    Returns:
        loss_value: scalar tensor, ``<Theta, y_hat> - <Theta, P_star>``. A lower bound on
            the true loss, so it may be negative and has no meaningful zero.
        gradient: shape (N, N), ``y_hat_eps(Theta) - P_star``.

    The true perturbed Fenchel-Young loss is

        L_eps(Theta, P*) = Omega^star_eps(Theta) + Omega_eps(P*) - <Theta, P*>  >= 0

    with equality iff ``y_hat_eps(Theta) == P*``. ``loss_value`` omits both
    ``Omega_eps(P*)`` and the residual ``eps E_Z[<Z, P_Z>]`` inside
    ``Omega^star_eps``.

    With ``detail_sink`` passed, the three terms are recorded. Keys::

        omega_star   MC estimate of Omega^star_eps(Theta), mean_m <Theta + eps Z_m, P_m>
        omega_target Omega_eps(P*) = -eps E_Z[<Z, P*>], on the same Gumbel draws
        theta_target <Theta, P*>
        fy_true      omega_star + omega_target - theta_target
        fy_reported  the returned loss_value

    ``Omega_eps(P*)`` has a closed form, ``-eps * N * gamma`` for Gumbel(0, 1) with gamma
    the Euler-Mascheroni constant, but is estimated from the sampled ``Z_m`` instead.
    Sharing the draws with ``omega_star`` makes ``fy_true`` identically the sample mean of

        max_P <Theta + eps Z_m, P> - <Theta + eps Z_m, P*>

    whose every summand is non-negative because ``P*`` is feasible in that max, so
    ``fy_true`` is non-negative sample by sample rather than only in expectation.

    Typical use::

        loss_value, grad = fenchel_young_loss(Theta, P_star)
        Theta.backward(grad)
    """
    y_hat = perturbed_expected_argmax(
        Theta,
        epsilon=epsilon,
        M=M,
        P_ref=P_star if detail_sink is not None else None,
        detail_sink=detail_sink,
    )
    gradient = y_hat - P_star
    loss_value = (Theta.detach() * y_hat).sum() - (Theta.detach() * P_star).sum()
    if detail_sink is not None:
        theta_target = float((Theta.detach() * P_star).sum())
        detail_sink["theta_target"] = theta_target
        detail_sink["fy_true"] = (
            detail_sink["omega_star"] + detail_sink["omega_target"] - theta_target
        )
        detail_sink["fy_reported"] = float(loss_value)
    return loss_value, gradient
