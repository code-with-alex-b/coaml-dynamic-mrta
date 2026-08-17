"""Log-space Sinkhorn operator (methodology section 3.3.4).

Differentiable doubly stochastic relaxation of the augmented score matrix
``Theta`` from the GNN scoring head. Used inside the gradient computation
for both method one and method two.

The operator solves
    max <Theta, P>  -  tau * sum_{r,j} P[r, j] log P[r, j]
on the Birkhoff polytope by repeated row and column normalisation of
``exp(Theta / tau)``. Iteration runs in log space so masked entries at
``-M_mask`` (default -100) do not underflow to zero.

Default hyperparameters per methodology:
    K_SINK_DEFAULT = 20    (number of row/column normalisation rounds)
    TAU_DEFAULT    = 1.0   (entropic temperature)
"""

from __future__ import annotations

import torch


K_SINK_DEFAULT = 20
TAU_DEFAULT = 1.0


def sinkhorn_log(
    Theta: torch.Tensor,
    tau: float = TAU_DEFAULT,
    num_iters: int = K_SINK_DEFAULT,
) -> torch.Tensor:
    """Log-space Sinkhorn iteration on a square cost matrix.

    Args:
        Theta: shape (N, N). May include large negative masked entries.
        tau: entropic temperature. Smaller tau approaches a hard assignment.
        num_iters: number of row/column normalisation passes.

    Returns:
        P: shape (N, N) doubly stochastic matrix (rows and columns sum to 1).
    """
    log_P = Theta / tau
    for _ in range(num_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=0, keepdim=True)
    return torch.exp(log_P)


def sinkhorn_log_batched(
    Theta_batch: torch.Tensor,
    tau: float = TAU_DEFAULT,
    num_iters: int = K_SINK_DEFAULT,
) -> torch.Tensor:
    """Batched Sinkhorn iteration.

    Args:
        Theta_batch: shape (B, N, N).
        tau: entropic temperature.
        num_iters: number of row/column normalisation passes.

    Returns:
        P_batch: shape (B, N, N). Each batch element is doubly stochastic.
    """
    log_P = Theta_batch / tau
    for _ in range(num_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=2, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
    return torch.exp(log_P)
