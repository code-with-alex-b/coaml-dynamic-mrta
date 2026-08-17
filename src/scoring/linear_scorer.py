"""Linear scoring head, a convexity diagnostic for the method-one CO-layer.

Schiffer et al. (2026) recommend validating a perturbed-optimiser pipeline with a
convex model before scaling to a deep encoder. The Fenchel-Young loss is convex in
``Theta`` (Berthet et al. 2020), and this scorer makes ``Theta`` an affine function of
its own parameters, so the composed objective is convex in the weights. Trained on a
fixed cache rather than on-policy, a convex problem has a guaranteed-converging loss.
If it fails to converge the fault lies in the CO-layer, the cached labels or the loss
implementation, not in optimisation dynamics or model capacity.

Convexity requires the raw scores to be affine in the trainable weights, not merely
differentiable. No hidden layers, no activations, no learned interaction terms, and the
robot and task features must be fixed non-learned functions of the state, which holds
since ``build_features`` has no trainable parameters.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from instances.synthetic_generator import SyntheticInstance
from scoring.gnn_scorer import (
    F_ROBOT,
    F_TASK,
    _kaiming_init_all_linears,
    build_augmented_matrix,
    build_features,
)
from simulator.dynamic_simulator import SimulatorState


class LinearScorer(nn.Module):
    """Affine scorer. Returns the doubly augmented matrix Theta.

    ``Theta`` is an affine function of this module's parameters: the raw
    ``(R, T)`` scores come from one ``nn.Linear(F_ROBOT + F_TASK, 1)`` applied
    per (robot, task) pair with no nonlinearity, and
    ``scoring.gnn_scorer.build_augmented_matrix`` only ever masks cells to a
    fixed constant or adds a fixed constant (``commit_bias``), both of which
    preserve affineness in the linear layer's weights for a fixed state.
    """

    def __init__(
        self,
        commit_bias: float = 0.0,
        use_queue_ahead_mask: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(F_ROBOT + F_TASK, 1)
        _kaiming_init_all_linears(self)
        self.commit_bias = float(commit_bias)
        self.use_queue_ahead_mask = bool(use_queue_ahead_mask)

    def get_init_kwargs(self) -> dict:
        """Constructor arguments needed to rebuild an architecturally
        identical scorer in a worker process before ``load_state_dict``.

        Mirrors ``GNNScorer.get_init_kwargs``.
        """
        return {
            "commit_bias": float(self.commit_bias),
            "use_queue_ahead_mask": bool(self.use_queue_ahead_mask),
        }

    def build_features(
        self, state: SimulatorState, instance: SyntheticInstance
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_features(state, instance)

    def build_augmented_matrix(
        self,
        scores: torch.Tensor,
        state: SimulatorState,
        instance: SyntheticInstance,
    ) -> torch.Tensor:
        return build_augmented_matrix(
            scores,
            state,
            instance,
            commit_bias=self.commit_bias,
            use_queue_ahead_mask=self.use_queue_ahead_mask,
        )

    def forward(
        self, state: SimulatorState, instance: SyntheticInstance
    ) -> torch.Tensor:
        robot_feats, task_feats = self.build_features(state, instance)
        param_device = next(self.parameters()).device
        robot_feats = robot_feats.to(param_device)
        task_feats = task_feats.to(param_device)

        R = int(robot_feats.shape[0])
        T = int(task_feats.shape[0])
        r_exp = robot_feats.unsqueeze(1).expand(R, T, -1)
        t_exp = task_feats.unsqueeze(0).expand(R, T, -1)
        concat = torch.cat([r_exp, t_exp], dim=-1)
        scores = self.linear(concat).squeeze(-1)  # affine in weights

        theta = self.build_augmented_matrix(scores, state, instance)
        return theta
