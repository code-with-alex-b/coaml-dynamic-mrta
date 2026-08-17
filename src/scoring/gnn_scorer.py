"""GNN scoring head.

Consumes a ``SimulatorState`` and its source ``SyntheticInstance`` and produces the
doubly augmented score matrix ``Theta``. Shared by method one and method two.

Bipartite GNN core, a SAGEConv stack at hidden dim 64 over two layers with ReLU
between and Kaiming init on all linears. R robot nodes and T task nodes, with edges
connecting every robot to every task in both directions. The pairwise scoring head is
a 2H to H to 1 MLP, again ReLU and Kaiming.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from instances.synthetic_generator import SyntheticInstance
from simulator.dynamic_simulator import SimulatorState


M_MASK = 100.0
HIDDEN_DIM = 64
# 3 global state features (backlog_ratio, fraction_committed, epoch_fraction) broadcast to every node so the scorer stays calibrated on high-backlog states.
N_GLOBAL = 3
F_ROBOT = 4 + N_GLOBAL
F_TASK = 7 + N_GLOBAL


def _kaiming_init_all_linears(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def build_features(
    state: SimulatorState, instance: SyntheticInstance
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure per-node feature construction, shared by every scorer.

    Depends only on ``state`` and ``instance``, never on any scorer's
    trainable parameters, so any scorer (GNN, linear, or otherwise) can call
    this directly and receive identical features. Extracted verbatim from
    ``GNNScorer.build_features``; that method is now a thin wrapper around
    this function, so existing behaviour and checkpoints are unaffected.
    """
    cfg = instance.config
    W = float(cfg["warehouse_size"])
    Delta = float(cfg["Delta"])
    mu_d = float(cfg["mu_d"])
    H = int(instance.H)
    R = int(instance.R)
    T = int(instance.T)

    n_pending = len(state.pending_tasks)
    n_available = len(state.available_robots)
    backlog_ratio = n_pending / max(1, n_available)
    fraction_committed = (T - n_pending) / T
    epoch_fraction = float(state.epoch) / float(H)
    globals_vec = [backlog_ratio, fraction_committed, epoch_fraction]

    robot_feats = torch.zeros(R, F_ROBOT, dtype=torch.float32)
    for r in range(R):
        pos = state.positions[r]
        busy = float(state.busy_times[r])
        avail = 1.0 if r in state.available_robots else 0.0
        robot_feats[r, 0] = float(pos[0]) / W
        robot_feats[r, 1] = float(pos[1]) / W
        robot_feats[r, 2] = busy / Delta
        robot_feats[r, 3] = avail
        robot_feats[r, 4] = backlog_ratio
        robot_feats[r, 5] = fraction_committed
        robot_feats[r, 6] = epoch_fraction

    task_feats = torch.zeros(T, F_TASK, dtype=torch.float32)
    tasks_by_id = {int(t["id"]): t for t in instance.tasks}
    pending = state.pending_tasks
    for j in range(T):
        if j in pending:
            t = tasks_by_id[j]
            pickup = t["pickup"]
            drop = t["drop"]
            duration = float(t["duration"])
            release_epoch = int(t["release_epoch"])
            task_feats[j, 0] = float(pickup[0]) / W
            task_feats[j, 1] = float(pickup[1]) / W
            task_feats[j, 2] = float(drop[0]) / W
            task_feats[j, 3] = float(drop[1]) / W
            task_feats[j, 4] = duration / mu_d
            task_feats[j, 5] = float(release_epoch) / float(H)
            task_feats[j, 6] = 1.0
        # Global features are broadcast to every task node, pending or not.
        task_feats[j, 7] = backlog_ratio
        task_feats[j, 8] = fraction_committed
        task_feats[j, 9] = epoch_fraction
    return robot_feats, task_feats


def build_augmented_matrix(
    scores: torch.Tensor,
    state: SimulatorState,
    instance: SyntheticInstance,
    commit_bias: float = 0.0,
    use_queue_ahead_mask: bool = False,
) -> torch.Tensor:
    """Mask, bias, and pad a raw ``(R, T)`` score matrix into ``Theta``.

    Affine in ``scores`` for a fixed state, since masking replaces cells with the
    constant ``-M_MASK`` and ``commit_bias`` adds a constant. That is what lets a scorer
    whose scores are affine in its own parameters, such as ``LinearScorer``, produce a
    ``Theta`` affine in those parameters.
    """
    R, T = int(scores.shape[0]), int(scores.shape[1])
    device = scores.device
    dtype = scores.dtype

    available = torch.zeros(R, dtype=torch.bool, device=device)
    for r in state.available_robots:
        available[int(r)] = True
    pending = torch.zeros(T, dtype=torch.bool, device=device)
    for j in state.pending_tasks:
        pending[int(j)] = True
    if use_queue_ahead_mask:
        # Path B: drop robot-availability so busy robots keep queue-ahead scores; valid iff task is pending.
        valid = pending.unsqueeze(0).expand(R, T)
    else:
        # Default: valid only if robot available and task pending; busy-robot rows masked to -M_MASK.
        valid = available.unsqueeze(1) & pending.unsqueeze(0)

    masked = scores.masked_fill(~valid, -M_MASK)

    # Runtime commit bias nudges valid assignment scores up so the oracle favors committing; default 0.0 keeps masked bit-identical.
    if commit_bias != 0.0:
        masked = masked + commit_bias * valid.to(dtype)

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
    theta = torch.cat([top, bottom], dim=0)
    return theta


class _BipartiteGNNCore(nn.Module):
    """SAGEConv stack over R + T nodes with bipartite robot-task edges."""

    def __init__(
        self,
        robot_input_dim: int = F_ROBOT,
        task_input_dim: int = F_TASK,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = 2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.robot_proj = nn.Linear(robot_input_dim, hidden_dim)
        self.task_proj = nn.Linear(task_input_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        _kaiming_init_all_linears(self)

    @staticmethod
    def _build_edge_index(R: int, T: int, device: torch.device) -> torch.Tensor:
        if R == 0 or T == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        r_idx = (
            torch.arange(R, device=device, dtype=torch.long)
            .unsqueeze(1)
            .expand(R, T)
            .flatten()
        )
        t_idx = (
            (torch.arange(T, device=device, dtype=torch.long) + R)
            .unsqueeze(0)
            .expand(R, T)
            .flatten()
        )
        src = torch.cat([r_idx, t_idx])
        dst = torch.cat([t_idx, r_idx])
        return torch.stack([src, dst], dim=0)

    def forward(
        self, robot_features: torch.Tensor, task_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        R = int(robot_features.shape[0])
        T = int(task_features.shape[0])
        h_r = self.robot_proj(robot_features)
        h_t = self.task_proj(task_features)
        x = torch.cat([h_r, h_t], dim=0)
        edge_index = self._build_edge_index(R, T, x.device)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x[:R], x[R:]


class PairwiseScorer(nn.Module):
    """Concatenation-based pairwise scorer. 2H -> H -> 1 MLP."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        _kaiming_init_all_linears(self)

    def forward(
        self, robot_emb: torch.Tensor, task_emb: torch.Tensor
    ) -> torch.Tensor:
        R = int(robot_emb.shape[0])
        T = int(task_emb.shape[0])
        r_exp = robot_emb.unsqueeze(1).expand(R, T, -1)
        t_exp = task_emb.unsqueeze(0).expand(R, T, -1)
        concat = torch.cat([r_exp, t_exp], dim=-1)
        h = F.relu(self.fc1(concat))
        out = self.fc2(h).squeeze(-1)
        return out


class GNNScorer(nn.Module):
    """End-to-end scorer. Returns the doubly augmented matrix Theta."""

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = 2,
        commit_bias: float = 0.0,
        use_queue_ahead_mask: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = _BipartiteGNNCore(
            robot_input_dim=F_ROBOT,
            task_input_dim=F_TASK,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.scorer = PairwiseScorer(hidden_dim)
        # Runtime-only bias; a plain attribute (not nn.Parameter/buffer) so checkpoints stay compatible and bit-identical at commit_bias=0.0.
        self.commit_bias = float(commit_bias)
        # Mask policy for Path B queue-ahead commits; plain attribute (not a parameter/buffer) so old checkpoints load unchanged.
        self.use_queue_ahead_mask = bool(use_queue_ahead_mask)

    def get_init_kwargs(self) -> dict:
        """Constructor arguments needed to rebuild an architecturally
        identical scorer in a worker process before ``load_state_dict``.

        Returns the ``hidden_dim``, ``num_layers``, ``commit_bias`` and
        ``use_queue_ahead_mask`` passed to ``__init__``, which together with a
        state dict fully reconstruct the model on CPU. ``commit_bias`` and
        ``use_queue_ahead_mask`` are runtime arguments (not in the state dict),
        so they are carried here to keep worker scorers consistent.
        """
        return {
            "hidden_dim": int(self.hidden_dim),
            "num_layers": int(self.encoder.num_layers),
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
        robot_emb, task_emb = self.encoder(robot_feats, task_feats)
        scores = self.scorer(robot_emb, task_emb)
        theta = self.build_augmented_matrix(scores, state, instance)
        return theta
