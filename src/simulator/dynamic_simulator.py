"""Dynamic simulator, the Path B continuous-time chain.

Internal state. The primary per-robot state is the continuous wall-clock finish time
``_robot_finish_times[r]``, not a Delta-decayed remaining busy time. The dispatcher's
epochs set the rhythm of decisions, not of physical execution. A commitment's
``start_time`` is the robot's previous physical finish, or zero for its first, and
``finish_wall_clock`` is ``start_time + c_j``. Nothing rounds up to an epoch boundary.

Asymmetric by design. The simulator accepts queue-ahead commits and does not reject an
action targeting a robot whose physical finish is in the future, which is what the
offline anticipative MILP relies on when its continuous chain schedule is replayed at
epoch boundaries with start times back-dated to keep the chain tight. Online policies
consult ``state.available_robots``, the physically idle set, so they do not over-commit.
Action validation enforces only structural rules, being no duplicate robot, no duplicate
task, task in pending and robot index in range. Physical availability is a policy
concern surfaced through the state view.

State view::

    busy_times[r]    = max(0, finish_times[r] - wall_clock)
    available_robots = {r : finish_times[r] <= wall_clock + 1e-9}

Release times. A robot arriving at a pickup before the task's release time waits there,
so ``finish_wall_clock`` becomes ``max(arrival_at_pickup, release_time) +
travel(pickup, drop) + duration``, matching the MILP's
``f[r, j] >= release_j + travel(pickup, drop) + d_j``. The service contribution
``completion_time = travel_distance + duration`` is unchanged, since the release wait is
idle time at the pickup rather than active work.

Cost::

    D           = sum over commitments of travel_distance
    M           = max over r of finish_times[r]
    tau_busy[r] = sum over commitments by r of (travel_distance + duration)
    B           = max(tau_busy) - min(tau_busy)
    Combined    = w_dist * D + w_make * M + w_bal * B
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

from instances.synthetic_generator import SyntheticInstance


_AVAIL_TOL = 1e-9


@dataclass
class SimulatorState:
    positions: np.ndarray
    busy_times: np.ndarray
    pending_tasks: set
    available_robots: set
    epoch: int
    wall_clock: float

    @classmethod
    def from_dict(cls, d: dict) -> "SimulatorState":
        """Reconstruct a state view from the dataset generator's serialised
        trajectory record (the inverse of ``_serialize_state_record``).

        That record stores the raw continuous ``finish_times`` rather than the
        ``busy_times`` view, so ``busy_times`` is recomputed here as
        ``max(0, finish_times - wall_clock)``, matching the simulator's own
        ``_busy_times_view``. A record that already carries ``busy_times`` is
        also accepted for symmetry."""
        positions = np.asarray(d["positions"], dtype=np.float64)
        wall_clock = float(d["wall_clock"])
        if "busy_times" in d:
            busy_times = np.asarray(d["busy_times"], dtype=np.float64)
        else:
            finish_times = np.asarray(d["finish_times"], dtype=np.float64)
            busy_times = np.maximum(0.0, finish_times - wall_clock)
        return cls(
            positions=positions,
            busy_times=busy_times,
            pending_tasks={int(j) for j in d["pending_tasks"]},
            available_robots={int(r) for r in d["available_robots"]},
            epoch=int(d["epoch"]),
            wall_clock=wall_clock,
        )


@dataclass
class TrajectoryRecord:
    commitments: list = field(default_factory=list)
    robot_finish_times: Optional[np.ndarray] = None
    robot_busy_times_total: Optional[np.ndarray] = None
    terminal_state: Optional[SimulatorState] = None
    simulator_failure: bool = False
    unserved_task_ids: set = field(default_factory=set)
    epochs_run: int = 0
    wall_clock_at_termination: float = 0.0


@dataclass
class CostBreakdown:
    distance: float
    makespan: float
    imbalance: float
    combined: float
    weights_used: dict


class DynamicSimulator:
    def __init__(self, instance: SyntheticInstance):
        self.instance = instance
        cfg = instance.config
        self.R = int(instance.R)
        self.T = int(instance.T)
        self.H = int(instance.H)
        self.Delta = float(cfg["Delta"])
        self.v = float(cfg["v"])
        self.wall_clock_cap = 5.0 * self.H * self.Delta

        self._tasks_by_id = {int(t["id"]): t for t in instance.tasks}
        self._release_buckets: dict = {}
        for t in instance.tasks:
            tau = int(t["release_epoch"])
            if 0 <= tau < self.H:
                self._release_buckets.setdefault(tau, set()).add(int(t["id"]))
        self._serviceable: Set[int] = {
            int(t["id"])
            for t in instance.tasks
            if 0 <= int(t["release_epoch"]) < self.H
        }
        self._n_serviceable = len(self._serviceable)

        self._reset_internal()

    def _reset_internal(self) -> None:
        self._positions = np.asarray(
            self.instance.initial_positions, dtype=np.float64
        ).copy()
        self._robot_finish_times = np.zeros(self.R, dtype=np.float64)
        self._pending: Set[int] = set(self._release_buckets.get(0, set()))
        self._epoch = 0
        self._wall_clock = 0.0
        self._trajectory = TrajectoryRecord()
        self._terminal = False
        self._check_and_handle_terminal()

    def reset(self) -> SimulatorState:
        self._reset_internal()
        return self.state

    def _available_robots_set(self) -> Set[int]:
        return {
            int(r)
            for r in range(self.R)
            if self._robot_finish_times[r] <= self._wall_clock + _AVAIL_TOL
        }

    def _busy_times_view(self) -> np.ndarray:
        return np.maximum(0.0, self._robot_finish_times - self._wall_clock)

    @property
    def state(self) -> SimulatorState:
        return SimulatorState(
            positions=self._positions.copy(),
            busy_times=self._busy_times_view(),
            pending_tasks=set(self._pending),
            available_robots=self._available_robots_set(),
            epoch=int(self._epoch),
            wall_clock=float(self._wall_clock),
        )

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    @property
    def trajectory(self) -> TrajectoryRecord:
        return self._trajectory

    def _validate_action(self, action: List[Tuple[int, int]]) -> None:
        """Structural validation only. Physical robot availability is NOT
        enforced here; the simulator accepts queue-ahead commits per the
        (b)+(i) design. Online policies that should not over-commit are
        expected to consult ``state.available_robots`` themselves."""
        seen_robots: Set[int] = set()
        seen_tasks: Set[int] = set()
        for pair in action:
            r, j = int(pair[0]), int(pair[1])
            if r in seen_robots:
                raise ValueError(f"Robot {r} appears more than once in action")
            if j in seen_tasks:
                raise ValueError(f"Task {j} appears more than once in action")
            if not (0 <= r < self.R):
                raise ValueError(f"Robot {r} out of range [0, {self.R})")
            if j not in self._pending:
                raise ValueError(
                    f"Task {j} not in pending tasks {sorted(self._pending)}"
                )
            seen_robots.add(r)
            seen_tasks.add(j)

    def _check_and_handle_terminal(self) -> None:
        if self._terminal:
            return
        success = (
            len(self._trajectory.commitments) == self._n_serviceable
            and bool(
                np.all(
                    self._robot_finish_times <= self._wall_clock + _AVAIL_TOL
                )
            )
        )
        failure = (self._wall_clock > self.wall_clock_cap) and not success
        if success or failure:
            self._terminal = True
            self._trajectory.simulator_failure = failure
            served_ids: Set[int] = set()
            for c in self._trajectory.commitments:
                if float(c["finish_wall_clock"]) <= self._wall_clock + _AVAIL_TOL:
                    served_ids.add(int(c["task_id"]))
            self._trajectory.unserved_task_ids = self._serviceable - served_ids
            self._trajectory.epochs_run = int(self._epoch)
            self._trajectory.wall_clock_at_termination = float(self._wall_clock)
            self._trajectory.terminal_state = self.state
            self._finalize_per_robot_aggregates()

    def _finalize_per_robot_aggregates(self) -> None:
        self._trajectory.robot_finish_times = self._robot_finish_times.copy()
        bt = np.zeros(self.R, dtype=np.float64)
        for c in self._trajectory.commitments:
            r = int(c["robot_id"])
            bt[r] += float(c["completion_time"])
        self._trajectory.robot_busy_times_total = bt

    def step(
        self, action: List[Tuple[int, int]]
    ) -> Tuple[SimulatorState, bool]:
        if self._terminal:
            raise RuntimeError("step() called after termination")
        self._validate_action(action)

        for pair in action:
            r, j = int(pair[0]), int(pair[1])
            task = self._tasks_by_id[j]
            start_pos = self._positions[r].copy()
            pickup = np.asarray(task["pickup"], dtype=np.float64).reshape(2)
            drop = np.asarray(task["drop"], dtype=np.float64).reshape(2)
            duration = float(task["duration"])
            tp = float(np.linalg.norm(start_pos - pickup) / self.v)
            td = float(np.linalg.norm(pickup - drop) / self.v)
            travel_distance = tp + td
            c_j = travel_distance + duration

            start_time = max(
                float(self._robot_finish_times[r]),
                float(self._epoch) * self.Delta,
            )
            arrival_at_pickup = start_time + tp
            release_time_j = float(int(task["release_epoch"])) * self.Delta
            effective_pickup_start = max(arrival_at_pickup, release_time_j)
            finish_time = effective_pickup_start + td + duration

            self._robot_finish_times[r] = finish_time
            self._positions[r] = drop
            self._pending.discard(j)

            self._trajectory.commitments.append(
                {
                    "epoch": int(self._epoch),
                    "robot_id": r,
                    "task_id": j,
                    "start_position": start_pos,
                    "pickup": pickup.copy(),
                    "drop": drop.copy(),
                    "duration": duration,
                    "completion_time": c_j,
                    "travel_distance": travel_distance,
                    "start_wall_clock": start_time,
                    "arrival_at_pickup": arrival_at_pickup,
                    "effective_pickup_start": effective_pickup_start,
                    "finish_wall_clock": finish_time,
                }
            )

        self._epoch += 1
        self._wall_clock = float(self._epoch) * self.Delta

        if self._epoch < self.H:
            new = self._release_buckets.get(self._epoch, set())
            self._pending |= new

        self._check_and_handle_terminal()

        return self.state, self._terminal

    def compute_cost(self, weights: dict) -> CostBreakdown:
        if not self._terminal:
            raise RuntimeError("compute_cost() called before terminal")
        D = float(
            sum(c["travel_distance"] for c in self._trajectory.commitments)
        )
        ft = self._trajectory.robot_finish_times
        bt = self._trajectory.robot_busy_times_total
        M = float(ft.max()) if ft.size > 0 else 0.0
        B = float(bt.max() - bt.min()) if bt.size > 0 else 0.0
        w_d = float(weights["w_dist"])
        w_m = float(weights["w_make"])
        w_b = float(weights["w_bal"])
        C = w_d * D + w_m * M + w_b * B
        return CostBreakdown(
            distance=D,
            makespan=M,
            imbalance=B,
            combined=float(C),
            weights_used={"w_dist": w_d, "w_make": w_m, "w_bal": w_b},
        )
