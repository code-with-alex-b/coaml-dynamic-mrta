"""Synthetic instance generator for Phase 0.5 viability check.

Implements the methodology section 3.7 distribution. One instance is the
tuple xi = (T, p_0) from section 3.1.2 where T is a list of tasks with
fields (id, release_epoch, pickup, drop, duration) and p_0 is the R by 2
matrix of initial robot positions.

All randomness flows from a single ``np.random.default_rng(seed)`` in the
fixed order documented in ``generate_instance``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import truncnorm


DEFAULT_CONFIG = {
    "warehouse_size": 50.0,
    "R": 10,
    "T": 60,
    "K_pick": 5,
    "K_drop": 2,
    "sigma_pick": 2.0,
    "sigma_drop": 3.0,
    "mu_d": 5.0,
    "sigma_d": 1.5,
    "d_min": 1.0,
    "d_max": 10.0,
    "H": 20,
    "Delta": 5.0,
    "v": 1.0,
}


@dataclass(eq=False)
class SyntheticInstance:
    tasks: list
    initial_positions: np.ndarray
    pickup_cluster_indices: np.ndarray
    drop_cluster_indices: np.ndarray
    config: dict
    seed: int
    R: int
    T: int
    H: int

    def to_dict(self) -> dict:
        """Serialise to the plain-dict form consumed by ``from_dict``.

        The inverse of ``from_dict``. Numpy arrays become nested lists so the
        result is picklable and JSON-safe, which is what the parallel rollout
        workers receive over the process boundary.
        """
        return {
            "tasks": [
                {
                    "id": int(t["id"]),
                    "release_epoch": int(t["release_epoch"]),
                    "pickup": np.asarray(t["pickup"], dtype=np.float64).tolist(),
                    "drop": np.asarray(t["drop"], dtype=np.float64).tolist(),
                    "duration": float(t["duration"]),
                }
                for t in self.tasks
            ],
            "initial_positions": np.asarray(
                self.initial_positions, dtype=np.float64
            ).tolist(),
            "pickup_cluster_indices": np.asarray(
                self.pickup_cluster_indices, dtype=np.int64
            ).tolist(),
            "drop_cluster_indices": np.asarray(
                self.drop_cluster_indices, dtype=np.int64
            ).tolist(),
            "config": dict(self.config),
            "seed": int(self.seed),
            "R": int(self.R),
            "T": int(self.T),
            "H": int(self.H),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SyntheticInstance":
        """Reconstruct an instance from the dataset generator's serialised
        form (the inverse of ``_serialize_instance`` in the expert dataset
        generator). Numeric fields are restored to numpy arrays."""
        tasks = [
            {
                "id": int(t["id"]),
                "release_epoch": int(t["release_epoch"]),
                "pickup": np.asarray(t["pickup"], dtype=np.float64),
                "drop": np.asarray(t["drop"], dtype=np.float64),
                "duration": float(t["duration"]),
            }
            for t in d["tasks"]
        ]
        return cls(
            tasks=tasks,
            initial_positions=np.asarray(
                d["initial_positions"], dtype=np.float64
            ),
            pickup_cluster_indices=np.asarray(
                d["pickup_cluster_indices"], dtype=np.int64
            ),
            drop_cluster_indices=np.asarray(
                d["drop_cluster_indices"], dtype=np.int64
            ),
            config=dict(d["config"]),
            seed=int(d["seed"]),
            R=int(d["R"]),
            T=int(d["T"]),
            H=int(d["H"]),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SyntheticInstance):
            return NotImplemented
        if (self.seed, self.R, self.T, self.H) != (
            other.seed,
            other.R,
            other.T,
            other.H,
        ):
            return False
        if self.config != other.config:
            return False
        if not np.array_equal(self.initial_positions, other.initial_positions):
            return False
        if not np.array_equal(
            self.pickup_cluster_indices, other.pickup_cluster_indices
        ):
            return False
        if not np.array_equal(self.drop_cluster_indices, other.drop_cluster_indices):
            return False
        if len(self.tasks) != len(other.tasks):
            return False
        for ta, tb in zip(self.tasks, other.tasks):
            if ta["id"] != tb["id"]:
                return False
            if ta["release_epoch"] != tb["release_epoch"]:
                return False
            if ta["duration"] != tb["duration"]:
                return False
            if not np.array_equal(ta["pickup"], tb["pickup"]):
                return False
            if not np.array_equal(ta["drop"], tb["drop"]):
                return False
        return True


def generate_instance(seed: int, config: dict | None = None) -> SyntheticInstance:
    """Generate one instance from the synthetic distribution.

    Unset config keys fall back to DEFAULT_CONFIG. Randomness order
    (single ``np.random.default_rng(seed)``):
        1. R initial robot positions, uniform in [0, W]^2.
        2. K_pick pickup cluster centres, uniform in [0, W]^2.
        3. K_drop drop cluster centres, uniform in [0, W]^2.
        4. For each task id 0..T-1: pickup cluster index, pickup loc
           (2D normal at centre, isotropic sigma_pick, clipped),
           drop cluster index, drop loc (sigma_drop, clipped),
           duration (truncnorm on [d_min, d_max]), release u (Beta(2,2)),
           tau = clip(floor(u * H), 0, H-1).
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    W = float(cfg["warehouse_size"])
    R = int(cfg["R"])
    T = int(cfg["T"])
    K_pick = int(cfg["K_pick"])
    K_drop = int(cfg["K_drop"])
    sigma_pick = float(cfg["sigma_pick"])
    sigma_drop = float(cfg["sigma_drop"])
    mu_d = float(cfg["mu_d"])
    sigma_d = float(cfg["sigma_d"])
    d_min = float(cfg["d_min"])
    d_max = float(cfg["d_max"])
    H = int(cfg["H"])

    rng = np.random.default_rng(seed)

    initial_positions = rng.uniform(0.0, W, size=(R, 2))
    pickup_centers = rng.uniform(0.0, W, size=(K_pick, 2))
    drop_centers = rng.uniform(0.0, W, size=(K_drop, 2))

    a_trunc = (d_min - mu_d) / sigma_d
    b_trunc = (d_max - mu_d) / sigma_d

    tasks: list = []
    pickup_idx_arr = np.empty(T, dtype=np.int64)
    drop_idx_arr = np.empty(T, dtype=np.int64)

    for j in range(T):
        p_idx = int(rng.integers(0, K_pick))
        pickup = rng.normal(loc=pickup_centers[p_idx], scale=sigma_pick, size=2)
        pickup = np.clip(pickup, 0.0, W)

        d_idx = int(rng.integers(0, K_drop))
        drop = rng.normal(loc=drop_centers[d_idx], scale=sigma_drop, size=2)
        drop = np.clip(drop, 0.0, W)

        duration = float(
            truncnorm.rvs(
                a_trunc, b_trunc, loc=mu_d, scale=sigma_d, random_state=rng
            )
        )

        u = float(rng.beta(2.0, 2.0))
        tau = int(np.clip(np.floor(u * H), 0, H - 1))

        tasks.append(
            {
                "id": j,
                "release_epoch": tau,
                "pickup": pickup,
                "drop": drop,
                "duration": duration,
            }
        )
        pickup_idx_arr[j] = p_idx
        drop_idx_arr[j] = d_idx

    return SyntheticInstance(
        tasks=tasks,
        initial_positions=initial_positions,
        pickup_cluster_indices=pickup_idx_arr,
        drop_cluster_indices=drop_idx_arr,
        config=cfg,
        seed=int(seed),
        R=R,
        T=T,
        H=H,
    )


def check_seed_non_overlap(seeds_a: Iterable[int], seeds_b: Iterable[int]) -> None:
    """Raise ValueError if the two seed iterables share any element."""
    set_a = set(seeds_a)
    set_b = set(seeds_b)
    overlap = sorted(set_a & set_b)
    if overlap:
        raise ValueError(f"Overlapping seeds between the two iterables: {overlap}")
