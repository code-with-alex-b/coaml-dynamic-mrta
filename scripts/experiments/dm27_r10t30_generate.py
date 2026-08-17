"""DM-27 Phase 1: generate the density-matched R=10, T=30, W=65 instance set.

Density-matched transfer experiment (see the planning-chat discussion this
follows from DM-25/DM-26 at R=10, T=60, where density and the robots-to-tasks
ratio both changed from the R=6, T=18 training distribution). Warehouse
65x65 is chosen so R=10, T=30 holds task/robot density to within ~1.4% of
the R=6, T=18, W=50 regime (exact match would be W=64.55) while preserving
the 1:3 robots-to-tasks ratio and 3.0 tasks/robot load exactly (10/30 reduces
to the same fraction as 6/18).

Every generator parameter except R, T, and warehouse_size is left byte-
identical to the R=6, T=18 training distribution's config (H=20, Delta=5.0,
v=1.0, K_pick=5, K_drop=2, sigma_pick=2.0, sigma_drop=3.0, mu_d=5.0,
sigma_d=1.5, d_min=1.0, d_max=10.0) -- confirmed against a live record from
cache/training_set_il_v4/train before writing this script. This means
cluster density, cluster spread relative to warehouse size, the duration-vs-
travel-time balance, and the per-epoch task arrival rate are NOT
regime-matched even though spatial density and the R/T ratio are; see the
planning-chat flag for the exact figures. This script does not attempt to
fix those; it generates exactly what was asked (counts and warehouse size
changed, nothing else).

Records are minimal: {R, T, seed, split, instance, weights}. No MILP
solution, no expert decisions or trajectory -- neither Phase 2 (cost-only
baselines) nor Phase 3 (RLOO fine-tuning, which needs no labels) reads them,
and matching cache/dm23_transfer_train's precedent (a MILP-free cache used
successfully for the same purpose at R=10, T=60).

Seeds 92000-92199, a fresh range: verified against the known existing cache
seed ranges (il_v3/v4 train 10000-10999, val 11000-11199, TEST 11200-11399,
rh_r10t60 90000-90199) before writing any file. Split 160 train
(92000-92159, "split": "train") and 40 held-out (92160-92199,
"split": "held_out"); the held-out seeds are never written into the derived
training-pool cache Phase 3 reads from.

Idempotent and resumable: an existing seed{N}.json is left untouched and
skipped, so a partial or interrupted run can simply be re-invoked.

Run:
    PYTHONPATH="$PWD/src" python scripts/experiments/dm27_r10t30_generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from instances.synthetic_generator import check_seed_non_overlap, generate_instance


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "cache" / "dm27_r10t30_d65"

TRAIN_SEEDS = list(range(92000, 92160))
HELD_OUT_SEEDS = list(range(92160, 92200))
ALL_SEEDS = TRAIN_SEEDS + HELD_OUT_SEEDS

WEIGHTS = {"w_dist": 0.0637, "w_make": 0.2398, "w_bal": 0.6965}

# Only R, T, and warehouse_size change; generate_instance() merges this over DEFAULT_CONFIG, which already equals the R=6, T=18 values for every other key.
CONFIG_OVERRIDE = {"warehouse_size": 65.0, "R": 10, "T": 30}

# Seed ranges this new range must not collide with; the test split must never be touched.
EXISTING_RANGES = {
    "il_v3/v4 train": range(10000, 11000),
    "il_v3 val": range(11000, 11200),
    "il_v3 TEST (never touch)": range(11200, 11400),
    "rh_r10t60": range(90000, 90200),
}


def verify_no_seed_overlap() -> None:
    for label, rng in EXISTING_RANGES.items():
        check_seed_non_overlap(ALL_SEEDS, rng)
    print(
        f"Verified: seeds {ALL_SEEDS[0]}-{ALL_SEEDS[-1]} do not overlap any "
        f"existing cache range ({', '.join(EXISTING_RANGES.keys())}).",
        flush=True,
    )


def write_record(seed: int, split: str) -> None:
    target = OUT_DIR / f"seed{seed}.json"
    if target.exists():
        return
    instance = generate_instance(seed, CONFIG_OVERRIDE)
    record = {
        "R": instance.R,
        "T": instance.T,
        "seed": seed,
        "split": split,
        "instance": instance.to_dict(),
        "weights": dict(WEIGHTS),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(record, f)
    tmp.rename(target)


def main() -> None:
    verify_no_seed_overlap()
    print(
        f"Generating {len(ALL_SEEDS)} instances "
        f"(R={CONFIG_OVERRIDE['R']}, T={CONFIG_OVERRIDE['T']}, "
        f"W={CONFIG_OVERRIDE['warehouse_size']}) into "
        f"{OUT_DIR.relative_to(REPO_ROOT)}: "
        f"{len(TRAIN_SEEDS)} train (seeds {TRAIN_SEEDS[0]}-{TRAIN_SEEDS[-1]}), "
        f"{len(HELD_OUT_SEEDS)} held-out (seeds {HELD_OUT_SEEDS[0]}-"
        f"{HELD_OUT_SEEDS[-1]}).",
        flush=True,
    )

    n_written, n_skipped = 0, 0
    for seed in TRAIN_SEEDS:
        existed = (OUT_DIR / f"seed{seed}.json").exists()
        write_record(seed, "train")
        n_skipped += int(existed)
        n_written += int(not existed)
    for seed in HELD_OUT_SEEDS:
        existed = (OUT_DIR / f"seed{seed}.json").exists()
        write_record(seed, "held_out")
        n_skipped += int(existed)
        n_written += int(not existed)

    present = sorted(
        int(p.name[len("seed"):-len(".json")])
        for p in OUT_DIR.glob("seed*.json")
    )
    expected = sorted(ALL_SEEDS)
    if present != expected:
        raise RuntimeError(
            f"{OUT_DIR} contents do not match the intended 200-instance set "
            f"after generation. Expected {len(expected)} seeds, found "
            f"{len(present)}."
        )
    print(
        f"Done: {n_written} written, {n_skipped} already present "
        f"(skipped). {len(present)}/{len(expected)} seeds confirmed on disk "
        f"in {OUT_DIR.relative_to(REPO_ROOT)}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
