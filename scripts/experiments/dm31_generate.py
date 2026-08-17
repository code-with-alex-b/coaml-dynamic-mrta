"""DM-31 Phase 1: generate three protocol-corrected, density-and-speed-matched
caches for the transfer curve (results/feature_scale_audit.md follow-up).

Prior transfer targets (DM-23/26/27/28) held task density and the
robots-to-tasks ratio but left robot speed v=1.0 fixed, so travel TIME grew
with the warehouse while the epoch length Delta stayed at 5.0. The feature
audit found this pushes busy/Delta and epoch_fraction out of their training
range by 7.6x and 2.5x at W=204. The correction here: scale robot speed
with the warehouse side (v = W / 50) so every time quantity -- travel time,
busy time, epochs to termination -- stays invariant across scale. Delta, H,
and mu_d are left untouched, matching the instruction not to change them.

Protocol, identical at every scale: W = 50 * sqrt(T / 18), v = W / 50,
wall_clock_cap = 5 * H * Delta = 500 (unchanged, since H and Delta are
unchanged). Only warehouse_size, R, T, and v are overridden from
DEFAULT_CONFIG; every other generator parameter (H=20, Delta=5.0, K_pick=5,
K_drop=2, sigma_pick=2.0, sigma_drop=3.0, mu_d=5.0, sigma_d=1.5, d_min=1.0,
d_max=10.0) is inherited unchanged.

Three scales, 60 instances each, seeds 96000-96259 (a fresh range, checked
against every known existing cache range before writing anything). Last 20
of each 60 are held out for Phase 3 evaluation; the first 40 are unused in
this experiment (no training happens here).

    Scale A: R=10,  T=30,  W=65,  v=1.29, seeds 96000-96059, cache/dm31a_r10t30
    Scale B: R=30,  T=90,  W=112, v=2.24, seeds 96100-96159, cache/dm31b_r30t90
    Scale C: R=100, T=300, W=204, v=4.08, seeds 96200-96259, cache/dm31c_r100t300

Records are minimal ({R, T, seed, split, instance, weights}, no MILP), same
convention as cache/dm28_r100t300_d204: no anticipative oracle is computed
or needed for this experiment.

Prints a projected total wall clock after the first 3 instances (across all
three scales) and aborts before writing further if that projection exceeds
30 minutes -- generation is pure numpy/scipy draws with no simulator or
solver calls, so this is expected to pass trivially, but the check runs for
real rather than being asserted.

Idempotent and resumable: an existing seed{N}.json is left untouched.

Run:
    PYTHONPATH="$PWD/src" python scripts/experiments/dm31_generate.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from instances.synthetic_generator import check_seed_non_overlap, generate_instance


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS = {"w_dist": 0.0637, "w_make": 0.2398, "w_bal": 0.6965}
RUNTIME_BUDGET_S = 30 * 60

SCALES = [
    {
        "label": "Scale A (R=10, T=30)",
        "out_dir": REPO_ROOT / "cache" / "dm31a_r10t30",
        "seeds": list(range(96000, 96060)),
        "config": {"warehouse_size": 65.0, "R": 10, "T": 30, "v": 1.29},
    },
    {
        "label": "Scale B (R=30, T=90)",
        "out_dir": REPO_ROOT / "cache" / "dm31b_r30t90",
        "seeds": list(range(96100, 96160)),
        "config": {"warehouse_size": 112.0, "R": 30, "T": 90, "v": 2.24},
    },
    {
        "label": "Scale C (R=100, T=300)",
        "out_dir": REPO_ROOT / "cache" / "dm31c_r100t300",
        "seeds": list(range(96200, 96260)),
        "config": {"warehouse_size": 204.0, "R": 100, "T": 300, "v": 4.08},
    },
]

# Existing cache seed ranges this new range must not collide with.
EXISTING_RANGES = {
    "il_v3/v4 train": range(10000, 11000),
    "il_v3 val": range(11000, 11200),
    "il_v3 TEST (never touch)": range(11200, 11400),
    "rh_r10t60 / dm23 / dm26": range(90000, 90200),
    "dm27_r10t30": range(92000, 92200),
    "dm28_r100t300": range(93000, 93060),
    "headroom_* scripts": range(99000, 99075),
}


def verify_no_seed_overlap() -> None:
    all_seeds = [s for scale in SCALES for s in scale["seeds"]]
    for label, rng in EXISTING_RANGES.items():
        check_seed_non_overlap(all_seeds, rng)
    print(
        f"Verified: seeds {all_seeds[0]}-{all_seeds[-1]} (3 disjoint blocks) "
        f"do not overlap any existing cache range "
        f"({', '.join(EXISTING_RANGES.keys())}).",
        flush=True,
    )


def write_record(out_dir: Path, seed: int, split: str, config: dict) -> bool:
    """Returns True if a new file was written, False if it already existed."""
    target = out_dir / f"seed{seed}.json"
    if target.exists():
        return False
    instance = generate_instance(seed, config)
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
    return True


def density_report(scale: dict) -> None:
    W = float(scale["config"]["warehouse_size"])
    R = int(scale["config"]["R"])
    T = int(scale["config"]["T"])
    v = float(scale["config"]["v"])
    area = W * W
    task_density = T / area
    robot_density = R / area
    ratio = R / T

    W0, R0, T0 = 50.0, 6, 18
    area0 = W0 * W0
    task_density0 = T0 / area0
    robot_density0 = R0 / area0
    ratio0 = R0 / T0

    task_pct = task_density / task_density0 * 100
    robot_pct = robot_density / robot_density0 * 100
    ratio_pct = ratio / ratio0 * 100

    print(
        f"  {scale['label']}: W={W:.1f} v={v:.2f} "
        f"task_density={task_density:.6f} ({task_pct:.2f}% of training) "
        f"robot_density={robot_density:.6f} ({robot_pct:.2f}% of training) "
        f"R:T={ratio:.4f} ({ratio_pct:.2f}% of training)",
        flush=True,
    )


def main() -> None:
    verify_no_seed_overlap()

    # Probe instances keep their real "unused" label -- not thrown away, just timed -- to project total runtime and abort early if it exceeds RUNTIME_BUDGET_S.
    probe_seeds = SCALES[0]["seeds"][:3]
    t0 = time.perf_counter()
    for seed in probe_seeds:
        write_record(SCALES[0]["out_dir"], seed, "unused", SCALES[0]["config"])
    probe_elapsed = time.perf_counter() - t0
    sec_per_instance = probe_elapsed / len(probe_seeds)
    total_instances = sum(len(s["seeds"]) for s in SCALES)
    projected_s = sec_per_instance * total_instances
    print(
        f"Phase 1 timing gate: {len(probe_seeds)} instances in "
        f"{probe_elapsed:.2f}s ({sec_per_instance:.3f}s/instance); "
        f"projected total for {total_instances} instances: "
        f"{projected_s:.1f}s ({projected_s/60:.1f} min).",
        flush=True,
    )
    if projected_s > RUNTIME_BUDGET_S:
        print(
            f"STOPPING: projected Phase 1 runtime ({projected_s/60:.1f} min) "
            f"exceeds the 30-minute budget. No further instances written "
            f"beyond the 3 already on disk.",
            flush=True,
        )
        return

    for scale in SCALES:
        seeds = scale["seeds"]
        unused_seeds = seeds[:40]
        held_out_seeds = seeds[40:60]
        n_written, n_skipped = 0, 0
        for seed in unused_seeds:
            written = write_record(scale["out_dir"], seed, "unused", scale["config"])
            n_written += int(written)
            n_skipped += int(not written)
        for seed in held_out_seeds:
            written = write_record(scale["out_dir"], seed, "held_out", scale["config"])
            n_written += int(written)
            n_skipped += int(not written)

        present = sorted(
            int(p.name[len("seed"):-len(".json")])
            for p in scale["out_dir"].glob("seed*.json")
        )
        expected = sorted(seeds)
        if present != expected:
            raise RuntimeError(
                f"{scale['out_dir']} contents do not match the intended "
                f"60-instance set. Expected {len(expected)} seeds, found "
                f"{len(present)}."
            )
        print(
            f"{scale['label']}: {n_written} written, {n_skipped} already "
            f"present, {len(present)}/{len(expected)} confirmed in "
            f"{scale['out_dir'].relative_to(REPO_ROOT)} "
            f"(40 unused {seeds[0]}-{seeds[39]}, 20 held-out "
            f"{seeds[40]}-{seeds[59]}).",
            flush=True,
        )

    print("\nDensity / ratio vs. R=6, T=18, W=50 training regime:", flush=True)
    for scale in SCALES:
        density_report(scale)


if __name__ == "__main__":
    main()
