"""Derive the objective weights from the reference magnitudes, per Equation 3.1.

Phase 1.5 step 3. The audit in CLEANUP_INVENTORY.md found that no file on disk
produces the objective weights: the triple 0.0637, 0.2398, 0.6965 is a hard
literal in twenty-two scripts and in every cached record, and the reference
magnitudes 869.4, 225.4 and 77.5 appear in exactly one prose file and in no
code. Section H2 of that audit concluded, from the reciprocal-rule residuals
alone, that the reported travel magnitude is the number that does not belong.

This script settles it by measurement rather than inference. It runs the
distance-only Hungarian assignment over the training split, takes the mean of
each raw objective term, and applies the reciprocal rule of Equation 3.1,

    w_i = (1 / X_i) / sum_j (1 / X_j)

The same magnitudes are also reported on the test split, so Table 3.1 can name
the split its numbers came from.

The rollout is imported from ``src/baselines/bipartite_policies.py`` and the
terms are read off the ``CostBreakdown`` the simulator already builds — nothing
is reimplemented. No solver is invoked, nothing is trained, and no cache record
or existing file is modified.

The distance-only Hungarian assignment does not read the objective weights:
``build_greedy_cost_matrix`` uses travel time only, so the rollout is fixed
regardless of the weight triple and there is no circularity in deriving
weights from its terms.

Usage::

    PYTHONPATH="$PWD/src" python scripts/regenerate/calibrate_weights.py \\
        --out-dir provenance/methodology
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "sweep"))

from baselines.bipartite_policies import run_greedy_policy  # noqa: E402
from evaluation.method_one_evaluator import rollout_failed  # noqa: E402
from evaluation.method_two_evaluator import load_records  # noqa: E402
from instances.synthetic_generator import SyntheticInstance  # noqa: E402
from evaluate_sweep import EVAL_WEIGHTS  # noqa: E402

TERMS = ("travel", "makespan", "balance")
HARDCODED = (0.0637, 0.2398, 0.6965)
REPORTED_MAGNITUDES = (869.4, 225.4, 77.5)  # as printed in Table 3.1

SPLITS = {
    "train": ("cache/training_set_il_v3/train", 10000, 10999),
    "test": ("cache/training_set_il_v3/test", 11200, 11399),
    "val": ("cache/training_set_il_v3/val", 11000, 11199),
}


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def reciprocal_rule(magnitudes) -> tuple:
    """Equation 3.1. Weights proportional to the reciprocal of each magnitude."""
    inv = np.array([1.0 / m for m in magnitudes], dtype=float)
    return tuple(float(x) for x in inv / inv.sum())


def contributions(weights, magnitudes) -> tuple:
    """Each term's share of the total weighted cost, in per cent."""
    prod = np.array([w * m for w, m in zip(weights, magnitudes)], dtype=float)
    return tuple(float(x) for x in prod / prod.sum() * 100.0)


def implied_magnitudes(weights, anchor_index=1, anchor_value=225.4) -> tuple:
    """Invert the rule: magnitudes proportional to 1/w, scaled to one anchor.

    The rule fixes magnitudes only up to a common scale, so one must be
    pinned; makespan is used since it's the one the reported triple and the
    reported weights already agree on.
    """
    inv = np.array([1.0 / w for w in weights], dtype=float)
    return tuple(float(x) for x in inv / inv[anchor_index] * anchor_value)


def _repo_relative(path: Path) -> str:
    """Path relative to the repository root, or unchanged if it lies outside.

    The output JSON is tracked and the repository is published, so an absolute
    path would carry the building machine's home directory into it. Matches
    ``figures.style._repo_relative``.
    """
    absolute = Path(path).resolve()
    try:
        return str(absolute.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def measure_split(cache_dir: Path, n: int) -> dict:
    """Mean raw objective terms under the distance-only Hungarian rollout."""
    records = load_records(cache_dir, n)
    if not records:
        raise SystemExit(f"No seed*.json records under {cache_dir}.")
    travel, makespan, balance, served, seeds = [], [], [], 0, []
    for rec in records:
        instance = SyntheticInstance.from_dict(rec["instance"])
        sim = run_greedy_policy(instance)
        # The weight triple here only scales the combined figure, unused below;
        # the three raw terms are weight-independent.
        cb = sim.compute_cost(EVAL_WEIGHTS)
        travel.append(cb.distance)
        makespan.append(cb.makespan)
        balance.append(cb.imbalance)
        served += 0 if rollout_failed(sim) else 1
        seeds.append(int(rec.get("seed", -1)))
    return {
        "cache_dir": _repo_relative(cache_dir),
        "n": len(records),
        "seed_min": min(seeds),
        "seed_max": max(seeds),
        "serve_all": f"{served}/{len(records)}",
        "mean": {
            "travel": float(np.mean(travel)),
            "makespan": float(np.mean(makespan)),
            "balance": float(np.mean(balance)),
        },
        "sd": {
            "travel": float(np.std(travel, ddof=1)),
            "makespan": float(np.std(makespan, ddof=1)),
            "balance": float(np.std(balance, ddof=1)),
        },
    }


# The committed artefact that carries this script's result. Named in the
# missing-input message so a fresh clone is told where the numbers live.
SHIPPED_ARTEFACT = "provenance/methodology/repro_weight_calibration.json"


def require_inputs(wanted: list) -> None:
    """Abort with one line if a gitignored input is absent."""
    for split in wanted:
        rel = SPLITS[split][0]
        if not (REPO_ROOT / rel).exists():
            raise SystemExit(
                f"Missing {rel}. It is gitignored and absent from a fresh "
                f"clone, so this script can only run on a tree that carries "
                f"it. The committed {SHIPPED_ARTEFACT} carries the result."
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="provenance/methodology")
    ap.add_argument("--instances", type=int, default=1000)
    ap.add_argument("--splits", default="train,test,val")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    require_inputs([s.strip() for s in args.splits.split(",")
                    if s.strip() and s.strip() in SPLITS])

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "repro_weight_calibration.json"
    if out_path.exists() and not args.force:
        raise SystemExit(f"REFUSING to overwrite {out_path}. Pass --force.")

    wanted = [s.strip() for s in args.splits.split(",") if s.strip()]
    results = {}
    for name in wanted:
        cache_rel, lo, hi = SPLITS[name]
        print(f"\n=== {name} split, {cache_rel}, expecting seeds {lo} to {hi}")
        m = measure_split(REPO_ROOT / cache_rel, args.instances)
        mags = tuple(m["mean"][t] for t in TERMS)
        w = reciprocal_rule(mags)
        contrib = contributions(w, mags)
        m["derived_weights"] = dict(zip(TERMS, w))
        m["contribution_pct"] = dict(zip(TERMS, contrib))
        m["abs_deviation_from_hardcoded"] = dict(
            zip(TERMS, [abs(a - b) for a, b in zip(w, HARDCODED)])
        )
        m["max_abs_deviation_from_hardcoded"] = max(
            abs(a - b) for a, b in zip(w, HARDCODED)
        )
        results[name] = m

        print(f"  n={m['n']}, seeds {m['seed_min']} to {m['seed_max']}, "
              f"serve-all {m['serve_all']}")
        print(f"  mean magnitudes : travel {mags[0]:.4f}  makespan {mags[1]:.4f}  "
              f"balance {mags[2]:.4f}")
        print(f"  reciprocal rule : {w[0]:.6f}  {w[1]:.6f}  {w[2]:.6f}")
        print(f"  hardcoded       : {HARDCODED[0]:.6f}  {HARDCODED[1]:.6f}  "
              f"{HARDCODED[2]:.6f}")
        print(f"  max deviation   : {m['max_abs_deviation_from_hardcoded']:.6f}")
        print(f"  contributions   : {contrib[0]:.2f} %  {contrib[1]:.2f} %  "
              f"{contrib[2]:.2f} %")

    # Magnitudes that would make the hardcoded triple the exact reciprocal
    # rule, anchored on the makespan magnitude both agree on.
    implied = implied_magnitudes(HARDCODED)
    w_from_reported = reciprocal_rule(REPORTED_MAGNITUDES)
    results["diagnosis"] = {
        "hardcoded_weights": dict(zip(TERMS, HARDCODED)),
        "reported_magnitudes_table_3_1": dict(zip(TERMS, REPORTED_MAGNITUDES)),
        "weights_implied_by_reported_magnitudes": dict(zip(TERMS, w_from_reported)),
        "max_dev_reported_magnitudes_vs_hardcoded": max(
            abs(a - b) for a, b in zip(w_from_reported, HARDCODED)
        ),
        "magnitudes_implied_by_hardcoded_weights": dict(zip(TERMS, implied)),
        "contributions_under_reported": dict(
            zip(TERMS, contributions(HARDCODED, REPORTED_MAGNITUDES))
        ),
    }
    print("\n=== diagnosis, arithmetic only")
    print(f"  weights implied by the reported magnitudes "
          f"{REPORTED_MAGNITUDES}: "
          f"{w_from_reported[0]:.6f}  {w_from_reported[1]:.6f}  "
          f"{w_from_reported[2]:.6f}")
    print(f"    max deviation from the hardcoded triple: "
          f"{results['diagnosis']['max_dev_reported_magnitudes_vs_hardcoded']:.6f}")
    print(f"  magnitudes implied by the hardcoded triple, makespan anchored at "
          f"225.4: {implied[0]:.2f}  {implied[1]:.2f}  {implied[2]:.2f}")
    c = results["diagnosis"]["contributions_under_reported"]
    print(f"  contributions under the reported numbers: "
          f"{c['travel']:.2f} %  {c['makespan']:.2f} %  {c['balance']:.2f} %")

    payload = {
        "script": "scripts/regenerate/calibrate_weights.py",
        "git_head": _git_head(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rule": "w_i = (1/X_i) / sum_j (1/X_j), Equation 3.1",
        "policy": "distance-only Hungarian, run_greedy_policy, which reads no weights",
        "splits": results,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")
    print("No solver was invoked and no existing file was modified.")


if __name__ == "__main__":
    main()
