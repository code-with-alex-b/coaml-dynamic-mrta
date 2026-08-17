"""Regenerate the Figure 4.2 series from cached per-instance costs.

Phase 1.5 step 5. Recomputes the seven rolling-horizon rungs of the budget
curve from the per-instance CSVs the A3 ladder wrote, with no solver and no
rollout. Every quantity is arithmetic over files already on disk.

Three quantities per rung.

  Gap closure, RATIO OF MEANS:
      (mean floor cost - mean rolling cost) / (mean floor cost - mean MILP cost)
  Floor is the distance-only Hungarian on the same 200 test instances; ceiling
  is each rung's own ``milp_oracle_cost_from_cache`` column. The mean of
  per-instance ratios is computed alongside so the two estimators can be told
  apart, since the audit found the Section 4.3 prose quotes C102 and C103 on
  one and C104 on the other.

  Measured compute per decision: the rung's wall clock divided by the instance
  count divided by the mean number of window solves per instance, from
  ``provenance/figure42_budget/a3_ladder_walltimes.txt`` (written by the ladder
  driver itself). This is what the figure plots on x, not the nominal budget,
  because the two diverge by up to a factor of nineteen.

  Fallback fraction: the share of window solves that returned no incumbent and
  fell through to Hungarian-on-kappa, a different policy deciding those solves.

No solver is invoked, nothing is evaluated or trained, and no cache record or
existing file is read for writing.

Usage::

    PYTHONPATH="$PWD/src" python scripts/regenerate/figure42_series.py \\
        --out-dir provenance/figure42_budget
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _d in ("regenerate", "analysis", "experiments"):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _d))

from transfer_table_stats import load_csv  # noqa: E402

BUDGETS = ["0.010", "0.025", "0.050", "0.100", "0.250", "1.0", "10.0"]
NOMINAL_MS = {"0.010": 10, "0.025": 25, "0.050": 50, "0.100": 100,
              "0.250": 250, "1.0": 1000, "10.0": 10000}

# Published, from THESIS_CLAIMS.md Section E and the figure's own manifest.
PUBLISHED_GAP = {"0.010": ("C102", 18.02), "1.0": ("C104", 73.22),
                 "10.0": ("C103", 76.37)}
PUBLISHED_FALLBACK = {"0.010": ("C105", "over a third"),
                      "0.050": ("C106", "under 7 per cent")}
# Recorded in figures/output/budget_curve.sources.txt by fig_budget_curve.py.
MANIFEST_COMPUTE_MS = {"0.010": 12.167, "0.025": 20.946, "0.050": 33.542,
                       "0.100": 56.386, "0.250": 105.601, "1.0": 225.607,
                       "10.0": 537.342}
MANIFEST_GAP = {"0.010": 20.396, "0.025": 34.950, "0.050": 47.721,
                "0.100": 52.584, "0.250": 62.857, "1.0": 73.223, "10.0": 75.886}


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def parse_walltimes(path: Path) -> dict:
    """Wall clock per budget, from the ladder driver's own timing log."""
    out = {}
    for m in re.finditer(r"budget=([\d.]+) rc=(\d+) wall_seconds=(\d+) rows=(\d+)",
                         path.read_text()):
        out[m.group(1)] = {"rc": int(m.group(2)), "wall_seconds": int(m.group(3)),
                           "rows": int(m.group(4))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="provenance/figure42_budget")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "repro_figure42_series.json"
    if json_path.exists() and not args.force:
        raise SystemExit(f"REFUSING to overwrite {json_path}. Pass --force.")

    P = REPO_ROOT / "provenance"
    floor_rows = {int(r["seed"]): float(r["policy_cost"])
                  for r in load_csv(P / "b3_hungarian_distance_only_test_20260731.csv")}
    walls = parse_walltimes(P / "a3_ladder_walltimes.txt")

    print(f"floor: distance-only Hungarian, {len(floor_rows)} instances, "
          f"mean {np.mean(list(floor_rows.values())):.4f}")
    print(f"wall clock source: provenance/figure42_budget/a3_ladder_walltimes.txt\n")

    rungs = {}
    for b in BUDGETS:
        rows = load_csv(P / f"a3_rh_test_h1_b{b}s.csv")
        seeds = [int(r["seed"]) for r in rows]
        if set(seeds) != set(floor_rows):
            raise SystemExit(f"budget {b}: seed set differs from the floor")
        rh = np.array([float(r["policy_cost"]) for r in rows])
        milp = np.array([float(r["milp_oracle_cost_from_cache"]) for r in rows])
        flo = np.array([floor_rows[s] for s in seeds])
        solves = np.array([float(r["n_window_solves"]) for r in rows])
        fallbacks = np.array([float(r["n_solves_fallback_fired"]) for r in rows])
        no_inc = np.array([float(r["n_solves_no_incumbent"]) for r in rows])
        served = np.array([int(r["serve_all_flag"]) for r in rows])

        gap_rom = float((flo.mean() - rh.mean()) / (flo.mean() - milp.mean())) * 100
        gap_mor = float(np.mean((flo - rh) / (flo - milp))) * 100
        wall = walls[b]["wall_seconds"]
        n = len(rows)
        ms_per_solve = wall / n / solves.mean() * 1000.0
        fb_frac = float(fallbacks.sum() / solves.sum()) * 100

        rungs[b] = {
            "nominal_ms": NOMINAL_MS[b],
            "n": n,
            "mean_rh_cost": float(rh.mean()),
            "gap_closure_ratio_of_means_pct": gap_rom,
            "gap_closure_mean_of_ratios_pct": gap_mor,
            "wall_seconds": wall,
            "mean_window_solves_per_instance": float(solves.mean()),
            "measured_ms_per_solved_decision": ms_per_solve,
            "measured_over_nominal": ms_per_solve / NOMINAL_MS[b],
            "fallback_pct_of_window_solves": fb_frac,
            "no_incumbent_pct": float(no_inc.sum() / solves.sum()) * 100,
            "serve_all": f"{int(served.sum())}/{n}",
        }
        print(f"budget {b:>6}s  gap ROM {gap_rom:7.3f} %  gap MOR {gap_mor:7.3f} %  "
              f"compute {ms_per_solve:8.3f} ms  fallback {fb_frac:6.2f} %  "
              f"serve-all {int(served.sum())}/{n}")

    # Compare against the figure's own manifest and against the claims file.
    print("\n=== against figures/output/budget_curve.sources.txt ===")
    worst = (0.0, None)
    for b in BUDGETS:
        dg = abs(rungs[b]["gap_closure_ratio_of_means_pct"] - MANIFEST_GAP[b])
        dc = abs(rungs[b]["measured_ms_per_solved_decision"] - MANIFEST_COMPUTE_MS[b])
        for d, what in ((dg, f"{b}/gap"), (dc, f"{b}/compute")):
            if d > worst[0]:
                worst = (d, what)
        print(f"  {b:>6}s  gap {rungs[b]['gap_closure_ratio_of_means_pct']:7.3f} vs "
              f"{MANIFEST_GAP[b]:7.3f} (d={dg:.4f})   compute "
              f"{rungs[b]['measured_ms_per_solved_decision']:8.3f} vs "
              f"{MANIFEST_COMPUTE_MS[b]:8.3f} (d={dc:.4f})")
    print(f"  largest discrepancy against the manifest: {worst[0]:.4f} at {worst[1]}")

    print("\n=== against THESIS_CLAIMS.md ===")
    claims = {}
    for b, (cid, val) in PUBLISHED_GAP.items():
        rom = rungs[b]["gap_closure_ratio_of_means_pct"]
        mor = rungs[b]["gap_closure_mean_of_ratios_pct"]
        claims[cid] = {
            "budget": b, "published": val,
            "ratio_of_means": rom, "mean_of_ratios": mor,
            "matches_ratio_of_means": abs(rom - val) < 0.005,
            "matches_mean_of_ratios": abs(mor - val) < 0.005,
        }
        which = ("ratio of means" if claims[cid]["matches_ratio_of_means"]
                 else "mean of ratios" if claims[cid]["matches_mean_of_ratios"]
                 else "NEITHER")
        print(f"  {cid} at {b}s: published {val}, ROM {rom:.3f}, MOR {mor:.3f}"
              f"  -> matches {which}")
    for b, (cid, phrase) in PUBLISHED_FALLBACK.items():
        fb = rungs[b]["fallback_pct_of_window_solves"]
        claims[cid] = {"budget": b, "published_phrase": phrase,
                       "fallback_pct": fb}
        print(f"  {cid} at {b}s: published '{phrase}', measured {fb:.2f} %")

    payload = {
        "script": "scripts/regenerate/figure42_series.py",
        "git_head": _git_head(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "floor": "provenance/table41_main_results/b3_hungarian_distance_only_test_20260731.csv",
        "ceiling": "milp_oracle_cost_from_cache column of each rung",
        "wall_clock": "provenance/figure42_budget/a3_ladder_walltimes.txt",
        "n_instances": 200,
        "split": "test, seeds 11200 to 11399",
        "rungs": rungs,
        "claims": claims,
        "largest_discrepancy_vs_manifest": {"value": worst[0], "where": worst[1]},
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")
    print("No solver was invoked and no existing file was modified.")


if __name__ == "__main__":
    main()
