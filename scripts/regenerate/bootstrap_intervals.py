"""Regenerate the 95 per cent bootstrap intervals for Table 4.1.

Phase 1.5 step 2. The audit in CLEANUP_INVENTORY.md found that
``provenance/results_tables_v2_20260731.md`` is written by no script in the
tree: its point estimates all reproduce from the per-instance CSVs, but no
interval in Table 4.1 could be recomputed since the bootstrap that produced
them was not in the repository. This script is the missing producer.

Protocol, per methodology section 3.7: 95 per cent percentile intervals from
10,000 resamples over instances, at a fixed seed. The bootstrap resamples
**instances, not observations within a policy**, and the same resampled index
vector is reused across every policy in a given replicate, so the paired
difference intervals are genuinely paired and the gap closure intervals carry
the correlation between a policy and its floor and ceiling.

The estimator machinery is imported from
``scripts/analysis/transfer_table_stats.py`` rather than reimplemented.
``boot_indices`` draws the index matrix from
``numpy.random.default_rng(BOOT_SEED)`` and ``ci`` takes the 2.5 and 97.5
percentiles — the same two functions behind every interval in the transfer
table and the rolling-horizon ladder report, so Table 4.1 lands on the same
convention as the rest of the thesis.

No solver is invoked, nothing is trained, no checkpoint is loaded and no cache
record is read or modified. This script reads per-instance CSVs and writes to
its own output directory only.

Usage::

    PYTHONPATH="$PWD/src" python scripts/regenerate/bootstrap_intervals.py \\
        --out-dir provenance/table41_main_results

    # try a small set of obvious seeds if the default does not reproduce
    PYTHONPATH="$PWD/src" python scripts/regenerate/bootstrap_intervals.py \\
        --out-dir provenance/table41_main_results --seed-trial
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _d in ("regenerate", "analysis", "experiments"):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _d))

from transfer_table_stats import (  # noqa: E402
    N_BOOT,
    BOOT_SEED,
    boot_indices,
    ci,
    load_csv,
)

# Tried only when --seed-trial is passed and the default does not reproduce.
# Deliberately short and obvious (project bootstrap seed, two neighbouring run
# dates, evaluation seed, Gurobi seed, two common defaults) — not a search for
# a seed that fits.
OBVIOUS_SEEDS = [20260731, 20260730, 20260729, 20260801, 42, 0, 1, 2026]

LEARNED = "learned_hard"
FLOOR = "hungarian_distance_only"
CEILING = "milp"


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def _cost(row: dict) -> float | None:
    """Cost under the all-200 convention the table reports.

    ``policy_cost`` is blank on a non-serving rollout, where the evaluator
    discards the cost. The table reports Method One over all 200 with failed
    rollouts' costs included via the ``cost_including_failed`` fallback column,
    per footnote f in ``results_tables_v2_20260731.md``.
    """
    v = row.get("policy_cost", "")
    if v not in ("", None):
        return float(v)
    v = row.get("cost_including_failed", "")
    return float(v) if v not in ("", None) else None


def load_policies() -> tuple[dict, dict, list[int]]:
    """Load every Table 4.1 policy onto one aligned seed index."""
    P = REPO_ROOT / "provenance"
    R = REPO_ROOT / "results"

    a2 = load_csv(R / "a2_per_instance_test_20260730.csv")
    hard = {int(r["seed"]): r for r in a2 if r["decode_mode"] == "hard"}
    pert = {int(r["seed"]): r for r in a2 if r["decode_mode"] == "perturbed"}
    seeds = sorted(hard)

    def from_csv(path, name):
        rows = {int(r["seed"]): r for r in load_csv(path)}
        missing = set(seeds) - set(rows)
        if missing:
            raise SystemExit(f"{name}: {len(missing)} seeds missing from {path}")
        return rows

    sources = {
        FLOOR: from_csv(P / "b3_hungarian_distance_only_test_20260731.csv", FLOOR),
        "hungarian_kappa_weighted":
            from_csv(P / "b3_hungarian_kappa_weighted_test_20260731.csv", "kappa"),
        "linear_scorer_matched":
            from_csv(P / "b3_linearscorer_matched_test_20260804.csv", "linear_matched"),
        "linear_scorer_dm24":
            from_csv(P / "b3_linearscorer_test_20260731.csv", "linear_dm24"),
        "coldstart": from_csv(P / "b3_coldstart_test_20260731.csv", "coldstart"),
        "methodone": from_csv(P / "b3_methodone_test_20260731.csv", "methodone"),
        "learned_perturbed": pert,
        LEARNED: hard,
    }
    for b in ("0.010", "0.025", "0.050", "0.100", "0.250", "1.0", "10.0"):
        sources[f"rh_{b}s"] = from_csv(P / f"a3_rh_test_h1_b{b}s.csv", f"rh {b}")

    costs, served = {}, {}
    for name, rows in sources.items():
        costs[name] = np.array([_cost(rows[s]) for s in seeds], dtype=float)
        if any(v is None for v in costs[name]):
            raise SystemExit(f"{name}: a seed has no cost under either column")
        flags = [rows[s].get("serve_all_flag") for s in seeds]
        served[name] = np.array(
            [1 if f in (None, "") else int(float(f)) for f in flags], dtype=int
        )

    # The ceiling is the cached MILP objective, carried on the learned rows.
    costs[CEILING] = np.array([float(hard[s]["cost_milp_oracle"]) for s in seeds])
    served[CEILING] = np.ones(len(seeds), dtype=int)
    return costs, served, seeds


def cells(costs: dict, name: str, idx: np.ndarray, keep: np.ndarray | None = None) -> dict:
    """The three interval-bearing cells of one Table 4.1 row.

    ``keep`` restricts to a subset of instances; when given, the index matrix
    must already be drawn at that subset's size.
    """
    sl = slice(None) if keep is None else keep
    pol = costs[name][sl]
    flo = costs[FLOOR][sl]
    cei = costs[CEILING][sl]
    ref = costs[LEARNED][sl]

    bm_pol = pol[idx].mean(axis=1)
    bm_flo = flo[idx].mean(axis=1)
    bm_cei = cei[idx].mean(axis=1)
    bm_ref = ref[idx].mean(axis=1)

    # Gap closure as the RATIO OF MEANS, the estimator section 3.7 commits to.
    # The mean of per-instance ratios is computed alongside so the two can be
    # told apart.
    gc_rom = float((flo.mean() - pol.mean()) / (flo.mean() - cei.mean()))
    bs_gc_rom = (bm_flo - bm_pol) / (bm_flo - bm_cei)

    per_inst = (flo - pol) / (flo - cei)
    gc_mor = float(per_inst.mean())
    bs_gc_mor = per_inst[idx].mean(axis=1)

    diffs = pol - ref
    return {
        "n": int(pol.size),
        "mean_cost": float(pol.mean()),
        "mean_cost_ci95": ci(bm_pol),
        "gap_closure_ratio_of_means_pct": gc_rom * 100.0,
        "gap_closure_ratio_of_means_ci95_pct": tuple(x * 100.0 for x in ci(bs_gc_rom)),
        "gap_closure_mean_of_ratios_pct": gc_mor * 100.0,
        "gap_closure_mean_of_ratios_ci95_pct": tuple(x * 100.0 for x in ci(bs_gc_mor)),
        "paired_diff_vs_learned_hard": float(diffs.mean()),
        "paired_diff_vs_learned_hard_ci95": ci(diffs[idx].mean(axis=1)),
    }


# Published Table 4.1, transcribed from provenance/results_tables_v2_20260731.md,
# except the linear scorer row (from provenance/b3_linearscorer_matched_test_20260804.md).
# Point, then interval.
PUBLISHED = {
    "linear_scorer_dm24": {
        "ids": "not in the thesis, DM-24 row, cross-check only",
        "mean_cost": (185.395, (181.534, 189.340)),
        "gap": (-40.564, (-47.14, -34.34)),
        "diff": (57.200, (54.317, 60.192)),
    },
    "linear_scorer_matched": {
        "ids": "C055, C056, C057",
        "mean_cost": (180.444, (176.712, 184.226)),
        "gap": (-32.692, (-38.29, -27.22)),
        "diff": (52.249, (49.490, 54.994)),
    },
    FLOOR: {
        "ids": "C047, C048, C049",
        "mean_cost": (159.880, (156.452, 163.280)),
        "gap": (0.000, None),
        "diff": (31.685, (29.226, 34.084)),
    },
    "hungarian_kappa_weighted": {
        "ids": "C051, C052, C053",
        "mean_cost": (159.104, (155.721, 162.417)),
        "gap": (1.233, (-0.30, 2.80)),
        "diff": (30.909, (28.441, 33.297)),
    },
    "coldstart": {
        "ids": "C059, C060, C061",
        "mean_cost": (153.357, (149.733, 156.980)),
        "gap": (10.370, (6.40, 14.18)),
        "diff": (25.162, (22.623, 27.734)),
    },
    "methodone": {
        "ids": "C063, C064, C065",
        "mean_cost": (148.383, (143.637, 153.118)),
        "gap": (18.278, (11.59, 24.52)),
        "diff": (20.188, (16.193, 24.267)),
    },
    "rh_0.010s": {"ids": "Table 4.1 extra row", "mean_cost": (147.050, (143.667, 150.431)),
                  "gap": (20.396, (17.35, 23.30)), "diff": (18.856, (16.475, 21.213))},
    "rh_0.025s": {"ids": "Table 4.1 extra row", "mean_cost": (137.896, (134.615, 141.138)),
                  "gap": (34.950, (31.94, 37.87)), "diff": (9.701, (7.350, 12.031))},
    "learned_perturbed": {
        "ids": "C068, C069, C070",
        "mean_cost": (135.147, (132.182, 138.088)),
        "gap": (39.320, (36.52, 42.05)),
        "diff": (6.952, (5.217, 8.602)),
    },
    "rh_0.050s": {"ids": "Table 4.1 extra row", "mean_cost": (129.862, (126.736, 132.978)),
                  "gap": (47.721, (45.04, 50.32)), "diff": (1.668, (-0.546, 3.874))},
    LEARNED: {
        "ids": "C072, C073, C074",
        "mean_cost": (128.195, (125.129, 131.289)),
        "gap": (50.372, (47.45, 53.19)),
        "diff": (0.0, None),
    },
    "rh_0.100s": {"ids": "Table 4.1 extra row", "mean_cost": (126.803, (123.901, 129.820)),
                  "gap": (52.584, (49.91, 55.14)), "diff": (-1.391, (-3.289, 0.542))},
    "rh_0.250s": {"ids": "Table 4.1 extra row", "mean_cost": (120.342, (117.580, 123.106)),
                  "gap": (62.857, (60.63, 65.01)), "diff": (-7.853, (-9.822, -5.919))},
    "rh_1.0s": {"ids": "Table 4.1 extra row", "mean_cost": (113.822, (111.360, 116.340)),
                "gap": (73.223, (71.12, 75.20)), "diff": (-14.373, (-16.189, -12.583))},
    "rh_10.0s": {"ids": "Table 4.1 extra row", "mean_cost": (112.146, (109.792, 114.592)),
                 "gap": (75.886, (73.73, 77.88)), "diff": (-16.048, (-18.044, -14.101))},
    CEILING: {
        "ids": "C075, C076, C077",
        "mean_cost": (96.978, (94.853, 99.069)),
        "gap": (100.000, None),
        "diff": (-31.216, (-33.032, -29.417)),
    },
}

ORDER = list(PUBLISHED.keys())


def compare_all(results: dict) -> dict:
    """Diff every regenerated interval against its published counterpart."""
    out, worst = {}, {"value": 0.0, "where": None}
    for name in ORDER:
        pub, got = PUBLISHED[name], results[name]
        row = {}
        for label, pub_pair, got_point, got_ci, prec in (
            ("mean_cost", pub["mean_cost"], got["mean_cost"],
             got["mean_cost_ci95"], 3),
            ("gap_closure", pub["gap"], got["gap_closure_ratio_of_means_pct"],
             got["gap_closure_ratio_of_means_ci95_pct"], 2),
            ("paired_diff", pub["diff"], got["paired_diff_vs_learned_hard"],
             got["paired_diff_vs_learned_hard_ci95"], 3),
        ):
            pub_point, pub_ci = pub_pair
            entry = {
                "published_point": pub_point,
                "regenerated_point": got_point,
                "point_abs_diff": abs(got_point - pub_point),
            }
            if pub_ci is None:
                entry["published_ci"] = None
                entry["regenerated_ci"] = got_ci
                entry["ci_reproduces"] = None
            else:
                dlo = abs(got_ci[0] - pub_ci[0])
                dhi = abs(got_ci[1] - pub_ci[1])
                # Gap closure is printed to two decimals, the others to three,
                # so a reproduction is a match at the printed precision.
                tol = 0.5 * 10 ** (-prec) + 1e-12
                entry.update({
                    "published_ci": list(pub_ci),
                    "regenerated_ci": list(got_ci),
                    "ci_abs_diff_low": dlo,
                    "ci_abs_diff_high": dhi,
                    "ci_reproduces": bool(dlo <= tol and dhi <= tol),
                    "tolerance": tol,
                })
                for d, end in ((dlo, "low"), (dhi, "high")):
                    if d > worst["value"]:
                        worst = {"value": d, "where": f"{name}/{label}/{end}"}
            row[label] = entry
        out[name] = row
    return {"per_row": out, "largest_ci_discrepancy": worst}


def run_at_seed(costs: dict, seed: int, n: int) -> dict:
    """Recompute every row's intervals at one seed."""
    global BOOT_SEED
    import transfer_table_stats as tts
    saved = tts.BOOT_SEED
    tts.BOOT_SEED = seed
    try:
        idx = tts.boot_indices(n)
    finally:
        tts.BOOT_SEED = saved
    return {name: cells(costs, name, idx) for name in ORDER}


# Gitignored input, paired with the committed artefact carrying the result.
# Checked up front so a fresh clone gets one diagnostic line rather than a
# FileNotFoundError from deep in the load.
REQUIRED_INPUTS = [
    ("results/a2_per_instance_test_20260730.csv",
     "provenance/table41_main_results/repro_table41_intervals.json"),
]


def require_inputs() -> None:
    """Abort with one line if a gitignored input is absent."""
    for rel, shipped in REQUIRED_INPUTS:
        if not (REPO_ROOT / rel).exists():
            raise SystemExit(
                f"Missing {rel}. It is gitignored and absent from a fresh "
                f"clone, so this script can only run on a tree that carries "
                f"it. The committed {shipped} carries the result."
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="provenance/table41_main_results")
    ap.add_argument("--seed", type=int, default=BOOT_SEED)
    ap.add_argument("--seed-trial", action="store_true",
                    help="If the default seed does not reproduce, retry a short "
                         "fixed list of obvious seeds and report. Not a search.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    require_inputs()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "repro_table41_intervals.json"
    md_path = out_dir / "repro_table41_intervals.md"
    for p in (json_path, md_path):
        if p.exists() and not args.force:
            raise SystemExit(f"REFUSING to overwrite {p}. Pass --force.")

    costs, served, seeds = load_policies()
    n = len(seeds)
    print(f"{n} instances, seeds {min(seeds)} to {max(seeds)}")
    print(f"bootstrap: {N_BOOT} resamples of INSTANCES, seed {args.seed}, "
          f"percentile interval at 2.5 and 97.5")
    print("one index matrix shared by every policy, so all intervals are paired\n")

    results = run_at_seed(costs, args.seed, n)
    cmp = compare_all(results)

    n_ci = sum(1 for r in cmp["per_row"].values() for c in r.values()
               if c["ci_reproduces"] is not None)
    n_ok = sum(1 for r in cmp["per_row"].values() for c in r.values()
               if c["ci_reproduces"] is True)
    print(f"intervals with a published counterpart : {n_ci}")
    print(f"reproduce at the printed precision      : {n_ok}")
    print(f"largest absolute discrepancy            : "
          f"{cmp['largest_ci_discrepancy']['value']:.6f} "
          f"at {cmp['largest_ci_discrepancy']['where']}\n")

    # Method One over the 165 serving instances as well as all 200.
    keep = served["methodone"] == 1
    idx165 = None
    import transfer_table_stats as tts
    saved = tts.BOOT_SEED
    tts.BOOT_SEED = args.seed
    try:
        idx165 = tts.boot_indices(int(keep.sum()))
    finally:
        tts.BOOT_SEED = saved
    m1_165 = cells(costs, "methodone", idx165, keep=keep)
    print(f"Method One over all 200 : mean {results['methodone']['mean_cost']:.3f} "
          f"{results['methodone']['mean_cost_ci95']}")
    print(f"Method One over the 165 : mean {m1_165['mean_cost']:.3f} "
          f"{m1_165['mean_cost_ci95']}\n")

    seed_trial = None
    if args.seed_trial and n_ok < n_ci:
        seed_trial = {}
        print("Default seed did not reproduce every interval. Trying the short list.")
        for s in OBVIOUS_SEEDS:
            r = run_at_seed(costs, s, n)
            c = compare_all(r)
            ok = sum(1 for row in c["per_row"].values() for cell in row.values()
                     if cell["ci_reproduces"] is True)
            seed_trial[s] = {
                "n_reproduced": ok,
                "n_compared": n_ci,
                "largest_discrepancy": c["largest_ci_discrepancy"],
            }
            print(f"  seed {s:>9}: {ok}/{n_ci} intervals reproduce, "
                  f"largest discrepancy {c['largest_ci_discrepancy']['value']:.6f}")

    payload = {
        "script": "scripts/regenerate/bootstrap_intervals.py",
        "git_head": _git_head(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "n_boot": N_BOOT,
            "seed": args.seed,
            "interval": "95 per cent percentile, 2.5 and 97.5",
            "resamples": "instances, not observations within a policy",
            "pairing": "one index matrix shared by every policy in a replicate",
            "gap_closure_estimator": "ratio of means, matching the point estimate "
                                     "and methodology section 3.7",
            "estimator_source": "boot_indices and ci imported from "
                                "scripts/analysis/transfer_table_stats.py",
        },
        "n_instances": n,
        "seed_min": min(seeds),
        "seed_max": max(seeds),
        "rows": results,
        "method_one_over_165_serving": m1_165,
        "comparison": cmp,
        "seed_trial": seed_trial,
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2, default=list)

    with md_path.open("w") as f:
        f.write("# Table 4.1 bootstrap intervals, regenerated\n\n")
        f.write(f"{N_BOOT} resamples of instances, seed {args.seed}, "
                f"95 per cent percentile interval, n={n}, "
                f"seeds {min(seeds)} to {max(seeds)}.\n")
        f.write("Gap closure intervals are on the ratio of means.\n\n")
        f.write("| Row | IDs | Mean cost, regenerated | Mean cost, published | "
                "Gap closure, regenerated | Gap closure, published | "
                "Paired diff, regenerated | Paired diff, published |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for name in ORDER:
            g, p = results[name], PUBLISHED[name]
            def fmt(pt, c, prec):
                if c is None or c[0] is None:
                    return f"{pt:.{prec}f}"
                return f"{pt:.{prec}f} [{c[0]:.{prec}f}, {c[1]:.{prec}f}]"
            f.write(
                f"| {name} | {p['ids']} | "
                f"{fmt(g['mean_cost'], g['mean_cost_ci95'], 3)} | "
                f"{fmt(p['mean_cost'][0], p['mean_cost'][1], 3)} | "
                f"{fmt(g['gap_closure_ratio_of_means_pct'], g['gap_closure_ratio_of_means_ci95_pct'], 2)} | "
                f"{fmt(p['gap'][0], p['gap'][1], 2)} | "
                f"{fmt(g['paired_diff_vs_learned_hard'], g['paired_diff_vs_learned_hard_ci95'], 3)} | "
                f"{fmt(p['diff'][0], p['diff'][1], 3)} |\n"
            )

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print("No solver was invoked, nothing was trained, and no existing file was modified.")


if __name__ == "__main__":
    main()
