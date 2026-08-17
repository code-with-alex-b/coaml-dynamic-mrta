"""Analyse the four-budget h=1 rolling-horizon ladder at the transfer scales.

Joins the ladder's per-instance costs to the learned policy and the two
Hungarian variants recorded for the same seeds in the transfer table, so every
comparison is paired on the instance.

Reports, per scale and budget, the mean cost with a 95 per cent bootstrap
interval, the serve-all count, how much of the episode the rolling horizon
actually decided (no-incumbent and fallback rates), the honest per-decision
compute including model build, and the paired cost ratio against the learned
policy with a sign test.

Read only. Writes one markdown report and one JSON summary under provenance/.
"""

from __future__ import annotations

import argparse
import csv
import json
from math import comb
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
N_BOOT, BOOT_SEED = 10_000, 20260731
TRANSFER = REPO_ROOT / "provenance" / "table42_transfer"
# Median per-decision latency of the learned policy (DM-40 method) from phase3_timing_summary.json; same machine, same Rosetta 2 environment, so timings are comparable.
SCALE_MAP = {
    "R=10,T=30": {"key": "r10t30", "csv": "transfer_r10t30_per_instance.csv"},
    "R=30,T=90": {"key": "r30t90", "csv": "transfer_r30t90_per_instance.csv"},
}


def ci(x):
    ok = x[np.isfinite(x)]
    return (float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5)))


def sign_test(d):
    nz = d[d != 0]
    n = int(nz.size)
    if n == 0:
        return {"n": 0, "k": 0, "p": None}
    k = int((nz < 0).sum())
    tail = min(k, n - k)
    p = min(1.0, sum(comb(n, i) for i in range(tail + 1)) / (2 ** n) * 2)
    return {"n": n, "k": k, "p": float(p)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = REPO_ROOT / args.dir

    rows = list(csv.DictReader((d / "rh_ladder_per_instance.csv").open()))
    timing = json.load((TRANSFER / "phase3_timing_summary.json").open())["timings"]

    ref = {}
    for label, m in SCALE_MAP.items():
        ref[label] = {
            int(r["seed"]): r
            for r in csv.DictReader((TRANSFER / m["csv"]).open())
        }

    out, md = [], []
    md.append("# Rolling horizon h=1 budget ladder at the transfer scales\n")
    md.append("Budgets 0.050, 0.100, 0.250 and 1.0 seconds per window solve, "
              "60 instances per scale per budget, h=1 so the policy is online "
              "and non-clairvoyant. Intervals are 95 per cent bootstrap over "
              "10,000 paired resamples, seed 20260731.\n")
    md.append("The rolling horizon decides a window only when the solver "
              "returns an incumbent inside the budget. Otherwise the window "
              "falls back to Hungarian-on-kappa, so the fallback rate is how "
              "much of each episode the rolling horizon did not decide.\n")
    md.append("Per-decision compute is model build plus solve. The build sits "
              "outside the Gurobi time limit, so it is not covered by the "
              "budget and the budget label understates the real cost.\n")

    for label, m in SCALE_MAP.items():
        sc = [r for r in rows if r["scale"] == label]
        if not sc:
            continue
        budgets = sorted({float(r["budget_seconds"]) for r in sc})
        pol_lat = timing[m["key"]]["stages"]["whole decision"]["median_us"] / 1e6

        md.append(f"\n## {label}\n")
        md.append("| Budget | n | Mean RH cost | Serve-all | Windows | "
                  "No incumbent | Fallback rate | Time-limit rate | "
                  "Mean build | Real per-decision | Ladder wall |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")

        for b in budgets:
            rs = [r for r in sc if float(r["budget_seconds"]) == b]
            seeds = [int(r["seed"]) for r in rs]
            cost = np.array([float(r["cost"]) for r in rs])
            idx = np.random.default_rng(BOOT_SEED).integers(0, len(rs), size=(N_BOOT, len(rs)))
            lo, hi = ci(cost[idx].mean(axis=1))

            nsolve = np.array([float(r["n_window_solves"]) for r in rs])
            nbuilt = np.array([float(r["n_models_built"]) for r in rs])
            noinc = np.array([float(r["n_no_incumbent"]) for r in rs])
            nfb = np.array([float(r["n_fallbacks"]) for r in rs])
            ntl = np.array([float(r["n_time_limit"]) for r in rs])
            build = np.array([float(r["mean_build_s"]) for r in rs if r["mean_build_s"]])
            perdec = np.array([float(r["mean_total_per_decision_s"]) for r in rs if r["mean_total_per_decision_s"]])
            wall = np.array([float(r["wall_seconds"]) for r in rs])
            served = int(sum(int(r["served_all"]) for r in rs))

            md.append(
                f"| {b:.3f} s | {len(rs)} | {cost.mean():.3f} [{lo:.3f}, {hi:.3f}] | "
                f"{served}/{len(rs)} | {nsolve.mean():.1f} | "
                f"{100 * noinc.sum() / nbuilt.sum():.1f} % | "
                f"{100 * nfb.sum() / nsolve.sum():.1f} % | "
                f"{100 * ntl.sum() / nbuilt.sum():.1f} % | "
                f"{build.mean() * 1e3:.1f} ms | {perdec.mean() * 1e3:.1f} ms | "
                f"{wall.sum():.0f} s |"
            )

            pol = np.array([float(ref[label][s]["policy_cost"]) for s in seeds])
            gre = np.array([float(ref[label][s]["hungarian_distance_only_cost"]) for s in seeds])
            kap = np.array([float(ref[label][s]["hungarian_kappa_cost"]) for s in seeds])
            rec = {
                "scale": label, "budget": b, "n": len(rs),
                "mean_rh_cost": float(cost.mean()), "ci": [lo, hi],
                "served_all": served,
                "mean_windows": float(nsolve.mean()),
                "no_incumbent_rate": float(noinc.sum() / nbuilt.sum()),
                "fallback_rate": float(nfb.sum() / nsolve.sum()),
                "time_limit_rate": float(ntl.sum() / nbuilt.sum()),
                "mean_build_s": float(build.mean()),
                "mean_per_decision_s": float(perdec.mean()),
                "policy_median_per_decision_s": pol_lat,
                "compute_multiple_vs_policy": float(perdec.mean() / pol_lat),
                "ladder_wall_s": float(wall.sum()),
            }
            for name, base in [("learned_policy", pol),
                               ("hungarian_distance_only", gre),
                               ("hungarian_kappa", kap)]:
                diff = cost - base
                rec[name] = {
                    "mean_cost": float(base.mean()),
                    "ratio_of_means": float(cost.mean() / base.mean()),
                    "ratio_ci": list(ci(cost[idx].mean(axis=1) / base[idx].mean(axis=1))),
                    "mean_paired_difference": float(diff.mean()),
                    "sign_test": sign_test(diff),
                }
            out.append(rec)

        md.append(f"\nLearned policy on the same 60 seeds: mean cost "
                  f"{np.mean([float(ref[label][s]['policy_cost']) for s in seeds]):.3f}, "
                  f"median per-decision {pol_lat * 1e6:.0f} us.\n")
        md.append("| Budget | RH / learned policy | Paired diff | Sign test | "
                  "RH / distance-only Hungarian | Compute vs policy |")
        md.append("|---|---|---|---|---|---|")
        for rec in [r for r in out if r["scale"] == label]:
            p = rec["learned_policy"]
            st = p["sign_test"]
            md.append(
                f"| {rec['budget']:.3f} s | {p['ratio_of_means']:.4f} "
                f"[{p['ratio_ci'][0]:.4f}, {p['ratio_ci'][1]:.4f}] | "
                f"{p['mean_paired_difference']:+.3f} | "
                f"RH better on {st['k']}/{st['n']}, p={st['p']:.2e} | "
                f"{rec['hungarian_distance_only']['ratio_of_means']:.4f} | "
                f"{rec['compute_multiple_vs_policy']:.0f}x |"
            )

    total_wall = sum(r["ladder_wall_s"] for r in out)
    md.append(f"\nTotal ladder wall clock, both scales, all budgets: "
              f"{total_wall:.0f} s = {total_wall / 3600:.2f} h.\n")

    jp, mp = d / "rh_ladder_summary.json", d / "rh_ladder_report.md"
    for p in (jp, mp):
        if p.exists():
            raise SystemExit(f"REFUSING to overwrite {p}")
    json.dump({"n_bootstrap": N_BOOT, "bootstrap_seed": BOOT_SEED,
               "results": out}, jp.open("w"), indent=2)
    mp.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {jp}\nwrote {mp}")


if __name__ == "__main__":
    main()
