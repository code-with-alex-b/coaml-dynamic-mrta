"""Statistics for the rolling-horizon h=1 density row (R=10, T=60, 1.0s budget).

Consumes the per-instance CSV written by scripts/experiments/rolling_horizon_baseline.py
and the learned policy's per-instance CSV from the zero-shot transfer table.
Nothing is re-solved and no cache record is read or written.

Conventions are taken from scripts/analysis/transfer_table_stats.py so the numbers are
comparable with the other three rows of that table: 95 per cent bootstrap
intervals over 10,000 paired resamples at seed 20260731, resampling instances
rather than costs, and reusing one resampled index vector for both arms of
every paired estimator.

Two instance sets are reported side by side:

  common   instances the rolling horizon and the learned policy both served
  table192 the 192 instances the transfer table already uses, that is the
           instances the policy and both Hungarian baselines all served

The policy's seconds per instance come from the DM-40 timing harness output
stored in phase3_timing_summary.json. The compute multiple is seconds per
instance on both sides, not per decision against per window solve, because the
two sides do not take the same number of decisions.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
N_BOOT = 10_000
BOOT_SEED = 20260731
FALLBACK_FLAG_THRESHOLD = 0.10

TRANSFER_DIR = REPO_ROOT / "provenance" / "table42_transfer"
POLICY_CSV = TRANSFER_DIR / "transfer_r10t60_per_instance.csv"
TIMING_JSON = TRANSFER_DIR / "phase3_timing_summary.json"


def _f(s):
    return None if s is None or s == "" else float(s)


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def boot_indices(n: int) -> np.ndarray:
    rng = np.random.default_rng(BOOT_SEED)
    return rng.integers(0, n, size=(N_BOOT, n))


def ci(samples: np.ndarray) -> tuple:
    ok = samples[np.isfinite(samples)]
    if ok.size == 0:
        return (None, None)
    return (float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5)))


def mean_ci(x: np.ndarray) -> dict:
    idx = boot_indices(x.size)
    lo, hi = ci(x[idx].mean(axis=1))
    return {"point": float(x.mean()), "lo": lo, "hi": hi, "n": int(x.size)}


def paired_ratio(num: np.ndarray, den: np.ndarray) -> dict:
    """Ratio of means and mean of per-instance ratios, sharing one index draw."""
    idx = boot_indices(num.size)
    rom = float(num.mean() / den.mean())
    rom_bs = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    per = num / den
    mor = float(per.mean())
    mor_bs = per[idx].mean(axis=1)
    rom_lo, rom_hi = ci(rom_bs)
    mor_lo, mor_hi = ci(mor_bs)
    return {
        "ratio_of_means": {"point": rom, "lo": rom_lo, "hi": rom_hi},
        "mean_of_ratios": {"point": mor, "lo": mor_lo, "hi": mor_hi},
        "n": int(num.size),
    }


def fmt(d: dict, places: int = 4) -> str:
    if d.get("lo") is None:
        return f"{d['point']:.{places}f}"
    return f"{d['point']:.{places}f} [{d['lo']:.{places}f}, {d['hi']:.{places}f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rh-csv", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    rh_rows = load_csv(Path(args.rh_csv))
    pol_rows = load_csv(POLICY_CSV)
    rh = {int(r["seed"]): r for r in rh_rows}
    pol = {int(r["seed"]): r for r in pol_rows}

    budgets = {r["budget_seconds"] for r in rh_rows}
    if budgets != {"1.0"}:
        raise SystemExit(f"Expected a single 1.0s budget, found {budgets}")
    missing = sorted(set(rh) - set(pol))
    if missing:
        raise SystemExit(f"Seeds in the RH run with no policy row: {missing}")

    seeds = sorted(rh)
    out: dict = {
        "rh_csv": str(args.rh_csv),
        "policy_csv": str(POLICY_CSV),
        "n_instances": len(seeds),
        "seed_range": [seeds[0], seeds[-1]],
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED},
    }

    rh_served = {s for s in seeds if rh[s]["serve_all_flag"] == "1"}
    pol_served = {s for s in seeds if pol[s]["serve_all_flag"] == "1"}
    greedy_served = {s for s in seeds if pol[s]["greedy_served_all"] == "1"}
    kappa_served = {s for s in seeds if pol[s]["kappa_served_all"] == "1"}
    table192 = sorted(pol_served & greedy_served & kappa_served)
    common = sorted(rh_served & pol_served)

    out["serve_all"] = {
        "rolling_horizon": {
            "n_served": len(rh_served),
            "n_total": len(seeds),
            "failed_seeds": sorted(set(seeds) - rh_served),
        },
        "policy": {
            "n_served": len(pol_served),
            "failed_seeds": sorted(set(seeds) - pol_served),
        },
        "hungarian_distance_only_failed": sorted(set(seeds) - greedy_served),
        "hungarian_kappa_failed": sorted(set(seeds) - kappa_served),
        "table192_n": len(table192),
        "table192_excluded": sorted(set(seeds) - set(table192)),
        "common_rh_and_policy_n": len(common),
        "common_excluded": sorted(set(seeds) - set(common)),
    }

    # On the 192 set an instance the rolling horizon failed has no cost, so each set is reported both as-is and restricted to what the rolling horizon actually served.
    def arrays(subset):
        ss = [s for s in subset if _f(rh[s]["policy_cost"]) is not None]
        r = np.array([_f(rh[s]["policy_cost"]) for s in ss])
        p = np.array([_f(pol[s]["policy_cost"]) for s in ss])
        return ss, r, p

    out["sets"] = {}
    for name, subset in (("common", common), ("table192", table192)):
        ss, r, p = arrays(subset)
        out["sets"][name] = {
            "n_requested": len(subset),
            "n_used": len(ss),
            "dropped_rh_failed": sorted(set(subset) - set(ss)),
            "rh_mean_cost": mean_ci(r),
            "policy_mean_cost": mean_ci(p),
            "ratio_rh_over_policy": paired_ratio(r, p),
            "mean_paired_difference": mean_ci(r - p),
            "n_rh_cheaper": int((r < p).sum()),
        }

    # Mean rolling-horizon cost over everything it served, no policy pairing.
    _, r_all, _ = arrays(sorted(rh_served))
    out["rh_mean_cost_all_served"] = mean_ci(r_all)

    tot_solves = sum(int(rh[s]["n_window_solves"]) for s in seeds)
    counters = {}
    for key in ("n_solves_hit_time_limit", "n_solves_no_incumbent",
                "n_solves_fallback_fired"):
        tot = sum(int(rh[s][key]) for s in seeds)
        counters[key] = {"count": tot, "fraction": tot / tot_solves}
    counters["n_window_solves"] = {"count": tot_solves, "fraction": 1.0}
    per_inst_fb = np.array([
        int(rh[s]["n_solves_fallback_fired"]) / int(rh[s]["n_window_solves"])
        for s in seeds
    ])
    counters["fallback_fraction_per_instance"] = {
        "mean": float(per_inst_fb.mean()),
        "min": float(per_inst_fb.min()),
        "max": float(per_inst_fb.max()),
        "n_instances_over_threshold": int(
            (per_inst_fb > FALLBACK_FLAG_THRESHOLD).sum()
        ),
    }
    counters["fallback_flag_threshold"] = FALLBACK_FLAG_THRESHOLD
    counters["fallback_flag_tripped"] = bool(
        counters["n_solves_fallback_fired"]["fraction"] > FALLBACK_FLAG_THRESHOLD
    )
    out["solve_counters"] = counters

    wall = np.array([_f(rh[s]["instance_wall_clock_seconds"]) for s in seeds])
    build = np.array([_f(rh[s]["measured_total_build_seconds"]) for s in seeds])
    solve = np.array([_f(rh[s]["measured_total_solve_seconds"]) for s in seeds])
    with open(TIMING_JSON) as f:
        timing = json.load(f)
    pol_s = float(timing["timings"]["r10t60"]["per_instance_total_s"])
    out["wall_clock"] = {
        "rh_seconds_per_instance": {
            "mean": float(wall.mean()), "min": float(wall.min()),
            "max": float(wall.max()),
        },
        "rh_build_seconds_per_instance": float(build.mean()),
        "rh_solve_seconds_per_instance": float(solve.mean()),
        "rh_other_seconds_per_instance": float((wall - build - solve).mean()),
        "build_share_of_wall": float(build.sum() / wall.sum()),
        "solve_share_of_wall": float(solve.sum() / wall.sum()),
        "mean_build_s_per_window_solve": float(
            build.sum() / sum(
                int(rh[s]["n_window_solves"]) for s in seeds
            )
        ),
        "mean_solve_s_per_window_solve": float(solve.sum() / tot_solves),
        "policy_seconds_per_instance": pol_s,
        "policy_timing_source": str(TIMING_JSON),
        "compute_multiple_per_instance": float(wall.mean() / pol_s),
        "compute_multiple_solve_time_only": float(solve.mean() / pol_s),
    }

    P = print
    P("=" * 72)
    P("Rolling horizon h=1, R=10 T=60, budget 1.0s, seeds 90000-90199")
    P("=" * 72)
    sa = out["serve_all"]
    N = len(seeds)
    P(f"\nServe all: rolling horizon {sa['rolling_horizon']['n_served']}/{N}, "
      f"policy {sa['policy']['n_served']}/{N}")
    P(f"  rolling horizon fails on: {sa['rolling_horizon']['failed_seeds']}")
    P(f"  policy fails on:          {sa['policy']['failed_seeds']}")
    P(f"  common served set n = {sa['common_rh_and_policy_n']}, "
      f"table 192 set n = {sa['table192_n']}")
    P(f"  common excludes: {sa['common_excluded']}")
    P(f"  192 set excludes: {sa['table192_excluded']}")

    P(f"\nMean rolling-horizon cost over all it served "
      f"(n={out['rh_mean_cost_all_served']['n']}): "
      f"{fmt(out['rh_mean_cost_all_served'])}")

    for name in ("common", "table192"):
        d = out["sets"][name]
        P(f"\n--- {name} set, n used = {d['n_used']} "
          f"(requested {d['n_requested']}, dropped {d['dropped_rh_failed']}) ---")
        P(f"  mean RH cost      {fmt(d['rh_mean_cost'])}")
        P(f"  mean policy cost  {fmt(d['policy_mean_cost'])}")
        P(f"  ratio of means    {fmt(d['ratio_rh_over_policy']['ratio_of_means'])}")
        P(f"  mean of ratios    {fmt(d['ratio_rh_over_policy']['mean_of_ratios'])}")
        P(f"  paired difference {fmt(d['mean_paired_difference'])}")
        P(f"  RH cheaper on     {d['n_rh_cheaper']}/{d['n_used']}")

    c = out["solve_counters"]
    P(f"\n--- solve counters, over {c['n_window_solves']['count']} window "
      f"solves ---")
    for k in ("n_solves_hit_time_limit", "n_solves_no_incumbent",
              "n_solves_fallback_fired"):
        P(f"  {k:28s} {c[k]['count']:6d}  {c[k]['fraction']:.4f}")
    pf = c["fallback_fraction_per_instance"]
    P(f"  per-instance fallback fraction: mean {pf['mean']:.4f}, "
      f"range [{pf['min']:.4f}, {pf['max']:.4f}], "
      f"{pf['n_instances_over_threshold']}/{N} instances above "
      f"{FALLBACK_FLAG_THRESHOLD:.2f}")
    if c["fallback_flag_tripped"]:
        P(f"  *** FLAG: fallback fired on "
          f"{c['n_solves_fallback_fired']['fraction']:.1%} of window solves, "
          f"above the one in ten threshold ***")

    w = out["wall_clock"]
    P(f"\n--- wall clock ---")
    P(f"  RH seconds per instance   {w['rh_seconds_per_instance']['mean']:.3f} "
      f"(range {w['rh_seconds_per_instance']['min']:.1f} to "
      f"{w['rh_seconds_per_instance']['max']:.1f})")
    P(f"    model build             {w['rh_build_seconds_per_instance']:.3f} "
      f"({w['build_share_of_wall']:.1%})")
    P(f"    solve                   {w['rh_solve_seconds_per_instance']:.3f} "
      f"({w['solve_share_of_wall']:.1%})")
    P(f"    simulator and other     {w['rh_other_seconds_per_instance']:.3f}")
    P(f"  mean build per window solve {w['mean_build_s_per_window_solve']:.4f}s")
    P(f"  mean solve per window solve {w['mean_solve_s_per_window_solve']:.4f}s")
    P(f"  policy seconds per instance {w['policy_seconds_per_instance']:.4f}")
    P(f"  compute multiple            "
      f"{w['compute_multiple_per_instance']:.1f}x "
      f"(solve time only {w['compute_multiple_solve_time_only']:.1f}x)")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        P(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
