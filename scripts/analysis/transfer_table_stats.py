"""Phase 2 statistics for the zero-shot transfer table.

Reads the per-instance CSVs written by ``transfer_table_eval.py`` and, per
scale and per Hungarian variant, reports

  * ratio of mean costs (headline estimator)
  * mean of per-instance ratios (alongside estimator)
  * 95 per cent bootstrap intervals, 10,000 paired resamples, seed 20260731
  * the paired difference on the same instances, with an exact two-sided sign
    test
  * gap closure under both estimators, with intervals, only where an offline
    anticipative MILP benchmark is stored
  * leave-one-out sensitivity, being the largest absolute change in each
    estimator from dropping any single instance, and the seed responsible

The bootstrap resamples instances, not costs, and reuses the same resampled
index vector for the policy and the baseline, so every interval respects the
pairing that both estimators depend on.

Instances enter the statistics only when the policy and both Hungarian
variants served every task, since a cost ratio against a baseline that dropped
tasks is not like for like. Every excluded instance is named in the output.
"""

from __future__ import annotations

import argparse
import csv
import json
from math import comb
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
N_BOOT = 10_000
BOOT_SEED = 20260731
BASELINES = [
    ("hungarian_distance_only_cost", "distance-only Hungarian"),
    ("hungarian_kappa_cost", "kappa-weighted Hungarian"),
]


def _f(s):
    return None if s == "" else float(s)


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def sign_test(diffs: np.ndarray) -> dict:
    """Exact two-sided sign test on paired differences, zeros discarded."""
    nz = diffs[diffs != 0]
    n = int(nz.size)
    k = int((nz < 0).sum())  # policy cheaper than baseline
    if n == 0:
        return {"n_nonzero": 0, "n_policy_better": 0, "p_value": None}
    tail = min(k, n - k)
    p = sum(comb(n, i) for i in range(0, tail + 1)) / (2 ** n) * 2
    return {
        "n_nonzero": n,
        "n_policy_better": k,
        "n_baseline_better": n - k,
        "p_value": float(min(1.0, p)),
    }


def boot_indices(n: int) -> np.ndarray:
    rng = np.random.default_rng(BOOT_SEED)
    return rng.integers(0, n, size=(N_BOOT, n))


def ci(samples: np.ndarray) -> tuple:
    ok = samples[np.isfinite(samples)]
    if ok.size == 0:
        return (None, None)
    return (float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5)))


def loo(values_fn, n: int, point: float, seeds: list[int]) -> dict:
    """Largest absolute change in an estimator from dropping one instance."""
    best_d, best_i = 0.0, None
    for i in range(n):
        keep = np.arange(n) != i
        d = abs(values_fn(keep) - point)
        if d > best_d:
            best_d, best_i = d, i
    return {
        "max_abs_change": float(best_d),
        "seed": (seeds[best_i] if best_i is not None else None),
        "value_without": (
            float(values_fn(np.arange(n) != best_i)) if best_i is not None else None
        ),
    }


def analyse_pair(policy: np.ndarray, base: np.ndarray, seeds: list[int],
                 idx: np.ndarray) -> dict:
    n = policy.size
    ratio_of_means = float(policy.mean() / base.mean())
    per_inst = policy / base
    mean_of_ratios = float(per_inst.mean())
    diffs = policy - base

    bs_rom = policy[idx].mean(axis=1) / base[idx].mean(axis=1)
    bs_mor = per_inst[idx].mean(axis=1)
    bs_diff = diffs[idx].mean(axis=1)

    return {
        "n": int(n),
        "mean_policy_cost": float(policy.mean()),
        "mean_baseline_cost": float(base.mean()),
        "ratio_of_means": ratio_of_means,
        "ratio_of_means_ci95": ci(bs_rom),
        "mean_of_ratios": mean_of_ratios,
        "mean_of_ratios_ci95": ci(bs_mor),
        "mean_paired_difference": float(diffs.mean()),
        "mean_paired_difference_ci95": ci(bs_diff),
        "median_paired_difference": float(np.median(diffs)),
        "sign_test": sign_test(diffs),
        "loo_ratio_of_means": loo(
            lambda k: policy[k].mean() / base[k].mean(), n, ratio_of_means, seeds),
        "loo_mean_of_ratios": loo(
            lambda k: per_inst[k].mean(), n, mean_of_ratios, seeds),
    }


def analyse_closure(policy, greedy, ref, seeds, idx, name) -> dict:
    """Gap closure against a reference, under both estimators."""
    rom = float((greedy.mean() - policy.mean()) / (greedy.mean() - ref.mean()))
    per_inst = (greedy - policy) / (greedy - ref)
    mor = float(per_inst.mean())
    bs_rom = ((greedy[idx].mean(axis=1) - policy[idx].mean(axis=1))
              / (greedy[idx].mean(axis=1) - ref[idx].mean(axis=1)))
    bs_mor = per_inst[idx].mean(axis=1)
    n = policy.size
    return {
        "reference": name,
        "n": int(n),
        "mean_reference_cost": float(ref.mean()),
        "ratio_of_means": rom,
        "ratio_of_means_ci95": ci(bs_rom),
        "mean_of_ratios": mor,
        "mean_of_ratios_ci95": ci(bs_mor),
        "loo_ratio_of_means": loo(
            lambda k: (greedy[k].mean() - policy[k].mean())
            / (greedy[k].mean() - ref[k].mean()), n, rom, seeds),
        "loo_mean_of_ratios": loo(lambda k: per_inst[k].mean(), n, mor, seeds),
    }


def analyse_scale(scale: dict, rows: list[dict]) -> dict:
    seeds_all = [int(r["seed"]) for r in rows]

    excluded = []
    keep = []
    for r in rows:
        why = []
        if r["serve_all_flag"] != "1":
            why.append(f"policy left {r['n_unserved_tasks']} unserved")
        if r["greedy_served_all"] != "1":
            why.append(f"distance-only Hungarian left {r['greedy_n_unserved']} unserved")
        if r["kappa_served_all"] != "1":
            why.append(f"kappa Hungarian left {r['kappa_n_unserved']} unserved")
        if r["policy_cost"] == "":
            why.append("no policy cost recorded")
        if why:
            excluded.append({"seed": int(r["seed"]), "reasons": why})
        else:
            keep.append(r)

    seeds = [int(r["seed"]) for r in keep]
    policy = np.array([_f(r["policy_cost"]) for r in keep])
    greedy = np.array([_f(r["hungarian_distance_only_cost"]) for r in keep])
    kappa = np.array([_f(r["hungarian_kappa_cost"]) for r in keep])
    idx = boot_indices(len(keep))

    out = {
        "scale": scale,
        "n_instances_total": len(rows),
        "n_instances_in_statistics": len(keep),
        "seed_range": [min(seeds_all), max(seeds_all)],
        "excluded_from_statistics": excluded,
        "vs_baselines": {},
        "gap_closure": None,
        "milp_sanity_violations": [],
    }
    arrays = {"hungarian_distance_only_cost": greedy,
              "hungarian_kappa_cost": kappa}
    for col, label in BASELINES:
        out["vs_baselines"][label] = analyse_pair(policy, arrays[col], seeds, idx)

    # Not like for like: a rollout that dropped tasks is costed over fewer served tasks and so is flattered, but this shows what the exclusion is worth.
    if excluded:
        seeds_a = [int(r["seed"]) for r in rows]
        pol_a = np.array([_f(r["policy_cost"]) for r in rows])
        arr_a = {c: np.array([_f(r[c]) for r in rows]) for c, _ in BASELINES}
        idx_a = boot_indices(len(rows))
        out["vs_baselines_all_instances"] = {
            label: analyse_pair(pol_a, arr_a[col], seeds_a, idx_a)
            for col, label in BASELINES
        }

    if scale["has_milp_benchmark"]:
        milp = np.array([_f(r["milp_objective"]) for r in keep])
        expert = np.array([_f(r["expert_replay_cost"]) for r in keep])
        # The MILP objective is a lower bound, so a value above expert replay or greedy is a bad optimality certificate, not a hard instance; dropped from the MILP closure only.
        bad = (milp > expert) | (milp > greedy)
        out["milp_sanity_violations"] = [
            {"seed": seeds[i], "milp": float(milp[i]),
             "expert_replay": float(expert[i]), "greedy": float(greedy[i])}
            for i in np.nonzero(bad)[0]
        ]
        good = ~bad
        idx_good = boot_indices(int(good.sum()))
        seeds_good = [s for s, g in zip(seeds, good) if g]
        out["gap_closure"] = {
            "vs_milp": analyse_closure(
                policy[good], greedy[good], milp[good], seeds_good, idx_good,
                "offline anticipative MILP (lower bound)"),
            "vs_expert_replay": analyse_closure(
                policy, greedy, expert, seeds, idx,
                "replayed expert trajectory (achievable ceiling)"),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    d = REPO_ROOT / args.dir
    inv = json.load((d / "phase0_inventory.json").open())

    results = []
    for scale in inv["scales"]:
        rows = load_csv(REPO_ROOT / scale["csv"])
        results.append(analyse_scale(scale, rows))

    out = d / "phase2_statistics.json"
    if out.exists():
        raise SystemExit(f"REFUSING to overwrite {out}")
    with out.open("w") as f:
        json.dump({"n_bootstrap": N_BOOT, "bootstrap_seed": BOOT_SEED,
                   "results": results}, f, indent=2)
    print(f"wrote {out}")

    for r in results:
        s = r["scale"]
        print("\n" + "=" * 78)
        print(f"{s['label']}   n={r['n_instances_in_statistics']}"
              f"/{r['n_instances_total']}  seeds {r['seed_range'][0]}-{r['seed_range'][1]}")
        print("=" * 78)
        for excl in r["excluded_from_statistics"]:
            print(f"  excluded seed {excl['seed']}: {'; '.join(excl['reasons'])}")
        for label, v in r["vs_baselines"].items():
            lo, hi = v["ratio_of_means_ci95"]
            lo2, hi2 = v["mean_of_ratios_ci95"]
            lo3, hi3 = v["mean_paired_difference_ci95"]
            st = v["sign_test"]
            print(f"\n  vs {label}")
            print(f"    mean policy {v['mean_policy_cost']:.4f}   "
                  f"mean baseline {v['mean_baseline_cost']:.4f}")
            print(f"    ratio of means   {v['ratio_of_means']:.4f}  "
                  f"[{lo:.4f}, {hi:.4f}]")
            print(f"    mean of ratios   {v['mean_of_ratios']:.4f}  "
                  f"[{lo2:.4f}, {hi2:.4f}]")
            print(f"    paired diff      {v['mean_paired_difference']:+.4f}  "
                  f"[{lo3:+.4f}, {hi3:+.4f}]   median "
                  f"{v['median_paired_difference']:+.4f}")
            print(f"    sign test        policy better on "
                  f"{st['n_policy_better']}/{st['n_nonzero']}, "
                  f"p={st['p_value']:.3e}")
            print(f"    LOO ratio-of-means max |change| "
                  f"{v['loo_ratio_of_means']['max_abs_change']:.5f} "
                  f"(seed {v['loo_ratio_of_means']['seed']})")
            print(f"    LOO mean-of-ratios max |change| "
                  f"{v['loo_mean_of_ratios']['max_abs_change']:.5f} "
                  f"(seed {v['loo_mean_of_ratios']['seed']})")
        if r["gap_closure"] is None:
            print("\n  no offline anticipative MILP benchmark stored: "
                  "gap closure undefined and not reported")
        else:
            for viol in r["milp_sanity_violations"]:
                print(f"\n  MILP sanity violation seed {viol['seed']}: "
                      f"claimed {viol['milp']:.3f} above expert replay "
                      f"{viol['expert_replay']:.3f} / greedy {viol['greedy']:.3f}")
            for key, g in r["gap_closure"].items():
                lo, hi = g["ratio_of_means_ci95"]
                lo2, hi2 = g["mean_of_ratios_ci95"]
                print(f"\n  gap closure vs {g['reference']} (n={g['n']})")
                print(f"    ratio of means   {g['ratio_of_means']:.4f}  "
                      f"[{lo:.4f}, {hi:.4f}]")
                print(f"    mean of ratios   {g['mean_of_ratios']:.4f}  "
                      f"[{lo2:.4f}, {hi2:.4f}]")
                print(f"    LOO ratio-of-means max |change| "
                      f"{g['loo_ratio_of_means']['max_abs_change']:.5f} "
                      f"(seed {g['loo_ratio_of_means']['seed']})")
                print(f"    LOO mean-of-ratios max |change| "
                      f"{g['loo_mean_of_ratios']['max_abs_change']:.5f} "
                      f"(seed {g['loo_mean_of_ratios']['seed']})")


if __name__ == "__main__":
    main()
