"""DM-27 Phase 2: baselines on the density-matched R=10, T=30, W=65 held-out 40.

Evaluates four baselines on the same 40 held-out instances (seeds
92160-92199 in cache/dm27_r10t30_d65), so every comparison in Phase 4 is
exact (identical instances, identical simulator, identical cost weights):

    1. distance-only Hungarian (baselines.bipartite_policies.run_greedy_policy)
    2. Hungarian-on-kappa (run_hungarian_kappa_policy: distance + makespan +
       balance terms per epoch)
    3. the R=6, T=18 warm-start checkpoint
       (checkpoints/sweep_G_v4warmstart_best.pt), zero-shot, no fine-tuning,
       hard decode (epsilon=0)
    4. the online h=1 rolling-horizon MILP (rolling_horizon_baseline.py),
       ONE budget only: TimeLimit=1s per epoch solve, MIPGap=0.01, gurobi
       seed 0. Per-epoch solve statistics are captured by monkeypatching
       gurobipy.Model (to read back Runtime/Status per solve) and
       rolling_horizon_baseline.solve_rolling_window (to read back window
       size), following the exact instrumentation pattern in
       scripts/rolling_horizon_h1_r10t60_probe.py. The wrapped calls run the
       original functions unmodified.

No MILP reference and no gap closure at this scale (matching DM-25/DM-26's
convention at R=10, T=60): this is cost and serve-all-rate only.

Per-instance rows append to results/dm27_r10t30_baselines.csv immediately;
seeds already present are skipped on rerun, so this is safely resumable.
Nothing in cache/, checkpoints/, or any other results/ file is written.

Run:
    PYTHONPATH="$PWD/src" python scripts/experiments/dm27_r10t30_baselines.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import gurobipy
from gurobipy import GRB

import rolling_horizon_baseline as rhb
from baselines.bipartite_policies import run_greedy_policy, run_hungarian_kappa_policy
from evaluation.method_one_evaluator import rollout_failed, rollout_policy
from instances.synthetic_generator import SyntheticInstance
from scoring.gnn_scorer import GNNScorer


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "cache" / "dm27_r10t30_d65"
WARM_START = REPO_ROOT / "checkpoints" / "sweep_G_v4warmstart_best.pt"
OUT_CSV = REPO_ROOT / "results" / "dm27_r10t30_baselines.csv"

HELD_OUT_SEEDS = list(range(92160, 92200))
WEIGHTS = {"w_dist": 0.0637, "w_make": 0.2398, "w_bal": 0.6965}

RH_HORIZON = 1
RH_TIME_LIMIT = 1  # seconds per epoch solve -- one budget only, per instructions
RH_GUROBI_SEED = 0

CSV_FIELDS = [
    "seed",
    "hungarian_cost", "hungarian_served",
    "kappa_cost", "kappa_served",
    "zeroshot_cost", "zeroshot_served",
    "rh1_cost", "rh1_served", "rh1_n_solves", "rh1_n_fallbacks",
    "rh1_n_time_limit", "rh1_mean_window", "rh1_max_window",
    "rh1_mean_solve_s", "rh1_max_solve_s",
]

# Instrumentation state, following rolling_horizon_h1_r10t60_probe.py exactly.
_CAPTURED = []
_ORIG_MODEL = gurobipy.Model
_ORIG_SOLVE = rhb.solve_rolling_window
_WINDOW_SIZES = []


def _capturing_model(*args, **kwargs):
    m = _ORIG_MODEL(*args, **kwargs)
    _CAPTURED.append(m)
    return m


def _wrapped_solve(instance, window_task_ids, *args, **kwargs):
    _WINDOW_SIZES.append(len(window_task_ids))
    return _ORIG_SOLVE(instance, window_task_ids, *args, **kwargs)


def _status_name(code: int) -> str:
    return {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.SUBOPTIMAL: "suboptimal",
        GRB.INTERRUPTED: "interrupted",
        GRB.INFEASIBLE: "infeasible",
    }.get(code, f"status_{code}")


def load_zero_shot_scorer() -> GNNScorer:
    scorer = GNNScorer()
    ckpt = torch.load(str(WARM_START), map_location="cpu", weights_only=False)
    try:
        scorer.load_state_dict(ckpt["model_state_dict"], strict=True)
    except RuntimeError as e:
        print(
            f"FAILED: {WARM_START} state dict does not load into a fresh "
            f"GNNScorer() with no shape mismatch:\n{e}",
            flush=True,
        )
        sys.exit(1)
    scorer.eval()
    return scorer


def existing_seeds() -> set:
    if not OUT_CSV.exists():
        return set()
    with OUT_CSV.open() as f:
        return {int(r["seed"]) for r in csv.DictReader(f)}


def append_row(row: dict) -> None:
    write_header = not OUT_CSV.exists()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


# Gitignored inputs paired with the committed artefact carrying the result; checked upfront so a fresh clone gets one diagnostic line, not a FileNotFoundError deep in the load.
REQUIRED_INPUTS = [
    ("cache/dm27_r10t30_d65", "provenance/table41_main_results/b3_methodone_test_20260731.csv"),
    ("checkpoints/sweep_G_v4warmstart_best.pt",
     "provenance/table41_main_results/b3_methodone_test_20260731.csv"),
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
    require_inputs()
    seeds = HELD_OUT_SEEDS
    print(
        f"DM-27 Phase 2 baselines: {len(seeds)} held-out instances "
        f"(seeds {seeds[0]}-{seeds[-1]}) from "
        f"{CACHE_DIR.relative_to(REPO_ROOT)}.",
        flush=True,
    )
    print(
        f"h=1 rolling-horizon budget: TimeLimit={RH_TIME_LIMIT}s/solve, "
        f"MIPGap=0.01, gurobi_seed={RH_GUROBI_SEED} (single budget only).",
        flush=True,
    )
    print(f"weights={WEIGHTS}", flush=True)

    scorer = load_zero_shot_scorer()
    print(f"Zero-shot scorer loaded from {WARM_START.relative_to(REPO_ROOT)}.",
          flush=True)

    done = existing_seeds()
    if done:
        print(f"Resuming: {len(done)} seeds already in "
              f"{OUT_CSV.relative_to(REPO_ROOT)}.", flush=True)

    gurobipy.Model = _capturing_model
    rhb.solve_rolling_window = _wrapped_solve
    t0 = time.perf_counter()
    try:
        for i, seed in enumerate(seeds, 1):
            if seed in done:
                continue
            rec = json.load((CACHE_DIR / f"seed{seed}.json").open())
            instance = SyntheticInstance.from_dict(rec["instance"])

            t_inst = time.perf_counter()

            sim_h = run_greedy_policy(instance)
            hung_served = rhb._served_all(sim_h)
            hung_cost = (
                float(sim_h.compute_cost(WEIGHTS).combined) if hung_served else None
            )

            sim_k = run_hungarian_kappa_policy(instance, WEIGHTS)
            kappa_served = rhb._served_all(sim_k)
            kappa_cost = (
                float(sim_k.compute_cost(WEIGHTS).combined) if kappa_served else None
            )

            sim_z = rollout_policy(scorer, instance, WEIGHTS, epsilon=0.0)
            zs_served = not rollout_failed(sim_z)
            zs_cost = (
                float(sim_z.compute_cost(WEIGHTS).combined) if zs_served else None
            )

            _CAPTURED.clear()
            _WINDOW_SIZES.clear()
            sim_r, n_solves, n_fallbacks = rhb.run_rolling_horizon_policy(
                instance, WEIGHTS, RH_HORIZON,
                time_limit_seconds=RH_TIME_LIMIT, seed=RH_GUROBI_SEED,
            )
            runtimes, statuses = [], []
            for m in _CAPTURED:
                try:
                    runtimes.append(float(m.Runtime))
                    statuses.append(_status_name(int(m.Status)))
                except Exception as e:  # noqa: BLE001
                    runtimes.append(float("nan"))
                    statuses.append(f"read_error:{e}")
            for m in list(_CAPTURED):
                try:
                    m.dispose()
                except Exception:  # noqa: BLE001
                    pass
            _CAPTURED.clear()

            rh_served = rhb._served_all(sim_r)
            rh_cost = (
                float(sim_r.compute_cost(WEIGHTS).combined) if rh_served else None
            )
            n_tl = statuses.count("time_limit")

            wall = time.perf_counter() - t_inst

            append_row({
                "seed": seed,
                "hungarian_cost": "" if hung_cost is None else f"{hung_cost:.6f}",
                "hungarian_served": int(hung_served),
                "kappa_cost": "" if kappa_cost is None else f"{kappa_cost:.6f}",
                "kappa_served": int(kappa_served),
                "zeroshot_cost": "" if zs_cost is None else f"{zs_cost:.6f}",
                "zeroshot_served": int(zs_served),
                "rh1_cost": "" if rh_cost is None else f"{rh_cost:.6f}",
                "rh1_served": int(rh_served),
                "rh1_n_solves": n_solves,
                "rh1_n_fallbacks": n_fallbacks,
                "rh1_n_time_limit": n_tl,
                "rh1_mean_window": f"{np.mean(_WINDOW_SIZES):.3f}" if _WINDOW_SIZES else "",
                "rh1_max_window": int(np.max(_WINDOW_SIZES)) if _WINDOW_SIZES else 0,
                "rh1_mean_solve_s": f"{np.nanmean(runtimes):.4f}" if runtimes else "",
                "rh1_max_solve_s": f"{np.nanmax(runtimes):.4f}" if runtimes else "",
            })

            hc = "FAIL" if hung_cost is None else f"{hung_cost:.2f}"
            kc = "FAIL" if kappa_cost is None else f"{kappa_cost:.2f}"
            zc = "FAIL" if zs_cost is None else f"{zs_cost:.2f}"
            rc = "FAIL" if rh_cost is None else f"{rh_cost:.2f}"
            print(
                f"[{i}/{len(seeds)}] seed={seed} hungarian={hc} kappa={kc} "
                f"zeroshot={zc} rh1={rc} solves={n_solves} "
                f"fallbacks={n_fallbacks} tl={n_tl} "
                f"win_max={int(np.max(_WINDOW_SIZES)) if _WINDOW_SIZES else 0} "
                f"wall={wall:.1f}s cum={time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    finally:
        gurobipy.Model = _ORIG_MODEL
        rhb.solve_rolling_window = _ORIG_SOLVE

    report(seeds)


def report(seeds: list) -> None:
    with OUT_CSV.open() as f:
        rows = [r for r in csv.DictReader(f) if int(r["seed"]) in set(seeds)]
    n = len(rows)

    def _summ(cost_key, served_key):
        served = [r for r in rows if int(r[served_key])]
        costs = [float(r[cost_key]) for r in served if r[cost_key]]
        rate = len(served) / n if n else 0.0
        mean = float(np.mean(costs)) if costs else float("nan")
        return mean, rate, len(served)

    hung_mean, hung_rate, hung_n = _summ("hungarian_cost", "hungarian_served")
    kappa_mean, kappa_rate, kappa_n = _summ("kappa_cost", "kappa_served")
    zs_mean, zs_rate, zs_n = _summ("zeroshot_cost", "zeroshot_served")
    rh_mean, rh_rate, rh_n = _summ("rh1_cost", "rh1_served")

    n_solves_total = sum(int(r["rh1_n_solves"]) for r in rows)
    n_tl_total = sum(int(r["rh1_n_time_limit"]) for r in rows)
    n_fb_total = sum(int(r["rh1_n_fallbacks"]) for r in rows)
    weighted_win_mean = (
        sum(float(r["rh1_mean_window"]) * int(r["rh1_n_solves"])
            for r in rows if r["rh1_mean_window"]) / n_solves_total
        if n_solves_total else float("nan")
    )
    weighted_solve_mean = (
        sum(float(r["rh1_mean_solve_s"]) * int(r["rh1_n_solves"])
            for r in rows if r["rh1_mean_solve_s"]) / n_solves_total
        if n_solves_total else float("nan")
    )
    max_window = max((int(r["rh1_max_window"]) for r in rows), default=0)
    max_solve = max(
        (float(r["rh1_max_solve_s"]) for r in rows if r["rh1_max_solve_s"]),
        default=float("nan"),
    )

    print("\n" + "=" * 70)
    print(f"DM-27 Phase 2 baseline summary, R=10 T=30 W=65, n={n}")
    print("=" * 70)
    print(f"  {'baseline':<24} {'mean cost':>10} {'serve-all':>12}")
    print(f"  {'-'*24} {'-'*10} {'-'*12}")
    print(f"  {'Hungarian (dist-only)':<24} {hung_mean:>10.4f} "
          f"{hung_n:>4}/{n} ({hung_rate*100:>5.1f}%)")
    print(f"  {'Hungarian-on-kappa':<24} {kappa_mean:>10.4f} "
          f"{kappa_n:>4}/{n} ({kappa_rate*100:>5.1f}%)")
    print(f"  {'Zero-shot warm start':<24} {zs_mean:>10.4f} "
          f"{zs_n:>4}/{n} ({zs_rate*100:>5.1f}%)")
    print(f"  {'h=1 rolling MILP (1s)':<24} {rh_mean:>10.4f} "
          f"{rh_n:>4}/{n} ({rh_rate*100:>5.1f}%)")
    print()
    print(f"  h=1 per-epoch solve stats ({n_solves_total} solves total):")
    print(f"    window size              mean {weighted_win_mean:.2f}  "
          f"max {max_window}")
    print(f"    solve time per epoch     mean {weighted_solve_mean:.3f}s  "
          f"max {max_solve:.2f}s")
    print(f"    epochs at {RH_TIME_LIMIT}s limit       {n_tl_total}/{n_solves_total} "
          f"({n_tl_total / n_solves_total * 100:.1f}%)" if n_solves_total else "n/a")
    print(f"    epochs falling back      {n_fb_total}/{n_solves_total} "
          f"({n_fb_total / n_solves_total * 100:.1f}%)" if n_solves_total else "n/a")


if __name__ == "__main__":
    main()
