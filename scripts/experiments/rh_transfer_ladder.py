"""Four-budget h=1 rolling-horizon ladder at the two ratio-matched transfer scales.

Budgets 0.050, 0.100, 0.250 and 1.0 seconds per window solve, ascending, over
60 instances at R=10, T=30 (cache/dm31a_r10t30, seeds 96000-96059) and 60 at
R=30, T=90 (cache/dm31b_r30t90, seeds 96100-96159).

h=1 admits no unreleased task, so this is the genuinely online, non-clairvoyant
rolling horizon rather than a lookahead oracle.

Two things this run records that the budget label alone hides. First, the model
build sits OUTSIDE the Gurobi time limit, so ``mean_build_s`` and the derived
``mean_total_per_decision_s`` are reported next to the budget and are the honest
per-decision cost. Second, when the solver returns no incumbent inside the limit
the window is decided by Hungarian-on-kappa instead, so ``n_no_incumbent`` and
``n_fallbacks`` say how much of each episode the rolling horizon actually
decided.

Read-only with respect to every cache and checkpoint. Writes only under
provenance/, refuses to overwrite, flushes and fsyncs after every instance, and
resumes by skipping (budget, scale, seed) triples already present.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
for _d in ("regenerate", "analysis", "experiments"):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _d))

from instances.synthetic_generator import SyntheticInstance  # noqa: E402
import dm32_hardened_rh as hrh  # noqa: E402
from gurobipy import GRB  # noqa: E402

WEIGHTS = {"w_dist": 0.0637, "w_make": 0.2398, "w_bal": 0.6965}
HORIZON = 1
GUROBI_SEED = 0
MIP_GAP = 0.01  # applied by the solver default, recorded here for provenance
BUDGETS = [0.050, 0.100, 0.250, 1.0]
SCALES = [
    {"key": "r10t30", "label": "R=10,T=30", "R": 10, "T": 30,
     "cache": "cache/dm31a_r10t30", "seeds": list(range(96000, 96060))},
    {"key": "r30t90", "label": "R=30,T=90", "R": 30, "T": 90,
     "cache": "cache/dm31b_r30t90", "seeds": list(range(96100, 96160))},
]
FORBIDDEN = range(11200, 11400)

STATUS = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit",
          GRB.SUBOPTIMAL: "suboptimal", GRB.INTERRUPTED: "interrupted",
          GRB.INFEASIBLE: "infeasible"}

INST_FIELDS = [
    "scale", "R", "T", "budget_seconds", "seed", "cost", "served_all",
    "frac_served", "n_unserved", "n_window_solves", "n_models_built",
    "n_time_limit", "n_no_incumbent", "n_fallbacks",
    "mean_solve_s", "max_solve_s", "mean_build_s", "max_build_s",
    "mean_total_per_decision_s", "mean_window", "max_window",
    "total_solver_seconds", "wall_seconds",
]
SOLVE_FIELDS = [
    "scale", "budget_seconds", "seed", "k", "window_size",
    "build_s", "gurobi_runtime_s", "solve_wall_s", "status", "sol_count",
]


def append(path: Path, fields, row) -> None:
    header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if header:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open() as f:
        return {(r["scale"], r["budget_seconds"], int(r["seed"]))
                for r in csv.DictReader(f)}


def run_one(scale: dict, seed: int, budget: float, out_solves: Path) -> dict:
    rec = json.load((REPO_ROOT / scale["cache"] / f"seed{seed}.json").open())
    if int(rec["R"]) != scale["R"] or int(rec["T"]) != scale["T"]:
        raise SystemExit(f"seed {seed}: cache is R={rec['R']} T={rec['T']}")
    instance = SyntheticInstance.from_dict(rec["instance"])

    timings: list = []
    t0 = time.perf_counter()
    sim, n_solves, n_fallbacks = hrh.run_rolling_horizon_policy_hardened(
        instance, WEIGHTS, HORIZON,
        # MIPGap isn't threaded through the policy; solve_rolling_window_hardened applies its own default of 0.01, the value DM-32 and DM-33 used.
        time_limit_seconds=budget, seed=GUROBI_SEED,
        log_seed=seed, timings=timings,
    )
    wall = time.perf_counter() - t0

    # An empty window returns before a model is built and carries no timing record, so solve statistics are over the models actually built, not n_solves.
    built = [t for t in timings if t and "gurobi_runtime_s" in t]
    rt = np.array([t["gurobi_runtime_s"] for t in built]) if built else np.array([])
    bd = np.array([t["build_s"] for t in built]) if built else np.array([])
    sw = np.array([t["solve_wall_s"] for t in built]) if built else np.array([])
    ws = np.array([t["window_size"] for t in built]) if built else np.array([])

    for k, t in enumerate(built):
        append(out_solves, SOLVE_FIELDS, {
            "scale": scale["label"], "budget_seconds": budget, "seed": seed,
            "k": k, "window_size": t["window_size"],
            "build_s": repr(t["build_s"]),
            "gurobi_runtime_s": repr(t["gurobi_runtime_s"]),
            "solve_wall_s": repr(t["solve_wall_s"]),
            "status": STATUS.get(t["status"], t["status"]),
            "sol_count": t["sol_count"],
        })

    n_serviceable = len(sim._serviceable)
    n_unserved = len(sim.trajectory.unserved_task_ids)
    served_all = not (sim.trajectory.simulator_failure
                      or sim.state.pending_tasks
                      or sim.trajectory.unserved_task_ids)
    return {
        "scale": scale["label"], "R": scale["R"], "T": scale["T"],
        "budget_seconds": budget, "seed": seed,
        "cost": repr(float(sim.compute_cost(WEIGHTS).combined)),
        "served_all": int(served_all),
        "frac_served": repr((n_serviceable - n_unserved) / n_serviceable
                            if n_serviceable else 1.0),
        "n_unserved": n_unserved,
        "n_window_solves": n_solves,
        "n_models_built": len(built),
        "n_time_limit": sum(1 for t in built if t["status"] == GRB.TIME_LIMIT),
        "n_no_incumbent": sum(1 for t in built if t["sol_count"] == 0),
        "n_fallbacks": n_fallbacks,
        "mean_solve_s": repr(float(rt.mean())) if rt.size else "",
        "max_solve_s": repr(float(rt.max())) if rt.size else "",
        "mean_build_s": repr(float(bd.mean())) if bd.size else "",
        "max_build_s": repr(float(bd.max())) if bd.size else "",
        "mean_total_per_decision_s": repr(float((bd + sw).mean())) if bd.size else "",
        "mean_window": repr(float(ws.mean())) if ws.size else "",
        "max_window": int(ws.max()) if ws.size else 0,
        "total_solver_seconds": repr(float((bd + sw).sum())) if bd.size else "",
        "wall_seconds": repr(wall),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_inst = out_dir / "rh_ladder_per_instance.csv"
    out_solves = out_dir / "rh_ladder_per_solve.csv"

    for s in SCALES:
        bad = [x for x in s["seeds"] if x in FORBIDDEN]
        if bad:
            raise SystemExit(f"REFUSING: {s['key']} touches held-out seeds {bad[:5]}")
        if any(p.lower() == "test" for p in Path(s["cache"]).parts):
            raise SystemExit(f"REFUSING a test split: {s['cache']}")

    already = done_keys(out_inst)
    if already:
        print(f"resuming, {len(already)} instance-budget pairs already done",
              flush=True)

    total = len(BUDGETS) * sum(len(s["seeds"]) for s in SCALES)
    n = 0
    t_start = time.perf_counter()
    for scale in SCALES:
        for budget in BUDGETS:
            costs, walls = [], []
            for seed in scale["seeds"]:
                n += 1
                if (scale["label"], str(budget), seed) in already:
                    continue
                row = run_one(scale, seed, budget, out_solves)
                append(out_inst, INST_FIELDS, row)
                costs.append(float(row["cost"]))
                walls.append(float(row["wall_seconds"]))
                el = time.perf_counter() - t_start
                print(
                    f"[{n}/{total}] {scale['label']} b={budget}s seed={seed} "
                    f"cost={float(row['cost']):.3f} served={row['served_all']} "
                    f"solves={row['n_window_solves']} "
                    f"no_inc={row['n_no_incumbent']} fb={row['n_fallbacks']} "
                    f"wall={float(row['wall_seconds']):.1f}s "
                    f"elapsed={el / 60:.1f}m",
                    flush=True,
                )
            if costs:
                print(f"  == {scale['label']} b={budget}s: mean cost "
                      f"{np.mean(costs):.4f}, mean wall {np.mean(walls):.2f}s, "
                      f"scale-budget wall {np.sum(walls):.0f}s\n", flush=True)

    print(f"\ntotal wall {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)
    print(f"wrote {out_inst}\nwrote {out_solves}", flush=True)


if __name__ == "__main__":
    main()
