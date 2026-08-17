"""Re-solve 20 validation-split anticipative MILPs at a 300 second time limit.

Denominator-sensitivity check for the headline 52.80% gap closure result.
The production val cache (cache/training_set_il_v3/val) was generated with a
60 second Gurobi time limit per instance. This script re-solves 20 of those
200 instances at 5x the production time limit (300 seconds), same
formulation, same weights, same tiebreak coefficient (all three are fixed
inside anticipative.anticipative_milp.solve_anticipative, which this script
calls unmodified), to check whether the MILP denominator is stable under
more solve time. The production cache is never written to.

The 20 seeds are the three val records already known to be internally
impossible (seed 11044, 11055, 11169: cached objective exceeds the greedy
cost, see docs and MEMORY headline-result-audit-facts) plus 17 more sampled
evenly across the 11000-11199 val seed range.

Run from the repository root with the coaml conda environment:

    PYTHONPATH="$PWD/src" python scripts/experiments/milp_resolve_300s.py

Results are appended to results/milp_resolve_300s.csv one row per instance,
flushed to disk immediately, so the file is valid to read at any point during
the run. If the script is interrupted, rerunning it resumes: seeds already
present in the CSV are skipped and re-read from disk rather than re-solved.

The 300 second re-solve for the two corrupted seeds checked so far (11044,
11055) came back certified optimal at the same implausible objectives as the
60 second cache, which rules out insufficient solve time as the explanation.
Given that, this script does not attempt to regenerate corrected cache
records for the corrupted seeds; it only reports whether each one's 300s
re-solve is sane (objective at or below the greedy Hungarian cost). Building
a replacement cache from a longer solve of the same formulation would just
reproduce the same numbers. That is a separate diagnosis task, not this
script's job.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from anticipative.anticipative_milp import solve_anticipative
from baselines.bipartite_policies import run_greedy_policy
from evaluation.method_one_evaluator import (
    gap_closure,
    load_scorer_from_checkpoint,
)
from evaluation.method_two_evaluator import evaluate_one_instance
from instances.synthetic_generator import SyntheticInstance


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VAL_CACHE_DIR = REPO_ROOT / "cache" / "training_set_il_v3" / "val"
RESULTS_CSV = REPO_ROOT / "results" / "milp_resolve_300s.csv"
CHECKPOINT = REPO_ROOT / "checkpoints" / "sweep_G_v4warmstart_best.pt"

CORRUPTED_SEEDS = [11044, 11055, 11169]
TIME_LIMIT_300S = 300
MIP_GAP_TARGET = 0.01
EVAL_WEIGHTS = {"w_dist": 0.0637, "w_make": 0.2398, "w_bal": 0.6965}
CSV_FIELDS = [
    "seed", "obj_60s", "obj_300s", "dual_bound_300s",
    "gap_60s", "gap_300s", "status",
]
MOVEMENT_FLAG_PCT = 0.5


def select_seeds() -> list:
    even = sorted(
        {int(round(v)) for v in np.linspace(11000, 11199, 17)}
    )
    seeds = sorted(set(CORRUPTED_SEEDS) | set(even))
    return seeds


def load_cached_record(seed: int) -> dict:
    path = VAL_CACHE_DIR / f"seed{seed}.json"
    with path.open("r") as f:
        return json.load(f)


def resolve_one(record: dict):
    instance = SyntheticInstance.from_dict(record["instance"])
    weights = record["weights"]
    solution = solve_anticipative(
        instance, weights, time_limit_seconds=TIME_LIMIT_300S,
        mip_gap=MIP_GAP_TARGET,
    )
    return instance, solution


def append_csv_row(row: dict) -> None:
    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def load_finished_rows() -> list:
    if not RESULTS_CSV.exists():
        return []
    with RESULTS_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append({
                "seed": int(r["seed"]),
                "obj_60s": float(r["obj_60s"]),
                "obj_300s": float(r["obj_300s"]),
                "dual_bound_300s": float(r["dual_bound_300s"]),
                "gap_60s": float(r["gap_60s"]),
                "gap_300s": float(r["gap_300s"]),
                "status": r["status"],
            })
    return rows


def print_interim_summary(rows: list) -> None:
    movements = [
        (r["obj_60s"] - r["obj_300s"]) / r["obj_60s"] * 100.0
        for r in rows
    ]
    print(
        f"\n--- interim summary after {len(rows)} instances ---\n"
        f"  mean incumbent movement: {np.mean(movements):+.3f}%\n"
        f"  max incumbent movement:  {np.max(movements):+.3f}%\n"
        f"  all moves under 0.5%:    "
        f"{all(abs(m) < MOVEMENT_FLAG_PCT for m in movements)}\n"
        "--- end interim summary ---\n",
        flush=True,
    )


# Gitignored inputs, checked before any work so a fresh clone gets one diagnostic line rather than a FileNotFoundError from deep in the load.
REQUIRED_INPUTS = [
    ("cache/training_set_il_v3/val", "provenance/appendix_c_benchmark/milp_resolve_300s.csv"),
    ("checkpoints/sweep_G_v4warmstart_best.pt",
     "provenance/appendix_c_benchmark/milp_resolve_300s.csv"),
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
    seeds = select_seeds()
    existing_rows = load_finished_rows()
    existing_seeds = {r["seed"] for r in existing_rows}
    remaining = [s for s in seeds if s not in existing_seeds]

    print(f"Selected {len(seeds)} seeds: {seeds}", flush=True)
    if existing_rows:
        print(
            f"Resuming: {len(existing_rows)}/{len(seeds)} instances already "
            f"in {RESULTS_CSV.relative_to(REPO_ROOT)}.",
            flush=True,
        )
    if remaining:
        print(f"{len(remaining)} remaining: {remaining}", flush=True)
    else:
        print("All seeds already resolved. Skipping to summary.", flush=True)

    total_done = len(existing_rows)
    for seed in remaining:
        record = load_cached_record(seed)
        obj_60s = float(record["milp_solution"]["objective_value"])
        gap_60s = float(record["milp_solution"]["mip_gap"])

        instance, solution = resolve_one(record)

        obj_300s = float(solution.objective_value)
        gap_300s = float(solution.mip_gap)
        dual_bound_300s = float(solution.obj_bound)
        movement_pct = (obj_60s - obj_300s) / obj_60s * 100.0

        row = {
            "seed": seed,
            "obj_60s": obj_60s,
            "obj_300s": obj_300s,
            "dual_bound_300s": dual_bound_300s,
            "gap_60s": gap_60s,
            "gap_300s": gap_300s,
            "status": solution.status,
        }
        append_csv_row(row)
        total_done += 1

        print(
            f"[{total_done}/{len(seeds)}] seed={seed} "
            f"obj_60s={obj_60s:.3f} obj_300s={obj_300s:.3f} "
            f"move={movement_pct:+.3f}% "
            f"gap_60s={gap_60s:.3f} gap_300s={gap_300s:.3f} "
            f"status={solution.status}",
            flush=True,
        )

        if total_done == 10:
            print_interim_summary(load_finished_rows())

    rows = load_finished_rows()
    assert len(rows) == len(seeds), (
        f"Expected {len(seeds)} finished rows, found {len(rows)}."
    )

    movements = [
        (r["obj_60s"] - r["obj_300s"]) / r["obj_60s"] * 100.0 for r in rows
    ]
    dual_bound_moves = []
    for r in rows:
        record = load_cached_record(r["seed"])
        old_bound = float(record["milp_solution"]["obj_bound"])
        if abs(old_bound) > 1e-9:
            dual_bound_moves.append(
                (r["dual_bound_300s"] - old_bound) / old_bound * 100.0
            )

    n_moved = sum(1 for m in movements if abs(m) > MOVEMENT_FLAG_PCT)

    print("\n=== Final summary ===")
    print(f"Instances resolved: {len(rows)}")
    print(f"Mean incumbent improvement: {np.mean(movements):+.3f}%")
    print(f"Max incumbent improvement:  {np.max(movements):+.3f}%")
    if dual_bound_moves:
        print(
            f"Mean dual bound improvement: {np.mean(dual_bound_moves):+.3f}% "
            f"(n={len(dual_bound_moves)})"
        )
    else:
        print("Mean dual bound improvement: n/a (all old bounds ~0)")
    print(
        f"Instances with incumbent movement > {MOVEMENT_FLAG_PCT}%: "
        f"{n_moved}/{len(rows)}"
    )

    print("\nLoading checkpoint for learned policy gap closure "
          f"({CHECKPOINT.relative_to(REPO_ROOT)})...", flush=True)
    scorer = load_scorer_from_checkpoint(str(CHECKPOINT))

    gaps_60s, gaps_300s = [], []
    for r in rows:
        record = load_cached_record(r["seed"])
        eval_record = evaluate_one_instance(
            scorer, record, modes=["hard"], weights=EVAL_WEIGHTS, epsilon=0.0
        )
        greedy = eval_record["greedy"]
        learned_cost = eval_record["hard_cost"]
        if learned_cost is None:
            continue
        g60 = gap_closure(greedy, learned_cost, r["obj_60s"])
        g300 = gap_closure(greedy, learned_cost, r["obj_300s"])
        if g60 is not None:
            gaps_60s.append(g60)
        if g300 is not None:
            gaps_300s.append(g300)

    def _fmt_gap(vals):
        if not vals:
            return "n/a (no instance with a positive gap)"
        return f"{np.mean(vals) * 100:.2f}% (n={len(vals)})"

    print(
        f"Learned policy gap closure, 60s incumbents as denominator:  "
        f"{_fmt_gap(gaps_60s)}"
    )
    print(
        f"Learned policy gap closure, 300s incumbents as denominator: "
        f"{_fmt_gap(gaps_300s)}"
    )

    print("\n=== Corrupted seeds (11044, 11055, 11169) ===")
    for seed in CORRUPTED_SEEDS:
        record = load_cached_record(seed)
        instance = SyntheticInstance.from_dict(record["instance"])
        row = next(r for r in rows if r["seed"] == seed)
        greedy_cost = float(
            run_greedy_policy(instance).compute_cost(record["weights"]).combined
        )
        obj_300s = row["obj_300s"]
        sane = obj_300s <= greedy_cost + 1e-6
        print(
            f"seed={seed} greedy={greedy_cost:.3f} obj_300s={obj_300s:.3f} "
            f"status={row['status']} sane={sane}"
        )


if __name__ == "__main__":
    main()
