"""Recompute the learned policy's gap closure under all three MILP denominators.

Gap closure is ``(greedy - learned) / (greedy - milp)``. A longer solve finds a
better (lower) incumbent, enlarging the denominator and shrinking the reported
closure; the headline result uses the 60 second production cache as that
denominator. This script reports what the same twenty instances give under the
60 s, 300 s and 3600 s incumbents, so the sensitivity of the headline to solver
budget is measured rather than assumed.

The policy side is held completely fixed across the three columns: one
checkpoint, one deterministic hard decode per instance (``epsilon=0.0``,
``modes=["hard"]``), evaluated once and reused for all three denominators, so
every point of movement between the columns comes from the denominator alone.

Self check. ``scripts/experiments/milp_resolve_300s.py`` already reported
49.27% under the 60 s denominator and 48.61% under the 300 s denominator over
17 instances. This script recomputes both from scratch and aborts if either
disagrees, which validates the 3600 s column rather than leaving it unanchored.

The three corrupted val records (11044, 11055, 11169) have a cached objective
above the greedy cost, so their denominator is non-positive and ``gap_closure``
returns None for them at every budget. They drop out of all three columns
identically, leaving n=17, so their exclusion is not a source of difference
between the columns.

Reads only. The label cache, the three budget CSVs and the checkpoint are
never written to.

Run from the repository root in the coaml environment:

    PYTHONPATH="$PWD/src" python scripts/regenerate/gap_closure_vs_budget.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

from evaluation.method_one_evaluator import (
    gap_closure,
    load_scorer_from_checkpoint,
)
from evaluation.method_two_evaluator import evaluate_one_instance


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VAL_CACHE_DIR = REPO_ROOT / "cache" / "training_set_il_v3" / "val"
CSV_300S = REPO_ROOT / "results" / "milp_resolve_300s.csv"
CSV_3600S = REPO_ROOT / "results" / "milp_resolve_3600s.csv"
CHECKPOINT = REPO_ROOT / "checkpoints" / "sweep_G_v4warmstart_best.pt"

OUT_PER_INSTANCE = REPO_ROOT / "provenance" / "gap_closure_vs_budget.csv"
OUT_SUMMARY = REPO_ROOT / "provenance" / "gap_closure_vs_budget_summary.csv"

# Identical to scripts/experiments/milp_resolve_300s.py, which is what makes the 60 s and
# 300 s columns comparable with the already-published numbers.
EVAL_WEIGHTS = {"w_dist": 0.0637, "w_make": 0.2398, "w_bal": 0.6965}

# Reference values recorded by scripts/experiments/milp_resolve_300s.py. The recomputation
# below must land on these or the evaluation path has drifted.
REFERENCE_GC_60S_PCT = 49.27
REFERENCE_GC_300S_PCT = 48.61
REFERENCE_N = 17
REFERENCE_TOL_PCT = 0.01

BUDGETS = ["60s", "300s", "3600s"]


def sha256_sidecar(path: Path) -> Path:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    side = path.with_name(path.name + ".sha256")
    side.write_text(f"{digest}  {path.name}\n")
    return side


def load_objectives() -> tuple:
    """Return the ordered seed list and {seed: {budget: milp objective}}."""
    for p in (CSV_300S, CSV_3600S, CHECKPOINT, VAL_CACHE_DIR):
        if not p.exists():
            raise SystemExit(
                f"Missing {p.relative_to(REPO_ROOT)}. It is gitignored and "
                f"absent from a fresh clone, so this script can only run on a "
                f"tree that carries it. The committed "
                f"provenance/appendix_c_benchmark/gap_closure_vs_budget.csv carries the result."
            )

    with CSV_300S.open("r", newline="") as f:
        rows_300 = list(csv.DictReader(f))
    with CSV_3600S.open("r", newline="") as f:
        rows_3600 = {int(r["seed"]): r for r in csv.DictReader(f)}

    order = [int(r["seed"]) for r in rows_300]
    missing = [s for s in order if s not in rows_3600]
    if missing:
        raise SystemExit(
            f"The 3600 s run is incomplete, missing {missing}. Finish it "
            f"before recomputing the denominator."
        )

    objectives = {}
    for r300 in rows_300:
        seed = int(r300["seed"])
        objectives[seed] = {
            "60s": float(r300["obj_60s"]),
            "300s": float(r300["obj_300s"]),
            "3600s": float(rows_3600[seed]["obj_val"]),
        }
    return order, objectives


def load_cached_record(seed: int) -> dict:
    with (VAL_CACHE_DIR / f"seed{seed}.json").open("r") as f:
        return json.load(f)


def evaluate(order: list, objectives: dict) -> list:
    """One deterministic policy rollout per instance, three denominators."""
    print(
        f"Loading {CHECKPOINT.relative_to(REPO_ROOT)} for the policy side "
        f"(held fixed across all three columns)...",
        flush=True,
    )
    scorer = load_scorer_from_checkpoint(str(CHECKPOINT))

    per_instance = []
    for i, seed in enumerate(order, start=1):
        record = load_cached_record(seed)
        eval_record = evaluate_one_instance(
            scorer, record, modes=["hard"], weights=EVAL_WEIGHTS, epsilon=0.0
        )
        greedy = eval_record["greedy"]
        learned = eval_record["hard_cost"]
        if learned is None:
            print(f"  [{i}/{len(order)}] seed={seed} no hard decode, skipped",
                  flush=True)
            continue

        row = {
            "seed": seed,
            "greedy": greedy,
            "learned_hard": learned,
        }
        for b in BUDGETS:
            obj = objectives[seed][b]
            gc = gap_closure(greedy, learned, obj)
            row[f"obj_{b}"] = obj
            row[f"gap_closure_{b}"] = "" if gc is None else gc
        row["denominator_defined"] = row["gap_closure_60s"] != ""
        per_instance.append(row)

        gc60 = row["gap_closure_60s"]
        gc3600 = row["gap_closure_3600s"]
        if gc60 == "" or gc3600 == "":
            note = "denominator non-positive, excluded at every budget"
        else:
            note = (
                f"gc 60s={gc60 * 100:6.2f}%  3600s={gc3600 * 100:6.2f}%  "
                f"move={(gc3600 - gc60) * 100:+.2f} pts"
            )
        print(f"  [{i}/{len(order)}] seed={seed} {note}", flush=True)

    return per_instance


def summarise(per_instance: list) -> dict:
    usable = [r for r in per_instance if r["denominator_defined"]]
    out = {"n_instances_in_gap_closure": len(usable)}
    for b in BUDGETS:
        vals = [r[f"gap_closure_{b}"] for r in usable]
        out[f"mean_gap_closure_{b}_pct"] = st.fmean(vals) * 100.0
        out[f"median_gap_closure_{b}_pct"] = st.median(vals) * 100.0
    base = out["mean_gap_closure_60s_pct"]
    for b in ("300s", "3600s"):
        out[f"movement_vs_60s_{b}_points"] = out[f"mean_gap_closure_{b}_pct"] - base
    return out


def self_check(summary: dict) -> None:
    """Abort if the 60 s and 300 s columns do not reproduce the published run."""
    problems = []
    if summary["n_instances_in_gap_closure"] != REFERENCE_N:
        problems.append(
            f"n={summary['n_instances_in_gap_closure']}, "
            f"expected {REFERENCE_N}"
        )
    for label, got, want in (
        ("60 s", summary["mean_gap_closure_60s_pct"], REFERENCE_GC_60S_PCT),
        ("300 s", summary["mean_gap_closure_300s_pct"], REFERENCE_GC_300S_PCT),
    ):
        if abs(round(got, 2) - want) > REFERENCE_TOL_PCT:
            problems.append(f"{label} column {got:.2f}%, expected {want:.2f}%")
    if problems:
        bar = "!" * 78
        raise SystemExit(
            f"\n{bar}\n"
            f"!! SELF CHECK FAILED. The recomputation does not reproduce\n"
            f"!! scripts/experiments/milp_resolve_300s.py:\n"
            + "".join(f"!!   {p}\n" for p in problems)
            + f"!! The 3600 s column cannot be trusted; the evaluation path\n"
            f"!! has drifted since the 300 s run. Investigate before using\n"
            f"!! any number from this script.\n"
            f"{bar}\n"
        )
    print(
        f"\nSelf check passed. The 60 s and 300 s columns reproduce "
        f"milp_resolve_300s.py exactly ({REFERENCE_GC_60S_PCT}% and "
        f"{REFERENCE_GC_300S_PCT}% over {REFERENCE_N} instances), so the "
        f"3600 s column is anchored to the published result."
    )


def print_table(summary: dict, per_instance: list) -> None:
    n = summary["n_instances_in_gap_closure"]
    print(
        f"\nLearned policy gap closure by MILP denominator, "
        f"n={n} of {len(per_instance)} instances"
    )
    print("=" * 62)
    print(f"{'denominator':<22s}{'mean':>12s}{'median':>12s}{'vs 60 s':>14s}")
    print("-" * 62)
    for b in BUDGETS:
        move = (
            "baseline" if b == "60s"
            else f"{summary[f'movement_vs_60s_{b}_points']:+.2f} pts"
        )
        print(
            f"{b + ' incumbents':<22s}"
            f"{summary[f'mean_gap_closure_{b}_pct']:>11.2f}%"
            f"{summary[f'median_gap_closure_{b}_pct']:>11.2f}%"
            f"{move:>14s}"
        )
    print("=" * 62)

    usable = [r for r in per_instance if r["denominator_defined"]]
    movers = sorted(
        usable,
        key=lambda r: r["gap_closure_3600s"] - r["gap_closure_60s"],
    )[:3]
    print("\nInstances whose closure fell most when the denominator moved:")
    for r in movers:
        drop = (r["gap_closure_3600s"] - r["gap_closure_60s"]) * 100
        obj_move = (r["obj_60s"] - r["obj_3600s"]) / r["obj_60s"] * 100
        print(
            f"  seed {r['seed']}: {r['gap_closure_60s'] * 100:6.2f}% -> "
            f"{r['gap_closure_3600s'] * 100:6.2f}%  ({drop:+.2f} pts, "
            f"MILP incumbent fell {obj_move:.2f}%)"
        )


def write_outputs(per_instance: list, summary: dict) -> None:
    OUT_PER_INSTANCE.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed", "greedy", "learned_hard",
        "obj_60s", "obj_300s", "obj_3600s",
        "gap_closure_60s", "gap_closure_300s", "gap_closure_3600s",
        "denominator_defined",
    ]
    with OUT_PER_INSTANCE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in per_instance:
            writer.writerow({k: r[k] for k in fields})
    with OUT_SUMMARY.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])
    for p in (OUT_PER_INSTANCE, OUT_SUMMARY):
        side = sha256_sidecar(p)
        print(f"Wrote {p.relative_to(REPO_ROOT)}")
        print(f"Wrote {side.relative_to(REPO_ROOT)}")


def main() -> int:
    order, objectives = load_objectives()
    per_instance = evaluate(order, objectives)
    summary = summarise(per_instance)
    self_check(summary)
    print_table(summary, per_instance)
    print()
    write_outputs(per_instance, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
