"""Summarise the 60 s, 300 s and 3600 s MILP budgets over the same twenty seeds.

One column per budget: instances proven optimal, instances terminated at the
time limit, mean and median optimality gap, mean and maximum incumbent
improvement on the 60 s run, and count whose incumbent did not move off the
60 s value. The 60 s column is the baseline the improvement rows are measured
against, so its three improvement entries are definitional rather than
informative: zero mean, zero maximum, and all twenty unmoved.

Sources, all read only:

  60 s    cache/training_set_il_v3/val/seed<seed>.json, milp_solution
  300 s   results/milp_resolve_300s.csv
  3600 s  results/milp_resolve_3600s.csv

The seed list comes from the 300 s file, which defines the subsample. All three
budgets must cover exactly that list or the script aborts, so a partial
overnight run cannot be silently summarised as if it were complete.

Prints the table and writes results/milp_budget_comparison.csv with a .sha256
sidecar.

Run from the repository root in the coaml environment:

    PYTHONPATH="$PWD/src" python scripts/regenerate/milp_budget_summary.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VAL_CACHE_DIR = REPO_ROOT / "cache" / "training_set_il_v3" / "val"
CSV_300S = REPO_ROOT / "results" / "milp_resolve_300s.csv"
CSV_3600S = REPO_ROOT / "results" / "milp_resolve_3600s.csv"
OUT_CSV = REPO_ROOT / "results" / "milp_budget_comparison.csv"

# Matches scripts/experiments/milp_resolve_3600s.py. Re-extracting an
# identical incumbent reproduces it to roughly 1e-8 relative, so 1e-6 sits
# well above extraction noise and well below any real incumbent movement.
MOVEMENT_REL_TOL = 1e-6
MOVEMENT_ABS_TOL = 1e-6

# Three val records whose cached objective exceeds the greedy cost yet certify
# optimal. Known and documented, reported here as a footnote so they are not
# mistaken for the solver closing hard instances.
KNOWN_CORRUPTED_SEEDS = [11044, 11055, 11169]

BUDGETS = ["60s", "300s", "3600s"]

METRIC_ROWS = [
    ("n_instances", "instances"),
    ("n_proven_optimal", "proven optimal"),
    ("n_terminated_at_limit", "terminated at limit"),
    ("mean_optimality_gap", "mean optimality gap"),
    ("median_optimality_gap", "median optimality gap"),
    ("mean_improvement_vs_60s_pct", "mean improvement vs 60 s (%)"),
    ("max_improvement_vs_60s_pct", "max improvement vs 60 s (%)"),
    ("n_incumbent_unmoved_vs_60s", "incumbent did not move"),
]


def sha256_sidecar(path: Path) -> Path:
    """Write ``<file>.sha256`` in the format ``shasum -a 256`` produces."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    side = path.with_name(path.name + ".sha256")
    side.write_text(f"{digest}  {path.name}\n")
    return side


def require(path: Path, hint: str) -> None:
    """Abort with one line if a gitignored input is absent.

    ``hint`` says what the input is for; the rest follows the shape every
    script under scripts/regenerate/ uses: what is missing, that it is
    gitignored and absent from a clone, and which committed file ships.
    """
    if not path.exists():
        raise SystemExit(
            f"Missing {path.relative_to(REPO_ROOT)}. {hint} It is gitignored "
            f"and absent from a fresh clone, so this script can only run on a "
            f"tree that carries it. The committed "
            f"provenance/appendix_c_benchmark/milp_budget_comparison.csv carries the result."
        )


def load_budgets() -> dict:
    """Return {seed: {budget: {obj, gap, status}}} plus the ordered seed list."""
    require(CSV_300S, "The 300 s run defines the subsample.")
    require(
        CSV_3600S,
        "Run scripts/experiments/milp_resolve_3600s.py to completion first; it writes that "
        "file one row at a time and only sidecars it when all rows are present.",
    )

    with CSV_300S.open("r", newline="") as f:
        rows_300 = list(csv.DictReader(f))
    with CSV_3600S.open("r", newline="") as f:
        rows_3600 = list(csv.DictReader(f))

    order = [int(r["seed"]) for r in rows_300]

    seeds_3600 = [int(r["seed"]) for r in rows_3600]
    missing = [s for s in order if s not in set(seeds_3600)]
    extra = [s for s in seeds_3600 if s not in set(order)]
    if missing:
        raise SystemExit(
            f"The 3600 s file is missing {len(missing)} of the {len(order)} "
            f"seeds: {missing}. It is an incomplete run. Resume it, do not "
            f"summarise it."
        )
    if extra:
        raise SystemExit(
            f"The 3600 s file holds seeds not in the 300 s subsample: {extra}. "
            f"The samples have drifted; investigate before summarising."
        )

    by_3600 = {int(r["seed"]): r for r in rows_3600}
    data: dict = {}

    for r300 in rows_300:
        seed = int(r300["seed"])
        cache_path = VAL_CACHE_DIR / f"seed{seed}.json"
        require(cache_path, "The 60 s column comes from the production cache.")
        with cache_path.open("r") as f:
            ms = json.load(f)["milp_solution"]

        obj_60_cache = float(ms["objective_value"])
        obj_60_csv = float(r300["obj_60s"])
        if abs(obj_60_cache - obj_60_csv) > 1e-6 * max(1.0, abs(obj_60_cache)):
            raise SystemExit(
                f"seed {seed}: 60 s objective disagrees between the cache "
                f"({obj_60_cache!r}) and {CSV_300S.name} ({obj_60_csv!r})."
            )

        r3600 = by_3600[seed]
        obj_3600_recorded_60s = float(r3600["obj_60s"])
        if abs(obj_3600_recorded_60s - obj_60_csv) > 1e-6 * max(1.0, abs(obj_60_csv)):
            raise SystemExit(
                f"seed {seed}: the 3600 s file recorded a different 60 s "
                f"reference ({obj_3600_recorded_60s!r}) than {CSV_300S.name} "
                f"({obj_60_csv!r})."
            )

        data[seed] = {
            "60s": {
                "obj": obj_60_cache,
                "gap": float(ms["mip_gap"]),
                "status": str(ms["status"]),
            },
            "300s": {
                "obj": float(r300["obj_300s"]),
                "gap": float(r300["gap_300s"]),
                "status": str(r300["status"]),
            },
            "3600s": {
                "obj": float(r3600["obj_val"]),
                "gap": float(r3600["mip_gap"]),
                "status": str(r3600["status"]),
            },
        }

    return data, order, rows_3600


def improvement_pct(obj_60: float, obj_b: float) -> float:
    return (obj_60 - obj_b) / obj_60 * 100.0


def unmoved(obj_60: float, obj_b: float) -> bool:
    slack = MOVEMENT_ABS_TOL + MOVEMENT_REL_TOL * abs(obj_60)
    return abs(obj_60 - obj_b) <= slack


def summarise(data: dict, order: list) -> dict:
    out: dict = {}
    for budget in BUDGETS:
        entries = [data[s][budget] for s in order]
        gaps = [e["gap"] for e in entries]
        improvements = [
            improvement_pct(data[s]["60s"]["obj"], data[s][budget]["obj"])
            for s in order
        ]
        n_unmoved = sum(
            1 for s in order
            if unmoved(data[s]["60s"]["obj"], data[s][budget]["obj"])
        )
        out[budget] = {
            "n_instances": len(order),
            "n_proven_optimal": sum(1 for e in entries if e["status"] == "optimal"),
            "n_terminated_at_limit": sum(
                1 for e in entries if e["status"] == "time_limit"
            ),
            "mean_optimality_gap": statistics.fmean(gaps),
            "median_optimality_gap": statistics.median(gaps),
            "mean_improvement_vs_60s_pct": statistics.fmean(improvements),
            "max_improvement_vs_60s_pct": max(improvements),
            "n_incumbent_unmoved_vs_60s": n_unmoved,
        }
    return out


def fmt(key: str, value) -> str:
    if key.startswith("n_"):
        return f"{int(value)}"
    if "gap" in key:
        return f"{value:.4f}"
    return f"{value:+.3f}"


def print_table(summary: dict, data: dict, order: list, rows_3600: list) -> None:
    label_w = max(len(lbl) for _, lbl in METRIC_ROWS)
    col_w = 12

    header = "  ".join(
        ["".ljust(label_w)] + [b.rjust(col_w) for b in BUDGETS]
    )
    print()
    print("MILP budget ladder, twenty validation instances, identical "
          "formulation, weights, gap target and thread settings")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for key, label in METRIC_ROWS:
        cells = [fmt(key, summary[b][key]).rjust(col_w) for b in BUDGETS]
        print("  ".join([label.ljust(label_w)] + cells))
    print("=" * len(header))

    print(
        "\nThe 60 s column is the baseline for the three improvement rows, so "
        "its entries there are\ndefinitional (zero movement against itself), "
        "not a finding."
    )

    present_corrupted = [s for s in KNOWN_CORRUPTED_SEEDS if s in data]
    if present_corrupted:
        print(
            f"\nFootnote. Seeds {present_corrupted} certify optimal at all "
            f"three budgets at objectives that\nexceed the greedy cost, and "
            f"are the documented corrupted records. They inflate the\n"
            f"'proven optimal' row in every column equally; the row is not "
            f"evidence of the solver\nclosing hard instances."
        )

    threads = sorted({r["gurobi_threads"] for r in rows_3600 if r["gurobi_threads"]})
    if threads:
        print(f"\nGurobi threads recorded in the 3600 s run: {threads}.")
        if len(threads) > 1:
            print(
                "  More than one thread count across the run. The budgets are "
                "NOT comparable\n  under identical thread settings. Report "
                "this rather than the table above."
            )
    else:
        print(
            "\nThread count was not captured in the 3600 s run (it was run "
            "with --quiet-solver).\nThread-count identity across the ladder "
            "rests on code inspection alone."
        )

    flagged = [r["seed"] for r in rows_3600 if r.get("regression_flag")]
    if flagged:
        print(
            f"\n*** {len(flagged)} MONOTONICITY REGRESSION(S) in the 3600 s "
            f"run, seeds {flagged}. ***\n"
            f"    A longer limit produced a worse incumbent on these. They are "
            f"included in the\n    table above unweighted; consider them "
            f"unreliable."
        )


def write_csv(summary: dict) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + [f"budget_{b}" for b in BUDGETS])
        for key, _label in METRIC_ROWS:
            writer.writerow([key] + [summary[b][key] for b in BUDGETS])
    side = sha256_sidecar(OUT_CSV)
    print(f"\nWrote {OUT_CSV.relative_to(REPO_ROOT)}")
    print(f"Wrote {side.relative_to(REPO_ROOT)}")


def main() -> int:
    data, order, rows_3600 = load_budgets()
    summary = summarise(data, order)
    print_table(summary, data, order, rows_3600)
    write_csv(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
