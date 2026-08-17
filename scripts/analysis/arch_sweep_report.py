#!/usr/bin/env python
"""Report the Method One architecture sweep, one block per cell.

Reads the per-evaluation CSV written by il_trainer and the per-cell training logs.
For each cell prints the parameter count, the evaluation trajectory, the mean over
the final three evaluations, the serve-all rate and the instance count.

The 2 layer 64 hidden cell is reported first and checked against the production
reference of +6.7% at step 9000. Method One is deterministic given a configuration,
so a materially different number there means the run was perturbed and the other
eight cells cannot be read as architecture effects.

Perturbed decode at epsilon 1.0 only.

Usage:
    python scripts/analysis/arch_sweep_report.py [--csv results/dm_arch_sweep.csv]
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import OrderedDict
from pathlib import Path

# Production Method One perturbed gap closure, from provenance/sensitivity_study_b_20260801.md; the L2_H64 cell is a replicate of production at a different RNG stream, so this is what it should return.

PRODUCTION_REFERENCE = 0.067
PRODUCTION_REFERENCE_STEP = 9000
# Study B's B1 replicate landed +0.04 pp from the reference, so treat anything inside a tenth of a point as "returns the production number".

REFERENCE_TOLERANCE = 0.001
YARDSTICK = "L2_H64"


def load_rows(csv_path: Path):
    """Return an ordered mapping run -> list of evaluation rows."""
    if not csv_path.exists():
        raise SystemExit(f"No eval CSV at {csv_path}; nothing to report.")
    by_run = OrderedDict()
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            by_run.setdefault(row["run"], []).append(row)
    return by_run


def resume_evidence(run: str, log_dir: Path):
    """Return (resumed, note) for one cell, read from its training log.

    A resume is visible two ways: the trainer prints a resume line, and the
    eval CSV gains a second row at a step it already had. Resume does not
    reproduce the torch RNG stream, so a resumed cell is not bit-comparable
    with an unresumed one and the flag has to travel with the number.
    """
    log_path = log_dir / f"dm_arch_{run}.log"
    if not log_path.exists():
        return None, f"no log at {log_path}"
    text = log_path.read_text(errors="replace")
    hits = re.findall(r"^Resum(?:ed|ing) from.*$", text, flags=re.MULTILINE)
    if hits:
        return True, hits[-1].strip()
    return False, "no resume line in the log"


def dedupe(rows):
    """Keep the last row per step, and report how many steps were re-run.

    A relaunch re-evaluates steps the first attempt already wrote, so the CSV
    can hold more than one row per step. The later row is the one the surviving
    checkpoint corresponds to.
    """
    by_step = OrderedDict()
    duplicated = 0
    for row in rows:
        step = int(row["step"])
        if step in by_step:
            duplicated += 1
        by_step[step] = row
    ordered = [by_step[s] for s in sorted(by_step)]
    return ordered, duplicated


def report_cell(run: str, rows, log_dir: Path, is_yardstick: bool) -> None:
    rows, duplicated = dedupe(rows)
    head = rows[0]
    n_params = head["n_params"]
    total_steps = head["total_steps"]

    print(f"=== {run} ===")
    print(f"  parameters        : {int(n_params):,}")
    print(f"  layers / hidden   : {head['num_layers']} / {head['hidden_dim']}")
    print(f"  evaluations       : {len(rows)} recorded, run budget "
          f"{int(total_steps)} steps")

    resumed, note = resume_evidence(run, log_dir)
    resumed_str = {True: "yes", False: "no", None: "unknown"}[resumed]
    print(f"  resumed from ckpt : {resumed_str} ({note})")
    if duplicated:
        print(f"  NOTE: {duplicated} evaluation step(s) appear more than once "
              f"in the CSV, which is the signature of a relaunch. The later "
              f"row is used.")

    print("  trajectory (perturbed gap closure, epsilon 1.0):")
    print("    step    gap        serve-all   n_serving/n_val   best?")
    for row in rows:
        flag = "*" if row["is_new_best"] == "1" else ""
        print(f"    {int(row['step']):<7d} "
              f"{float(row['perturbed_gap']):+.4f}   "
              f"{float(row['served_all_rate']):.3f}       "
              f"{row['n_serving']}/{row['n_val_instances']}"
              f"          {flag}")

    tail = rows[-3:]
    if len(tail) < 3:
        print(f"  mean of final 3   : not available, only {len(tail)} "
              f"evaluation(s) recorded")
        return
    mean_tail = sum(float(r["perturbed_gap"]) for r in tail) / len(tail)
    tail_steps = ", ".join(str(int(r["step"])) for r in tail)
    n_val = {r["n_val_instances"] for r in tail}
    n_serving = {r["n_serving"] for r in tail}
    serve_rates = [float(r["served_all_rate"]) for r in tail]
    print(f"  mean of final 3   : {mean_tail:+.4f} ({mean_tail * 100:+.2f}%) "
          f"over steps {tail_steps}")
    print(f"  serve-all (last 3): "
          f"{', '.join(f'{r:.3f}' for r in serve_rates)}")
    print(f"  computed over     : n_serving {sorted(n_serving)} of n_val "
          f"{sorted(n_val)} validation instances")

    if is_yardstick:
        at_ref = [r for r in rows
                  if int(r["step"]) == PRODUCTION_REFERENCE_STEP]
        print()
        print(f"  --- yardstick check against production "
              f"({PRODUCTION_REFERENCE:+.4f} at step "
              f"{PRODUCTION_REFERENCE_STEP}) ---")
        if not at_ref:
            print(f"  No evaluation at step {PRODUCTION_REFERENCE_STEP}; "
                  f"cannot check the reference. Treat the sweep as "
                  f"unverified.")
            return
        value = float(at_ref[-1]["perturbed_gap"])
        delta = value - PRODUCTION_REFERENCE
        print(f"  {run} at step {PRODUCTION_REFERENCE_STEP}: {value:+.4f} "
              f"({value * 100:+.2f}%), delta {delta * 100:+.2f} pp")
        if abs(delta) <= REFERENCE_TOLERANCE:
            print("  VERDICT: matches production. The new flags did not "
                  "perturb the run, so the other cells read as clean "
                  "architecture effects.")
        else:
            print("  VERDICT: DOES NOT match production. Method One is "
                  "deterministic given a configuration, so this means the "
                  "new flags perturbed the run. Every other cell in this "
                  "sweep is contaminated and none of the architecture "
                  "deltas can be trusted.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results/dm_arch_sweep.csv")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    by_run = load_rows(Path(args.csv))
    log_dir = Path(args.log_dir)

    print("Method One architecture sweep report")
    print(f"Source: {args.csv}")
    print("Metric: perturbed-decode gap closure versus the MILP bound at "
          "epsilon 1.0, on the 200 instance validation cache (seeds 11000 to "
          "11199).")
    print("Method One hard decode is deliberately not reported.")
    print()

    # Yardstick first, so a contaminated sweep is visible before any architecture comparison is read.

    order = ([YARDSTICK] if YARDSTICK in by_run else []) + [
        r for r in by_run if r != YARDSTICK
    ]
    for run in order:
        report_cell(run, by_run[run], log_dir, is_yardstick=(run == YARDSTICK))
        print()

    expected = {f"L{layer}_H{hidden}"
                for layer in (1, 2, 3) for hidden in (32, 64, 128)}
    missing = sorted(expected - set(by_run))
    if missing:
        print(f"Cells with no evaluations in the CSV: {', '.join(missing)}")
    if YARDSTICK not in by_run:
        print(f"WARNING: the {YARDSTICK} yardstick cell is absent, so nothing "
              f"here has been checked against production.")


if __name__ == "__main__":
    main()
