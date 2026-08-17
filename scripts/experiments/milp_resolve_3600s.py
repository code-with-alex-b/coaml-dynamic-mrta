"""Re-solve the same twenty validation MILPs at a 3600 second time limit.

Third rung of the budget ladder. The production label cache
(``cache/training_set_il_v3/val``) was solved at 60 seconds per instance by
``src/training/expert_dataset_generator.py``. ``scripts/experiments/milp_resolve_300s.py``
re-solved a twenty instance subsample at 300 seconds. This script re-solves
that same subsample at 3600 seconds so the dual bound has a chance to move.
The point of the run is the certificate rather than the incumbent, so the
column that matters most is ``mip_gap``.

Comparability of the three budgets
----------------------------------
All three runs go through ``anticipative.anticipative_milp.solve_anticipative``
unmodified, so the formulation, the objective weights, the ``1e-6`` tiebreak
coefficient and the ``MIPGap`` target of 0.01 are identical by construction.
The only Gurobi parameters that function sets are ``OutputFlag``, ``TimeLimit``
and ``MIPGap``. In particular ``Threads`` and ``Seed`` are left at Gurobi's
defaults in all three runs, and all three loop strictly sequentially in a
single process with one solve in flight at a time. So the thread count is the
same across the ladder, namely every core on the machine.

That equality only holds while nothing else perturbs it, so this script
refuses to start if a ``gurobi.env`` file is present in the working directory,
since such a file silently overrides ``Threads`` for every environment Gurobi
creates. Run this on the same machine as the earlier two budgets, and do not
run anything else heavy alongside it.

No warm start
-------------
The earlier incumbents are read for comparison only. They are never handed to
the solver. ``solve_anticipative`` has no warm start path, so there is nothing
to switch off, but the point is load bearing: seeding the 3600 second solve
with the 300 second incumbent would change the search tree and destroy
comparability with the two shorter budgets.

Instance list
-------------
The seeds are read from ``results/milp_resolve_300s.csv`` rather than
reconstructed from ``select_seeds()``, so the subsample cannot drift. The
60 second objective for each seed is read from the same file and cross checked
against the production cache; a mismatch aborts the run.

Checkpointing
-------------
One row is appended, flushed and fsynced to the output CSV per instance, so a
crash at instance seventeen costs only instance seventeen. On restart the
script reads the output file, skips every seed already present, and continues
in the original order.

Monotonicity check
------------------
Before each row is written the script checks that the 3600 second objective is
no worse than both the 300 second and the 60 second objective for that seed.
A longer time limit explores a superset of the shorter run's search tree, so a
regression indicates nondeterminism and needs to be seen rather than averaged
in. On a violation the row is still written, but with ``regression_flag``
populated, a loud banner printed, and a non-zero exit status at the end.
Writing the row keeps an overnight run from throwing away its own evidence.
Pass ``--halt-on-regression`` to stop dead at the first violation instead.

Outputs
-------
``results/milp_resolve_3600s.csv`` plus a ``.sha256`` sidecar written once all
twenty rows are present. Per instance Gurobi logs land in
``logs/milp_resolve_3600s/seed<seed>.log`` and are parsed for the node count,
the solver's own runtime, and the number of threads Gurobi actually used. That
last one is the hard evidence that this run matched the earlier thread count.

Nothing under ``results/milp_resolve_300s.csv``, ``cache/`` or any
``milp_solution`` record is written to or modified.

Run from the repository root in the coaml environment:

    PYTHONPATH="$PWD/src" python scripts/experiments/milp_resolve_3600s.py
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import re
import signal
import sys
from datetime import datetime
from pathlib import Path

from anticipative.anticipative_milp import solve_anticipative
from instances.synthetic_generator import SyntheticInstance


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VAL_CACHE_DIR = REPO_ROOT / "cache" / "training_set_il_v3" / "val"
SOURCE_300S_CSV = REPO_ROOT / "results" / "milp_resolve_300s.csv"
RESULTS_CSV = REPO_ROOT / "results" / "milp_resolve_3600s.csv"
GUROBI_LOG_DIR = REPO_ROOT / "logs" / "milp_resolve_3600s"

# Threads and Seed deliberately absent: all three budgets (60s/300s/3600s) leave them at Gurobi's default.
TIME_LIMIT_3600S = 3600
MIP_GAP_TARGET = 0.01

# 1e-6 sits comfortably above the ~1e-8 relative extraction noise observed re-extracting an identical incumbent (seed 11000) and far below any real movement.
MONOTONICITY_REL_TOL = 1e-6
MONOTONICITY_ABS_TOL = 1e-6

# Anything outside optimal/time_limit (notably GRB.INTERRUPTED, "status_11", from a trapped SIGINT) means the search stopped early, so the row must not be written.
ACCEPTABLE_STATUSES = ("optimal", "time_limit")

# A "time_limit" solve should have consumed essentially all of its budget.
BUDGET_CONSUMED_FRACTION = 0.95

CSV_FIELDS = [
    "seed",
    "time_limit_s",
    "obj_val",
    "obj_bound",
    "mip_gap",
    "runtime_s",
    "status",
    "node_count",
    "wall_start",
    "wall_end",
    "gurobi_runtime_s",
    "gurobi_threads",
    "obj_60s",
    "obj_300s",
    "improvement_vs_60s_pct",
    "improvement_vs_300s_pct",
    "regression_flag",
]


_STOP_REQUESTED = False


def _request_stop(signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(
        f"\n>>> Signal {signum} received. Finishing the instance currently in "
        f"flight, writing its row, then stopping at the boundary. This can "
        f"take up to one full time limit. Send SIGKILL if you need to stop "
        f"now, at the cost of the in-flight instance only.",
        flush=True,
    )


def install_signal_handlers() -> None:
    """Turn SIGTERM into a stop-at-the-next-instance-boundary request.

    Note on SIGINT. Gurobi installs its own SIGINT handler at the C level and
    swallows the signal, aborting the solve and returning GRB.INTERRUPTED
    rather than letting Python raise KeyboardInterrupt. So SIGINT does not
    stop this script cleanly and must not be used; the status guard in the
    main loop catches the resulting short solve and refuses to record it.
    Handlers also only run when Python holds the interpreter, which is between
    instances, so a graceful stop waits for the current solve to finish.
    """
    signal.signal(signal.SIGTERM, _request_stop)


def preflight() -> None:
    """Refuse to start under conditions that would break comparability."""
    stray_env = Path.cwd() / "gurobi.env"
    if stray_env.exists():
        raise SystemExit(
            f"Refusing to run: {stray_env} exists. A gurobi.env in the working "
            f"directory silently overrides Threads for every Gurobi "
            f"environment, which would break thread-count comparability with "
            f"the 60 s and 300 s runs. Move it aside and rerun."
        )
    if not SOURCE_300S_CSV.exists():
        raise SystemExit(f"Missing source instance list: {SOURCE_300S_CSV}")
    if not VAL_CACHE_DIR.is_dir():
        raise SystemExit(f"Missing production val cache: {VAL_CACHE_DIR}")


# Read from the 300 s output so the sample cannot drift.
def load_300s_rows() -> list:
    with SOURCE_300S_CSV.open("r", newline="") as f:
        rows = [
            {
                "seed": int(r["seed"]),
                "obj_60s": float(r["obj_60s"]),
                "obj_300s": float(r["obj_300s"]),
                "gap_60s": float(r["gap_60s"]),
                "gap_300s": float(r["gap_300s"]),
                "status_300s": r["status"],
            }
            for r in csv.DictReader(f)
        ]
    if not rows:
        raise SystemExit(f"{SOURCE_300S_CSV} has no rows.")
    seeds = [r["seed"] for r in rows]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"{SOURCE_300S_CSV} contains duplicate seeds.")
    return rows


def load_cached_record(seed: int) -> dict:
    """Read a production cache record. Read only, never written back."""
    with (VAL_CACHE_DIR / f"seed{seed}.json").open("r") as f:
        return json.load(f)


def cross_check_against_cache(rows: list) -> None:
    """Confirm the 300 s file's 60 s column still matches the label cache."""
    for r in rows:
        record = load_cached_record(r["seed"])
        cached_obj = float(record["milp_solution"]["objective_value"])
        if abs(cached_obj - r["obj_60s"]) > 1e-6 * max(1.0, abs(cached_obj)):
            raise SystemExit(
                f"seed {r['seed']}: obj_60s in {SOURCE_300S_CSV.name} is "
                f"{r['obj_60s']!r} but the cache holds {cached_obj!r}. The "
                f"ladder is not self-consistent; investigate before running."
            )
        stored_tl = record.get("mip_time_limit")
        if stored_tl is not None and int(stored_tl) != 60:
            raise SystemExit(
                f"seed {r['seed']}: cache record says mip_time_limit="
                f"{stored_tl}, expected 60."
            )
        stored_gap = record.get("mip_gap_target")
        if stored_gap is not None and abs(float(stored_gap) - MIP_GAP_TARGET) > 1e-12:
            raise SystemExit(
                f"seed {r['seed']}: cache record says mip_gap_target="
                f"{stored_gap}, expected {MIP_GAP_TARGET}."
            )
    print(
        f"Cross check passed: all {len(rows)} seeds agree with the production "
        f"cache on the 60 s objective, time limit and gap target.",
        flush=True,
    )


@contextlib.contextmanager
def capture_stdout_fd(path: Path):
    """Redirect file descriptor 1 to ``path`` for the duration of the block.

    gurobipy writes its log at the C level, so ``contextlib.redirect_stdout``
    does not see it. Redirecting the descriptor itself does. This affects only
    where the log text lands; ``OutputFlag`` has no bearing on the search, so
    the solve is identical to one run with the log suppressed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    saved_fd = os.dup(1)
    log_file = path.open("w")
    try:
        os.dup2(log_file.fileno(), 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        log_file.close()


_NODES_RE = re.compile(r"Explored\s+([\d,]+)\s+nodes?")
_SECONDS_RE = re.compile(r"in\s+([\d.]+)\s+seconds")
_THREADS_RE = re.compile(r"using up to (\d+) threads")


def parse_gurobi_log(path: Path) -> dict:
    """Best effort scrape of node count, runtime and thread count.

    Parsing failures are not fatal. The authoritative objective, bound, gap and
    status all come from the returned solution object; this only enriches the
    provenance columns.
    """
    out = {"node_count": "", "gurobi_runtime_s": "", "gurobi_threads": ""}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return out
    m = _THREADS_RE.search(text)
    if m:
        out["gurobi_threads"] = int(m.group(1))
    for line in text.splitlines():
        if not line.startswith("Explored "):
            continue
        m = _NODES_RE.search(line)
        if m:
            out["node_count"] = int(m.group(1).replace(",", ""))
        m = _SECONDS_RE.search(line)
        if m:
            out["gurobi_runtime_s"] = float(m.group(1))
    return out


def load_finished_rows() -> list:
    if not RESULTS_CSV.exists():
        return []
    with RESULTS_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_FIELDS:
            raise SystemExit(
                f"{RESULTS_CSV} exists but its header does not match this "
                f"script's schema.\n  on disk: {reader.fieldnames}\n"
                f"  expected: {CSV_FIELDS}\n"
                f"Appending would produce a mixed-schema file. Move the old "
                f"file aside and rerun."
            )
        return list(reader)


def append_csv_row(row: dict) -> None:
    write_header = not RESULTS_CSV.exists()
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def write_sha256_sidecar(path: Path) -> Path:
    """Write ``<file>.sha256`` in the format ``shasum -a 256`` produces.

    One line holding the hex digest, two spaces and the base name, matching the
    existing sidecars under results/ and provenance/.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    side = path.with_name(path.name + ".sha256")
    side.write_text(f"{digest}  {path.name}\n")
    return side


def monotonicity_violations(obj_3600: float, obj_300: float, obj_60: float) -> list:
    """Return the labels of any budget the 3600 s objective is worse than.

    A larger time limit continues the same deterministic search rather than
    restarting it, so the 3600 s incumbent must be no worse than either shorter
    budget. Anything else is nondeterminism and must be surfaced.
    """
    violations = []
    for label, reference in (("300s", obj_300), ("60s", obj_60)):
        slack = MONOTONICITY_ABS_TOL + MONOTONICITY_REL_TOL * abs(reference)
        if obj_3600 > reference + slack:
            violations.append(
                f"worse_than_{label}(+{obj_3600 - reference:.6g})"
            )
    return violations


def print_regression_banner(seed: int, violations: list, obj_3600: float,
                            obj_300: float, obj_60: float) -> None:
    bar = "!" * 78
    print(
        f"\n{bar}\n"
        f"!! MONOTONICITY REGRESSION, seed {seed}\n"
        f"!!   obj_3600s = {obj_3600!r}\n"
        f"!!   obj_300s  = {obj_300!r}\n"
        f"!!   obj_60s   = {obj_60!r}\n"
        f"!!   violated  = {', '.join(violations)}\n"
        f"!! A longer time limit produced a WORSE incumbent. This is not a\n"
        f"!! rounding artefact at these tolerances. Treat the ladder as\n"
        f"!! nondeterministic and do not average this instance in.\n"
        f"{bar}\n",
        flush=True,
    )


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--halt-on-regression",
        action="store_true",
        help=(
            "Stop at the first monotonicity violation instead of flagging it, "
            "writing the row and continuing."
        ),
    )
    parser.add_argument(
        "--quiet-solver",
        action="store_true",
        help=(
            "Suppress the Gurobi log. Leaves node_count, gurobi_runtime_s and "
            "gurobi_threads blank; the solve itself is unaffected."
        ),
    )
    args = parser.parse_args()

    install_signal_handlers()
    preflight()
    source_rows = load_300s_rows()
    cross_check_against_cache(source_rows)

    by_seed = {r["seed"]: r for r in source_rows}
    order = [r["seed"] for r in source_rows]

    existing = load_finished_rows()
    done_seeds = {int(r["seed"]) for r in existing}
    remaining = [s for s in order if s not in done_seeds]

    print(
        f"Budget: {TIME_LIMIT_3600S}s, MIPGap target {MIP_GAP_TARGET}, "
        f"Threads and Seed at Gurobi defaults (as in the 60 s and 300 s runs).",
        flush=True,
    )
    print(f"Instance list from {SOURCE_300S_CSV.name}: {len(order)} seeds.", flush=True)
    if existing:
        print(
            f"Resuming: {len(existing)}/{len(order)} already in "
            f"{RESULTS_CSV.relative_to(REPO_ROOT)}.",
            flush=True,
        )
    if remaining:
        print(f"{len(remaining)} remaining: {remaining}", flush=True)
    else:
        print("All seeds already solved. Nothing to do.", flush=True)

    n_regressions = sum(1 for r in existing if r.get("regression_flag"))
    total_done = len(existing)

    for seed in remaining:
        if _STOP_REQUESTED:
            print(
                f"\nStopping at instance boundary as requested. "
                f"{total_done}/{len(order)} rows on disk. Rerun the same "
                f"command to resume.",
                flush=True,
            )
            break

        src = by_seed[seed]
        obj_60s = src["obj_60s"]
        obj_300s = src["obj_300s"]

        record = load_cached_record(seed)
        instance = SyntheticInstance.from_dict(record["instance"])
        weights = record["weights"]

        wall_start = now_iso()
        log_path = GUROBI_LOG_DIR / f"seed{seed}.log"
        print(
            f"[{total_done + 1}/{len(order)}] seed={seed} starting at "
            f"{wall_start} (limit {TIME_LIMIT_3600S}s)",
            flush=True,
        )

        # No warm start; the earlier incumbents above are for comparison only.
        if args.quiet_solver:
            solution = solve_anticipative(
                instance,
                weights,
                time_limit_seconds=TIME_LIMIT_3600S,
                mip_gap=MIP_GAP_TARGET,
            )
            log_info = {"node_count": "", "gurobi_runtime_s": "",
                        "gurobi_threads": ""}
        else:
            with capture_stdout_fd(log_path):
                solution = solve_anticipative(
                    instance,
                    weights,
                    time_limit_seconds=TIME_LIMIT_3600S,
                    mip_gap=MIP_GAP_TARGET,
                    verbose=True,
                )
            log_info = parse_gurobi_log(log_path)

        wall_end = now_iso()

        # A short search recorded as a 3600 s certificate is exactly what this run exists to prevent.
        runtime_s = float(solution.solve_time_seconds)
        reject = None
        if solution.status not in ACCEPTABLE_STATUSES:
            reject = (
                f"status={solution.status!r} is neither 'optimal' nor "
                f"'time_limit'. status_11 means GRB.INTERRUPTED, i.e. the "
                f"solve was stopped by a signal."
            )
        elif (
            solution.status == "time_limit"
            and runtime_s < BUDGET_CONSUMED_FRACTION * TIME_LIMIT_3600S
        ):
            reject = (
                f"status='time_limit' but the solve ran only {runtime_s:.1f}s "
                f"of its {TIME_LIMIT_3600S}s budget."
            )
        if reject is not None:
            bar = "!" * 78
            raise SystemExit(
                f"\n{bar}\n"
                f"!! REFUSING TO RECORD seed {seed}\n"
                f"!! {reject}\n"
                f"!! No row written, so rerunning this script re-solves this "
                f"seed from scratch.\n"
                f"!! Rows already on disk are unaffected.\n"
                f"{bar}\n"
            )

        obj_3600s = float(solution.objective_value)
        violations = monotonicity_violations(obj_3600s, obj_300s, obj_60s)
        if violations:
            print_regression_banner(seed, violations, obj_3600s, obj_300s, obj_60s)
            n_regressions += 1

        row = {
            "seed": seed,
            "time_limit_s": TIME_LIMIT_3600S,
            "obj_val": obj_3600s,
            "obj_bound": float(solution.obj_bound),
            "mip_gap": float(solution.mip_gap),
            "runtime_s": runtime_s,
            "status": solution.status,
            "node_count": log_info["node_count"],
            "wall_start": wall_start,
            "wall_end": wall_end,
            "gurobi_runtime_s": log_info["gurobi_runtime_s"],
            "gurobi_threads": log_info["gurobi_threads"],
            "obj_60s": obj_60s,
            "obj_300s": obj_300s,
            "improvement_vs_60s_pct": (obj_60s - obj_3600s) / obj_60s * 100.0,
            "improvement_vs_300s_pct": (obj_300s - obj_3600s) / obj_300s * 100.0,
            "regression_flag": ";".join(violations),
        }

        if violations and args.halt_on_regression:
            raise SystemExit(
                f"Halting on monotonicity regression at seed {seed} as "
                f"requested. Row NOT written. Rerun without "
                f"--halt-on-regression to record it and continue."
            )

        append_csv_row(row)
        total_done += 1

        print(
            f"[{total_done}/{len(order)}] seed={seed} done at {wall_end} "
            f"obj={obj_3600s:.4f} bound={row['obj_bound']:.4f} "
            f"gap={row['mip_gap']:.4f} (300s gap was {src['gap_300s']:.4f}) "
            f"status={solution.status} runtime={row['runtime_s']:.1f}s "
            f"nodes={log_info['node_count']} threads={log_info['gurobi_threads']}",
            flush=True,
        )

    rows = load_finished_rows()
    print(f"\n=== {len(rows)}/{len(order)} instances present ===", flush=True)

    if len(rows) != len(order):
        print(
            "Incomplete. Rerun the same command to resume from where it "
            "stopped. No sidecar written for a partial file.",
            flush=True,
        )
        return 2

    side = write_sha256_sidecar(RESULTS_CSV)
    print(f"Wrote {RESULTS_CSV.relative_to(REPO_ROOT)}", flush=True)
    print(f"Wrote {side.relative_to(REPO_ROOT)}", flush=True)

    threads_seen = sorted(
        {r["gurobi_threads"] for r in rows if r["gurobi_threads"]}
    )
    print(f"Gurobi thread counts observed across the run: {threads_seen or 'not captured'}")
    n_optimal = sum(1 for r in rows if r["status"] == "optimal")
    n_limit = sum(1 for r in rows if r["status"] == "time_limit")
    print(f"Proven optimal: {n_optimal}. Terminated at the limit: {n_limit}.")

    if n_regressions:
        flagged = [r["seed"] for r in rows if r["regression_flag"]]
        print(
            f"\n*** {n_regressions} MONOTONICITY REGRESSION(S): seeds "
            f"{flagged}. Do not average these in. Exiting non-zero. ***",
            flush=True,
        )
        return 1

    print("No monotonicity regressions. Objectives are monotone across the ladder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
