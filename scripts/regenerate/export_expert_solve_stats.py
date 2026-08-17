"""Export the expert MILP solve statistics into tracked provenance.

Appendix A quotes the Gurobi cost of building the imitation training set, but
until now those numbers existed only as an aggregate over the gitignored
``cache/training_set_il_v4/train/seed*.json``, unreproducible from a fresh
clone. This script reads the cache and writes the aggregate to
``provenance/methodology/expert_solve_stats.csv`` with a sha256 sidecar, so the
figures survive independently of the cache.

Reported over the 1,000 training instances: total solve time (sum of
``milp_solution.solve_time_seconds``), count at the time limit
(``status == "time_limit"``), count proven optimal (``status == "optimal"``),
median solve time and mean residual gap (mean of ``milp_solution.mip_gap``).

Caveat, recorded in the output. Every v4 record carries
``reextracted_from: cache/training_set_il_v3`` and its solve times are
bit-identical to v3's: the MILP was solved once, for v3, and v4 reused those
solutions and re-ran only the expert-decision extraction. The total below is
therefore a one-time cost attributed to the training set, not a cost paid
again to build v4; the script verifies this against the v3 cache when present
and records the outcome.

Solves were sequential (``training.expert_dataset_generator.generate_split``
is a plain loop with no pool), so the total is wall clock for the sequence.
Each individual solve ran multi-threaded, so it is not core-hours.

Read only with respect to ``cache/``. Writes two files under ``provenance/``,
overwriting its own previous output so the export is idempotent.

Run from the repository root in the coaml environment:

    python scripts/regenerate/export_expert_solve_stats.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics as st
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_V4 = REPO_ROOT / "cache" / "training_set_il_v4" / "train"
CACHE_V3 = REPO_ROOT / "cache" / "training_set_il_v3" / "train"

OUT_CSV = REPO_ROOT / "provenance" / "expert_solve_stats.csv"

# The training split is 1,000 instances; a short read is a truncated cache,
# not a smaller experiment, so it aborts rather than reporting a partial total.
EXPECTED_N = 1000

# Production cache-generation settings, fixed in
# src/training/expert_dataset_generator.py (MIP_TIME_LIMIT, MIP_GAP). Every
# record is checked against these so a cache built under different settings
# cannot be silently aggregated into the same row.
EXPECTED_TIME_LIMIT = 60
EXPECTED_GAP_TARGET = 0.01


def sha256_sidecar(path: Path) -> Path:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    side = path.with_name(path.name + ".sha256")
    side.write_text(f"{digest}  {path.name}\n")
    return side


def load_split(cache_dir: Path) -> dict:
    """Return {seed: milp_solution dict} for every record under ``cache_dir``."""
    out = {}
    for p in sorted(cache_dir.glob("seed*.json")):
        with p.open() as fh:
            rec = json.load(fh)
        out[int(rec["seed"])] = {
            "solve_time_seconds": float(rec["milp_solution"]["solve_time_seconds"]),
            "status": str(rec["milp_solution"]["status"]),
            "mip_gap": float(rec["milp_solution"]["mip_gap"]),
            "mip_time_limit": rec.get("mip_time_limit"),
            "mip_gap_target": rec.get("mip_gap_target"),
            "reextracted_from": rec.get("reextracted_from"),
        }
    return out


def check_settings_uniform(records: dict) -> None:
    """Abort unless every record was solved under the production settings."""
    limits = {r["mip_time_limit"] for r in records.values()}
    gaps = {r["mip_gap_target"] for r in records.values()}
    problems = []
    if limits != {EXPECTED_TIME_LIMIT}:
        problems.append(f"mip_time_limit values {sorted(limits)}, expected "
                        f"{{{EXPECTED_TIME_LIMIT}}}")
    if gaps != {EXPECTED_GAP_TARGET}:
        problems.append(f"mip_gap_target values {sorted(gaps)}, expected "
                        f"{{{EXPECTED_GAP_TARGET}}}")
    if problems:
        raise SystemExit(
            "REFUSING to aggregate: the cache is not uniform under the "
            "production settings:\n" + "".join(f"  {p}\n" for p in problems)
        )


def check_reuse_against_v3(v4: dict) -> tuple:
    """Verify the v3-reuse claim. Returns (verdict, max_abs_difference)."""
    sources = {r["reextracted_from"] for r in v4.values()}
    if sources != {"cache/training_set_il_v3"}:
        return (f"unexpected reextracted_from {sorted(sources)}", None)
    if not CACHE_V3.is_dir():
        return ("v3 cache absent, claim not checked", None)
    v3 = load_split(CACHE_V3)
    common = set(v3) & set(v4)
    if not common:
        return ("v3 cache present but shares no seeds", None)
    worst = max(
        abs(v3[s]["solve_time_seconds"] - v4[s]["solve_time_seconds"])
        for s in common
    )
    verdict = (
        f"verified identical on {len(common)} shared seeds"
        if worst == 0.0
        else f"DIFFERS on shared seeds, max {worst:.6g} s"
    )
    return (verdict, worst)


def summarise(records: dict, reuse_verdict: str) -> list:
    times = [r["solve_time_seconds"] for r in records.values()]
    statuses = [r["status"] for r in records.values()]
    gaps = [r["mip_gap"] for r in records.values()]

    n_limit = sum(1 for s in statuses if s == "time_limit")
    n_opt = sum(1 for s in statuses if s == "optimal")
    other = sorted({s for s in statuses} - {"time_limit", "optimal"})
    if other:
        raise SystemExit(
            f"REFUSING to aggregate: unexpected solver statuses {other}. "
            f"The time-limit and optimal counts would not sum to n."
        )

    total = float(sum(times))
    return [
        ("n_instances", len(records)),
        ("mip_time_limit_seconds", EXPECTED_TIME_LIMIT),
        ("mip_gap_target", EXPECTED_GAP_TARGET),
        ("total_solve_time_seconds", f"{total:.6f}"),
        ("total_solve_time_hours", f"{total / 3600.0:.6f}"),
        ("n_terminating_at_time_limit", n_limit),
        ("n_proven_optimal", n_opt),
        ("median_solve_time_seconds", f"{st.median(times):.6f}"),
        ("mean_solve_time_seconds", f"{st.fmean(times):.6f}"),
        ("min_solve_time_seconds", f"{min(times):.6f}"),
        ("max_solve_time_seconds", f"{max(times):.6f}"),
        ("mean_residual_mip_gap", f"{st.fmean(gaps):.6f}"),
        ("mean_residual_mip_gap_pct", f"{100.0 * st.fmean(gaps):.4f}"),
        ("solves_were_sequential", "true"),
        ("source_cache", "cache/training_set_il_v4/train"),
        ("solve_times_reused_from", "cache/training_set_il_v3"),
        ("reuse_check", reuse_verdict),
    ]


def print_table(summary: list) -> None:
    width = max(len(k) for k, _ in summary)
    print("\nExpert MILP solve statistics, imitation training split")
    print("=" * (width + 34))
    for k, v in summary:
        print(f"{k:<{width}}  {v}")
    print("=" * (width + 34))


def write_outputs(summary: list) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for k, v in summary:
            writer.writerow([k, v])
    side = sha256_sidecar(OUT_CSV)
    print(f"\nWrote {OUT_CSV.relative_to(REPO_ROOT)}")
    print(f"Wrote {side.relative_to(REPO_ROOT)}")


def main() -> int:
    if not CACHE_V4.is_dir():
        raise SystemExit(
            f"Missing {CACHE_V4.relative_to(REPO_ROOT)}. It is gitignored and "
            f"absent from a fresh clone, so this export can only be regenerated "
            f"on a tree that carries it. The committed "
            f"provenance/methodology/expert_solve_stats.csv is the shipped artefact."
        )

    records = load_split(CACHE_V4)
    if len(records) != EXPECTED_N:
        raise SystemExit(
            f"REFUSING to aggregate: found {len(records)} records under "
            f"{CACHE_V4.relative_to(REPO_ROOT)}, expected {EXPECTED_N}. A short "
            f"read is a truncated cache, not a smaller experiment."
        )

    check_settings_uniform(records)
    reuse_verdict, _ = check_reuse_against_v3(records)
    summary = summarise(records, reuse_verdict)
    print_table(summary)
    write_outputs(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
