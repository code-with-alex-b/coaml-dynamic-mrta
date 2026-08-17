"""Export checkpoint-only and log-only series into tracked provenance CSVs.

Phase 1.5, final step. Two classes of quantity have no CSV behind them: the
per-step ``cost_history`` and ``grad_norm_history`` inside two rolling training
checkpoints, which produce Figure 4.1 and claims C087 to C100, and ten claims
across five log groups that live only in the gitignored, single-machine
``logs/``. This script reads both and writes plain CSVs under ``provenance/``
so notebook one can regenerate Figure 4.1 and every log-only claim without
opening a checkpoint or a log.

Read only with respect to everything it reads. It writes new CSVs and touches
nothing else. No model is loaded onto a device, nothing is trained, and no
solver is invoked.

Usage::

    PYTHONPATH="$PWD/src" python scripts/regenerate/export_series_and_logs.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# provenance/ subdirectories this script writes into. Its ten outputs back
# four different thesis artefacts, so destinations are per output rather than
# one shared --out-dir; listed here so a typo fails at startup, not midway.
DEST_GROUPS = (
    "figure41_training",     # the two training series, and this script's manifest
    "methodology",           # dataset size, linear-scorer loss, benchmark quality
    "appendix_a_sweeps",     # warm-start evals and the step-400 arms
    "appendix_c_benchmark",  # the 300 s re-solve log extract
)

WARM_CKPT = REPO_ROOT / "checkpoints" / "sweep_G_v4warmstart.pt"
COLD_CKPT = REPO_ROOT / "checkpoints" / "sweep_G_coldstart.pt"

NUM_STEPS = 1000

ARMS = [
    ("warm", WARM_CKPT, "series_warm_start_training.csv",
     "Warm start from Method One, production run"),
    ("cold", COLD_CKPT, "series_cold_start_training.csv",
     "Cold start, DM-17"),
]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def repr_float(value) -> str:
    """Full-precision round-trippable float, so an export loses nothing."""
    return repr(float(value))


# Part 1. The two training series

def export_series(out_dir: Path) -> dict:
    report = {}
    for key, ckpt_path, filename, label in ARMS:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cost = np.asarray(ckpt["cost_history"], dtype=float)
        grad = np.asarray(ckpt["grad_norm_history"], dtype=float)
        final_step = int(ckpt["step"])

        if not (len(cost) == len(grad) == final_step == NUM_STEPS):
            raise SystemExit(
                f"{ckpt_path}: expected 1000 entries against a final step of "
                f"1000, got cost {len(cost)}, grad {len(grad)}, step "
                f"{final_step}. Refusing to export a partial series."
            )
        if not (np.isfinite(cost).all() and np.isfinite(grad).all()):
            raise SystemExit(f"{ckpt_path}: non-finite entries in a history.")

        rows = [
            {"step": s + 1,
             "cost_history": repr_float(cost[s]),
             "grad_norm_history": repr_float(grad[s])}
            for s in range(NUM_STEPS)
        ]
        path = out_dir / filename
        write_csv(path, ["step", "cost_history", "grad_norm_history"], rows)

        # Per-step or a ten-step block mean? A block mean would repeat each
        # value ten times: 100 distinct values, 900 consecutive-equal pairs.
        n_unique_grad = int(len(np.unique(grad)))
        n_equal_neighbours = int((np.diff(grad) == 0).sum())
        block_means = grad.reshape(-1, 10).mean(axis=1)

        report[key] = {
            "label": label,
            "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
            "checkpoint_sha256": sha256_of(ckpt_path),
            "csv": str(path.relative_to(REPO_ROOT)),
            "csv_sha256": sha256_of(path),
            "n_rows": len(rows),
            "final_step": final_step,
            "length_is_1000": len(cost) == NUM_STEPS,
            "grad_n_unique": n_unique_grad,
            "grad_n_consecutive_equal": n_equal_neighbours,
            "grad_is_true_per_step": n_unique_grad == NUM_STEPS
                                     and n_equal_neighbours == 0,
            "grad_peak": float(grad.max()),
            "grad_peak_step": int(grad.argmax()) + 1,
            "grad_min": float(grad.min()),
            "grad_largest_10step_block_mean": float(block_means.max()),
            "grad_largest_block_mean_ending_step": int(block_means.argmax() + 1) * 10,
            "grad_mean_first_150": float(grad[:150].mean()),
            "grad_mean_last_150": float(grad[-150:].mean()),
            "cost_step_1": float(cost[0]),
            "cost_step_1000": float(cost[-1]),
            "cost_fall_raw_endpoints": float(cost[0] - cost[-1]),
        }
        print(f"  wrote {path.relative_to(REPO_ROOT)}  {len(rows)} rows, "
              f"length 1000 confirmed")
    return report


def verify_log_is_block_mean_of_history(report: dict) -> dict:
    """Cross-check the warm log against the exported warm series.

    Establishes which carries the per-step value. The log prints one line
    every ten steps; if that value equals the mean of the preceding ten
    history entries and never the single entry at that step, the history is
    per-step and the log is the block mean.
    """
    log_path = REPO_ROOT / "logs" / "sweep_G_v4warmstart.log"
    if not log_path.exists():
        return {"available": False}
    ckpt = torch.load(WARM_CKPT, map_location="cpu", weights_only=False)
    grad = np.asarray(ckpt["grad_norm_history"], dtype=float)
    matches_block, matches_step, total = 0, 0, 0
    for line in log_path.read_text().splitlines():
        m = re.search(r"\[step (\d+)/1000\].*grad_norm=([\d.]+)", line)
        if not m:
            continue
        step, value = int(m.group(1)), float(m.group(2))
        total += 1
        if abs(value - grad[step - 10:step].mean()) < 5e-4:
            matches_block += 1
        if abs(value - grad[step - 1]) < 5e-4:
            matches_step += 1
    return {
        "available": True,
        "log": "logs/sweep_G_v4warmstart.log",
        "logged_values_checked": total,
        "equal_to_10_step_block_mean_of_history": matches_block,
        "equal_to_single_history_entry": matches_step,
        "conclusion": ("the checkpoint history is per-step and the LOG is the "
                       "ten-step block mean of it"
                       if matches_block == total and matches_step == 0
                       else "inconclusive, see counts"),
    }


# Part 2. The log-only claims, one CSV per claim group

def export_il_dataset_size(out_dir: Path) -> dict:
    """C024, C025. Expert dataset size, from logs/il_training_v4.log."""
    log = REPO_ROOT / "logs" / "il_training_v4.log"
    rows = []
    lines = log.read_text().splitlines()
    for i, line in enumerate(lines):
        m = re.search(r"ILDataset loaded (\d+) commit-epoch examples from "
                      r"(\d+) instances", line)
        if not m:
            continue
        examples, instances = int(m.group(1)), int(m.group(2))
        steps = ""
        for back in lines[max(0, i - 2):i]:
            s = re.search(r"Starting IL training with (\d+) steps", back)
            if s:
                steps = s.group(1)
        rows.append({
            "log_line_number": i + 1,
            "training_run_steps": steps,
            "commit_epoch_examples": examples,
            "instances": instances,
            "examples_per_instance": repr_float(examples / instances),
            "is_the_v4_run_behind_C024_C025": int(examples == 14498),
        })
    path = out_dir / "logclaims_il_dataset_size.csv"
    write_csv(path, list(rows[0].keys()), rows)
    v4 = [r for r in rows if r["is_the_v4_run_behind_C024_C025"] == 1]
    print(f"  wrote {path.relative_to(REPO_ROOT)}  {len(rows)} rows")
    return {"csv": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256_of(path), "n_rows": len(rows),
            "claims": ["C024", "C025"],
            "C025_examples": v4[0]["commit_epoch_examples"] if v4 else None,
            "C024_examples_per_instance": v4[0]["examples_per_instance"] if v4 else None,
            "note": "the log holds two IL runs; the 13711-example run is the "
                    "earlier 30000-step run and is NOT the source of C024 or C025"}


def export_milp_resolve(out_dir: Path) -> dict:
    """C031, C032. Re-solve at five times budget, from logs/milp_resolve_300s.log."""
    log = REPO_ROOT / "logs" / "milp_resolve_300s.log"
    text = log.read_text()
    rows = []
    for line in text.splitlines():
        m = re.search(
            r"\[\d+/20\] seed=(\d+) obj_60s=([\d.]+) obj_300s=([\d.]+) "
            r"move=\+([\d.]+)% gap_60s=([\d.]+) gap_300s=([\d.]+) status=(\w+)",
            line)
        if m:
            rows.append({
                "seed": int(m.group(1)),
                "obj_60s": m.group(2), "obj_300s": m.group(3),
                "incumbent_move_pct": m.group(4),
                "gap_60s": m.group(5), "gap_300s": m.group(6),
                "status": m.group(7),
                "sane": "", "greedy_cost": "",
            })
    by_seed = {r["seed"]: r for r in rows}
    # The corrupted-seed block, which is the only source of C032.
    for line in text.splitlines():
        m = re.search(r"seed=(\d+) greedy=([\d.]+) obj_300s=[\d.]+ "
                      r"status=\w+ sane=(\w+)", line)
        if m and int(m.group(1)) in by_seed:
            by_seed[int(m.group(1))]["greedy_cost"] = m.group(2)
            by_seed[int(m.group(1))]["sane"] = m.group(3)
    for r in rows:
        if r["sane"] == "":
            r["sane"] = "True"
    path = out_dir / "logclaims_milp_resolve_300s.csv"
    write_csv(path, list(rows[0].keys()), rows)

    summary = {}
    for key, pattern in [
        ("mean_incumbent_improvement_pct",
         r"Mean incumbent improvement:\s*\+([\d.]+)%"),
        ("max_incumbent_improvement_pct",
         r"Max incumbent improvement:\s*\+([\d.]+)%"),
        ("gap_closure_60s_denominator_pct",
         r"60s incumbents as denominator:\s*([\d.]+)%"),
        ("gap_closure_300s_denominator_pct",
         r"300s incumbents as denominator:\s*([\d.]+)%"),
    ]:
        m = re.search(pattern, text)
        summary[key] = m.group(1) if m else None
    n_eval = re.search(r"denominator:\s*[\d.]+% \(n=(\d+)\)", text)
    summary["n_instances_in_gap_closure"] = int(n_eval.group(1)) if n_eval else None
    summary["gap_closure_movement_points"] = repr_float(
        float(summary["gap_closure_60s_denominator_pct"])
        - float(summary["gap_closure_300s_denominator_pct"]))
    summary["n_insane_seeds"] = sum(1 for r in rows if r["sane"] == "False")
    summary["insane_seeds"] = ";".join(
        str(r["seed"]) for r in rows if r["sane"] == "False")

    # Carried in this manifest entry rather than its own CSV: nothing read that
    # CSV, milp_budget_comparison.csv supersedes it on every figure it held,
    # and writing it would only put an orphan back into provenance/.
    print(f"  wrote {path.relative_to(REPO_ROOT)}  {len(rows)} rows")
    return {"csv": str(path.relative_to(REPO_ROOT)), "sha256": sha256_of(path),
            "n_rows": len(rows), "claims": ["C031", "C032"], **summary}


def export_linear_scorer_loss(out_dir: Path) -> dict:
    """C040. Affine scorer loss per step, matched-hyperparameter run."""
    log = REPO_ROOT / "logs" / "il_convexity_diag_linear_matched.log"
    rows = []
    for line in log.read_text().splitlines():
        m = re.search(r"\[step (\d+)/(\d+)\] loss=(-?[\d.]+) "
                      r"grad_norm=([\d.]+) epsilon=([\d.]+)"
                      r"(?: fy_true=(-?[\d.]+))?", line)
        if m:
            rows.append({
                "step": int(m.group(1)), "total_steps": int(m.group(2)),
                "loss": m.group(3), "grad_norm": m.group(4),
                "epsilon": m.group(5), "fy_true": m.group(6) or "",
            })
    path = out_dir / "logclaims_linear_scorer_loss.csv"
    write_csv(path, list(rows[0].keys()), rows)
    at2500 = [r for r in rows if r["step"] == 2500]
    print(f"  wrote {path.relative_to(REPO_ROOT)}  {len(rows)} rows")
    return {"csv": str(path.relative_to(REPO_ROOT)), "sha256": sha256_of(path),
            "n_rows": len(rows), "claims": ["C040"],
            "step_2500_present": bool(at2500),
            "loss_at_step_2500": at2500[0]["loss"] if at2500 else None,
            "step_range": f"{rows[0]['step']} to {rows[-1]['step']}"}


def export_warmstart_evals(out_dir: Path) -> dict:
    """C043, C044, C096, C097. The in-loop validation evaluations."""
    log = REPO_ROOT / "logs" / "sweep_G_v4warmstart.log"
    text = log.read_text()
    rows = []
    for line in text.splitlines():
        m = re.search(r"\[eval step (\d+)/(\d+)\] hard gap_vs_milp=([\d.]+) "
                      r"served_all_rate=([\d.]+) \(n_val=(\d+)\)", line)
        if m:
            rows.append({
                "step": int(m.group(1)), "total_steps": int(m.group(2)),
                "hard_gap_vs_milp": m.group(3),
                "gap_closure_pct": repr_float(float(m.group(3)) * 100.0),
                "served_all_rate": m.group(4),
                "n_val": int(m.group(5)),
            })
    path = out_dir / "logclaims_sweep_G_warmstart_evals.csv"
    write_csv(path, list(rows[0].keys()), rows)

    cadence = sorted({rows[i + 1]["step"] - rows[i]["step"]
                      for i in range(len(rows) - 1)})
    plateau = [r for r in rows if r["step"] >= 600]
    first50 = next((r for r in rows
                    if float(r["hard_gap_vs_milp"]) >= 0.50), None)
    n_vals = sorted({r["n_val"] for r in rows})
    print(f"  wrote {path.relative_to(REPO_ROOT)}  {len(rows)} rows")
    return {
        "csv": str(path.relative_to(REPO_ROOT)), "sha256": sha256_of(path),
        "n_rows": len(rows), "claims": ["C043", "C044", "C096", "C097"],
        "C043_cadence_steps": cadence,
        "C044_n_evaluations": len(rows),
        "C096_first_step_at_or_above_50pct": first50["step"] if first50 else None,
        "C096_value_there_pct": first50["gap_closure_pct"] if first50 else None,
        "C097_plateau_mean_pct": repr_float(
            np.mean([float(r["gap_closure_pct"]) for r in plateau])),
        "C097_n_evaluations_in_plateau": len(plateau),
        "n_val_values_present": n_vals,
    }


def export_method_two_arms(out_dir: Path) -> dict:
    """The Method Two sensitivity arms, from the overnight chain logs.

    Appendix A compares four Method Two runs at step 400. Production comes from
    ``logs/sweep_G_v4warmstart.log``, already covered by
    ``export_warmstart_evals``. The other three ran under
    ``scripts/experiments/overnight_chain.sh`` stages 3, 4 and 5; their logs
    are gitignored, so their trajectories are exported here.

    The stage 5 arm previously stopped at step 150 in this tree because the
    process died and only the V2 tree held the resumed run; the completed log
    was brought across in the final reconciliation, so this arm now reaches
    step 400, but ``reached_step_400`` is still reported per arm rather than
    assumed.

    Two further gitignored-log arms ask whether a Method One gain survives
    into Method Two, each on a 1000 step budget against the same validation
    split, so the two appendix rows they back stop depending on an untracked
    file.
    """
    ARMS = [
        ("production", "sweep_G_v4warmstart.log",
         "the production Method Two run"),
        ("production_replicate", "overnight_s3_archwinner.log",
         "stage 3, a production replicate because no architecture cell cleared the "
         "selection threshold"),
        ("epsilon_endpoint_035", "overnight_s4_eps035.log",
         "stage 4, epsilon endpoint 0.5 to 0.35"),
        ("k40_batch64", "overnight_s5_k40_b64.log",
         "stage 5, K 20 to 40 and batch 128 to 64 at fixed K times batch"),
        ("archL2H128_warmstart", "m2_archL2H128_warmstart.log",
         "1000 step budget, does the Method One L2_H128 architecture gain survive "
         "into Method Two"),
        ("eps060_warmstart", "m2_eps060_warmstart.log",
         "1000 step budget, does the Method One epsilon floor 0.60 gain survive "
         "into Method Two"),
    ]
    rows_out, summary = [], {}
    for tag, name, note in ARMS:
        log = REPO_ROOT / "logs" / name
        if not log.exists():
            summary[tag] = {"log_present": 0}
            continue
        evals = []
        for line in log.read_text(errors="replace").splitlines():
            m = re.search(r"\[eval step (\d+)/(\d+)\] hard gap_vs_milp=(-?[\d.]+) "
                          r"served_all_rate=([\d.]+) \(n_val=(\d+)\)", line)
            if m:
                evals.append(dict(step=int(m.group(1)), total_steps=int(m.group(2)),
                                  hard_gap_vs_milp=m.group(3),
                                  served_all_rate=m.group(4), n_val=int(m.group(5))))
        at400 = next((e for e in evals if e["step"] == 400), None)
        for e in evals:
            rows_out.append({"arm": tag, "log": name, **e})
        summary[tag] = {
            "log_present": 1, "note": note, "sha256": sha256_of(log),
            "n_evaluations": len(evals),
            "last_step": evals[-1]["step"] if evals else None,
            "reached_step_400": int(at400 is not None),
            "gap_at_step_400": at400["hard_gap_vs_milp"] if at400 else None,
            "last_recorded_gap": evals[-1]["hard_gap_vs_milp"] if evals else None,
        }
    path = out_dir / "logclaims_method_two_step400.csv"
    write_csv(path, list(rows_out[0].keys()), rows_out)
    spath = out_dir / "logclaims_method_two_step400_summary.csv"
    srows = [{"arm": k, **v} for k, v in summary.items()]
    keys = sorted({k for r in srows for k in r})
    write_csv(spath, keys, [{k: r.get(k, "") for k in keys} for r in srows])
    print(f"  wrote {path.relative_to(REPO_ROOT)}  {len(rows_out)} rows")
    print(f"  wrote {spath.relative_to(REPO_ROOT)}  {len(srows)} rows")
    for k, v in summary.items():
        print(f"     {k:<22} step400={v.get('gap_at_step_400')}  "
              f"last={v.get('last_recorded_gap')} at step {v.get('last_step')}")
    return {"csv": str(path.relative_to(REPO_ROOT)), "sha256": sha256_of(path),
            "summary_csv": str(spath.relative_to(REPO_ROOT)),
            "summary_sha256": sha256_of(spath),
            "n_rows": len(rows_out), "arms": summary,
            "claims": ["appendix A Method Two sensitivity"]}


def export_benchmark_quality(out_dir: Path) -> dict:
    """Benchmark solver configuration and quality, summarised out of the label cache.

    Closes C010, C011 and C026 to C029 without a notebook reading 595 MB of
    cache. One row per split, over every record in it.

    Draft 17 reports validation figures on the 197 records that survive the
    lower bound check rather than all 200, so both bases are exported here.
    The three excluded records are all proven optimal at the sixty second
    budget with a zero residual gap, so dropping them lowers the optimal count
    by exactly three and RAISES the mean residual gap; reporting only the 200
    basis is what let Section 3.3 and Chapter 6 disagree with each other.
    """
    excluded = {11044, 11055, 11169}
    rows_out = []
    for split in ("val", "test"):
        d = REPO_ROOT / "cache" / "training_set_il_v3" / split
        if not d.is_dir():
            continue
        limits, targets, gaps, optimal, n = set(), set(), [], 0, 0
        gaps_kept, optimal_kept, n_kept = [], 0, 0
        for p in sorted(d.glob("seed*.json")):
            rec = json.loads(p.read_text())
            n += 1
            limits.add(rec.get("mip_time_limit"))
            targets.add(rec.get("mip_gap_target"))
            sol = rec.get("milp_solution", {})
            is_opt = str(sol.get("status", "")).lower() == "optimal"
            gap = float(sol["mip_gap"]) if sol.get("mip_gap") is not None else None
            if is_opt:
                optimal += 1
            if gap is not None:
                gaps.append(gap)
            if int(rec["seed"]) not in excluded:
                n_kept += 1
                if is_opt:
                    optimal_kept += 1
                if gap is not None:
                    gaps_kept.append(gap)
        rows_out.append({
            "split": split,
            "n_records": n,
            "mip_time_limit_seconds": sorted(limits)[0] if len(limits) == 1 else
                                      ";".join(str(x) for x in sorted(limits)),
            "mip_time_limit_uniform": int(len(limits) == 1),
            "mip_gap_target": sorted(targets)[0] if len(targets) == 1 else
                              ";".join(str(x) for x in sorted(targets)),
            "mip_gap_target_uniform": int(len(targets) == 1),
            "n_proven_optimal": optimal,
            "mean_residual_mip_gap": repr_float(float(np.mean(gaps))) if gaps else "",
            "mean_residual_mip_gap_pct": repr_float(100.0 * float(np.mean(gaps)))
                                          if gaps else "",
            "n_records_kept": n_kept,
            "n_proven_optimal_kept": optimal_kept,
            "mean_residual_mip_gap_kept_pct": repr_float(
                100.0 * float(np.mean(gaps_kept))) if gaps_kept else "",
            "excluded_seeds": ";".join(str(x) for x in sorted(excluded)
                                       if split == "val") or "none",
        })
    if not rows_out:
        print("  cache/training_set_il_v3 absent, benchmark quality not exported")
        return {"available": False}
    path = out_dir / "cache_benchmark_quality.csv"
    write_csv(path, list(rows_out[0].keys()), rows_out)
    print(f"  wrote {path.relative_to(REPO_ROOT)}  {len(rows_out)} rows")
    for r in rows_out:
        print(f"     {r['split']:<5} n={r['n_records']} limit={r['mip_time_limit_seconds']}s "
              f"target={r['mip_gap_target']} optimal={r['n_proven_optimal']} "
              f"mean gap={float(r['mean_residual_mip_gap_pct']):.2f} %")
    return {"csv": str(path.relative_to(REPO_ROOT)), "sha256": sha256_of(path),
            "n_rows": len(rows_out),
            "claims": ["C010", "C011", "C026", "C027", "C028", "C029"]}


# Gitignored inputs, paired with the committed artefact that carries the
# result. Checked before any work so a fresh clone gets one diagnostic line
# rather than a FileNotFoundError from deep in the load.
REQUIRED_INPUTS = [
    ("checkpoints/sweep_G_v4warmstart.pt",
     "provenance/figure41_training/export_series_and_logs_manifest.json"),
    ("checkpoints/sweep_G_coldstart.pt",
     "provenance/figure41_training/export_series_and_logs_manifest.json"),
    ("logs/sweep_G_v4warmstart.log",
     "provenance/figure41_training/export_series_and_logs_manifest.json"),
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="provenance")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    require_inputs()

    # provenance/ groups files by the thesis artefact they back, and this
    # script's ten outputs back four different ones, so each exporter gets its
    # own destination rather than sharing a single --out-dir.
    base = REPO_ROOT / args.out_dir
    dest = {group: base / group for group in DEST_GROUPS}
    for path in dest.values():
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        dest["figure41_training"] / "export_series_and_logs_manifest.json"
    )
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"REFUSING to overwrite {manifest_path}. Pass --force.")

    print("=== 1. Per-step training series from the two rolling checkpoints ===")
    series = export_series(dest["figure41_training"])

    print("\n=== 2. Is the stored gradient norm per-step or a block mean? ===")
    block_check = verify_log_is_block_mean_of_history(series)
    print(json.dumps(block_check, indent=1))

    print("\n=== 3. Log-only claims, one CSV per claim group ===")
    groups = {
        "il_dataset_size": export_il_dataset_size(dest["methodology"]),
        "milp_resolve_300s": export_milp_resolve(dest["appendix_c_benchmark"]),
        "linear_scorer_loss": export_linear_scorer_loss(dest["methodology"]),
        "sweep_G_warmstart_evals": export_warmstart_evals(dest["appendix_a_sweeps"]),
        "method_two_step400": export_method_two_arms(dest["appendix_a_sweeps"]),
        "benchmark_quality": export_benchmark_quality(dest["methodology"]),
    }
    covered = sorted({c for g in groups.values() for c in g["claims"]})
    print(f"\n  claims now covered by a CSV: {', '.join(covered)} "
          f"({len(covered)} claims across {len(groups)} groups)")

    payload = {
        "script": "scripts/regenerate/export_series_and_logs.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "series": series,
        "per_step_or_block_mean": block_check,
        "log_claim_groups": groups,
        "claims_covered": covered,
    }
    with manifest_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote {manifest_path.relative_to(REPO_ROOT)}")
    print("No checkpoint, log or existing provenance file was modified.")


if __name__ == "__main__":
    main()
