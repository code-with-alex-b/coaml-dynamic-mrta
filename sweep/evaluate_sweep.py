"""Evaluate the hyperparameter sweep on the VALIDATION set.

Uses the programmatic interface of ``method_two_evaluator``
(``load_scorer_from_checkpoint``, ``load_records``, ``evaluate_one_instance``,
``summarize``) directly; it never shells out to the evaluator CLI.

For each config in ``sweep/configs.json`` it inspects the sweep checkpoint, and
when present evaluates the scorer on the validation split under hard and
perturbed inference. The configs are ranked by hard serve-all rate (gap closure
as tiebreaker); pending and in-progress configs are listed unranked at the
bottom. The table is printed to stdout and written to ``logs/sweep_results.txt``.

Run with PYTHONPATH pointed at src/, e.g.::

    PYTHONPATH="$PWD/src" python sweep/evaluate_sweep.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from evaluation.method_two_evaluator import (
    load_scorer_from_checkpoint,
    load_records,
    evaluate_one_instance,
    summarize,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_PATH = Path(__file__).resolve().parent / "configs.json"
RESULTS_PATH = REPO_ROOT / "logs" / "sweep_results.txt"

MODES = ["hard", "perturbed"]
EVAL_WEIGHTS = {"w_dist": 0.0637, "w_make": 0.2398, "w_bal": 0.6965}
# Perturbed-decode epsilon comes from configs.json (shared["epsilon"]); the single-checkpoint path can override it via --epsilon.

# One CSV row per (instance, decode mode); with the two default modes that is two rows per instance.
PER_INSTANCE_COLUMNS = [
    "seed",
    "decode_mode",
    "policy_cost",
    "cost_hungarian_distance_only",
    "cost_hungarian_kappa_weighted",
    "cost_milp_oracle",
    "gap_closure_vs_milp",
    "gap_closure_secondary",
    "term_travel_time",
    "term_makespan",
    "term_balance",
    "serve_all_flag",
    "sim_time_at_termination",
    "epoch_count",
    "per_robot_task_counts",
    # The aggregate path drops a non-serving rollout's cost; these columns keep the instance visible with its cost recorded.
    "cost_including_failed",
    "cost_is_from_failed_rollout",
    "n_unserved_tasks",
    # So both Hungarian variants carry the same decomposition the learned policy already had.
    "term_travel_time_hungarian_distance",
    "term_makespan_hungarian_distance",
    "term_balance_hungarian_distance",
    "term_travel_time_hungarian_kappa",
    "term_makespan_hungarian_kappa",
    "term_balance_hungarian_kappa",
]


def _cell(value):
    """Render one CSV cell. None becomes an empty cell, never a substitute."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return value


def _per_instance_rows(record, modes):
    """Build the CSV rows for one already-evaluated instance record.

    Reads only values the evaluator has already produced for this record. It
    computes no cost, rolls out nothing, and draws no random number.
    """
    rows = []
    gt = record.get("greedy_terms", {})
    kt = record.get("kappa_terms", {})
    for mode in modes:
        detail = record.get(f"{mode}_detail", {})
        rows.append([
            _cell(record["seed"]),
            _cell(mode),
            _cell(record.get(f"{mode}_cost")),
            _cell(record["greedy"]),
            _cell(record.get("kappa")),
            _cell(record["milp"]),
            _cell(record.get(f"{mode}_gap_vs_milp")),
            _cell(record.get(f"{mode}_gap_vs_expert")),
            _cell(detail.get("term_travel_time")),
            _cell(detail.get("term_makespan")),
            _cell(detail.get("term_balance")),
            _cell(record.get(f"{mode}_served_all")),
            _cell(detail.get("sim_time_at_termination")),
            _cell(detail.get("epoch_count")),
            _cell(detail.get("per_robot_task_counts")),
            _cell(detail.get("cost_including_failed")),
            _cell(detail.get("cost_is_from_failed_rollout")),
            _cell(detail.get("n_unserved_tasks")),
            _cell(gt.get("term_travel_time")),
            _cell(gt.get("term_makespan")),
            _cell(gt.get("term_balance")),
            _cell(kt.get("term_travel_time")),
            _cell(kt.get("term_makespan")),
            _cell(kt.get("term_balance")),
        ])
    return rows


def _pct(fraction):
    """Format a 0-1 fraction as a percentage string, or '-' for None."""
    if fraction is None:
        return "-"
    return f"{fraction * 100:.1f}%"


def _fmt(value, prec=3):
    """Format a float, or '-' for None."""
    if value is None:
        return "-"
    return f"{value:.{prec}f}"


def evaluate_scorer_on_val(scorer, shared, modes=MODES, per_instance_out=None):
    """Per-instance validation evaluation of one scorer, returning the summary.

    Shared by the full sweep loop and the single-checkpoint path so both use
    identical records, the same eval seed, weights, and epsilon.

    ``per_instance_out``, when given, is a path to write one CSV row per
    (instance, decode mode). It is additive: with it absent the evaluation is
    unchanged, and with it present the only extra work is a deterministic,
    torch-RNG-free kappa-Hungarian rollout plus reads off the simulator, so
    every returned aggregate is identical either way.
    """
    # Anchor a relative cache path to the repo root so the script behaves the same from any working directory (train_one.py does the same).
    cache_dir = Path(shared["val_cache"])
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    n_requested = int(shared["eval_instances"])
    records = load_records(str(cache_dir), n_requested)
    if len(records) < n_requested:
        raise SystemExit(
            f"Expected {n_requested} validation records in {cache_dir} but "
            f"found {len(records)}. Refusing to report metrics on a partial "
            f"or empty validation set."
        )
    # Seed the Gumbel draws so perturbed inference is reproducible.
    torch.manual_seed(int(shared["eval_seed"]))
    epsilon = float(shared["epsilon"])
    collect = per_instance_out is not None
    if not collect:
        per_instance = [
            evaluate_one_instance(scorer, rec, modes, EVAL_WEIGHTS, epsilon)
            for rec in records
        ]
        return summarize(per_instance, modes)

    out_path = Path(per_instance_out)
    if out_path.exists():
        raise SystemExit(
            f"REFUSING to overwrite an existing file: {out_path}. "
            "Choose a new --per-instance-out path."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_instance = []
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(PER_INSTANCE_COLUMNS)
        for rec in records:
            result = evaluate_one_instance(
                scorer, rec, modes, EVAL_WEIGHTS, epsilon,
                collect_detail=True,
            )
            per_instance.append(result)
            # Written inside the iteration that produced it, so a row cannot be misaligned with its seed.
            for row in _per_instance_rows(result, modes):
                writer.writerow(row)
    print(f"Wrote per-instance rows to {out_path}", flush=True)
    return summarize(per_instance, modes)


def evaluate_config(entry, shared):
    """Return a result dict for one config entry.

    Keys: id, k, batch, bk, status, done (bool), ranked (bool), step,
    target_steps, hard_serve_all (str), hard_serve_rate (float|None),
    gap_closure (float|None), perturbed_gap (float|None).
    """
    config_id = entry["id"]
    target_steps = int(shared["steps"])
    ckpt_path = REPO_ROOT / "checkpoints" / f"sweep_bk_grid_{config_id}.pt"

    base = {
        "id": config_id,
        "k": entry["k"],
        "batch": entry["batch"],
        "bk": entry["bk"],
        "target_steps": target_steps,
        "step": None,
        "hard_serve_all": "-",
        "hard_serve_rate": None,
        "gap_closure": None,
        "perturbed_gap": None,
    }

    if not ckpt_path.exists():
        base.update(status="Pending", done=False, ranked=False)
        return base

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = int(ckpt.get("step", 0))
    base["step"] = step
    done = step >= target_steps
    status = "Done" if done else f"In progress (step {step}/{target_steps})"

    scorer = load_scorer_from_checkpoint(str(ckpt_path))
    summary = evaluate_scorer_on_val(scorer, shared)

    n = summary["n_evaluated"]
    hard = summary["modes"]["hard"]
    pert = summary["modes"]["perturbed"]

    base.update(
        status=status,
        done=done,
        ranked=done,
        step=step,
        hard_serve_all=f"{hard['n_served_all']}/{n} ({hard['served_all_rate'] * 100:.0f}%)",
        hard_serve_rate=hard["served_all_rate"],
        gap_closure=hard["serving_gap_vs_milp"]["mean"],
        perturbed_gap=pert["serving_gap_vs_milp"]["mean"],
    )
    return base


def build_table(results):
    """Render the ranked summary table as a list of text lines."""
    header = (
        f"{'Rank':<5}| {'Config':<6}| {'K':<3}| {'Batch':<6}| {'B*K':<5}| "
        f"{'Hard serve-all':<16}| {'Gap closure':<12}| {'Perturbed gap':<14}| "
        f"{'Steps':<10}| Status"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    ranked = [r for r in results if r["ranked"]]
    unranked = [r for r in results if not r["ranked"]]

    # Ranked: hard serve-all rate desc, gap closure desc as tiebreaker.
    ranked.sort(
        key=lambda r: (
            r["hard_serve_rate"] if r["hard_serve_rate"] is not None else -1,
            r["gap_closure"] if r["gap_closure"] is not None else -1,
        ),
        reverse=True,
    )

    def steps_str(r):
        if r["step"] is None:
            return "-"
        return f"{r['step']}/{r['target_steps']}"

    def row(rank_label, r):
        return (
            f"{rank_label:<5}| {r['id']:<6}| {r['k']:<3}| {r['batch']:<6}| "
            f"{r['bk']:<5}| {r['hard_serve_all']:<16}| "
            f"{_pct(r['gap_closure']):<12}| {_pct(r['perturbed_gap']):<14}| "
            f"{steps_str(r):<10}| {r['status']}"
        )

    for i, r in enumerate(ranked, start=1):
        lines.append(row(str(i), r))
    for r in unranked:
        lines.append(row("", r))

    if ranked:
        w = ranked[0]
        n_served = (
            w["hard_serve_all"].split(" ")[0] if w["hard_serve_all"] != "-" else "-"
        )
        lines.append("")
        lines.append(
            f"Winner: Config {w['id']} (K={w['k']}, batch={w['batch']}, "
            f"B*K={w['bk']}) -- {_pct(w['gap_closure'])} gap closure, "
            f"{n_served} hard serve-all"
        )

    return lines


def _print_full_precision(summary) -> None:
    """Print every aggregate at full float precision, unrounded.

    DM-38 recorded only the ``:.1f`` percentages, so the exact values behind
    the headline figure were lost the moment the process exited. Every number
    here goes through ``repr``, which round-trips a Python float exactly, so
    the printed text is the value and not a rendering of it. This is a printer
    only: it computes nothing and changes nothing.
    """
    print("\n  FULL PRECISION AGGREGATES (repr, exact round-trip):", flush=True)
    print(f"    n_evaluated                       = {summary['n_evaluated']!r}", flush=True)
    print(f"    mean_greedy                       = {summary['mean_greedy']!r}", flush=True)
    print(f"    mean_milp                         = {summary['mean_milp']!r}", flush=True)
    print(f"    mean_expert                       = {summary['mean_expert']!r}", flush=True)
    for mode in sorted(summary["modes"]):
        m = summary["modes"][mode]
        print(f"    [{mode}]", flush=True)
        print(f"      n_success                       = {m['n_success']!r}", flush=True)
        print(f"      success_rate                    = {m['success_rate']!r}", flush=True)
        print(f"      n_served_all                    = {m['n_served_all']!r}", flush=True)
        print(f"      served_all_rate                 = {m['served_all_rate']!r}", flush=True)
        for key in ("serving_cost", "serving_gap_vs_milp", "serving_gap_vs_expert"):
            s = m[key]
            print(f"      {key}.n                 = {s['n']!r}", flush=True)
            print(f"      {key}.mean              = {s['mean']!r}", flush=True)
            print(f"      {key}.std               = {s['std']!r}", flush=True)
            print(f"      {key}.p25               = {s['p25']!r}", flush=True)
            print(f"      {key}.p50               = {s['p50']!r}", flush=True)
            print(f"      {key}.p75               = {s['p75']!r}", flush=True)


def evaluate_single_checkpoint(checkpoint_path, shared, per_instance_out=None) -> None:
    """Evaluate one checkpoint on the validation split and print its metrics.

    Skips the config loop. Reuses the same per-instance evaluation as the sweep
    (``evaluate_scorer_on_val``) and prints the hard-decode gap closure versus
    the MILP bound, the serve-all rate, and the raw mean costs for the
    Hungarian greedy baseline, the MILP, and the learned policy.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    step = int(ckpt.get("step", 0))
    scorer = load_scorer_from_checkpoint(str(ckpt_path))
    summary = evaluate_scorer_on_val(
        scorer, shared, per_instance_out=per_instance_out
    )

    n = summary["n_evaluated"]
    hard = summary["modes"]["hard"]
    pert = summary["modes"]["perturbed"]

    print(f"\nCheckpoint: {ckpt_path} (step {step})", flush=True)
    # Names the cache actually evaluated; a hardcoded "validation instances" would mislead when --cache-dir is passed.
    print(
        f"Evaluated {n} instances from {shared['val_cache']} "
        f"(eval_seed={shared['eval_seed']}, perturbed epsilon={shared['epsilon']})",
        flush=True,
    )
    print(
        f"  gap_vs_milp (hard):      {_pct(hard['serving_gap_vs_milp']['mean'])}",
        flush=True,
    )
    print(
        f"  gap_vs_milp (perturbed): {_pct(pert['serving_gap_vs_milp']['mean'])}",
        flush=True,
    )
    print(
        f"  served_all_rate (hard):  {hard['n_served_all']}/{n} "
        f"({_pct(hard['served_all_rate'])})",
        flush=True,
    )
    print("  raw mean costs:", flush=True)
    print(f"    hungarian (greedy):           {_fmt(summary['mean_greedy'])}", flush=True)
    print(f"    milp:                         {_fmt(summary['mean_milp'])}", flush=True)
    print(
        f"    learned (hard, serving-only): {_fmt(hard['serving_cost']['mean'])}",
        flush=True,
    )
    _print_full_precision(summary)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate the method-two sweep on the validation set."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Evaluate only this single checkpoint and skip the config loop.",
    )
    parser.add_argument(
        "--eval-instances", "--eval_instances", dest="eval_instances",
        type=int, default=None,
        help="Number of validation instances (single-checkpoint mode; "
        "default uses the shared configs.json value).",
    )
    parser.add_argument(
        "--eval-seed", "--eval_seed", dest="eval_seed", type=int, default=None,
        help="Seed for perturbed-inference Gumbel draws (single-checkpoint "
        "mode; default uses the shared configs.json value).",
    )
    parser.add_argument(
        "--cache-dir", "--cache_dir", dest="cache_dir", type=str, default=None,
        help="Validation cache directory (single-checkpoint mode; default "
        "uses the shared configs.json val_cache).",
    )
    parser.add_argument(
        "--epsilon", dest="epsilon", type=float, default=None,
        help="Perturbed-decode epsilon (single-checkpoint mode; default uses "
        "the shared configs.json epsilon).",
    )
    parser.add_argument(
        "--per-instance-out", "--per_instance_out", dest="per_instance_out",
        type=str, default=None,
        help="Write one CSV row per (instance, decode mode) to this path "
        "(single-checkpoint mode). Refuses to overwrite an existing file. "
        "Omitting it leaves behaviour exactly as before.",
    )
    args = parser.parse_args()

    with CONFIGS_PATH.open("r") as f:
        spec = json.load(f)
    shared = spec["shared"]

    if args.checkpoint is not None:
        # CLI overrides apply to single-checkpoint evaluation only; the full sweep loop is unaffected.
        shared = dict(shared)
        if args.eval_instances is not None:
            shared["eval_instances"] = args.eval_instances
        if args.eval_seed is not None:
            shared["eval_seed"] = args.eval_seed
        if args.cache_dir is not None:
            shared["val_cache"] = args.cache_dir
        if args.epsilon is not None:
            shared["epsilon"] = args.epsilon
        evaluate_single_checkpoint(
            args.checkpoint, shared, per_instance_out=args.per_instance_out
        )
        return

    sweep_name = spec["sweep_name"]

    print(f"Evaluating sweep '{sweep_name}' ({len(spec['configs'])} configs)", flush=True)

    results = []
    for entry in spec["configs"]:
        print(f"  config {entry['id']}: checking checkpoint...", flush=True)
        results.append(evaluate_config(entry, shared))

    lines = build_table(results)
    text = "\n".join(lines)
    print("\n" + text, flush=True)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        f.write(text + "\n")
    print(f"\nWrote results to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
