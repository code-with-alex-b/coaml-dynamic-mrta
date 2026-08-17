"""Zero-shot transfer evaluation at four scales, hard decode, per-instance CSV.

Runs the instrumented method-two evaluator (``evaluate_one_instance`` with
``collect_detail`` on) over each scale's cache and writes one CSV per scale.

Every evaluated instance appears in the CSV. A rollout that failed to serve
every task is written with ``serve_all_flag`` zero and its cost still recorded,
because the evaluator's aggregate path sets ``{mode}_cost`` to None unless all
tasks are served, and dropping those rows would score the policy on a
self-selected easier subset. The recorded cost for such a row comes from
``hard_detail.cost_including_failed`` and is flagged
``cost_is_from_failed_rollout``.

Only the R=6, T=18 reference scale has an offline anticipative MILP stored per
record. The transfer caches carry instances only, so ``milp_objective`` is
empty there and gap closure is undefined and must not be reported.

Inference only. No training, no cache writes, hard decode (epsilon 0) so no
randomness is drawn.

Usage::

    PYTHONPATH="$PWD/src" python scripts/experiments/transfer_table_eval.py \
        --out-dir provenance/transfer_table_20260802
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from evaluation.method_two_evaluator import evaluate_one_instance, load_records
from evaluation.method_one_evaluator import load_scorer_from_checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CKPT = "checkpoints/sweep_G_v4warmstart_best.pt"
CKPT_SHA = "c0ad09e854564c0e32752df3a9cd680a39ec00361d4b6868e785c6f62f366f20"
WEIGHTS = {"w_dist": 0.0637, "w_make": 0.2398, "w_bal": 0.6965}

# ``group`` separates the ratio-matched size-scaling series from the density row so the two are never read as one series.
SCALES = [
    {"key": "r6t18", "R": 6, "T": 18, "ratio": 3.0, "group": "size",
     "cache": "cache/training_set_il_v3/val", "n": 200,
     "label": "R=6, T=18 (training scale, reference)"},
    {"key": "r10t30", "R": 10, "T": 30, "ratio": 3.0, "group": "size",
     "cache": "cache/dm31a_r10t30", "n": 60,
     "label": "R=10, T=30 (size only)"},
    {"key": "r30t90", "R": 30, "T": 90, "ratio": 3.0, "group": "size",
     "cache": "cache/dm31b_r30t90", "n": 60,
     "label": "R=30, T=90 (size only, fivefold)"},
    {"key": "r10t60", "R": 10, "T": 60, "ratio": 6.0, "group": "density",
     "cache": "cache/training_set_rh_r10t60", "n": 200,
     "label": "R=10, T=60 (size and density together)"},
]

CSV_FIELDS = [
    "seed", "R", "T",
    "policy_cost", "hungarian_distance_only_cost", "hungarian_kappa_cost",
    "milp_objective",
    "serve_all_flag", "n_unserved_tasks",
    "term_travel_time", "term_makespan", "term_balance",
    "n_decisions", "n_commitments",
    "cost_is_from_failed_rollout",
    "greedy_served_all", "greedy_n_unserved",
    "kappa_served_all", "kappa_n_unserved",
    "greedy_term_travel_time", "greedy_term_makespan", "greedy_term_balance",
    "kappa_term_travel_time", "kappa_term_makespan", "kappa_term_balance",
    "expert_replay_cost", "epoch_count", "simulator_failure",
]

FORBIDDEN_SEEDS = range(11200, 11400)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _n_commitments(detail) -> int:
    """Assignments the rollout committed.

    ``per_robot_task_counts`` is the per-robot commitment tally the evaluator
    already recorded, so its sum is the commitment count without re-running
    anything. This is NOT the decision count: a decision is one dispatcher
    invocation and may commit zero, one or several assignments, and a committed
    task can still finish after the wall-clock cap and so count as unserved.
    The decision count is ``epoch_count``, the number of ``step`` calls, which
    is the same quantity the DM-40 timing harness counts per instance.
    """
    counts = detail.get("per_robot_task_counts", "")
    return sum(int(x) for x in counts.split(";") if x != "")


def run_scale(scorer, scale: dict, out_dir: Path) -> dict:
    cache_dir = REPO_ROOT / scale["cache"]
    if any(p.lower() == "test" for p in Path(scale["cache"]).parts):
        raise SystemExit(f"REFUSING to evaluate a test split ({scale['cache']})")

    records = load_records(cache_dir, scale["n"])
    if not records:
        raise SystemExit(f"No records under {cache_dir}")

    seeds = [int(r.get("seed", -1)) for r in records]
    clashes = [s for s in seeds if s in FORBIDDEN_SEEDS]
    if clashes:
        raise SystemExit(
            f"REFUSING: {scale['key']} touches held-out seeds {clashes[:10]}"
        )
    for r in records:
        if int(r["R"]) != scale["R"] or int(r["T"]) != scale["T"]:
            raise SystemExit(
                f"{scale['key']}: cache record seed={r.get('seed')} is "
                f"R={r['R']} T={r['T']}, expected R={scale['R']} T={scale['T']}"
            )

    rows, evaluated = [], []
    for i, rec in enumerate(records):
        res = evaluate_one_instance(
            scorer, rec, ["hard"], WEIGHTS, epsilon=0.0,
            num_perturbed_samples=1, collect_detail=True,
        )
        d = res["hard_detail"]
        served_all = bool(res["hard_served_all"])
        # cost is None whenever the rollout dropped a task; fall back to the failed rollout's recorded cost so the row is never dropped.
        cost = res["hard_cost"]
        if cost is None:
            cost = d.get("cost_including_failed")

        rows.append({
            "seed": res["seed"], "R": scale["R"], "T": scale["T"],
            "policy_cost": repr(cost) if cost is not None else "",
            "hungarian_distance_only_cost": repr(res["greedy"]),
            "hungarian_kappa_cost": repr(res["kappa"]),
            "milp_objective": repr(res["milp"]) if res["milp"] is not None else "",
            "serve_all_flag": int(served_all),
            "n_unserved_tasks": d["n_unserved_tasks"],
            "term_travel_time": repr(d["term_travel_time"]) if d["term_travel_time"] is not None else "",
            "term_makespan": repr(d["term_makespan"]) if d["term_makespan"] is not None else "",
            "term_balance": repr(d["term_balance"]) if d["term_balance"] is not None else "",
            "n_decisions": d["epoch_count"],
            "n_commitments": _n_commitments(d),
            "cost_is_from_failed_rollout": int(bool(d["cost_is_from_failed_rollout"])),
            "greedy_served_all": int(bool(res["greedy_served_all"])),
            "greedy_n_unserved": res["greedy_n_unserved"],
            "kappa_served_all": int(bool(res["kappa_served_all"])),
            "kappa_n_unserved": res["kappa_n_unserved"],
            "greedy_term_travel_time": repr(res["greedy_terms"]["term_travel_time"]),
            "greedy_term_makespan": repr(res["greedy_terms"]["term_makespan"]),
            "greedy_term_balance": repr(res["greedy_terms"]["term_balance"]),
            "kappa_term_travel_time": repr(res["kappa_terms"]["term_travel_time"]),
            "kappa_term_makespan": repr(res["kappa_terms"]["term_makespan"]),
            "kappa_term_balance": repr(res["kappa_terms"]["term_balance"]),
            "expert_replay_cost": repr(res["expert"]) if res["expert"] is not None else "",
            "epoch_count": d["epoch_count"],
            "simulator_failure": int(bool(d["simulator_failure"])),
        })
        evaluated.append(res)
        print(
            f"[{scale['key']} {i + 1}/{len(records)}] seed={res['seed']} "
            f"policy={'FAIL' if not served_all else f'{cost:.2f}'} "
            f"dist_only={res['greedy']:.2f} kappa={res['kappa']:.2f} "
            f"unserved={d['n_unserved_tasks']}",
            flush=True,
        )

    csv_path = out_dir / f"transfer_{scale['key']}_per_instance.csv"
    if csv_path.exists():
        raise SystemExit(f"REFUSING to overwrite {csv_path}")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_served = sum(r["serve_all_flag"] for r in rows)
    failures = [
        {"seed": r["seed"], "n_unserved_tasks": r["n_unserved_tasks"]}
        for r in rows if not r["serve_all_flag"]
    ]
    meta = {
        **{k: scale[k] for k in ("key", "R", "T", "ratio", "group", "label", "cache")},
        "n_instances": len(rows),
        "seed_min": min(seeds), "seed_max": max(seeds),
        "seeds_contiguous": (max(seeds) - min(seeds) + 1) == len(seeds),
        "has_milp_benchmark": rows[0]["milp_objective"] != "",
        "has_expert_replay": rows[0]["expert_replay_cost"] != "",
        "n_serve_all": n_served,
        "serve_all_rate": n_served / len(rows),
        "policy_failures": failures,
        "greedy_n_failures": sum(1 for r in rows if not r["greedy_served_all"]),
        "kappa_n_failures": sum(1 for r in rows if not r["kappa_served_all"]),
        "csv": str(csv_path.relative_to(REPO_ROOT)),
    }
    print(
        f"\n{scale['label']}: serve-all {n_served}/{len(rows)}, "
        f"{len(failures)} policy failures, "
        f"dist-only failures {meta['greedy_n_failures']}, "
        f"kappa failures {meta['kappa_n_failures']}\n",
        flush=True,
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=CKPT)
    args = ap.parse_args()

    ckpt_path = REPO_ROOT / args.checkpoint
    sha_start = _sha256(ckpt_path)
    if sha_start != CKPT_SHA:
        raise SystemExit(
            f"ABORT: checkpoint sha256 {sha_start} != expected {CKPT_SHA}"
        )
    print(f"checkpoint sha256 (start) = {sha_start}  OK", flush=True)

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scorer = load_scorer_from_checkpoint(str(ckpt_path))
    scorer.eval()

    metas = []
    with torch.no_grad():
        for scale in SCALES:
            metas.append(run_scale(scorer, scale, out_dir))

    sha_end = _sha256(ckpt_path)
    print(f"checkpoint sha256 (end)   = {sha_end}  "
          f"{'OK' if sha_end == CKPT_SHA else 'CHANGED'}", flush=True)

    inv_path = out_dir / "phase0_inventory.json"
    if inv_path.exists():
        raise SystemExit(f"REFUSING to overwrite {inv_path}")
    with inv_path.open("w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "checkpoint_sha256_start": sha_start,
            "checkpoint_sha256_end": sha_end,
            "weights": WEIGHTS,
            "decode": "hard (epsilon=0)",
            "held_out_seeds_untouched": "11200-11399",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "proc_translated": subprocess.run(
                ["sysctl", "-n", "sysctl.proc_translated"],
                capture_output=True, text=True).stdout.strip() or "unset",
            "scales": metas,
        }, f, indent=2)
    print(f"wrote {inv_path}", flush=True)


if __name__ == "__main__":
    main()
