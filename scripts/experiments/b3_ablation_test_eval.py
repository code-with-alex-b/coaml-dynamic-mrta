"""B3. Evaluate one ablation checkpoint on the frozen test split, hard decode.

Fills the pending ablation rows of Table 1. One checkpoint per invocation, one
pass, hard decode only. No solver is invoked, nothing is trained, and no cache
record is written.

Failure handling. The evaluator's aggregate path discards a non-serving
rollout's cost, which would score a weak policy on a self-selected easier
subset. This script writes every instance to the CSV regardless, with
``serve_all_flag`` false and the cost recorded in ``cost_including_failed``,
so the excluded instances stay visible. ``policy_cost`` keeps the evaluator's
own convention and is blank on a failed rollout, so both readings are available
side by side.

Usage::

    PYTHONPATH="$PWD/src" python scripts/experiments/b3_ablation_test_eval.py \
        --checkpoint checkpoints/sweep_G_coldstart_best.pt \
        --scorer-type gnn --label coldstart \
        --cache-dir cache/training_set_il_v3/test --allow-test-split \
        --out provenance/b3_coldstart_test.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "sweep"))

from evaluation.method_two_evaluator import (  # noqa: E402
    evaluate_one_instance,
    load_records,
)
from scoring.gnn_scorer import GNNScorer  # noqa: E402
from scoring.linear_scorer import LinearScorer  # noqa: E402
from evaluate_sweep import (  # noqa: E402
    EVAL_WEIGHTS,
    PER_INSTANCE_COLUMNS,
    _per_instance_rows,
)


def load_scorer(path: str, scorer_type: str):
    """Rebuild the right architecture and load its state dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cls = {"gnn": GNNScorer, "linear": LinearScorer}[scorer_type]
    scorer = cls()
    scorer.load_state_dict(ckpt["model_state_dict"])
    scorer.eval()
    return scorer, int(ckpt.get("step", -1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scorer-type", choices=("gnn", "linear"), default="gnn")
    ap.add_argument("--label", required=True)
    ap.add_argument("--cache-dir", default="cache/training_set_il_v3/val")
    ap.add_argument("--eval-instances", type=int, default=200)
    ap.add_argument("--eval-seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-test-split", action="store_true")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if any(p.lower() == "test" for p in cache_dir.parts):
        if not args.allow_test_split:
            raise SystemExit(
                f"REFUSING to evaluate on a test split ({cache_dir}). "
                "Pass --allow-test-split to override deliberately."
            )
        print(f"*** TEST SPLIT OPT-IN: {cache_dir} ***", flush=True)

    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"REFUSING to overwrite {out_path}")

    records = load_records(str(REPO_ROOT / cache_dir), args.eval_instances)
    if len(records) < args.eval_instances:
        raise SystemExit(
            f"Expected {args.eval_instances} records, found {len(records)}."
        )

    scorer, step = load_scorer(str(REPO_ROOT / args.checkpoint), args.scorer_type)
    # Hard decode draws no randomness, but seed anyway for the same reproducibility convention as every other evaluation.
    torch.manual_seed(args.eval_seed)
    print(
        f"{args.label}: {args.checkpoint} (step {step}, {args.scorer_type}), "
        f"{len(records)} instances from {cache_dir}, hard decode, "
        f"eval_seed={args.eval_seed}",
        flush=True,
    )

    modes = ["hard"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_served = 0
    t0 = time.perf_counter()
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PER_INSTANCE_COLUMNS)
        for i, rec in enumerate(records):
            r = evaluate_one_instance(
                scorer, rec, modes, EVAL_WEIGHTS, 1.0, collect_detail=True
            )
            for row in _per_instance_rows(r, modes):
                w.writerow(row)
            f.flush()
            served = bool(r["hard_served_all"])
            n_served += int(served)
            if (i + 1) % 25 == 0 or not served:
                d = r["hard_detail"]
                cost = r.get("hard_cost")
                shown = cost if cost is not None else d.get("cost_including_failed")
                print(
                    f"  [{i+1}/{len(records)}] seed={r['seed']} "
                    f"served_all={served} cost={shown:.3f} "
                    f"unserved={d.get('n_unserved_tasks')}",
                    flush=True,
                )
    el = time.perf_counter() - t0
    print(
        f"\n{args.label}: served_all {n_served}/{len(records)}, "
        f"wall {el:.1f}s, wrote {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
