"""A6. Per-decision inference latency of the learned policy at R=6, T=18.

Measures the four stages that make up one dispatcher decision at inference:

    1. feature construction   build_features(state, instance)
    2. graph build            _BipartiteGNNCore._build_edge_index(R, T, device)
    3. GNN forward            projections, SAGEConv stack, pairwise scorer and
                              the augmented-matrix assembly
    4. Hungarian solve        _decode_action, which is the negate/detach/numpy
                              conversion plus scipy linear_sum_assignment plus
                              the extraction of valid picks

The policy's per-decision cost is not the forward pass alone, so all four are
timed separately and the whole decision is timed independently so the sum can be
checked against it.

This script does not modify the inference path. It calls exactly the callables
``GNNScorer.forward`` and ``rollout_policy`` call, in the same order, and it
asserts on every decision that the decomposed Theta is bitwise identical to the
Theta the unmodified ``scorer(state, instance)`` returns. The simulator is
always advanced by the unmodified path.

Validation split only. Hard decode only, epsilon 0, which draws no randomness.

Usage::

    PYTHONPATH="$PWD/src" python scripts/experiments/a6_inference_timing.py \
        --cache-dir cache/training_set_il_v3/val \
        --instances 60 --min-decisions 2000 --warmup 40 \
        --out results/a6_inference_timing.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
import torch
import torch.nn.functional as F

from evaluation.method_one_evaluator import (
    _decode_action,
    load_scorer_from_checkpoint,
)
from instances.synthetic_generator import SyntheticInstance
from scoring.gnn_scorer import build_features, _BipartiteGNNCore
from simulator.dynamic_simulator import DynamicSimulator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE = "cache/training_set_il_v3/val"
DEFAULT_CKPT = "checkpoints/sweep_G_v4warmstart_best.pt"

CSV_FIELDS = [
    "seed", "decision_index", "n_pending", "n_available", "n_committed",
    "t_features_s", "t_graph_build_s", "t_gnn_forward_s", "t_hungarian_s",
    "t_stages_sum_s", "t_total_s", "t_discrepancy_s",
]


def _decomposed(scorer, state, instance):
    """Run one decision stage by stage, returning (theta, timings dict).

    Mirrors ``GNNScorer.forward`` exactly, except that the edge index is built
    outside the encoder so it can be timed on its own. The ``.to(param_device)``
    hop in the real forward is a no-op here because both are CPU.
    """
    enc = scorer.encoder
    R, T = int(instance.R), int(instance.T)

    t0 = time.perf_counter()
    robot_feats, task_feats = build_features(state, instance)
    t1 = time.perf_counter()

    edge_index = _BipartiteGNNCore._build_edge_index(R, T, robot_feats.device)
    t2 = time.perf_counter()

    h_r = enc.robot_proj(robot_feats)
    h_t = enc.task_proj(task_feats)
    x = torch.cat([h_r, h_t], dim=0)
    for i, conv in enumerate(enc.convs):
        x = conv(x, edge_index)
        if i < len(enc.convs) - 1:
            x = F.relu(x)
    robot_emb, task_emb = x[:R], x[R:]
    scores = scorer.scorer(robot_emb, task_emb)
    theta = scorer.build_augmented_matrix(scores, state, instance)
    t3 = time.perf_counter()

    _decode_action(theta, state, R, T, epsilon=0.0)
    t4 = time.perf_counter()

    return theta, {
        "t_features_s": t1 - t0,
        "t_graph_build_s": t2 - t1,
        "t_gnn_forward_s": t3 - t2,
        "t_hungarian_s": t4 - t3,
    }


def _pct(a, q):
    return float(np.percentile(a, q))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--instances", type=int, default=60)
    ap.add_argument("--min-decisions", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if any(p.lower() == "test" for p in cache_dir.parts):
        raise SystemExit(
            f"REFUSING to time on a test split ({cache_dir}). "
            "This is a latency measurement; use the validation split."
        )
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"REFUSING to overwrite {out_path}")

    paths = sorted((REPO_ROOT / cache_dir).glob("seed*.json"))[: args.instances]
    records = [json.load(p.open()) for p in paths]
    if not records:
        raise SystemExit(f"No records under {cache_dir}")

    scorer = load_scorer_from_checkpoint(str(REPO_ROOT / args.checkpoint))
    scorer.eval()

    # Full decisions, discarded, so lazy init and first-call allocation never enter the recorded distribution.
    done = 0
    with torch.no_grad():
        for rec in records:
            if done >= args.warmup:
                break
            inst = SyntheticInstance.from_dict(rec["instance"])
            sim = DynamicSimulator(inst)
            while not sim.is_terminal and done < args.warmup:
                st = sim.state
                th = scorer(st, inst)
                # Warm both paths: separate call sequences, either could carry a first-call allocation cost into the recorded window.
                _decomposed(scorer, st, inst)
                sim.step(_decode_action(th, st, inst.R, inst.T, epsilon=0.0))
                done += 1
    print(f"warm-up decisions discarded: {done}", flush=True)

    rows = []
    n_inst = 0
    mismatches = 0
    with torch.no_grad():
        for rec in records:
            if len(rows) >= args.min_decisions and n_inst >= 50:
                break
            inst = SyntheticInstance.from_dict(rec["instance"])
            seed = int(rec.get("seed", -1))
            sim = DynamicSimulator(inst)
            n_inst += 1
            idx = 0
            while not sim.is_terminal:
                state = sim.state
                n_pend, n_avail = len(state.pending_tasks), len(state.available_robots)

                # Whole decision, unmodified path, timed first.
                t0 = time.perf_counter()
                theta_real = scorer(state, inst)
                commits = _decode_action(
                    theta_real, state, inst.R, inst.T, epsilon=0.0
                )
                t1 = time.perf_counter()
                total = t1 - t0

                theta_dec, st_t = _decomposed(scorer, state, inst)
                if not torch.equal(theta_dec, theta_real):
                    mismatches += 1

                ssum = sum(st_t.values())
                rows.append({
                    "seed": seed, "decision_index": idx,
                    "n_pending": n_pend, "n_available": n_avail,
                    "n_committed": len(commits),
                    **{k: repr(v) for k, v in st_t.items()},
                    "t_stages_sum_s": repr(ssum),
                    "t_total_s": repr(total),
                    "t_discrepancy_s": repr(ssum - total),
                })
                idx += 1
                sim.step(commits)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n = len(rows)
    print(f"\ninstances={n_inst}  decisions={n}  theta mismatches={mismatches}")
    print(f"mean decisions per instance = {n / n_inst!r}")
    print(f"wrote {out_path}")

    print("\n=== configuration ===")
    print(f"torch.get_num_threads()         = {torch.get_num_threads()}")
    print(f"torch.get_num_interop_threads() = {torch.get_num_interop_threads()}")
    print(f"device                          = cpu (map_location='cpu', no .to(device) in rollout)")
    print(f"mps available / built           = {torch.backends.mps.is_available()} / {torch.backends.mps.is_built()}")
    print(f"worker pool                     = none, single process")
    print(f"platform                        = {platform.platform()}")
    print(f"machine / processor             = {platform.machine()} / {platform.processor()}")
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
        ncpu = subprocess.run(["sysctl", "-n", "hw.ncpu"],
                              capture_output=True, text=True).stdout.strip()
        print(f"chip                            = {chip}  ({ncpu} logical cores)")
    except Exception as e:  # pragma: no cover
        print(f"chip                            = unavailable ({e})")
    print(f"python                          = {platform.python_version()}")
    print(f"torch                           = {torch.__version__}")
    print(f"scipy                           = {scipy.__version__}")
    print(f"numpy                           = {np.__version__}")

    print("\n=== per-decision timings, microseconds ===")
    hdr = f"{'stage':22s}{'median':>10s}{'mean':>10s}{'p5':>10s}{'p25':>10s}{'p75':>10s}{'p95':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for key, label in [
        ("t_features_s", "feature construction"),
        ("t_graph_build_s", "graph build"),
        ("t_gnn_forward_s", "GNN forward"),
        ("t_hungarian_s", "Hungarian (scipy)"),
        ("t_stages_sum_s", "SUM of stages"),
        ("t_total_s", "whole decision"),
    ]:
        a = np.array([float(r[key]) for r in rows]) * 1e6
        print(f"{label:22s}{np.median(a):10.2f}{a.mean():10.2f}"
              f"{_pct(a,5):10.2f}{_pct(a,25):10.2f}{_pct(a,75):10.2f}{_pct(a,95):10.2f}")
    d = np.array([float(r["t_discrepancy_s"]) for r in rows]) * 1e6
    print(f"\ndiscrepancy (sum of stages minus whole decision), microseconds:")
    print(f"  median={np.median(d):.2f}  mean={d.mean():.2f}  p5={_pct(d,5):.2f}  p95={_pct(d,95):.2f}")


if __name__ == "__main__":
    main()
