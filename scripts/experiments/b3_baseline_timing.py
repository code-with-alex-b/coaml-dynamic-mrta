"""B3 Phase 3. Per-decision compute for the two Hungarian baselines and for the
learned policy, measured on the frozen test split.

Same method as DM-40: warm up at least 20 full decisions, time with
``time.perf_counter`` around each stage and around the whole decision, and
record at least 2,000 decisions. Read-only. No solver, no training, no cache
write.

The Hungarian baselines decompose into two stages rather than four, because
they have no network. Their per-epoch work is the cost-matrix build followed by
the scipy assignment solve, which is what ``_run_with_per_epoch_builder`` does.

Usage::

    PYTHONPATH="$PWD/src" python scripts/experiments/b3_baseline_timing.py \
        --cache-dir cache/training_set_il_v3/test --allow-test-split \
        --out provenance/b3_timing_test.csv
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy
import torch
import torch.nn.functional as F

from baselines.bipartite_policies import (
    build_greedy_cost_matrix,
    build_kappa_cost_matrix,
    hungarian_action,
)
from evaluation.method_one_evaluator import (
    _decode_action,
    load_scorer_from_checkpoint,
)
from evaluation.method_two_evaluator import load_records
from instances.synthetic_generator import SyntheticInstance
from scoring.gnn_scorer import build_features, _BipartiteGNNCore
from simulator.dynamic_simulator import DynamicSimulator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
W = {"w_dist": 0.0637, "w_make": 0.2398, "w_bal": 0.6965}
FIELDS = ["policy", "seed", "decision_index", "t_stage1_s", "t_stage2_s",
          "t_stages_sum_s", "t_total_s", "t_discrepancy_s"]


def _pct(a, q):
    return float(np.percentile(a, q))


def time_hungarian(instance, seed, builder, rows, label):
    """One episode of a Hungarian baseline, timing each decision."""
    sim = DynamicSimulator(instance)
    idx = 0
    while not sim.is_terminal:
        state = sim.state
        t0 = time.perf_counter()
        cost, rids, tids = builder(state, sim)
        t1 = time.perf_counter()
        action = hungarian_action(cost, rids, tids)
        t2 = time.perf_counter()
        total = t2 - t0
        rows.append({
            "policy": label, "seed": seed, "decision_index": idx,
            "t_stage1_s": repr(t1 - t0), "t_stage2_s": repr(t2 - t1),
            "t_stages_sum_s": repr(total), "t_total_s": repr(total),
            "t_discrepancy_s": repr(0.0),
        })
        idx += 1
        sim.step(action)


def time_learned(scorer, instance, seed, rows):
    """One episode of the learned policy, decomposed as in DM-40."""
    enc = scorer.encoder
    R, T = int(instance.R), int(instance.T)
    sim = DynamicSimulator(instance)
    idx = 0
    with torch.no_grad():
        while not sim.is_terminal:
            state = sim.state
            t0 = time.perf_counter()
            theta = scorer(state, instance)
            commits = _decode_action(theta, state, R, T, epsilon=0.0)
            t1 = time.perf_counter()
            total = t1 - t0

            s0 = time.perf_counter()
            rf, tf = build_features(state, instance)
            s1 = time.perf_counter()
            ei = _BipartiteGNNCore._build_edge_index(R, T, rf.device)
            s2 = time.perf_counter()
            h = torch.cat([enc.robot_proj(rf), enc.task_proj(tf)], dim=0)
            for i, conv in enumerate(enc.convs):
                h = conv(h, ei)
                if i < len(enc.convs) - 1:
                    h = F.relu(h)
            sc = scorer.scorer(h[:R], h[R:])
            th2 = scorer.build_augmented_matrix(sc, state, instance)
            s3 = time.perf_counter()
            _decode_action(th2, state, R, T, epsilon=0.0)
            s4 = time.perf_counter()
            rows.append({
                "policy": "learned_hard", "seed": seed, "decision_index": idx,
                "t_stage1_s": repr((s1 - s0) + (s2 - s1) + (s3 - s2)),
                "t_stage2_s": repr(s4 - s3),
                "t_stages_sum_s": repr(s4 - s0),
                "t_total_s": repr(total),
                "t_discrepancy_s": repr((s4 - s0) - total),
            })
            idx += 1
            sim.step(commits)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="cache/training_set_il_v3/val")
    ap.add_argument("--checkpoint", default="checkpoints/sweep_G_v4warmstart_best.pt")
    ap.add_argument("--instances", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-test-split", action="store_true")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if any(p.lower() == "test" for p in cache_dir.parts) and not args.allow_test_split:
        raise SystemExit(f"REFUSING to time on a test split ({cache_dir}).")
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"REFUSING to overwrite {out_path}")

    records = load_records(str(REPO_ROOT / cache_dir), args.instances)
    scorer = load_scorer_from_checkpoint(str(REPO_ROOT / args.checkpoint))
    scorer.eval()

    gb = lambda st, sim: build_greedy_cost_matrix(st, sim.instance)
    kb = lambda st, sim: build_kappa_cost_matrix(
        st, sim.instance, W, sim.trajectory.commitments)

    warm = []
    done = 0
    for rec in records:
        if done >= args.warmup:
            break
        inst = SyntheticInstance.from_dict(rec["instance"])
        time_hungarian(inst, -1, gb, warm, "warm")
        time_hungarian(inst, -1, kb, warm, "warm")
        time_learned(scorer, inst, -1, warm)
        done = len(warm)
    print(f"warm-up decisions discarded: {len(warm)}", flush=True)

    rows = []
    for rec in records:
        inst = SyntheticInstance.from_dict(rec["instance"])
        seed = int(rec.get("seed", -1))
        time_hungarian(inst, seed, gb, rows, "hungarian_distance_only")
        time_hungarian(inst, seed, kb, rows, "hungarian_kappa_weighted")
        time_learned(scorer, inst, seed, rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_path}  ({len(rows)} rows)", flush=True)

    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    val = ctypes.c_int(0); size = ctypes.c_size_t(4)
    libc.sysctlbyname(b"sysctl.proc_translated", ctypes.byref(val),
                      ctypes.byref(size), None, 0)
    print("\n=== environment ===")
    print(f"sysctl.proc_translated (in process) = {val.value}")
    print(f"platform.machine()                  = {platform.machine()}")
    print(f"torch.get_num_threads()             = {torch.get_num_threads()}")
    print(f"torch.get_num_interop_threads()     = {torch.get_num_interop_threads()}")
    print(f"device                              = cpu")
    print(f"python {platform.python_version()}  torch {torch.__version__} "
          f"scipy {scipy.__version__}  numpy {np.__version__}")
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
        print(f"chip = {chip}")
    except Exception:
        pass

    print("\n=== per-decision timings, microseconds ===")
    hdr = f"{'policy':26s}{'n':>7s}{'median':>10s}{'mean':>10s}{'p5':>9s}{'p25':>9s}{'p75':>9s}{'p95':>10s}"
    print(hdr); print("-" * len(hdr))
    for pol in ("hungarian_distance_only", "hungarian_kappa_weighted", "learned_hard"):
        sub = [r for r in rows if r["policy"] == pol]
        a = np.array([float(r["t_total_s"]) for r in sub]) * 1e6
        print(f"{pol:26s}{len(sub):7d}{np.median(a):10.2f}{a.mean():10.2f}"
              f"{_pct(a,5):9.2f}{_pct(a,25):9.2f}{_pct(a,75):9.2f}{_pct(a,95):10.2f}")
    print("\n=== per-instance totals, seconds ===")
    for pol in ("hungarian_distance_only", "hungarian_kappa_weighted", "learned_hard"):
        per = {}
        for r in rows:
            if r["policy"] == pol:
                per[r["seed"]] = per.get(r["seed"], 0.0) + float(r["t_total_s"])
        v = np.array(list(per.values()))
        nd = len([r for r in rows if r["policy"] == pol]) / len(per)
        print(f"{pol:26s} mean={v.mean():.6f}  median={np.median(v):.6f}  "
              f"decisions/instance={nd:.2f}  instances={len(per)}")


if __name__ == "__main__":
    main()
