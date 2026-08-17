"""Per-decision inference latency of the learned policy at all four scales.

Phase 1.5 step 4. The audit in CLEANUP_INVENTORY.md found that two of the four
compute figures in Table 4.2 do not reproduce, being C134 at 3.42 ms against a
recomputed 3.5840 and C139 at 2.20 ms against 2.3081, and that the R=6, T=18
row is split-mixed, with test-split costs beside a validation-split latency.
The published figures also came from two different timing runs on two different
days.

This script measures all four scales in one process on one machine, so the four
numbers are comparable with each other for the first time. R=6, T=18 is
measured on the TEST split, which puts every quantity in that row on one split.

Method, unchanged from DM-40 and ``scripts/experiments/a6_inference_timing.py``. A decision
is the whole dispatcher step the evaluator executes,

    theta   = scorer(state, instance)
    commits = _decode_action(theta, state, R, T, epsilon=0.0)

timed with ``time.perf_counter`` around both calls together. The simulator step
is excluded, being environment cost rather than policy cost. Warm-up decisions
are run and discarded before any recording. Hard decode throughout, epsilon 0,
which draws no randomness, so the rollouts are deterministic and only the
timings vary between runs.

Absolute milliseconds are machine and environment dependent. The environment
block is recorded with every run so a figure is never quoted without it.

No solver is invoked, nothing is trained, and no cache record or existing file
is modified.

Usage::

    PYTHONPATH="$PWD/src" python scripts/experiments/timing_all.py \\
        --allow-test-split --out-dir provenance/table42_transfer
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluation.method_one_evaluator import (  # noqa: E402
    _decode_action,
    load_scorer_from_checkpoint,
)
from evaluation.method_two_evaluator import load_records  # noqa: E402
from instances.synthetic_generator import SyntheticInstance  # noqa: E402
from simulator.dynamic_simulator import DynamicSimulator  # noqa: E402

DEFAULT_CKPT = "checkpoints/sweep_G_v4warmstart_best.pt"

# R=6, T=18 is deliberately the test split, not the validation split the transfer table used.
SCALES = [
    ("r6t18", "cache/training_set_il_v3/test", "test", 6, 18,
     {"C120_transfer_table": 0.96, "C108_figure_4_2_marker": 1.11}),
    ("r10t30", "cache/dm31a_r10t30", "transfer, seeds 96000-96059", 10, 30,
     {"C127_transfer_table": 1.47}),
    ("r30t90", "cache/dm31b_r30t90", "transfer, seeds 96100-96159", 30, 90,
     {"C134_transfer_table": 3.42}),
    ("r10t60", "cache/training_set_rh_r10t60", "transfer, seeds 90000-90199", 10, 60,
     {"C139_transfer_table": 2.20}),
]


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def _proc_translated() -> int | None:
    """Read sysctl.proc_translated in process, as DM-40 did.

    A subprocess would report its own translation state, so the value must come
    from inside this interpreter.
    """
    try:
        libc = ctypes.CDLL("libc.dylib")
        out = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(out))
        rc = libc.sysctlbyname(
            b"sysctl.proc_translated", ctypes.byref(out), ctypes.byref(size), None, 0
        )
        return int(out.value) if rc == 0 else None
    except Exception:
        return None


def environment() -> dict:
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        chip = "unavailable"
    try:
        uname_m = subprocess.run(
            ["uname", "-m"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        uname_m = "unavailable"
    return {
        "platform_machine": platform.machine(),
        "uname_m": uname_m,
        "sysctl_proc_translated_in_process": _proc_translated(),
        "cpu_brand": chip,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "device": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "os": platform.platform(),
    }


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def _cpu_idle_samples(n: int, interval: int) -> list[float]:
    """Per-sample idle CPU percentage from ``top -l``.

    The first ``top`` sample is a since-boot average and is discarded, so
    ``n`` recorded samples need ``n + 1`` requested.
    """
    out = subprocess.run(
        ["top", "-l", str(n + 1), "-n", "0", "-s", str(interval)],
        capture_output=True, text=True, check=True).stdout
    idles = []
    for line in out.splitlines():
        if line.startswith("CPU usage:"):
            for tok in line.split(","):
                if "idle" in tok:
                    idles.append(float(tok.strip().split("%")[0]))
    return idles[1:]


def _swap_counters() -> dict:
    """Cumulative swapins and swapouts, plus the swapfile usage."""
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True,
                        check=True).stdout
    counters = {}
    for line in vm.splitlines():
        low = line.lower()
        for key in ("swapins", "swapouts", "pages occupied by compressor"):
            if low.startswith(key):
                counters[key] = int(line.rsplit(":", 1)[1].strip().rstrip("."))
    swapusage = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                               capture_output=True, text=True,
                               check=True).stdout.strip()
    counters["vm.swapusage"] = swapusage
    return counters


def machine_gate(min_idle: float, samples: int, interval: int) -> dict:
    """Gate on idle CPU and on swapouts, not on load average.

    Load average is deliberately not a gate condition. After a restart it
    decays slowly from the boot transient and reads high while the machine is
    in fact quiet, so it would block a run that idle CPU and the swap counters
    both show is fine.
    """
    idles = _cpu_idle_samples(samples, interval)
    swap = _swap_counters()
    loadavg = subprocess.run(["uptime"], capture_output=True, text=True,
                             check=True).stdout.strip()
    idle_ok = bool(idles) and all(i > min_idle for i in idles)
    swap_ok = swap.get("swapouts", -1) == 0
    report = {
        "idle_samples_pct": idles,
        "idle_min_pct": min(idles) if idles else None,
        "idle_threshold_pct": min_idle,
        "idle_pass": idle_ok,
        "swapouts": swap.get("swapouts"),
        "swapins": swap.get("swapins"),
        "pages_occupied_by_compressor": swap.get("pages occupied by compressor"),
        "vm_swapusage": swap.get("vm.swapusage"),
        "swapouts_pass": swap_ok,
        "loadavg_line_recorded_not_gated": loadavg,
        "pass": idle_ok and swap_ok,
    }
    return report


def cache_size(cache_dir: Path) -> int:
    """Number of seed*.json records available in a cache, the full split."""
    return len(sorted(Path(cache_dir).glob("seed*.json")))


def time_scale(scorer, cache_dir: Path, n_instances: int, warmup: int,
               min_decisions: int) -> tuple[list, dict]:
    records = load_records(cache_dir, n_instances)
    if not records:
        raise SystemExit(f"No seed*.json records under {cache_dir}.")

    # Warm-up, discarded. Same code path, no recording.
    done = 0
    with torch.no_grad():
        for rec in records:
            if done >= warmup:
                break
            inst = SyntheticInstance.from_dict(rec["instance"])
            sim = DynamicSimulator(inst)
            while not sim.is_terminal and done < warmup:
                st = sim.state
                th = scorer(st, inst)
                sim.step(_decode_action(th, st, inst.R, inst.T, epsilon=0.0))
                done += 1

    rows, n_inst, seeds = [], 0, []
    with torch.no_grad():
        for rec in records:
            if len(rows) >= min_decisions and n_inst >= n_instances:
                break
            inst = SyntheticInstance.from_dict(rec["instance"])
            seed = int(rec.get("seed", -1))
            sim = DynamicSimulator(inst)
            n_inst += 1
            seeds.append(seed)
            idx = 0
            while not sim.is_terminal:
                state = sim.state
                t0 = time.perf_counter()
                theta = scorer(state, inst)
                commits = _decode_action(theta, state, inst.R, inst.T, epsilon=0.0)
                t1 = time.perf_counter()
                rows.append({
                    "seed": seed, "decision_index": idx,
                    "n_pending": len(state.pending_tasks),
                    "n_available": len(state.available_robots),
                    "n_committed": len(commits),
                    "t_total_s": repr(t1 - t0),
                })
                idx += 1
                sim.step(commits)

    t_ms = np.array([float(r["t_total_s"]) for r in rows]) * 1000.0
    per_inst = {}
    for r in rows:
        per_inst.setdefault(r["seed"], 0.0)
        per_inst[r["seed"]] += float(r["t_total_s"])
    return rows, {
        "n_decisions": len(rows),
        "n_instances": n_inst,
        "seed_min": min(seeds), "seed_max": max(seeds),
        "warmup_discarded": done,
        "decisions_per_instance": len(rows) / n_inst,
        "median_ms": float(np.median(t_ms)),
        "mean_ms": float(t_ms.mean()),
        "p5_ms": _pct(t_ms, 5), "p25_ms": _pct(t_ms, 25),
        "p75_ms": _pct(t_ms, 75), "p95_ms": _pct(t_ms, 95),
        "iqr_ms": _pct(t_ms, 75) - _pct(t_ms, 25),
        "per_instance_total_ms": float(np.mean(list(per_inst.values())) * 1000.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", default="provenance/table42_transfer")
    ap.add_argument("--instances", type=int, default=50,
                    help="Instances per scale. The historic default of 50 is "
                         "kept so earlier runs stay reproducible. Override it "
                         "with --all-instances.")
    ap.add_argument("--all-instances", action="store_true",
                    help="Time the FULL cache at every scale, ignoring "
                         "--instances and --min-decisions.")
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--min-decisions", type=int, default=2000)
    ap.add_argument("--allow-test-split", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-idle-pct", type=float, default=80.0,
                    help="Gate: every idle CPU sample must exceed this.")
    ap.add_argument("--gate-samples", type=int, default=4)
    ap.add_argument("--gate-interval", type=int, default=3)
    ap.add_argument("--skip-gate", action="store_true")
    args = ap.parse_args()

    needs_test = any(
        any(p.lower() == "test" for p in Path(c).parts) for _, c, _, _, _, _ in SCALES
    )
    if needs_test and not args.allow_test_split:
        raise SystemExit(
            "REFUSING to time on a test split without --allow-test-split. "
            "The R=6, T=18 row is measured on cache/training_set_il_v3/test."
        )
    if needs_test:
        print("OPT-IN: --allow-test-split passed, R=6, T=18 is timed on the test split.")

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "repro_timing_all_scales.json"
    if json_path.exists() and not args.force:
        raise SystemExit(f"REFUSING to overwrite {json_path}. Pass --force.")

    gate_before = None
    if args.skip_gate:
        print("Machine-state gate SKIPPED by --skip-gate.")
    else:
        print(f"=== machine-state gate: idle CPU > {args.min_idle_pct} % on every "
              f"sample, and zero swapouts ===", flush=True)
        gate_before = machine_gate(args.min_idle_pct, args.gate_samples,
                                   args.gate_interval)
        print(json.dumps(gate_before, indent=1))
        if not gate_before["pass"]:
            reasons = []
            if not gate_before["idle_pass"]:
                reasons.append(
                    f"idle CPU fell to {gate_before['idle_min_pct']} % against a "
                    f"{args.min_idle_pct} % floor")
            if not gate_before["swapouts_pass"]:
                reasons.append(f"swapouts is {gate_before['swapouts']}, not zero")
            raise SystemExit("ABORTING, machine-state gate failed: " +
                             "; ".join(reasons))
        print("Gate PASSED.\n", flush=True)

    env = environment()
    print(json.dumps(env, indent=1))
    if env["sysctl_proc_translated_in_process"] == 1:
        print("\nNOTE: this interpreter is running under Rosetta 2 translation. "
              "Timings are emulated x86_64, as in DM-40.\n")

    scorer = load_scorer_from_checkpoint(REPO_ROOT / args.checkpoint)
    scorer.eval()

    results = {}
    for key, cache_rel, split, R, T, published in SCALES:
        full_n = cache_size(REPO_ROOT / cache_rel)
        n_req = full_n if args.all_instances else args.instances
        min_dec = 0 if args.all_instances else args.min_decisions
        print(f"=== {key}  R={R} T={T}  {cache_rel}  ({split})", flush=True)
        print(f"  cache holds {full_n} records, timing {n_req}"
              f"{'  [FULL SPLIT]' if args.all_instances else '  [capped]'}",
              flush=True)
        rows, stats = time_scale(
            scorer, REPO_ROOT / cache_rel, n_req, args.warmup, min_dec)
        stats["cache_records_available"] = full_n
        stats["instances_requested"] = n_req
        stats["full_split_timed"] = (stats["n_instances"] == full_n)
        stats.update({"scale": key, "R": R, "T": T, "cache_dir": cache_rel,
                      "split": split, "published": published})
        results[key] = stats
        csv_path = out_dir / f"repro_timing_{key}.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  {stats['n_decisions']} decisions over {stats['n_instances']} "
              f"instances, seeds {stats['seed_min']}-{stats['seed_max']}")
        print(f"  median {stats['median_ms']:.4f} ms   mean {stats['mean_ms']:.4f} ms"
              f"   per instance {stats['per_instance_total_ms']:.2f} ms")
        print(f"  IQR [{stats['p25_ms']:.4f}, {stats['p75_ms']:.4f}] "
              f"= {stats['iqr_ms']:.4f} ms   p95 {stats['p95_ms']:.4f} ms")
        for name, val in published.items():
            print(f"  vs {name} = {val}: difference "
                  f"{stats['median_ms'] - val:+.4f} ms "
                  f"({(stats['median_ms'] / val - 1) * 100:+.1f} %)")
        print(f"  wrote {csv_path}", flush=True)

    print("\n=== do the published C134 (3.42) and C139 (2.20) appear anywhere ===")
    hunt = {}
    for target in (3.42, 2.20):
        hits = []
        for key, s in results.items():
            for stat in ("median_ms", "mean_ms", "p5_ms", "p25_ms", "p75_ms", "p95_ms"):
                if abs(s[stat] - target) < 0.005:
                    hits.append(f"{key}/{stat}={s[stat]:.4f}")
        hunt[str(target)] = hits
        print(f"  {target}: {hits if hits else 'NOT FOUND as any of median, mean, p5, p25, p75, p95 at any scale'}")

    payload = {
        "script": "scripts/experiments/timing_all.py",
        "git_head": _git_head(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "method": "whole decision = scorer(state, instance) + _decode_action(...), "
                  "time.perf_counter, hard decode epsilon 0, simulator step excluded",
        "environment": env,
        "warmup_target": args.warmup,
        "instances_per_scale": "all" if args.all_instances else args.instances,
        "all_instances": args.all_instances,
        "min_decisions": 0 if args.all_instances else args.min_decisions,
        "machine_gate_rule": (
            f"idle CPU > {args.min_idle_pct} % on every one of "
            f"{args.gate_samples} samples, and cumulative swapouts == 0. "
            "Load average is recorded but NOT gated on, because it decays "
            "slowly from a restart and reads high on a quiet machine."),
        "machine_gate_before": gate_before,
        "machine_gate_after": None if args.skip_gate else machine_gate(
            args.min_idle_pct, args.gate_samples, args.gate_interval),
        "scales": results,
        "published_value_hunt": hunt,
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")
    print("No solver was invoked and no existing file was modified.")


if __name__ == "__main__":
    main()
