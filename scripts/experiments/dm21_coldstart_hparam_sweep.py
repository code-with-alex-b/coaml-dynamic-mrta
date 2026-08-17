"""DM-21 cold-start hyperparameter screen for method two.

Tests whether cold-start's poor result (sweep_G_coldstart_best.pt, 10.8% best
gap closure, DM-17) is caused by initialisation itself or by the production
hyperparameters (lr=1e-4, clip=1.0), which were tuned in the warm-start
regime where grad norms run 50-160. Cold-start grad norms run 500-1200, so
that clip discards nearly all gradient and the small lr compounds it. CS1-CS5
vary lr and clip (and, for CS4, the epsilon floor) to see whether a cold-start
run with headroom in the clip and a larger step size closes more of the gap.

All runs share the config-G structure (K=20, batch_size=128, bk=2560), rloo
baseline, --no-warm-start, and evaluate every 25 steps on the 200 validation
instances with the production evaluator (hard gap vs MILP, serve-all rate).
The five configs (CS1-CS5) live in sweep/configs.json.

Resume model: this script never contains its own resume logic. It always
invokes sweep/train_one.py with a fixed --checkpoint-path
checkpoints/sweep_cs_<id>.pt and the target --max-steps for this phase.
train_one.py's own resume block (checkpoint_path.exists() -> resume) picks
up automatically, so re-running this script with a larger --max-steps later
continues every run from wherever it stopped, including the epsilon
schedule, because epsilon_anneal_steps=300 is pinned in configs.json to the
full intended budget rather than left to float with num_steps.

Per-run stdout+stderr is appended (not overwritten) to logs/sweep_cs_<id>.log
so a later phase's output lands after phase 1's in the same file. After each
run reaches its target step for this phase, its eval-step lines are parsed
out of that log and matched against the exact per-step cost and grad_norm
recorded in the rolling checkpoint's cost_history/grad_norm_history (indexed
by step - 1), and appended to results/sweep_cs_summary.csv. An interim
ranking table (plus the DM-17 reference point) prints after every run.

If a run exits non-zero and the tail of its log looks like the known ARM64
spawn-related crash (AssertionError mentioning "spawn", or a
BrokenProcessPool), it is relaunched once automatically (the resume block
picks it up from its last rolling checkpoint) and the sweep continues. Any
other crash, or a second failure of the same run, stops the whole sweep with
the log tail printed, rather than silently pressing on past something the
run-order was designed to let you catch early.

Run (phase 1, 150 steps per run):
    PYTHONPATH="$PWD/src" python scripts/experiments/dm21_coldstart_hparam_sweep.py --max-steps 150

Run later (extend every run to the full 300-step budget, resuming from 150):
    PYTHONPATH="$PWD/src" python scripts/experiments/dm21_coldstart_hparam_sweep.py --max-steps 300
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_ONE = REPO_ROOT / "sweep" / "train_one.py"
SRC_DIR = REPO_ROOT / "src"
CSV_PATH = REPO_ROOT / "results" / "sweep_cs_summary.csv"
CSV_HEADER = ["run", "step", "gap_closure", "serve_all", "cost", "grad_norm"]

# Highest-signal configs first so the sweep can be stopped early once the picture is clear.
PRIORITY_ORDER = ["CS3", "CS4", "CS2", "CS5", "CS1"]

# Display-only; the actual training values come from sweep/configs.json.
CS_HPARAMS = {
    "CS1": "lr=0.0001 clip=10.0 eps->0.5",
    "CS2": "lr=0.0005 clip=10.0 eps->0.5",
    "CS3": "lr=0.001  clip=10.0 eps->0.5",
    "CS4": "lr=0.0005 clip=10.0 eps->0.3",
    "CS5": "lr=0.001  clip=1.0  eps->0.5",
}

# checkpoints/sweep_G_coldstart_best.pt, matched-hyperparameter cold start, full 1000-step budget; verified best_gap = 0.10800738496983303 at best_step = 975.
DM17_LABEL = "DM-17 (matched hparams, lr=0.0001 clip=1.0, full 1000-step budget)"
DM17_BEST_GAP_PCT = 10.8
DM17_BEST_STEP = 975

STEP_RE = re.compile(
    r"\[step (\d+)/(\d+)\].*elapsed=([0-9.]+)s"
)
EVAL_RE = re.compile(
    r"\[eval step (\d+)/(\d+)\] hard gap_vs_milp=([0-9.]+|n/a) "
    r"served_all_rate=([0-9.]+)"
)
SPAWN_CRASH_MARKERS = ("BrokenProcessPool",)


def looks_like_spawn_crash(text: str) -> bool:
    if "BrokenProcessPool" in text:
        return True
    return "AssertionError" in text and "spawn" in text.lower()


def run_config(config_id: str, max_steps: int, num_workers: int, eval_every: int):
    """Run (or resume) one CS config to ``max_steps``. Returns True on success."""
    ckpt_path = REPO_ROOT / "checkpoints" / f"sweep_cs_{config_id}.pt"
    log_path = REPO_ROOT / "logs" / f"sweep_cs_{config_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(TRAIN_ONE),
        "--config-id", config_id,
        "--num-workers", str(num_workers),
        "--max-steps", str(max_steps),
        "--checkpoint-path", str(ckpt_path),
        "--eval-every", str(eval_every),
        "--no-warm-start",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)

    resuming = ckpt_path.exists()
    print(
        f"\n=== {config_id}: "
        f"{'resuming' if resuming else 'starting'} toward step {max_steps} "
        f"(log: {log_path}) ===",
        flush=True,
    )

    for attempt in (1, 2):
        with log_path.open("a") as log_fh:
            log_fh.write(
                f"\n----- orchestrator: launching attempt {attempt} toward "
                f"max_steps={max_steps} -----\n"
            )
            log_fh.flush()
            result = subprocess.run(
                cmd, cwd=REPO_ROOT, env=env, stdout=log_fh, stderr=subprocess.STDOUT
            )
        if result.returncode == 0:
            return True

        tail = log_path.read_text()[-4000:]
        if attempt == 1 and looks_like_spawn_crash(tail):
            print(
                f"{config_id}: crashed with what looks like the ARM64 spawn "
                f"assertion; relaunching once (will resume from its last "
                f"rolling checkpoint).",
                flush=True,
            )
            continue

        print(
            f"\n{config_id}: FAILED (exit {result.returncode}), attempt "
            f"{attempt}. Last part of {log_path}:\n{tail}\n",
            flush=True,
        )
        return False

    return False


def parse_log(log_path: Path):
    """Return the list of (step, gap_pct_or_None, serve_all) eval rows."""
    rows = []
    if not log_path.exists():
        return rows
    with log_path.open() as f:
        for line in f:
            m = EVAL_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            gap = None if m.group(2) == "n/a" else float(m.group(2)) * 100.0
            serve = float(m.group(3))
            rows.append((step, gap, serve))
    return rows


def load_step_histories(ckpt_path: Path):
    """Return (cost_history, grad_norm_history) lists from a rolling checkpoint."""
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("cost_history", []), ckpt.get("grad_norm_history", [])


def append_csv_rows(config_id: str, log_path: Path, ckpt_path: Path):
    """Append this run's new eval rows to results/sweep_cs_summary.csv.

    Skips (run, step) pairs already present so re-running a later phase over
    the same log does not duplicate phase 1's rows.
    """
    rows = parse_log(log_path)
    if not rows:
        return []
    cost_hist, grad_hist = load_step_histories(ckpt_path)

    existing = set()
    write_header = not CSV_PATH.exists()
    if CSV_PATH.exists():
        with CSV_PATH.open() as f:
            for r in csv.DictReader(f):
                existing.add((r["run"], int(r["step"])))

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_rows = []
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        for step, gap, serve in rows:
            if (config_id, step) in existing:
                continue
            idx = step - 1
            cost = cost_hist[idx] if idx < len(cost_hist) else ""
            grad_norm = grad_hist[idx] if idx < len(grad_hist) else ""
            gap_frac = "" if gap is None else gap / 100.0
            row = [config_id, step, gap_frac, serve, cost, grad_norm]
            writer.writerow(row)
            new_rows.append((step, gap, serve))
    return new_rows


def print_wall_clock_estimate(first_config: str, max_steps: int, n_configs: int):
    log_path = REPO_ROOT / "logs" / f"sweep_cs_{first_config}.log"
    if not log_path.exists():
        return
    points = []
    with log_path.open() as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                points.append((int(m.group(1)), float(m.group(3))))
    if len(points) < 2:
        print(
            f"\n(Not enough step timing points in {log_path} yet for a "
            f"wall-clock estimate.)",
            flush=True,
        )
        return
    (s0, t0), (s1, t1) = points[0], points[-1]
    if s1 <= s0:
        return
    sec_per_step = (t1 - t0) / (s1 - s0)
    remaining_steps_total = n_configs * max_steps - s1
    est_sec = remaining_steps_total * sec_per_step
    h, rem = divmod(int(est_sec), 3600)
    m, sec = divmod(rem, 60)
    print(
        f"\nThroughput from {first_config}: {sec_per_step:.1f}s/step "
        f"(from step {s0} to {s1}). Estimated remaining wall-clock for the "
        f"rest of this phase ({remaining_steps_total} steps across "
        f"{n_configs} runs): {h}h{m:02d}m{sec:02d}s.\n",
        flush=True,
    )


def print_ranking_table(results: dict, max_steps: int):
    """results: config_id -> list of (step, gap_pct_or_None, serve_all)."""
    print(f"\n--- Interim ranking (toward step {max_steps}) ---")
    header = f"{'run':>6} | {'best gap':>9} | {'at step':>7} | {'last serve-all':>14} | hparams"
    print(header)
    print("-" * len(header))

    entries = []
    for cfg_id in PRIORITY_ORDER:
        rows = results.get(cfg_id)
        if not rows:
            entries.append((cfg_id, None, None, None, "pending"))
            continue
        serving = [(s, g, sv) for s, g, sv in rows if g is not None]
        if serving:
            best_step, best_gap, _ = max(serving, key=lambda t: t[1])
        else:
            best_step, best_gap = None, None
        last_serve = rows[-1][2]
        entries.append((cfg_id, best_gap, best_step, last_serve, "done"))

    entries.sort(
        key=lambda e: (e[1] is None, -(e[1] if e[1] is not None else 0.0))
    )
    for cfg_id, best_gap, best_step, last_serve, status in entries:
        gap_s = "pending" if best_gap is None else f"{best_gap:5.1f}%"
        step_s = "-" if best_step is None else str(best_step)
        serve_s = "-" if last_serve is None else f"{last_serve*100:4.0f}%"
        print(
            f"{cfg_id:>6} | {gap_s:>9} | {step_s:>7} | {serve_s:>14} | "
            f"{CS_HPARAMS[cfg_id]}"
        )
    print(
        f"{'DM-17':>6} | {DM17_BEST_GAP_PCT:8.1f}% | {DM17_BEST_STEP:>7} | "
        f"{'n/a':>14} | {DM17_LABEL}"
    )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument(
        "--config-ids", type=str, default=None,
        help="Comma-separated override of the run order (default: the fixed "
        "CS3,CS4,CS2,CS5,CS1 priority order).",
    )
    args = ap.parse_args()

    config_ids = (
        args.config_ids.split(",") if args.config_ids else list(PRIORITY_ORDER)
    )

    print(
        f"DM-21 cold-start hyperparameter screen: {len(config_ids)} runs "
        f"toward max_steps={args.max_steps}, order={config_ids}.",
        flush=True,
    )

    results: dict = {}
    for i, cfg_id in enumerate(config_ids):
        ok = run_config(cfg_id, args.max_steps, args.num_workers, args.eval_every)
        log_path = REPO_ROOT / "logs" / f"sweep_cs_{cfg_id}.log"
        ckpt_path = REPO_ROOT / "checkpoints" / f"sweep_cs_{cfg_id}.pt"
        if ckpt_path.exists():
            append_csv_rows(cfg_id, log_path, ckpt_path)
        results[cfg_id] = parse_log(log_path)

        if i == 0:
            print_wall_clock_estimate(cfg_id, args.max_steps, len(config_ids))

        print_ranking_table(results, args.max_steps)

        if not ok:
            print(
                f"{cfg_id} did not complete; stopping the sweep here so this "
                f"can be looked at before continuing. Already-collected "
                f"results above and in {CSV_PATH} are unaffected.",
                flush=True,
            )
            sys.exit(1)

    print(f"Phase complete. Full results in {CSV_PATH}.", flush=True)


if __name__ == "__main__":
    main()
