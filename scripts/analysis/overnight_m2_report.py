#!/usr/bin/env python
"""Report a Method Two overnight arm from its rolling checkpoint.

Reads the ``eval_history`` list that method_two_trainer now persists in every
periodic checkpoint, and prints the whole validation trajectory rather than a
best value. Earlier runs stored only ``best_gap`` and ``best_step``, so the
shape of the curve was recoverable only from stdout and was lost when the log
was not kept. That cost a figure.

The comparison point is production Method Two at step 400, which scored 0.3881
on the same 200 instance validation cache under hard decode. Run-to-run
variance at that step is 7.32 points, so a difference smaller than roughly 7
points is not a difference.

Usage:
    python scripts/analysis/overnight_m2_report.py checkpoints/overnight_s4_eps035.pt
    python scripts/analysis/overnight_m2_report.py checkpoints/overnight_s*.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

PRODUCTION_AT_400 = 0.3881
COMPARISON_STEP = 400
# Observed run-to-run spread at step 400 on this configuration, in points of gap closure; anything inside it is noise.
RUN_TO_RUN_POINTS = 7.32


def report(path: Path) -> None:
    print(f"=== {path} ===")
    if not path.exists():
        print("  checkpoint does not exist; the stage did not reach a save.")
        print()
        return
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"  unreadable: {exc}")
        print()
        return

    step = ckpt.get("step")
    init_kwargs = ckpt.get("init_kwargs")
    config = ckpt.get("config") or {}
    print(f"  reached step      : {step}")
    print(f"  scorer shape      : {init_kwargs or 'not recorded (defaults)'}")
    print(f"  warm start        : {config.get('warm_start_from')}")
    print(f"  resumed from      : {config.get('resume_from')}")
    print(f"  K / batch         : {config.get('rloo_k')} / "
          f"{config.get('batch_size')}")
    print(f"  lr / clip         : {config.get('learning_rate')} / "
          f"{config.get('gradient_clip_norm')}")
    print(f"  epsilon           : {config.get('epsilon_initial')} -> "
          f"{config.get('epsilon_terminal')} over "
          f"{config.get('epsilon_anneal_steps')} steps")
    print(f"  workers           : {config.get('num_workers')}")

    history = ckpt.get("eval_history") or []
    if not history:
        best_gap = ckpt.get("best_gap")
        best_step = ckpt.get("best_step")
        print("  eval_history      : absent. This checkpoint predates "
              "eval_history, or no evaluation has run yet.")
        if best_gap is not None:
            print(f"  best (only figure): {best_gap:+.4f} at step {best_step}")
        print()
        return

    print("  trajectory (hard decode gap closure vs MILP, validation split):")
    print("    step    gap        serve-all   n_val")
    for entry in history:
        gap = entry.get("gap")
        gap_str = "   n/a  " if gap is None else f"{gap:+.4f}"
        print(f"    {entry['step']:<7d} {gap_str}   "
              f"{entry.get('served_all_rate', float('nan')):.3f}       "
              f"{entry.get('n_val')}")

    at_step = [e for e in history if e["step"] == COMPARISON_STEP]
    print()
    if at_step and at_step[-1].get("gap") is not None:
        value = at_step[-1]["gap"]
        delta_points = (value - PRODUCTION_AT_400) * 100
        print(f"  step {COMPARISON_STEP}: {value:.4f}  "
              f"versus production {PRODUCTION_AT_400:.4f}  "
              f"(delta {delta_points:+.2f} points)")
        if abs(delta_points) < RUN_TO_RUN_POINTS:
            print(f"  Run-to-run variance at this step is "
                  f"{RUN_TO_RUN_POINTS} points, so this is NOT a difference.")
        else:
            print(f"  Exceeds the {RUN_TO_RUN_POINTS} point run-to-run "
                  f"variance at this step, so it is worth a second look. It "
                  f"is one sample, not a confirmed effect.")
    else:
        print(f"  No evaluation at step {COMPARISON_STEP}; the run did not "
              f"reach the comparison point. Production's value there is "
              f"{PRODUCTION_AT_400:.4f}.")
        print(f"  Reminder: run-to-run variance at step {COMPARISON_STEP} is "
              f"{RUN_TO_RUN_POINTS} points, so anything below roughly 0.46 "
              f"is not a difference.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()
    print("Method Two overnight arms")
    print(f"Comparison point: production {PRODUCTION_AT_400:.4f} at step "
          f"{COMPARISON_STEP}, same 200 instance validation cache.")
    print(f"Run-to-run variance at step {COMPARISON_STEP} is "
          f"{RUN_TO_RUN_POINTS} points: anything below roughly 0.46 is not a "
          f"difference.")
    print("Validation split only. No test split figure appears here.")
    print()
    for raw in args.checkpoints:
        report(Path(raw))


if __name__ == "__main__":
    main()
