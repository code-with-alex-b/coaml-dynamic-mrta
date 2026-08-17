#!/usr/bin/env python
"""Pick the architecture sweep winner for the Method Two warm start.

Selection is on the mean over a cell's final three evaluations, not its best
single evaluation. A single evaluation on this metric moves several points
between adjacent steps (see any trajectory in results/dm_arch_sweep.csv), so a
best-of picks the luckiest evaluation rather than the strongest cell.

If no cell's mean beats production Method One's +6.7%, there is no candidate
worth changing the warm start for, and the caller is told to warm start from
checkpoints/il_method_one_v4_best.pt and record the run as a production
replicate. That is still a useful run: it is a second sample at production
settings, comparable against production's 0.3881 at step 400.

Emits shell-eval-able assignments on stdout:

    WINNER_CELL=L3_H128
    WINNER_CKPT=checkpoints/dm_arch_L3_H128_best.pt
    WINNER_MEAN=0.0812
    WINNER_ROLE=candidate            # or production_replicate
    WINNER_NOTE=...

Usage:
    python scripts/analysis/arch_sweep_winner.py
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

PRODUCTION_MEAN = 0.067
PRODUCTION_CKPT = "checkpoints/il_method_one_v4_best.pt"
TAIL = 3


def emit(**kwargs) -> None:
    for key, value in kwargs.items():
        print(f"{key}={value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results/dm_arch_sweep.csv")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        emit(
            WINNER_CELL="none",
            WINNER_CKPT=PRODUCTION_CKPT,
            WINNER_MEAN="nan",
            WINNER_ROLE="production_replicate",
            WINNER_NOTE=f"no eval CSV at {csv_path}",
        )
        return

    by_run = OrderedDict()
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            # Last row wins per step, so a relaunch that re-evaluated a step is not counted twice in the tail mean.
            by_run.setdefault(row["run"], {})[int(row["step"])] = row

    means = {}
    for run, rows_by_step in by_run.items():
        ordered = [rows_by_step[s] for s in sorted(rows_by_step)]
        if len(ordered) < TAIL:
            # A cell that never reached three evaluations is not eligible to be the winner.
            continue
        tail = ordered[-TAIL:]
        means[run] = sum(float(r["perturbed_gap"]) for r in tail) / TAIL

    if not means:
        emit(
            WINNER_CELL="none",
            WINNER_CKPT=PRODUCTION_CKPT,
            WINNER_MEAN="nan",
            WINNER_ROLE="production_replicate",
            WINNER_NOTE="no cell has three or more evaluations",
        )
        return

    best_cell = max(means, key=means.get)
    best_mean = means[best_cell]

    if best_mean <= PRODUCTION_MEAN:
        emit(
            WINNER_CELL=best_cell,
            WINNER_CKPT=PRODUCTION_CKPT,
            WINNER_MEAN=f"{best_mean:.4f}",
            WINNER_ROLE="production_replicate",
            WINNER_NOTE=(
                f"best cell {best_cell} mean {best_mean:+.4f} does not beat "
                f"production {PRODUCTION_MEAN:+.4f}; warm starting from "
                f"production instead"
            ),
        )
        return

    # Warm start from the cell's best checkpoint (matches production, whose il_method_one_v4_best.pt is also a best not a final checkpoint); fall back to rolling if missing.
    ckpt_dir = Path(args.ckpt_dir)
    best_ckpt = ckpt_dir / f"dm_arch_{best_cell}_best.pt"
    rolling = ckpt_dir / f"dm_arch_{best_cell}.pt"
    if best_ckpt.exists():
        chosen, note = best_ckpt, "cell best checkpoint"
    elif rolling.exists():
        chosen, note = rolling, "cell best checkpoint missing, using rolling"
    else:
        emit(
            WINNER_CELL=best_cell,
            WINNER_CKPT=PRODUCTION_CKPT,
            WINNER_MEAN=f"{best_mean:.4f}",
            WINNER_ROLE="production_replicate",
            WINNER_NOTE=(
                f"cell {best_cell} won on mean {best_mean:+.4f} but neither "
                f"{best_ckpt} nor {rolling} exists; falling back to production"
            ),
        )
        return

    emit(
        WINNER_CELL=best_cell,
        WINNER_CKPT=str(chosen),
        WINNER_MEAN=f"{best_mean:.4f}",
        WINNER_ROLE="candidate",
        WINNER_NOTE=(
            f"{note}; mean of final {TAIL} evaluations {best_mean:+.4f} beats "
            f"production {PRODUCTION_MEAN:+.4f}"
        ),
    )


if __name__ == "__main__":
    main()
