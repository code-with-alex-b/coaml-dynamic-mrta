"""Figure 4.1. Warm start against cold start over the 1000 step training budget.

Both panels are built from the training cost and gradient norm histories of the
two runs, not from validation gap closure, so both runs are covered at every one
of the 1000 steps with no gaps.

The two series are read from tracked CSVs under ``provenance/figure41_training/`` rather
than from the rolling checkpoints they originally lived in. Those CSVs were
exported by ``scripts/regenerate/export_series_and_logs.py`` at full float precision, so
the values are bitwise identical to the checkpoint arrays and this figure is
byte-for-byte the one that was published. The point of the change is that
notebook one can regenerate Figure 4.1 without opening a 349 KB checkpoint.

Left panel is the training cost per step, which is the mean combined simulator
cost over the 2560 rollouts of that step's minibatch. Lower is better. Right
panel is the pre-clip gradient norm per step on a logarithmic axis, scaled to
the data.

The raw cost series carries about 1.9 units of per-step sampling noise, which is
enough to bury the cold started run's total fall of 11.63 units, so both panels
draw a centred rolling median over 51 steps on top of the raw series. The raw
series stays visible faintly underneath in both panels, so nothing is hidden.

Reads two CSVs, writes a PDF and a sidecar. Opens no checkpoint, trains nothing
and evaluates nothing.

Run with
    PYTHONPATH="$PWD/src" python figures/fig_warm_vs_cold.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import csv

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import (  # noqa: E402
    COLD_STYLE,
    COLOR_COLD,
    COLOR_WARM,
    PANEL_LETTER_PT,
    RAW_ALPHA,
    RAW_LINEWIDTH,
    WARM_STYLE,
    apply_style,
    figure_size,
    hairline_grid,
    write_sidecar,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WARM_SERIES = os.path.join(
    REPO_ROOT, "provenance", "figure41_training", "series_warm_start_training.csv")
COLD_SERIES = os.path.join(
    REPO_ROOT, "provenance", "figure41_training", "series_cold_start_training.csv")

WARM_LABEL = "Warm start from imitation"
COLD_LABEL = "Cold start"

OUT_DIR = os.path.join(REPO_ROOT, "figures", "output")
OUT_PDF = os.path.join(OUT_DIR, "warm_vs_cold_start.pdf")

# Window of the centred rolling median. Chosen against the cold run, which is
# the harder of the two to read: its cost falls 11.63 units over 1000 steps, so
# the underlying trend is about 0.29 units per 25 steps, while the per-step
# sampling noise is about 1.9 units. At a 25 step window the smoothed line still
# jitters 0.16 units step to step, which is over half the trend it is meant to
# show. At 51 steps the jitter drops to 0.09 units against 0.59 units of trend
# over the same span, a ratio of about one in seven, so the cold trend reads
# cleanly without the window growing large enough to flatten the ends.
SMOOTH_WINDOW = 51

CLIP_THRESHOLD = 1.0
NUM_STEPS = 1000


def load_history(path: str):
    """Return (cost_history, grad_norm_history) as arrays, with checks.

    Reads the exported per-step CSV rather than a checkpoint. The checks that
    used to compare the history length against the checkpoint's stored ``step``
    are replaced by a check that the ``step`` column is exactly 1 to 1000 with
    no gap and no duplicate, which is the same guarantee expressed on the data
    the CSV actually carries.
    """
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    steps = np.asarray([int(r["step"]) for r in rows], dtype=int)
    cost = np.asarray([float(r["cost_history"]) for r in rows], dtype=float)
    grad = np.asarray([float(r["grad_norm_history"]) for r in rows], dtype=float)

    if len(cost) != NUM_STEPS:
        raise ValueError(f"{path}: expected {NUM_STEPS} steps, got {len(cost)}")
    if not np.array_equal(steps, np.arange(1, NUM_STEPS + 1)):
        raise ValueError(
            f"{path}: the step column is not exactly 1 to {NUM_STEPS} with one "
            "row each, so steps are missing, duplicated or out of order"
        )
    if not np.isfinite(cost).all() or not np.isfinite(grad).all():
        raise ValueError(f"{path}: history contains non-finite entries")
    return cost, grad


def rolling_median(series: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling median, shrinking the window at the two ends."""
    n = len(series)
    half = window // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(series[lo:hi])
    return out


def main() -> None:
    apply_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    warm_cost, warm_grad = load_history(WARM_SERIES)
    cold_cost, cold_grad = load_history(COLD_SERIES)
    steps = np.arange(1, NUM_STEPS + 1)

    warm_smooth = rolling_median(warm_cost, SMOOTH_WINDOW)
    cold_smooth = rolling_median(cold_cost, SMOOTH_WINDOW)
    warm_grad_smooth = rolling_median(warm_grad, SMOOTH_WINDOW)
    cold_grad_smooth = rolling_median(cold_grad, SMOOTH_WINDOW)

    fig, (ax_cost, ax_grad) = plt.subplots(
        1,
        2,
        figsize=figure_size(1.0, 0.44),
        sharex=True,
    )

    # Left panel, training cost. Raw series faint, rolling median on top.
    ax_cost.plot(
        steps, warm_cost, color=COLOR_WARM, linewidth=RAW_LINEWIDTH,
        alpha=RAW_ALPHA, zorder=2,
    )
    ax_cost.plot(
        steps, cold_cost, color=COLOR_COLD, linewidth=RAW_LINEWIDTH,
        alpha=RAW_ALPHA, zorder=2,
    )
    line_warm, = ax_cost.plot(
        steps, warm_smooth, label=WARM_LABEL, zorder=4, **WARM_STYLE
    )
    line_cold, = ax_cost.plot(
        steps, cold_smooth, label=COLD_LABEL, zorder=3, **COLD_STYLE
    )

    hairline_grid(ax_cost, axis="y")
    ax_cost.set_xlabel("Training step")
    ax_cost.set_ylabel("Training cost")
    ax_cost.set_xlim(1, NUM_STEPS)
    ax_cost.text(
        0.0, 1.02, "a", transform=ax_cost.transAxes, fontsize=PANEL_LETTER_PT,
        fontweight="bold", ha="left", va="bottom",
    )

    # Right panel, gradient norm before clipping, logarithmic. Same treatment
    # as the left panel, since at 1000 points per run the raw traces overplot
    # into two solid bands and the dashed and solid line styles stop being
    # distinguishable.
    ax_grad.plot(
        steps, warm_grad, color=COLOR_WARM, linewidth=RAW_LINEWIDTH,
        alpha=RAW_ALPHA, zorder=2,
    )
    ax_grad.plot(
        steps, cold_grad, color=COLOR_COLD, linewidth=RAW_LINEWIDTH,
        alpha=RAW_ALPHA, zorder=2,
    )
    ax_grad.plot(steps, warm_grad_smooth, zorder=4, **WARM_STYLE)
    ax_grad.plot(steps, cold_grad_smooth, zorder=3, **COLD_STYLE)
    ax_grad.set_yscale("log")
    hairline_grid(ax_grad, axis="y")
    ax_grad.set_xlabel("Training step")
    ax_grad.set_ylabel("Gradient norm before clipping")
    ax_grad.set_xlim(1, NUM_STEPS)

    # The clipping threshold sits two decades below the smallest observed norm,
    # so drawing it would spend most of the panel on empty axis. The y limits
    # follow the data instead. That every step was clipped is still recorded in
    # the sidecar.
    ax_grad.autoscale(axis="y")
    ax_grad.text(
        0.0, 1.02, "b", transform=ax_grad.transAxes, fontsize=PANEL_LETTER_PT,
        fontweight="bold", ha="left", va="bottom",
    )

    for ax in (ax_cost, ax_grad):
        ax.set_xticks([1, 250, 500, 750, 1000])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    # One legend for the whole figure, under both panels. tight_layout runs
    # first, reserving only the band the 8 pt legend actually needs, so the
    # freed margin goes to the two plot areas rather than to whitespace.
    fig.tight_layout(pad=0.3, w_pad=1.3, rect=(0.0, 0.105, 1.0, 1.0))
    fig.legend(
        handles=[line_warm, line_cold],
        labels=[WARM_LABEL, COLD_LABEL],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.savefig(OUT_PDF)
    plt.close(fig)

    def fmt(value):
        return f"{value:.4f}"

    extra = {
        "steps covered": f"1 to {NUM_STEPS}, both runs, both series, no gaps",
        "rollouts per step": "2560 (batch 128 instances, RLOO k=20)",
        "warm cost at steps 100/500/1000": ", ".join(
            fmt(warm_cost[s - 1]) for s in (100, 500, 1000)
        ),
        "cold cost at steps 100/500/1000": ", ".join(
            fmt(cold_cost[s - 1]) for s in (100, 500, 1000)
        ),
        "warm grad norm at steps 100/500/1000": ", ".join(
            fmt(warm_grad[s - 1]) for s in (100, 500, 1000)
        ),
        "cold grad norm at steps 100/500/1000": ", ".join(
            fmt(cold_grad[s - 1]) for s in (100, 500, 1000)
        ),
        "warm cost first-difference std": fmt(np.diff(warm_cost).std(ddof=1)),
        "cold cost first-difference std": fmt(np.diff(cold_cost).std(ddof=1)),
        "warm residual std about rolling median": fmt(
            (warm_cost - warm_smooth).std(ddof=1)
        ),
        "cold residual std about rolling median": fmt(
            (cold_cost - cold_smooth).std(ddof=1)
        ),
        "warm cost fall, step 1 to 1000": fmt(warm_cost[0] - warm_cost[-1]),
        "cold cost fall, step 1 to 1000": fmt(cold_cost[0] - cold_cost[-1]),
        "steps clipped at 1.0": (
            f"warm {(warm_grad > CLIP_THRESHOLD).sum()}/{NUM_STEPS}, "
            f"cold {(cold_grad > CLIP_THRESHOLD).sum()}/{NUM_STEPS}"
        ),
    }

    notes = [
        "Runs plotted. Warm is the production run whose best checkpoint is "
        "checkpoints/sweep_G_v4warmstart_best.pt (step 750); the per-step "
        "histories originate in its rolling checkpoint sweep_G_v4warmstart.pt. "
        "Cold is DM-17, best checkpoint sweep_G_coldstart_best.pt (step 975), "
        "rolling checkpoint sweep_G_coldstart.pt. Rolling and best checkpoints "
        "carry byte-identical configs in each run, so the pairing is exact.",
        "Source of the plotted series. This figure reads two tracked CSVs, "
        "provenance/figure41_training/series_{warm,cold}_start_training.csv, exported from "
        "the two rolling checkpoints by scripts/regenerate/export_series_and_logs.py at "
        "full float precision. The exported values are bitwise identical to "
        "the checkpoint arrays, verified by numpy array_equal on both series "
        "of both arms, so this figure is the published one and no checkpoint "
        "is opened to build it.",
        "Both panels smoothed: centred rolling median over "
        f"{SMOOTH_WINDOW} steps, window shrunk at the two ends. The raw series "
        "is drawn faint behind the smoothed line for both runs in both panels, "
        "so nothing is hidden. The cost panel needs it because the cold run's "
        "total fall of 11.63 units is comparable to the 1.9 unit per-step "
        "sampling noise; the gradient panel needs it because 1000 raw points "
        "per run overplot into two solid bands.",
        "cost_history is metrics['mean_cost'], the unweighted mean over the "
        "2560 rollouts of that step's minibatch of the combined episode cost "
        "w_dist*D + w_make*M + w_bal*B from "
        "DynamicMRTASimulator.compute_cost. Rollouts that fail to serve every "
        "task are included in that mean and are NOT penalised for the tasks "
        "they dropped: the cost is computed over the commitments actually "
        "made, so an episode that drops tasks contributes less distance. The "
        "runs predate eval_history, so the training-time serve-all rate cannot "
        "be recovered from these checkpoints.",
        "grad_norm_history is the value returned by clip_grad_norm_ in "
        "rollout_worker.reduce_and_apply, which is the total norm of the "
        "summed worker gradient BEFORE clipping. Both runs used "
        "gradient_clip_norm 1.0 and every one of the 1000 steps exceeded it, "
        "so clipping was active on every step of both runs. The threshold is "
        "not drawn in panel b: the smallest observed norm is 6.31, two decades "
        "above the threshold, so the reference line would have pushed the data "
        "into the top third of the panel. The y limits follow the data and the "
        "clipped-step counts are recorded under Values above instead.",
        "Comparability. Both runs draw from the same 1000 instance training "
        "cache (cache/training_set_il_v3/train), batch_size 128, "
        "baseline_mode rloo with rloo_k 20, so both series are means over 2560 "
        "rollouts per point. Configs differ only in warm_start_from, "
        "resume_from, start_step, checkpoint_path and best_serve_all_floor, "
        "none of which changes what cost_history measures.",
        "Minibatch selection is unseeded (random.sample in "
        "src/training/method_two_trainer.py, DM-44), so the two runs see "
        "different instance samples at each step. At 2560 rollouts per point "
        "the resulting per-step noise is about 1.9 units against a "
        "between-run separation of about 22 units at step 1000, so the "
        "sampling difference is roughly a tenth of the effect and unbiased in "
        "either direction.",
        "The cold run was resumed once at step 801 (start_step 801, "
        "resume_from its own rolling checkpoint). The trainer reloads the "
        "stored history and sets start_step to the stored step plus one, so "
        "the two segments concatenate without gap or duplication. Both "
        "histories hold exactly 1000 entries against a final step of 1000, "
        "which is only possible if every step contributed exactly one entry.",
        "No validation gap closure is plotted in this figure.",
    ]

    sidecar = write_sidecar(
        OUT_PDF,
        sources=[
            WARM_SERIES,
            COLD_SERIES,
            os.path.join(
                REPO_ROOT, "scripts", "regenerate", "export_series_and_logs.py"
            ),
            os.path.join(REPO_ROOT, "src", "training", "method_two_trainer.py"),
            os.path.join(REPO_ROOT, "src", "training", "rollout_worker.py"),
            os.path.join(REPO_ROOT, "src", "simulator", "dynamic_simulator.py"),
            os.path.join(REPO_ROOT, "figures", "style.py"),
            os.path.abspath(__file__),
        ],
        notes=notes,
        extra=extra,
    )

    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {sidecar}")


if __name__ == "__main__":
    main()
