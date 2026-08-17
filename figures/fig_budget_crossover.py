"""The advantage flips as instances grow.

One panel, one line, three points. Mean cost of the rolling horizon at a one
second solve budget divided by the mean cost of the learned policy, on the same
instances at each scale. Above one the learned policy is cheaper, below one the
rolling horizon is cheaper. The line crosses one between six robots and ten.

Every number is computed here from the per-instance CSVs. Nothing is read from a
summary table, a report or a JSON of aggregates.

Intervals are 95 per cent percentile bootstrap over 10,000 resamples of
instances, paired, with a fixed seed of 20260731 matching the convention used
throughout the provenance record. Pairing matters: the rolling horizon and the
learned policy ran on identical instances at every scale, so a resample draws
an instance and takes both policies' costs on it.

Reads CSVs, writes a PDF and a sidecar. Solves nothing, evaluates nothing, and
modifies no cache record.

Run with
    PYTHONPATH="$PWD/src" python figures/fig_budget_crossover.py
"""

from __future__ import annotations

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import (  # noqa: E402
    ANNOTATION_PT,
    COLOR_REFERENCE,
    COLOR_WARM,
    apply_style,
    figure_size,
    hairline_grid,
    write_sidecar,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def prov(*parts):
    return os.path.join(REPO_ROOT, "provenance", *parts)


def tracked_or_results(name):
    """Prefer the tracked copy under provenance/ over the one under results/.

    Both copies are byte identical. results/ is gitignored, so a script that
    reads it works in the tree that produced it and fails in a fresh clone,
    which is where an examiner starts. The provenance copy is checked first for
    that reason, and results/ is kept as a fallback so nothing breaks in a tree
    that predates the copy being made.
    """
    tracked = prov("table41_main_results", name)
    return tracked if os.path.exists(tracked) else os.path.join(
        REPO_ROOT, "results", name)


# Training scale. The one second rung of the seven budget test split ladder,
# against the headline test evaluation of the learned policy. Both are on the
# frozen test split, seeds 11200 to 11399.
RH_R6 = prov("figure42_budget", "a3_rh_test_h1_b1.0s.csv")
LEARNED_R6 = tracked_or_results("a2_per_instance_test_20260730.csv")
# Wall clock for the training scale ladder was recorded per budget for the whole
# sweep rather than per instance, so this log is the only route to a compute
# figure at that scale that includes model construction.
A3_WALLTIMES = prov("figure42_budget", "a3_ladder_walltimes.txt")
TIMING_R6 = prov("figure42_budget", "b3_timing_test_20260731.csv")

# Transfer scales. The one second rung of the four budget ladder, against the
# transfer table evaluation of the learned policy.
LADDER = prov("table42_transfer", "rh_ladder_per_instance.csv")
TRANSFER = {
    "R=10,T=30": prov(
        "table42_transfer", "transfer_r10t30_per_instance.csv"
    ),
    "R=30,T=90": prov(
        "table42_transfer", "transfer_r30t90_per_instance.csv"
    ),
}

TRANSFER_TIMING = {
    "R=10,T=30": prov("table42_transfer", "timing_r10t30.csv"),
    "R=30,T=90": prov("table42_transfer", "timing_r30t90.csv"),
}

OUT_DIR = os.path.join(REPO_ROOT, "figures", "output")
OUT_PDF = os.path.join(OUT_DIR, "budget_crossover.pdf")

BUDGET_SECONDS = 1.0
BOOTSTRAP_SEED = 20260731
BOOTSTRAP_DRAWS = 10000

SCALES = ["R=6,T=18", "R=10,T=30", "R=30,T=90"]
TICK_LABELS = {
    "R=6,T=18": "R=6\nT=18",
    "R=10,T=30": "R=10\nT=30",
    "R=30,T=90": "R=30\nT=90",
}


def paired_ratio_ci(rh_costs, learned_costs):
    """Ratio of mean costs with a paired percentile bootstrap interval.

    One resample draws instance indices with replacement and takes BOTH
    policies' costs on those instances, so the correlation between the two
    policies on an instance is preserved and the interval is on the ratio, not
    on either mean separately.
    """
    rh = np.asarray(rh_costs, dtype=float)
    learned = np.asarray(learned_costs, dtype=float)
    if rh.shape != learned.shape:
        raise ValueError("paired arrays must have the same length")
    point = rh.mean() / learned.mean()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(rh)
    idx = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    draws = rh[idx].mean(axis=1) / learned[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), n


def load_training_scale():
    """R=6,T=18 on the frozen test split, seeds 11200 to 11399."""
    rh = pd.read_csv(RH_R6)
    learned = pd.read_csv(LEARNED_R6)
    learned = learned[learned.decode_mode == "hard"]
    if set(rh.seed) != set(learned.seed):
        raise ValueError(
            f"{RH_R6} and {LEARNED_R6} do not cover the same instances"
        )
    if not (rh.budget_seconds == BUDGET_SECONDS).all():
        raise ValueError(f"{RH_R6} is not the {BUDGET_SECONDS} s budget")
    merged = rh[["seed", "policy_cost"]].rename(
        columns={"policy_cost": "rh_cost"}
    ).merge(
        learned[["seed", "policy_cost"]].rename(
            columns={"policy_cost": "learned_cost"}
        ),
        on="seed",
    )
    return merged


def load_transfer_scale(scale):
    """R=10,T=30 or R=30,T=90, ladder against the transfer table evaluation."""
    ladder = pd.read_csv(LADDER)
    ladder = ladder[
        (ladder.scale == scale) & (ladder.budget_seconds == BUDGET_SECONDS)
    ]
    if ladder.empty:
        raise ValueError(f"{LADDER}: no {scale} rows at {BUDGET_SECONDS} s")
    transfer = pd.read_csv(TRANSFER[scale])
    if set(ladder.seed) != set(transfer.seed):
        raise ValueError(
            f"{scale}: ladder seeds differ from {TRANSFER[scale]}"
        )
    return ladder[["seed", "cost"]].rename(columns={"cost": "rh_cost"}).merge(
        transfer[["seed", "policy_cost"]].rename(
            columns={"policy_cost": "learned_cost"}
        ),
        on="seed",
    )


def parse_ladder_walltimes(path):
    """Map budget in seconds to (wall clock seconds, rows) for the whole sweep."""
    out = {}
    pattern = re.compile(
        r"^budget=([\d.]+)\s+rc=(\d+)\s+wall_seconds=(\d+)\s+rows=(\d+)"
    )
    with open(path) as handle:
        for line in handle:
            match = pattern.match(line.strip())
            if match:
                budget, rc, wall, rows = match.groups()
                if int(rc) != 0:
                    raise ValueError(f"{path}: budget {budget} exited rc={rc}")
                out[float(budget)] = (int(wall), int(rows))
    if not out:
        raise ValueError(f"{path}: no budget lines parsed")
    return out


def compute_multiples():
    """Compute per decision the rolling horizon used, over the policy's own.

    Both sides are measured, not nominal. The rolling horizon figure is wall
    clock per instance divided by the dispatcher invocations that reached the
    solver, so it includes model construction, which sits outside the Gurobi
    time limit. The learned policy figure is the median whole-decision wall
    clock from the timing runs. Every value is for the 1 s budget, which is the
    budget this figure plots, not for any other rung of the ladder.
    """
    out = {}

    ladder6 = pd.read_csv(RH_R6)
    wall_seconds, rows_recorded = parse_ladder_walltimes(A3_WALLTIMES)[
        BUDGET_SECONDS
    ]
    if rows_recorded != len(ladder6):
        raise ValueError(f"{RH_R6}: row count disagrees with the wall clock log")
    timing6 = pd.read_csv(TIMING_R6)
    learned_ms = timing6[timing6.policy == "learned_hard"].t_total_s.median() * 1000.0
    out["R=6,T=18"] = dict(
        rh_ms=wall_seconds / len(ladder6) / ladder6.n_window_solves.mean() * 1000.0,
        learned_ms=learned_ms,
    )

    ladder = pd.read_csv(LADDER)
    for scale in ("R=10,T=30", "R=30,T=90"):
        group = ladder[
            (ladder.scale == scale) & (ladder.budget_seconds == BUDGET_SECONDS)
        ]
        timing = pd.read_csv(TRANSFER_TIMING[scale])
        out[scale] = dict(
            rh_ms=(group.wall_seconds / group.n_window_solves).mean() * 1000.0,
            learned_ms=timing.t_total_s.median() * 1000.0,
        )

    for value in out.values():
        value["multiple"] = value["rh_ms"] / value["learned_ms"]
    return out


def main() -> None:
    apply_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    paired = {"R=6,T=18": load_training_scale()}
    for scale in ("R=10,T=30", "R=30,T=90"):
        paired[scale] = load_transfer_scale(scale)

    results = {}
    for scale in SCALES:
        data = paired[scale]
        point, lo, hi, n = paired_ratio_ci(data.rh_cost, data.learned_cost)
        results[scale] = dict(
            ratio=point, lo=lo, hi=hi, n=n,
            rh_mean=data.rh_cost.mean(),
            learned_mean=data.learned_cost.mean(),
        )

    x = np.arange(len(SCALES), dtype=float)
    ratios = np.array([results[s]["ratio"] for s in SCALES])
    lows = np.array([results[s]["lo"] for s in SCALES])
    highs = np.array([results[s]["hi"] for s in SCALES])

    compute = compute_multiples()

    # Full text width, wider than it is high. Three categorical positions in a
    # near-square panel read as a tall chart of very little; at this aspect the
    # line's rise across the scales is what the panel is shaped around.
    fig, ax = plt.subplots(figsize=figure_size(1.0, 0.45))

    # Parity first, so the data sits on top of it.
    ax.axhline(
        1.0, color=COLOR_REFERENCE, linewidth=0.7, linestyle=(0, (4, 2)),
        zorder=2,
    )

    ax.errorbar(
        x, ratios,
        yerr=np.vstack([ratios - lows, highs - ratios]),
        color=COLOR_WARM, linewidth=1.2, marker="o", markersize=4.0,
        capsize=2.5, capthick=0.8, elinewidth=0.8, zorder=4,
    )

    # The compute each point was bought with, above the top of its interval so
    # nothing collides with the error bars or with the parity line.
    for i, scale in enumerate(SCALES):
        ax.text(
            i, highs[i] + 0.010,
            f"{compute[scale]['multiple']:.0f}× compute",
            fontsize=ANNOTATION_PT, color="#777777", ha="center", va="bottom",
        )

    # Region labels rather than shading. Shading would put two large blocks of
    # tone behind a figure whose entire content is one line, and the line is
    # what should carry the eye.
    label_x = 0.985
    ax.text(
        label_x, 1.004, "learned policy cheaper",
        transform=ax.get_yaxis_transform(), ha="right", va="bottom",
        fontsize=ANNOTATION_PT, color="#555555",
    )
    ax.text(
        label_x, 0.996, "rolling horizon cheaper",
        transform=ax.get_yaxis_transform(), ha="right", va="top",
        fontsize=ANNOTATION_PT, color="#555555",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([TICK_LABELS[s] for s in SCALES])
    ax.set_xlim(-0.4, 2.4)
    ax.set_xlabel("Instance size")
    ax.set_ylabel(
        "Rolling horizon cost / learned policy cost\n"
        "(rolling horizon at a 1 s solve budget)"
    )
    hairline_grid(ax, axis="y")

    span = highs.max() - lows.min()
    ax.set_ylim(lows.min() - 0.25 * span, highs.max() + 0.25 * span)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    extra = {}
    for scale in SCALES:
        r = results[scale]
        extra[scale] = (
            f"ratio {r['ratio']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}], "
            f"n={r['n']}, RH mean {r['rh_mean']:.4f}, "
            f"learned mean {r['learned_mean']:.4f}"
        )
    for scale in SCALES:
        c = compute[scale]
        extra[f"{scale} compute"] = (
            f"{c['multiple']:.1f}x (rolling horizon {c['rh_ms']:.2f} ms per "
            f"decision, learned policy {c['learned_ms']:.4f} ms)"
        )
    extra["bootstrap"] = (
        f"{BOOTSTRAP_DRAWS} paired resamples, percentile interval, "
        f"numpy default_rng seed {BOOTSTRAP_SEED}"
    )
    extra["crossover"] = (
        "below one at R=6,T=18 and above one at both larger scales, so the "
        "advantage flips between six robots and ten"
    )

    notes = [
        "Quantity plotted: mean cost of the rolling horizon at a 1 s solve "
        "budget divided by mean cost of the learned policy, ratio of means, on "
        "the same instances at each scale. Above one the learned policy is "
        "cheaper. Computed here from the per-instance CSVs listed above; no "
        "value is taken from any summary table, report or aggregate JSON.",
        "Instance matching is asserted, not assumed. The script raises if the "
        "rolling horizon and learned policy seed sets differ at any scale. All "
        "three passed: n=200 on seeds 11200 to 11399 at R=6,T=18, n=60 on "
        "96000 to 96059 at R=10,T=30, and n=60 on 96100 to 96159 at R=30,T=90.",
        "Provenance of the two transfer scales. The 1 s rolling horizon costs "
        "are the 1.0 s rung of rh_ladder_per_instance.csv and the learned "
        "policy costs are from the transfer table's own per-instance files, so "
        "these points are the same runs the transfer table was built from. "
        "Confirmed numerically: the learned policy means computed here, "
        "173.3594 and 354.7814, reproduce the figures quoted in "
        "rh_ladder_report.md, 173.359 and 354.781.",
        "Provenance of the training scale. R=6,T=18 is NOT from the transfer "
        "table. Its rolling horizon point is the 1.0 s rung of the seven "
        "budget A3 ladder on the frozen test split and its learned policy "
        "costs are the headline test evaluation, both on seeds 11200 to 11399. "
        "The transfer table's own r6t18 file covers seeds 11000 to 11199, a "
        "different split, and is deliberately not used, because pairing the "
        "ladder against it would compare costs on different instances.",
        "Intervals are paired: one bootstrap resample draws instances and "
        "takes both policies' costs on the drawn instances, preserving the "
        "correlation between the two policies on an instance. An unpaired "
        "interval on this quantity would be wider and would not answer the "
        "question the figure asks.",
        "The R=10,T=30 interval straddles one. The crossover is therefore "
        "located between six and ten robots, but parity at ten robots is not "
        "resolved by this evidence and the figure should not be read as "
        "showing the learned policy ahead at that scale.",
        "Compute annotations. Each point is labelled with the compute the "
        "rolling horizon used at that point divided by the learned policy's "
        "own compute at the same scale. The rolling horizon figure is measured "
        "wall clock per instance over the dispatcher invocations that reached "
        "the solver, so it includes model construction, which sits outside the "
        "Gurobi time limit. The learned policy figure is the median "
        "whole-decision wall clock from the timing runs, 1.1092 ms, 1.4687 ms "
        "and 3.5840 ms. All three are for the 1 s budget, which is the budget "
        "this figure plots.",
        "The compute advantage is large at every point but it is NOT monotone: "
        "203x, 341x, 284x. It falls from ten robots to thirty because the same "
        "1 s budget is being divided by a learned policy that is itself slower "
        "at the larger scale, 3.5840 ms against 1.4687 ms per decision, not "
        "because the rolling horizon was given less absolute compute. In "
        "absolute terms it rises throughout, 225.61 ms, 500.39 ms and 1016.48 "
        "ms per decision. The figure should not be captioned as a growing "
        "compute advantage.",
        "No solver was invoked, nothing was evaluated and no cache record was "
        "read or modified by this script. It reads CSVs only.",
    ]

    sources = [
        RH_R6,
        A3_WALLTIMES,
        LEARNED_R6,
        TIMING_R6,
        LADDER,
        TRANSFER["R=10,T=30"],
        TRANSFER["R=30,T=90"],
        TRANSFER_TIMING["R=10,T=30"],
        TRANSFER_TIMING["R=30,T=90"],
        os.path.join(REPO_ROOT, "figures", "style.py"),
        os.path.abspath(__file__),
    ]
    sidecar = write_sidecar(OUT_PDF, sources=sources, notes=notes, extra=extra)

    for scale in SCALES:
        r, c = results[scale], compute[scale]
        print(
            f"{scale:10s} ratio {r['ratio']:.4f} "
            f"[{r['lo']:.4f}, {r['hi']:.4f}]  n={r['n']:3d}  "
            f"RH {r['rh_mean']:8.4f} / learned {r['learned_mean']:8.4f}   "
            f"compute {c['rh_ms']:7.2f} ms / {c['learned_ms']:.4f} ms "
            f"= {c['multiple']:5.1f}x"
        )
    print(f"\nWrote {OUT_PDF}")
    print(f"Wrote {sidecar}")


if __name__ == "__main__":
    main()
