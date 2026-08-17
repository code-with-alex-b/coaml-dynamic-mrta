"""Gap closure against measured compute for the seven budget A3 ladder.

One panel. The rolling horizon at h=1 on the frozen test split, seeds 11200 to
11399, at seven solve budgets from 10 ms to 10 s, plotted against the compute it
actually used rather than against the budget it was given. The learned policy
enters as a horizontal reference at its own gap closure, with its own measured
per-decision compute marked on the x axis, so the budget at which the rolling
horizon overtakes it can be read straight off the crossing.

Why measured compute and not the nominal budget. The nominal budget is a
Gurobi TimeLimit applied to model.optimize() alone. It bounds neither the model
construction that precedes each solve nor the solves that finish early, so it
is wrong in both directions: at 10 ms the measured cost per decision is above
the budget because construction is not inside it, and at 10 s it is far below
because almost every solve terminates long before the limit. Plotting against
the nominal budget would therefore misstate the compute axis by up to a factor
of nineteen. Both columns are recorded in the sidecar.

Quantities. Gap closure is the ratio of mean costs,
(mean greedy - mean policy) / (mean greedy - mean MILP), on the 200 test
instances, matching the convention of Table 4.1 in the results chapter. It is
recomputed here from the per-instance CSVs; no value is read from any summary
table or report. The distance-only Hungarian supplies the floor and the cached
MILP objective the ceiling.

No solver is invoked, nothing is evaluated, nothing is trained, and no cache
record is read or modified. This script reads CSVs and one text log.
"""

from __future__ import annotations

import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import (  # noqa: E402
    ANNOTATION_PT,
    COLOR_COLD,
    COLOR_WARM,
    apply_style,
    figure_size,
    hairline_grid,
    write_sidecar,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo(*parts: str) -> str:
    return os.path.join(REPO_ROOT, *parts)


OUT_DIR = repo("figures", "output")
OUT_PDF = os.path.join(OUT_DIR, "budget_curve.pdf")

# The seven rungs, as (nominal budget in seconds, filename stem).
BUDGETS = [
    (0.010, "0.010"),
    (0.025, "0.025"),
    (0.050, "0.050"),
    (0.100, "0.100"),
    (0.250, "0.250"),
    (1.0, "1.0"),
    (10.0, "10.0"),
]

# The two shortest rungs are drawn with open markers. At these budgets the
# window solve frequently returns no incumbent and the epoch falls back to a
# Hungarian-on-kappa assignment, so a substantial share of the decisions were
# not made by the rolling horizon at all. The threshold is stated rather than
# assumed: any rung whose fallback share exceeds this is drawn open, and the
# script asserts that exactly the two shortest rungs qualify.
FALLBACK_OPEN_THRESHOLD_PCT = 10.0

def tracked_or_results(name: str) -> str:
    """Prefer the tracked copy under provenance/ over the one under results/.

    Both copies are byte identical. results/ is gitignored, so a script that
    reads it works in the tree that produced it and fails in a fresh clone,
    which is where an examiner starts. The provenance copy is checked first for
    that reason, and results/ is kept as a fallback so nothing breaks in a tree
    that predates the copy being made.
    """
    tracked = repo("provenance", "table41_main_results", name)
    return tracked if os.path.exists(tracked) else repo("results", name)


WALLTIMES = repo("provenance", "figure42_budget", "a3_ladder_walltimes.txt")
HUNGARIAN = repo("provenance", "table41_main_results", "b3_hungarian_distance_only_test_20260731.csv")
LEARNED = tracked_or_results("a2_per_instance_test_20260730.csv")
TIMING = repo("provenance", "figure42_budget", "b3_timing_test_20260731.csv")
STYLE = repo("figures", "style.py")
THIS = os.path.abspath(__file__)


def rung_csv(stem: str) -> str:
    return repo("provenance", "figure42_budget", f"a3_rh_test_h1_b{stem}s.csv")


def parse_walltimes(path: str) -> dict:
    """Map nominal budget in seconds to (wall clock seconds, rows recorded).

    This is the whole-sweep wall clock for one rung of the ladder, so it
    includes model construction, the simulator step and every epoch that
    advanced without a solve. It is the only measured timing in the project
    that brackets the entire rolling-horizon episode.
    """
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


def ratio_of_means(greedy, policy, milp) -> float:
    """Gap closure as the ratio of mean costs, the Table 1 convention."""
    denominator = greedy.mean() - milp.mean()
    if denominator <= 1e-9:
        raise ValueError("degenerate gap-closure denominator")
    return (greedy.mean() - policy.mean()) / denominator


def load_points():
    """Build the seven ladder points, each measured rather than nominal."""
    hungarian = pd.read_csv(HUNGARIAN).set_index("seed")["policy_cost"]
    walltimes = parse_walltimes(WALLTIMES)

    rows = []
    for budget, stem in BUDGETS:
        frame = pd.read_csv(rung_csv(stem)).set_index("seed").sort_index()

        # Every rung must cover the same instances as the floor, otherwise the
        # gap closure denominator would be built on a different instance set.
        if list(frame.index) != sorted(hungarian.index):
            raise ValueError(f"{rung_csv(stem)}: seed set differs from the floor")
        if int(frame.serve_all_flag.sum()) != len(frame):
            raise ValueError(f"{rung_csv(stem)}: not every instance served all")
        if not np.allclose(frame.budget_seconds.values, budget):
            raise ValueError(f"{rung_csv(stem)}: budget column disagrees")

        wall_seconds, rows_recorded = walltimes[budget]
        if rows_recorded != len(frame):
            raise ValueError(f"{rung_csv(stem)}: row count disagrees with the log")

        greedy = hungarian.loc[frame.index].values
        policy = frame.policy_cost.values
        milp = frame.milp_oracle_cost_from_cache.values

        # Measured compute per decision. Wall clock for the whole rung, over
        # the instances, over the dispatcher invocations that reached the
        # solver. Dividing by n_window_solves rather than n_decisions is the
        # convention already used by fig_budget_crossover.py, and it is the
        # denominator the nominal budget should be compared against, because
        # the budget applies once per solve and epochs with nothing pending
        # advance without one. The alternative denominator is recorded in the
        # sidecar so the choice is visible.
        per_solve_ms = wall_seconds / len(frame) / frame.n_window_solves.mean() * 1e3
        per_epoch_ms = wall_seconds / len(frame) / frame.n_decisions.mean() * 1e3

        fallback_pct = (
            frame.n_solves_fallback_fired.sum() / frame.n_window_solves.sum() * 100.0
        )

        rows.append(
            dict(
                budget_s=budget,
                nominal_ms=budget * 1e3,
                measured_ms=per_solve_ms,
                measured_ms_per_epoch=per_epoch_ms,
                gap_pct=ratio_of_means(greedy, policy, milp) * 100.0,
                fallback_pct=fallback_pct,
                wall_seconds=wall_seconds,
                n_window_solves=frame.n_window_solves.mean(),
                n_decisions=frame.n_decisions.mean(),
                n=len(frame),
            )
        )

    points = pd.DataFrame(rows)

    # The open-marker rule is asserted, not hand-placed. Exactly the two
    # shortest rungs must be the ones above the fallback threshold; if a future
    # re-run changes that, the figure should fail rather than mislabel.
    open_mask = points.fallback_pct > FALLBACK_OPEN_THRESHOLD_PCT
    if list(np.flatnonzero(open_mask.values)) != [0, 1]:
        raise ValueError(
            "the rungs above the fallback threshold are no longer the two "
            f"shortest: {points.loc[open_mask, 'budget_s'].tolist()}"
        )
    points["open_marker"] = open_mask
    return points


def load_learned():
    """The learned policy's gap closure and its own measured compute."""
    hungarian = pd.read_csv(HUNGARIAN).set_index("seed")["policy_cost"]
    frame = pd.read_csv(LEARNED)
    frame = frame[frame.decode_mode == "hard"].set_index("seed").sort_index()
    if list(frame.index) != sorted(hungarian.index):
        raise ValueError(f"{LEARNED}: seed set differs from the floor")

    gap_pct = (
        ratio_of_means(
            hungarian.loc[frame.index].values,
            frame.policy_cost.values,
            frame.cost_milp_oracle.values,
        )
        * 100.0
    )

    # Split-consistent per-decision compute, the median whole decision over the
    # test-split timing run. This is the value the figure marks, so the compute
    # axis and the cost axis are measured on the same 200 instances.
    timing = pd.read_csv(TIMING)
    test_median_ms = (
        timing[timing.policy == "learned_hard"].t_total_s.median() * 1e3
    )
    return gap_pct, test_median_ms, len(frame)


# The learned policy's compute mark is the median whole decision from the TEST
# split timing run, 1.1092 ms over 2587 decisions, taken from
# b3_timing_test_20260731.csv. It is read from that file rather than written
# here as a literal, so the mark cannot drift from the data. The whole figure
# is therefore on one split: gap closure, costs and compute all come from the
# 200 test instances on seeds 11200 to 11399. The A6 timing run's 0.92 ms
# median is a VALIDATION-split measurement over 50 instances and is
# deliberately not used.


def main() -> None:
    apply_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    points = load_points()
    learned_gap_pct, learned_test_ms, n_learned = load_learned()
    # Single source for the compute mark, read from the test-split timing run.
    learned_ms = learned_test_ms

    crossing = bracket_crossing(points, learned_gap_pct, learned_ms)

    fig, ax = plt.subplots(figsize=figure_size(1.0, 0.45))

    # The rolling horizon is the baseline, so it recedes: mid grey, thinner
    # than the default, grey markers. The learned policy carries the accent.
    # This inverts the earlier draft, where the accent sat on the baseline and
    # pulled the eye to the wrong series.
    ax.plot(
        points.measured_ms,
        points.gap_pct,
        color=COLOR_COLD,
        linewidth=0.9,
        linestyle="-",
        marker="o",
        markersize=3.2,
        markevery=list(np.flatnonzero(~points.open_marker.values)),
        zorder=2,
    )
    opened = points[points.open_marker]
    ax.plot(
        opened.measured_ms,
        opened.gap_pct,
        marker="o",
        markersize=3.2,
        markerfacecolor="white",
        markeredgecolor=COLOR_COLD,
        markeredgewidth=0.8,
        linestyle="none",
        zorder=3,
    )

    # Learned policy: accent throughout, so the reader lands here first. The
    # star is sized to sit alongside the ladder's markers rather than to
    # dominate them; the accent colour already carries the emphasis.
    ax.axhline(
        learned_gap_pct,
        color=COLOR_WARM,
        linewidth=0.9,
        linestyle=(0, (4, 2)),
        zorder=5,
    )
    ax.plot(
        [learned_ms],
        [learned_gap_pct],
        marker="*",
        markersize=6.0,
        color=COLOR_WARM,
        linestyle="none",
        zorder=7,
    )
    # The policy's compute is stated beside its own marker, so the reader does
    # not have to track down to the axis to find it.
    ax.annotate(
        f"{learned_ms:.2f} ms",
        xy=(learned_ms, learned_gap_pct),
        xytext=(6, 4),
        textcoords="offset points",
        fontsize=ANNOTATION_PT,
        color=COLOR_WARM,
        ha="left",
        va="bottom",
    )

    # Nominal budget beside each point, which is the whole reason the x axis is
    # measured compute: the reader can see the two diverge, and by how much.
    for row in points.itertuples():
        label = (
            f"{row.budget_s:g} s" if row.budget_s >= 1 else f"{int(row.nominal_ms)} ms"
        )
        # Below and to the right of each marker. The curve rises to the right
        # throughout, so this side is always clear of it; centring the label
        # under the marker puts it on the line on the steep middle rungs.
        # The final rung is the exception: it is the rightmost point on the
        # panel, so its label goes below and to the LEFT, where the curve has
        # already passed. Placing it right would push the tight bounding box
        # out past the axes and shrink the figure when it is set at text width.
        is_last = row.Index == len(points) - 1
        ax.annotate(
            label,
            xy=(row.measured_ms, row.gap_pct),
            xytext=(-6, -8) if is_last else (6, -8),
            textcoords="offset points",
            fontsize=ANNOTATION_PT,
            color="#777777",
            ha="right" if is_last else "left",
            va="top",
        )

    # The value label stays on the dashed line, where it names the reference
    # the whole figure is read against.
    ax.text(
        points.measured_ms.iloc[-1] * 1.9,
        learned_gap_pct,
        f"Learned policy, {learned_gap_pct:.2f} %",
        fontsize=ANNOTATION_PT,
        color=COLOR_WARM,
        ha="right",
        va="bottom",
    )

    # The ladder is labelled directly in its own colour, so no legend is
    # needed and nothing competes with the data.
    ax.text(
        points.measured_ms.iloc[-1] * 0.92,
        points.gap_pct.iloc[-1] + 3.4,
        "Rolling horizon, h = 1",
        fontsize=ANNOTATION_PT,
        color=COLOR_COLD,
        ha="right",
        va="bottom",
    )

    # One short note for the open markers, in the empty lower right below the
    # curve. The glyph is a real open marker rather than a word, so the note
    # names the encoding by showing it. The threshold is interpolated from the
    # constant the code enforces, so the note cannot drift from the rule.
    # Sat below the lowest rung rather than level with it, so the glyph cannot
    # be misread as an eighth data point.
    note_x, note_y = 34.0, 16.4
    ax.plot(
        [note_x],
        [note_y],
        marker="o",
        markersize=3.2,
        markerfacecolor="white",
        markeredgecolor="#777777",
        markeredgewidth=0.8,
        linestyle="none",
        zorder=4,
    )
    ax.annotate(
        f"fallback decided $\\geq$ {FALLBACK_OPEN_THRESHOLD_PCT:g} % "
        "of window solves",
        xy=(note_x, note_y),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=ANNOTATION_PT,
        color="#777777",
        ha="left",
        va="center",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Measured compute per decision (ms, log scale)")
    ax.set_ylabel("Gap closure vs MILP (%)")
    ax.set_xlim(0.62, 1500.0)
    ax.set_ylim(12.0, 82.0)
    ax.set_yticks([20, 30, 40, 50, 60, 70, 80])
    hairline_grid(ax, axis="y")

    fig.savefig(OUT_PDF)
    plt.close(fig)

    extra = {}
    for row in points.itertuples():
        extra[f"budget {row.budget_s:g} s"] = (
            f"nominal {row.nominal_ms:.0f} ms, measured {row.measured_ms:.3f} ms "
            f"per solved decision ({row.measured_ms / row.nominal_ms:.3f}x nominal), "
            f"{row.measured_ms_per_epoch:.3f} ms per dispatcher epoch, "
            f"gap closure {row.gap_pct:.3f} %, fallback {row.fallback_pct:.2f} %, "
            f"sweep wall {row.wall_seconds} s, "
            f"mean solves/instance {row.n_window_solves:.2f}, "
            f"mean epochs/instance {row.n_decisions:.2f}, n={row.n}"
        )
    extra["learned policy"] = (
        f"gap closure {learned_gap_pct:.3f} % (ratio of means, n={n_learned}), "
        f"compute marked {learned_ms:.4f} ms, being the test-split "
        f"median whole decision over 2587 timed decisions"
    )
    extra["crossing"] = crossing["text"]

    write_sidecar(
        OUT_PDF,
        sources=[rung_csv(stem) for _, stem in BUDGETS]
        + [WALLTIMES, HUNGARIAN, LEARNED, TIMING, STYLE, THIS],
        extra=extra,
        notes=[
            "Quantity plotted on y: gap closure as the ratio of mean costs, "
            "(mean greedy - mean policy) / (mean greedy - mean MILP), over the "
            "200 test instances on seeds 11200 to 11399. The floor is the "
            "distance-only Hungarian from b3_hungarian_distance_only_test_"
            "20260731.csv and the ceiling is the cached MILP objective carried "
            "in each rung's own milp_oracle_cost_from_cache column. Recomputed "
            "here from per-instance CSVs; no value is read from any summary "
            "table, report or aggregate JSON. The seven values reproduce "
            "Table 1 of results_tables_v2_20260731.md exactly, as does the "
            "learned policy's 50.372 %.",
            "Quantity plotted on x: measured wall clock for the whole rung, "
            "divided by the 200 instances, divided by the mean number of "
            "window solves per instance. It therefore includes model "
            "construction, the simulator step and the epochs that advanced "
            "without a solve, none of which the Gurobi TimeLimit covers. The "
            "wall clock comes from a3_ladder_walltimes.txt, the timing log "
            "written by the ladder driver itself.",
            "Measured compute is NOT the nominal budget and the two diverge in "
            "both directions. At the 10 ms rung the measured cost is 1.217x "
            "the budget because model construction sits outside the limit. At "
            "the 10 s rung it is 0.054x the budget because almost every solve "
            "terminates well before the limit. Plotting against the nominal "
            "budget would stretch the x axis by up to a factor of nineteen and "
            "would misplace the crossing.",
            "Denominator choice on x. n_window_solves counts the dispatcher "
            "invocations that reached the solver, which is what the budget is "
            "applied to once each. n_decisions counts every simulator epoch, "
            "including those that advanced with nothing pending, and is about "
            "1.4x larger. The former is plotted, matching "
            "fig_budget_crossover.py; the latter is recorded per rung above so "
            "the figure can be rebuilt on either convention.",
            "Open markers mark the rungs where the fallback assignment "
            "decided more than 10 % of window solves, being 36.63 % at the "
            "10 ms budget and 17.68 % at 25 ms. On those two rungs a "
            "substantial share of decisions was made by Hungarian-on-kappa "
            "rather than by the rolling horizon, so the points describe a "
            "hybrid policy and should not be read as the rolling horizon's own "
            "performance. The script asserts that exactly the two shortest "
            "rungs exceed the threshold and fails if that ever changes.",
            "The learned policy compute mark is "
            f"{learned_ms:.4f} ms, the median whole decision over the 2587 "
            "timed decisions in b3_timing_test_20260731.csv, which is the "
            "TEST split. Every quantity in this figure is therefore on one "
            "split: gap closure, costs and both compute axes all come from the "
            "200 test instances on seeds 11200 to 11399. It is the same value "
            "fig_budget_crossover.py marks at this scale, so the two figures "
            "agree. An earlier draft marked 0.92 ms, the median from the A6 "
            "run (a6_inference_timing_20260730.md), which is a VALIDATION "
            "measurement over 50 instances and 2234 decisions and is 21 % "
            "lower. That value is not used here. The change does not move the "
            "crossing, which is fixed by the y axis, but it does change the "
            "crossing's compute multiple from about 48x to about 40x.",
            "All timings were measured under x86 emulation on this machine, "
            "per the environment note in a6_inference_timing_20260730.md. The "
            "comparison is internally consistent because Gurobi and the policy "
            "ran under the same emulation, but the absolute milliseconds are "
            "not native-hardware figures.",
            "Serve-all is 200 of 200 on every rung and on the learned policy, "
            "asserted by the script, so no point is scored on a self-selected "
            "subset.",
            "No solver was invoked, nothing was evaluated or trained, and no "
            "cache record was read or modified. This script reads six CSVs and "
            "one text log.",
        ],
    )

    print(f"wrote {OUT_PDF}")
    print(f"wrote {os.path.splitext(OUT_PDF)[0]}.sources.txt")
    print()
    print(
        f"{'budget':>8} {'nominal ms':>11} {'measured ms':>12} "
        f"{'ratio':>7} {'gap %':>8} {'fallback %':>11}"
    )
    for row in points.itertuples():
        print(
            f"{row.budget_s:>8g} {row.nominal_ms:>11.0f} {row.measured_ms:>12.3f} "
            f"{row.measured_ms / row.nominal_ms:>7.3f} {row.gap_pct:>8.3f} "
            f"{row.fallback_pct:>11.2f}"
        )
    print()
    print(f"learned policy: {learned_gap_pct:.3f} % at {learned_ms:.4f} ms marked")
    print(f"crossing: {crossing['text']}")


def bracket_crossing(points, learned_gap_pct: float, learned_ms: float) -> dict:
    """Locate where the ladder crosses the learned policy's gap closure.

    Returns the bracketing rungs, the interpolated crossing abscissa and the
    compute multiple, so that the figure draws the span from the same numbers
    the sidecar reports. Nothing about the crossing is hand-placed.
    """
    below = points[points.gap_pct < learned_gap_pct]
    above = points[points.gap_pct > learned_gap_pct]
    if below.empty or above.empty:
        return dict(found=False, text="no crossing within the plotted range")
    last_below = below.iloc[-1]
    first_above = above.iloc[0]
    # Log-linear interpolation in x, which is how the eye reads a log axis.
    span = np.log10(first_above.measured_ms) - np.log10(last_below.measured_ms)
    frac = (learned_gap_pct - last_below.gap_pct) / (
        first_above.gap_pct - last_below.gap_pct
    )
    x_cross = 10 ** (np.log10(last_below.measured_ms) + frac * span)
    multiple = x_cross / learned_ms
    return dict(
        found=True,
        x_cross=float(x_cross),
        multiple=float(multiple),
        text=(
            f"between the {last_below.budget_s:g} s rung "
            f"({last_below.measured_ms:.3f} ms, {last_below.gap_pct:.3f} %) and "
            f"the {first_above.budget_s:g} s rung "
            f"({first_above.measured_ms:.3f} ms, {first_above.gap_pct:.3f} %); "
            f"log-linear interpolation puts it at {x_cross:.1f} ms, about "
            f"{multiple:.0f}x the learned policy's marked compute"
        ),
    )


if __name__ == "__main__":
    main()
