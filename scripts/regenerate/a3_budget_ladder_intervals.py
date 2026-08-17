"""95 per cent bootstrap intervals on gap closure for the seven budget A3 rungs.

The rolling horizon at h=1 on the frozen test split, seeds 11200 to 11399, at
the seven solve budgets from 10 ms to 10 s. ``figures/fig_budget_curve.py``
already computes and plots the point estimate for each rung but carries no
uncertainty, so a reader cannot tell which rungs are separated by more than
sampling noise. This script supplies the intervals for those same seven point
estimates and nothing else.

Estimator: gap closure is the ratio of mean costs,
(mean greedy - mean policy) / (mean greedy - mean MILP), over the 200 test
instances, the Table 4.1 convention and the one the figure uses. It is a ratio
of two aggregates rather than a mean of per-instance ratios, so its interval
has to come from resampling the instances underneath it.

Reuse rather than reimplementation. ``load_points`` and ``ratio_of_means`` are
imported from ``figures/fig_budget_curve.py``, so the seven point estimates
are by construction the ones the figure draws, and the figure's own
validation (seed sets against the floor, serve-all on every instance, the
budget column agreeing with the rung, row count against the wall clock log,
the fallback-share rule) runs as a side effect. The bootstrap machinery comes
from ``scripts/analysis/transfer_table_stats.py``, whose ``boot_indices`` and
``ci`` are behind every other interval in the thesis, so these seven land on
the same convention as Table 4.1 and the transfer table.

Pairing. The rolling horizon cost, the distance-only Hungarian floor and the
cached anticipative ceiling are three measurements on the SAME 200 instances;
resampling them independently would break that correlation and report
intervals that are far too wide. One index matrix is therefore drawn once and
reused for the floor, the policy and the ceiling within each replicate and
across all seven rungs, so every interval is paired and the rungs are
directly comparable.

Verification gate. The 1 s rung must reproduce the published interval
[71.12, 75.20] from Table 4.1 of ``provenance/results_tables_v2_20260731.md``.
If it does not, the script writes nothing, since a protocol that cannot land
on the one published value is not the one the thesis used and the other six
would then be unsupported.

No solver is invoked, nothing is evaluated, nothing is trained, no checkpoint
is loaded and no cache record is read or modified. This script reads nine
CSVs and one text log, and writes two new files into its own output directory.

Output goes to ``provenance/``, alongside every other regenerator, since these
seven intervals are reported in the thesis and this is their only tracked
source. The sidecar records the bare basename, matching the other sidecars
under ``provenance/``, so ``shasum -a 256 -c`` verifies it from the directory
holding the file.

Usage::

    PYTHONPATH="$PWD/src" python scripts/regenerate/a3_budget_ladder_intervals.py

    # Non-default seed, for a stability check on the interval width. This one
    # is a diagnostic rather than a reported value, so it goes to the
    # gitignored results/ and is deliberately not published.
    PYTHONPATH="$PWD/src" python scripts/regenerate/a3_budget_ladder_intervals.py \\
        --seed 20260730 --out results/a3_ladder_seedcheck_20260812.csv
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

# The figure module imports pyplot at module scope; fix a non-interactive
# backend before that happens so this script runs headless. It never draws.
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _d in ("regenerate", "analysis", "experiments"):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _d))
sys.path.insert(0, str(REPO_ROOT / "figures"))

import transfer_table_stats as tts  # noqa: E402
from fig_budget_curve import (  # noqa: E402
    BUDGETS,
    HUNGARIAN,
    load_points,
    ratio_of_means,
    rung_csv,
)
from style import sha256_of  # noqa: E402
from transfer_table_stats import BOOT_SEED, N_BOOT, boot_indices, ci  # noqa: E402

DEFAULT_OUT = (
    REPO_ROOT / "provenance" / "a3_budget_ladder_gap_intervals_test_20260812.csv"
)

# Table 4.1 of provenance/results_tables_v2_20260731.md, gap closure as the
# ratio of means per rung, to the two decimals the table prints. Transcribed
# targets, not computed values; the script checks itself against them.
PUBLISHED_POINT_PCT = {
    0.010: "20.40",
    0.025: "34.95",
    0.050: "47.72",
    0.100: "52.58",
    0.250: "62.86",
    1.0: "73.22",
    10.0: "75.89",
}

# The single rung whose interval is published, so it validates the protocol.
GATE_BUDGET_S = 1.0
GATE_INTERVAL_PCT = ("71.12", "75.20")

DECIMALS = 2


def fmt(value: float) -> str:
    """The printed precision of the output, used for every comparison too.

    The published table gives two decimals, so two decimals is the strongest
    claim that can be checked against it.
    """
    return f"{value:.{DECIMALS}f}"


@contextlib.contextmanager
def boot_seed(seed: int):
    """Run ``boot_indices`` at a chosen seed without reimplementing the draw.

    ``transfer_table_stats.boot_indices`` reads the module-level ``BOOT_SEED``,
    so it is rebound around the call, keeping the RNG construction in one
    place while letting the seed be an argument. Same pattern as
    ``scripts/regenerate/bootstrap_intervals.py``.
    """
    saved = tts.BOOT_SEED
    tts.BOOT_SEED = seed
    try:
        yield
    finally:
        tts.BOOT_SEED = saved


def load_series() -> tuple[dict, np.ndarray, np.ndarray]:
    """Per-instance costs for each rung, plus the shared floor and ceiling.

    Returns the per-rung policy cost arrays keyed by budget, the floor and the
    ceiling, all on one ascending seed order, so a single index vector indexes
    all three consistently.

    The ceiling is each rung's own ``milp_oracle_cost_from_cache`` column, the
    same one the figure reads and byte-identical to the ``cost_milp_oracle``
    column on the learned rows Table 4.1 used; the script asserts every rung
    agrees on it so the ceiling cannot silently differ between rungs.
    """
    floor_series = pd.read_csv(HUNGARIAN).set_index("seed").sort_index()
    seeds = list(floor_series.index)
    floor = floor_series["policy_cost"].values

    policy, ceiling = {}, None
    for budget, stem in BUDGETS:
        frame = pd.read_csv(rung_csv(stem)).set_index("seed").sort_index()
        if list(frame.index) != seeds:
            raise SystemExit(f"{rung_csv(stem)}: seed set differs from the floor")
        policy[budget] = frame["policy_cost"].values

        rung_ceiling = frame["milp_oracle_cost_from_cache"].values
        if ceiling is None:
            ceiling = rung_ceiling
        elif not np.array_equal(rung_ceiling, ceiling):
            raise SystemExit(
                f"{rung_csv(stem)}: cached MILP ceiling differs from earlier rungs, "
                "so the rungs are not scored against one common ceiling"
            )

    # The floor must not sit at or above the ceiling, or the gap closure
    # denominator is degenerate or wrong-signed. ratio_of_means checks this on
    # the means; this checks the arrays are the right way round.
    if floor.mean() <= ceiling.mean():
        raise SystemExit("the distance-only floor is not above the MILP ceiling")

    return policy, floor, ceiling


def intervals(policy: dict, floor, ceiling, seed: int) -> list[dict]:
    """Point estimate and 95 per cent percentile interval for every rung.

    One index matrix, drawn once, reused for the floor, the policy and the
    ceiling in each replicate and shared across all seven rungs, which is what
    makes the intervals paired.
    """
    n = floor.size
    with boot_seed(seed):
        idx = boot_indices(n)
    if idx.shape != (N_BOOT, n):
        raise SystemExit(f"index matrix is {idx.shape}, expected {(N_BOOT, n)}")

    # Same in every replicate for every rung, so computed once rather than seven times.
    boot_floor = floor[idx].mean(axis=1)
    boot_ceiling = ceiling[idx].mean(axis=1)

    rows = []
    for budget, _ in BUDGETS:
        costs = policy[budget]
        boot_policy = costs[idx].mean(axis=1)

        # Same estimator on resampled means as on observed means; the point
        # estimate goes through the imported ratio_of_means so it can't drift.
        boot_gap_pct = (boot_floor - boot_policy) / (boot_floor - boot_ceiling) * 100.0
        low, high = ci(boot_gap_pct)

        n_finite = int(np.isfinite(boot_gap_pct).sum())
        if n_finite != N_BOOT:
            raise SystemExit(
                f"budget {budget:g} s: {N_BOOT - n_finite} of {N_BOOT} replicates "
                "had a non-finite gap closure, so the percentiles are not on the "
                "full resample set"
            )

        rows.append(
            dict(
                budget_seconds=budget,
                gap_closure_pct=ratio_of_means(floor, costs, ceiling) * 100.0,
                ci95_low_pct=low,
                ci95_high_pct=high,
            )
        )
    return rows


def check_points(rows: list[dict], figure_points: pd.DataFrame) -> None:
    """The point estimates must be the figure's, and the published seven."""
    problems = []
    for row in rows:
        budget = row["budget_seconds"]
        mine = row["gap_closure_pct"]

        drawn = figure_points.loc[
            np.isclose(figure_points.budget_s, budget), "gap_pct"
        ]
        if len(drawn) != 1:
            problems.append(f"{budget:g} s: not exactly one matching figure rung")
            continue

        # Against the figure: full float agreement required, same estimator on
        # the same arrays. Any difference means the two aren't reading the same instances.
        if not np.isclose(mine, float(drawn.iloc[0]), rtol=0.0, atol=1e-9):
            problems.append(
                f"{budget:g} s: {mine:.6f} % here against "
                f"{float(drawn.iloc[0]):.6f} % in fig_budget_curve.py"
            )

        # Against the published table, at the printed precision.
        expected = PUBLISHED_POINT_PCT[budget]
        if fmt(mine) != expected:
            problems.append(
                f"{budget:g} s: {fmt(mine)} % here against "
                f"{expected} % published in Table 4.1"
            )

    if problems:
        raise SystemExit(
            "STOPPING. The point estimates do not reproduce, so no interval is "
            "reported. Disagreements:\n  " + "\n  ".join(problems)
        )


def check_gate(rows: list[dict]) -> None:
    """The published interval on the 1 s rung is the check on the protocol."""
    gate = [r for r in rows if np.isclose(r["budget_seconds"], GATE_BUDGET_S)]
    if len(gate) != 1:
        raise SystemExit(f"no {GATE_BUDGET_S:g} s rung to check the protocol on")
    got = (fmt(gate[0]["ci95_low_pct"]), fmt(gate[0]["ci95_high_pct"]))
    if got != GATE_INTERVAL_PCT:
        raise SystemExit(
            f"STOPPING. The {GATE_BUDGET_S:g} s rung gives "
            f"[{got[0]}, {got[1]}] against the published "
            f"[{GATE_INTERVAL_PCT[0]}, {GATE_INTERVAL_PCT[1]}]. The bootstrap "
            "protocol is therefore not the one behind Table 4.1, so the other "
            "six intervals are not reported. Nothing was written."
        )


def write_outputs(rows: list[dict], out_path: Path, force: bool) -> Path:
    """Write the CSV and its sha256 sidecar, refusing to clobber."""
    sidecar_path = out_path.with_suffix(out_path.suffix + ".sha256")
    for path in (out_path, sidecar_path):
        if path.exists() and not force:
            raise SystemExit(f"REFUSING to overwrite {path}. Pass --force.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "budget_seconds": f"{r['budget_seconds']:g}",
                "gap_closure_pct": fmt(r["gap_closure_pct"]),
                "ci95_low_pct": fmt(r["ci95_low_pct"]),
                "ci95_high_pct": fmt(r["ci95_high_pct"]),
            }
            for r in rows
        ]
    )
    frame.to_csv(out_path, index=False)

    # sha256sum format, so `shasum -a 256 -c <sidecar>` verifies it directly;
    # bare basename, matching every other sidecar under provenance/.
    sidecar_path.write_text(f"{sha256_of(str(out_path))}  {out_path.name}\n")
    return sidecar_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=BOOT_SEED,
        help=f"resampling seed (default {BOOT_SEED}, the project bootstrap seed; "
        "the published intervals reproduce only at this value)",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    # Also runs every validation fig_budget_curve.py performs on the rungs.
    figure_points = load_points()
    policy, floor, ceiling = load_series()

    print(f"{floor.size} instances, seeds 11200 to 11399")
    print(
        f"bootstrap: {N_BOOT} resamples of INSTANCES, seed {args.seed}, "
        "95 per cent percentile interval at 2.5 and 97.5"
    )
    print(
        "one index matrix shared by the floor, the policy and the ceiling in "
        "each replicate and across all seven rungs, so every interval is paired"
    )
    print("estimator: gap closure as the ratio of mean costs\n")

    rows = intervals(policy, floor, ceiling, args.seed)

    check_points(rows, figure_points)
    print("point estimates reproduce fig_budget_curve.py and Table 4.1 on all seven")

    check_gate(rows)
    print(
        f"protocol check: the {GATE_BUDGET_S:g} s rung reproduces the published "
        f"[{GATE_INTERVAL_PCT[0]}, {GATE_INTERVAL_PCT[1]}]\n"
    )

    print(f"{'budget':>8} {'gap %':>8} {'ci low':>8} {'ci high':>8} {'width':>7}")
    for row in rows:
        print(
            f"{row['budget_seconds']:>8g} {fmt(row['gap_closure_pct']):>8} "
            f"{fmt(row['ci95_low_pct']):>8} {fmt(row['ci95_high_pct']):>8} "
            f"{row['ci95_high_pct'] - row['ci95_low_pct']:>7.2f}"
        )
    print()

    sidecar_path = write_outputs(rows, out_path, args.force)
    print(f"wrote {out_path}")
    print(f"wrote {sidecar_path}")
    print(
        f"verify with: (cd {sidecar_path.parent.relative_to(REPO_ROOT)} && "
        f"shasum -a 256 -c {sidecar_path.name})"
    )
    print(
        "No solver was invoked, nothing was trained, and no existing file was "
        "modified."
    )


if __name__ == "__main__":
    main()
