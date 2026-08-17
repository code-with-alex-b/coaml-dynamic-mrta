"""Per-term decomposition of the policy advantage, and weight re-scoring.

Everything the thesis says about where the learned policy's advantage sits is
arithmetic on three recorded term columns. This module holds that arithmetic in
one place so the regeneration notebook, the weight sensitivity script and any
test call the same code rather than three copies of the same expression.

The analysis functions are pure: dataframes in, dataframes out, none of them
holds a file path or reads a cache record, a checkpoint or a solver. The file
helpers at the bottom are the only functions that touch disk, and each takes
its path as an argument.

The objective is C = w1*D + w2*M + w3*B, with D total fleet travel time, M
makespan and B the range of busy time across robots. Locked production triple
is 0.0637, 0.2398, 0.6965.

Bootstrap intervals come from ``boot_indices`` and ``ci`` in
``scripts/analysis/transfer_table_stats.py`` rather than a second
implementation, so every interval here is on the convention behind every
other interval in the thesis. ``boot_indices`` is a pure function of the seed
and instance count, so drawing it once per call gives every weight vector the
identical index matrix without threading one matrix through the call sites —
the property the weight sensitivity study depends on.

Lives in ``scripts/`` next to ``transfer_table_stats.py`` and
``bootstrap_intervals.py``, where the project's post-hoc analysis of
per-instance CSVs already lives, rather than in ``src/analysis/`` — ``src/``
holds the pipeline (generator, simulator, scorers, trainers, evaluators) and
nothing in it reads a results CSV, so putting this module there would have
created a backwards import from ``src/`` into ``scripts/`` to reach the
estimator helpers.

No solver is invoked anywhere in this module and ``gurobipy`` is not imported.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# transfer_table_stats lives in scripts/analysis/, only on the path when the
# caller put it there (notebook one does; a direct invocation does not, and
# without this the import fails on every tree). Same three entries, same
# order, as scripts/regenerate/bootstrap_intervals.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _d in ("regenerate", "analysis", "experiments"):
    _p = str(_REPO_ROOT / "scripts" / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import transfer_table_stats as tts  # noqa: E402
from transfer_table_stats import BOOT_SEED, N_BOOT, ci  # noqa: E402,F401

TERM_COLUMNS = ("term_travel_time", "term_makespan", "term_balance")
TERM_LABELS = ("travel", "makespan", "balance")

# The scalar cost column, whichever of these the frame carries. The learned
# policy and the Hungarian baselines record ``policy_cost``, the anticipative
# benchmark records ``milp_objective_value``.
COST_COLUMNS = ("policy_cost", "milp_objective_value")

LOCKED_WEIGHTS = (0.0637, 0.2398, 0.6965)

# Tolerances for identity_check. The two rollout policies reproduce their
# recorded cost to floating point. The benchmark does not, for the reason given
# in the identity_check docstring.
POLICY_IDENTITY_TOL = 1e-9
MILP_IDENTITY_TOL = 5e-3


# Internal helpers

def _as_weights(weights) -> np.ndarray:
    w = np.asarray(list(weights), dtype=float)
    if w.shape != (3,):
        raise ValueError(
            f"a weight vector must hold exactly three numbers, got {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"weight vector holds a non-finite entry, {weights}")
    return w


def _terms(df: pd.DataFrame) -> np.ndarray:
    missing = [c for c in TERM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"frame is missing term columns {missing}")
    return df[list(TERM_COLUMNS)].to_numpy(dtype=float)


def _cost_column(df: pd.DataFrame, cost_column: str | None) -> str:
    if cost_column is not None:
        if cost_column not in df.columns:
            raise ValueError(f"frame has no column {cost_column!r}")
        return cost_column
    found = [c for c in COST_COLUMNS if c in df.columns]
    if len(found) != 1:
        raise ValueError(
            "frame must carry exactly one recorded cost column out of "
            f"{list(COST_COLUMNS)}, found {found}")
    return found[0]


def _boot_indices_at(n: int, boot_seed: int, n_boot: int) -> np.ndarray:
    """Index matrix from transfer_table_stats, at an explicit seed and size.

    ``boot_indices`` reads its seed and resample count off module globals, so
    those two are set for the duration of the draw and restored afterwards —
    same approach as ``scripts/regenerate/bootstrap_intervals.py``, so no
    second bootstrap implementation enters the thesis.
    """
    saved_seed, saved_n = tts.BOOT_SEED, tts.N_BOOT
    tts.BOOT_SEED, tts.N_BOOT = int(boot_seed), int(n_boot)
    try:
        return tts.boot_indices(int(n))
    finally:
        tts.BOOT_SEED, tts.N_BOOT = saved_seed, saved_n


def pair_on_seed(baseline_df: pd.DataFrame,
                 policy_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort both frames onto one seed index, or raise if the seed sets differ.

    Silently intersecting two seed sets is how a paired statistic quietly
    becomes an unpaired one on a subset nobody named, so a difference in
    either direction is an error here rather than a filter.
    """
    for name, df in (("baseline", baseline_df), ("policy", policy_df)):
        if "seed" not in df.columns:
            raise ValueError(f"the {name} frame has no seed column")
        if df["seed"].duplicated().any():
            dup = sorted(df.loc[df["seed"].duplicated(), "seed"].unique())
            raise ValueError(f"the {name} frame repeats seeds {dup}")

    b = baseline_df.sort_values("seed").reset_index(drop=True)
    p = policy_df.sort_values("seed").reset_index(drop=True)
    bs = set(int(s) for s in b["seed"])
    ps = set(int(s) for s in p["seed"])
    if bs != ps:
        only_baseline = sorted(bs - ps)
        only_policy = sorted(ps - bs)
        raise ValueError(
            "the two frames do not cover the same seeds, so no paired "
            f"statistic is defined. {len(only_baseline)} seeds are in the "
            f"baseline only, {only_baseline[:10]}, and {len(only_policy)} are "
            f"in the policy only, {only_policy[:10]}")
    return b, p


# The four analysis functions

def weighted_terms(df: pd.DataFrame, weights) -> pd.DataFrame:
    """The three weighted term means of one policy.

    Returns a frame indexed by term name, carrying the weight applied, the raw
    mean of that term over the instances in ``df``, and the product of the two.
    The weighted means sum to the mean cost of the policy under ``weights``.
    """
    w = _as_weights(weights)
    raw = _terms(df).mean(axis=0)
    out = pd.DataFrame(
        {
            "weight": w,
            "raw_mean": raw,
            "weighted_mean": w * raw,
        },
        index=pd.Index(list(TERM_LABELS), name="term"),
    )
    out.attrs["n_instances"] = int(len(df))
    out.attrs["total_weighted_mean"] = float((w * raw).sum())
    return out


def attribution(baseline_df: pd.DataFrame, policy_df: pd.DataFrame,
                weights) -> pd.DataFrame:
    """Per-term decomposition of the reduction from baseline to policy.

    Instances are paired on seed and the seed sets must be identical, so this
    raises rather than intersecting. Reduction is baseline minus policy, so
    positive means the policy is cheaper on that term, negative means worse.

    Returns a frame indexed by term name with the two raw means, the raw
    reduction, the weight, the weighted reduction in absolute cost units, and
    that weighted reduction as a share of the total reduction in C. Shares sum
    to one by construction (asserted before returning). A share can exceed
    one, and another go negative, whenever the policy loses on one term and
    pays for it out of the others — not an error, and the substantive finding
    on the travel term.

    The total reduction and instance count are attached as frame attributes
    under ``total_weighted_reduction`` and ``n_instances``.
    """
    b, p = pair_on_seed(baseline_df, policy_df)
    w = _as_weights(weights)

    base_raw = _terms(b).mean(axis=0)
    pol_raw = _terms(p).mean(axis=0)
    raw_reduction = base_raw - pol_raw
    weighted_reduction = w * raw_reduction
    total = float(weighted_reduction.sum())
    if total == 0.0:
        raise ValueError(
            "the total weighted reduction is exactly zero, so no term can be "
            "given a share of it")

    share = weighted_reduction / total
    if abs(float(share.sum()) - 1.0) > 1e-9:
        raise ValueError(
            f"term shares sum to {float(share.sum())!r} rather than one")

    out = pd.DataFrame(
        {
            "baseline_raw_mean": base_raw,
            "policy_raw_mean": pol_raw,
            "raw_reduction": raw_reduction,
            "weight": w,
            "weighted_reduction": weighted_reduction,
            "share_of_total": share,
        },
        index=pd.Index(list(TERM_LABELS), name="term"),
    )
    out.attrs["n_instances"] = int(len(b))
    out.attrs["total_weighted_reduction"] = total
    out.attrs["mean_baseline_cost"] = float((w * base_raw).sum())
    out.attrs["mean_policy_cost"] = float((w * pol_raw).sum())
    return out


def identity_check(df: pd.DataFrame, weights, tol: float = POLICY_IDENTITY_TOL,
                   cost_column: str | None = None) -> pd.DataFrame:
    """Assert that w1 D + w2 M + w3 B reproduces the recorded scalar cost.

    Raises ``ValueError`` when any instance breaks the identity by more than
    ``tol``; otherwise returns a one row frame holding the instance count, the
    max and mean absolute residual, the tolerance and the column checked.

    Tolerances. The learned policy and the Hungarian baselines reproduce their
    recorded ``policy_cost`` to floating point (the simulator computes that
    cost as exactly this weighted sum), so they are checked at 1e-9.

    The anticipative benchmark is checked at 5e-3, and the residual there is
    not an error: the Gurobi objective carries a tie-break term of 1e-6 on the
    sum of task start times in addition to the three weighted terms (claim
    C012), breaking degenerate ties between equal-cost schedules and
    deliberately far below the resolution of any reported number. It is
    present in ``milp_objective_value`` and absent from the three recorded
    terms, so the recorded objective sits a little above their weighted sum by
    exactly that contribution. On the frozen test split the largest such
    residual is about 3.3e-3 on costs of order 1e2, matching the order 1e-3
    claim C013 predicts. A materially larger residual, or one of the opposite
    sign, would mean something other than the tie-break is in the objective
    and the export would be wrong.
    """
    w = _as_weights(weights)
    col = _cost_column(df, cost_column)
    recorded = df[col].to_numpy(dtype=float)
    rebuilt = _terms(df) @ w
    residual = rebuilt - recorded
    max_abs = float(np.abs(residual).max()) if residual.size else 0.0
    mean_abs = float(np.abs(residual).mean()) if residual.size else 0.0

    if max_abs > tol:
        worst = int(np.argmax(np.abs(residual)))
        seed = int(df["seed"].to_numpy()[worst]) if "seed" in df.columns else -1
        raise ValueError(
            f"the weighted terms do not reproduce {col} within {tol}. The "
            f"largest absolute residual is {max_abs!r} at row {worst}, seed "
            f"{seed}, where the terms rebuild {rebuilt[worst]!r} against a "
            f"recorded {recorded[worst]!r}")

    return pd.DataFrame([{
        "cost_column": col,
        "n_instances": int(len(df)),
        "max_abs_residual": max_abs,
        "mean_abs_residual": mean_abs,
        "tolerance": float(tol),
        "passes": True,
    }])


def reweight(baseline_df: pd.DataFrame, policy_df: pd.DataFrame, weights,
             boot_seed: int = BOOT_SEED, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Re-score both policies under an arbitrary weight vector and compare.

    Both frames are re-scored from their recorded term columns, valid for the
    learned policy and the distance-only Hungarian since neither assignment
    depends on the weights. Not valid for the kappa-weighted Hungarian (cost
    matrix built from the weights) or the anticipative benchmark (optimum
    moves with the weights) — both would have to be re-run, not re-scored.

    Returns a one row frame holding the cost ratio of policy over baseline, a
    95 per cent percentile bootstrap interval on that ratio, and the count of
    instances where the policy is cheaper. Mean costs and the mean paired
    difference with its own interval come back too, free once the index
    matrix is drawn.

    The bootstrap resamples instances with replacement ``n_boot`` times,
    reusing one index matrix for policy and baseline so the interval respects
    the pairing. ``boot_seed`` defaults to ``BOOT_SEED`` from
    ``transfer_table_stats`` (20260731, the seed behind every other interval
    in the thesis) as a fixed default rather than a hidden draw, since an
    unseeded interval is not reproducible. The draw is deterministic in the
    seed and instance count, so calling this once per weight vector gives
    every vector the identical index matrix.
    """
    b, p = pair_on_seed(baseline_df, policy_df)
    w = _as_weights(weights)

    base_cost = _terms(b) @ w
    pol_cost = _terms(p) @ w
    n = int(base_cost.size)

    idx = _boot_indices_at(n, boot_seed, n_boot)
    bs_ratio = pol_cost[idx].mean(axis=1) / base_cost[idx].mean(axis=1)
    ratio_lo, ratio_hi = ci(bs_ratio)

    diffs = pol_cost - base_cost
    diff_lo, diff_hi = ci(diffs[idx].mean(axis=1))

    return pd.DataFrame([{
        "n_instances": n,
        "w1_travel": float(w[0]),
        "w2_makespan": float(w[1]),
        "w3_balance": float(w[2]),
        "mean_policy_cost": float(pol_cost.mean()),
        "mean_baseline_cost": float(base_cost.mean()),
        "cost_ratio": float(pol_cost.mean() / base_cost.mean()),
        "ratio_ci95_low": ratio_lo,
        "ratio_ci95_high": ratio_hi,
        "mean_paired_difference": float(diffs.mean()),
        "paired_difference_ci95_low": diff_lo,
        "paired_difference_ci95_high": diff_hi,
        "n_instances_policy_cheaper": int((diffs < 0).sum()),
        "n_boot": int(n_boot),
        "bootstrap_seed": int(boot_seed),
    }])


# File input and output. Every path is an argument.

def sha256_of(path) -> str:
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sidecar_path(path) -> Path:
    """The sidecar that carries a file's hash, being the file plus .sha256."""
    p = Path(path)
    return p.with_name(p.name + ".sha256")


def write_sha256_sidecar(path) -> Path:
    """Write ``<file>.sha256``: hex digest, two spaces, base name, matching
    the ``shasum -a 256`` format so ``shasum -a 256 -c <file>.sha256``
    verifies it from the containing directory with no project-specific tooling.
    """
    p = Path(path)
    side = sidecar_path(p)
    side.write_text(f"{sha256_of(p)}  {p.name}\n")
    return side


def read_sha256_sidecar(path) -> str:
    """The digest recorded in a file's sidecar."""
    side = sidecar_path(path)
    if not side.exists():
        raise FileNotFoundError(f"no sha256 sidecar at {side}")
    first = side.read_text().strip().splitlines()[0]
    digest = first.split()[0].strip()
    if len(digest) != 64:
        raise ValueError(f"{side} does not start with a 64 character digest")
    return digest


def verify_sha256(path) -> str:
    """Recompute a file's hash and compare it against its sidecar.

    Returns the digest when they agree and raises otherwise: a mismatch means
    the file changed since export, so anything computed from it is not the
    published quantity.
    """
    recorded = read_sha256_sidecar(path)
    actual = sha256_of(path)
    if recorded != actual:
        raise ValueError(
            f"sha256 mismatch on {Path(path).name}. The sidecar records "
            f"{recorded} and the file on disk hashes to {actual}")
    return actual


def load_terms_csv(path, verify: bool = True) -> pd.DataFrame:
    """Read one of the per-term provenance exports into a dataframe.

    Lines starting with a hash are the provenance header and are skipped. When
    ``verify`` is true the sha256 sidecar is checked before the file is parsed,
    so a cell that reads through this function cannot silently use a file that
    has drifted from the one that was exported.
    """
    p = Path(path)
    if verify:
        verify_sha256(p)
    frame = pd.read_csv(p, comment="#")
    if "seed" in frame.columns:
        frame["seed"] = frame["seed"].astype(int)
    return frame


def write_terms_csv(path, rows, fieldnames, header_lines) -> Path:
    """Write a provenance export with its hash-prefixed header comment.

    ``header_lines`` is a sequence of plain sentences. Each is written as a
    comment line so that the file states its own provenance, and ``read_csv``
    with ``comment="#"`` skips them.
    """
    p = Path(path)
    with p.open("w", newline="") as handle:
        for line in header_lines:
            handle.write(f"# {line}\n")
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return p
