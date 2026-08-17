"""Regenerate the two Hungarian baseline per-instance CSVs.

Phase 1.5 step 1. The audit in CLEANUP_INVENTORY.md found that
``provenance/table41_main_results/b3_hungarian_distance_only_test_20260731.csv`` and
``provenance/table41_main_results/b3_hungarian_kappa_weighted_test_20260731.csv`` are written by no
script in the tree, yet they are the zero point of Table 4.1, the floor of the
Figure 4.2 budget curve and the baseline of the weight sensitivity study. This
script is the missing producer.

Nothing here is reimplemented: both rollouts come from
``src/baselines/bipartite_policies.py``, the cost from
``DynamicSimulator.compute_cost``, the serve-all test from
``method_one_evaluator.rollout_failed``, the gap closure from
``method_one_evaluator.gap_closure``, and the weights from ``EVAL_WEIGHTS`` in
``sweep/evaluate_sweep.py`` — the same call sequence, in the same order, that
``method_two_evaluator.evaluate_one_instance`` uses for its ``greedy`` and
``kappa`` figures.

Both policies are deterministic (no RNG draw, no solver beyond
``scipy.optimize.linear_sum_assignment``). No cache record, checkpoint or
configuration is written or modified, and the existing ``b3_*`` files are
never opened for writing.

Usage::

    PYTHONPATH="$PWD/src" python scripts/regenerate/reproduce_hungarian_baselines.py \\
        --cache-dir cache/training_set_il_v3/test --allow-test-split \\
        --eval-instances 200 --out-dir provenance/table41_main_results

    # add --compare-against-published to diff each instance against the
    # existing b3_* files, which are opened read only.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "sweep"))

from baselines.bipartite_policies import (  # noqa: E402
    run_greedy_policy,
    run_hungarian_kappa_policy,
)
from evaluation.method_one_evaluator import gap_closure, rollout_failed  # noqa: E402
from evaluation.method_two_evaluator import load_records  # noqa: E402
from instances.synthetic_generator import SyntheticInstance  # noqa: E402
from evaluate_sweep import EVAL_WEIGHTS  # noqa: E402

# The published files carry these nine columns in this order. Reproduced
# exactly so the two sets diff column by column without reordering.
COLUMNS = [
    "seed",
    "policy",
    "policy_cost",
    "cost_milp_oracle",
    "gap_closure_vs_milp",
    "serve_all_flag",
    "term_travel_time",
    "term_makespan",
    "term_balance",
]

PUBLISHED = {
    "hungarian_distance_only":
        "provenance/table41_main_results/b3_hungarian_distance_only_test_20260731.csv",
    "hungarian_kappa_weighted":
        "provenance/table41_main_results/b3_hungarian_kappa_weighted_test_20260731.csv",
}


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def _resolve_weights(records: list, override: str | None) -> tuple[dict, str]:
    """Return the weight triple and a one-line statement of where it came from.

    Default is ``EVAL_WEIGHTS`` from ``sweep/evaluate_sweep.py``, what the
    keep-set evaluation path uses and what the 30 July headline run recorded.
    Cached records carry their own ``weights`` field too; every record is
    checked against the triple in use and the run aborts on disagreement
    rather than silently scoring under two conventions.
    """
    if override:
        w = json.loads(override)
        source = f"--weights on the command line, {override}"
    else:
        w = dict(EVAL_WEIGHTS)
        source = "EVAL_WEIGHTS in sweep/evaluate_sweep.py line 39"

    seen = set()
    for rec in records:
        rw = rec.get("weights")
        if rw is None:
            raise SystemExit(
                f"Record for seed {rec.get('seed')} carries no 'weights' field. "
                f"Refusing to fall back to a module constant silently."
            )
        seen.add(tuple(round(float(rw[k]), 12) for k in ("w_dist", "w_make", "w_bal")))
    if len(seen) != 1:
        raise SystemExit(
            f"Cached records disagree on the objective weights: {sorted(seen)}. "
            f"Scoring a heterogeneous cache under one triple would be wrong."
        )
    cached = seen.pop()
    in_use = tuple(round(float(w[k]), 12) for k in ("w_dist", "w_make", "w_bal"))
    if cached != in_use:
        raise SystemExit(
            f"Weights in use {in_use} differ from the weights stored in every "
            f"cached record {cached}. Refusing to proceed."
        )
    source += f"; confirmed identical to the 'weights' field of all {len(records)} cached records"
    return w, source


def evaluate_one(record: dict, weights: dict) -> dict:
    """Run both baselines on one cached record.

    Same calls, same order, as method_two_evaluator.evaluate_one_instance
    makes for its ``greedy`` and ``kappa`` figures.
    """
    instance = SyntheticInstance.from_dict(record["instance"])

    sim_greedy = run_greedy_policy(instance)
    cb_greedy = sim_greedy.compute_cost(weights)
    greedy_cost = float(cb_greedy.combined)

    sim_kappa = run_hungarian_kappa_policy(instance, weights)
    cb_kappa = sim_kappa.compute_cost(weights)
    kappa_cost = float(cb_kappa.combined)

    milp_sol = record.get("milp_solution")
    milp = (
        float(milp_sol["objective_value"])
        if isinstance(milp_sol, dict) and milp_sol.get("objective_value") is not None
        else None
    )

    out = {}
    for policy, sim, cb, cost in (
        ("hungarian_distance_only", sim_greedy, cb_greedy, greedy_cost),
        ("hungarian_kappa_weighted", sim_kappa, cb_kappa, kappa_cost),
    ):
        # Denominator is the distance-only rollout cost minus the cached MILP
        # objective for BOTH rows, so the floor row scores exactly 0.0 and the
        # kappa row is measured against the same floor as every other Table 4.1 row.
        gc = None if milp is None else gap_closure(greedy_cost, cost, milp)
        out[policy] = {
            "seed": int(record.get("seed", -1)),
            "policy": policy,
            "policy_cost": repr(cost),
            "cost_milp_oracle": "" if milp is None else repr(milp),
            "gap_closure_vs_milp": "" if gc is None else repr(gc),
            "serve_all_flag": 0 if rollout_failed(sim) else 1,
            "term_travel_time": repr(cb.distance),
            "term_makespan": repr(cb.makespan),
            "term_balance": repr(cb.imbalance),
        }
    return out


def _repo_relative(path) -> str:
    """Path relative to the repository root, or unchanged if it lies outside.

    The summary this script writes can land under provenance/ and be tracked
    in a published repository, so an absolute path would carry the running
    machine's home directory into it. Matches ``figures.style._repo_relative``.
    """
    absolute = Path(path).resolve()
    try:
        return str(absolute.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def compare(rows: list, published_path: Path, key: str) -> dict:
    """Diff regenerated rows against a published CSV. Read only."""
    if not published_path.is_file():
        return {"status": "published file absent",
                "path": _repo_relative(published_path)}
    with published_path.open() as f:
        pub = {int(r["seed"]): r for r in csv.DictReader(f)}
    mine = {int(r["seed"]): r for r in rows}
    only_mine = sorted(set(mine) - set(pub))
    only_pub = sorted(set(pub) - set(mine))
    common = sorted(set(mine) & set(pub))

    disagreements, worst, worst_seed = 0, 0.0, None
    per_col_worst = {}
    for s in common:
        for col in ("policy_cost", "term_travel_time", "term_makespan", "term_balance",
                    "gap_closure_vs_milp", "cost_milp_oracle"):
            a, b = mine[s][col], pub[s].get(col, "")
            if a == "" or b == "":
                if a != b:
                    disagreements += 1
                continue
            d = abs(float(a) - float(b))
            if col not in per_col_worst or d > per_col_worst[col][0]:
                per_col_worst[col] = (d, s)
            if col == "policy_cost":
                if d > 0:
                    disagreements += 1
                if d > worst:
                    worst, worst_seed = d, s
        if int(mine[s]["serve_all_flag"]) != int(pub[s]["serve_all_flag"]):
            disagreements += 1
    return {
        "status": "compared",
        "path": _repo_relative(published_path),
        "n_common": len(common),
        "n_only_regenerated": len(only_mine),
        "n_only_published": len(only_pub),
        "n_cost_disagreements": sum(
            1 for s in common
            if float(mine[s]["policy_cost"]) != float(pub[s]["policy_cost"])
        ),
        "n_field_disagreements_total": disagreements,
        "max_abs_cost_difference": worst,
        "max_abs_cost_difference_seed": worst_seed,
        "max_abs_difference_by_column": {
            c: {"value": v[0], "seed": v[1]} for c, v in sorted(per_col_worst.items())
        },
        "serve_all_regenerated": sum(int(mine[s]["serve_all_flag"]) for s in common),
        "serve_all_published": sum(int(pub[s]["serve_all_flag"]) for s in common),
    }


# The committed artefact that carries this script's result. Named in the
# missing-input message so a fresh clone is told where the numbers live.
SHIPPED_ARTEFACT = "provenance/table41_main_results/b3_hungarian_distance_only_test_20260731.csv"


def require_cache(cache_dir, empty: bool = False) -> None:
    """Abort with one line if the gitignored label cache is absent or empty."""
    if empty or not (REPO_ROOT / cache_dir).exists():
        raise SystemExit(
            f"Missing {cache_dir}. It is gitignored and absent from a fresh "
            f"clone, so this script can only run on a tree that carries it. "
            f"The committed {SHIPPED_ARTEFACT} carries the result."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="cache/training_set_il_v3/test")
    ap.add_argument("--eval-instances", type=int, default=200)
    ap.add_argument("--out-dir", default="provenance/table41_main_results")
    ap.add_argument("--allow-test-split", action="store_true",
                    help="Explicit opt-in required to read a test split.")
    ap.add_argument("--weights", default=None,
                    help='JSON override, e.g. \'{"w_dist":0.0637,...}\'. '
                         'Omit to use EVAL_WEIGHTS.')
    ap.add_argument("--compare-against-published", action="store_true",
                    help="Diff against the existing b3_* files, opened read only.")
    ap.add_argument("--force", action="store_true",
                    help="Allow overwriting files in the output directory.")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    # Same guard as scripts/experiments/b3_ablation_test_eval.py and
    # scripts/experiments/rolling_horizon_baseline.py: aborts by default on a test path.
    if any(p.lower() == "test" for p in cache_dir.parts) and not args.allow_test_split:
        raise SystemExit(
            f"REFUSING to run on a test split ({cache_dir}) without "
            f"--allow-test-split. The test set is reserved for the final thesis."
        )
    if any(p.lower() == "test" for p in cache_dir.parts):
        print("OPT-IN: --allow-test-split passed, reading the frozen test split.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        p: out_dir / f"repro_{p}_{cache_dir.name}.csv" for p in PUBLISHED
    }
    summary_path = out_dir / f"repro_hungarian_summary_{cache_dir.name}.json"
    for p in list(targets.values()) + [summary_path]:
        if p.exists() and not args.force:
            raise SystemExit(f"REFUSING to overwrite {p}. Pass --force to replace it.")

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    require_cache(cache_dir)
    records = load_records(cache_dir, args.eval_instances)
    if not records:
        require_cache(cache_dir, empty=True)
    weights, weights_source = _resolve_weights(records, args.weights)

    seeds = [int(r.get("seed", -1)) for r in records]
    print(f"{len(records)} records, seeds {min(seeds)} to {max(seeds)}")
    print(f"weights in use: {weights}")
    print(f"weights source: {weights_source}")

    rows = {p: [] for p in PUBLISHED}
    handles = {p: targets[p].open("w", newline="") for p in PUBLISHED}
    writers = {}
    try:
        for p, fh in handles.items():
            writers[p] = csv.DictWriter(fh, fieldnames=COLUMNS)
            writers[p].writeheader()
        for i, rec in enumerate(records, 1):
            out = evaluate_one(rec, weights)
            for p in PUBLISHED:
                rows[p].append(out[p])
                writers[p].writerow(out[p])
                handles[p].flush()  # a crash costs one instance, not the run
            if i % 25 == 0 or i == len(records):
                print(f"  [{i}/{len(records)}] seed={rec.get('seed')}")
    finally:
        for fh in handles.values():
            fh.close()

    elapsed = time.perf_counter() - t0

    summary = {
        "script": "scripts/regenerate/reproduce_hungarian_baselines.py",
        "git_head": _git_head(),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": elapsed,
        "cache_dir": _repo_relative(cache_dir),
        "n_records": len(records),
        "seed_min": min(seeds),
        "seed_max": max(seeds),
        "weights": weights,
        "weights_source": weights_source,
        "outputs": {p: str(targets[p]) for p in PUBLISHED},
        "policies": {},
    }
    for p in PUBLISHED:
        costs = [float(r["policy_cost"]) for r in rows[p]]
        served = sum(int(r["serve_all_flag"]) for r in rows[p])
        summary["policies"][p] = {
            "mean_cost": sum(costs) / len(costs),
            "n": len(costs),
            "serve_all": f"{served}/{len(costs)}",
        }
        print(f"\n{p}: mean cost {summary['policies'][p]['mean_cost']:.6f}, "
              f"serve-all {served}/{len(costs)}")

    if args.compare_against_published:
        summary["comparison"] = {}
        for p, rel in PUBLISHED.items():
            cmp = compare(rows[p], REPO_ROOT / rel, p)
            summary["comparison"][p] = cmp
            print(f"\nvs {rel}")
            if cmp["status"] != "compared":
                print(f"  {cmp['status']}")
                continue
            print(f"  common instances        : {cmp['n_common']}")
            print(f"  cost disagreements      : {cmp['n_cost_disagreements']}")
            print(f"  any-field disagreements : {cmp['n_field_disagreements_total']}")
            print(f"  max abs cost difference : {cmp['max_abs_cost_difference']:.6e}"
                  f" (seed {cmp['max_abs_cost_difference_seed']})")
            print(f"  serve-all regenerated   : {cmp['serve_all_regenerated']}")
            print(f"  serve-all published     : {cmp['serve_all_published']}")

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {targets['hungarian_distance_only']}")
    print(f"Wrote {targets['hungarian_kappa_weighted']}")
    print(f"Wrote {summary_path}")
    print(f"Wall clock {elapsed:.1f} s. No cache record, checkpoint or "
          f"existing provenance file was modified.")


if __name__ == "__main__":
    main()
