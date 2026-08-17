"""Regenerate the nine vector weight sensitivity study on the test split.

``provenance/weight_sensitivity_test_20260804.md`` reports a re-scoring study
over nine weight vectors and states in its closing section that the analysis
script is not committed. This script is that missing producer: it reads the
tracked per-term exports from ``scripts/regenerate/export_term_provenance.py``,
calls ``scripts/regenerate/term_decomposition.py`` for every quantity, writes
the results as a CSV, then compares every number against the markdown and
reports a per-vector pass or fail.

Nothing is overwritten in the markdown. A disagreement is printed and recorded
so it can be settled before the study is cited, since a result whose producing
code no longer exists may not be reproducible at all. If more than two of the
nine vectors disagree the run stops before writing the CSV, so the decision to
cite the study is taken by a person rather than papered over by a fresh file.

The nine vectors are recovered from the markdown exactly and are not
renormalised. Three need a word: W2 is printed to six decimals but its exact
construction (one third divided by each Table 3.1 reference magnitude 869.4,
225.4, 77.5) is used here and rounds to the printed row. W2r is printed as
0.3333 on all three terms and is exactly one third on each; the cost ratio is
invariant to a common positive rescaling, so printed rounding and the exact
third give the same ratio either way. W5 as written is identical to W4 — the
markdown says so explicitly and reports the row twice on purpose, and both
rows are kept here so the regenerated table matches the published nine rows.
W5b, W7 and W2r are marked supplementary in the markdown rather than part of
the requested set; that marking carries through to the CSV.

The bootstrap is 10,000 paired resamples of instances at seed 20260731
(``BOOT_SEED`` in ``scripts/analysis/transfer_table_stats.py``). The markdown
records the same seed and the same single index matrix reused across every
vector; since ``boot_indices`` is a pure function of the seed and instance
count, drawing it once per vector reproduces that exactly.

No solver is invoked, ``gurobipy`` is not imported, no Gurobi licence is read,
nothing is trained, no checkpoint is opened and no cache record is read. This
script reads three committed CSVs and writes into its own output directory.

Usage::

    PYTHONPATH="$PWD/src" python scripts/regenerate/weight_sensitivity.py

    # overwrite an earlier run
    PYTHONPATH="$PWD/src" python scripts/regenerate/weight_sensitivity.py --force
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _d in ("regenerate", "analysis", "experiments"):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / _d))

from term_decomposition import (  # noqa: E402
    BOOT_SEED,
    N_BOOT,
    LOCKED_WEIGHTS,
    load_terms_csv,
    reweight,
    sha256_of,
    write_sha256_sidecar,
)

SOURCE_MD = "provenance/weight_sensitivity_test_20260804.md"
POLICY_CSV = "repro_terms_policy_hard_test.csv"
GREEDY_CSV = "repro_terms_hungarian_distance_only_test.csv"
CALIBRATION_JSON = "provenance/methodology/repro_weight_calibration.json"

OUT_CSV = "repro_weight_sensitivity.csv"
OUT_JSON = "repro_weight_sensitivity_comparison.json"

# How many vectors may disagree before the run refuses to write a CSV.
MAX_DISAGREEING_VECTORS = 2

THIRD = 1.0 / 3.0

# The nine vectors, recovered from the markdown in the order it prints them.
# Nothing here is renormalised and nothing is invented.
VECTORS = [
    {
        "id": "W1",
        "label": "production",
        "weights": LOCKED_WEIGHTS,
        "supplementary": False,
        "note": "the locked production triple",
    },
    {
        "id": "W2",
        "label": "equal thirds, normalised",
        "weights": (THIRD / 869.4, THIRD / 225.4, THIRD / 77.5),
        "supplementary": False,
        "note": "one third divided by each Table 3.1 reference magnitude, "
                "printed in the markdown as 0.000383, 0.001479 and 0.004301",
    },
    {
        "id": "W2r",
        "label": "equal thirds, raw",
        "weights": (THIRD, THIRD, THIRD),
        "supplementary": True,
        "note": "printed as 0.3333 on all three terms, exactly one third here, "
                "and the ratio is invariant to the common scale either way",
    },
    {
        "id": "W3",
        "label": "travel-heavy",
        "weights": (0.60, 0.20, 0.20),
        "supplementary": False,
        "note": "",
    },
    {
        "id": "W4",
        "label": "makespan-heavy",
        "weights": (0.20, 0.60, 0.20),
        "supplementary": False,
        "note": "",
    },
    {
        "id": "W5",
        "label": "as written",
        "weights": (0.20, 0.60, 0.20),
        "supplementary": False,
        "note": "identical to W4 as specified, which the markdown states and "
                "reports twice on purpose",
    },
    {
        "id": "W5b",
        "label": "balance-light",
        "weights": (0.475, 0.475, 0.05),
        "supplementary": True,
        "note": "",
    },
    {
        "id": "W7",
        "label": "balance-zero",
        "weights": (0.50, 0.50, 0.00),
        "supplementary": True,
        "note": "",
    },
    {
        "id": "W6",
        "label": "reciprocal rule, test split",
        "weights": (0.06236405280502964, 0.23557337994916025,
                    0.7020625672458101),
        "supplementary": False,
        "note": "the reciprocal rule on the test-split distance-only Hungarian "
                "term means, printed in the markdown as 0.062364, 0.235573 "
                "and 0.702063, and carried at full precision from "
                "provenance/methodology/repro_weight_calibration.json",
    },
]

# Every number the markdown prints for these vectors, with the number of
# decimals it prints it to. The tolerance for a match is half a unit in the last
# printed place, so a reproduction is a match at the precision the document
# commits to and nothing tighter.
PUBLISHED = {
    "W1": {
        "mean_policy_cost": (128.195, 3),
        "mean_baseline_cost": (159.880, 3),
        "cost_ratio": (0.8018, 4),
        "ratio_ci95_low": (0.7884, 4),
        "ratio_ci95_high": (0.8156, 4),
        "mean_paired_difference": (-31.685, 3),
        "paired_difference_ci95_low": (-34.084, 3),
        "paired_difference_ci95_high": (-29.226, 3),
        "n_instances_policy_cheaper": (195, 0),
    },
    "W2": {
        "mean_policy_cost": (0.7827, 4),
        "mean_baseline_cost": (0.9784, 4),
        "cost_ratio": (0.8000, 4),
        "ratio_ci95_low": (0.7865, 4),
        "ratio_ci95_high": (0.8139, 4),
        "mean_paired_difference": (-0.1957, 4),
        "paired_difference_ci95_low": (-0.2105, 4),
        "paired_difference_ci95_high": (-0.1806, 4),
        "n_instances_policy_cheaper": (195, 0),
    },
    "W2r": {
        "mean_policy_cost": (366.096, 3),
        "mean_baseline_cost": (381.564, 3),
        "cost_ratio": (0.9595, 4),
        "ratio_ci95_low": (0.9523, 4),
        "ratio_ci95_high": (0.9669, 4),
        "mean_paired_difference": (-15.468, 3),
        "paired_difference_ci95_low": (-18.212, 3),
        "paired_difference_ci95_high": (-12.609, 3),
        "n_instances_policy_cheaper": (164, 0),
    },
    "W3": {
        "mean_policy_cost": (560.158, 3),
        "mean_baseline_cost": (567.213, 3),
        "cost_ratio": (0.9876, 4),
        "ratio_ci95_low": (0.9808, 4),
        "ratio_ci95_high": (0.9945, 4),
        "mean_paired_difference": (-7.055, 3),
        "paired_difference_ci95_low": (-10.880, 3),
        "paired_difference_ci95_high": (-3.135, 3),
        "n_instances_policy_cheaper": (129, 0),
    },
    "W4": {
        "mean_policy_cost": (305.570, 3),
        "mean_baseline_cost": (318.491, 3),
        "cost_ratio": (0.9594, 4),
        "ratio_ci95_low": (0.9522, 4),
        "ratio_ci95_high": (0.9669, 4),
        "mean_paired_difference": (-12.920, 3),
        "paired_difference_ci95_low": (-15.213, 3),
        "paired_difference_ci95_high": (-10.539, 3),
        "n_instances_policy_cheaper": (163, 0),
    },
    "W5": {
        "mean_policy_cost": (305.570, 3),
        "mean_baseline_cost": (318.491, 3),
        "cost_ratio": (0.9594, 4),
        "ratio_ci95_low": (0.9522, 4),
        "ratio_ci95_high": (0.9669, 4),
        "mean_paired_difference": (-12.920, 3),
        "paired_difference_ci95_low": (-15.213, 3),
        "paired_difference_ci95_high": (-10.539, 3),
        "n_instances_policy_cheaper": (163, 0),
    },
    "W5b": {
        "mean_policy_cost": (507.979, 3),
        "mean_baseline_cost": (511.801, 3),
        "cost_ratio": (0.9925, 4),
        "ratio_ci95_low": (0.9860, 4),
        "ratio_ci95_high": (0.9992, 4),
        "mean_paired_difference": (-3.823, 3),
        "paired_difference_ci95_low": (-7.156, 3),
        "paired_difference_ci95_high": (-0.406, 3),
        "n_instances_policy_cheaper": (116, 0),
    },
    "W7": {
        "mean_policy_cost": (533.017, 3),
        "mean_baseline_cost": (534.784, 3),
        "cost_ratio": (0.9967, 4),
        "ratio_ci95_low": (0.9902, 4),
        "ratio_ci95_high": (1.0033, 4),
        "mean_paired_difference": (-1.768, 3),
        "paired_difference_ci95_low": (-5.250, 3),
        "paired_difference_ci95_high": (1.793, 3),
        "n_instances_policy_cheaper": (110, 0),
    },
    "W6": {
        "mean_policy_cost": (126.329, 3),
        "mean_baseline_cost": (158.221, 3),
        "cost_ratio": (0.7984, 4),
        "ratio_ci95_low": (0.7849, 4),
        "ratio_ci95_high": (0.8124, 4),
        "mean_paired_difference": (-31.893, 3),
        "paired_difference_ci95_low": (-34.298, 3),
        "paired_difference_ci95_high": (-29.429, 3),
        "n_instances_policy_cheaper": (195, 0),
    },
}

CSV_COLUMNS = [
    "vector", "label", "supplementary",
    "w1_travel", "w2_makespan", "w3_balance",
    "n_instances", "mean_policy_cost", "mean_baseline_cost",
    "cost_ratio", "ratio_ci95_low", "ratio_ci95_high",
    "mean_paired_difference",
    "paired_difference_ci95_low", "paired_difference_ci95_high",
    "n_instances_policy_cheaper", "interval_excludes_one",
    "n_boot", "bootstrap_seed",
]


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def tolerance(decimals: int) -> float:
    """Half a unit in the last place the markdown prints."""
    return 0.5 * 10 ** (-decimals) + 1e-12


def compare_vector(vector_id: str, computed: dict) -> dict:
    """Every published number for one vector against what was computed."""
    published = PUBLISHED[vector_id]
    fields, disagreements = {}, []
    for field, (expected, decimals) in published.items():
        got = float(computed[field])
        tol = tolerance(decimals)
        diff = abs(got - expected)
        agrees = diff <= tol
        fields[field] = {
            "published": expected,
            "regenerated": got,
            "abs_diff": diff,
            "tolerance": tol,
            "agrees": agrees,
        }
        if not agrees:
            disagreements.append(
                f"{field} published {expected:.{decimals}f} against "
                f"regenerated {got:.{decimals + 2}f}, a difference of "
                f"{diff:.{decimals + 2}f} against a tolerance of {tol:.2e}")
    return {
        "vector": vector_id,
        "n_compared": len(fields),
        "n_agreeing": sum(1 for f in fields.values() if f["agrees"]),
        "passes": not disagreements,
        "disagreements": disagreements,
        "fields": fields,
    }


def check_w6_against_calibration(path: Path) -> dict | None:
    """Cross-check the W6 literal against the committed calibration record.

    The markdown prints W6 to six decimals; the full precision triple lives in
    the JSON ``scripts/regenerate/calibrate_weights.py`` wrote, so the literal
    here is checked against it rather than trusted. A report, not a guard,
    since the study must remain reproducible from the markdown alone.
    """
    if not path.exists():
        return None
    with path.open() as handle:
        payload = json.load(handle)
    derived = payload["splits"]["test"]["derived_weights"]
    recorded = (derived["travel"], derived["makespan"], derived["balance"])
    literal = next(v["weights"] for v in VECTORS if v["id"] == "W6")
    return {
        "source": str(path.relative_to(REPO_ROOT)),
        "calibration_record": list(recorded),
        "literal_in_this_script": list(literal),
        "max_abs_difference": max(abs(a - b) for a, b in zip(recorded, literal)),
        "identical": recorded == tuple(literal),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", default="provenance/table41_main_results",
                    help="where the per-term exports live")
    ap.add_argument("--out-dir", default="provenance/appendix_a_sweeps")
    ap.add_argument("--seed", type=int, default=BOOT_SEED)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    in_dir = REPO_ROOT / args.in_dir
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / OUT_CSV
    json_path = out_dir / OUT_JSON
    for path in (csv_path, json_path):
        if path.exists() and not args.force:
            raise SystemExit(f"REFUSING to overwrite {path}. Pass --force.")

    policy_path = in_dir / POLICY_CSV
    greedy_path = in_dir / GREEDY_CSV
    for path in (policy_path, greedy_path):
        if not path.exists():
            raise SystemExit(
                f"{path} is missing. Run scripts/regenerate/export_term_provenance.py "
                "first, since this script reads only committed exports.")

    # load_terms_csv verifies each sha256 sidecar before parsing.
    policy = load_terms_csv(policy_path)
    greedy = load_terms_csv(greedy_path)
    print(f"policy rows {len(policy)}, baseline rows {len(greedy)}, "
          f"seeds {int(policy['seed'].min())} to {int(policy['seed'].max())}")
    print(f"bootstrap {args.n_boot} paired resamples of instances at seed "
          f"{args.seed}, 95 per cent percentile interval")
    print(f"comparing against {SOURCE_MD}")
    print()

    w6 = check_w6_against_calibration(REPO_ROOT / CALIBRATION_JSON)
    if w6 is not None:
        print(f"W6 literal against {w6['source']}, max absolute difference "
              f"{w6['max_abs_difference']:.3e}, identical {w6['identical']}")
        print()

    rows, comparisons = [], []
    header = (f"{'vec':<5} {'mean pol':>10} {'mean base':>10} {'ratio':>8} "
              f"{'CI low':>8} {'CI high':>8} {'cheaper':>8}  result")
    print(header)
    print("-" * len(header))

    for spec in VECTORS:
        result = reweight(greedy, policy, spec["weights"],
                          boot_seed=args.seed, n_boot=args.n_boot).iloc[0]
        computed = {k: result[k] for k in result.index}
        excludes_one = bool(result["ratio_ci95_high"] < 1.0
                            or result["ratio_ci95_low"] > 1.0)
        row = {
            "vector": spec["id"],
            "label": spec["label"],
            "supplementary": int(spec["supplementary"]),
            "w1_travel": repr(float(spec["weights"][0])),
            "w2_makespan": repr(float(spec["weights"][1])),
            "w3_balance": repr(float(spec["weights"][2])),
            "n_instances": int(result["n_instances"]),
            "mean_policy_cost": repr(float(result["mean_policy_cost"])),
            "mean_baseline_cost": repr(float(result["mean_baseline_cost"])),
            "cost_ratio": repr(float(result["cost_ratio"])),
            "ratio_ci95_low": repr(float(result["ratio_ci95_low"])),
            "ratio_ci95_high": repr(float(result["ratio_ci95_high"])),
            "mean_paired_difference":
                repr(float(result["mean_paired_difference"])),
            "paired_difference_ci95_low":
                repr(float(result["paired_difference_ci95_low"])),
            "paired_difference_ci95_high":
                repr(float(result["paired_difference_ci95_high"])),
            "n_instances_policy_cheaper":
                int(result["n_instances_policy_cheaper"]),
            "interval_excludes_one": int(excludes_one),
            "n_boot": int(result["n_boot"]),
            "bootstrap_seed": int(result["bootstrap_seed"]),
        }
        rows.append(row)

        comparison = compare_vector(spec["id"], computed)
        comparisons.append(comparison)
        verdict = "PASS" if comparison["passes"] else "FAIL  <<<<<< DISAGREES"
        print(f"{spec['id']:<5} {result['mean_policy_cost']:>10.4f} "
              f"{result['mean_baseline_cost']:>10.4f} "
              f"{result['cost_ratio']:>8.4f} "
              f"{result['ratio_ci95_low']:>8.4f} "
              f"{result['ratio_ci95_high']:>8.4f} "
              f"{int(result['n_instances_policy_cheaper']):>5} /200  {verdict}")
        for line in comparison["disagreements"]:
            print(f"        {line}")

    failing = [c["vector"] for c in comparisons if not c["passes"]]
    n_compared = sum(c["n_compared"] for c in comparisons)
    n_agree = sum(c["n_agreeing"] for c in comparisons)

    print()
    print(f"numbers compared against the markdown : {n_compared}")
    print(f"agreeing at the printed precision      : {n_agree}")
    print(f"vectors that disagree                  : {len(failing)} "
          f"{failing if failing else ''}")
    print()
    print("The markdown was not modified. A disagreement above is a finding "
          "about the markdown or about this reconstruction, and it has to be "
          "settled before the study is cited.")

    payload = {
        "script": "scripts/regenerate/weight_sensitivity.py",
        "git_head": _git_head(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "compared_against": SOURCE_MD,
        "protocol": {
            "n_boot": int(args.n_boot),
            "bootstrap_seed": int(args.seed),
            "interval": "95 per cent percentile, 2.5 and 97.5",
            "resamples": "instances, with replacement, paired across the two "
                         "policies",
            "estimator_source": "reweight in scripts/regenerate/term_decomposition.py, "
                                "which draws its index matrix with "
                                "boot_indices and takes percentiles with ci, "
                                "both from scripts/analysis/transfer_table_stats.py",
        },
        "inputs": {
            "policy": {"path": f"{args.in_dir}/{POLICY_CSV}",
                       "sha256": sha256_of(policy_path)},
            "baseline": {"path": f"{args.in_dir}/{GREEDY_CSV}",
                         "sha256": sha256_of(greedy_path)},
        },
        "w6_cross_check": w6,
        "vectors": [
            {"id": v["id"], "label": v["label"],
             "weights": list(v["weights"]),
             "supplementary": v["supplementary"], "note": v["note"]}
            for v in VECTORS
        ],
        "comparison": {
            "n_numbers_compared": n_compared,
            "n_agreeing": n_agree,
            "vectors_disagreeing": failing,
            "per_vector": comparisons,
        },
    }

    if len(failing) > MAX_DISAGREEING_VECTORS:
        with json_path.open("w") as handle:
            json.dump(payload, handle, indent=2)
        write_sha256_sidecar(json_path)
        raise SystemExit(
            f"\nSTOPPING before the CSV is written. {len(failing)} of "
            f"{len(VECTORS)} vectors disagree with {SOURCE_MD}, which is more "
            f"than the {MAX_DISAGREEING_VECTORS} this script tolerates. The "
            f"comparison is in {json_path} and the markdown is untouched. "
            "Several results in this project were produced by code that no "
            "longer exists, so a markdown with no committed script may not be "
            "reproducible at all. Decide whether to cite it before a CSV is "
            "written around it.")

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_sha256_sidecar(csv_path)
    payload["outputs"] = {
        "csv": f"{args.out_dir}/{OUT_CSV}",
        "sha256": sha256_of(csv_path),
        "n_rows": len(rows),
    }
    with json_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    write_sha256_sidecar(json_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print("No solver was invoked, nothing was trained, and no existing file "
          "was modified.")


if __name__ == "__main__":
    main()
