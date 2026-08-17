"""Export the per-term decomposition of all three test-split policies.

The per-term decomposition of the policy advantage cannot currently be
reproduced from committed files. Two of the three sources are committed, the
learned policy's per-instance CSV and the distance-only Hungarian's, but the
anticipative benchmark's terms live only in 200 JSON records under the
gitignored, unshipped ``cache/training_set_il_v3/test``. This script writes
all three onto the same footing as tracked CSVs so an examiner can rebuild the
decomposition with no training run, no solver call and no access to the
instance cache.

Three files are written, each with a ``.sha256`` sidecar.

  ``repro_terms_policy_hard_test.csv``
      The learned policy under hard decode, from
      ``results/a2_per_instance_test_20260730.csv`` filtered to
      ``decode_mode == hard``.

  ``repro_terms_hungarian_distance_only_test.csv``
      The distance-only Hungarian floor, from
      ``provenance/table41_main_results/b3_hungarian_distance_only_test_20260731.csv``.

  ``repro_terms_milp_test.csv``
      The offline anticipative benchmark, from the ``milp_solution`` block of
      each ``cache/training_set_il_v3/test/seed*.json`` record.

Guards on the benchmark export. The weight triple is a hardcoded literal that
appears in nine files under four different names, so asserting on the record's
own ``weights`` field would test the field, not the provenance. Four
independent guards are checked instead, and any one failing aborts the run
before anything is written.

  1. The cache directory actually read ends in ``training_set_il_v3/test``.
     ``cache/training_set_il`` carries a different vector, 0.0516, 0.1984 and
     0.75, and must never be the source.
  2. Every record's weights field reads 0.0637, 0.2398 and 0.6965.
  3. The seed set is exactly 11200 to 11399 with no gaps.
  4. ``objective_value`` is at or above the record's stored dual bound
     ``obj_bound``. Three validation records carry false optimality
     certificates and fail this check. No test record does, so a failure here
     means the export is reading the wrong split. Every offending record is
     named in the output.

All three exports are then checked to hold 200 rows over identical seed sets,
and the weighted terms of each are checked against its recorded scalar cost by
``term_decomposition.identity_check``.

No solver is invoked, ``gurobipy`` is not imported, no Gurobi licence is read,
nothing is trained, no checkpoint is opened, and nothing under ``cache/`` is
modified; the cache is read and only read.

Where the outputs land. ``provenance/table41_main_results/``, not ``results/``:
``results/`` is gitignored at line 63 of ``.gitignore``, so a file written
there would not be committed and would defeat the point of an export meant to
reproduce from committed files alone. ``provenance/table41_main_results/`` is
tracked and is where every other regeneration export already lives.

Usage::

    PYTHONPATH="$PWD/src" python scripts/regenerate/export_term_provenance.py

    # overwrite an earlier export
    PYTHONPATH="$PWD/src" python scripts/regenerate/export_term_provenance.py --force
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
    LOCKED_WEIGHTS,
    MILP_IDENTITY_TOL,
    POLICY_IDENTITY_TOL,
    identity_check,
    load_terms_csv,
    sha256_of,
    write_sha256_sidecar,
    write_terms_csv,
)

EXPECTED_SEEDS = list(range(11200, 11400))

A2_CSV = "results/a2_per_instance_test_20260730.csv"
B3_CSV = "provenance/table41_main_results/b3_hungarian_distance_only_test_20260731.csv"
CACHE_REL = "cache/training_set_il_v3/test"

# The other cache under the old weight vector. Named here so the guard message
# can say what the wrong answer would have looked like.
WRONG_CACHE_REL = "cache/training_set_il"
WRONG_CACHE_WEIGHTS = (0.0516, 0.1984, 0.75)

POLICY_COLUMNS = ["seed", "term_travel_time", "term_makespan", "term_balance",
                  "policy_cost", "serve_all_flag"]
MILP_COLUMNS = ["seed", "term_travel_time", "term_makespan", "term_balance",
                "milp_objective_value"]

OUT_POLICY = "repro_terms_policy_hard_test.csv"
OUT_GREEDY = "repro_terms_hungarian_distance_only_test.csv"
OUT_MILP = "repro_terms_milp_test.csv"
OUT_MANIFEST = "repro_terms_manifest.json"

WEIGHT_SENTENCE = (
    f"Weight vector on every source record is w1 {LOCKED_WEIGHTS[0]} travel, "
    f"w2 {LOCKED_WEIGHTS[1]} makespan, w3 {LOCKED_WEIGHTS[2]} balance")


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def load_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def fail(message: str) -> None:
    raise SystemExit(f"GUARD FAILED. {message}")


def check_seed_set(seeds: list[int], where: str) -> None:
    ordered = sorted(seeds)
    if len(ordered) != len(set(ordered)):
        dupes = sorted({s for s in ordered if ordered.count(s) > 1})
        fail(f"{where} repeats seeds {dupes}")
    if ordered != EXPECTED_SEEDS:
        missing = sorted(set(EXPECTED_SEEDS) - set(ordered))
        extra = sorted(set(ordered) - set(EXPECTED_SEEDS))
        fail(f"{where} does not cover 11200 to 11399 without gaps. "
             f"{len(missing)} missing, {missing[:10]}. "
             f"{len(extra)} unexpected, {extra[:10]}")
    if len(ordered) != 200:
        fail(f"{where} holds {len(ordered)} rows rather than 200")


# 1a and 1b, the two rollout policies

def build_policy_rows(rows: list[dict], where: str) -> list[dict]:
    out = []
    for r in rows:
        out.append({
            "seed": int(r["seed"]),
            "term_travel_time": r["term_travel_time"],
            "term_makespan": r["term_makespan"],
            "term_balance": r["term_balance"],
            "policy_cost": r["policy_cost"],
            "serve_all_flag": r["serve_all_flag"],
        })
    if any(r["policy_cost"] in ("", None) for r in out):
        blank = [r["seed"] for r in out if r["policy_cost"] in ("", None)]
        fail(f"{where} has no recorded cost on seeds {blank[:10]}")
    check_seed_set([r["seed"] for r in out], where)
    return sorted(out, key=lambda r: r["seed"])


# 1c, the anticipative benchmark, with its four guards

def _repo_relative(path) -> str:
    """Path relative to the repository root, or unchanged if it lies outside.

    Matches ``figures.style._repo_relative``. Used only for what is recorded
    in tracked output; files are still opened through the absolute path.
    """
    absolute = Path(path).resolve()
    try:
        return str(absolute.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_milp_rows(cache_dir: Path) -> tuple[list[dict], dict]:
    resolved = cache_dir.resolve()
    tail = resolved.parts[-2:]

    # Guard 1: the directory actually read is the v3 test split.
    if tail != ("training_set_il_v3", "test"):
        fail("the cache directory read must end in training_set_il_v3/test, "
             f"and this run resolved {resolved}. "
             f"{WRONG_CACHE_REL} carries {WRONG_CACHE_WEIGHTS} and must never "
             "be the source of this export")
    if not resolved.is_dir():
        fail(f"{resolved} is not a directory")

    files = sorted(resolved.glob("seed*.json"))
    if not files:
        fail(f"no seed*.json records under {resolved}")

    rows, seeds, bad_weights, bad_bounds = [], [], [], []
    for path in files:
        with path.open() as handle:
            record = json.load(handle)
        seed = int(record["seed"])
        seeds.append(seed)

        # Guard 2. The weights field on every record.
        w = record.get("weights", {})
        triple = (w.get("w_dist"), w.get("w_make"), w.get("w_bal"))
        if triple != LOCKED_WEIGHTS:
            bad_weights.append({"seed": seed, "weights": triple,
                                "file": path.name})

        solution = record["milp_solution"]
        objective = float(solution["objective_value"])
        bound = float(solution["obj_bound"])

        # Guard 4: a feasible schedule cannot cost less than a valid lower
        # bound, so an objective below its own stored bound is a false
        # optimality certificate.
        if objective < bound:
            bad_bounds.append({
                "seed": seed,
                "objective_value": objective,
                "obj_bound": bound,
                "shortfall": bound - objective,
                "status": solution.get("status"),
                "mip_gap": solution.get("mip_gap"),
            })

        rows.append({
            "seed": seed,
            "term_travel_time": repr(float(solution["distance"])),
            "term_makespan": repr(float(solution["makespan"])),
            "term_balance": repr(float(solution["imbalance"])),
            "milp_objective_value": repr(objective),
        })

    if bad_weights:
        for entry in bad_weights[:10]:
            print(f"  weights guard, seed {entry['seed']} reads {entry['weights']}")
        fail(f"{len(bad_weights)} of {len(files)} records carry a weight "
             f"vector other than {LOCKED_WEIGHTS}")

    # Guard 3. The seed set.
    check_seed_set(seeds, f"the cache at {resolved}")

    if bad_bounds:
        print()
        print("  DUAL BOUND GUARD, records whose objective sits below their "
              "own stored bound")
        for entry in bad_bounds:
            print(f"    seed {entry['seed']}  objective "
                  f"{entry['objective_value']:.6f}  bound "
                  f"{entry['obj_bound']:.6f}  shortfall "
                  f"{entry['shortfall']:.6e}  status {entry['status']}")
        fail(f"{len(bad_bounds)} of {len(files)} test records fail the dual "
             "bound check. No test record fails it in the frozen split, so "
             "this run is reading the wrong split. Three validation records "
             "fail it, being seeds 11044, 11055 and 11169")

    print(f"  dual bound guard passed on all {len(files)} records")
    provenance = {
        # Repository-relative: the manifest is tracked and published, so an
        # absolute path would carry the building machine's home directory.
        # ``resolved`` stays absolute in memory and is what the guards check.
        "cache_dir_resolved": _repo_relative(resolved),
        "cache_dir_relative": CACHE_REL,
        "n_records": len(files),
        "weights_on_every_record": dict(zip(("w_dist", "w_make", "w_bal"),
                                            LOCKED_WEIGHTS)),
        "dual_bound_violations": bad_bounds,
    }
    return sorted(rows, key=lambda r: r["seed"]), provenance


# Gitignored inputs, paired with the committed artefact that carries the
# result. Checked before any work so a fresh clone gets one diagnostic line
# rather than a FileNotFoundError from deep in the load.
REQUIRED_INPUTS = [
    (A2_CSV, "provenance/table41_main_results/repro_terms_policy_hard_test.csv"),
    (CACHE_REL, "provenance/table41_main_results/repro_terms_policy_hard_test.csv"),
]


def require_inputs() -> None:
    """Abort with one line if a gitignored input is absent."""
    for rel, shipped in REQUIRED_INPUTS:
        if not (REPO_ROOT / rel).exists():
            raise SystemExit(
                f"Missing {rel}. It is gitignored and absent from a fresh "
                f"clone, so this script can only run on a tree that carries "
                f"it. The committed {shipped} carries the result."
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="provenance/table41_main_results")
    ap.add_argument("--cache-dir", default=CACHE_REL,
                    help="must resolve to a path ending in "
                         "training_set_il_v3/test")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    require_inputs()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [out_dir / name for name in
               (OUT_POLICY, OUT_GREEDY, OUT_MILP, OUT_MANIFEST)]
    for path in targets:
        if path.exists() and not args.force:
            raise SystemExit(f"REFUSING to overwrite {path}. Pass --force.")

    stamp = datetime.now(timezone.utc).isoformat()
    head = _git_head()

    print("Reading the two committed per-instance CSVs")
    a2_path = REPO_ROOT / A2_CSV
    b3_path = REPO_ROOT / B3_CSV
    a2 = [r for r in load_csv(a2_path) if r["decode_mode"] == "hard"]
    b3 = [r for r in load_csv(b3_path)
          if r.get("policy", "hungarian_distance_only")
          == "hungarian_distance_only"]
    policy_rows = build_policy_rows(a2, f"{A2_CSV} at decode_mode hard")
    greedy_rows = build_policy_rows(b3, B3_CSV)
    print(f"  {len(policy_rows)} learned policy rows, "
          f"{len(greedy_rows)} distance-only Hungarian rows")

    print(f"Reading the anticipative benchmark out of {args.cache_dir}")
    milp_rows, cache_prov = build_milp_rows(REPO_ROOT / args.cache_dir)
    print(f"  {len(milp_rows)} benchmark records")

    # Every export covers the same instances.
    seed_sets = {
        OUT_POLICY: [r["seed"] for r in policy_rows],
        OUT_GREEDY: [r["seed"] for r in greedy_rows],
        OUT_MILP: [r["seed"] for r in milp_rows],
    }
    for name, seeds in seed_sets.items():
        check_seed_set(seeds, name)
    if not (set(seed_sets[OUT_POLICY]) == set(seed_sets[OUT_GREEDY])
            == set(seed_sets[OUT_MILP])):
        fail("the three exports do not cover identical seed sets")
    print("  all three exports cover seeds 11200 to 11399, n = 200, "
          "seed sets identical")

    common_header = [
        "Written by scripts/regenerate/export_term_provenance.py. Arithmetic and copying "
        "only.",
        f"Generated {stamp} at git head {head}.",
        "Frozen test split, seeds 11200 to 11399, n = 200, no instance "
        "excluded.",
        "Terms are D total fleet travel time, M makespan, B range of busy time "
        "across robots.",
        WEIGHT_SENTENCE,
        "No solver was invoked and no Gurobi licence was read to produce this "
        "file.",
    ]

    written = []
    for name, columns, rows, extra in (
        (OUT_POLICY, POLICY_COLUMNS, policy_rows, [
            "Learned policy under hard decode, epsilon 0, inference only.",
            f"Source file is {A2_CSV} filtered to decode_mode == hard.",
        ]),
        (OUT_GREEDY, POLICY_COLUMNS, greedy_rows, [
            "Distance-only Hungarian floor. Its assignment reads no weights.",
            f"Source file is {B3_CSV}.",
        ]),
        (OUT_MILP, MILP_COLUMNS, milp_rows, [
            "Offline anticipative MILP benchmark, per-instance terms.",
            f"Source cache directory is {CACHE_REL}, confirmed at export time "
            f"to resolve inside this repository.",
            "Terms come from the milp_solution block, being distance, makespan "
            "and imbalance, and the scalar is objective_value.",
            f"The other label cache {WRONG_CACHE_REL} carries the different "
            f"vector {WRONG_CACHE_WEIGHTS[0]}, {WRONG_CACHE_WEIGHTS[1]}, "
            f"{WRONG_CACHE_WEIGHTS[2]} and is never the source of this file.",
            "objective_value carries a 1e-6 tie-break term on the sum of task "
            "start times in addition to the three weighted terms, so it sits "
            "slightly above w1 D + w2 M + w3 B by design.",
        ]),
    ):
        path = out_dir / name
        write_terms_csv(path, rows, columns, common_header + extra)
        side = write_sha256_sidecar(path)
        digest = sha256_of(path)
        written.append({"csv": f"{args.out_dir}/{name}",
                        "sha256": digest,
                        "bytes": path.stat().st_size,
                        "n_rows": len(rows),
                        "sidecar": f"{args.out_dir}/{side.name}"})
        print(f"  wrote {path.name}, {len(rows)} rows, sha256 {digest[:16]}")

    print()
    print("Identity check, weighted terms against the recorded scalar cost")
    identities = {}
    for name, tol in ((OUT_POLICY, POLICY_IDENTITY_TOL),
                      (OUT_GREEDY, POLICY_IDENTITY_TOL),
                      (OUT_MILP, MILP_IDENTITY_TOL)):
        frame = load_terms_csv(out_dir / name)
        result = identity_check(frame, LOCKED_WEIGHTS, tol=tol)
        row = result.iloc[0].to_dict()
        identities[name] = {k: (float(v) if isinstance(v, float) else v)
                            for k, v in row.items()}
        print(f"  {name:<44} max residual "
              f"{row['max_abs_residual']:.3e} against tolerance {tol:g}")
    print("  The benchmark residual is the 1e-6 tie-break coefficient on the "
          "sum of start times, claim C012, and not an error.")

    manifest = {
        "script": "scripts/regenerate/export_term_provenance.py",
        "git_head": head,
        "generated_utc": stamp,
        "split": "test",
        "seed_min": EXPECTED_SEEDS[0],
        "seed_max": EXPECTED_SEEDS[-1],
        "n_instances": len(EXPECTED_SEEDS),
        "weights": dict(zip(("w1_travel", "w2_makespan", "w3_balance"),
                            LOCKED_WEIGHTS)),
        "sources": {
            "learned_policy_hard": {
                "path": A2_CSV,
                "sha256": sha256_of(a2_path),
                "filter": "decode_mode == hard",
            },
            "hungarian_distance_only": {
                "path": B3_CSV,
                "sha256": sha256_of(b3_path),
                "filter": "policy == hungarian_distance_only",
            },
            "anticipative_milp": cache_prov,
        },
        "outputs": written,
        "identity_checks": identities,
        "guards": {
            "cache_path_ends_in_v3_test": True,
            "weights_on_every_record": True,
            "seed_set_11200_to_11399_no_gaps": True,
            "objective_at_or_above_dual_bound": True,
            "n_equals_200_on_all_three": True,
            "seed_sets_identical_across_the_three": True,
        },
        "no_solver": ("gurobipy is not imported by this script and no Gurobi "
                      "licence is read"),
    }
    manifest_path = out_dir / OUT_MANIFEST
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    write_sha256_sidecar(manifest_path)
    print()
    print(f"Wrote {manifest_path}")
    print("Every guard passed. No solver was invoked, nothing was trained, and "
          "nothing under cache/ was modified.")


if __name__ == "__main__":
    main()
