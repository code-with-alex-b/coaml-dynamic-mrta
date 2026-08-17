"""Rolling-horizon Gurobi baseline for Phase 0.5.

At each dispatcher epoch this baseline solves a MILP over only the next h
epochs, starting from the simulator's current state, executes the
assignments that begin in the current epoch, then advances one epoch and
re-solves. As h grows toward H the lookahead approaches the offline
anticipative oracle (modulo the epoch-constrained idle-wait cost the
oracle does not model).

The window solve (``solve_rolling_window``) is modelled directly on
``src/anticipative/anticipative_milp.py`` (same indicator-constraint
sequence formulation, same objective and tie-break) but is parameterised by
the current state rather than the instance origin:

    initial position    -> the robot's current position (drop of its last
                           committed task, or its start position)
    ready time          -> the robot's current physical finish time, so a
                           robot still busy on a prior commit cannot begin
                           its next task before it frees up
    task set            -> only the unserved tasks releasing within the next
                           h epochs (pending now plus soon-to-release)
    accumulated load    -> each robot's already-incurred busy time and
                           current finish time enter the makespan and
                           balance auxiliaries as constants, so the window
                           solve optimises the global episode objective the
                           simulator will ultimately score, not just the
                           window's incremental contribution

The locked oracle in ``anticipative_milp.py`` is not modified; this baseline
reuses its helpers (``_travel``, ``_start_node``, ``_end_node``) and mirrors
its formulation here.

Instances and the MILP oracle reference are read from cached records under
``--cache-dir`` (the same records the IL pipeline consumes), following the
convention in ``sweep/evaluate_sweep.py``. ``--checkpoint`` is accepted for
signature parity with the learned-policy evaluators and is otherwise unused.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from anticipative.anticipative_milp import _travel, _start_node, _end_node
from baselines.bipartite_policies import (
    build_kappa_cost_matrix,
    hungarian_action,
    run_greedy_policy,
    run_hungarian_kappa_policy,
)
from instances.synthetic_generator import SyntheticInstance
from simulator.dynamic_simulator import DynamicSimulator, SimulatorState


DEFAULT_CACHE_DIR = "cache/training_set_il_v3/val"
DEFAULT_WEIGHTS = {"w_dist": 0.0637, "w_make": 0.2398, "w_bal": 0.6965}
_AVAIL_TOL = 1e-9

# measured_mean_per_decision_seconds averages only solves that invoked Gurobi; n_decisions counts every epoch, including no-solve ones, so it is larger.
PER_INSTANCE_FIELDS = [
    "seed",
    "budget_seconds",
    "policy_cost",
    "milp_oracle_cost_from_cache",
    "gap_closure",
    "serve_all_flag",
    "n_window_solves",
    "n_solves_hit_time_limit",
    "n_solves_no_incumbent",
    "n_solves_fallback_fired",
    "measured_total_solve_seconds",
    "measured_mean_per_decision_seconds",
    "n_decisions",
    # Appended after the A3 ladder ran; kept at the end so csv.DictReader still reads earlier files byte-identically.
    "measured_total_build_seconds",
    "mean_build_s",
    "mean_solve_s",
    "instance_wall_clock_seconds",
]


def solve_rolling_window(
    instance: SyntheticInstance,
    window_task_ids: List[int],
    positions: np.ndarray,
    finish_times: np.ndarray,
    wall_clock: float,
    weights: dict,
    past_busy: np.ndarray,
    time_limit_seconds: float = 10.0,
    mip_gap: float = 0.01,
    seed: int = 0,
    stats_sink: Optional[dict] = None,
) -> Optional[Dict[tuple, int]]:
    """Solve the lookahead MILP over ``window_task_ids`` from the current state.

    Returns the arc solution (``{(r, i, j): 1}`` in the per-robot sequence
    graph over the windowed tasks, indexed by window position ``j``) or
    ``None`` if Gurobi found no incumbent within the time limit. Robot ids are
    the true 0..R-1 robot indices; task index ``j`` is the position into the
    sorted ``window_task_ids`` list.

    ``stats_sink``, when given, is a dict populated in place with this solve's
    measured wall clock, Gurobi status, incumbent count and whether the time
    limit was hit. It is a pure observer: it does not alter the model, the
    solve or the return value. It is left untouched when the window is empty
    and no solve happens.
    """
    _t_build0 = time.perf_counter()
    R = int(instance.R)
    win_ids = sorted(int(j) for j in window_task_ids)
    n = len(win_ids)
    if n == 0:
        return {}

    cfg = instance.config
    Delta = float(cfg["Delta"])
    v = float(cfg["v"])
    tasks_by_id = {int(t["id"]): t for t in instance.tasks}

    pickups = {
        j: np.asarray(tasks_by_id[win_ids[j]]["pickup"], dtype=np.float64)
        for j in range(n)
    }
    drops = {
        j: np.asarray(tasks_by_id[win_ids[j]]["drop"], dtype=np.float64)
        for j in range(n)
    }
    durations = {j: float(tasks_by_id[win_ids[j]]["duration"]) for j in range(n)}
    release_times = {
        j: float(int(tasks_by_id[win_ids[j]]["release_epoch"])) * Delta
        for j in range(n)
    }

    # Matches the simulator's start = max(finish, epoch * Delta): idle robots start now, busy ones wait for their physical finish.
    r_start = {r: max(float(finish_times[r]), float(wall_clock)) for r in range(R)}

    w_dist = float(weights["w_dist"])
    w_make = float(weights["w_make"])
    w_bal = float(weights["w_bal"])

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("rolling_window_mrta", env=env)
    model.setParam("TimeLimit", float(time_limit_seconds))
    model.setParam("MIPGap", float(mip_gap))
    model.setParam("Seed", int(seed) % (2**31 - 1))

    x: Dict[tuple, gp.Var] = {}
    for r in range(R):
        s_r = _start_node(r)
        e_r = _end_node(r)
        for j in range(n):
            x[(r, s_r, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_r{r}_S_t{j}")
        x[(r, s_r, e_r)] = model.addVar(vtype=GRB.BINARY, name=f"x_r{r}_S_E")
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[(r, i, j)] = model.addVar(
                        vtype=GRB.BINARY, name=f"x_r{r}_t{i}_t{j}"
                    )
        for i in range(n):
            x[(r, i, e_r)] = model.addVar(
                vtype=GRB.BINARY, name=f"x_r{r}_t{i}_E"
            )

    f_var: Dict[tuple, gp.Var] = {}
    for r in range(R):
        for j in range(n):
            f_var[(r, j)] = model.addVar(
                vtype=GRB.CONTINUOUS, lb=0.0, name=f"f_r{r}_t{j}"
            )

    # Lower-bounded by the robot's current physical finish so it still contributes to makespan even if it serves no window task.
    tau_free = {
        r: model.addVar(
            vtype=GRB.CONTINUOUS, lb=max(0.0, float(finish_times[r])),
            name=f"tau_free_r{r}",
        )
        for r in range(R)
    }
    # Lower-bounded by the robot's already-incurred busy load, so balance reflects history, not only the window.
    tau_busy = {
        r: model.addVar(
            vtype=GRB.CONTINUOUS, lb=max(0.0, float(past_busy[r])),
            name=f"tau_busy_r{r}",
        )
        for r in range(R)
    }
    M_make = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="M_make")
    B_plus = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="B_plus")
    B_minus = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="B_minus")

    served: Dict[tuple, gp.Var] = {}
    for r in range(R):
        s_r = _start_node(r)
        for j in range(n):
            served[(r, j)] = model.addVar(
                vtype=GRB.BINARY, name=f"served_r{r}_t{j}"
            )
            in_arcs = [x[(r, s_r, j)]] + [
                x[(r, i, j)] for i in range(n) if i != j
            ]
            model.addConstr(
                served[(r, j)] == gp.quicksum(in_arcs),
                name=f"def_served_r{r}_t{j}",
            )

    model.update()

    # (a) Each windowed task served exactly once (serve-all-windowed).
    for j in range(n):
        model.addConstr(
            gp.quicksum(served[(r, j)] for r in range(R)) == 1,
            name=f"task_t{j}_served_once",
        )

    # (b) Flow conservation
    for r in range(R):
        e_r = _end_node(r)
        for j in range(n):
            out_arcs = [x[(r, j, e_r)]] + [
                x[(r, j, k)] for k in range(n) if k != j
            ]
            model.addConstr(
                served[(r, j)] == gp.quicksum(out_arcs),
                name=f"flow_r{r}_t{j}",
            )

    # (c) Each robot starts and ends exactly once
    for r in range(R):
        s_r = _start_node(r)
        e_r = _end_node(r)
        out_start = [x[(r, s_r, j)] for j in range(n)] + [x[(r, s_r, e_r)]]
        model.addConstr(gp.quicksum(out_start) == 1, name=f"start_r{r}")
        in_end = [x[(r, i, e_r)] for i in range(n)] + [x[(r, s_r, e_r)]]
        model.addConstr(gp.quicksum(in_end) == 1, name=f"end_r{r}")
        model.addConstr(
            gp.quicksum(x[(r, s_r, j)] for j in range(n)) <= 1,
            name=f"chain_start_r{r}",
        )

    # (d) Completion-time consistency: the start arc anchors on r_start[r]; chain arcs are relative, exactly as in the offline oracle.
    for r in range(R):
        s_r = _start_node(r)
        pos_r = np.asarray(positions[r], dtype=np.float64)
        for j in range(n):
            tp_sj = _travel(pos_r, pickups[j], v)
            tpd_j = _travel(pickups[j], drops[j], v)
            delta_sj = r_start[r] + tp_sj + tpd_j + durations[j]
            model.addGenConstrIndicator(
                x[(r, s_r, j)], 1, f_var[(r, j)],
                GRB.GREATER_EQUAL, delta_sj, name=f"comp_start_r{r}_t{j}",
            )
            for i in range(n):
                if i != j:
                    tp_ij = _travel(drops[i], pickups[j], v)
                    delta_ij = tp_ij + tpd_j + durations[j]
                    model.addGenConstrIndicator(
                        x[(r, i, j)], 1, f_var[(r, j)] - f_var[(r, i)],
                        GRB.GREATER_EQUAL, delta_ij, name=f"comp_r{r}_t{i}_t{j}",
                    )

    # (e) Release time (absolute frame, same as the oracle).
    for r in range(R):
        for j in range(n):
            tpd_j = _travel(pickups[j], drops[j], v)
            rhs = release_times[j] + tpd_j + durations[j]
            model.addGenConstrIndicator(
                served[(r, j)], 1, f_var[(r, j)],
                GRB.GREATER_EQUAL, rhs, name=f"release_r{r}_t{j}",
            )

    # (f) Robot finish time
    for r in range(R):
        for j in range(n):
            model.addGenConstrIndicator(
                served[(r, j)], 1, tau_free[r] - f_var[(r, j)],
                GRB.GREATER_EQUAL, 0.0, name=f"tfree_r{r}_t{j}",
            )

    # (g) Busy-time accumulation, offset by the robot's past load.
    for r in range(R):
        s_r = _start_node(r)
        pos_r = np.asarray(positions[r], dtype=np.float64)
        terms = []
        for j in range(n):
            sc = (
                _travel(pos_r, pickups[j], v)
                + _travel(pickups[j], drops[j], v)
                + durations[j]
            )
            terms.append(sc * x[(r, s_r, j)])
        for i in range(n):
            for j in range(n):
                if i != j:
                    sc = (
                        _travel(drops[i], pickups[j], v)
                        + _travel(pickups[j], drops[j], v)
                        + durations[j]
                    )
                    terms.append(sc * x[(r, i, j)])
        model.addConstr(
            tau_busy[r] == float(past_busy[r]) + gp.quicksum(terms),
            name=f"busy_r{r}",
        )

    # (h) Makespan / imbalance auxiliaries
    for r in range(R):
        model.addConstr(M_make >= tau_free[r], name=f"M_geq_tfree_r{r}")
        model.addConstr(B_plus >= tau_busy[r], name=f"Bp_geq_tbusy_r{r}")
        model.addConstr(B_minus <= tau_busy[r], name=f"Bm_leq_tbusy_r{r}")

    # Distance over windowed arcs only, since past distance is a constant that does not change the argmin; tie-break matches the oracle's 1e-6 coefficient.
    dist_terms = []
    for r in range(R):
        s_r = _start_node(r)
        pos_r = np.asarray(positions[r], dtype=np.float64)
        for j in range(n):
            ell = _travel(pos_r, pickups[j], v) + _travel(pickups[j], drops[j], v)
            dist_terms.append(ell * x[(r, s_r, j)])
        for i in range(n):
            for j in range(n):
                if i != j:
                    ell = _travel(drops[i], pickups[j], v) + _travel(
                        pickups[j], drops[j], v
                    )
                    dist_terms.append(ell * x[(r, i, j)])
    D_expr = gp.quicksum(dist_terms)
    tiebreak_expr = gp.quicksum(
        f_var[(r, j)] for r in range(R) for j in range(n)
    )
    model.setObjective(
        w_dist * D_expr
        + w_make * M_make
        + w_bal * (B_plus - B_minus)
        + 1e-6 * tiebreak_expr,
        GRB.MINIMIZE,
    )

    _t0 = time.perf_counter()
    model.optimize()
    _t1 = time.perf_counter()

    if stats_sink is not None:
        stats_sink.update({
            # Separated from solve time because model build is not bounded by TimeLimit and dominates at large windows.
            "build_seconds": _t0 - _t_build0,
            "solve_seconds": _t1 - _t0,
            "status": int(model.Status),
            "sol_count": int(model.SolCount),
            "hit_time_limit": bool(model.Status == GRB.TIME_LIMIT),
            "no_incumbent": bool(model.SolCount == 0),
        })

    if model.SolCount == 0:
        return None

    arc_solution: Dict[tuple, int] = {}
    for key, var in x.items():
        if var.X > 0.5:
            arc_solution[key] = 1
    return arc_solution


def _window_commits(
    arc_solution: Dict[tuple, int],
    win_ids: List[int],
    state: SimulatorState,
    instance: SyntheticInstance,
) -> List[Tuple[int, int]]:
    """Extract the assignments to execute this epoch from the window solution.

    Commit a robot's first planned task (its ``S_r -> j`` arc) only when the
    robot is physically available now and the task is currently pending. A
    robot the MILP plans to start on a not-yet-released task naturally yields
    no commit this epoch (that task is not pending) and is re-planned next
    epoch after re-solving, which is the rolling-horizon execute-first-step
    rule.
    """
    tasks_by_id = {int(t["id"]): t for t in instance.tasks}
    commits: List[Tuple[int, int]] = []
    for (r, i, j), _ in arc_solution.items():
        if i != _start_node(r):
            continue
        if j == _end_node(r):
            continue
        real_id = win_ids[j]
        if r in state.available_robots and real_id in state.pending_tasks:
            commits.append((int(r), int(real_id)))
    return commits


def run_rolling_horizon_policy(
    instance: SyntheticInstance,
    weights: dict,
    horizon: int,
    time_limit_seconds: float = 10.0,
    seed: int = 0,
    verbose: bool = False,
    solve_stats_out: Optional[List[dict]] = None,
) -> Tuple[DynamicSimulator, int, int]:
    """Roll the rolling-horizon MILP policy out on one instance to termination.

    Returns ``(terminal_simulator, n_solves, n_fallbacks)`` where a fallback is
    an epoch in which Gurobi returned no incumbent and the epoch fell back to a
    Hungarian-on-kappa action so the episode still progresses.

    ``solve_stats_out``, when given, is a list appended to once per window solve
    with that solve's measured wall clock, status, incumbent count, whether the
    time limit was hit and whether the fallback fired. The return value and the
    policy's behaviour are unchanged when it is None, which is the default.
    """
    H = int(instance.H)
    sim = DynamicSimulator(instance)
    serviceable = {
        int(t["id"])
        for t in instance.tasks
        if 0 <= int(t["release_epoch"]) < H
    }
    n_solves = 0
    n_fallbacks = 0

    while not sim.is_terminal:
        state = sim.state
        epoch = int(state.epoch)

        if not state.pending_tasks:
            # Nothing to assign this epoch; advance and let busy robots finish.
            sim.step([])
            continue

        committed_ids: Set[int] = {
            int(c["task_id"]) for c in sim.trajectory.commitments
        }
        window_ids = sorted(
            int(t["id"])
            for t in instance.tasks
            if int(t["id"]) in serviceable
            and int(t["id"]) not in committed_ids
            and int(t["release_epoch"]) < epoch + horizon
        )

        finish_times = state.wall_clock + state.busy_times
        past_busy = np.zeros(int(instance.R), dtype=np.float64)
        for c in sim.trajectory.commitments:
            past_busy[int(c["robot_id"])] += float(c["completion_time"])

        n_solves += 1
        _stats: dict = {}
        arc_solution = solve_rolling_window(
            instance,
            window_ids,
            positions=state.positions,
            finish_times=finish_times,
            wall_clock=float(state.wall_clock),
            weights=weights,
            past_busy=past_busy,
            time_limit_seconds=time_limit_seconds,
            seed=seed,
            stats_sink=_stats,
        )
        if solve_stats_out is not None:
            # An empty window runs no solve, so _stats stays empty and is flagged as such rather than counted as a zero-second solve.
            solve_stats_out.append({
                "epoch": epoch,
                "window_size": len(window_ids),
                "solve_ran": bool(_stats),
                "fallback_fired": bool(arc_solution is None),
                **_stats,
            })

        if arc_solution is None:
            # No incumbent within the time limit; fall back to Hungarian-on-kappa so tasks still get served this epoch.
            n_fallbacks += 1
            cost, robot_ids, task_ids = build_kappa_cost_matrix(
                state, instance, weights, sim.trajectory.commitments
            )
            action = hungarian_action(cost, robot_ids, task_ids)
        else:
            action = _window_commits(arc_solution, window_ids, state, instance)

        sim.step(action)
        if verbose:
            print(
                f"    epoch {epoch:2d} pending={len(state.pending_tasks)} "
                f"window={len(window_ids)} committed={len(action)}",
                flush=True,
            )

    return sim, n_solves, n_fallbacks


def _served_all(sim: DynamicSimulator) -> bool:
    return not (
        sim.trajectory.simulator_failure
        or sim.state.pending_tasks
        or sim.trajectory.unserved_task_ids
    )


def load_cached_records(
    cache_dir: str, max_instances: Optional[int]
) -> List[dict]:
    paths = sorted(Path(cache_dir).glob("seed*.json"))
    if max_instances is not None:
        paths = paths[:max_instances]
    records = []
    for p in paths:
        with p.open("r") as fh:
            records.append(json.load(fh))
    return records


def _resume_seeds(out_path: Path, time_limit: float) -> Set[int]:
    """Return the seeds already complete in ``out_path``, repairing a tail tear.

    Rows are flushed one per completed instance, so an interrupted run leaves
    either whole rows or, if it died mid-flush, one truncated final line. A
    truncated line is dropped and the file rewritten without it, so the resumed
    run redoes that instance rather than inheriting a half-written record.

    Refuses to resume a file written under a different column set or a
    different budget, either of which would silently mix incomparable rows into
    one CSV.
    """
    with out_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != PER_INSTANCE_FIELDS:
            raise SystemExit(
                f"REFUSING to resume {out_path}: its header does not match the "
                f"current column set. Found {reader.fieldnames}."
            )
        good: List[dict] = []
        torn = 0
        for row in reader:
            complete = (
                None not in row.values()
                and all(row[k] != "" for k in (
                    "seed", "budget_seconds", "n_window_solves",
                    "instance_wall_clock_seconds",
                ))
            )
            if not complete:
                torn += 1
                continue
            if row["budget_seconds"] != repr(float(time_limit)):
                raise SystemExit(
                    f"REFUSING to resume {out_path}: it holds rows at budget "
                    f"{row['budget_seconds']}s but this run is at "
                    f"{float(time_limit)}s."
                )
            good.append(row)

    if torn:
        # Only rewritten after every good row is in memory, so the file is never left shorter than its content.
        with out_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=PER_INSTANCE_FIELDS)
            writer.writeheader()
            writer.writerows(good)
        print(
            f"RESUME: dropped {torn} incomplete row(s) from {out_path}; "
            f"those instances will be rerun.",
            flush=True,
        )

    seeds = [int(r["seed"]) for r in good]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(
            f"REFUSING to resume {out_path}: it holds duplicate seeds."
        )
    return set(seeds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rolling-horizon Gurobi baseline for Phase 0.5."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Accepted for parity with policy evaluators; unused (this is a "
        "model-free baseline).",
    )
    parser.add_argument("--eval-instances", "--eval_instances",
                        dest="eval_instances", type=int, default=200)
    parser.add_argument("--eval-seed", "--eval_seed",
                        dest="eval_seed", type=int, default=0,
                        help="Gurobi seed for the per-epoch solves.")
    parser.add_argument("--cache-dir", "--cache_dir",
                        dest="cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--horizon", type=int, required=True,
                        help="Lookahead h in epochs.")
    parser.add_argument("--time-limit", "--time_limit",
                        dest="time_limit", type=float, default=10.0,
                        help="Gurobi time limit per epoch solve, in seconds. "
                             "Accepts sub-second budgets, e.g. 0.01.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--per-instance-out", "--per_instance_out",
                        dest="per_instance_out", type=str, default=None,
                        help="Write one CSV row per instance to this path, "
                             "flushed after every instance. Refuses to "
                             "overwrite an existing file.")
    parser.add_argument("--resume", action="store_true",
                        help="Append to an existing --per-instance-out instead "
                             "of refusing, skipping every seed already present "
                             "in it. An instance is written as one flushed row "
                             "only after its rollout completes, so resuming is "
                             "exact at instance granularity: no instance is "
                             "ever half-recorded.")
    parser.add_argument("--allow-test-split", action="store_true",
                        help="Explicit opt-in required to run against a test "
                             "split. Without it the guard below aborts.")
    args = parser.parse_args()

    # Mirrors the refusal guards in method_two_evaluator and il_trainer: the test split is reserved for the final thesis, --allow-test-split is a deliberate opt-in.
    if any(part.lower() == "test" for part in Path(args.cache_dir).parts):
        if not args.allow_test_split:
            raise SystemExit(
                f"REFUSING to run the baseline on a test split "
                f"({args.cache_dir}). The test set is reserved for the final "
                f"thesis. Pass --allow-test-split to override deliberately."
            )
        print(
            f"*** TEST SPLIT OPT-IN: running against {args.cache_dir} "
            f"because --allow-test-split was passed. ***",
            flush=True,
        )

    records = load_cached_records(args.cache_dir, args.eval_instances)
    if not records:
        raise SystemExit(f"No cached records found under {args.cache_dir}")

    # Weights come from the cache, not a module constant; every record must agree or a single weights dict cannot score the run comparably.
    seen = {}
    for rec in records:
        w = rec.get("weights")
        if w is None:
            raise SystemExit(
                f"Record seed={rec.get('seed')} carries no 'weights' field. "
                f"Refusing to fall back to the module default, which would "
                f"silently score this run under unverified weights."
            )
        seen.setdefault(json.dumps(w, sort_keys=True), []).append(rec.get("seed"))
    if len(seen) != 1:
        detail = "; ".join(
            f"{k} on {len(v)} records (e.g. seeds {v[:3]})" for k, v in seen.items()
        )
        raise SystemExit(
            f"ABORTING: cached records disagree on objective weights. {detail}"
        )
    weights = dict(json.loads(next(iter(seen))))
    print(
        f"Rolling-horizon baseline: h={args.horizon}, "
        f"time_limit={args.time_limit}s, {len(records)} instances from "
        f"{args.cache_dir}",
        flush=True,
    )
    print(f"Weights: {weights}", flush=True)

    csv_file = None
    csv_writer = None
    done_seeds: Set[int] = set()
    if args.per_instance_out is not None:
        out_path = Path(args.per_instance_out)
        if out_path.exists():
            if not args.resume:
                raise SystemExit(
                    f"REFUSING to overwrite {out_path}. Pass --resume to "
                    f"append and skip the seeds it already holds."
                )
            done_seeds = _resume_seeds(out_path, float(args.time_limit))
            print(
                f"RESUME: {out_path} already holds {len(done_seeds)} complete "
                f"rows; those seeds will be skipped.",
                flush=True,
            )
            csv_file = out_path.open("a", newline="")
            csv_writer = csv.DictWriter(csv_file, fieldnames=PER_INSTANCE_FIELDS)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = out_path.open("w", newline="")
            csv_writer = csv.DictWriter(csv_file, fieldnames=PER_INSTANCE_FIELDS)
            csv_writer.writeheader()
            csv_file.flush()

    rows: List[dict] = []
    t_start = time.time()
    for idx, rec in enumerate(records):
        instance = SyntheticInstance.from_dict(rec["instance"])
        seed = int(rec.get("seed", idx))
        if seed in done_seeds:
            print(f"[{idx + 1}/{len(records)}] seed={seed} SKIPPED, "
                  f"already in the resume file", flush=True)
            continue

        solve_stats: List[dict] = []
        _t_inst0 = time.perf_counter()
        sim_roll, n_solves, n_fallbacks = run_rolling_horizon_policy(
            instance, weights, args.horizon,
            time_limit_seconds=args.time_limit,
            seed=args.eval_seed, verbose=args.verbose,
            solve_stats_out=solve_stats,
        )
        # Excludes the greedy and Hungarian reference rollouts below, so this figure is comparable with the policy's per-instance seconds.
        inst_wall = time.perf_counter() - _t_inst0
        served_all = _served_all(sim_roll)
        roll_cost = (
            sim_roll.compute_cost(weights).combined if served_all else None
        )

        sim_greedy = run_greedy_policy(instance)
        greedy_cost = sim_greedy.compute_cost(weights).combined
        sim_hung = run_hungarian_kappa_policy(instance, weights)
        hungarian_cost = sim_hung.compute_cost(weights).combined
        # Caches without an offline anticipative oracle store milp_solution as null; left empty rather than fabricating a gap closure against a nonexistent bound.
        milp_rec = rec.get("milp_solution")
        milp_cost = (
            float(milp_rec["objective_value"]) if milp_rec is not None else None
        )

        rows.append({
            "seed": seed,
            "rolling_cost": roll_cost,
            "greedy_cost": greedy_cost,
            "hungarian_cost": hungarian_cost,
            "milp_cost": milp_cost,
            "served_all": served_all,
            "n_fallbacks": n_fallbacks,
        })
        if csv_writer is not None:
            ran = [s for s in solve_stats if s.get("solve_ran")]
            total_solve = sum(float(s["solve_seconds"]) for s in ran)
            total_build = sum(float(s["build_seconds"]) for s in ran)
            n_hit = sum(1 for s in ran if s.get("hit_time_limit"))
            n_noinc = sum(1 for s in ran if s.get("no_incumbent"))
            n_fb = sum(1 for s in solve_stats if s.get("fallback_fired"))
            # Uses the MILP objective stored in the cache record; nothing is re-solved and the denominator is not recomputed.
            denom = None if milp_cost is None else greedy_cost - milp_cost
            closure = (
                (greedy_cost - roll_cost) / denom
                if (roll_cost is not None and denom is not None and denom > 1e-9)
                else None
            )
            csv_writer.writerow({
                "seed": seed,
                "budget_seconds": repr(float(args.time_limit)),
                "policy_cost": "" if roll_cost is None else repr(roll_cost),
                "milp_oracle_cost_from_cache": (
                    "" if milp_cost is None else repr(milp_cost)
                ),
                "gap_closure": "" if closure is None else repr(closure),
                "serve_all_flag": 1 if served_all else 0,
                "n_window_solves": len(solve_stats),
                "n_solves_hit_time_limit": n_hit,
                "n_solves_no_incumbent": n_noinc,
                "n_solves_fallback_fired": n_fb,
                "measured_total_solve_seconds": repr(total_solve),
                "measured_mean_per_decision_seconds": repr(
                    total_solve / len(ran) if ran else 0.0
                ),
                "n_decisions": int(sim_roll.trajectory.epochs_run),
                "measured_total_build_seconds": repr(total_build),
                "mean_build_s": repr(total_build / len(ran) if ran else 0.0),
                "mean_solve_s": repr(total_solve / len(ran) if ran else 0.0),
                "instance_wall_clock_seconds": repr(inst_wall),
            })
            # Flush every row so a crash costs one instance, not the run.
            csv_file.flush()

        roll_str = "FAILED" if roll_cost is None else f"{roll_cost:.3f}"
        milp_str = "none" if milp_cost is None else f"{milp_cost:.3f}"
        print(
            f"[{idx + 1}/{len(records)}] seed={seed} rolling={roll_str} "
            f"hungarian={hungarian_cost:.3f} greedy={greedy_cost:.3f} "
            f"milp_lb={milp_str} solves={n_solves} "
            f"fallbacks={n_fallbacks} wall={inst_wall:.1f}s",
            flush=True,
        )

    if csv_file is not None:
        csv_file.close()
        print(f"\nWrote per-instance rows to {args.per_instance_out}", flush=True)

    if done_seeds:
        print(
            f"\nNOTE: this was a resumed run. The summary below covers only "
            f"the {len(rows)} instances executed in this invocation, not the "
            f"{len(done_seeds)} inherited from the resume file. Use the CSV "
            f"for figures over the whole set.",
            flush=True,
        )
    _report(rows, args.horizon, time.time() - t_start)


def _report(rows: List[dict], horizon: int, elapsed: float) -> None:
    n = len(rows)
    served = [r for r in rows if r["served_all"]]
    served_rate = len(served) / n if n else 0.0

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    roll_mean = _mean([r["rolling_cost"] for r in served])
    greedy_mean = _mean([r["greedy_cost"] for r in rows])
    hung_mean = _mean([r["hungarian_cost"] for r in rows])
    milp_mean = _mean([r["milp_cost"] for r in rows])

    # Per-instance relative gaps, averaged over instances the baseline served.
    gap_vs_hung = _mean([
        (r["hungarian_cost"] - r["rolling_cost"]) / r["hungarian_cost"]
        for r in served if r["hungarian_cost"] > 0
    ])
    gap_vs_milp = _mean([
        (r["rolling_cost"] - r["milp_cost"]) / r["milp_cost"]
        for r in served if r["milp_cost"] is not None and r["milp_cost"] > 0
    ])
    # Gap closure greedy -> MILP lower bound (Phase 0.5 framing).
    closure = _mean([
        (r["greedy_cost"] - r["rolling_cost"])
        / (r["greedy_cost"] - r["milp_cost"])
        for r in served
        if r["milp_cost"] is not None
        and (r["greedy_cost"] - r["milp_cost"]) > 1e-9
    ])
    total_fallbacks = sum(r["n_fallbacks"] for r in rows)

    print("\n" + "=" * 64, flush=True)
    print(f"Rolling-horizon baseline summary (h={horizon})", flush=True)
    print("=" * 64, flush=True)
    print(f"  instances evaluated      {n}", flush=True)
    print(
        f"  served-all rate          {served_rate:.3f} "
        f"({len(served)}/{n})", flush=True)
    print(f"  epochs using fallback    {total_fallbacks}", flush=True)
    print(f"  mean rolling cost        {roll_mean:.4f} "
          f"(over served instances)", flush=True)
    print(f"  mean hungarian cost      {hung_mean:.4f}", flush=True)
    print(f"  mean greedy cost         {greedy_mean:.4f}", flush=True)
    print(f"  mean MILP lower bound    {milp_mean:.4f}", flush=True)
    print(
        f"  gap vs Hungarian         {gap_vs_hung:+.4f}  "
        f"(positive = rolling beats Hungarian)", flush=True)
    print(
        f"  gap vs MILP oracle       {gap_vs_milp:+.4f}  "
        f"(fraction above the MILP lower bound)", flush=True)
    print(
        f"  gap closure greedy->MILP {closure:.4f}  "
        f"(MILP is a lower bound, so >1.0 is possible)", flush=True)
    print(f"  wall time                {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
