"""DM-32 Phase 0: hardened rolling-horizon h=1 solver.

Outstanding since DM-27a. The offline anticipative MILP already has a known
Gurobi 12.0.2 presolve dual-bound bug producing wrong optimality
certificates on some val seeds (see the corrupted-val-milp memory / prior
session). This hardens the ONLINE h=1 rolling-horizon solver
(scripts/experiments/rolling_horizon_baseline.py, not modified) against the same class
of bug, for every window solve, not just at the end of an episode.

For every window solve:
  1. Build a feasible greedy chain assignment over the same window
     (cheapest-insertion: each task, in release-time order, goes to
     whichever robot completes it soonest given its current chain cursor).
     This is always feasible -- every window task gets served by exactly
     one robot in a well-formed per-robot chain -- and its objective is
     computed with the *exact same formula* solve_rolling_window optimises
     (same D/M_make/B_plus-B_minus/tiebreak terms), so it is a genuine
     point in the MILP's own feasible region with a directly comparable
     objective value.
  2. Warm-start Gurobi with this greedy assignment (set .Start on every arc
     variable) before optimizing -- a real MIP start, not just a
     comparison point computed on the side.
  3. After optimize(), if Gurobi returns an incumbent (SolCount > 0),
     assert ObjVal <= greedy_objective + a small numerical tolerance. Since
     greedy is a feasible point in the MILP's own search space, no
     correctly-functioning solver can report an incumbent worse than it
     (a restriction cannot beat its relaxation). A violation is logged
     (seed, epoch, returned objective, greedy objective, Gurobi status) and
     the epoch falls back to the greedy assignment -- violations are
     recorded, never silently repaired or substituted with a "corrected"
     objective.

solve_rolling_window's variable/constraint-building code is reproduced here
verbatim (not imported and monkeypatched) so the warm start can be attached
before optimize() is called; scripts/experiments/rolling_horizon_baseline.py itself is
never modified. run_rolling_horizon_policy_hardened mirrors
run_rolling_horizon_policy's loop, calling the hardened solve and
threading seed/epoch through for violation logging.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from anticipative.anticipative_milp import _travel, _start_node, _end_node
from baselines.bipartite_policies import build_kappa_cost_matrix, hungarian_action
from instances.synthetic_generator import SyntheticInstance
from simulator.dynamic_simulator import DynamicSimulator

import time

import rolling_horizon_baseline as rhb


OBJ_TOLERANCE = 1e-4  # absolute; accounts for floating-point solve precision only


def greedy_window_solution(
    instance: SyntheticInstance,
    win_ids: List[int],
    positions: np.ndarray,
    finish_times: np.ndarray,
    wall_clock: float,
    weights: dict,
    past_busy: np.ndarray,
) -> Tuple[Dict[tuple, int], float]:
    """Cheapest-insertion greedy chain assignment over the window.

    Every task (in release-time order) goes to whichever robot completes it
    soonest given that robot's current chain cursor (position and time).
    Always feasible: every task ends up served by exactly one robot in a
    well-formed sequential chain, satisfying every constraint
    solve_rolling_window encodes (flow conservation and start/end-once by
    construction; completion-time and release-time constraints by
    construction of the completion-time formula itself, with equality
    where solve_rolling_window's indicator constraints require ">=").

    Returns (arc_solution dict in the same {(r,i,j): 1} format
    solve_rolling_window returns, objective value computed with the exact
    same formula solve_rolling_window optimises).
    """
    R = int(instance.R)
    n = len(win_ids)
    cfg = instance.config
    v = float(cfg["v"])
    Delta = float(cfg["Delta"])
    tasks_by_id = {int(t["id"]): t for t in instance.tasks}

    pickups = {j: np.asarray(tasks_by_id[win_ids[j]]["pickup"], dtype=np.float64) for j in range(n)}
    drops = {j: np.asarray(tasks_by_id[win_ids[j]]["drop"], dtype=np.float64) for j in range(n)}
    durations = {j: float(tasks_by_id[win_ids[j]]["duration"]) for j in range(n)}
    release_times = {
        j: float(int(tasks_by_id[win_ids[j]]["release_epoch"])) * Delta for j in range(n)
    }
    r_start = {r: max(float(finish_times[r]), float(wall_clock)) for r in range(R)}

    if n == 0:
        arc_solution = {(r, _start_node(r), _end_node(r)): 1 for r in range(R)}
        w_make = float(weights["w_make"])
        w_bal = float(weights["w_bal"])
        tau_free = np.array([max(0.0, float(finish_times[r])) for r in range(R)])
        tau_busy = np.array([max(0.0, float(past_busy[r])) for r in range(R)])
        M_make = float(np.max(tau_free)) if R else 0.0
        B_plus = float(np.max(tau_busy)) if R else 0.0
        B_minus = float(np.min(tau_busy)) if R else 0.0
        return arc_solution, w_make * M_make + w_bal * (B_plus - B_minus)

    cursor_pos = {r: np.asarray(positions[r], dtype=np.float64) for r in range(R)}
    cursor_time = {r: r_start[r] for r in range(R)}
    chains: Dict[int, List[int]] = {r: [] for r in range(R)}
    f_val: Dict[tuple, float] = {}

    order = sorted(range(n), key=lambda j: (release_times[j], j))
    for j in order:
        best_r, best_completion = None, None
        for r in range(R):
            tp = _travel(cursor_pos[r], pickups[j], v)
            arrival = cursor_time[r] + tp
            start_service = max(arrival, release_times[j])
            completion = start_service + _travel(pickups[j], drops[j], v) + durations[j]
            if best_completion is None or completion < best_completion:
                best_r, best_completion = r, completion
        chains[best_r].append(j)
        f_val[(best_r, j)] = best_completion
        cursor_pos[best_r] = drops[j]
        cursor_time[best_r] = best_completion

    arc_solution: Dict[tuple, int] = {}
    for r in range(R):
        s_r, e_r = _start_node(r), _end_node(r)
        chain = chains[r]
        if not chain:
            arc_solution[(r, s_r, e_r)] = 1
        else:
            arc_solution[(r, s_r, chain[0])] = 1
            for a, b in zip(chain, chain[1:]):
                arc_solution[(r, a, b)] = 1
            arc_solution[(r, chain[-1], e_r)] = 1

    w_dist = float(weights["w_dist"])
    w_make = float(weights["w_make"])
    w_bal = float(weights["w_bal"])

    D = 0.0
    tau_busy = np.array([float(past_busy[r]) for r in range(R)])
    for r in range(R):
        prev_pos = np.asarray(positions[r], dtype=np.float64)
        for j in chains[r]:
            tp = _travel(prev_pos, pickups[j], v)
            tpd = _travel(pickups[j], drops[j], v)
            D += tp + tpd
            tau_busy[r] += tp + tpd + durations[j]
            prev_pos = drops[j]

    tau_free = np.array([
        max(float(finish_times[r]), f_val[(r, chains[r][-1])]) if chains[r]
        else max(0.0, float(finish_times[r]))
        for r in range(R)
    ])
    M_make = float(np.max(tau_free)) if R else 0.0
    B_plus = float(np.max(tau_busy)) if R else 0.0
    B_minus = float(np.min(tau_busy)) if R else 0.0
    tiebreak = sum(f_val.values())

    objective = w_dist * D + w_make * M_make + w_bal * (B_plus - B_minus) + 1e-6 * tiebreak
    return arc_solution, objective


def solve_rolling_window_hardened(
    instance: SyntheticInstance,
    window_task_ids: List[int],
    positions: np.ndarray,
    finish_times: np.ndarray,
    wall_clock: float,
    weights: dict,
    past_busy: np.ndarray,
    time_limit_seconds: int = 1,
    mip_gap: float = 0.01,
    seed: int = 0,
    log_seed: int = -1,
    log_epoch: int = -1,
    violations: Optional[list] = None,
    timing_out: Optional[dict] = None,
) -> Optional[Dict[tuple, int]]:
    """solve_rolling_window, hardened with a greedy MIP warm start and a
    post-solve no-worse-than-greedy assertion. Variable/constraint-building
    code is reproduced verbatim from rolling_horizon_baseline.solve_rolling_window
    (that module is never imported-and-monkeypatched, nor modified) so the
    warm start can be attached before optimize().

    Returns the arc solution as usual, or the greedy fallback (with the
    violation recorded in ``violations``, never silently repaired) if
    Gurobi's returned objective is worse than the greedy warm start's own
    objective -- a mathematical impossibility for a correct solve, since
    greedy is itself a feasible point in the MILP's search space.
    """
    R = int(instance.R)
    win_ids = sorted(int(j) for j in window_task_ids)
    n = len(win_ids)
    if n == 0:
        return {}

    cfg = instance.config
    Delta = float(cfg["Delta"])
    v = float(cfg["v"])
    tasks_by_id = {int(t["id"]): t for t in instance.tasks}

    pickups = {j: np.asarray(tasks_by_id[win_ids[j]]["pickup"], dtype=np.float64) for j in range(n)}
    drops = {j: np.asarray(tasks_by_id[win_ids[j]]["drop"], dtype=np.float64) for j in range(n)}
    durations = {j: float(tasks_by_id[win_ids[j]]["duration"]) for j in range(n)}
    release_times = {
        j: float(int(tasks_by_id[win_ids[j]]["release_epoch"])) * Delta for j in range(n)
    }
    r_start = {r: max(float(finish_times[r]), float(wall_clock)) for r in range(R)}

    w_dist = float(weights["w_dist"])
    w_make = float(weights["w_make"])
    w_bal = float(weights["w_bal"])

    # Computed before the model so its objective is available for the post-solve assertion regardless of solve outcome.
    greedy_arc, greedy_obj = greedy_window_solution(
        instance, win_ids, positions, finish_times, wall_clock, weights, past_busy
    )

    t_build_start = time.perf_counter()
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("rolling_window_mrta_hardened", env=env)
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
                    x[(r, i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_r{r}_t{i}_t{j}")
        for i in range(n):
            x[(r, i, e_r)] = model.addVar(vtype=GRB.BINARY, name=f"x_r{r}_t{i}_E")

    f_var: Dict[tuple, gp.Var] = {}
    for r in range(R):
        for j in range(n):
            f_var[(r, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"f_r{r}_t{j}")

    tau_free = {
        r: model.addVar(vtype=GRB.CONTINUOUS, lb=max(0.0, float(finish_times[r])), name=f"tau_free_r{r}")
        for r in range(R)
    }
    tau_busy = {
        r: model.addVar(vtype=GRB.CONTINUOUS, lb=max(0.0, float(past_busy[r])), name=f"tau_busy_r{r}")
        for r in range(R)
    }
    M_make = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="M_make")
    B_plus = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="B_plus")
    B_minus = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="B_minus")

    served: Dict[tuple, gp.Var] = {}
    for r in range(R):
        s_r = _start_node(r)
        for j in range(n):
            served[(r, j)] = model.addVar(vtype=GRB.BINARY, name=f"served_r{r}_t{j}")
            in_arcs = [x[(r, s_r, j)]] + [x[(r, i, j)] for i in range(n) if i != j]
            model.addConstr(served[(r, j)] == gp.quicksum(in_arcs), name=f"def_served_r{r}_t{j}")

    model.update()

    for j in range(n):
        model.addConstr(gp.quicksum(served[(r, j)] for r in range(R)) == 1, name=f"task_t{j}_served_once")

    for r in range(R):
        e_r = _end_node(r)
        for j in range(n):
            out_arcs = [x[(r, j, e_r)]] + [x[(r, j, k)] for k in range(n) if k != j]
            model.addConstr(served[(r, j)] == gp.quicksum(out_arcs), name=f"flow_r{r}_t{j}")

    for r in range(R):
        s_r = _start_node(r)
        e_r = _end_node(r)
        out_start = [x[(r, s_r, j)] for j in range(n)] + [x[(r, s_r, e_r)]]
        model.addConstr(gp.quicksum(out_start) == 1, name=f"start_r{r}")
        in_end = [x[(r, i, e_r)] for i in range(n)] + [x[(r, s_r, e_r)]]
        model.addConstr(gp.quicksum(in_end) == 1, name=f"end_r{r}")
        model.addConstr(gp.quicksum(x[(r, s_r, j)] for j in range(n)) <= 1, name=f"chain_start_r{r}")

    for r in range(R):
        s_r = _start_node(r)
        pos_r = np.asarray(positions[r], dtype=np.float64)
        for j in range(n):
            tp_sj = _travel(pos_r, pickups[j], v)
            tpd_j = _travel(pickups[j], drops[j], v)
            delta_sj = r_start[r] + tp_sj + tpd_j + durations[j]
            model.addGenConstrIndicator(
                x[(r, s_r, j)], 1, f_var[(r, j)], GRB.GREATER_EQUAL, delta_sj, name=f"comp_start_r{r}_t{j}",
            )
            for i in range(n):
                if i != j:
                    tp_ij = _travel(drops[i], pickups[j], v)
                    delta_ij = tp_ij + tpd_j + durations[j]
                    model.addGenConstrIndicator(
                        x[(r, i, j)], 1, f_var[(r, j)] - f_var[(r, i)],
                        GRB.GREATER_EQUAL, delta_ij, name=f"comp_r{r}_t{i}_t{j}",
                    )

    for r in range(R):
        for j in range(n):
            tpd_j = _travel(pickups[j], drops[j], v)
            rhs = release_times[j] + tpd_j + durations[j]
            model.addGenConstrIndicator(
                served[(r, j)], 1, f_var[(r, j)], GRB.GREATER_EQUAL, rhs, name=f"release_r{r}_t{j}",
            )

    for r in range(R):
        for j in range(n):
            model.addGenConstrIndicator(
                served[(r, j)], 1, tau_free[r] - f_var[(r, j)], GRB.GREATER_EQUAL, 0.0, name=f"tfree_r{r}_t{j}",
            )

    for r in range(R):
        s_r = _start_node(r)
        pos_r = np.asarray(positions[r], dtype=np.float64)
        terms = []
        for j in range(n):
            sc = _travel(pos_r, pickups[j], v) + _travel(pickups[j], drops[j], v) + durations[j]
            terms.append(sc * x[(r, s_r, j)])
        for i in range(n):
            for j in range(n):
                if i != j:
                    sc = _travel(drops[i], pickups[j], v) + _travel(pickups[j], drops[j], v) + durations[j]
                    terms.append(sc * x[(r, i, j)])
        model.addConstr(tau_busy[r] == float(past_busy[r]) + gp.quicksum(terms), name=f"busy_r{r}")

    for r in range(R):
        model.addConstr(M_make >= tau_free[r], name=f"M_geq_tfree_r{r}")
        model.addConstr(B_plus >= tau_busy[r], name=f"Bp_geq_tbusy_r{r}")
        model.addConstr(B_minus <= tau_busy[r], name=f"Bm_leq_tbusy_r{r}")

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
                    ell = _travel(drops[i], pickups[j], v) + _travel(pickups[j], drops[j], v)
                    dist_terms.append(ell * x[(r, i, j)])
    D_expr = gp.quicksum(dist_terms)
    tiebreak_expr = gp.quicksum(f_var[(r, j)] for r in range(R) for j in range(n))
    model.setObjective(
        w_dist * D_expr + w_make * M_make + w_bal * (B_plus - B_minus) + 1e-6 * tiebreak_expr,
        GRB.MINIMIZE,
    )

    # Continuous variables are left unset for Gurobi's own MIP-start completion.
    for key, var in x.items():
        var.Start = 1.0 if key in greedy_arc else 0.0

    build_s = time.perf_counter() - t_build_start
    t_solve_start = time.perf_counter()
    model.optimize()
    solve_wall_s = time.perf_counter() - t_solve_start

    if timing_out is not None:
        timing_out["build_s"] = build_s
        timing_out["solve_wall_s"] = solve_wall_s
        timing_out["gurobi_runtime_s"] = float(model.Runtime)
        timing_out["window_size"] = n
        timing_out["status"] = int(model.Status)
        timing_out["sol_count"] = int(model.SolCount)

    status = int(model.Status)
    if model.SolCount == 0:
        model.dispose()
        env.dispose()
        return None

    returned_obj = float(model.ObjVal)
    if returned_obj > greedy_obj + OBJ_TOLERANCE:
        if violations is not None:
            violations.append({
                "seed": log_seed, "epoch": log_epoch,
                "returned_objective": returned_obj, "greedy_objective": greedy_obj,
                "gurobi_status": status,
            })
        # Do not silently repair: record and fall back to the greedy point.
        model.dispose()
        env.dispose()
        return greedy_arc

    arc_solution: Dict[tuple, int] = {}
    for key, var in x.items():
        if var.X > 0.5:
            arc_solution[key] = 1
    model.dispose()
    env.dispose()
    return arc_solution


def run_rolling_horizon_policy_hardened(
    instance: SyntheticInstance,
    weights: dict,
    horizon: int,
    time_limit_seconds: int = 1,
    seed: int = 0,
    log_seed: int = -1,
    violations: Optional[list] = None,
    timings: Optional[list] = None,
    on_solve: Optional[Callable[[], None]] = None,
) -> Tuple[DynamicSimulator, int, int]:
    """run_rolling_horizon_policy, calling the hardened solve and threading
    seed/epoch through for violation logging. Mirrors the original loop
    (rolling_horizon_baseline.run_rolling_horizon_policy) exactly otherwise,
    including the Hungarian-on-kappa fallback when Gurobi returns no
    incumbent.

    ``timings``, if given, gets one dict appended per window solve (the same
    fields solve_rolling_window_hardened's own ``timing_out`` populates:
    build_s, solve_wall_s, gurobi_runtime_s, window_size, status, sol_count).
    ``on_solve``, if given, is called after every window solve completes (no
    arguments) -- e.g. for external RSS logging every N solves. Both are
    optional and default to None, so existing callers are unaffected."""
    H = int(instance.H)
    sim = DynamicSimulator(instance)
    serviceable = {int(t["id"]) for t in instance.tasks if 0 <= int(t["release_epoch"]) < H}
    n_solves = 0
    n_fallbacks = 0

    while not sim.is_terminal:
        state = sim.state
        epoch = int(state.epoch)

        if not state.pending_tasks:
            sim.step([])
            continue

        committed_ids: Set[int] = {int(c["task_id"]) for c in sim.trajectory.commitments}
        window_ids = sorted(
            int(t["id"]) for t in instance.tasks
            if int(t["id"]) in serviceable
            and int(t["id"]) not in committed_ids
            and int(t["release_epoch"]) < epoch + horizon
        )

        finish_times = state.wall_clock + state.busy_times
        past_busy = np.zeros(int(instance.R), dtype=np.float64)
        for c in sim.trajectory.commitments:
            past_busy[int(c["robot_id"])] += float(c["completion_time"])

        n_solves += 1
        timing_out: Optional[dict] = {} if timings is not None else None
        arc_solution = solve_rolling_window_hardened(
            instance, window_ids,
            positions=state.positions, finish_times=finish_times,
            wall_clock=float(state.wall_clock), weights=weights, past_busy=past_busy,
            time_limit_seconds=time_limit_seconds, seed=seed,
            log_seed=log_seed, log_epoch=epoch, violations=violations,
            timing_out=timing_out,
        )
        if timings is not None:
            timings.append(timing_out)
        if on_solve is not None:
            on_solve()

        if arc_solution is None:
            n_fallbacks += 1
            cost, robot_ids, task_ids = build_kappa_cost_matrix(
                state, instance, weights, sim.trajectory.commitments
            )
            action = hungarian_action(cost, robot_ids, task_ids)
        else:
            action = rhb._window_commits(arc_solution, window_ids, state, instance)

        sim.step(action)

    return sim, n_solves, n_fallbacks
