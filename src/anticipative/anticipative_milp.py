"""Offline anticipative MILP for Phase 0.5.

Sequence formulation per methodology section 3.2 with Gurobi indicator
constraints in place of big-M.

Decision variables. ``x[r, i, j]`` are binary arcs in the per-robot
sequence graph with nodes ``{S_r} U J U {E_r}``. ``f[r, j]`` are
continuous completion times. ``tau_free[r]`` and ``tau_busy[r]`` per
robot. ``M_make``, ``B_plus``, ``B_minus`` are objective auxiliaries.

Objective. ``w_dist * D + w_make * M + w_bal * (B_plus - B_minus)``
where ``D = sum over used arcs of ell[r,i,j]`` and the makespan and
imbalance auxiliaries are linked by the standard max/min linearisations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from instances.synthetic_generator import SyntheticInstance


@dataclass
class AnticipativeSolution:
    objective_value: float
    distance: float
    makespan: float
    imbalance: float
    arc_solution: dict
    completion_times: dict
    origin_to_pickup_travel: dict
    solve_time_seconds: float
    mip_gap: float
    status: str
    n_unserved_tasks: int
    obj_bound: float = 0.0


def _travel(a: np.ndarray, b: np.ndarray, v: float) -> float:
    return float(np.linalg.norm(a - b)) / v


def _start_node(r: int) -> tuple:
    return ("S", int(r))


def _end_node(r: int) -> tuple:
    return ("E", int(r))


def solve_anticipative(
    instance: SyntheticInstance,
    weights: dict,
    time_limit_seconds: int = 1800,
    mip_gap: float = 0.01,
    verbose: bool = False,
) -> AnticipativeSolution:
    R = int(instance.R)
    T = int(instance.T)
    cfg = instance.config
    Delta = float(cfg["Delta"])
    v = float(cfg["v"])

    w_dist = float(weights["w_dist"])
    w_make = float(weights["w_make"])
    w_bal = float(weights["w_bal"])

    tasks = instance.tasks
    pickups = {int(t["id"]): np.asarray(t["pickup"], dtype=np.float64) for t in tasks}
    drops = {int(t["id"]): np.asarray(t["drop"], dtype=np.float64) for t in tasks}
    durations = {int(t["id"]): float(t["duration"]) for t in tasks}
    release_times = {
        int(t["id"]): float(int(t["release_epoch"])) * Delta for t in tasks
    }
    p0 = np.asarray(instance.initial_positions, dtype=np.float64)

    env = gp.Env(empty=True)
    if not verbose:
        env.setParam("OutputFlag", 0)
    env.start()

    model = gp.Model("anticipative_mrta", env=env)
    model.setParam("TimeLimit", float(time_limit_seconds))
    model.setParam("MIPGap", float(mip_gap))

    x: Dict[tuple, gp.Var] = {}
    for r in range(R):
        s_r = _start_node(r)
        e_r = _end_node(r)
        for j in range(T):
            x[(r, s_r, j)] = model.addVar(
                vtype=GRB.BINARY, name=f"x_r{r}_S_t{j}"
            )
        x[(r, s_r, e_r)] = model.addVar(vtype=GRB.BINARY, name=f"x_r{r}_S_E")
        for i in range(T):
            for j in range(T):
                if i != j:
                    x[(r, i, j)] = model.addVar(
                        vtype=GRB.BINARY, name=f"x_r{r}_t{i}_t{j}"
                    )
        for i in range(T):
            x[(r, i, e_r)] = model.addVar(
                vtype=GRB.BINARY, name=f"x_r{r}_t{i}_E"
            )

    f_var: Dict[tuple, gp.Var] = {}
    for r in range(R):
        for j in range(T):
            f_var[(r, j)] = model.addVar(
                vtype=GRB.CONTINUOUS, lb=0.0, name=f"f_r{r}_t{j}"
            )

    tau_free = {
        r: model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"tau_free_r{r}")
        for r in range(R)
    }
    tau_busy = {
        r: model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"tau_busy_r{r}")
        for r in range(R)
    }
    M_make = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="M_make")
    B_plus = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="B_plus")
    B_minus = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="B_minus")

    served: Dict[tuple, gp.Var] = {}
    for r in range(R):
        s_r = _start_node(r)
        for j in range(T):
            served[(r, j)] = model.addVar(
                vtype=GRB.BINARY, name=f"served_r{r}_t{j}"
            )
            in_arcs = [x[(r, s_r, j)]] + [
                x[(r, i, j)] for i in range(T) if i != j
            ]
            model.addConstr(
                served[(r, j)] == gp.quicksum(in_arcs),
                name=f"def_served_r{r}_t{j}",
            )

    model.update()

    # (a) Each task served exactly once
    for j in range(T):
        model.addConstr(
            gp.quicksum(served[(r, j)] for r in range(R)) == 1,
            name=f"task_t{j}_served_once",
        )

    # (b) Flow conservation
    for r in range(R):
        e_r = _end_node(r)
        for j in range(T):
            out_arcs = [x[(r, j, e_r)]] + [
                x[(r, j, k)] for k in range(T) if k != j
            ]
            model.addConstr(
                served[(r, j)] == gp.quicksum(out_arcs),
                name=f"flow_r{r}_t{j}",
            )

    # (c) Each robot starts and ends exactly once
    for r in range(R):
        s_r = _start_node(r)
        e_r = _end_node(r)
        out_start = [x[(r, s_r, j)] for j in range(T)] + [x[(r, s_r, e_r)]]
        model.addConstr(gp.quicksum(out_start) == 1, name=f"start_r{r}")
        in_end = [x[(r, i, e_r)] for i in range(T)] + [x[(r, s_r, e_r)]]
        model.addConstr(gp.quicksum(in_end) == 1, name=f"end_r{r}")

    # (c.2) Chain-start bound: redundant given (c), but the explicit task-arc-only bound strengthens the LP relaxation for the presolver.
    for r in range(R):
        s_r = _start_node(r)
        model.addConstr(
            gp.quicksum(x[(r, s_r, j)] for j in range(T)) <= 1,
            name=f"chain_start_r{r}",
        )

    # (d) Completion time consistency via indicators
    for r in range(R):
        s_r = _start_node(r)
        p0r = p0[r]
        for j in range(T):
            tp_sj = _travel(p0r, pickups[j], v)
            tpd_j = _travel(pickups[j], drops[j], v)
            delta_sj = tp_sj + tpd_j + durations[j]
            model.addGenConstrIndicator(
                x[(r, s_r, j)],
                1,
                f_var[(r, j)],
                GRB.GREATER_EQUAL,
                delta_sj,
                name=f"comp_start_r{r}_t{j}",
            )
            for i in range(T):
                if i != j:
                    tp_ij = _travel(drops[i], pickups[j], v)
                    delta_ij = tp_ij + tpd_j + durations[j]
                    model.addGenConstrIndicator(
                        x[(r, i, j)],
                        1,
                        f_var[(r, j)] - f_var[(r, i)],
                        GRB.GREATER_EQUAL,
                        delta_ij,
                        name=f"comp_r{r}_t{i}_t{j}",
                    )

    # (e) Release time
    for r in range(R):
        for j in range(T):
            tpd_j = _travel(pickups[j], drops[j], v)
            rhs = release_times[j] + tpd_j + durations[j]
            model.addGenConstrIndicator(
                served[(r, j)],
                1,
                f_var[(r, j)],
                GRB.GREATER_EQUAL,
                rhs,
                name=f"release_r{r}_t{j}",
            )

    # (f) Robot finish time
    for r in range(R):
        for j in range(T):
            model.addGenConstrIndicator(
                served[(r, j)],
                1,
                tau_free[r] - f_var[(r, j)],
                GRB.GREATER_EQUAL,
                0.0,
                name=f"tfree_r{r}_t{j}",
            )

    # (g) Busy time accumulation
    for r in range(R):
        s_r = _start_node(r)
        p0r = p0[r]
        terms = []
        for j in range(T):
            sc = (
                _travel(p0r, pickups[j], v)
                + _travel(pickups[j], drops[j], v)
                + durations[j]
            )
            terms.append(sc * x[(r, s_r, j)])
        for i in range(T):
            for j in range(T):
                if i != j:
                    sc = (
                        _travel(drops[i], pickups[j], v)
                        + _travel(pickups[j], drops[j], v)
                        + durations[j]
                    )
                    terms.append(sc * x[(r, i, j)])
        model.addConstr(tau_busy[r] == gp.quicksum(terms), name=f"busy_r{r}")

    # (h) Makespan / imbalance auxiliaries
    for r in range(R):
        model.addConstr(M_make >= tau_free[r], name=f"M_geq_tfree_r{r}")
        model.addConstr(B_plus >= tau_busy[r], name=f"Bp_geq_tbusy_r{r}")
        model.addConstr(B_minus <= tau_busy[r], name=f"Bm_leq_tbusy_r{r}")

    dist_terms = []
    for r in range(R):
        s_r = _start_node(r)
        p0r = p0[r]
        for j in range(T):
            ell = _travel(p0r, pickups[j], v) + _travel(pickups[j], drops[j], v)
            dist_terms.append(ell * x[(r, s_r, j)])
        for i in range(T):
            for j in range(T):
                if i != j:
                    ell = _travel(drops[i], pickups[j], v) + _travel(
                        pickups[j], drops[j], v
                    )
                    dist_terms.append(ell * x[(r, i, j)])

    D_expr = gp.quicksum(dist_terms)
    # Tiebreak coefficient 1e-6 matches the existing cache and is verified to pin f_var to its lower bound without altering the primary (D, M, imbalance) trade-off.
    tiebreak_expr = gp.quicksum(
        f_var[(r, j)] for r in range(R) for j in range(T)
    )
    model.setObjective(
        w_dist * D_expr
        + w_make * M_make
        + w_bal * (B_plus - B_minus)
        + 1e-6 * tiebreak_expr,
        GRB.MINIMIZE,
    )

    t0 = time.time()
    model.optimize()
    solve_time = time.time() - t0

    status_map = {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        GRB.UNBOUNDED: "unbounded",
        GRB.SUBOPTIMAL: "suboptimal",
    }
    status = status_map.get(model.Status, f"status_{model.Status}")

    if status == "infeasible":
        raise RuntimeError(
            f"Anticipative MILP infeasible for instance seed={instance.seed}"
        )
    if model.SolCount == 0:
        raise RuntimeError(
            f"Anticipative MILP returned no feasible solution "
            f"(status={status}, solve_time={solve_time:.2f}s)"
        )

    arc_solution: Dict[tuple, int] = {}
    for key, var in x.items():
        if var.X > 0.5:
            arc_solution[key] = 1

    served_set = set()
    completion_times: Dict[tuple, float] = {}
    for (r, j), var in served.items():
        if var.X > 0.5:
            completion_times[(r, j)] = float(f_var[(r, j)].X)
            served_set.add((r, j))

    # Origin is p0[r] if the incoming arc starts at S_r, else the predecessor task's drop; this is the tp_sj/tp_ij term needed downstream to back out start-of-travel from completion time.
    origin_to_pickup_travel: Dict[tuple, float] = {}
    for (r, i, j) in arc_solution:
        s_r = _start_node(r)
        e_r = _end_node(r)
        if j == e_r:
            continue
        if i == s_r:
            origin_to_pickup_travel[(r, j)] = _travel(p0[r], pickups[j], v)
        else:
            origin_to_pickup_travel[(r, j)] = _travel(drops[i], pickups[j], v)

    D_val = 0.0
    for (r, i, j) in arc_solution:
        s_r = _start_node(r)
        e_r = _end_node(r)
        if j == e_r:
            continue
        if i == s_r:
            ell = _travel(p0[r], pickups[j], v) + _travel(
                pickups[j], drops[j], v
            )
        else:
            ell = _travel(drops[i], pickups[j], v) + _travel(
                pickups[j], drops[j], v
            )
        D_val += ell

    M_val = float(M_make.X)
    B_val = float(B_plus.X) - float(B_minus.X)
    obj_val = float(model.ObjVal)
    mip_gap_final = float(model.MIPGap) if model.SolCount > 0 else float("inf")
    obj_bound = float(model.ObjBound)
    n_unserved = T - len(served_set)

    return AnticipativeSolution(
        objective_value=obj_val,
        distance=D_val,
        makespan=M_val,
        imbalance=B_val,
        arc_solution=arc_solution,
        completion_times=completion_times,
        origin_to_pickup_travel=origin_to_pickup_travel,
        solve_time_seconds=solve_time,
        mip_gap=mip_gap_final,
        status=status,
        n_unserved_tasks=n_unserved,
        obj_bound=obj_bound,
    )


def extract_per_epoch_decisions(
    solution: AnticipativeSolution,
    instance: SyntheticInstance,
) -> List[List[Tuple[int, int]]]:
    Delta = float(instance.config["Delta"])
    v = float(instance.config["v"])
    tasks_by_id = {int(t["id"]): t for t in instance.tasks}

    per_epoch: Dict[int, List[Tuple[int, int]]] = {}
    for (r, j), f_rj in solution.completion_times.items():
        task = tasks_by_id[j]
        d_j = float(task["duration"])
        release_epoch_j = int(task["release_epoch"])
        # The simulator commits a robot at start-of-travel (f_rj - tp_rj - tpd_j - d_j), not at arrival-at-drop, so back that out to match its convention.
        tp_rj = float(solution.origin_to_pickup_travel[(r, j)])
        pickup = np.asarray(task["pickup"], dtype=np.float64)
        drop = np.asarray(task["drop"], dtype=np.float64)
        tpd_j = _travel(pickup, drop, v)
        start_time = f_rj - tp_rj - tpd_j - d_j
        epoch = max(int(np.floor(start_time / Delta)), release_epoch_j)
        if epoch < 0:
            epoch = 0
        per_epoch.setdefault(epoch, []).append((int(r), int(j)))

    if not per_epoch:
        return []
    max_epoch = max(per_epoch.keys())
    return [per_epoch.get(t, []) for t in range(max_epoch + 1)]
