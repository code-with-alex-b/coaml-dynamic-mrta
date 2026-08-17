"""Smoke tests for the method-one evaluation harness on tiny configs."""

from __future__ import annotations

import json

import numpy as np
import torch

from instances.synthetic_generator import generate_instance
from baselines.bipartite_policies import run_greedy_policy
from scoring.gnn_scorer import GNNScorer
from training.expert_dataset_generator import _serialize_instance
from training.il_trainer import save_checkpoint
from evaluation.method_one_evaluator import (
    DEFAULT_WEIGHTS,
    evaluate_one_instance,
    evaluate_split,
    gap_closure,
    load_scorer_from_checkpoint,
    rollout_policy,
    summarize,
)


SMALL_CONFIG = {"R": 3, "T": 4, "H": 5}


class _NeverCommitScorer(GNNScorer):
    """Scorer that drives the policy to never commit a task.

    Forces the top-left robot-task scores far below the zero-cost idle and
    pending diagonals, so Hungarian decoding always parks robots idle and
    tasks pending. The rollout then exhausts the wall-clock cap with tasks
    unserved, a guaranteed failure."""

    def forward(self, state, instance):
        theta = super().forward(state, instance).clone()
        theta[: instance.R, : instance.T] = -1e3
        return theta


def _greedy_as_record(seed, config=SMALL_CONFIG):
    """Build a cache-shaped record using greedy as a stand-in expert and a
    MILP surrogate set strictly below the greedy cost (so gap closure vs MILP
    has a positive denominator). Gurobi-free."""
    inst = generate_instance(seed, config=config)
    sim = run_greedy_policy(inst)
    cb = sim.compute_cost(DEFAULT_WEIGHTS)

    commits = sim.trajectory.commitments
    max_epoch = max((c["epoch"] for c in commits), default=-1)
    decisions = [[] for _ in range(max_epoch + 1)]
    for c in commits:
        decisions[c["epoch"]].append([int(c["robot_id"]), int(c["task_id"])])

    record = {
        "seed": int(seed),
        "R": int(inst.R),
        "T": int(inst.T),
        "instance": _serialize_instance(inst),
        "milp_solution": {
            "objective_value": cb.combined * 0.5,
            "distance": cb.distance * 0.5,
            "makespan": cb.makespan * 0.5,
            "imbalance": cb.imbalance * 0.5,
            "status": "optimal",
            "mip_gap": 0.0,
        },
        "expert_decisions": decisions,
    }
    return record, inst


def _write_records(cache_dir, records):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        with (cache_dir / f"seed{rec['seed']}.json").open("w") as f:
            json.dump(rec, f)


def test_gap_closure_handles_zero_and_normal_denominator():
    assert gap_closure(10.0, 6.0, 2.0) == (10.0 - 6.0) / (10.0 - 2.0)
    # Denominator exactly zero -> None.
    assert gap_closure(5.0, 4.0, 5.0) is None
    # Denominator below tolerance -> None.
    assert gap_closure(5.0, 4.0, 5.0 - 1e-12) is None


def test_rollout_policy_terminates_and_costs_finite():
    inst = generate_instance(70000, config=SMALL_CONFIG)
    torch.manual_seed(0)
    scorer = GNNScorer()
    sim = rollout_policy(scorer, inst)
    assert sim.is_terminal
    cb = sim.compute_cost(DEFAULT_WEIGHTS)
    assert np.isfinite(cb.combined)
    assert np.isfinite(cb.distance)
    assert np.isfinite(cb.makespan)
    assert np.isfinite(cb.imbalance)


def test_evaluate_one_instance_returns_all_keys_and_handles_none():
    record, _ = _greedy_as_record(70001)
    torch.manual_seed(0)
    scorer = GNNScorer()
    result = evaluate_one_instance(scorer, record)

    expected_keys = {
        "seed",
        "policy_cost", "policy_distance", "policy_makespan", "policy_imbalance",
        "greedy_cost", "greedy_distance", "greedy_makespan", "greedy_imbalance",
        "milp_cost", "milp_distance", "milp_makespan", "milp_imbalance",
        "replayed_expert_cost", "replayed_expert_distance",
        "replayed_expert_makespan", "replayed_expert_imbalance",
        "gap_closure_vs_milp", "gap_closure_vs_expert", "mip_gap",
        "policy_failed", "policy_simulator_failure", "greedy_simulator_failure",
    }
    assert expected_keys.issubset(result.keys())

    for src in ("greedy", "milp", "replayed_expert"):
        assert np.isfinite(result[f"{src}_cost"])

    if result["policy_failed"]:
        assert result["policy_cost"] is None
        assert result["policy_distance"] is None
        assert result["gap_closure_vs_milp"] is None
    else:
        assert np.isfinite(result["policy_cost"])
        # MILP surrogate is strictly below greedy -> finite gap closure.
        assert result["gap_closure_vs_milp"] is not None
        assert np.isfinite(result["gap_closure_vs_milp"])

    # Expert is the replayed greedy decisions, so identical cost gives a zero denominator -> None.
    assert result["replayed_expert_cost"] == result["greedy_cost"]
    assert result["gap_closure_vs_expert"] is None


def test_evaluate_split_summary_structure(tmp_path):
    cache_dir = tmp_path / "test"
    records = [_greedy_as_record(seed)[0] for seed in (70010, 70011, 70012)]
    _write_records(cache_dir, records)

    torch.manual_seed(0)
    scorer = GNNScorer()
    out = evaluate_split(scorer, cache_dir=cache_dir)

    assert len(out["per_instance"]) == 3
    summary = out["summary"]
    assert summary["n_instances_evaluated"] == 3
    assert (
        summary["n_policy_failures"] + summary["n_instances_succeeded"] == 3
    )

    for src in ("greedy", "milp", "replayed_expert"):
        assert summary["combined_cost"][src]["mean"] is not None
        assert summary["combined_cost"][src]["n"] == 3

    assert (
        summary["combined_cost"]["policy"]["n"]
        == summary["n_instances_succeeded"]
    )

    for key in ("gap_closure_vs_milp", "gap_closure_vs_expert"):
        block = summary[key]
        assert "count_positive" in block
        assert "count_above_0_30" in block
        assert "n" in block

    for term in ("distance", "makespan", "imbalance"):
        tblock = summary["per_term"][term]
        for src in ("policy", "greedy", "milp", "replayed_expert"):
            assert src in tblock
        assert "gap_closure_vs_milp" in tblock
        assert "gap_closure_vs_expert" in tblock


def test_failed_rollout_reports_none_and_flag():
    record, _ = _greedy_as_record(70030)
    torch.manual_seed(0)
    scorer = _NeverCommitScorer()
    result = evaluate_one_instance(scorer, record)

    # A non-serving rollout is a hard failure, not a low cost.
    assert result["policy_failed"] is True
    assert result["policy_cost"] is None
    assert result["policy_distance"] is None
    assert result["policy_makespan"] is None
    assert result["policy_imbalance"] is None
    assert result["gap_closure_vs_milp"] is None
    assert result["gap_closure_vs_expert"] is None
    assert np.isfinite(result["greedy_cost"])
    assert np.isfinite(result["milp_cost"])

    summary = summarize([result])
    assert summary["n_instances_evaluated"] == 1
    assert summary["n_policy_failures"] == 1
    assert summary["n_instances_succeeded"] == 0
    # No succeeded rollouts -> empty policy and gap-closure statistics.
    assert summary["combined_cost"]["policy"]["n"] == 0
    assert summary["combined_cost"]["policy"]["mean"] is None
    assert summary["gap_closure_vs_milp"]["n"] == 0
    assert summary["combined_cost"]["greedy"]["n"] == 1


def test_load_scorer_from_checkpoint_round_trips(tmp_path):
    torch.manual_seed(0)
    model = GNNScorer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, step=5, path=ckpt)

    loaded = load_scorer_from_checkpoint(ckpt)
    for (na, pa), (nb, pb) in zip(
        model.named_parameters(), loaded.named_parameters()
    ):
        assert na == nb
        assert torch.allclose(pa, pb)

    inst = generate_instance(70020, config=SMALL_CONFIG)
    sim = rollout_policy(loaded, inst)
    assert sim.is_terminal
    assert np.isfinite(sim.compute_cost(DEFAULT_WEIGHTS).combined)
