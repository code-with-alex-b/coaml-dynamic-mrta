"""Best-checkpoint tracking tests for method two.

Exercises the in-loop validation hook and best-checkpoint logic without the
real validation cache, by scripting the gap-closure values the trainer sees.
Verifies that the best checkpoint is a separate file, updates only on
improvement, never overwrites the periodic checkpoint, and that the best
metric and step survive in both the best and periodic checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

torch.set_num_threads(1)

from instances.synthetic_generator import generate_instance
from training import method_two_trainer as M
from training.method_two_trainer import MethodTwoConfig, best_checkpoint_path


def test_best_checkpoint_path_naming():
    assert best_checkpoint_path("checkpoints/foo.pt") == "checkpoints/foo_best.pt"
    assert best_checkpoint_path("a/b/c") == "a/b/c_best.pt"


def _write_tiny_cache(cache_dir: Path, seeds):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        inst = generate_instance(s)
        rec = {"seed": int(s), "instance": inst.to_dict()}
        with (cache_dir / f"seed{int(s):06d}.json").open("w") as f:
            json.dump(rec, f)


def test_best_checkpoint_tracking(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    _write_tiny_cache(cache_dir, [900, 901, 902, 903])

    # Force CPU for speed and determinism.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    # Script the per-step validation gap closure: best is 0.6 at step 4.
    scripted = iter([0.30, 0.50, 0.40, 0.60, 0.55])

    def fake_eval(scorer, val_records, weights):
        gap = next(scripted)
        return gap, 1.0, {"modes": {"hard": {}}}

    monkeypatch.setattr(M, "evaluate_val_gap_closure", fake_eval)

    ckpt = tmp_path / "run.pt"
    config = MethodTwoConfig(
        cache_dir=str(cache_dir),
        checkpoint_path=str(ckpt),
        baseline_mode=M.BASELINE_RLOO,
        rloo_k=2,
        batch_size=2,
        K_sink=10,
        epsilon_initial=1.0,
        epsilon_terminal=1.0,
        log_every_steps=0,
        checkpoint_every_steps=2,
        num_workers=1,
        eval_every_steps=1,
        val_cache_dir=str(cache_dir),  # records loaded but unused by fake_eval
        num_val_instances=4,
    )

    torch.manual_seed(0)
    M.train(config, num_steps=5)

    best_path = Path(best_checkpoint_path(str(ckpt)))

    assert best_path.exists()
    assert ckpt.exists()
    assert best_path != ckpt

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    assert best["best_gap"] == 0.60
    assert best["best_step"] == 4
    assert "model_state_dict" in best
    # The best checkpoint is a model snapshot, not the optimiser/history blob.
    assert "optimizer_state_dict" not in best

    periodic = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert periodic["best_gap"] == 0.60
    assert periodic["best_step"] == 4
    assert "optimizer_state_dict" in periodic
    # Periodic checkpoint reflects the last loop step, not the best step.
    assert periodic["step"] == 5


def test_no_best_when_eval_disabled(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    _write_tiny_cache(cache_dir, [910, 911, 912, 913])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    # If the trainer evaluates at all here, that is a failure (eval disabled).
    def boom(*a, **k):
        raise AssertionError("evaluate_val_gap_closure called with eval disabled")

    monkeypatch.setattr(M, "evaluate_val_gap_closure", boom)

    ckpt = tmp_path / "run.pt"
    config = MethodTwoConfig(
        cache_dir=str(cache_dir),
        checkpoint_path=str(ckpt),
        baseline_mode=M.BASELINE_RLOO,
        rloo_k=2,
        batch_size=2,
        K_sink=10,
        epsilon_initial=1.0,
        epsilon_terminal=1.0,
        log_every_steps=0,
        checkpoint_every_steps=2,
        num_workers=1,
        eval_every_steps=0,  # disabled
    )
    torch.manual_seed(0)
    M.train(config, num_steps=3)

    assert not Path(best_checkpoint_path(str(ckpt))).exists()
    assert ckpt.exists()
