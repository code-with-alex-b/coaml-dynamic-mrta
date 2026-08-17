"""Train a single hyperparameter-sweep configuration.

Loads ``sweep/configs.json``, merges the shared block with one config entry
(selected by ``--config-id``), and calls ``method_two_trainer.train`` with the
right ``MethodTwoConfig``.

Resume behaviour: if the config's sweep checkpoint already exists on disk, the
run resumes from it (the trainer's resume block restores model, optimizer,
baseline, history, and start_step). Otherwise it warm-starts from the shared
method-one checkpoint. Periodic checkpointing in the trainer guarantees the
last saved checkpoint is a valid resume point even after a crash.

Run with PYTHONPATH pointed at src/, e.g.::

    PYTHONPATH="$PWD/src" python sweep/train_one.py --config-id A
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from training.method_two_trainer import MethodTwoConfig, train


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_PATH = Path(__file__).resolve().parent / "configs.json"


def load_sweep():
    """Return ``(shared, configs_by_id)`` from ``sweep/configs.json``."""
    with CONFIGS_PATH.open("r") as f:
        spec = json.load(f)
    shared = spec["shared"]
    configs_by_id = {c["id"]: c for c in spec["configs"]}
    return shared, configs_by_id


def build_config(
    config_id: str,
    num_workers: int = 1,
    max_steps=None,
    checkpoint_path=None,
    eval_every=None,
    lr=None,
    grad_clip=None,
    batch_size=None,
    epsilon_end=None,
    epsilon_anneal_steps=None,
    no_warm_start=False,
    warm_start_checkpoint=None,
    rng_base_seed=None,
    rloo_k=None,
    cache_dir=None,
    val_cache_dir=None,
    checkpoint_every=None,
):
    """Build the ``(MethodTwoConfig, num_steps)`` pair for one sweep config.

    ``num_workers`` selects sequential (1) or parallel (>=2) rollout
    generation. ``max_steps`` overrides the shared step count when given, which
    keeps smoke tests short. ``checkpoint_path`` overrides the default sweep
    checkpoint location, which lets a smoke run write to a private path instead
    of the shared sweep checkpoint. ``eval_every`` overrides the in-loop
    validation cadence; ``None`` uses the shared checkpoint cadence and ``0``
    disables in-loop validation and best-checkpoint tracking.

    ``lr``, ``grad_clip`` and ``epsilon_end`` override the learning rate, the
    gradient clip norm and the terminal annealing epsilon respectively. The
    resolution order is the CLI value if given, then a per-config entry value
    if present, then the shared default. ``epsilon_end`` controls the terminal
    epsilon only; the initial epsilon stays the shared value, so passing it
    turns the constant-epsilon schedule into an annealed one. ``batch_size``
    overrides the per-config ``batch`` when given, otherwise the per-config
    entry value is used.

    ``warm_start_checkpoint`` overrides the warm-start source. When ``None`` the
    shared ``warm_start`` from configs.json (the default
    ``il_method_one_v3.pt``) is used; when given, that path is used instead. It
    has no effect when an existing sweep checkpoint is resumed, or when
    ``no_warm_start`` is set.

    ``rng_base_seed`` overrides the deterministic per-rollout seed base. When
    ``None`` the shared ``rng_base_seed`` from configs.json is used if present,
    otherwise the ``MethodTwoConfig`` default is kept.
    """
    shared, configs_by_id = load_sweep()
    if config_id not in configs_by_id:
        raise SystemExit(
            f"Unknown config-id {config_id!r}. "
            f"Known ids: {sorted(configs_by_id)}"
        )
    entry = configs_by_id[config_id]

    # Resolution order for overridable scalars: CLI, then per-config entry, then shared default.
    resolved_lr = lr if lr is not None else entry.get(
        "lr", shared["learning_rate"]
    )
    resolved_clip = grad_clip if grad_clip is not None else entry.get(
        "grad_clip", shared["gradient_clip_norm"]
    )
    resolved_eps_end = epsilon_end if epsilon_end is not None else entry.get(
        "epsilon_end", shared["epsilon"]
    )
    # Pinning the anneal span decouples the epsilon schedule from num_steps; without it a resume with a larger step budget stretches the schedule and rewinds epsilon mid-training (observed in sweep_G_tuned_continued).
    resolved_anneal_steps = (
        epsilon_anneal_steps if epsilon_anneal_steps is not None
        else entry.get("epsilon_anneal_steps", shared.get("epsilon_anneal_steps"))
    )
    resolved_batch = batch_size if batch_size is not None else entry["batch"]
    # Overridable so a K-against-batch swap at fixed B*K can anchor on an existing config entry (which pins lr, clip, epsilon and warm start).
    resolved_rloo_k = rloo_k if rloo_k is not None else entry["k"]
    # CLI value if given, else the shared configs.json default, else None (keeps the MethodTwoConfig default).
    resolved_rng_base_seed = (
        rng_base_seed if rng_base_seed is not None
        else shared.get("rng_base_seed")
    )

    if checkpoint_path is None:
        checkpoint_path = (
            REPO_ROOT / "checkpoints" / f"sweep_bk_grid_{config_id}.pt"
        )
    else:
        # A CLI override arrives as a str; normalise so the resume check below (Path.exists) works regardless.
        checkpoint_path = Path(checkpoint_path)

    config = MethodTwoConfig(
        cache_dir=(str(cache_dir) if cache_dir is not None
                   else str(REPO_ROOT / shared["train_cache"])),
        checkpoint_path=str(checkpoint_path),
        checkpoint_every_steps=int(checkpoint_every if checkpoint_every is not None
                                   else shared["checkpoint_every"]),
        baseline_mode=shared["baseline_mode"],
        rloo_k=int(resolved_rloo_k),
        batch_size=int(resolved_batch),
        # Initial epsilon stays the shared value; the terminal epsilon is the resolved override, equal by default (constant-epsilon schedule).
        epsilon_initial=float(shared["epsilon"]),
        epsilon_terminal=float(resolved_eps_end),
        epsilon_anneal_steps=(
            int(resolved_anneal_steps) if resolved_anneal_steps is not None
            else None
        ),
        learning_rate=float(resolved_lr),
        gradient_clip_norm=float(resolved_clip),
        # Eval and checkpoint messages are gated separately (eval_every_steps, checkpoint_every_steps), so they print regardless of this cadence.
        log_every_steps=10,
        num_workers=int(num_workers),
        # Default cadence is the periodic checkpoint cadence; selection is by hard gap closure vs the MILP bound on the shared validation cache.
        eval_every_steps=(
            int(checkpoint_every if checkpoint_every is not None
                else shared["checkpoint_every"]) if eval_every is None
            else int(eval_every)
        ),
        val_cache_dir=(str(val_cache_dir) if val_cache_dir is not None
                       else str(REPO_ROOT / shared["val_cache"])),
        num_val_instances=int(shared["eval_instances"]),
    )

    # Only set when a value resolved, so the MethodTwoConfig default is preserved when neither the CLI nor configs.json supplies one.
    if resolved_rng_base_seed is not None:
        config.rng_base_seed = int(resolved_rng_base_seed)

    # Step count resolution: CLI max_steps, then per-config entry, then shared.
    if max_steps is not None:
        num_steps = int(max_steps)
    else:
        num_steps = int(entry.get("steps", shared["steps"]))

    if checkpoint_path.exists():
        config.resume_from = str(checkpoint_path)
        print(
            f"Existing checkpoint found at {checkpoint_path}; resuming.",
            flush=True,
        )
    elif no_warm_start:
        # warm_start_from stays None so train() builds a fresh GNNScorer and skips loading any checkpoint.
        config.warm_start_from = None
        print(
            f"No checkpoint at {checkpoint_path}; --no-warm-start set, "
            f"training from a randomly initialised model.",
            flush=True,
        )
    else:
        if warm_start_checkpoint is not None:
            config.warm_start_from = str(warm_start_checkpoint)
        else:
            # Per-config warm_start (if present) beats the shared default, so a config entry can pin its own warm-start checkpoint.
            config.warm_start_from = str(
                REPO_ROOT / entry.get("warm_start", shared["warm_start"])
            )
        print(
            f"No checkpoint at {checkpoint_path}; warm-starting from "
            f"{config.warm_start_from}.",
            flush=True,
        )

    return config, num_steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one hyperparameter-sweep config (bk_grid)."
    )
    parser.add_argument(
        "--config-id", "--config_id", dest="config_id", type=str, required=True
    )
    parser.add_argument(
        "--num-workers", "--num_workers", dest="num_workers", type=int, default=1
    )
    parser.add_argument(
        "--max-steps", "--max_steps", dest="max_steps", type=int, default=None
    )
    parser.add_argument(
        "--checkpoint-path", "--checkpoint_path", "--checkpoint",
        dest="checkpoint_path", type=str, default=None,
        help="Override the sweep checkpoint path (e.g. for smoke runs).",
    )
    parser.add_argument(
        "--eval-every", "--eval_every", dest="eval_every", type=int, default=None,
        help="In-loop validation cadence in steps. 0 disables best tracking; "
        "default uses the shared checkpoint cadence.",
    )
    parser.add_argument(
        "--lr", dest="lr", type=float, default=None,
        help="Override the learning rate (else per-config or configs.json).",
    )
    parser.add_argument(
        "--grad-clip", "--grad_clip", dest="grad_clip", type=float, default=None,
        help="Override the gradient clip norm (else configs.json).",
    )
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=None,
        help="Override the per-config batch size (else configs.json).",
    )
    parser.add_argument(
        "--epsilon-end", "--epsilon_end", dest="epsilon_end", type=float,
        default=None,
        help="Override the terminal annealing epsilon (else configs.json).",
    )
    parser.add_argument(
        "--epsilon-anneal-steps", "--epsilon_anneal_steps",
        dest="epsilon_anneal_steps", type=int, default=None,
        help="Pin the epsilon anneal span in steps so the schedule is "
        "independent of the total step budget (else configs.json, else the "
        "span floats with num_steps).",
    )
    parser.add_argument(
        "--no-warm-start", "--no_warm_start", dest="no_warm_start",
        action="store_true", default=False,
        help="Skip the warm-start checkpoint and train from a randomly "
        "initialised model (ignored when an existing checkpoint is resumed).",
    )
    parser.add_argument(
        "--warm-start-checkpoint", "--warm_start_checkpoint",
        dest="warm_start_checkpoint", type=str, default=None,
        help="Override the warm-start checkpoint path. Default uses the shared "
        "warm_start from configs.json (il_method_one_v3.pt). Ignored when an "
        "existing checkpoint is resumed or --no-warm-start is set.",
    )
    parser.add_argument(
        "--rloo-k", "--rloo_k", "--k", dest="rloo_k", type=int, default=None,
        help="Override the RLOO group size K (else the per-config k). Use with "
        "--batch-size to move along a fixed B*K contour.",
    )
    # Without these overrides the entry point can only read the production caches named in configs.json, unusable from a demo notebook or reduced-scale run.
    parser.add_argument(
        "--cache-dir", dest="cache_dir", type=str, default=None,
        help="Override the training cache from configs.json shared.train_cache.")
    parser.add_argument(
        "--val-cache-dir", dest="val_cache_dir", type=str, default=None,
        help="Override the validation cache from configs.json shared.val_cache.")
    parser.add_argument(
        "--checkpoint-every", dest="checkpoint_every", type=int, default=None,
        help="Override the periodic checkpoint cadence from configs.json.")
    parser.add_argument(
        "--rng-base-seed", "--rng_base_seed", dest="rng_base_seed", type=int,
        default=None,
        help="Override the per-rollout RNG base seed (else configs.json or the "
        "MethodTwoConfig default).",
    )
    args = parser.parse_args()

    config, num_steps = build_config(
        args.config_id,
        num_workers=args.num_workers,
        max_steps=args.max_steps,
        lr=args.lr,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        epsilon_end=args.epsilon_end,
        epsilon_anneal_steps=args.epsilon_anneal_steps,
        checkpoint_path=args.checkpoint_path,
        eval_every=args.eval_every,
        no_warm_start=args.no_warm_start,
        warm_start_checkpoint=args.warm_start_checkpoint,
        rng_base_seed=args.rng_base_seed,
        rloo_k=args.rloo_k,
        cache_dir=args.cache_dir,
        val_cache_dir=args.val_cache_dir,
        checkpoint_every=args.checkpoint_every,
    )

    print(
        f"=== Sweep config {args.config_id}: "
        f"rloo_k={config.rloo_k} batch_size={config.batch_size} "
        f"steps={num_steps} baseline={config.baseline_mode} "
        f"epsilon={config.epsilon_initial}->{config.epsilon_terminal} "
        f"lr={config.learning_rate} grad_clip={config.gradient_clip_norm} "
        f"num_workers={config.num_workers} ===",
        flush=True,
    )

    try:
        train(config, num_steps)
    except Exception:
        traceback.print_exc()
        print(
            f"Config {args.config_id} FAILED. The last periodic checkpoint "
            f"(if any) at {config.checkpoint_path} remains valid for resume.",
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
