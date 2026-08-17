"""Method one imitation learning training loop.

Samples ``(state, expert_decision)`` pairs from the cached dataset, scores each state
with the GNN head, and takes an Adam step on the perturbed Fenchel-Young gradient. Only
epochs where the expert started a task carry supervision; drain epochs are skipped. State
comes from the cache rather than from replaying the simulator.

The target is the expert permutation ``P_star``, shaped like ``Theta``, with assignments
top-left, idle robots and pending tasks on the two diagonals, and the bottom-right block
absorbing assigned task rows into the freed robot columns. That last block is what makes
``P_star`` a true permutation, since each assigned robot vacates its idle column and the
task row must claim a column somewhere. Theta scores those cells at zero, so the pairing
is a valid optimum of the linear oracle.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch.nn.utils import clip_grad_norm_

from instances.synthetic_generator import SyntheticInstance
from losses.fenchel_young import (
    EPSILON_DEFAULT,
    M_SAMPLES_DEFAULT,
    fenchel_young_loss,
)
from scoring.gnn_scorer import HIDDEN_DIM, GNNScorer
from scoring.linear_scorer import LinearScorer
from simulator.dynamic_simulator import SimulatorState


DEFAULT_LR = 1e-3
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_CACHE_DIR = "cache/training_set_il_v3"
DEFAULT_SPLIT = "train"
DEFAULT_CHECKPOINT_PATH = "checkpoints/il_method_one.pt"
DEFAULT_BATCH_SIZE = 16
DEFAULT_LOG_EVERY_STEPS = 50
# TrainingConfig.scorer_type options; "linear" swaps in the affine LinearScorer diagnostic with no other change to loss, CO-layer, or cache loading.
SCORER_TYPES = ("gnn", "linear")


@dataclass
class TrainingConfig:
    """Hyperparameters for the method-one IL training loop.

    ``num_steps`` is supplied separately to ``train`` so a single config can
    drive both a smoke run and a full run.
    """

    cache_dir: str = DEFAULT_CACHE_DIR
    split: Optional[str] = DEFAULT_SPLIT
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    # "constant" leaves Adam's lr fixed (original behaviour, no scheduler, bit-identical trajectory); "cosine" anneals it to zero via CosineAnnealingLR.
    lr_schedule: str = "constant"
    M: int = M_SAMPLES_DEFAULT
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    seed: int = 0
    log_every_steps: int = DEFAULT_LOG_EVERY_STEPS
    # Epsilon decays log-linearly from epsilon_initial to epsilon_terminal (methodology 3.4.6) so train-time decode converges to the hard-Hungarian inference boundary.
    epsilon_initial: float = 1.0
    epsilon_terminal: float = 0.01
    # None anneals over the full num_steps; an int anneals over that many steps then holds at epsilon_terminal.
    epsilon_anneal_steps: Optional[int] = None
    # Crash-safety checkpoint every N steps (0 disables); overwrites checkpoint_path and never advances the RNG, so it doesn't change the training trajectory.
    checkpoint_every_steps: int = 0
    # Saves to `{checkpoint_path stem}_best.pt` whenever the tracked loss (best_loss_window-smoothed, 1 = raw) hits a new minimum; never advances the RNG.
    save_best: bool = True
    best_loss_window: int = 1
    # When val_cache_dir is set and eval_every_steps > 0, best checkpoint is selected by perturbed gap-vs-MILP (same metric as sweep/evaluate_sweep) instead of loss; RNG state is saved/restored so training is unaffected.
    eval_every_steps: int = 0
    val_cache_dir: Optional[str] = None
    # Checkpoint only eligible as best if perturbed served_all_rate meets this floor; guards against a high-gap, near-zero-serve-all checkpoint winning. No effect on loss-based tracking.
    best_serve_all_floor: float = 0.95
    # Runtime-only scalar added to the valid robot-task block of GNNScorer's augmented matrix (gnn_scorer.py); default 0.0 is bit-identical to unbiased, not saved in the checkpoint.
    commit_bias: float = 0.0
    # Runtime-only Path B mask policy for GNNScorer; True drops robot-availability so busy robots keep scores and can be committed queue-ahead. Not saved in the checkpoint.
    use_queue_ahead_mask: bool = False
    # "linear" swaps in the affine LinearScorer diagnostic (scoring/linear_scorer.py); nothing else changes behaviour based on this flag beyond which class train() instantiates.
    scorer_type: str = "gnn"
    # GNNScorer capacity; defaults (64, 2 layers) reproduce the il_method_one_v4 scorer exactly. Unused by LinearScorer, which is affine.
    hidden_dim: int = HIDDEN_DIM
    num_layers: int = 2
    # One row per in-loop validation pass (None disables); opened in append mode and flushed per row so a run that dies mid-sweep leaves a valid file.
    eval_csv_path: Optional[str] = None
    # Label for the eval CSV's run column; falls back to the output checkpoint stem when unset.
    run_tag: Optional[str] = None
    # Resume from a rolling checkpoint (checkpoint_every_steps); see train() for what is and isn't reproduced.
    resume_from: Optional[str] = None


# Matched to sweep/evaluate_sweep.py + configs.json so the in-loop perturbed gap is directly comparable to the sweep's reported gap.
EVAL_WEIGHTS = {"w_dist": 0.0637, "w_make": 0.2398, "w_bal": 0.6965}
EVAL_EPSILON = 1.0
EVAL_SEED = 42


@dataclass
class ILExample:
    """One supervised training pair drawn from a commit epoch."""

    seed: int
    epoch: int
    instance: SyntheticInstance
    state: SimulatorState
    decision: List[Tuple[int, int]]


def reconstruct_state(state_record: dict) -> SimulatorState:
    """Rebuild a ``SimulatorState`` from a cached trajectory record.

    Thin wrapper over ``SimulatorState.from_dict`` so callers have a named
    entry point; the cached record stores raw ``finish_times`` and the view's
    ``busy_times`` is recomputed inside ``from_dict``.
    """
    return SimulatorState.from_dict(state_record)


def build_expert_permutation(
    decision: List[Tuple[int, int]], R: int, T: int
) -> torch.Tensor:
    """Construct the expert permutation matrix ``P_star`` of shape (R+T, R+T).

    Rows 0..R-1 are robots, rows R..R+T-1 are tasks. Columns 0..T-1 are
    tasks, columns T..T+R-1 are robot (idle) slots, mirroring the column
    layout of the augmented ``Theta`` from the GNN scorer.
    """
    N = R + T
    P = torch.zeros((N, N), dtype=torch.float32)

    assigned_robots = set()
    assigned_tasks = set()
    for pair in decision:
        r, j = int(pair[0]), int(pair[1])
        P[r, j] = 1.0  # top-left: robot r serves task j
        P[R + j, T + r] = 1.0  # bottom-right: task row claims freed robot col
        assigned_robots.add(r)
        assigned_tasks.add(j)

    for r in range(R):
        if r not in assigned_robots:
            P[r, T + r] = 1.0  # top-right diagonal: robot r stays idle
    for j in range(T):
        if j not in assigned_tasks:
            P[R + j, j] = 1.0  # bottom-left diagonal: task j stays pending

    return P


class ILDataset:
    """Index of commit-epoch ``(state, expert_decision)`` pairs on disk.

    Loads every ``seed*.json`` record under ``cache_dir`` (optionally scoped
    to a ``split`` subdirectory), reconstructs the instance once per record,
    and keeps one ``ILExample`` per commit epoch. Epochs whose cached expert
    decision is empty are skipped.
    """

    def __init__(self, cache_dir, split: Optional[str] = None):
        root = Path(cache_dir)
        if split is not None:
            root = root / split
        self.examples: List[ILExample] = []
        self.n_instances = 0

        for record_path in sorted(root.glob("seed*.json")):
            try:
                with record_path.open("r") as f:
                    record = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                print(
                    f"ILDataset: skipping {record_path.name} ({exc})",
                    flush=True,
                )
                continue
            self.n_instances += 1
            instance = SyntheticInstance.from_dict(record["instance"])
            seed = int(record["seed"])
            expert_decisions = record["expert_decisions"]
            expert_trajectory = record["expert_trajectory"]
            for epoch, commits in enumerate(expert_decisions):
                if not commits:
                    continue  # pure-drain epoch, no supervision
                state = reconstruct_state(expert_trajectory[epoch])
                decision = [(int(r), int(j)) for (r, j) in commits]
                self.examples.append(
                    ILExample(
                        seed=seed,
                        epoch=epoch,
                        instance=instance,
                        state=state,
                        decision=decision,
                    )
                )

        print(
            f"ILDataset loaded {len(self.examples)} commit-epoch examples "
            f"from {self.n_instances} instances",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ILExample:
        return self.examples[idx]

    def sample(self, batch_size: int, rng: random.Random) -> List[ILExample]:
        """Sample a minibatch of examples with replacement."""
        if not self.examples:
            raise ValueError("ILDataset is empty; nothing to sample.")
        return [rng.choice(self.examples) for _ in range(batch_size)]


def train_one_step(
    model: Union[GNNScorer, LinearScorer],
    optimizer: torch.optim.Optimizer,
    batch: List[ILExample],
    epsilon: float = EPSILON_DEFAULT,
    M: int = M_SAMPLES_DEFAULT,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    detail_sink: Optional[dict] = None,
) -> Tuple[float, float]:
    """Run one Adam step on the Fenchel-Young gradient over a minibatch.

    Scores each state to obtain ``Theta``, builds the expert permutation ``P_star``, and
    backpropagates the FY gradient scaled by ``1 / batch_size``. Gradients accumulate
    across the batch, are clipped to ``max_grad_norm``, then applied.

    Returns the mean loss value and the pre-clip gradient norm. The loss is the
    monitoring value, which omits ``Omega(P*)`` and can be negative.

    ``detail_sink``, when a dict, receives ``fy_true`` and the batch means of its three
    terms. It is a pure observer, adding no Hungarian solve and no RNG draw, so the
    training trajectory is bit-identical with or without it.
    """
    if not batch:
        raise ValueError("Cannot train on an empty batch.")

    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    totals = {"fy_true": 0.0, "omega_star": 0.0, "omega_target": 0.0}
    scale = 1.0 / float(len(batch))
    for example in batch:
        theta = model(example.state, example.instance)
        R = int(example.instance.R)
        T = int(example.instance.T)
        P_star = build_expert_permutation(example.decision, R, T)
        P_star = P_star.to(device=theta.device, dtype=theta.dtype)
        example_sink = {} if detail_sink is not None else None
        loss_value, gradient = fenchel_young_loss(
            theta, P_star, epsilon=epsilon, M=M, detail_sink=example_sink
        )
        theta.backward(gradient * scale)
        total_loss += float(loss_value.item())
        if example_sink is not None:
            for key in totals:
                totals[key] += example_sink[key]

    grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    if detail_sink is not None:
        detail_sink.update({k: v * scale for k, v in totals.items()})

    return total_loss * scale, float(grad_norm)


def save_checkpoint(
    model: Union[GNNScorer, LinearScorer],
    optimizer: torch.optim.Optimizer,
    step: int,
    path,
) -> None:
    """Persist model and optimiser state to ``path``.

    ``init_kwargs`` records the constructor arguments needed to rebuild an
    architecturally identical scorer (hidden dim, layer count) so a checkpoint
    from a non-default architecture can be reloaded without the caller having
    to remember its shape. It is an extra key only; every existing reader
    indexes ``model_state_dict`` and is unaffected.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if hasattr(model, "get_init_kwargs"):
        payload["init_kwargs"] = model.get_init_kwargs()
    torch.save(payload, path)


def best_checkpoint_path(checkpoint_path) -> Path:
    """Derive the best-checkpoint path from the output checkpoint path.

    Inserts ``_best`` before the extension so ``checkpoints/il_method_one.pt``
    becomes ``checkpoints/il_method_one_best.pt`` rather than appending a second
    ``.pt`` suffix.
    """
    p = Path(checkpoint_path)
    return p.with_name(f"{p.stem}_best{p.suffix}")


def evaluate_perturbed_gap(
    model: Union[GNNScorer, LinearScorer],
    records: List[dict],
    epsilon: float = EVAL_EPSILON,
    seed: int = EVAL_SEED,
    weights: Optional[dict] = None,
):
    """Evaluate the in-memory model's perturbed gap on cached val records.

    Reuses ``method_two_evaluator`` so the metric is identical to the one
    ``sweep/evaluate_sweep`` reports: the mean serving gap closure versus the
    MILP lower bound under perturbed Hungarian decode. Returns
    ``(perturbed_gap, served_all_rate, n_serving)`` where ``perturbed_gap`` is
    ``None`` when no rollout served all tasks.

    The torch RNG state is saved before seeding for the perturbed Gumbel draws
    and restored afterwards, so calling this mid-training leaves the training
    RNG stream (and thus the training trajectory) untouched. The model is set to
    eval mode for the rollouts and returned to train mode before this returns.
    """
    # Lazy import: only pulled in when validation is actually used.
    from evaluation.method_two_evaluator import (
        evaluate_one_instance,
        summarize,
    )

    weights = weights if weights is not None else EVAL_WEIGHTS
    rng_state = torch.get_rng_state()
    was_training = model.training
    try:
        torch.manual_seed(int(seed))
        model.eval()
        per_instance = [
            evaluate_one_instance(model, rec, ["perturbed"], weights, epsilon)
            for rec in records
        ]
    finally:
        torch.set_rng_state(rng_state)
        if was_training:
            model.train()

    summary = summarize(per_instance, ["perturbed"])
    pert = summary["modes"]["perturbed"]
    gap = pert["serving_gap_vs_milp"]["mean"]
    return gap, pert["served_all_rate"], pert["serving_gap_vs_milp"]["n"]


EVAL_CSV_COLUMNS = [
    "run",
    "scorer",
    "hidden_dim",
    "num_layers",
    "n_params",
    "step",
    "total_steps",
    "perturbed_gap",
    "served_all_rate",
    "n_serving",
    "n_val_instances",
    "is_new_best",
]


def append_eval_row(csv_path, row: dict) -> None:
    """Append one in-loop validation result to ``csv_path``.

    Opens, writes and closes per row so the file on disk is complete and valid
    after every evaluation rather than only at the end of a run. The header is
    written only when the file does not yet exist or is empty, so several runs
    can append to one sweep-wide file.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in EVAL_CSV_COLUMNS})


def _best_gap_from_csv(
    csv_path, run_tag: str, serve_all_floor: float
) -> Tuple[Optional[float], Optional[int]]:
    """Highest floor-eligible gap already recorded for ``run_tag``.

    Returns ``(gap, step)``, or ``(None, None)`` when the file is missing or
    holds no eligible row for that run. Applies the same serve-all floor the
    live best-checkpoint rule applies, so a row that was blocked from saving
    when it was written is not treated as a best on resume.
    """
    if not csv_path:
        return None, None
    path = Path(csv_path)
    if not path.exists():
        return None, None
    best_gap: Optional[float] = None
    best_step: Optional[int] = None
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("run") != run_tag or not row.get("perturbed_gap"):
                continue
            try:
                gap = float(row["perturbed_gap"])
                served = float(row["served_all_rate"])
                step = int(row["step"])
            except (TypeError, ValueError):
                continue
            if served < serve_all_floor:
                continue
            if best_gap is None or gap > best_gap:
                best_gap, best_step = gap, step
    return best_gap, best_step


def annealed_epsilon(
    step: int,
    num_steps: int,
    epsilon_initial: float,
    epsilon_terminal: float,
    anneal_steps: Optional[int] = None,
) -> float:
    """Log-linear (exponential) interpolation of the FY perturbation epsilon.

    ``step`` is 0-indexed. Progress runs from 0 at the first step to 1 at the
    end of the anneal window (``anneal_steps`` if given, else ``num_steps``),
    after which epsilon holds at ``epsilon_terminal``. Interpolating on the log
    scale gives an exponential decay, so epsilon halves over equal step
    intervals rather than dropping linearly.
    """
    log_initial = math.log(epsilon_initial)
    log_terminal = math.log(epsilon_terminal)
    span = anneal_steps if anneal_steps else num_steps
    progress = step / max(1, span - 1)
    progress = min(progress, 1.0)
    return math.exp(log_initial + progress * (log_terminal - log_initial))


def train(config: TrainingConfig, num_steps: int = 2000) -> List[dict]:
    """Train method one for ``num_steps`` and save a final checkpoint.

    The FY perturbation epsilon is annealed log-linearly per step from
    ``config.epsilon_initial`` to ``config.epsilon_terminal`` (methodology
    section 3.4.6). Returns the per-step history (step, loss, grad_norm,
    epsilon) for monitoring.
    """
    dataset = ILDataset(config.cache_dir, split=config.split)
    print(
        f"Training for {num_steps} steps. Dataset size: {len(dataset)}",
        flush=True,
    )
    print(
        f"Epsilon anneal: {config.epsilon_initial} -> {config.epsilon_terminal}"
        f" over {config.epsilon_anneal_steps or num_steps} steps (log scale)",
        flush=True,
    )
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)

    if config.scorer_type == "gnn":
        model = GNNScorer(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            commit_bias=config.commit_bias,
            use_queue_ahead_mask=config.use_queue_ahead_mask,
        )
    elif config.scorer_type == "linear":
        model = LinearScorer(
            commit_bias=config.commit_bias,
            use_queue_ahead_mask=config.use_queue_ahead_mask,
        )
    else:
        raise ValueError(
            f"Unknown scorer_type: {config.scorer_type!r} "
            f"(expected one of {SCORER_TYPES})"
        )
    n_params = sum(p.numel() for p in model.parameters())
    arch = (
        f", hidden_dim={config.hidden_dim}, num_layers={config.num_layers}"
        if config.scorer_type == "gnn"
        else ""
    )
    print(
        f"Scorer: {config.scorer_type} ({n_params} trainable parameters"
        f"{arch})",
        flush=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Resume reproduces the model, Adam state, epsilon/LR schedules, and minibatch stream, but not the Gumbel-perturbation RNG stream, which restarts from the seed — a faithful continuation, not a bit-identical replay.
    start_step = 1
    if config.resume_from:
        ckpt = torch.load(
            config.resume_from, map_location="cpu", weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt["step"]) + 1
        if start_step > num_steps:
            raise SystemExit(
                f"Nothing to resume: {config.resume_from} is already at step "
                f"{ckpt['step']} of {num_steps}."
            )
        for _ in range(start_step - 1):
            dataset.sample(config.batch_size, rng)
        print(
            f"Resuming from {config.resume_from} at step {start_step} "
            f"(checkpoint step {ckpt['step']})",
            flush=True,
        )

    scheduler = None
    if config.lr_schedule == "cosine":
        # Reset param groups to the base lr before constructing the scheduler (it snapshots initial_lr from whatever it finds), then walk it forward so the curve matches an uninterrupted run.
        for group in optimizer.param_groups:
            group["lr"] = config.lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps
        )
        for _ in range(start_step - 1):
            scheduler.step()
    elif config.lr_schedule != "constant":
        raise ValueError(
            f"Unknown lr_schedule: {config.lr_schedule!r} "
            f"(expected 'constant' or 'cosine')"
        )

    best_path = best_checkpoint_path(config.checkpoint_path)
    best_step = None
    loss_window: deque = deque(maxlen=max(1, int(config.best_loss_window)))

    # Active when a val cache and positive cadence are set; loss-based tracking is suppressed so the two never clobber the same _best file.
    eval_active = bool(
        config.val_cache_dir and config.eval_every_steps
        and config.eval_every_steps > 0
    )
    val_records: List[dict] = []
    best_gap = -math.inf  # perturbed gap: higher is better
    best_loss = math.inf  # trailing-window loss: lower is better
    if eval_active:
        from evaluation.method_two_evaluator import load_records

        # Guard mirrors method_two_evaluator: never evaluate on a test split.
        if any(
            part.lower() == "test"
            for part in Path(config.val_cache_dir).parts
        ):
            raise SystemExit(
                f"REFUSING to validate on a test split ({config.val_cache_dir}). "
                f"The test set is reserved for the final thesis."
            )
        val_records = load_records(config.val_cache_dir, 10**9)
        print(
            f"In-loop validation on every {config.eval_every_steps} steps: "
            f"{len(val_records)} instances from {config.val_cache_dir} "
            f"(perturbed gap vs MILP, epsilon={EVAL_EPSILON}, seed={EVAL_SEED}); "
            f"best by perturbed gap -> {best_path}",
            flush=True,
        )
    elif config.save_best:
        print(
            f"Best-checkpoint tracking on (loss window="
            f"{max(1, int(config.best_loss_window))}); saving lowest-loss "
            f"weights to {best_path}",
            flush=True,
        )

    run_tag = config.run_tag or Path(config.checkpoint_path).stem
    # On resume, recover the run's best gap from the eval CSV so a later but worse evaluation can't overwrite the surviving _best.pt.
    if config.resume_from and eval_active:
        prior_best, prior_step = _best_gap_from_csv(
            config.eval_csv_path, run_tag, config.best_serve_all_floor
        )
        if prior_best is not None:
            best_gap = prior_best
            best_step = prior_step
            print(
                f"Resumed best-gap tracking at {best_gap * 100:.1f}% "
                f"(step {best_step}), read back from {config.eval_csv_path}",
                flush=True,
            )

    history: List[dict] = []
    for step in range(start_step, num_steps + 1):
        epsilon = annealed_epsilon(
            step - 1,
            num_steps,
            config.epsilon_initial,
            config.epsilon_terminal,
            config.epsilon_anneal_steps,
        )
        batch = dataset.sample(config.batch_size, rng)
        # True FY loss detail costs ~20% of a step at production shape, so it's collected only on logged steps (trajectory is unaffected either way, per train_one_step).
        logging_this_step = bool(
            config.log_every_steps and step % config.log_every_steps == 0
        )
        fy_detail = {} if logging_this_step else None
        loss, grad_norm = train_one_step(
            model,
            optimizer,
            batch,
            epsilon=epsilon,
            M=config.M,
            max_grad_norm=config.max_grad_norm,
            detail_sink=fy_detail,
        )
        # No-op when the schedule is constant (scheduler is None).
        if scheduler is not None:
            scheduler.step()
        history.append(
            {
                "step": step,
                "loss": loss,
                "grad_norm": grad_norm,
                "epsilon": epsilon,
                # Extra keys only (logged steps); existing readers index the four above and are unaffected.
                **(fy_detail or {}),
            }
        )
        if logging_this_step:
            print(
                f"[step {step}/{num_steps}] "
                f"loss={loss:.4f} grad_norm={grad_norm:.4f} "
                f"epsilon={epsilon:.4f} "
                f"fy_true={fy_detail['fy_true']:.4f}",
                flush=True,
            )

        if eval_active and step % config.eval_every_steps == 0:
            gap, served_rate, n_serving = evaluate_perturbed_gap(
                model, val_records
            )
            gap_str = "n/a" if gap is None else f"{gap * 100:.1f}%"
            meets_floor = served_rate >= config.best_serve_all_floor
            improved = gap is not None and gap > best_gap and meets_floor
            if improved:
                best_gap = gap
                best_step = step
                save_checkpoint(model, optimizer, step, best_path)
            blocked = (
                gap is not None and gap > best_gap and not meets_floor
            )
            if improved:
                tag = "  *new best, saved*"
            elif blocked:
                tag = (
                    f"  (gap improved but below serve-all floor "
                    f"{config.best_serve_all_floor * 100:.0f}%, not saved)"
                )
            else:
                tag = ""
            print(
                f"[eval step {step}/{num_steps}] perturbed gap vs MILP="
                f"{gap_str} (served-all {served_rate * 100:.0f}%, "
                f"n_serving={n_serving})"
                f"{tag}",
                flush=True,
            )
            if config.eval_csv_path:
                append_eval_row(
                    config.eval_csv_path,
                    {
                        "run": run_tag,
                        "scorer": config.scorer_type,
                        "hidden_dim": config.hidden_dim,
                        "num_layers": config.num_layers,
                        "n_params": n_params,
                        "step": step,
                        "total_steps": num_steps,
                        "perturbed_gap": "" if gap is None else f"{gap:.6f}",
                        "served_all_rate": f"{served_rate:.6f}",
                        "n_serving": n_serving,
                        "n_val_instances": len(val_records),
                        "is_new_best": int(bool(improved)),
                    },
                )

        # Suppressed when in-loop validation is active so gap-based selection owns the _best file.
        if config.save_best and not eval_active:
            loss_window.append(loss)
            tracked = sum(loss_window) / len(loss_window)
            if tracked < best_loss:
                best_loss = tracked
                best_step = step
                save_checkpoint(model, optimizer, step, best_path)

        # Skipped on the final step to avoid a double write (saved unconditionally below).
        if (
            config.checkpoint_every_steps
            and step % config.checkpoint_every_steps == 0
            and step != num_steps
        ):
            save_checkpoint(model, optimizer, step, config.checkpoint_path)
            print(
                f"Saved intermediate checkpoint at step {step} to "
                f"{config.checkpoint_path}",
                flush=True,
            )

    save_checkpoint(model, optimizer, num_steps, config.checkpoint_path)
    print(f"Saved checkpoint to {config.checkpoint_path}", flush=True)
    if eval_active and best_step is not None:
        print(
            f"Best checkpoint: step {best_step} (perturbed gap "
            f"{best_gap * 100:.1f}%) at {best_path}",
            flush=True,
        )
    elif config.save_best and best_step is not None:
        print(
            f"Best checkpoint: step {best_step} (tracked loss "
            f"{best_loss:.4f}) at {best_path}",
            flush=True,
        )
    return history


class _Tee:
    """Write to several streams at once (console plus a log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Method one IL training loop (methodology section 3.5.3)."
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--gumbel-samples",
        type=int,
        default=M_SAMPLES_DEFAULT,
        help="Number of Gumbel samples M in the Fenchel-Young gradient.",
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=["constant", "cosine"],
        help="Learning-rate schedule over the run.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Initial value of the log-linear epsilon anneal (methodology "
        "section 3.4.6). The perturbation epsilon decays from this value down "
        "to --epsilon-end. Default 1.0.",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.01,
        help="Terminal (floor) value of the log-linear epsilon anneal "
        "(methodology section 3.4.6). The perturbation epsilon decays from "
        "epsilon_initial down to this value. Default 0.01.",
    )
    parser.add_argument(
        "--epsilon-anneal-steps",
        type=int,
        default=None,
        help="Number of steps over which to anneal epsilon before holding at "
        "--epsilon-end for the remainder of the run. When omitted, epsilon "
        "anneals over the full run (unchanged behaviour).",
    )
    parser.add_argument(
        "--output-checkpoint",
        "--checkpoint-path",
        dest="checkpoint_path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint save path. Refuses to overwrite an existing file "
        "unless --force is set.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Allow overwriting an existing output checkpoint.",
    )
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY_STEPS)
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="If set, tee all training output to this file as well as stdout.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Dataset cache root; the split subdirectory is appended.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Write an intermediate checkpoint every N steps (0 disables).",
    )
    parser.add_argument(
        "--no-save-best",
        dest="save_best",
        action="store_false",
        default=True,
        help="Disable best-checkpoint tracking (the lowest-loss weights saved "
        "to <output-checkpoint stem>_best.pt). Tracking is on by default.",
    )
    parser.add_argument(
        "--best-loss-window",
        type=int,
        default=1,
        help="Smooth the loss over a trailing window of this many steps before "
        "comparing for a new minimum. 1 (default) tracks the raw per-step loss; "
        "a larger window is more robust to a single lucky-easy batch.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Evaluate the model on the val set every N steps (0 disables). "
        "When active, the best checkpoint is selected by the perturbed gap vs "
        "MILP rather than by loss. Requires --val-cache-dir.",
    )
    parser.add_argument(
        "--val-cache-dir",
        type=str,
        default=None,
        help="Validation cache directory of seed*.json records. When omitted, "
        "in-loop validation is skipped.",
    )
    parser.add_argument(
        "--best-serve-all-floor",
        type=float,
        default=0.95,
        help="Minimum perturbed served-all rate for a checkpoint to be eligible "
        "to be saved as best under in-loop validation. Guards against saving a "
        "high-gap but near-zero serve-all checkpoint as best. Default 0.95.",
    )
    parser.add_argument(
        "--commit-bias",
        type=float,
        default=0.0,
        help="Positive scalar added to the valid robot-task assignment block of "
        "the GNN scorer's augmented matrix (see gnn_scorer.py). Default 0.0 "
        "leaves the scorer bit-identical to the unbiased path.",
    )
    parser.add_argument(
        "--use-queue-ahead-mask",
        dest="use_queue_ahead_mask",
        action="store_true",
        default=False,
        help="Path B mask. Drop the robot-availability factor from the GNN "
        "scorer's valid mask so busy robots keep their scores and can be "
        "committed queue-ahead, rather than being masked to -M_MASK. Default "
        "off (busy robots masked).",
    )
    parser.add_argument(
        "--scorer",
        dest="scorer_type",
        type=str,
        default="gnn",
        choices=list(SCORER_TYPES),
        help="Scorer architecture. 'gnn' (default) is the original deep "
        "GNNScorer path, unchanged. 'linear' swaps in the affine LinearScorer "
        "convexity diagnostic (see scoring/linear_scorer.py): the "
        "Fenchel-Young loss, CO-layer, and cache loading are identical either "
        "way.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=HIDDEN_DIM,
        help="GNN scorer hidden dimension. Default 64, the production Method "
        "One width.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of message-passing layers in the GNN scorer. Default 2, "
        "the production Method One depth.",
    )
    parser.add_argument(
        "--eval-csv",
        dest="eval_csv_path",
        type=str,
        default=None,
        help="Append one row per in-loop validation pass to this CSV. Written "
        "and flushed per evaluation, so the file stays valid if the run is "
        "interrupted. Several runs may append to the same file.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Label written into the eval CSV's run column. Defaults to the "
        "output checkpoint stem.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from a rolling checkpoint written by --checkpoint-every. "
        "Restores weights, optimiser state and the LR schedule, and restarts "
        "at the saved step + 1. Implies --force for the output checkpoint.",
    )
    args = parser.parse_args()

    # Checked before training so a long run never starts only to be blocked at save time; waived on resume since it writes back to the checkpoint it's continuing.
    if args.resume_from:
        args.force = True
    if Path(args.checkpoint_path).exists() and not args.force:
        raise SystemExit(
            f"Output checkpoint {args.checkpoint_path} already exists. "
            f"Pass --force to overwrite, or choose another --output-checkpoint."
        )

    log_fh = None
    original_stdout = sys.stdout
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(args.log_file, "a")
        sys.stdout = _Tee(original_stdout, log_fh)

    try:
        print(f"Starting IL training with {args.steps} steps", flush=True)
        config = TrainingConfig(
            cache_dir=args.cache_dir,
            log_every_steps=args.log_every,
            checkpoint_path=args.checkpoint_path,
            checkpoint_every_steps=args.checkpoint_every,
            lr=args.lr,
            lr_schedule=args.lr_schedule,
            batch_size=args.batch_size,
            M=args.gumbel_samples,
            epsilon_initial=args.epsilon_start,
            epsilon_terminal=args.epsilon_end,
            epsilon_anneal_steps=args.epsilon_anneal_steps,
            save_best=args.save_best,
            best_loss_window=args.best_loss_window,
            eval_every_steps=args.eval_every,
            val_cache_dir=args.val_cache_dir,
            best_serve_all_floor=args.best_serve_all_floor,
            commit_bias=args.commit_bias,
            use_queue_ahead_mask=args.use_queue_ahead_mask,
            scorer_type=args.scorer_type,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            eval_csv_path=args.eval_csv_path,
            run_tag=args.run_tag,
            resume_from=args.resume_from,
        )
        train(config, num_steps=args.steps)
    finally:
        if log_fh is not None:
            sys.stdout = original_stdout
            log_fh.close()


if __name__ == "__main__":
    main()
