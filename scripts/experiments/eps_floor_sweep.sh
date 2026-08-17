#!/usr/bin/env bash
# Method One epsilon floor axis, completion run.
#
# Stage 2 of the overnight chain sampled the epsilon floor at 0.40, 0.60, 0.80
# and constant 1.00 only, and 0.60 scored highest. This script fills in 0.20,
# 0.30, 0.50 and 0.70 at otherwise identical settings, closing the axis to
# eight points and showing whether 0.60 is a peak or a monotone trend's top.
#
# Every setting other than the epsilon floor is held at the production Method
# One configuration that produced checkpoints/il_method_one_v4_best.pt: lr
# 5e-4 cosine, batch 32, 30 Gumbel samples, epsilon annealed log-linearly from
# 1.0 over 3000 steps and held at the floor thereafter, 10000 steps, gradient
# clip 1.0 (il_trainer.py has no --grad-clip flag; DEFAULT_MAX_GRAD_NORM is
# 1.0, so every arm is clipped at the production value with no way to
# diverge), 2 layers, hidden 64, the v4 training cache and v3 validation
# cache, in-loop validation every 1000 steps.
#
# WRITES ONLY NEW PATHS: checkpoints/eps_floor_*, provenance/eps_floor_*.csv,
# provenance/eps_floor_*.json, logs/eps_floor_*. The two production checkpoints
# are hashed before and after and a change fails loudly.
#
# VALIDATION SPLIT ONLY. No arm touches the test split. Method Two is not
# trained here.
#
# BASH 3.2 SAFETY. The 1 August stage 2 failure was an empty array expanded
# under `set -u`, putting a stray empty argument in front of the trainer and
# killing all eight arms before they trained a step. This script has no
# optional-argument array at all; the only optional flag is --resume-from,
# handled by two explicit invocations in an if/else rather than by splicing a
# possibly-empty array. The one array here, the fixed argument list, is never
# empty.
#
# RESUME ACCOUNTING. A resumed arm restores weights, optimiser state and the
# LR schedule but not the torch RNG stream, so its trajectory is not
# comparable step for step with a fresh arm. Every arm records whether it ran
# fresh or resumed, and from which step, in
# provenance/eps_floor_<tag>_runinfo.json and in the chain log.
#
# Usage:
#   nohup caffeinate -i bash scripts/experiments/eps_floor_sweep.sh \
#       > logs/eps_floor_nohup.log 2>&1 &
#
#   bash scripts/experiments/eps_floor_sweep.sh 0.50     (one arm only)
#   DRY_RUN=1 bash scripts/experiments/eps_floor_sweep.sh

set -uo pipefail

cd "$(dirname "$0")/../.."
REPO="$PWD"
export PYTHONPATH="$REPO/src"

# Interpreter from the active environment (a bare `python` in a detached
# shell resolves to whatever that shell inherited, not necessarily coaml).
# Set PY to override.
PY="${PY:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
: "${PY:=$(command -v python3 || command -v python)}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
CHAIN_LOG="logs/eps_floor_chain_${RUN_STAMP}.log"
mkdir -p logs provenance checkpoints

STEPS=10000
BATCH=32
SAMPLES=30
LR=5e-4
LR_SCHEDULE=cosine
EPS_START=1.0
ANNEAL_STEPS=3000
EVAL_EVERY=1000
CKPT_EVERY=1000
NUM_LAYERS=2
HIDDEN_DIM=64
TRAIN_CACHE="cache/training_set_il_v4"
VAL_CACHE="cache/training_set_il_v3/val"

PROTECTED_A="checkpoints/il_method_one_v4_best.pt"
PROTECTED_B="checkpoints/sweep_G_v4warmstart_best.pt"

log () {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$CHAIN_LOG"
}

hash_protected () {
    local p
    for p in "$PROTECTED_A" "$PROTECTED_B"; do
        if [ -e "$p" ]; then
            shasum -a 256 "$p"
        else
            echo "MISSING  $p"
        fi
    done
}

# Step count stored inside a checkpoint, or -1 when missing or unreadable.
ckpt_step () {
    "$PY" - "$1" <<'PY' 2>/dev/null || echo -1
import sys, torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu",
                         weights_only=False)["step"]))
except Exception:
    print(-1)
PY
}

# $1 tag, $2 epsilon floor.
run_arm () {
    local tag="$1"
    local floor="$2"
    local ckpt="checkpoints/eps_floor_${tag}.pt"
    local csv="provenance/eps_floor_${tag}.csv"
    local info="provenance/eps_floor_${tag}_runinfo.json"
    local arm_log="logs/eps_floor_${tag}.log"
    local started ended status done_step mode resume_step

    # Fixed argument list, always non-empty, so expanding it under `set -u`
    # is safe on bash 3.2. No optional flag is ever spliced in here.
    local common=(
        --steps "$STEPS"
        --batch-size "$BATCH"
        --gumbel-samples "$SAMPLES"
        --lr "$LR"
        --lr-schedule "$LR_SCHEDULE"
        --epsilon-start "$EPS_START"
        --epsilon-end "$floor"
        --epsilon-anneal-steps "$ANNEAL_STEPS"
        --num-layers "$NUM_LAYERS"
        --hidden-dim "$HIDDEN_DIM"
        --cache-dir "$TRAIN_CACHE"
        --val-cache-dir "$VAL_CACHE"
        --eval-every "$EVAL_EVERY"
        --checkpoint-every "$CKPT_EVERY"
        --output-checkpoint "$ckpt"
        --eval-csv "$csv"
        --run-tag "$tag"
        --log-every 500
        --log-file "$arm_log"
    )

    if [ -n "${DRY_RUN:-}" ]; then
        echo "--- ${tag} (epsilon floor ${floor}) ---"
        echo "  ${PY} -u src/training/il_trainer.py ${common[*]}"
        return 0
    fi

    done_step="$(ckpt_step "$ckpt")"
    if [ "$done_step" = "$STEPS" ]; then
        log "ARM ${tag} SKIP    already complete at step ${STEPS} (${ckpt})"
        return 0
    fi

    started="$(date '+%Y-%m-%d %H:%M:%S')"
    if [ "$done_step" -gt 0 ] 2>/dev/null; then
        mode="resumed"
        resume_step="$done_step"
        log "ARM ${tag} START   epsilon floor ${floor}, RESUMING from step ${done_step}"
        log "  NOTE: ${tag} resumed, so its torch RNG stream differs from a fresh run and it is not step-for-step comparable with the fresh arms."
        "$PY" -u src/training/il_trainer.py "${common[@]}" --resume-from "$ckpt"
        status=$?
    else
        mode="fresh"
        resume_step="null"
        log "ARM ${tag} START   epsilon floor ${floor}, running FRESH from step 1"
        "$PY" -u src/training/il_trainer.py "${common[@]}"
        status=$?
    fi
    ended="$(date '+%Y-%m-%d %H:%M:%S')"

    # One relaunch on a crash, resuming from the rolling checkpoint. A relaunch
    # is itself a resume and is recorded as one.
    if [ "$status" -ne 0 ]; then
        log "ARM ${tag} FAIL    exit=${status}. Relaunching once from the rolling checkpoint."
        resume_step="$(ckpt_step "$ckpt")"
        if [ "$resume_step" -gt 0 ] 2>/dev/null; then
            mode="resumed"
            log "ARM ${tag} RETRY   resuming from step ${resume_step}"
            "$PY" -u src/training/il_trainer.py "${common[@]}" --resume-from "$ckpt"
            status=$?
        else
            mode="fresh"
            resume_step="null"
            log "ARM ${tag} RETRY   no usable rolling checkpoint, restarting from step 1"
            "$PY" -u src/training/il_trainer.py "${common[@]}"
            status=$?
        fi
        ended="$(date '+%Y-%m-%d %H:%M:%S')"
        if [ "$status" -ne 0 ]; then
            log "ARM ${tag} FAIL    exit=${status} on the relaunch. Continuing to the next arm."
            FAILED="${FAILED} ${tag}"
        fi
    fi

    local final_step
    final_step="$(ckpt_step "$ckpt")"

    cat > "$info" <<JSON
{
  "tag": "${tag}",
  "epsilon_end": ${floor},
  "epsilon_start": ${EPS_START},
  "epsilon_anneal_steps": ${ANNEAL_STEPS},
  "lr": "${LR}",
  "lr_schedule": "${LR_SCHEDULE}",
  "batch_size": ${BATCH},
  "gumbel_samples": ${SAMPLES},
  "num_layers": ${NUM_LAYERS},
  "hidden_dim": ${HIDDEN_DIM},
  "gradient_clip": 1.0,
  "total_steps": ${STEPS},
  "eval_every": ${EVAL_EVERY},
  "checkpoint_every": ${CKPT_EVERY},
  "train_cache": "${TRAIN_CACHE}",
  "val_cache": "${VAL_CACHE}",
  "run_mode": "${mode}",
  "resumed_from_step": ${resume_step},
  "comparable_to_fresh_arms": $( [ "$mode" = "fresh" ] && echo true || echo false ),
  "started": "${started}",
  "ended": "${ended}",
  "exit_code": ${status},
  "final_checkpoint_step": ${final_step},
  "checkpoint": "${ckpt}",
  "trajectory_csv": "${csv}",
  "chain_log": "${CHAIN_LOG}"
}
JSON

    log "ARM ${tag} END     exit=${status}  mode=${mode}  final_step=${final_step}  csv=${csv}"
    return 0
}

log "=================================================================="
log "Method One epsilon floor completion sweep. PID $$."
log "Repo ${REPO}"
log "Interpreter ${PY}"
log "Validation split only. Method Two is not trained."
log "Production settings held fixed: lr ${LR} ${LR_SCHEDULE}, batch ${BATCH}, ${SAMPLES} Gumbel samples, epsilon start ${EPS_START} annealed over ${ANNEAL_STEPS} steps, ${STEPS} steps, grad clip 1.0, ${NUM_LAYERS} layers, hidden ${HIDDEN_DIM}."
log "Train cache ${TRAIN_CACHE}, val cache ${VAL_CACHE}, eval every ${EVAL_EVERY}."
log "Protected checkpoint hashes BEFORE:"
hash_protected | while read -r line; do log "  ${line}"; done
BEFORE_HASHES="$(hash_protected)"

FAILED=""

if [ "$#" -eq 1 ]; then
    case "$1" in
        0.20) run_arm e020 0.20 ;;
        0.30) run_arm e030 0.30 ;;
        0.50) run_arm e050 0.50 ;;
        0.70) run_arm e070 0.70 ;;
        *) log "Unknown floor '$1'. Valid: 0.20 0.30 0.50 0.70"; exit 2 ;;
    esac
else
    run_arm e020 0.20
    run_arm e030 0.30
    run_arm e050 0.50
    run_arm e070 0.70
fi

log "Protected checkpoint hashes AFTER:"
hash_protected | while read -r line; do log "  ${line}"; done
AFTER_HASHES="$(hash_protected)"

EXIT_STATUS=0
if [ "$BEFORE_HASHES" != "$AFTER_HASHES" ]; then
    log "!!! A PROTECTED CHECKPOINT CHANGED DURING THIS SWEEP."
    log "!!! Treat every result from this run as suspect."
    EXIT_STATUS=2
else
    log "Protected checkpoints unchanged."
fi

if [ -n "${FAILED// /}" ]; then
    log "Arms that failed twice:${FAILED}"
    [ "$EXIT_STATUS" -eq 0 ] && EXIT_STATUS=1
fi

log "Epsilon floor sweep complete. Exit ${EXIT_STATUS}."
log "=================================================================="
exit "$EXIT_STATUS"
