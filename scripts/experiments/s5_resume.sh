#!/usr/bin/env bash
# Resume stage 5 of the overnight chain to step 400.
#
# Stage 5 is the Method Two K against batch arm: rloo_k 40, batch 64, so B*K
# is 2560, the same rollout budget per step as production's K 20 batch 128.
# It started 3 August 06:34:57, reached step 170, and the process vanished at
# roughly 09:42 with no termination line in any log and no watchdog trigger
# (the watchdog would not have fired until 20:34, so this was not a timeout).
#
# The last periodic checkpoint is step 150, written 09:21. Steps 151-170 were
# trained but never checkpointed and are lost, so this run restarts at step
# 151 and redoes them.
#
# RESUMED, NOT REPLAYED. sweep/train_one.py sets resume_from automatically
# when the checkpoint path exists, and the trainer restores model, optimiser,
# baseline, history and start_step, but not the torch RNG stream, so steps
# 151 onward will not be bit-identical to the original process's 151-170.
# This is a faithful continuation from the step 150 weights, not a replay --
# recorded in the chain log and in
# provenance/run_metadata/overnight_s5_k40_b64_runinfo.json.
#
# HYPERPARAMETERS UNCHANGED. Every flag below is exactly the one the chain's
# stage5_body used, so train_one.py rebuilds the identical MethodTwoConfig,
# verified field by field against the config dictionary stored inside
# checkpoints/overnight_s5_k40_b64.pt before launching. --warm-start-checkpoint
# is passed for exactness but is ignored on a resume, by design in train_one.py.
#
# PROTECTED CHECKPOINTS. Both production artefacts are hashed before and
# after; a change fails loudly and marks the run suspect.
#
# VALIDATION SPLIT ONLY. Nothing else is trained; the test split is untouched.
#
# BASH 3.2 SAFETY. No optional-argument array anywhere; every argument is
# fixed and passed positionally, so the empty-array expansion that killed
# stage 2 on 1 August cannot occur here.
#
# Expected runtime: 250 steps at roughly 66 s/step, measured off this arm's
# own first 170 steps, so about 4.6 hours plus ten validation passes.
#
# Usage:
#   nohup caffeinate -i bash scripts/experiments/s5_resume.sh \
#       > logs/s5_resume_nohup.log 2>&1 &

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
CHAIN_LOG="logs/s5_resume_chain_${RUN_STAMP}.log"
ARM_LOG="logs/overnight_s5_k40_b64.log"
CKPT="checkpoints/overnight_s5_k40_b64.pt"
INFO="provenance/run_metadata/overnight_s5_k40_b64_runinfo.json"
mkdir -p logs provenance

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

log "=================================================================="
log "Stage 5 resume. PID $$. Repo ${REPO}."
log "Interpreter ${PY}"
log "Validation split only. Nothing else is trained."

RESUME_FROM_STEP="$(ckpt_step "$CKPT")"
if [ "$RESUME_FROM_STEP" -lt 1 ] 2>/dev/null; then
    log "FATAL: no readable checkpoint at ${CKPT} (ckpt_step returned ${RESUME_FROM_STEP})."
    log "Refusing to start, because a fresh run here would silently discard the partial arm."
    exit 3
fi
if [ "$RESUME_FROM_STEP" = "400" ]; then
    log "Checkpoint already at step 400. Nothing to do."
    exit 0
fi

log "Partial checkpoint ${CKPT} holds step ${RESUME_FROM_STEP}."
log "RESUMING from step ${RESUME_FROM_STEP}. This arm is a continuation, NOT a bit-identical replay:"
log "  the trainer restores weights, optimiser, baseline and history but not the torch RNG stream."
log "Protected checkpoint hashes BEFORE:"
hash_protected | while read -r line; do log "  ${line}"; done
BEFORE_HASHES="$(hash_protected)"

STARTED="$(date '+%Y-%m-%d %H:%M:%S')"
log "ARM s5_k40_b64 START   rloo_k 40, batch 64, B*K 2560, resuming at step ${RESUME_FROM_STEP}, target 400"

"$PY" -u sweep/train_one.py \
    --config-id G_v4warmstart \
    --num-workers 12 \
    --max-steps 400 \
    --rloo-k 40 \
    --batch-size 64 \
    --epsilon-anneal-steps 1000 \
    --eval-every 25 \
    --warm-start-checkpoint checkpoints/il_method_one_v4_best.pt \
    --checkpoint-path "$CKPT" \
    >> "$ARM_LOG" 2>&1
STATUS=$?
ENDED="$(date '+%Y-%m-%d %H:%M:%S')"

FINAL_STEP="$(ckpt_step "$CKPT")"
log "ARM s5_k40_b64 END     exit=${STATUS}  final_step=${FINAL_STEP}  log=${ARM_LOG}"

log "Protected checkpoint hashes AFTER:"
hash_protected | while read -r line; do log "  ${line}"; done
AFTER_HASHES="$(hash_protected)"

EXIT_STATUS="$STATUS"
if [ "$BEFORE_HASHES" != "$AFTER_HASHES" ]; then
    log "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    log "!!! A PROTECTED CHECKPOINT CHANGED DURING THIS RUN."
    log "!!! Before:"
    echo "$BEFORE_HASHES" | while read -r line; do log "!!!   ${line}"; done
    log "!!! After:"
    echo "$AFTER_HASHES" | while read -r line; do log "!!!   ${line}"; done
    log "!!! Treat every result from this run as suspect."
    log "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    EXIT_STATUS=2
else
    log "Protected checkpoints unchanged."
fi

cat > "$INFO" <<JSON
{
  "arm": "overnight_s5_k40_b64",
  "description": "Method Two K against batch at fixed B*K 2560",
  "rloo_k": 40,
  "batch_size": 64,
  "b_times_k": 2560,
  "learning_rate": 0.0001,
  "gradient_clip_norm": 1.0,
  "epsilon_initial": 1.0,
  "epsilon_terminal": 0.5,
  "epsilon_anneal_steps": 1000,
  "eval_every_steps": 25,
  "checkpoint_every_steps": 25,
  "num_workers": 12,
  "warm_start_from": "checkpoints/il_method_one_v4_best.pt",
  "target_steps": 400,
  "run_mode": "resumed",
  "resumed_from_step": ${RESUME_FROM_STEP},
  "bit_identical_replay": false,
  "rng_stream_continuous": false,
  "steps_retrained": "151 to 170 were trained by the original process but never checkpointed, so they are redone here on a different RNG stream",
  "original_run_started": "2026-08-03 06:34:57",
  "original_run_died_near": "2026-08-03 09:42",
  "resume_started": "${STARTED}",
  "resume_ended": "${ENDED}",
  "exit_code": ${STATUS},
  "final_checkpoint_step": ${FINAL_STEP},
  "checkpoint": "${CKPT}",
  "arm_log": "${ARM_LOG}",
  "chain_log": "${CHAIN_LOG}"
}
JSON

log "Run metadata written to ${INFO}"
log "Stage 5 resume complete. Exit ${EXIT_STATUS}."
log "=================================================================="
exit "$EXIT_STATUS"
