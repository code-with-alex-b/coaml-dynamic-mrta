#!/usr/bin/env bash
# Method Two from the epsilon floor 0.60 Method One checkpoint.
#
# Tests whether the imitation stage's five point gain at floor 0.60 survives
# into the pipeline. The Method One epsilon floor sweep put floors 0.50, 0.60
# and 0.70 on a plateau near 8.4-9.4% perturbed gap closure against
# production's 3.78% at floor 0.40; the ordering inside that plateau is within
# one standard error, so 0.60 is the midpoint of the plateau, not the winner.
#
# THE ONLY CHANGE IS THE WARM START SOURCE. Every Method Two hyperparameter is
# the production value read out of the config dictionary stored inside
# checkpoints/sweep_G_v4warmstart_best.pt: rloo_k 20, batch 128 so B*K is 2560,
# lr 1e-4, gradient clip 1.0, epsilon 1.0 to 0.5, 12 workers, 1000 steps,
# validation every 25 steps on the 200 instance v3 validation cache, the v3
# training cache, rng_base_seed 0.
#
# Warm start parity: production warm started from il_method_one_v4_best.pt, a
# BEST checkpoint selected on perturbed gap, not a final one. The matching
# artefact on the 0.60 arm is therefore overnight_s2_eps060_best.pt (step
# 8000), not the step 10000 final -- using the final would change two things
# at once and not be a clean single-variable test.
#
# --epsilon-anneal-steps 1000 is not a deviation. The production entry leaves
# the span unset, stretching the anneal over the whole run; pinning it to
# 1000 (this run's length) reproduces production's epsilon step for step and
# makes the schedule resume-safe (train_one.py notes an unpinned span
# stretches and rewinds epsilon on a resume under a different budget, as
# observed in sweep_G_tuned_continued).
#
# THE GATE is 0.4447 at step 400, the production replicate on an independent
# run. Production's own 0.3881 is the lower of the two production
# observations, so beating it alone would mean nothing. THIS SCRIPT DOES NOT
# STOP AT THE GATE: it runs to 1000 unless killed; the gate is a reporting
# checkpoint only.
#
# WRITES ONLY NEW PATHS: checkpoints/m2_eps060_warmstart*.pt,
# logs/m2_eps060_warmstart.log, logs/m2_eps060_status.log,
# logs/m2_eps060_chain_*.log, provenance/run_metadata/m2_eps060_warmstart_runinfo.json.
# Both production checkpoints are hashed before and after and a change fails
# loudly; neither is ever a write target.
#
# VALIDATION SPLIT ONLY. The test split is never evaluated; adoption is
# decided on validation alone.
#
# RESUMABLE. Checkpoint every 25 steps, matching the evaluation cadence.
# train_one.py sets resume_from automatically when the checkpoint path
# exists, so relaunching this script continues rather than restarting.
#
# BASH 3.2 SAFETY. No optional-argument array anywhere; every argument is fixed.
#
# Expected runtime: roughly 40 s/step at this shape, measured off
# overnight_s3_archwinner, so about 4.5 hours to the step 400 gate and about
# 11.3 hours to step 1000.
#
# Usage:
#   nohup caffeinate -i bash scripts/experiments/m2_eps060_warmstart.sh \
#       > logs/m2_eps060_nohup.log 2>&1 &

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
CHAIN_LOG="logs/m2_eps060_chain_${RUN_STAMP}.log"
ARM_LOG="logs/m2_eps060_warmstart.log"
STATUS_LOG="logs/m2_eps060_status.log"
CKPT="checkpoints/m2_eps060_warmstart.pt"
INFO="provenance/run_metadata/m2_eps060_warmstart_runinfo.json"
WARM_START="checkpoints/overnight_s2_eps060_best.pt"
mkdir -p logs provenance checkpoints

PROTECTED_A="checkpoints/il_method_one_v4_best.pt"
PROTECTED_B="checkpoints/sweep_G_v4warmstart_best.pt"

GATE_STEP=400
GATE_REFERENCE=0.4447
PRODUCTION_OWN=0.3881

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
log "Method Two from the epsilon floor 0.60 Method One checkpoint. PID $$."
log "Repo ${REPO}"
log "Interpreter ${PY}"
log "VALIDATION SPLIT ONLY. The test split is never evaluated."
log "Gate: step ${GATE_STEP} against ${GATE_REFERENCE} (production replicate); production's own is ${PRODUCTION_OWN}."
log "This script does NOT stop at the gate. It runs to 1000 steps unless killed."

if [ ! -e "$WARM_START" ]; then
    log "FATAL: warm start checkpoint ${WARM_START} does not exist."
    exit 3
fi
log "Warm start source: ${WARM_START}"
log "  sha256: $(shasum -a 256 "$WARM_START" | cut -d' ' -f1)"
log "  step:   $(ckpt_step "$WARM_START")"

EXISTING="$(ckpt_step "$CKPT")"
if [ "$EXISTING" = "1000" ]; then
    log "Checkpoint ${CKPT} already at step 1000. Nothing to do."
    exit 0
fi
if [ "$EXISTING" -gt 0 ] 2>/dev/null; then
    RUN_MODE="resumed"
    log "Existing checkpoint at step ${EXISTING}. RESUMING."
    log "  NOTE: a resumed run does not reproduce the torch RNG stream, so it is a"
    log "  faithful continuation rather than a bit-identical replay."
else
    RUN_MODE="fresh"
    EXISTING="null"
    log "No existing checkpoint. Running FRESH from step 1."
fi

log "Protected checkpoint hashes BEFORE:"
hash_protected | while read -r line; do log "  ${line}"; done
BEFORE_HASHES="$(hash_protected)"

# Status follower. Mirrors each validation line into a compact, timestamped log
# the user can tail. Purely a reader on ARM_LOG; it never touches training.
: > "$STATUS_LOG"
(
    tail -n +1 -F "$ARM_LOG" 2>/dev/null \
    | grep --line-buffered -E "eval step|Best gap closure|Resumed from|Warm-starting" \
    | while IFS= read -r line; do
        printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line" >> "$STATUS_LOG"
      done
) &
FOLLOWER_PID=$!
log "Status follower PID ${FOLLOWER_PID}, writing ${STATUS_LOG}"

cleanup_follower () {
    if [ -n "${FOLLOWER_PID:-}" ]; then
        # Kill the follower and the tail beneath it; verified rather than
        # assumed, since a bulk kill with suppressed stderr can silently fail.
        local kid
        for kid in $(pgrep -P "$FOLLOWER_PID" 2>/dev/null); do
            kill -TERM "$kid" 2>/dev/null
        done
        kill -TERM "$FOLLOWER_PID" 2>/dev/null
        sleep 2
        for kid in $(pgrep -P "$FOLLOWER_PID" 2>/dev/null); do
            kill -KILL "$kid" 2>/dev/null
        done
        kill -KILL "$FOLLOWER_PID" 2>/dev/null
        if kill -0 "$FOLLOWER_PID" 2>/dev/null; then
            log "  WARNING: status follower ${FOLLOWER_PID} survived; kill it by hand."
        else
            log "  status follower stopped."
        fi
    fi
}
trap cleanup_follower EXIT

STARTED="$(date '+%Y-%m-%d %H:%M:%S')"
log "RUN START   mode=${RUN_MODE}  rloo_k 20, batch 128, B*K 2560, lr 1e-4, clip 1.0, epsilon 1.0->0.5 over 1000, 12 workers, 1000 steps"

# Retry loop for the Rosetta 2 fault. This x86_64-under-Rosetta environment
# throws a sporadic `assertion failed [arm_interval().contains(address)]`
# (CodeFragmentMetadata.cpp:53 instruction_extents_for_arm_address), aborting
# with SIGABRT (exit 134), sometimes during worker pool startup before a
# single step runs. scripts/experiments/dm_arch_sweep.sh documents the same
# fault and relaunches once -- an environment fault, not a configuration error.
#
# Each attempt after the first resumes from the rolling checkpoint when one
# exists (train_one.py sets resume_from automatically), so a mid-flight abort
# costs at most 25 steps. Orphaned workers are reaped between attempts,
# verified per PID, since a surviving pool holding a pipe open is what
# stalled an earlier chain and a bulk kill can fail silently.
MAX_ATTEMPTS=6
ATTEMPT=0
STATUS=1
ABORT_COUNT=0

reap_workers () {
    local p kid alive
    alive=""
    for p in $(pgrep -f 'multiprocessing.spawn|multiprocessing.resource_tracker' 2>/dev/null); do
        kill -KILL "$p" 2>/dev/null
        alive="${alive} ${p}"
    done
    [ -z "${alive// /}" ] && return 0
    sleep 3
    for p in $alive; do
        if kill -0 "$p" 2>/dev/null; then
            log "    WARNING: worker ${p} survived SIGKILL"
        fi
    done
    log "    reaped orphaned workers:${alive}"
}

while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
    ATTEMPT=$((ATTEMPT + 1))
    RESUME_AT="$(ckpt_step "$CKPT")"
    if [ "$RESUME_AT" = "1000" ]; then
        log "  attempt ${ATTEMPT}: checkpoint already at step 1000, done."
        STATUS=0
        break
    fi
    if [ "$ATTEMPT" -gt 1 ]; then
        if [ "$RESUME_AT" -gt 0 ] 2>/dev/null; then
            log "  attempt ${ATTEMPT}/${MAX_ATTEMPTS}: resuming from step ${RESUME_AT}"
        else
            log "  attempt ${ATTEMPT}/${MAX_ATTEMPTS}: no usable checkpoint, starting fresh"
        fi
    fi

    "$PY" -u sweep/train_one.py \
        --config-id G_v4warmstart \
        --num-workers 12 \
        --max-steps 1000 \
        --epsilon-anneal-steps 1000 \
        --eval-every 25 \
        --warm-start-checkpoint "$WARM_START" \
        --checkpoint-path "$CKPT" \
        >> "$ARM_LOG" 2>&1
    STATUS=$?

    [ "$STATUS" -eq 0 ] && break

    if [ "$STATUS" -eq 134 ]; then
        ABORT_COUNT=$((ABORT_COUNT + 1))
        log "  attempt ${ATTEMPT} hit the Rosetta abort (exit 134). Reaping and retrying."
    else
        log "  attempt ${ATTEMPT} exited ${STATUS} (not the Rosetta abort). Reaping and retrying."
    fi
    reap_workers
    sleep 10
done

if [ "$STATUS" -ne 0 ]; then
    log "  EXHAUSTED ${MAX_ATTEMPTS} attempts, last exit ${STATUS}. Giving up."
fi
log "  attempts used: ${ATTEMPT}, of which Rosetta aborts: ${ABORT_COUNT}"
ENDED="$(date '+%Y-%m-%d %H:%M:%S')"

FINAL_STEP="$(ckpt_step "$CKPT")"
log "RUN END     exit=${STATUS}  final_step=${FINAL_STEP}"

# The gate figure, read back out of the checkpoint's own eval history.
GATE_VALUE="$("$PY" - "$CKPT" "$GATE_STEP" <<'PY' 2>/dev/null || echo "unavailable"
import sys, torch
try:
    d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    want = int(sys.argv[2])
    for e in d.get("eval_history", []):
        if int(e["step"]) == want:
            print(f"{e['gap']:.4f}")
            break
    else:
        print("not reached")
except Exception:
    print("unavailable")
PY
)"
log "GATE  step ${GATE_STEP}: ${GATE_VALUE}   against ${GATE_REFERENCE} (replicate) and ${PRODUCTION_OWN} (production's own)"

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
  "run": "m2_eps060_warmstart",
  "purpose": "does the Method One epsilon floor 0.60 gain survive into Method Two",
  "warm_start_from": "${WARM_START}",
  "warm_start_sha256": "$(shasum -a 256 "$WARM_START" | cut -d' ' -f1)",
  "warm_start_step": $(ckpt_step "$WARM_START"),
  "warm_start_rationale": "production warm started from a BEST checkpoint (il_method_one_v4_best.pt), so the matching artefact on the 0.60 arm is its best checkpoint, not its step 10000 final",
  "rloo_k": 20,
  "batch_size": 128,
  "b_times_k": 2560,
  "learning_rate": 0.0001,
  "gradient_clip_norm": 1.0,
  "epsilon_initial": 1.0,
  "epsilon_terminal": 0.5,
  "epsilon_anneal_steps": 1000,
  "num_workers": 12,
  "eval_every_steps": 25,
  "checkpoint_every_steps": 25,
  "target_steps": 1000,
  "split": "validation only, cache/training_set_il_v3/val, 200 instances",
  "test_split_evaluated": false,
  "gate_step": ${GATE_STEP},
  "gate_reference_replicate": ${GATE_REFERENCE},
  "production_own": ${PRODUCTION_OWN},
  "gate_value": "${GATE_VALUE}",
  "run_mode": "${RUN_MODE}",
  "resumed_from_step": ${EXISTING},
  "attempts_used": ${ATTEMPT},
  "rosetta_aborts": ${ABORT_COUNT},
  "started": "${STARTED}",
  "ended": "${ENDED}",
  "exit_code": ${STATUS},
  "final_checkpoint_step": ${FINAL_STEP},
  "checkpoint": "${CKPT}",
  "arm_log": "${ARM_LOG}",
  "status_log": "${STATUS_LOG}",
  "chain_log": "${CHAIN_LOG}"
}
JSON

log "Run metadata written to ${INFO}"
log "Complete. Exit ${EXIT_STATUS}."
log "=================================================================="
exit "$EXIT_STATUS"
