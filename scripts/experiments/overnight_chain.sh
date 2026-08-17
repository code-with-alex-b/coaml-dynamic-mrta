#!/usr/bin/env bash
# Overnight chain. Five stages, strictly sequential, one worker pool at a time.
#
#   STAGE 1  Wait for the running Method One architecture sweep, then report it.
#   STAGE 2  Method One optimisation sweep, eight arms, one axis each.
#   STAGE 3  Method Two warm started from the architecture sweep winner.
#   STAGE 4  Method Two, epsilon endpoint arm.
#   STAGE 5  Method Two, K against batch swap at fixed B*K.
#
# Nothing here touches the architecture sweep, its checkpoints or its logs;
# the sweep is waited on, never signalled.
#
# Failure policy: a stage that exits non-zero is relaunched once, and the
# chain continues to the next stage regardless -- no failure halts the chain.
# Every stage runs under a watchdog at roughly twice its expected runtime so
# a hung stage cannot eat the window; this machine has no timeout(1), so the
# watchdog is implemented here.
#
# Between stages every orphaned rollout worker is reaped. A reparented worker
# pool holding a pipe open is what stalled an earlier chain. il_trainer uses
# no multiprocessing, so reaping on the multiprocessing markers cannot touch
# the architecture sweep even if it were somehow still alive.
#
# The chain writes only to new paths. checkpoints/sweep_G_v4warmstart_best.pt
# and checkpoints/il_method_one_v4_best.pt are hashed at start and end, both
# hashes go to the log, and a change fails loudly.
#
# Validation split only; no stage evaluates the test split. This script
# commits nothing and modifies no cache record.
#
# Usage:  nohup caffeinate -i bash scripts/experiments/overnight_chain.sh \
#             > logs/overnight_nohup.log 2>&1 &

set -uo pipefail

cd "$(dirname "$0")/../.."
REPO="$PWD"
export PYTHONPATH="$REPO/src"

# Interpreter from the active environment (a bare `python` in a detached
# shell resolves to whatever that shell inherited, not necessarily coaml).
# Set PY to override.
PY="${PY:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
: "${PY:=$(command -v python3 || command -v python)}"

RUN_DATE="$(date +%Y%m%d)"
CHAIN_LOG="logs/overnight_chain_${RUN_DATE}.log"
mkdir -p logs results checkpoints

# SMOKE=1 replaces every training body with a stub and shortens the waits, so
# the scaffolding (logging, hashing, watchdog, orphan reaping, winner
# selection, failure/relaunch accounting) can be exercised end to end without
# a single training step or checkpoint write. This machine runs bash 3.2,
# unforgiving about empty array expansion under `set -u`, so the scaffolding
# is worth proving before a multi-hour detached run depends on it.
SMOKE="${SMOKE:-}"
if [ -n "$SMOKE" ]; then
    CHAIN_LOG="${SMOKE_LOG:-/tmp/overnight_chain_smoke.log}"
    : > "$CHAIN_LOG"
fi

PROTECTED=(
    "checkpoints/sweep_G_v4warmstart_best.pt"
    "checkpoints/il_method_one_v4_best.pt"
)

# ---------------------------------------------------------------- logging ---

log () {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$CHAIN_LOG"
}

stage_start () {
    log "STAGE ${1} START  ${2}"
}

stage_end () {
    log "STAGE ${1} END    exit=${2}  ${3}"
}

# Run a command with a wall-clock limit; returns its exit status, or 124 on
# timeout (matching timeout(1)'s convention). The command is backgrounded
# rather than piped so $! is the command itself and the kill lands on it.
#
# A stage body is a shell function, so $! is the subshell, not the python
# process inside it. Killing the subshell alone would leave python running
# and holding the machine for the next stage -- exactly the stall this chain
# is meant to avoid -- so the watchdog kills the whole descendant tree.

descendants () {
    local parent="$1" kid
    for kid in $(pgrep -P "$parent" 2>/dev/null || true); do
        descendants "$kid"
        echo "$kid"
    done
}

kill_tree () {
    local root="$1" sig="$2" p
    # Deepest first, so a parent cannot respawn a child between signals.
    for p in $(descendants "$root") "$root"; do
        kill "-${sig}" "$p" 2>/dev/null || true
    done
}

run_with_limit () {
    local limit="$1"; shift
    local waited=0
    "$@" &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        if [ "$waited" -ge "$limit" ]; then
            log "  WATCHDOG: ${limit}s limit reached, terminating PID ${pid} and its tree."
            kill_tree "$pid" TERM
            sleep 30
            kill_tree "$pid" KILL
            wait "$pid" 2>/dev/null
            return 124
        fi
        sleep 15
        waited=$((waited + 15))
    done
    wait "$pid"
    return $?
}

# Kill rollout workers left behind by a stage. Matches only the
# multiprocessing spawn markers, which method_two_trainer's pool produces and
# il_trainer never does, so this can never hit the architecture sweep.

reap_workers () {
    local label="$1"
    local protected_pids="" sweep_pid p pids keep

    # Belt and braces: never signal the architecture sweep or anything below
    # it, even though il_trainer cannot produce a process matching the pattern.
    sweep_pid="$(pgrep -f 'dm_arch_sweep.sh' | head -1 || true)"
    if [ -n "$sweep_pid" ]; then
        protected_pids="$sweep_pid $(descendants "$sweep_pid" | tr '\n' ' ')"
        log "  reap after ${label}: sweep tree protected (${protected_pids})"
    fi

    pids=""
    for p in $(pgrep -f 'multiprocessing.spawn|multiprocessing.resource_tracker' 2>/dev/null || true); do
        keep=0
        for q in $protected_pids; do
            [ "$p" = "$q" ] && keep=1
        done
        [ "$keep" -eq 0 ] && pids="$pids $p"
    done

    if [ -z "${pids// /}" ]; then
        log "  reap after ${label}: no orphaned workers."
        return 0
    fi
    log "  reap after ${label}: terminating orphaned workers${pids}"
    kill -TERM $pids 2>/dev/null || true
    sleep 10
    for p in $pids; do
        kill -0 "$p" 2>/dev/null && {
            log "  reap after ${label}: forcing ${p}"
            kill -KILL "$p" 2>/dev/null || true
        }
    done
    sleep 5
    log "  reap after ${label}: done."
}

hash_protected () {
    local p
    for p in "${PROTECTED[@]}"; do
        if [ -e "$p" ]; then
            shasum -a 256 "$p"
        else
            echo "MISSING  $p"
        fi
    done
}

# Run one stage under the watchdog, relaunch once on failure, then return
# regardless. $1 stage id, $2 limit seconds, $3 description, rest the command.

run_stage () {
    local id="$1"; shift
    local limit="$1"; shift
    local desc="$1"; shift

    stage_start "$id" "$desc (watchdog ${limit}s)"
    run_with_limit "$limit" "$@"
    local status=$?

    if [ "$status" -ne 0 ]; then
        log "  STAGE ${id} exited ${status}. Relaunching once."
        run_with_limit "$limit" "$@"
        local retry=$?
        if [ "$retry" -ne 0 ]; then
            log "  STAGE ${id} failed again (exit ${retry}). Continuing to the next stage."
            FAILED+=("stage${id}")
        else
            log "  STAGE ${id} succeeded on the relaunch."
        fi
        status=$retry
    fi

    reap_workers "stage ${id}"
    stage_end "$id" "$status" "$desc"
    return 0
}

log "=================================================================="
log "Overnight chain starting. PID $$. Repo ${REPO}."
log "Interpreter ${PY}"
log "PYTHONPATH ${PYTHONPATH}"
log "Validation split only. No stage evaluates the test split."
log "Protected checkpoint hashes BEFORE the chain:"
hash_protected | while read -r line; do log "  ${line}"; done
BEFORE_HASHES="$(hash_protected)"

FAILED=()

# STAGE 1: wait for the architecture sweep by polling for its exit. No
# duration is assumed and no signal is ever sent to it. The cap is a floor
# under the whole chain, not an expectation: four cells at roughly twenty
# minutes each is about eighty minutes from the amendment time.

STAGE1_WAIT_LIMIT=$((6 * 3600))
[ -n "$SMOKE" ] && STAGE1_WAIT_LIMIT=1

stage_start 1 "wait for the architecture sweep, then report it"

SWEEP_PID="$(pgrep -f 'dm_arch_sweep.sh' | head -1 || true)"
if [ -z "$SWEEP_PID" ]; then
    log "  No architecture sweep process found. It has already finished."
else
    log "  Architecture sweep is PID ${SWEEP_PID}. Polling for its exit."
    waited=0
    while kill -0 "$SWEEP_PID" 2>/dev/null; do
        if [ "$waited" -ge "$STAGE1_WAIT_LIMIT" ]; then
            log "  Sweep still running after ${waited}s. NOT killing it."
            log "  Reporting on the evaluations recorded so far and moving on."
            break
        fi
        sleep 60
        waited=$((waited + 60))
        if [ $((waited % 1800)) -eq 0 ]; then
            log "  still waiting on the sweep, ${waited}s elapsed"
        fi
    done
    if ! kill -0 "$SWEEP_PID" 2>/dev/null; then
        log "  Architecture sweep PID ${SWEEP_PID} has exited after ${waited}s."
    fi
fi

STAGE1_REPORT="results/overnight_stage1_arch_report_${RUN_DATE}.txt"
[ -n "$SMOKE" ] && STAGE1_REPORT="/tmp/smoke_stage1_report.txt"
"$PY" scripts/analysis/arch_sweep_report.py \
    --csv results/dm_arch_sweep.csv \
    --log-dir logs > "$STAGE1_REPORT" 2>&1
stage1_status=$?
log "  Stage 1 report written to ${STAGE1_REPORT}"
stage_end 1 "$stage1_status" "architecture sweep report"

# STAGE 2: Method One optimisation sweep. Eight arms, each anchored at the
# production Method One configuration (lr 5e-4 cosine, batch 32, 30 Gumbel
# samples, epsilon annealed 1.0 to 0.40 over 3000 steps, 10000 steps,
# gradient clip 1.0 -- the il_trainer default, no flag exists -- 2 layers,
# hidden 64) and varying exactly one axis. No arm chains onto another arm's
# winner.
#
#   lr5e5       learning rate one order below production
#   lr5e4       learning rate at production (replicate, the noise yardstick)
#   lr5e3       learning rate one order above production
#   eps060      epsilon floor 0.60
#   eps080      epsilon floor 0.80
#   epsconst10  constant epsilon 1.0, the no-anneal control
#   m15         15 Gumbel samples, below production
#   m60         60 Gumbel samples, above production
#
# The epsilon floor arms both probe upward because downward is already
# measured: study B4 ran floor 0.25 at otherwise identical settings and lost
# 3.88 points. The constant 1.0 control is the limit of the upward direction.
#
# Expected runtime is roughly 20 minutes per arm at 30 samples (measured off
# the architecture sweep's own cells; m60 nearer 35), about 2.9 hours total.
# The watchdog is set to 6 hours rather than twice 2.4 because a truncation
# here loses whole arms.

STAGE2_LIMIT=$((6 * 3600))
STAGE2_CSV="results/overnight_stage2_method_one.csv"

# Step count inside a checkpoint, or -1 when missing or unreadable. Lets a
# relaunch of the whole stage skip arms that already finished instead of
# re-running them.
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

stage2_arm () {
    local tag="$1" lr="$2" eps_start="$3" eps_end="$4" samples="$5"
    local ckpt="checkpoints/overnight_s2_${tag}.pt"
    local log_file="logs/overnight_s2_${tag}.log"
    # Left unset rather than set to (): this machine runs bash 3.2, where
    # expanding an empty array under `set -u` is fatal, so every array here
    # is expanded with the ${a[@]+"${a[@]}"} guard. The explicit unset is
    # load bearing: in bash 3.2 a bare `local resume` leaves the name
    # set-but-empty as far as that guard is concerned, so it expands to one
    # empty string instead of nothing, and il_trainer dies on `unrecognized
    # arguments:` before training a step -- exactly how all eight arms
    # failed on the 20260801 run.
    local resume; unset resume
    local done_step

    done_step="$(ckpt_step "$ckpt")"
    if [ "$done_step" = "10000" ]; then
        log "  arm ${tag}: already complete at step 10000, skipping"
        return 0
    fi
    # Resume a partial arm so a relaunch costs at most 1000 steps.
    if [ "$done_step" -gt 0 ] 2>/dev/null; then
        log "  arm ${tag}: partial checkpoint at step ${done_step}, resuming"
        resume=(--resume-from "$ckpt")
    fi

    log "  arm ${tag}: lr=${lr} epsilon ${eps_start}->${eps_end} samples=${samples}"
    "$PY" -u src/training/il_trainer.py \
        --steps 10000 \
        --batch-size 32 \
        --gumbel-samples "${samples}" \
        --lr "${lr}" \
        --lr-schedule cosine \
        --epsilon-start "${eps_start}" \
        --epsilon-end "${eps_end}" \
        --epsilon-anneal-steps 3000 \
        --num-layers 2 \
        --hidden-dim 64 \
        --cache-dir cache/training_set_il_v4 \
        --val-cache-dir cache/training_set_il_v3/val \
        --eval-every 1000 \
        --checkpoint-every 1000 \
        --output-checkpoint "${ckpt}" \
        --eval-csv "${STAGE2_CSV}" \
        --run-tag "${tag}" \
        --log-every 500 \
        --log-file "${log_file}" \
        ${resume[@]+"${resume[@]}"} >> "${log_file%.log}.out" 2>&1
}

stage2_all () {
    local rc=0
    stage2_arm lr5e5      5e-5  1.0 0.40 30 || rc=1
    stage2_arm lr5e4      5e-4  1.0 0.40 30 || rc=1
    stage2_arm lr5e3      5e-3  1.0 0.40 30 || rc=1
    stage2_arm eps060     5e-4  1.0 0.60 30 || rc=1
    stage2_arm eps080     5e-4  1.0 0.80 30 || rc=1
    stage2_arm epsconst10 5e-4  1.0 1.00 30 || rc=1
    stage2_arm m15        5e-4  1.0 0.40 15 || rc=1
    stage2_arm m60        5e-4  1.0 0.40 60 || rc=1
    return $rc
}

if [ -n "$SMOKE" ]; then
    # Stub: succeeds immediately, trains nothing.
    stage2_all () { log "  SMOKE stub: stage 2 body, no training"; return 0; }
    STAGE2_LIMIT=60
fi

run_stage 2 "$STAGE2_LIMIT" "Method One optimisation sweep, eight arms" stage2_all

# STAGE 3 warm start selection. Method Two has no architecture of its own; it
# inherits the shape of its warm start, so this is the only stage that tests
# the architecture lever at all.
#
# The winner is the cell with the highest mean over its final three
# evaluations, not its best single evaluation. If no cell beats production's
# +6.7%, the warm start falls back to il_method_one_v4_best.pt and the run is
# recorded as a production replicate rather than a candidate -- still a
# second sample at production settings, comparable against 0.3881 at step 400.

log "Selecting the architecture sweep winner for the stage 3 warm start."
WINNER_OUT="$("$PY" scripts/analysis/arch_sweep_winner.py \
    --csv results/dm_arch_sweep.csv 2>&1)"
log "  winner selection output:"
echo "$WINNER_OUT" | while read -r line; do log "    ${line}"; done

WINNER_CELL="none"
WINNER_CKPT="checkpoints/il_method_one_v4_best.pt"
WINNER_ROLE="production_replicate"
# Only consume the well-formed assignment lines, so a traceback cannot be
# evaluated as shell.
eval "$(echo "$WINNER_OUT" | grep -E '^WINNER_(CELL|CKPT|MEAN|ROLE)=[^ ]*$' || true)"

log "  stage 3 warm start: ${WINNER_CKPT}"
log "  stage 3 role: ${WINNER_ROLE} (winning cell ${WINNER_CELL})"
if [ ! -e "$WINNER_CKPT" ]; then
    log "  WARNING: ${WINNER_CKPT} does not exist. Falling back to production."
    WINNER_CKPT="checkpoints/il_method_one_v4_best.pt"
    WINNER_ROLE="production_replicate"
fi

# Method Two stage bodies. Every stage anchors on the G_v4warmstart entry in
# sweep/configs.json, which pins the production settings: K 20, batch 128,
# B*K 2560, lr 1e-4, gradient clip 1.0, epsilon 1.0 to 0.5, 12 workers, the
# v3 train cache and the 200 instance v3 validation cache.
#
# --epsilon-anneal-steps 1000 is not a deviation. The entry leaves the span
# unset, stretching it over the whole run, so a 400 step run would reach
# epsilon 0.5 at step 400 whereas production was at 0.7582 there. Pinning the
# span to 1000 reproduces production's epsilon step for step, making the
# step 400 comparison equal budget AND equal perturbation.
#
# --eval-every 25 puts a validation pass and a rolling checkpoint on the same
# 25 step cadence; the trainer writes eval_history into that rolling
# checkpoint, so the whole trajectory survives without the stdout log.
#
# train_one.py resumes automatically when the checkpoint path already
# exists, making every one of these stages resumable at a cost of at most 25
# steps on relaunch.

M2_WORKERS=12
M2_STEPS=400
M2_ANNEAL=1000
M2_EVAL_EVERY=25
# Expected 7 hours each, so the watchdog sits at twice that.
M2_LIMIT=$((14 * 3600))

S3_CKPT="checkpoints/overnight_s3_archwinner.pt"
S4_CKPT="checkpoints/overnight_s4_eps035.pt"
S5_CKPT="checkpoints/overnight_s5_k40_b64.pt"

stage3_body () {
    "$PY" -u sweep/train_one.py \
        --config-id G_v4warmstart \
        --num-workers "$M2_WORKERS" \
        --max-steps "$M2_STEPS" \
        --epsilon-anneal-steps "$M2_ANNEAL" \
        --eval-every "$M2_EVAL_EVERY" \
        --warm-start-checkpoint "$WINNER_CKPT" \
        --checkpoint-path "$S3_CKPT" \
        >> logs/overnight_s3_archwinner.log 2>&1
}

# STAGE 4. Epsilon endpoint arm at 0.35 (from "v4-eps035" in
# sweep/configs.json, the epsilon arm of the v3 sweep); production is 0.5,
# pinned by the G_v4warmstart entry in the same file.
stage4_body () {
    "$PY" -u sweep/train_one.py \
        --config-id G_v4warmstart \
        --num-workers "$M2_WORKERS" \
        --max-steps "$M2_STEPS" \
        --epsilon-end 0.35 \
        --epsilon-anneal-steps "$M2_ANNEAL" \
        --eval-every "$M2_EVAL_EVERY" \
        --warm-start-checkpoint checkpoints/il_method_one_v4_best.pt \
        --checkpoint-path "$S4_CKPT" \
        >> logs/overnight_s4_eps035.log 2>&1
}

# STAGE 5. K against batch swap at fixed B*K 2560: production is K 20 batch
# 128, this arm is K 40 batch 64, same 2560 rollouts per step.
#
# Why up in K and not down: in sweep/configs.json the halve-K-double-batch
# move at fixed B*K is already run -- v4-B (K 20 batch 64) and v4-C (K 10
# batch 128), both B*K 1280, landed at 0.3792 and 0.3812 at step 300, a 0.2
# point difference inside noise. K above 20 appears nowhere in that file, so
# K 40 is the only untested direction on this contour.
stage5_body () {
    "$PY" -u sweep/train_one.py \
        --config-id G_v4warmstart \
        --num-workers "$M2_WORKERS" \
        --max-steps "$M2_STEPS" \
        --rloo-k 40 \
        --batch-size 64 \
        --epsilon-anneal-steps "$M2_ANNEAL" \
        --eval-every "$M2_EVAL_EVERY" \
        --warm-start-checkpoint checkpoints/il_method_one_v4_best.pt \
        --checkpoint-path "$S5_CKPT" \
        >> logs/overnight_s5_k40_b64.log 2>&1
}

if [ -n "$SMOKE" ]; then
    # Stubs exercising all three outcomes: fail then succeed on the relaunch,
    # hit the watchdog, and fail twice. None of them trains.
    M2_LIMIT=45
    rm -f /tmp/smoke_stage3_attempts
    stage3_body () {
        local n=0
        [ -e /tmp/smoke_stage3_attempts ] && n="$(cat /tmp/smoke_stage3_attempts)"
        n=$((n + 1)); echo "$n" > /tmp/smoke_stage3_attempts
        log "  SMOKE stub: stage 3 attempt ${n}"
        [ "$n" -ge 2 ] && return 0
        return 7
    }
    # Spawns a real nested python child so the watchdog's tree kill is proven.
    stage4_body () {
        log "  SMOKE stub: stage 4 hangs, expect the watchdog"
        "$PY" -c "import time; time.sleep(600)"
    }
    stage5_body () { log "  SMOKE stub: stage 5 always fails"; return 9; }
fi

run_stage 3 "$M2_LIMIT" \
    "Method Two from the architecture sweep winner (${WINNER_ROLE}, warm start ${WINNER_CKPT})" \
    stage3_body

run_stage 4 "$M2_LIMIT" \
    "Method Two epsilon endpoint 0.35 (production 0.5, from sweep/configs.json v4-eps035)" \
    stage4_body

run_stage 5 "$M2_LIMIT" \
    "Method Two K 40 batch 64 at B*K 2560 (production K 20 batch 128)" \
    stage5_body

log "All stages finished. Writing the Method Two summary."
M2_SUMMARY="results/overnight_method_two_summary_${RUN_DATE}.txt"
[ -n "$SMOKE" ] && M2_SUMMARY="/tmp/smoke_m2_summary.txt"
"$PY" scripts/analysis/overnight_m2_report.py \
    "$S3_CKPT" "$S4_CKPT" "$S5_CKPT" > "$M2_SUMMARY" 2>&1
log "  Method Two summary written to ${M2_SUMMARY}"

log "Protected checkpoint hashes AFTER the chain:"
hash_protected | while read -r line; do log "  ${line}"; done
AFTER_HASHES="$(hash_protected)"

CHAIN_STATUS=0
if [ "$BEFORE_HASHES" != "$AFTER_HASHES" ]; then
    log "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    log "!!! A PROTECTED CHECKPOINT CHANGED DURING THE CHAIN."
    log "!!! Before:"
    echo "$BEFORE_HASHES" | while read -r line; do log "!!!   ${line}"; done
    log "!!! After:"
    echo "$AFTER_HASHES" | while read -r line; do log "!!!   ${line}"; done
    log "!!! Treat every result from this chain as suspect."
    log "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    CHAIN_STATUS=2
else
    log "Protected checkpoints unchanged."
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
    log "Stages that failed twice: ${FAILED[*]}"
    [ "$CHAIN_STATUS" -eq 0 ] && CHAIN_STATUS=1
fi

log "Overnight chain complete. Exit ${CHAIN_STATUS}."
log "Status command: bash scripts/analysis/overnight_status.sh"
log "=================================================================="
exit "$CHAIN_STATUS"
