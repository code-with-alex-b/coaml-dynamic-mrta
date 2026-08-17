#!/usr/bin/env bash
# Method One architecture sweep. Appendix evidence on whether the imitation
# stage's result depends on encoder depth and width.
#
# Nine imitation-learning runs on a full 3x3 grid: message-passing layers 1/2/3
# by hidden dimension 32/64/128. Everything else is held at the production
# Method One configuration that produced checkpoints/il_method_one_v4_best.pt:
# lr 5e-4 cosine, batch 32, 30 Gumbel samples, epsilon annealed log-linearly
# 1.0 to 0.4 over 3000 steps and held there, 10000 steps, gradient clip 1.0
# (il_trainer.py has no --grad-clip flag; DEFAULT_MAX_GRAD_NORM is 1.0, so
# every config is clipped at the production value with no way to diverge), the
# v4 training cache and v3 validation cache, in-loop validation every 1000
# steps, and the same best-checkpoint rule (highest perturbed gap subject to a
# 95% serve-all floor). L2/H64 is the production architecture, so that cell is
# a replicate at a different RNG stream and the yardstick for the other eight.
#
# Reported metric: perturbed-decode gap closure versus the MILP bound on the
# 200 validation instances (seeds 11000-11199), an imitation-stage quality
# measure only. No run here is a candidate to replace any checkpoint.
#
# Estimated wall clock (batch 32, 30 samples): 0.087 s/step at L1/H32 rising
# to 0.115 s/step at L3/H128, plus 8-11 s per in-loop validation pass (ten of
# them). That is 16-21 minutes per run, roughly 2.7 hours for all nine.
#
# Crash handling: this x86_64-under-Rosetta-2 environment has produced
# sporadic assertion failures that killed an earlier sweep outright. Each
# config checkpoints every 1000 steps; a non-zero exit is relaunched once from
# that rolling checkpoint, and the sweep continues either way, so one crash
# costs at most 1000 steps of that config and never the remaining configs.
#
# Re-running is safe: a config already at the full step count is skipped, a
# config with a partial rolling checkpoint resumes from it.
#
# Usage:  bash scripts/experiments/dm_arch_sweep.sh              (all nine, sequentially)
#         bash scripts/experiments/dm_arch_sweep.sh 2 64         (one cell only)
#         DRY_RUN=1 bash scripts/experiments/dm_arch_sweep.sh    (print resolved commands)

set -uo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src"

STEPS=10000
BATCH=32
SAMPLES=30
LR=5e-4
EPS_START=1.0
EPS_END=0.40
ANNEAL_STEPS=3000
EVAL_EVERY=1000
CKPT_EVERY=1000
TRAIN_CACHE="cache/training_set_il_v4"
VAL_CACHE="cache/training_set_il_v3/val"
SUMMARY_CSV="results/dm_arch_sweep.csv"

LAYER_GRID=(1 2 3)
HIDDEN_GRID=(32 64 128)

# Checkpoints that must not be touched; every path this script writes is
# dm_arch_*, so this is a belt-and-braces guard hashing before/after.
PROTECTED=(
    "checkpoints/il_method_one_v4_best.pt"
    "checkpoints/sweep_G_v4warmstart_best.pt"
)

hash_protected () {
    for p in "${PROTECTED[@]}"; do
        if [ -e "$p" ]; then shasum -a 256 "$p"; fi
    done
}

# Step count stored inside a checkpoint, or -1 when it is missing or unreadable.
ckpt_step () {
    python - "$1" <<'PY' 2>/dev/null || echo -1
import sys, torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu",
                         weights_only=False)["step"]))
except Exception:
    print(-1)
PY
}

run_cell () {
    local layers="$1"
    local hidden="$2"
    local tag="L${layers}_H${hidden}"
    local ckpt="checkpoints/dm_arch_${tag}.pt"
    local log="logs/dm_arch_${tag}.log"

    local common=(
        --steps "${STEPS}"
        --batch-size "${BATCH}"
        --gumbel-samples "${SAMPLES}"
        --lr "${LR}"
        --lr-schedule cosine
        --epsilon-start "${EPS_START}"
        --epsilon-end "${EPS_END}"
        --epsilon-anneal-steps "${ANNEAL_STEPS}"
        --num-layers "${layers}"
        --hidden-dim "${hidden}"
        --cache-dir "${TRAIN_CACHE}"
        --val-cache-dir "${VAL_CACHE}"
        --eval-every "${EVAL_EVERY}"
        --checkpoint-every "${CKPT_EVERY}"
        --output-checkpoint "${ckpt}"
        --eval-csv "${SUMMARY_CSV}"
        --run-tag "${tag}"
        --log-every 500
        --log-file "${log}"
    )

    if [ -n "${DRY_RUN:-}" ]; then
        echo "--- ${tag} ---"
        echo "  python -u src/training/il_trainer.py ${common[*]}"
        return 0
    fi

    local done_step
    done_step="$(ckpt_step "${ckpt}")"
    if [ "${done_step}" = "${STEPS}" ]; then
        echo "=== ${tag}: already complete at step ${STEPS}, skipping ==="
        return 0
    fi

    echo "=== Arch sweep ${tag}: ${layers} layers, hidden ${hidden} ==="
    if [ "${done_step}" -gt 0 ] 2>/dev/null; then
        echo "    partial checkpoint at step ${done_step}, resuming"
        python -u src/training/il_trainer.py "${common[@]}" \
            --resume-from "${ckpt}"
    else
        python -u src/training/il_trainer.py "${common[@]}"
    fi
    local status=$?

    # One relaunch on a crash, resuming from the rolling checkpoint; the sweep
    # continues to the next cell whether or not the relaunch succeeds.
    if [ "${status}" -ne 0 ]; then
        echo "!!! ${tag} exited ${status}. Relaunching once from the rolling checkpoint." >&2
        local resume_step
        resume_step="$(ckpt_step "${ckpt}")"
        if [ "${resume_step}" -gt 0 ] 2>/dev/null; then
            echo "    resuming ${tag} from step ${resume_step}" >&2
            python -u src/training/il_trainer.py "${common[@]}" \
                --resume-from "${ckpt}"
            status=$?
        else
            echo "    no usable rolling checkpoint, restarting ${tag} from step 1" >&2
            python -u src/training/il_trainer.py "${common[@]}"
            status=$?
        fi
        if [ "${status}" -ne 0 ]; then
            echo "!!! ${tag} failed again (exit ${status}). Continuing with the next cell." >&2
            FAILED+=("${tag}")
        fi
    fi
    return 0
}

echo "Protected checkpoint hashes before the sweep:"
hash_protected
BEFORE="$(hash_protected)"

FAILED=()
if [ "$#" -eq 2 ]; then
    run_cell "$1" "$2"
else
    for layers in "${LAYER_GRID[@]}"; do
        for hidden in "${HIDDEN_GRID[@]}"; do
            run_cell "${layers}" "${hidden}"
        done
    done
fi

AFTER="$(hash_protected)"
if [ "${BEFORE}" != "${AFTER}" ]; then
    echo "!!! A protected checkpoint changed during the sweep." >&2
    exit 2
fi
echo "Protected checkpoints unchanged."

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Sweep finished with failed cells: ${FAILED[*]}" >&2
    exit 1
fi
echo "Sweep finished. Per-evaluation summary: ${SUMMARY_CSV}"
