#!/usr/bin/env bash
# One command status for the overnight chain. Read-only: it starts nothing,
# kills nothing and writes nothing.
#
# Usage:  bash scripts/analysis/overnight_status.sh

set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src"
# Interpreter from the active environment rather than a hardcoded path.
# Set PY to override. CONDA_PREFIX is used when an environment is active,
# which is the point of the note above: a bare `python` in a detached shell
# resolves to whatever that shell inherited.
PY="${PY:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
: "${PY:=$(command -v python3 || command -v python)}"

RUN_DATE="${1:-$(date +%Y%m%d)}"
CHAIN_LOG="logs/overnight_chain_${RUN_DATE}.log"

rule () { printf '\n==================== %s ====================\n\n' "$1"; }

rule "PROCESSES"
if pgrep -f 'overnight_chain.sh' >/dev/null 2>&1; then
    echo "Chain is RUNNING, PID(s): $(pgrep -f 'overnight_chain.sh' | tr '\n' ' ')"
else
    echo "Chain is not running."
fi
if pgrep -f 'dm_arch_sweep.sh' >/dev/null 2>&1; then
    echo "Architecture sweep is RUNNING, PID(s): $(pgrep -f 'dm_arch_sweep.sh' | tr '\n' ' ')"
else
    echo "Architecture sweep is not running."
fi
echo "Live rollout workers: $(pgrep -f 'multiprocessing.spawn' 2>/dev/null | wc -l | tr -d ' ')"

rule "CHAIN LOG, STAGE BOUNDARIES"
if [ -e "$CHAIN_LOG" ]; then
    grep -E 'STAGE [0-9]+ (START|END)|PROTECTED|failed|WATCHDOG|Overnight chain' \
        "$CHAIN_LOG" || echo "(no stage lines yet)"
else
    echo "No chain log at ${CHAIN_LOG}"
fi

rule "CHAIN LOG, LAST 30 LINES"
[ -e "$CHAIN_LOG" ] && tail -30 "$CHAIN_LOG"

rule "STAGE 1, METHOD ONE ARCHITECTURE SWEEP"
if [ -e "results/overnight_stage1_arch_report_${RUN_DATE}.txt" ]; then
    cat "results/overnight_stage1_arch_report_${RUN_DATE}.txt"
else
    echo "Stage 1 report not written yet. Live view from the CSV:"
    "$PY" scripts/analysis/arch_sweep_report.py 2>&1 || true
fi

rule "STAGE 2, METHOD ONE OPTIMISATION SWEEP, EIGHT ARMS"
if [ -e results/overnight_stage2_method_one.csv ]; then
    "$PY" scripts/analysis/arch_sweep_report.py \
        --csv results/overnight_stage2_method_one.csv 2>&1 || true
else
    echo "No stage 2 evaluations yet."
fi

rule "STAGES 3 TO 5, METHOD TWO"
"$PY" scripts/analysis/overnight_m2_report.py \
    checkpoints/overnight_s3_archwinner.pt \
    checkpoints/overnight_s4_eps035.pt \
    checkpoints/overnight_s5_k40_b64.pt 2>&1 || true

rule "PROTECTED CHECKPOINTS NOW"
for p in checkpoints/sweep_G_v4warmstart_best.pt \
         checkpoints/il_method_one_v4_best.pt; do
    [ -e "$p" ] && shasum -a 256 "$p" || echo "MISSING  $p"
done
echo
echo "Compare these against the BEFORE and AFTER hashes in ${CHAIN_LOG}."
