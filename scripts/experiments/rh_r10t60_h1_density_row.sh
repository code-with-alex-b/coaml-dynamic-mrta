#!/bin/bash
# Rolling-horizon h=1 at the density scale (R=10, T=60), for the fourth row of
# the zero-shot transfer table. Budget 1.0s per window solve, matching every
# other scale in the table. h=5 is deliberately not run: the stored
# rolling_horizon_h5 trajectory in this cache is a lookahead oracle/hybrid, so
# it is not an online baseline.
#
# The cache path contains no "test" component, so the runner's test-split
# guard is not engaged and --allow-test-split is not passed; the test split
# is not read.
#
# Safe to stop and restart: rows are flushed one per completed instance, so
# rerunning picks up from the last finished instance and skips everything
# already recorded, costing at most the one instance in flight.
set -u
cd "$(dirname "$0")/../.."
# Activate coaml only if no environment is active; CONDA_EXE locates conda.sh without a hardcoded path.
if [ -z "${CONDA_PREFIX:-}" ] && [ -n "${CONDA_EXE:-}" ]; then
  . "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh"
  conda activate coaml
fi

STAMP=20260804
OUT="provenance/rh_r10t60_h1_b1.0s_${STAMP}.csv"
LOG="logs/rh_r10t60_h1_b1.0s_${STAMP}.log"
TIMES="provenance/rh_r10t60_h1_b1.0s_${STAMP}_walltime.txt"

mkdir -p logs provenance

RESUME=""
if [ -e "$OUT" ]; then
    RESUME="--resume"
    echo "resume_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) existing_rows=$(( $(wc -l < "$OUT") - 1 ))" >> "$TIMES"
else
    {
        echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "cache=cache/training_set_rh_r10t60 seeds=90000-90199 n=200"
        echo "horizon=1 budget_seconds=1.0 gurobi_seed=0"
    } > "$TIMES"
fi

S=$(date +%s)
PYTHONPATH="$PWD/src" python -u scripts/experiments/rolling_horizon_baseline.py \
    --horizon 1 \
    --cache-dir cache/training_set_rh_r10t60 \
    --eval-instances 200 \
    --time-limit 1.0 \
    --eval-seed 0 \
    --per-instance-out "$OUT" \
    $RESUME \
    >> "$LOG" 2>&1
RC=$?
E=$(date +%s)

{
    echo "segment_end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "rc=${RC} segment_wall_seconds=$((E-S)) total_rows=$(( $(wc -l < "$OUT" 2>/dev/null || echo 1) - 1 ))"
} >> "$TIMES"
echo "RH DENSITY ROW SEGMENT DONE rc=${RC} wall=$((E-S))s rows=$(( $(wc -l < "$OUT" 2>/dev/null || echo 1) - 1 ))"
