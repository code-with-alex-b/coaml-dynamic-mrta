#!/bin/bash
# A3 Phase 3. Pre-registered budget ladder, h=1, frozen test split, 200 instances.
# Budgets run ascending, each writing its own CSV and wall-clock entry. Nothing
# here re-solves or writes any cache record.
set -u
cd "$(dirname "$0")/../.."
# Activate coaml only if no environment is active; CONDA_EXE locates conda.sh without a hardcoded path.
if [ -z "${CONDA_PREFIX:-}" ] && [ -n "${CONDA_EXE:-}" ]; then
  . "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh"
  conda activate coaml
fi

TIMES=provenance/figure42_budget/a3_ladder_walltimes.txt
: > "$TIMES"
echo "ladder_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$TIMES"

for B in 0.010 0.025 0.050 0.100 0.250 1.0 10.0; do
    OUT="provenance/a3_rh_test_h1_b${B}s.csv"
    LOG="logs/a3_rh_test_h1_b${B}s.log"
    if [ -e "$OUT" ]; then
        echo "SKIP ${B}: $OUT already exists" >> "$TIMES"
        continue
    fi
    echo "=== budget ${B}s starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$TIMES"
    S=$(date +%s)
    PYTHONPATH="$PWD/src" python -u scripts/experiments/rolling_horizon_baseline.py \
        --horizon 1 \
        --cache-dir cache/training_set_il_v3/test \
        --allow-test-split \
        --eval-instances 200 \
        --time-limit "$B" \
        --eval-seed 0 \
        --per-instance-out "$OUT" \
        > "$LOG" 2>&1
    RC=$?
    E=$(date +%s)
    echo "budget=${B} rc=${RC} wall_seconds=$((E-S)) rows=$(( $(wc -l < "$OUT" 2>/dev/null || echo 1) - 1 ))" >> "$TIMES"
    echo "budget=${B} done rc=${RC} wall=$((E-S))s"
done
echo "ladder_end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$TIMES"
echo "LADDER COMPLETE"
