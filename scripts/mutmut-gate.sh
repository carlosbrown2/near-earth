#!/usr/bin/env bash
# mutmut-gate.sh — Mutation testing gate for the Ralph loop.
# Runs mutmut on a given module and fails if survival rate exceeds threshold.
#
# Usage: scripts/mutmut-gate.sh <module_path> [max_survival_pct]
#   module_path       — Python file to mutate (e.g. prospector/scoring/scorer.py)
#   max_survival_pct  — Maximum allowed survival rate (default: 10)
#
# Exit codes:
#   0 — Gate passed (survival rate within threshold)
#   1 — Gate failed (too many surviving mutants)
#   2 — Usage error or mutmut not installed

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <module_path> [max_survival_pct]"
    echo "  module_path      — Python file to mutate (e.g. prospector/scoring/scorer.py)"
    echo "  max_survival_pct — Maximum allowed survival rate, default 10"
    exit 2
fi

MODULE="$1"
MAX_SURVIVAL="${2:-10}"

if [[ ! -f "$MODULE" ]]; then
    echo "Error: Module not found: $MODULE"
    exit 2
fi

if ! command -v mutmut &>/dev/null; then
    echo "Error: mutmut is not installed. Run: pip install mutmut"
    exit 2
fi

echo "=== Mutation Testing Gate ==="
echo "Module:          $MODULE"
echo "Max survival:    ${MAX_SURVIVAL}%"
echo ""

# Clean previous results
mutmut run --paths-to-mutate="$MODULE" --runner="python -m pytest -x --tb=no -q" --no-progress 2>&1 || true

# Parse results from mutmut
RESULTS=$(mutmut results 2>&1 || true)

# Count mutants by status
KILLED=$(echo "$RESULTS" | grep -c "^Killed" || true)
SURVIVED=$(echo "$RESULTS" | grep -c "^Survived" || true)
TIMEOUT=$(echo "$RESULTS" | grep -c "^Timeout" || true)
SUSPICIOUS=$(echo "$RESULTS" | grep -c "^Suspicious" || true)
SKIPPED=$(echo "$RESULTS" | grep -c "^Skipped" || true)

TOTAL=$((KILLED + SURVIVED + TIMEOUT + SUSPICIOUS + SKIPPED))

if [[ "$TOTAL" -eq 0 ]]; then
    echo "No mutants generated for $MODULE — gate passes (nothing to test)."
    exit 0
fi

# Timeouts and suspicious count as caught (not survived)
CAUGHT=$((KILLED + TIMEOUT + SUSPICIOUS))
SURVIVAL_PCT=$(( (SURVIVED * 100) / TOTAL ))

echo ""
echo "=== Results ==="
echo "Total mutants:   $TOTAL"
echo "Killed:          $KILLED"
echo "Survived:        $SURVIVED"
echo "Timeout:         $TIMEOUT"
echo "Suspicious:      $SUSPICIOUS"
echo "Skipped:         $SKIPPED"
echo "Survival rate:   ${SURVIVAL_PCT}%"
echo ""

if [[ "$SURVIVAL_PCT" -gt "$MAX_SURVIVAL" ]]; then
    echo "FAILED: Survival rate ${SURVIVAL_PCT}% exceeds threshold ${MAX_SURVIVAL}%"
    echo ""
    echo "Surviving mutants:"
    mutmut results 2>&1 | grep "^Survived" || true
    exit 1
else
    echo "PASSED: Survival rate ${SURVIVAL_PCT}% is within threshold ${MAX_SURVIVAL}%"
    exit 0
fi
