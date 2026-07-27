#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/venv/bin:$PATH"

LOG="results/logs/full_test_ablations_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Full-test ablations (N=226 each) ==="
date

# Run full grid on the freest GPUs available.
python3 scripts/10_run_full_test_ablations.py --min-free-gb 12

echo "=== Done ==="
date
echo "Log: $LOG"

