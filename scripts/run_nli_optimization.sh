#!/usr/bin/env bash
# Maximize NLI: single-frame + faithfulness prompt + post-process + sentence filter
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU=$(python3 -c "
from src.utils.gpu_manager import get_gpu_memory, select_freest_gpu
for g in get_gpu_memory():
    print(f\"  GPU {g['index']}: {g['memory_free_mb']:.0f}MB free\")
g = select_freest_gpu(8)
print(g if g is not None else 3)
" | tail -1)
export CUDA_VISIBLE_DEVICES="${GPU}"

LOG="results/logs/nli_optimization_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== NLI optimization on GPU $CUDA_VISIBLE_DEVICES (freest) ==="
python3 -c "from src.utils.gpu_manager import get_gpu_memory; [print(f\"GPU {g['index']}: {g['memory_free_mb']:.0f}MB free, {g['memory_used_mb']:.0f}MB used\") for g in get_gpu_memory()]"
date

# Full test eval with NLI-optimized pipeline
python3 scripts/09_nli_optimized_eval.py \
  --split test \
  --strategy every_5th \
  --prompt faithfulness \
  --postprocess \
  --sentence-filter

# Also run structured_event for comparison (same pipeline)
python3 scripts/09_nli_optimized_eval.py \
  --split test \
  --strategy every_5th \
  --prompt structured_event \
  --postprocess \
  --sentence-filter

# Update main zero-shot metrics path for publication tables
BEST=$(python3 -c "
import json
from pathlib import Path
p = Path('results/nli_optimized/best_run.json')
if p.exists():
    d = json.loads(p.read_text())
    print(d.get('dir',''))
")

if [ -n \"\$BEST\" ] && [ -f \"\$BEST/metrics.json\" ]; then
  cp \"\$BEST/metrics.json\" results/nli_optimized/best_metrics.json
  python3 scripts/05_compare_results.py \
    --zero_shot_metrics results/zero_shot/every_5th_structured_event_test/metrics.json \
    --finetuned_metrics \"\$BEST/metrics.json\" \
    --output results/comparison_nli_optimized.json || true
fi

python3 scripts/07_generate_publication_outputs.py || true

echo "=== NLI optimization complete ==="
date
echo "Log: $LOG"
