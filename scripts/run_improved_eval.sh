#!/usr/bin/env bash
# Re-evaluate with multi-frame collage + best val checkpoint (epoch 2)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU=$(python3 -c "from src.utils.gpu_manager import select_freest_gpu; g=select_freest_gpu(12); print(g if g is not None else 3)")
export CUDA_VISIBLE_DEVICES="${GPU}"

LOG="results/logs/improved_eval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Improved evaluation on GPU $CUDA_VISIBLE_DEVICES ==="
date

# 1. Restore full zero-shot metrics from original 226-sample run log
python3 scripts/08_recompute_metrics.py --restore-zero-shot --recompute-finetuned

# 2. Re-eval epoch-2 (best validation loss) with multi-frame collage
python3 scripts/04_evaluate_finetuned.py \
  --checkpoint results/checkpoints/checkpoint_epoch_2.pt \
  --split test \
  --use-collage

# 3. Re-run zero-shot full test with multi-frame collage
python3 scripts/02_evaluate_zero_shot.py \
  --split test \
  --strategy every_5th \
  --prompt structured_event \
  --use-collage

# 4. Regenerate comparison tables
python3 scripts/05_compare_results.py \
  --zero_shot_metrics results/zero_shot/every_5th_structured_event_test_collage/metrics.json \
  --finetuned_metrics results/finetuned/checkpoint_epoch_2_collage/metrics.json \
  --output results/comparison_report_improved.json || true

python3 scripts/07_generate_publication_outputs.py

echo "=== Improved evaluation complete ==="
date
echo "Log: $LOG"
