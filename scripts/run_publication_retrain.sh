#!/usr/bin/env bash
# Publication-quality retraining: multi-frame collage, early stopping, lower LR
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU=$(python3 -c "from src.utils.gpu_manager import select_freest_gpu; g=select_freest_gpu(12); print(g if g is not None else 1)")
export CUDA_VISIBLE_DEVICES="${GPU}"

LOG="results/logs/publication_retrain_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Publication retraining on GPU $CUDA_VISIBLE_DEVICES ==="
date
echo "Settings: LR=1e-5, max_epochs=3, early_stop=1, collage=4 frames, structured_event prompt"

# Save old checkpoints
if [ -d results/checkpoints ] && [ ! -d results/checkpoints_v1 ]; then
  cp -a results/checkpoints results/checkpoints_v1
  echo "Backed up old checkpoints to results/checkpoints_v1"
fi

python3 scripts/03_finetune.py

python3 scripts/04_evaluate_finetuned.py \
  --checkpoint results/checkpoints/best_checkpoint.pt \
  --split test \
  --use-collage

python3 scripts/05_compare_results.py \
  --zero_shot_metrics results/zero_shot/every_5th_structured_event_test/metrics.json \
  --finetuned_metrics results/finetuned/best_checkpoint_collage/metrics.json \
  --output results/comparison_report_publication.json

python3 scripts/07_generate_publication_outputs.py

echo "=== Publication retraining complete ==="
date
echo "Log: $LOG"
