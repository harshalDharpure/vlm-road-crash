#!/usr/bin/env bash
# Run complete Crash-1500 experiments in background on the freest GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/venv/bin/python" ]]; then
  export PATH="$ROOT/venv/bin:$PATH"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p results/logs results/tables results/checkpoints results/zero_shot results/ablation results/finetuned

LOG="results/logs/complete_experiments_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo "Complete experiments (background)"
echo "ROOT=$ROOT"
echo "Started: $(date)"
echo "=========================================="

# Pick freest GPU with at least 12GB free
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU=$(python3 -c "
from src.utils.gpu_manager import get_gpu_memory, select_freest_gpu
g = select_freest_gpu(12.0)
if g is None:
    g = 3
for info in get_gpu_memory():
    print(f'  GPU {info[\"index\"]}: {info[\"memory_free_mb\"]/1024:.1f}GB free', flush=True)
print(g)
" | tail -1)
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python3 -c "from src.utils.gpu_manager import get_gpu_memory; [print(f'  GPU {g[\"index\"]}: {g[\"memory_free_mb\"]/1024:.1f}GB free') for g in get_gpu_memory()]"

# NLTK assets
python3 - <<'PY'
import nltk
for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
print("NLTK ready")
PY

ZS_METRICS="results/zero_shot/every_5th_structured_event_test/metrics.json"
PROCESSED="data/processed/split_info.json"

# Phase 1: Data (skip if splits exist; skip frame re-extract if every_5th done)
if [[ ! -f "$PROCESSED" ]]; then
  echo "--- Phase 1: Data processing ---"
  python3 scripts/01_process_data.py --skip-frames
else
  echo "--- Phase 1: Skipped (split_info.json exists) ---"
  if [[ ! -d "data/processed/frames/every_5th" ]] || [[ -z "$(ls -A data/processed/frames/every_5th 2>/dev/null | head -1)" ]]; then
    echo "Extracting every_5th frames only..."
    python3 scripts/01_process_data.py --strategy every_5th
  fi
fi

# Phase 2: Zero-shot (wait if another job is writing the same output)
echo "--- Phase 2: Zero-shot evaluation (test split, 226 videos) ---"
if [[ -f "$ZS_METRICS" ]]; then
  echo "Zero-shot metrics already exist: $ZS_METRICS"
else
  # Wait up to 24h for an external zero-shot job to finish
  WAIT_SEC=0
  MAX_WAIT=$((24 * 3600))
  while [[ ! -f "$ZS_METRICS" ]] && pgrep -f "02_evaluate_zero_shot.py.*structured_event" >/dev/null 2>&1; do
    if (( WAIT_SEC % 300 == 0 )); then
      echo "Waiting for in-flight zero-shot job... (${WAIT_SEC}s)"
      tail -3 results/logs/zero_shot_full_test.log 2>/dev/null || true
    fi
    sleep 60
    WAIT_SEC=$((WAIT_SEC + 60))
    if (( WAIT_SEC > MAX_WAIT )); then
      echo "Timeout waiting; starting zero-shot on this GPU."
      break
    fi
  done
  if [[ ! -f "$ZS_METRICS" ]]; then
    python3 scripts/02_evaluate_zero_shot.py \
      --split test \
      --strategy every_5th \
      --prompt structured_event
  fi
fi

# Phase 3: Ablation grid (pilot=25 samples per config for tractability)
echo "--- Phase 3: Ablation studies ---"
python3 scripts/06_run_ablations.py --pilot || echo "Ablation phase had errors (continuing)"

# Phase 4: LoRA fine-tuning
echo "--- Phase 4: Fine-tuning (LoRA) ---"
python3 scripts/03_finetune.py

# Phase 5: Finetuned evaluation
echo "--- Phase 5: Finetuned evaluation ---"
if [[ -f results/checkpoints/best_checkpoint.pt ]]; then
  python3 scripts/04_evaluate_finetuned.py \
    --checkpoint results/checkpoints/best_checkpoint.pt \
    --split test
else
  echo "No best_checkpoint.pt found; skipping finetuned eval"
fi

# Phase 6: Reports and publication artifacts
echo "--- Phase 6: Comparison & publication outputs ---"
python3 scripts/05_compare_results.py \
  --zero_shot_metrics "$ZS_METRICS" \
  --finetuned_metrics results/finetuned/best_checkpoint/metrics.json \
  --output results/comparison_report.json 2>/dev/null || true

python3 scripts/export_results_tables.py || true
python3 scripts/07_generate_publication_outputs.py

echo "=========================================="
echo "Complete experiments finished: $(date)"
echo "Log: $LOG"
echo "=========================================="
