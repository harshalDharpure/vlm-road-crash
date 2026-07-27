#!/usr/bin/env bash
# Full publication-ready research pipeline for Crash-1500
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/venv/bin/python" ]]; then
  export PATH="$ROOT/venv/bin:$PATH"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p results/logs results/tables results/checkpoints results/zero_shot results/ablation

LOG="results/logs/publication_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Publication Pipeline ==="
echo "ROOT=$ROOT"
date

# Auto-select freest GPU if not set
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU=$(python3 -c "from src.utils.gpu_manager import select_freest_gpu; g=select_freest_gpu(8); print(g if g is not None else 1)")
  export CUDA_VISIBLE_DEVICES="${GPU}"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

python3 - <<'PY'
import nltk
for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
PY

# Phase 1: Data (full dataset)
echo "--- Phase 1: Data processing ---"
python3 scripts/01_process_data.py

# Phase 2: Zero-shot pilot then full test eval
echo "--- Phase 2: Zero-shot evaluation ---"
python3 scripts/02_evaluate_zero_shot.py --split test --strategy every_5th --prompt structured_event

# Phase 3: Ablations (pilot mode for speed; remove --pilot for full)
echo "--- Phase 3: Ablation studies ---"
python3 scripts/06_run_ablations.py --pilot

# Phase 4: Fine-tuning
echo "--- Phase 4: Fine-tuning ---"
python3 scripts/03_finetune.py

# Phase 5: Finetuned eval
echo "--- Phase 5: Finetuned evaluation ---"
if [[ -f results/checkpoints/best_checkpoint.pt ]]; then
  python3 scripts/04_evaluate_finetuned.py --checkpoint results/checkpoints/best_checkpoint.pt --split test
fi

# Phase 6: Compare and export
echo "--- Phase 6: Comparison & publication outputs ---"
python3 scripts/05_compare_results.py \
  --zero_shot_metrics results/zero_shot/every_5th_structured_event_test/metrics.json \
  --finetuned_metrics results/finetuned/best_checkpoint/metrics.json \
  --output results/comparison_report.json 2>/dev/null || true

python3 scripts/export_results_tables.py 2>/dev/null || true
python3 scripts/07_generate_publication_outputs.py

echo "=== Pipeline complete ==="
date
echo "Log: $LOG"
