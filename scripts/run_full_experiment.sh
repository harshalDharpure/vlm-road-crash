#!/usr/bin/env bash
# Full research pipeline per RESEARCH_PLAN.md:
#   01 data -> 02 zero-shot -> 03 fine-tune -> 04 eval finetuned -> 05 compare -> export tables
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=1          # pick a free GPU
#   export VLM_DATA_ROOT=/path/to/dataset    # folder containing videos/ and Excel (optional if under project root)
#   nohup bash scripts/run_full_experiment.sh > results/logs/nohup_console.log 2>&1 &
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -x "$ROOT/venv/bin/python" ]]; then
  export PATH="$ROOT/venv/bin:$PATH"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p results/logs results/tables results/checkpoints results/zero_shot

LOG="results/logs/full_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Full experiment ==="
echo "ROOT=$ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "VLM_DATA_ROOT=${VLM_DATA_ROOT:-<unset, using config.yaml root_dir>}"
echo "Log: $LOG"
date

DATA_ROOT="${VLM_DATA_ROOT:-$ROOT}"
VIDEOS_DIR="${DATA_ROOT}/video1500"
if [[ ! -d "$VIDEOS_DIR" ]]; then
  VIDEOS_DIR="${DATA_ROOT}/videos"
fi
if [[ ! -d "$VIDEOS_DIR" ]] || [[ -z "$(find "$VIDEOS_DIR" -maxdepth 1 -name '*.mp4' -print -quit 2>/dev/null)" ]]; then
  echo "ERROR: No MP4 files under $VIDEOS_DIR (expected video1500/ or videos/)"
  exit 1
fi

# Ensure required NLTK assets exist (prevents BLEU tokenization failures mid-run)
python3 - <<'PY'
import nltk
for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
print("NLTK assets ensured")
PY

python3 scripts/01_process_data.py
python3 scripts/02_evaluate_zero_shot.py
python3 scripts/03_finetune.py
python3 scripts/04_evaluate_finetuned.py \
  --checkpoint results/checkpoints/best_checkpoint.pt \
  --split test
python3 scripts/05_compare_results.py \
  --zero_shot_metrics results/zero_shot/metrics.json \
  --finetuned_metrics results/finetuned/best_checkpoint/metrics.json \
  --output results/comparison_report.json
python3 scripts/export_results_tables.py

echo "=== Done ==="
date
