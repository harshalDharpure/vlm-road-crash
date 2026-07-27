# Efficient Vision-Language Video Summarization for Road Crash Analysis

Publication-ready research pipeline using **Crash-1500**, sparse temporal sampling, LLaVA-NeXT, LoRA fine-tuning, and NLI-based semantic evaluation.

## Dataset

| Item | Path |
|------|------|
| Videos (1500) | `video1500/*.mp4` |
| Ground truth (11 attributes) | `Car_Crash_Text_Dataset_ground_truth.xlsx` |
| Processed metadata | `data/processed/` |

### Excel attributes
`Video Number`, `Severity of the Crash`, `Type of Vehicles involved`, `No. of Vehicles involved`, `Location of impact`, `Start/End of Crash`, `Explanation`, `Ambiguity`, `Camera View`, `Weather Conditions`

### Splits (seed=42)
- Train: 1048 | Val: 224 | Test: 226

## Quick start

```bash
cd /DATA/vaneet_2221cs15/vlm-road-crash
source venv/bin/activate
pip install -r requirements.txt

# 1. Data: validate, annotate, split, extract frames
python scripts/01_process_data.py

# 2. Zero-shot evaluation
export CUDA_VISIBLE_DEVICES=1
python scripts/02_evaluate_zero_shot.py --split test --strategy every_5th --prompt structured_event

# 3. Fine-tune (LoRA)
python scripts/03_finetune.py

# 4. Evaluate fine-tuned model
python scripts/04_evaluate_finetuned.py --checkpoint results/checkpoints/best_checkpoint.pt

# 5. Ablation studies (pilot: 25 samples)
python scripts/06_run_ablations.py --pilot

# 6. Publication outputs (tables, figures, LaTeX, paper sections)
python scripts/07_generate_publication_outputs.py
```

### Full automated pipeline

```bash
nohup bash scripts/run_publication_pipeline.sh > results/logs/nohup_publication.log 2>&1 &
```

## Architecture

```
video1500/ + Excel
    → 01_process_data (validate, 11-field GT, 70/15/15 split, frame cache)
    → 02_evaluate_zero_shot (LLaVA-NeXT + metrics)
    → 03_finetune (QLoRA)
    → 04_evaluate_finetuned
    → 06_run_ablations
    → 07_generate_publication_outputs
```

## Frame sampling strategies
- `dense` (every frame)
- `every_3rd`, `every_5th`, `every_10th`
- Cached under `data/processed/frames/{strategy}/{video_id}/`

## Prompt strategies
`basic_caption`, `temporal_sequence`, `safety_critical`, `chain_of_thought`, `structured_event`

## Metrics
BLEU-1–4, ROUGE-L, METEOR, BERTScore, CIDEr, SPICE, **NLI entailment** (RoBERTa-large-MNLI)

## Outputs

| Artifact | Location |
|----------|----------|
| Experiment tables | `results/experiment_tables.md` |
| Comparison JSON | `results/comparison_report.json` |
| Figures | `results/publication_figures/` |
| LaTeX tables | `results/latex_tables/` |
| Ablation CSV | `results/ablation/ablation_tables.csv` |
| Error analysis | `results/qualitative_examples/` |
| Human eval sheets | `results/qualitative_examples/*/human_eval_sheet.csv` |
| Paper sections | `results/paper_support/` |

## GPU management
Auto-selects freest GPU when `gpu.auto_select: true` in `config/config.yaml`. Override with `export CUDA_VISIBLE_DEVICES=N`.

## Configuration
Edit `config/config.yaml` for paths, LoRA ranks, learning rates, ablation grids, and experiment flags.
