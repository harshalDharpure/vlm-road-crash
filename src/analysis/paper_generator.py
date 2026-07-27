"""Auto-generate paper-ready methodology and results sections."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


def generate_methodology(config: Dict) -> str:
    ds = config.get("dataset", {})
    return f"""## Methodology

We propose an efficient vision-language pipeline for road crash video summarization on the Crash-1500 dataset ({ds.get('train_ratio', 0.7)*100:.0f}/{ds.get('val_ratio', 0.15)*100:.0f}/{ds.get('test_ratio', 0.15)*100:.0f} train/validation/test split). Videos are processed with sparse temporal frame sampling (dense, every-3rd, every-5th, every-10th) within the first {ds.get('segment_duration', 5)} seconds. Frames are resized to {ds.get('frame_resize', [336,336])} and fed to an instruction-tuned multimodal model (LLaVA-NeXT) with structured prompting.

Fine-tuning uses LoRA/QLoRA (rank ∈ {{4,8,16,32}}) with mixed precision and gradient checkpointing. Semantic faithfulness is evaluated using NLI entailment (RoBERTa-large-MNLI) alongside BLEU, ROUGE-L, METEOR, BERTScore, CIDEr, and SPICE.
"""


def generate_experimental_setup(config: Dict) -> str:
    tr = config.get("training", {})
    return f"""## Experimental Setup

- **Model**: {config.get('model', {}).get('vision_model', 'LLaVA-NeXT')}
- **Epochs**: {tr.get('num_epochs', 5)}
- **Learning rates**: {tr.get('learning_rates', [2e-5])}
- **LoRA ranks**: {tr.get('lora_ranks', [8])}
- **Batch size**: {config.get('model', {}).get('batch_size', 1)} (with gradient accumulation)
- **Evaluation**: Zero-shot and fine-tuned; bootstrap 95% CIs over seeds where applicable.
"""


def generate_results_section(metrics: Dict) -> str:
    flat = metrics if "bleu_1" in metrics else metrics.get("zero_shot", metrics)
    lines = ["## Results\n", "| Metric | Score |", "|--------|-------|"]
    for k in ["bleu_1", "bleu_4", "rouge_l", "meteor", "bertscore", "cider", "spice"]:
        if k in flat:
            lines.append(f"| {k.upper()} | {flat[k]:.4f} |")
    nli = flat.get("nli", {})
    if nli:
        lines.append(f"| NLI Entailment Acc | {nli.get('entailment_accuracy', 0):.4f} |")
    return "\n".join(lines)


def generate_figure_captions() -> str:
    return """## Figure Captions

**Fig. 1.** Metric comparison across frame sampling strategies and prompting variants on the Crash-1500 test set.

**Fig. 2.** Ablation heatmap for LoRA rank, temporal sampling, and number of input frames.

**Fig. 3.** Qualitative failure cases: hallucinations, missing events, and temporal inconsistencies.
"""


def write_paper_support(output_dir: Path, config: Dict, metrics: Optional[Dict] = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sections = {
        "methodology.md": generate_methodology(config),
        "experimental_setup.md": generate_experimental_setup(config),
        "figure_captions.md": generate_figure_captions(),
    }
    if metrics:
        sections["results.md"] = generate_results_section(metrics)
    for name, content in sections.items():
        (output_dir / name).write_text(content, encoding="utf-8")
    with open(output_dir / "paper_bundle.json", "w") as f:
        json.dump({"config_snapshot": config, "metrics": metrics}, f, indent=2)
