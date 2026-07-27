#!/usr/bin/env python3
"""Systematic ablation studies: frame sampling, prompts, LoRA rank."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_config


def run_cmd(cmd: list) -> int:
    print(">>", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Use max 25 test samples per run")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = get_config(args.config)
    cfg = config.config
    ab = cfg.get("ablation", {})
    max_samples = cfg.get("experiment", {}).get("pilot_samples", 25) if args.pilot else None
    py = sys.executable
    root = project_root
    results = []

    for strategy in ab.get("frame_strategies", ["every_5th"]):
        for prompt in ab.get("prompt_strategies", ["structured_event"]):
            cmd = [
                py, str(root / "scripts/02_evaluate_zero_shot.py"),
                "--split", "test",
                "--strategy", strategy,
                "--prompt", prompt,
            ]
            if max_samples:
                cmd.extend(["--max-samples", str(max_samples)])
            rc = run_cmd(cmd)
            metrics_path = (
                Path(cfg["paths"]["zero_shot"]) / f"{strategy}_{prompt}_test" / "metrics.json"
            )
            if metrics_path.exists():
                with open(metrics_path) as f:
                    m = json.load(f)
                m["strategy"] = strategy
                m["prompt"] = prompt
                results.append(m)

    out_dir = Path(cfg["paths"].get("ablation", "results/ablation"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Heatmap data: strategy x rouge_l
    heat = {}
    for r in results:
        key = f"{r.get('strategy')}|{r.get('prompt')}"
        heat[key] = {
            "rouge_l": r.get("rouge_l", 0),
            "bertscore": r.get("bertscore", 0),
            "nli_entail": r.get("nli", {}).get("entailment_accuracy", 0),
        }
    with open(out_dir / "ablation_tables.csv", "w") as f:
        f.write("config,rouge_l,bertscore,nli_entail\n")
        for k, v in heat.items():
            f.write(f"{k},{v['rouge_l']},{v['bertscore']},{v['nli_entail']}\n")

    print(f"Ablation complete. {len(results)} runs -> {out_dir}")


if __name__ == "__main__":
    main()
