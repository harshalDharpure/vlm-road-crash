#!/usr/bin/env python3
"""Recompute metrics from saved predictions; restore zero-shot metrics from full run log."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics_suite import MetricsSuite
from src.utils import get_config


def load_pred_ref(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    preds = [r["prediction"] for r in data]
    refs = [r["reference"] for r in data]
    return preds, refs, data


def restore_zero_shot_from_log(log_path: Path, out_dir: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = list(re.finditer(r'"num_samples":\s*226', text))
    if not matches:
        raise ValueError(f"No 226-sample metrics JSON found in {log_path}")
    start = matches[-1].start()
    # Walk back to opening brace
    brace = text.rfind("{", 0, start)
    if brace < 0:
        brace = start - 1
    depth = 0
    end = brace
    for i, ch in enumerate(text[brace:], brace):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    flat = json.loads(text[brace:end])
    metrics = {
        "num_samples": flat["num_samples"],
        "bleu": {k: flat[k] for k in flat if k.startswith("bleu_")},
        "meteor": flat.get("meteor"),
        "rouge_1": flat.get("rouge_1"),
        "rouge_2": flat.get("rouge_2"),
        "rouge_l": flat.get("rouge_l"),
        "bertscore": flat.get("bertscore"),
        "cider": flat.get("cider", 0.0),
        "nli": {
            "entailment_accuracy": flat.get("nli_entailment_acc", 0),
            "avg_entailment_prob": flat.get("nli_avg_entailment_prob", 0),
            "total_samples": flat["num_samples"],
        },
        "strategy": "every_5th",
        "prompt": "structured_event",
        "split": "test",
        "restored_from_log": str(log_path),
        "note": "Restored from full 226-sample zero-shot run (pre-ablation overwrite)",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_full_226.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-zero-shot", action="store_true")
    parser.add_argument("--recompute-finetuned", action="store_true")
    parser.add_argument("--recompute-all-detailed", action="store_true")
    args = parser.parse_args()

    config = get_config()
    cfg = config.config
    suite = MetricsSuite(cfg)
    root = Path(cfg["dataset"]["root_dir"])

    if args.restore_zero_shot:
        log_path = root / "results/logs/zero_shot_full_test.log"
        out_dir = Path(cfg["paths"]["zero_shot"]) / "every_5th_structured_event_test"
        metrics = restore_zero_shot_from_log(log_path, out_dir)
        print(f"Restored zero-shot metrics (n={metrics.get('num_samples')}) -> {out_dir}")

    if args.recompute_finetuned:
        for name in ("best_checkpoint", "checkpoint_epoch_2"):
            det = root / "results/finetuned" / name / "detailed_results.json"
            if not det.exists():
                print(f"Skip missing {det}")
                continue
            preds, refs, _ = load_pred_ref(det)
            metrics = suite.compute_all(preds, refs)
            metrics["checkpoint"] = name
            metrics["split"] = "test"
            metrics["note"] = "recomputed with corpus CIDEr"
            out = det.parent / "metrics_recomputed.json"
            out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            flat = suite.flatten_for_table(metrics)
            print(f"{name} (n={len(preds)}): BLEU-1={flat.get('bleu_1',0):.4f} "
                  f"ROUGE-L={flat.get('rouge_l',0):.4f} CIDEr={flat.get('cider',0):.4f} "
                  f"NLI={flat.get('nli_entailment_acc',0):.4f}")

    if args.recompute_all_detailed:
        zs_root = Path(cfg["paths"]["zero_shot"])
        for det in zs_root.glob("*/detailed_results.json"):
            preds, refs, _ = load_pred_ref(det)
            if len(preds) < 10:
                continue
            metrics = suite.compute_all(preds, refs)
            (det.parent / "metrics_recomputed.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
            print(f"Recomputed {det.parent.name} n={len(preds)}")


if __name__ == "__main__":
    main()
