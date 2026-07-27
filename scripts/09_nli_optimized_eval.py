#!/usr/bin/env python3
"""Zero-shot evaluation optimized for high NLI faithfulness."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.frame_sampler import FrameSampler
from src.evaluation.faithfulness import filter_sentences_by_nli, postprocess_prediction
from src.evaluation.metrics_suite import MetricsSuite
from src.evaluation.nli_evaluator import NLIEvaluator
from src.models.unified_vlm import UnifiedVLM
from src.utils import get_config
from src.utils.gpu_manager import log_vram, setup_gpu_from_config
from src.utils.video_ids import annotation_key_from_path


def load_frames(video_path: str, strategy: str, frames_root: Path, max_frames: int):
    stem = Path(video_path).stem
    cached, indices = FrameSampler.load_cached_frames(stem, strategy, frames_root)
    if cached:
        return cached[:max_frames], indices[:max_frames]
    from src.data_processing.frame_sampler import extract_frames_from_video

    frames, indices, _ = extract_frames_from_video(
        video_path, strategy=strategy, max_frames=max_frames
    )
    return frames, indices


def main():
    parser = argparse.ArgumentParser(description="NLI-optimized zero-shot evaluation")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--strategy", default="every_5th")
    parser.add_argument("--prompt", default="faithfulness")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--postprocess", action="store_true", default=True)
    parser.add_argument("--no-postprocess", action="store_false", dest="postprocess")
    parser.add_argument("--sentence-filter", action="store_true", default=True)
    parser.add_argument("--no-sentence-filter", action="store_false", dest="sentence_filter")
    parser.add_argument("--compare-prompts", action="store_true", help="Try faithfulness + structured_event on val subset")
    args = parser.parse_args()

    config = get_config(args.config)
    cfg = config.config
    setup_gpu_from_config(cfg)

    root = Path(cfg["dataset"]["root_dir"])
    processed_dir = root / cfg["dataset"]["processed_dir"]
    frames_root = processed_dir / "frames"

    with open(processed_dir / "split_info.json") as f:
        split_info = json.load(f)
    with open(processed_dir / f"annotations_{args.split}.json") as f:
        annotations = json.load(f)

    video_paths = split_info["splits"][args.split]
    if args.max_samples:
        video_paths = video_paths[: args.max_samples]

    prompts = [args.prompt]
    if args.compare_prompts:
        prompts = ["faithfulness", "structured_event"]

    vlm = UnifiedVLM(model_name=cfg["model"].get("primary", "llava-next"), config=cfg)

    best_run = None

    for prompt in prompts:
        print("=" * 60)
        print(f"NLI-OPT | split={args.split} | strategy={args.strategy} | prompt={prompt}")
        print(f"postprocess={args.postprocess} | sentence_filter={args.sentence_filter}")
        print(f"single middle frame (no collage)")
        print("=" * 60)

        raw_items = []
        runtimes = []

        for video_path in tqdm(video_paths, desc=f"Inference [{prompt}]"):
            ann_key = annotation_key_from_path(video_path)
            if ann_key not in annotations:
                continue
            gt = annotations[ann_key].get("text_summary", "") or annotations[ann_key].get("explanation", "")
            if not gt.strip():
                continue

            frames, indices = load_frames(
                video_path, args.strategy, frames_root, cfg["model"]["max_frames"]
            )
            if not frames:
                continue

            t0 = time.perf_counter()
            try:
                out = vlm.generate_summary(
                    frames,
                    prompt_strategy=prompt,
                    frame_indices=indices,
                    use_collage=False,
                    num_key_frames=1,
                )
                pred = out.get("text_summary", "").strip()
            except Exception as e:
                print(f"Error {ann_key}: {e}")
                continue
            runtimes.append(time.perf_counter() - t0)

            if pred:
                raw_items.append({"video_id": ann_key, "prediction_raw": pred, "reference": gt})

        print(f"\nInference done: {len(raw_items)} samples. Post-processing...")
        nli_filter = None
        if args.sentence_filter:
            nli_cfg = cfg.get("evaluation", {}).get("nli", {})
            nli_filter = NLIEvaluator(
                model_name=nli_cfg.get("model_name", "roberta-large-mnli"),
                device=nli_cfg.get("device", "cpu"),
                batch_size=nli_cfg.get("batch_size", 16),
            )

        predictions, references, results = [], [], []
        for item in tqdm(raw_items, desc="Post-process"):
            pred = item["prediction_raw"]
            gt = item["reference"]
            if args.postprocess:
                pred = postprocess_prediction(pred)
            if args.sentence_filter and nli_filter:
                pred = filter_sentences_by_nli(pred, gt, nli_filter)
            if pred:
                predictions.append(pred)
                references.append(gt)
                results.append({
                    "video_id": item["video_id"],
                    "prediction": pred,
                    "prediction_raw": item["prediction_raw"],
                    "reference": gt,
                    "strategy": args.strategy,
                    "prompt": prompt,
                })

        if not predictions:
            print(f"No predictions for prompt={prompt}")
            continue

        print("\nComputing metrics...")
        metrics_suite = MetricsSuite(cfg)
        metrics = metrics_suite.compute_all(predictions, references)
        metrics["runtime_sec_mean"] = sum(runtimes) / len(runtimes) if runtimes else 0
        metrics["strategy"] = args.strategy
        metrics["prompt"] = prompt
        metrics["split"] = args.split
        metrics["postprocess"] = args.postprocess
        metrics["sentence_filter"] = args.sentence_filter
        metrics["use_collage"] = False

        tag = f"{args.strategy}_{prompt}_{args.split}_nli_opt"
        out_dir = Path(cfg["paths"]["results"]) / "nli_optimized" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "detailed_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        flat = metrics_suite.flatten_for_table(metrics)
        nli_acc = flat.get("nli_entailment_acc", 0)
        print("\nResults:", json.dumps(flat, indent=2))
        print(f"Saved to {out_dir}")

        if best_run is None or nli_acc > best_run["nli"]:
            best_run = {"prompt": prompt, "nli": nli_acc, "dir": str(out_dir), "flat": flat}

    if best_run:
        summary_path = Path(cfg["paths"]["results"]) / "nli_optimized" / "best_run.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(best_run, f, indent=2)
        print("\n" + "=" * 60)
        print(f"BEST NLI: {best_run['nli']:.4f} (prompt={best_run['prompt']})")
        print(f"Output: {best_run['dir']}")
        print("=" * 60)

    log_vram("Post-eval ")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
