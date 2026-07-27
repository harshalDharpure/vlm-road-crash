#!/usr/bin/env python3
"""Zero-shot evaluation with full metric suite and prompt/sampling options."""
import argparse
import json
import sys
import time
from pathlib import Path

import nltk
import torch
from tqdm import tqdm

for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.frame_sampler import FrameSampler
from src.evaluation.metrics_suite import MetricsSuite
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
    frames, indices, _ = extract_frames_from_video(video_path, strategy=strategy, max_frames=max_frames)
    return frames, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--strategy", default=None, help="Frame sampling strategy")
    parser.add_argument("--prompt", default=None, help="Prompt strategy")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--use-collage", action="store_true", default=False)
    parser.add_argument("--no-collage", action="store_false", dest="use_collage")
    parser.add_argument("--num-key-frames", type=int, default=4)
    parser.add_argument(
        "--output-name",
        default=None,
        help="Override output folder name under results/zero_shot",
    )
    parser.add_argument(
        "--fail-if-exists",
        action="store_true",
        help="Fail if output directory already exists (protect canonical runs)",
    )
    args = parser.parse_args()

    config = get_config(args.config)
    cfg = config.config
    setup_gpu_from_config(cfg)

    root = Path(cfg["dataset"]["root_dir"])
    processed_dir = root / cfg["dataset"]["processed_dir"]
    strategy = args.strategy or cfg["dataset"].get("default_sampling", "every_5th")
    prompt = args.prompt or cfg["prompts"].get("default_strategy", "structured_event")
    frames_root = processed_dir / "frames"
    suffix = "_collage" if args.use_collage else ""
    default_name = f"{strategy}_{prompt}_{args.split}{suffix}"
    out_name = args.output_name or default_name
    out_dir = Path(cfg["paths"]["zero_shot"]) / out_name
    if args.fail_if_exists and out_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {out_dir}. "
            "Use --output-name for a new run directory."
        )

    with open(processed_dir / "split_info.json") as f:
        split_info = json.load(f)
    with open(processed_dir / f"annotations_{args.split}.json") as f:
        annotations = json.load(f)

    video_paths = split_info["splits"][args.split]
    if args.max_samples:
        video_paths = video_paths[: args.max_samples]

    print("=" * 60)
    print(f"ZERO-SHOT | split={args.split} | strategy={strategy} | prompt={prompt}")
    print(f"Multi-frame collage: {args.use_collage} (key frames={args.num_key_frames})")
    print(f"Videos: {len(video_paths)}")
    print("=" * 60)

    vlm = UnifiedVLM(model_name=cfg["model"].get("primary", "llava-next"), config=cfg)
    metrics_suite = MetricsSuite(cfg)

    predictions, references, results = [], [], []
    runtimes = []

    for video_path in tqdm(video_paths, desc="Inference"):
        ann_key = annotation_key_from_path(video_path)
        if ann_key not in annotations:
            continue
        gt = annotations[ann_key].get("text_summary", "") or annotations[ann_key].get("explanation", "")
        if not gt.strip():
            continue

        frames, indices = load_frames(video_path, strategy, frames_root, cfg["model"]["max_frames"])
        if not frames:
            continue

        t0 = time.perf_counter()
        try:
            out = vlm.generate_summary(
                frames,
                prompt_strategy=prompt,
                frame_indices=indices,
                use_collage=args.use_collage,
                num_key_frames=args.num_key_frames,
            )
            pred = out.get("text_summary", "").strip()
        except Exception as e:
            print(f"Error {ann_key}: {e}")
            continue
        runtimes.append(time.perf_counter() - t0)

        if pred:
            predictions.append(pred)
            references.append(gt)
            results.append({
                "video_id": ann_key,
                "prediction": pred,
                "reference": gt,
                "strategy": strategy,
                "prompt": prompt,
            })

    if not predictions:
        print("No predictions generated.")
        return

    print("\nComputing metrics...")
    metrics = metrics_suite.compute_all(predictions, references)
    metrics["runtime_sec_mean"] = sum(runtimes) / len(runtimes) if runtimes else 0
    metrics["strategy"] = strategy
    metrics["prompt"] = prompt
    metrics["split"] = args.split
    metrics["use_collage"] = args.use_collage
    metrics["num_key_frames"] = args.num_key_frames if args.use_collage else 1

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(
            {
                "script": "scripts/02_evaluate_zero_shot.py",
                "split": args.split,
                "strategy": strategy,
                "prompt": prompt,
                "use_collage": args.use_collage,
                "num_key_frames": args.num_key_frames if args.use_collage else 1,
                "output_name": out_name,
            },
            f,
            indent=2,
        )

    flat = metrics_suite.flatten_for_table(metrics)
    print("\nResults:", json.dumps(flat, indent=2))
    print(f"Saved to {out_dir}")
    log_vram("Post-eval ")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
