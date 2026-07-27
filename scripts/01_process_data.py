#!/usr/bin/env python3
"""Full Crash-1500 data processing: validate, annotate, split, extract frames."""
import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing import (
    DatasetSplitter,
    FrameSampler,
    GroundTruthParser,
    VideoValidator,
)
from src.utils import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-frames", action="store_true", help="Skip frame extraction")
    parser.add_argument("--strategy", default=None, help="Single sampling strategy only")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = get_config(args.config)
    root = Path(config.get("dataset.root_dir"))
    videos_dir = root / config.get("dataset.videos_dir")
    gt_file = root / config.get("dataset.ground_truth_file")
    processed_dir = root / config.get("dataset.processed_dir")

    print("=" * 70)
    print("Crash-1500 Data Processing Pipeline")
    print("=" * 70)
    print(f"Videos: {videos_dir}")
    print(f"Ground truth: {gt_file}")

    # Step 1: Validate videos
    print("\n[1/5] Validating videos...")
    validator = VideoValidator(videos_dir)
    validation = validator.collect_and_validate()
    print(f"  Valid: {len(validation['valid'])} | Corrupt: {len(validation['corrupt'])}")

    # Step 2: Parse all 11 Excel attributes
    print("\n[2/5] Parsing ground truth (all attributes)...")
    gt_parser = GroundTruthParser(str(gt_file))
    all_annotations = gt_parser.build_all_annotations()
    video_annotations = gt_parser.map_videos_to_annotations(validation["valid"])
    stats = gt_parser.get_statistics(video_annotations)
    print(f"  Matched with annotations: {stats.get('videos_with_annotations', 0)}")
    print(f"  Avg summary length: {stats.get('avg_summary_length', 0):.1f} words")

    matching = validator.match_annotations(validation["valid"], all_annotations)
    validator.save_report(processed_dir, validation, matching)

    annotations_file = processed_dir / "annotations.json"
    gt_parser.save_annotations(str(annotations_file), video_annotations)
    gt_parser.save_csv(str(processed_dir / "annotations.csv"), video_annotations)
    print(f"  Saved: {annotations_file}")

    # Step 3: Train/val/test split (70/15/15)
    print("\n[3/5] Creating splits...")
    splitter = DatasetSplitter(
        train_ratio=config.get("dataset.train_ratio", 0.70),
        val_ratio=config.get("dataset.val_ratio", 0.15),
        test_ratio=config.get("dataset.test_ratio", 0.15),
        random_seed=config.get("dataset.random_seed", 42),
    )
    valid_paths = [m["path"] for m in matching["matched"]]
    splits = splitter.split_videos(valid_paths)
    print(f"  Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

    split_info_file = processed_dir / "split_info.json"
    splitter.save_split_info(str(split_info_file), splits, video_annotations)

    for split_name in ("train", "val", "test"):
        split_anns = splitter.create_annotation_splits(video_annotations, splits)[split_name]
        out = processed_dir / f"annotations_{split_name}.json"
        gt_parser.save_annotations(str(out), split_anns)
        print(f"  Saved {split_name}: {len(split_anns)} samples -> {out}")

    # Step 4: Frame extraction (all strategies)
    if not args.skip_frames:
        print("\n[4/5] Extracting frames (multiprocessing)...")
        strategies_cfg = config.get("dataset.sampling_strategies", [])
        if args.strategy:
            strategies = [args.strategy]
        else:
            strategies = [s["name"] if isinstance(s, dict) else s for s in strategies_cfg]
            if not strategies:
                strategies = ["dense", "every_3rd", "every_5th", "every_10th"]

        resize = tuple(config.get("dataset.frame_resize", [336, 336]))
        sampler = FrameSampler(
            output_base=str(processed_dir / "frames"),
            strategies=strategies,
            segment_duration=config.get("dataset.segment_duration", 5),
            resize=resize,
            max_frames=config.get("model.max_frames", 30),
            num_workers=config.get("dataset.num_workers", 8),
        )
        frame_results = sampler.process_all_strategies(valid_paths)
        summary = {k: {"ok": len(v["ok"]), "skipped": len(v["skipped"]), "failed": len(v["failed"])} for k, v in frame_results.items()}
        with open(processed_dir / "frame_extraction_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Frame extraction summary: {summary}")
    else:
        print("\n[4/5] Skipping frame extraction (--skip-frames)")

    # Step 5: Dataset metadata manifest
    print("\n[5/5] Writing dataset metadata...")
    metadata = {
        "dataset": "Crash-1500",
        "total_videos": validation["total"],
        "valid_videos": len(validation["valid"]),
        "corrupt_videos": len(validation["corrupt"]),
        "matched_annotations": matching["n_matched"],
        "statistics": stats,
        "splits": {k: len(v) for k, v in splits.items()},
        "attributes": GroundTruthParser.ATTRIBUTE_COLUMNS,
    }
    with open(processed_dir / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("Data processing complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
