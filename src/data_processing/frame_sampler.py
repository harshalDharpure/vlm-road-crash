"""Multi-strategy frame extraction with caching and multiprocessing."""
from __future__ import annotations

import json
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


SAMPLING_STRATEGIES = {
    "dense": 1,
    "every_3rd": 3,
    "every_5th": 5,
    "every_10th": 10,
}


def _resize_frame(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    if size is None:
        return frame
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def extract_frames_from_video(
    video_path: str,
    strategy: str = "every_5th",
    segment_duration: int = 5,
    resize: Optional[Tuple[int, int]] = (336, 336),
    max_frames: int = 30,
) -> Tuple[List[np.ndarray], List[int], Dict]:
    interval = SAMPLING_STRATEGIES.get(strategy, 5)
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_segment_frames = int(segment_duration * fps)
    frames: List[np.ndarray] = []
    indices: List[int] = []
    frame_count = 0
    segment_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or segment_count >= max_segment_frames:
            break
        if frame_count % interval == 0:
            frames.append(_resize_frame(frame, resize))
            indices.append(frame_count)
            if len(frames) >= max_frames:
                break
        frame_count += 1
        segment_count += 1

    cap.release()
    meta = {
        "video_name": video_path.stem,
        "strategy": strategy,
        "frame_interval": interval,
        "num_frames": len(frames),
        "frame_indices": indices,
        "fps": fps,
    }
    return frames, indices, meta


def _process_one(args: Tuple) -> Dict:
    video_path, output_dir, strategy, segment_duration, resize, max_frames = args
    video_path = Path(video_path)
    out_dir = Path(output_dir) / strategy / video_path.stem
    marker = out_dir / ".done"
    if marker.exists():
        return {"status": "skipped", "video": video_path.stem, "strategy": strategy}

    try:
        frames, indices, meta = extract_frames_from_video(
            str(video_path), strategy, segment_duration, resize, max_frames
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (frame, idx) in enumerate(zip(frames, indices)):
            cv2.imwrite(str(out_dir / f"frame_{idx:05d}.jpg"), frame)
        meta["video_path"] = str(video_path)
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        marker.touch()
        return {"status": "ok", "video": video_path.stem, "strategy": strategy, "n": len(frames)}
    except Exception as e:
        return {"status": "error", "video": video_path.stem, "strategy": strategy, "error": str(e)}


class FrameSampler:
    """Extract and cache frames under data/processed/frames/{strategy}/."""

    def __init__(
        self,
        output_base: str,
        strategies: Optional[List[str]] = None,
        segment_duration: int = 5,
        resize: Tuple[int, int] = (336, 336),
        max_frames: int = 30,
        num_workers: int = 4,
    ):
        self.output_base = Path(output_base)
        self.strategies = strategies or list(SAMPLING_STRATEGIES.keys())
        self.segment_duration = segment_duration
        self.resize = resize
        self.max_frames = max_frames
        self.num_workers = num_workers

    def process_videos(self, video_paths: List[str], strategy: str) -> Dict:
        self.output_base.mkdir(parents=True, exist_ok=True)
        tasks = [
            (vp, self.output_base, strategy, self.segment_duration, self.resize, self.max_frames)
            for vp in video_paths
        ]
        results = {"ok": [], "skipped": [], "failed": []}
        worker_fn = _process_one
        if self.num_workers > 1:
            with mp.Pool(self.num_workers) as pool:
                for r in tqdm(pool.imap_unordered(worker_fn, tasks), total=len(tasks), desc=f"frames/{strategy}"):
                    self._collect(r, results)
        else:
            for t in tqdm(tasks, desc=f"frames/{strategy}"):
                self._collect(worker_fn(t), results)
        summary_path = self.output_base / strategy / "processing_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        return results

    def process_all_strategies(self, video_paths: List[str]) -> Dict[str, Dict]:
        all_results = {}
        for strategy in self.strategies:
            all_results[strategy] = self.process_videos(video_paths, strategy)
        return all_results

    @staticmethod
    def _collect(r: Dict, results: Dict) -> None:
        st = r.get("status", "error")
        if st == "ok":
            results["ok"].append(r)
        elif st == "skipped":
            results["skipped"].append(r)
        else:
            results["failed"].append(r)

    @staticmethod
    def load_cached_frames(video_stem: str, strategy: str, frames_root: Path) -> Tuple[List[np.ndarray], List[int]]:
        frame_dir = frames_root / strategy / video_stem
        if not frame_dir.exists():
            return [], []
        paths = sorted(frame_dir.glob("frame_*.jpg"))
        frames, indices = [], []
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
                idx = int(p.stem.split("_")[-1])
                indices.append(idx)
        return frames, indices
