"""Validate videos and match against annotations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from tqdm import tqdm

from src.utils.video_ids import extract_video_id


def validate_video(path: Path) -> Tuple[bool, str, Dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, "cannot_open", {}
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, _ = cap.read()
    cap.release()
    if not ret or frame_count <= 0:
        return False, "empty_or_corrupt", {}
    return True, "ok", {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": frame_count / fps if fps else 0,
    }


class VideoValidator:
    def __init__(self, videos_dir: Path):
        self.videos_dir = Path(videos_dir)

    def collect_and_validate(self, pattern: str = "*.mp4") -> Dict:
        video_files = sorted(self.videos_dir.glob(pattern))
        valid, corrupt, report = [], [], []
        for vp in tqdm(video_files, desc="Validating videos"):
            ok, reason, info = validate_video(vp)
            vid = extract_video_id(vp.stem)
            entry = {"video_id": vid, "path": str(vp), "reason": reason, **info}
            report.append(entry)
            if ok:
                valid.append(str(vp))
            else:
                corrupt.append(str(vp))
        return {
            "total": len(video_files),
            "valid": valid,
            "corrupt": corrupt,
            "report": report,
        }

    def match_annotations(
        self, valid_paths: List[str], annotations: Dict[str, Dict]
    ) -> Dict:
        matched, missing_ann, missing_video = [], [], []
        ann_ids = set(annotations.keys())
        video_ids = set()
        for p in valid_paths:
            vid = extract_video_id(Path(p).stem)
            video_ids.add(vid)
            if vid in annotations and annotations[vid].get("text_summary", "").strip():
                matched.append({"video_id": vid, "path": p, **annotations[vid]})
            else:
                missing_ann.append(vid)
        for aid in ann_ids:
            if aid not in video_ids:
                missing_video.append(aid)
        return {
            "matched": matched,
            "missing_annotation": missing_ann,
            "missing_video": missing_video,
            "n_matched": len(matched),
        }

    @staticmethod
    def save_report(output_dir: Path, validation: Dict, matching: Dict) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "validation_report.json", "w") as f:
            json.dump(validation, f, indent=2)
        with open(output_dir / "matching_report.json", "w") as f:
            json.dump(
                {k: v for k, v in matching.items() if k != "matched"},
                f,
                indent=2,
            )
        with open(output_dir / "dataset_manifest.json", "w") as f:
            json.dump(matching["matched"], f, indent=2, ensure_ascii=False)
