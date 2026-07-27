"""Match video file paths to annotation dict keys (padded numeric IDs)."""
import re
from pathlib import Path


def extract_video_id(stem_or_path: str) -> str:
    stem = Path(stem_or_path).stem if "/" in stem_or_path or "\\" in stem_or_path else stem_or_path
    match = re.search(r"(\d+)", stem)
    if match:
        return match.group(1).zfill(6)
    return stem


def annotation_key_from_path(video_path: str) -> str:
    return extract_video_id(video_path)


def video_path_from_id(video_id: str, videos_dir: Path) -> Path:
    vid = extract_video_id(video_id)
    for pattern in (f"{vid}.mp4", f"{int(vid)}.mp4", f"{vid.lstrip('0') or '0'}.mp4"):
        p = videos_dir / pattern
        if p.exists():
            return p
    return videos_dir / f"{vid}.mp4"
