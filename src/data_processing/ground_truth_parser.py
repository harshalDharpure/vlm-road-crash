"""Parse full ground-truth annotations from Crash-1500 Excel file."""
from __future__ import annotations

import json
import re
from datetime import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _format_time(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, time):
        return value.strftime("%H:%M:%S")
    return str(value).strip() or None


def _safe_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def extract_video_id(raw: Any) -> str:
    text = str(raw).strip()
    match = re.search(r"(\d+)", text)
    if match:
        return match.group(1).zfill(6)
    return text


class GroundTruthParser:
    """Parse all 11 Crash-1500 annotation attributes."""

    ATTRIBUTE_COLUMNS = [
        "Video Number",
        "Severity of the Crash",
        "Type of Vehicles involved",
        "No. of Vehicles involved",
        "Location of impact",
        "Start of Crash",
        "End of Crash",
        "Explanation",
        "Ambiguity",
        "Camera View",
        "Weather Conditions",
    ]

    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        self.df: Optional[pd.DataFrame] = None
        self.annotations: Dict[str, Dict[str, Any]] = {}

    def load_excel(self) -> pd.DataFrame:
        self.df = pd.read_excel(self.excel_path)
        return self.df

    def _row_to_record(self, row: pd.Series) -> Dict[str, Any]:
        video_id = extract_video_id(row["Video Number"])
        explanation = _safe_str(row.get("Explanation", ""))
        vehicles = _safe_str(row.get("Type of Vehicles involved", ""))
        severity = _safe_str(row.get("Severity of the Crash", ""))
        location = _safe_str(row.get("Location of impact", ""))
        ambiguity = _safe_str(row.get("Ambiguity", ""))
        camera = _safe_str(row.get("Camera View", ""))
        weather = _safe_str(row.get("Weather Conditions", ""))

        num_vehicles = row.get("No. of Vehicles involved", None)
        if pd.isna(num_vehicles):
            num_vehicles = None
        else:
            num_vehicles = int(float(num_vehicles))

        record = {
            "video_id": video_id,
            "video_number": int(float(row["Video Number"])) if not pd.isna(row["Video Number"]) else None,
            "severity": severity,
            "vehicles_involved": vehicles,
            "num_vehicles": num_vehicles,
            "impact_location": location,
            "crash_start": _format_time(row.get("Start of Crash")),
            "crash_end": _format_time(row.get("End of Crash")),
            "explanation": explanation,
            "text_summary": explanation,
            "ambiguity": ambiguity,
            "camera_view": camera,
            "weather": weather,
            "structured_summary": self._build_structured_summary(
                severity, vehicles, num_vehicles, location, explanation, weather, camera
            ),
        }
        return record

    @staticmethod
    def _build_structured_summary(
        severity: str,
        vehicles: str,
        num_vehicles: Optional[int],
        location: str,
        explanation: str,
        weather: str,
        camera: str,
    ) -> str:
        parts = []
        if severity:
            parts.append(f"Severity: {severity}.")
        if vehicles:
            nv = f" ({num_vehicles} vehicles)" if num_vehicles else ""
            parts.append(f"Vehicles{nv}: {vehicles}.")
        if location:
            parts.append(f"Impact location: {location}.")
        if weather:
            parts.append(f"Weather: {weather}.")
        if camera:
            parts.append(f"Camera view: {camera}.")
        if explanation:
            parts.append(explanation)
        return " ".join(parts).strip()

    def build_all_annotations(self) -> Dict[str, Dict[str, Any]]:
        if self.df is None:
            self.load_excel()
        annotations: Dict[str, Dict[str, Any]] = {}
        for _, row in self.df.iterrows():
            record = self._row_to_record(row)
            annotations[record["video_id"]] = record
        self.annotations = annotations
        return annotations

    def map_videos_to_annotations(self, video_files: List[str]) -> Dict[str, Dict[str, Any]]:
        if not self.annotations:
            self.build_all_annotations()
        video_annotations: Dict[str, Dict[str, Any]] = {}
        for video_file in video_files:
            vid = extract_video_id(Path(video_file).stem)
            if vid in self.annotations:
                video_annotations[vid] = {**self.annotations[vid], "video_file": str(video_file)}
            else:
                alt = vid.lstrip("0") or "0"
                alt_padded = alt.zfill(6)
                if alt_padded in self.annotations:
                    video_annotations[vid] = {**self.annotations[alt_padded], "video_file": str(video_file)}
                else:
                    video_annotations[vid] = {
                        "video_id": vid,
                        "text_summary": "",
                        "explanation": "",
                        "video_file": str(video_file),
                    }
        return video_annotations

    def save_annotations(self, output_path: str, annotations: Optional[Dict] = None) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = annotations if annotations is not None else self.annotations
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_csv(self, output_path: str, annotations: Optional[Dict] = None) -> None:
        data = annotations if annotations is not None else self.annotations
        rows = list(data.values())
        pd.DataFrame(rows).to_csv(output_path, index=False)

    def get_statistics(self, annotations: Optional[Dict] = None) -> Dict[str, Any]:
        data = annotations if annotations is not None else self.annotations
        if not data:
            return {}
        lengths = [len(a.get("text_summary", "").split()) for a in data.values()]
        with_text = sum(1 for a in data.values() if a.get("text_summary", "").strip())
        return {
            "total_videos": len(data),
            "videos_with_annotations": with_text,
            "avg_summary_length": sum(lengths) / len(lengths) if lengths else 0,
            "min_summary_length": min(lengths) if lengths else 0,
            "max_summary_length": max(lengths) if lengths else 0,
            "severity_counts": pd.Series(
                [a.get("severity", "") for a in data.values()]
            ).value_counts().to_dict(),
        }
