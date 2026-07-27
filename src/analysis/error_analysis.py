"""Automatic error analysis for crash summarization predictions."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

GENERIC_PATTERNS = [
    r"a car crash",
    r"vehicles collided",
    r"accident occurred",
    r"crash happened",
    r"unable to determine",
    r"cannot be determined",
]


def _word_overlap(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def analyze_sample(prediction: str, reference: str, video_id: str) -> Dict:
    issues = []
    pred_l, ref_l = prediction.lower(), reference.lower()

    if len(prediction.split()) < 15:
        issues.append("too_short")
    if any(re.search(p, pred_l) for p in GENERIC_PATTERNS):
        issues.append("generic_summary")
    if _word_overlap(prediction, reference) < 0.15:
        issues.append("missing_events")
    ref_vehicles = re.findall(r"\b(car|truck|bus|motorcycle|bike|van|suv)\b", ref_l)
    pred_vehicles = re.findall(r"\b(car|truck|bus|motorcycle|bike|van|suv)\b", pred_l)
    if ref_vehicles and not pred_vehicles:
        issues.append("missing_vehicle_mention")
    if "not" in pred_l and any(w in ref_l for w in pred_l.split() if w != "not"):
        issues.append("possible_contradiction")
    if len(prediction.split()) > 5 and _word_overlap(prediction, reference) < 0.05:
        issues.append("likely_hallucination")

    temporal_words = ["before", "then", "after", "first", "finally", "during"]
    if not any(w in pred_l for w in temporal_words) and len(reference.split()) > 40:
        issues.append("temporal_inconsistency")

    return {
        "video_id": video_id,
        "prediction": prediction,
        "reference": reference,
        "issues": issues,
        "failure_explanation": "; ".join(issues) if issues else "none",
    }


def run_error_analysis(results: List[Dict], output_dir: Path, top_k: int = 20) -> Dict:
    analyzed = [analyze_sample(r["prediction"], r["reference"], r["video_id"]) for r in results]
    by_issue: Dict[str, List] = {}
    for a in analyzed:
        for issue in a["issues"]:
            by_issue.setdefault(issue, []).append(a)

    failures = [a for a in analyzed if a["issues"]]
    summary = {
        "total": len(analyzed),
        "failure_count": len(failures),
        "failure_rate": len(failures) / max(len(analyzed), 1),
        "issue_counts": {k: len(v) for k, v in by_issue.items()},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "error_analysis.json", "w") as f:
        json.dump({"summary": summary, "samples": analyzed}, f, indent=2, ensure_ascii=False)

    qualitative = sorted(failures, key=lambda x: len(x["issues"]), reverse=True)[:top_k]
    with open(output_dir / "qualitative_failures.json", "w") as f:
        json.dump(qualitative, f, indent=2, ensure_ascii=False)

    return summary
