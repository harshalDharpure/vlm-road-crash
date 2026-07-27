"""Human evaluation sheet generation and agreement metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


CRITERIA = ["factual_accuracy", "temporal_correctness", "fluency", "completeness"]
LIKERT_SCALE = "Rate 1 (poor) to 5 (excellent)"


def generate_annotation_sheet(
    results: List[Dict],
    output_path: Path,
    n_samples: int = 100,
) -> pd.DataFrame:
    subset = results[:n_samples] if len(results) > n_samples else results
    rows = []
    for r in subset:
        rows.append({
            "video_id": r.get("video_id", ""),
            "reference": r.get("reference", ""),
            "prediction": r.get("prediction", ""),
            "factual_accuracy": "",
            "temporal_correctness": "",
            "fluency": "",
            "completeness": "",
            "annotator_id": "",
            "notes": "",
        })
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    instructions = {
        "criteria": CRITERIA,
        "scale": LIKERT_SCALE,
        "instructions": "Compare prediction against reference and video frames when available.",
    }
    with open(output_path.parent / "human_eval_instructions.json", "w") as f:
        json.dump(instructions, f, indent=2)
    return df


def cohens_kappa(rater1: List[int], rater2: List[int]) -> float:
    r1, r2 = np.array(rater1), np.array(rater2)
    n = len(r1)
    if n == 0:
        return 0.0
    categories = np.unique(np.concatenate([r1, r2]))
    conf = np.zeros((len(categories), len(categories)))
    cat_idx = {c: i for i, c in enumerate(categories)}
    for a, b in zip(r1, r2):
        conf[cat_idx[a], cat_idx[b]] += 1
    conf /= n
    po = np.trace(conf)
    pe = np.sum(conf.sum(axis=0) * conf.sum(axis=1))
    if pe == 1:
        return 1.0
    return float((po - pe) / (1 - pe))


def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    n_items, n_raters = ratings_matrix.shape[0], ratings_matrix.shape[1]
    categories = int(ratings_matrix.max()) + 1
    p_j = np.zeros(categories)
    for j in range(categories):
        p_j[j] = np.sum(ratings_matrix == j) / (n_items * n_raters)
    P_i = []
    for i in range(n_items):
        counts = [np.sum(ratings_matrix[i] == j) for j in range(categories)]
        P_i.append((np.sum(np.array(counts) ** 2) - n_raters) / (n_raters * (n_raters - 1)))
    P_bar = np.mean(P_i)
    P_e = np.sum(p_j ** 2)
    if P_e == 1:
        return 1.0
    return float((P_bar - P_e) / (1 - P_e))
