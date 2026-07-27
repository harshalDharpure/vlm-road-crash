"""Statistical analysis for publication-ready results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else (float(arr.mean()), 0.0)


def bootstrap_ci(values: List[float], n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))


def paired_ttest(a: List[float], b: List[float]) -> Dict:
    if len(a) != len(b) or len(a) < 2:
        return {"statistic": None, "pvalue": None}
    t, p = stats.ttest_rel(a, b)
    return {"statistic": float(t), "pvalue": float(p)}


def wilcoxon_test(a: List[float], b: List[float]) -> Dict:
    if len(a) != len(b) or len(a) < 2:
        return {"statistic": None, "pvalue": None}
    try:
        w, p = stats.wilcoxon(a, b)
        return {"statistic": float(w), "pvalue": float(p)}
    except Exception:
        return {"statistic": None, "pvalue": None}


def aggregate_seeds(seed_metrics: List[Dict], metric_key: str) -> Dict:
    values = [m.get(metric_key, 0) for m in seed_metrics if metric_key in m]
    m, s = mean_std(values)
    lo, hi = bootstrap_ci(values) if values else (0, 0)
    return {"mean": m, "std": s, "ci_low": lo, "ci_high": hi, "n": len(values)}


def compare_methods(method_a: Dict, method_b: Dict, per_sample_a: List[float], per_sample_b: List[float]) -> Dict:
    return {
        "method_a": method_a,
        "method_b": method_b,
        "paired_ttest": paired_ttest(per_sample_a, per_sample_b),
        "wilcoxon": wilcoxon_test(per_sample_a, per_sample_b),
        "delta_mean": float(np.mean(per_sample_a) - np.mean(per_sample_b)) if per_sample_a else 0,
    }


def save_analysis(output_dir: Path, results: Dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "statistical_summary.json", "w") as f:
        json.dump(results, f, indent=2)
