#!/usr/bin/env python3
"""Build publication-grade statistical/provenance proof artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_config


def _read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))
    return max(0.0, center - margin), min(1.0, center + margin)


def two_prop_ztest(k1: int, n1: int, k2: int, n2: int) -> Dict[str, float]:
    if min(n1, n2) <= 0:
        return {"z": 0.0, "p_value": 1.0}
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * ((1 / n1) + (1 / n2)))
    if se == 0:
        return {"z": 0.0, "p_value": 1.0}
    z = (p1 - p2) / se
    p_val = 2 * (1 - _normal_cdf(abs(z)))
    return {"z": z, "p_value": p_val}


def nli_from_metrics(m: Dict) -> Tuple[int, int, float]:
    nli = m.get("nli", {})
    n = int(nli.get("total_samples") or m.get("num_samples") or 0)
    p = float(nli.get("entailment_accuracy", 0.0))
    k = int(round(p * n))
    return k, n, p


def main() -> None:
    cfg = get_config().config
    results_root = Path(cfg["paths"]["results"])
    zero_root = Path(cfg["paths"]["zero_shot"])
    fin_root = Path(cfg["paths"]["finetuned"])

    canonical_metrics_path = zero_root / "every_5th_structured_event_test" / "metrics_full_226.json"
    canonical_log_path = results_root / "logs" / "zero_shot_full_test.log"
    fin_metrics_path = fin_root / "best_checkpoint" / "metrics.json"
    nli_best_path = results_root / "nli_optimized" / "best_run.json"
    nli_opt_metrics_path = (
        project_root / _read_json(nli_best_path).get("dir", "") / "metrics.json"
        if nli_best_path.exists()
        else None
    )

    if not canonical_metrics_path.exists():
        raise FileNotFoundError(f"Missing canonical metrics: {canonical_metrics_path}")
    if not fin_metrics_path.exists():
        raise FileNotFoundError(f"Missing finetuned metrics: {fin_metrics_path}")
    if not (nli_opt_metrics_path and nli_opt_metrics_path.exists()):
        raise FileNotFoundError(f"Missing NLI-optimized metrics: {nli_opt_metrics_path}")

    canonical = _read_json(canonical_metrics_path)
    fin = _read_json(fin_metrics_path)
    nli_opt = _read_json(nli_opt_metrics_path)

    kc, nc, pc = nli_from_metrics(canonical)
    kf, nf, pf = nli_from_metrics({"nli": fin.get("nli_scores", {}), "num_samples": fin.get("num_samples", 0)})
    kn, nn, pn = nli_from_metrics(nli_opt)

    ci_c = wilson_ci(kc, nc)
    ci_f = wilson_ci(kf, nf)
    ci_n = wilson_ci(kn, nn)

    z_cf = two_prop_ztest(kc, nc, kf, nf)
    z_cn = two_prop_ztest(kc, nc, kn, nn)
    z_nf = two_prop_ztest(kn, nn, kf, nf)

    proof = {
        "runs": {
            "canonical_zero_shot": {
                "metrics_path": str(canonical_metrics_path),
                "sha256": _sha256(canonical_metrics_path),
                "nli_entailment_accuracy": pc,
                "nli_entailment_count": kc,
                "n_samples": nc,
                "nli_ci_95_wilson": {"low": ci_c[0], "high": ci_c[1]},
                "source_log": str(canonical_log_path) if canonical_log_path.exists() else None,
            },
            "finetuned_v1": {
                "metrics_path": str(fin_metrics_path),
                "sha256": _sha256(fin_metrics_path),
                "nli_entailment_accuracy": pf,
                "nli_entailment_count": kf,
                "n_samples": nf,
                "nli_ci_95_wilson": {"low": ci_f[0], "high": ci_f[1]},
            },
            "nli_optimized": {
                "metrics_path": str(nli_opt_metrics_path),
                "sha256": _sha256(nli_opt_metrics_path),
                "nli_entailment_accuracy": pn,
                "nli_entailment_count": kn,
                "n_samples": nn,
                "nli_ci_95_wilson": {"low": ci_n[0], "high": ci_n[1]},
            },
        },
        "nli_significance_tests": {
            "canonical_vs_finetuned": z_cf,
            "canonical_vs_nli_optimized": z_cn,
            "nli_optimized_vs_finetuned": z_nf,
        },
    }

    out_dir = Path(cfg["paths"].get("statistical_analysis", "results/statistical_analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "journal_proof.json"
    md_out = out_dir / "journal_proof.md"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(proof, f, indent=2)

    md_lines = [
        "# Journal Proof Report",
        "",
        "## NLI with 95% CI (Wilson)",
        "",
        "| Run | Entailment | 95% CI | n |",
        "|-----|------------|--------|---|",
        f"| canonical_zero_shot | {pc:.4f} | [{ci_c[0]:.4f}, {ci_c[1]:.4f}] | {nc} |",
        f"| finetuned_v1 | {pf:.4f} | [{ci_f[0]:.4f}, {ci_f[1]:.4f}] | {nf} |",
        f"| nli_optimized | {pn:.4f} | [{ci_n[0]:.4f}, {ci_n[1]:.4f}] | {nn} |",
        "",
        "## Two-Proportion Significance Tests (two-sided)",
        "",
        "| Comparison | z | p-value |",
        "|------------|---|---------|",
        f"| canonical vs finetuned | {z_cf['z']:.4f} | {z_cf['p_value']:.4e} |",
        f"| canonical vs nli_optimized | {z_cn['z']:.4f} | {z_cn['p_value']:.4e} |",
        f"| nli_optimized vs finetuned | {z_nf['z']:.4f} | {z_nf['p_value']:.4e} |",
        "",
        "## Provenance (SHA256)",
        "",
        f"- canonical_zero_shot: `{proof['runs']['canonical_zero_shot']['sha256']}`",
        f"- finetuned_v1: `{proof['runs']['finetuned_v1']['sha256']}`",
        f"- nli_optimized: `{proof['runs']['nli_optimized']['sha256']}`",
    ]
    md_out.write_text("\n".join(md_lines), encoding="utf-8")

    print("Generated:")
    print(f"  - {json_out}")
    print(f"  - {md_out}")


if __name__ == "__main__":
    main()

