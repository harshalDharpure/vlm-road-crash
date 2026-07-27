#!/usr/bin/env python3
"""
Aggregate experiment outputs under results/ into Markdown and CSV tables.

Run after the pipeline (or whenever metrics JSON files exist).
Writes to results/tables/ by default.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def flatten_metrics(m: Dict[str, Any]) -> Dict[str, float]:
    """Scalar metrics for CSV (one row per key)."""
    out: Dict[str, float] = {}
    if not m:
        return out
    for k in ("num_samples", "meteor", "rouge_1", "rouge_2", "rouge_l", "bertscore", "cider"):
        if k in m and isinstance(m[k], (int, float)):
            out[k] = float(m[k])
    bs = m.get("bleu_scores") or {}
    for i in range(1, 5):
        key = f"bleu_{i}"
        if key in bs and isinstance(bs[key], (int, float)):
            out[key] = float(bs[key])
    nli = m.get("nli_scores") or {}
    for nk, nv in nli.items():
        if isinstance(nv, (int, float)):
            out[f"nli_{nk}"] = float(nv)
    return out


def main() -> int:
    results_dir = project_root / "results"
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    train_loss = _load_json(results_dir / "training_loss.json")
    val_loss = _load_json(results_dir / "validation_loss.json")
    zs = _load_json(results_dir / "zero_shot" / "metrics.json")
    comp = _load_json(results_dir / "comparison_report.json")

    finetuned_path: Optional[Path] = None
    ft: Optional[Dict[str, Any]] = None
    finetuned_dir = results_dir / "finetuned"
    if finetuned_dir.is_dir():
        for p in sorted(finetuned_dir.iterdir()):
            if p.is_dir() and (p / "metrics.json").exists():
                finetuned_path = p / "metrics.json"
                ft = _load_json(finetuned_path)
                break

    lines: List[str] = []
    lines.append("# Experiment results tables (generated)")
    lines.append("")
    lines.append(f"Project root: `{project_root}`")
    lines.append("")

    # Training loss table
    lines.append("## Training and validation loss")
    lines.append("")
    if train_loss or val_loss:
        epochs = sorted(
            {int(k.replace("epoch_", "")) for k in (train_loss or {}).keys() if k.startswith("epoch_")}
            | {int(k.replace("epoch_", "")) for k in (val_loss or {}).keys() if k.startswith("epoch_")}
        )
        rows = []
        for e in epochs:
            ek = f"epoch_{e}"
            tr = (train_loss or {}).get(ek, {})
            va = (val_loss or {}).get(ek, {})
            tl = tr.get("loss") if isinstance(tr, dict) else None
            vl = va.get("loss") if isinstance(va, dict) else None
            rows.append(
                [
                    str(e),
                    "" if tl is None else f"{float(tl):.4f}",
                    "" if vl is None else f"{float(vl):.4f}",
                ]
            )
        lines.append(_md_table(["Epoch", "Train loss", "Val loss"], rows))
    else:
        lines.append("_No `training_loss.json` / `validation_loss.json` yet._")
    lines.append("")

    # Zero-shot metrics
    lines.append("## Zero-shot evaluation metrics")
    lines.append("")
    if zs:
        flat = flatten_metrics(zs)
        rows = [[k, f"{v:.6f}"] for k, v in sorted(flat.items())]
        lines.append(_md_table(["Metric", "Value"], rows))
    else:
        lines.append("_No `results/zero_shot/metrics.json` yet._")
    lines.append("")

    # Fine-tuned
    lines.append("## Fine-tuned evaluation metrics")
    lines.append("")
    if ft:
        lines.append(f"Source: `{finetuned_path}`")
        lines.append("")
        flat = flatten_metrics(ft)
        rows = [[k, f"{v:.6f}"] for k, v in sorted(flat.items())]
        lines.append(_md_table(["Metric", "Value"], rows))
    else:
        lines.append("_No `results/finetuned/*/metrics.json` yet._")
    lines.append("")

    # Side-by-side
    lines.append("## Zero-shot vs fine-tuned (scalars)")
    lines.append("")
    if zs and ft:
        keys = sorted(set(flatten_metrics(zs).keys()) & set(flatten_metrics(ft).keys()))
        fz = flatten_metrics(zs)
        fft = flatten_metrics(ft)
        rows = []
        for k in keys:
            a, b = fz[k], fft[k]
            pct = ((b - a) / a * 100.0) if a else float("nan")
            rows.append([k, f"{a:.6f}", f"{b:.6f}", f"{pct:+.2f}%"])
        lines.append(_md_table(["Metric", "Zero-shot", "Fine-tuned", "Delta %"], rows))
    else:
        lines.append("_Need both zero-shot and fine-tuned metrics._")
    lines.append("")

    # Comparison JSON summary
    lines.append("## Comparison report (JSON)")
    lines.append("")
    if comp:
        lines.append("Key improvements from `comparison_report.json`:")
        lines.append("")
        comp_block = comp.get("comparison") or {}
        rows = []
        for mk, mv in sorted(comp_block.items()):
            if not isinstance(mv, dict):
                continue
            if "zero_shot" in mv and "finetuned" in mv:
                imp = mv.get("improvement") or {}
                pct = imp.get("percentage")
                rows.append(
                    [
                        mk,
                        f"{mv['zero_shot']:.6f}",
                        f"{mv['finetuned']:.6f}",
                        "" if pct is None else f"{float(pct):+.2f}%",
                    ]
                )
        if rows:
            lines.append(_md_table(["Metric", "Zero-shot", "Fine-tuned", "Improvement %"], rows))
    else:
        lines.append("_No `results/comparison_report.json` yet._")

    out_md = tables_dir / "experiment_tables.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")

    # CSV side-by-side
    if zs and ft:
        fz, fft = flatten_metrics(zs), flatten_metrics(ft)
        keys = sorted(set(fz.keys()) & set(fft.keys()))
        csv_path = tables_dir / "evaluation_side_by_side.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "zero_shot", "fine_tuned", "delta_pct"])
            for k in keys:
                a, b = fz[k], fft[k]
                pct = ((b - a) / a * 100.0) if a else ""
                w.writerow([k, f"{a:.8f}", f"{b:.8f}", f"{pct:.4f}" if pct != "" else ""])
        print(f"Wrote {csv_path}")

    # Loss CSV
    if train_loss or val_loss:
        csv_path = tables_dir / "training_loss_by_epoch.csv"
        epochs = sorted(
            {int(k.replace("epoch_", "")) for k in (train_loss or {}).keys() if k.startswith("epoch_")}
            | {int(k.replace("epoch_", "")) for k in (val_loss or {}).keys() if k.startswith("epoch_")}
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss"])
            for e in epochs:
                ek = f"epoch_{e}"
                tr = (train_loss or {}).get(ek, {})
                va = (val_loss or {}).get(ek, {})
                tl = tr.get("loss") if isinstance(tr, dict) else ""
                vl = va.get("loss") if isinstance(va, dict) else ""
                w.writerow([e, tl, vl])
        print(f"Wrote {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
