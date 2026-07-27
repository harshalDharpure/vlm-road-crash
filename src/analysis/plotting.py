"""Publication-quality figure generation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def plot_metric_comparison(metrics_by_method: Dict[str, Dict], output_path: Path, title: str = "Metric Comparison") -> None:
    methods = list(metrics_by_method.keys())
    metric_names = ["bleu_1", "rouge_l", "meteor", "bertscore", "cider", "nli_entailment_acc"]
    available = [m for m in metric_names if any(m in metrics_by_method[met] or m.replace("_", "") in str(metrics_by_method[met]) for met in methods)]
    if not available:
        available = ["bleu_1", "rouge_l", "meteor"]

    x = np.arange(len(available))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, method in enumerate(methods):
        vals = []
        flat = metrics_by_method[method]
        bleu = flat.get("bleu", flat)
        for m in available:
            if m.startswith("bleu"):
                vals.append(bleu.get(m, flat.get(m, 0)))
            elif m == "nli_entailment_acc":
                nli = flat.get("nli", {})
                vals.append(nli.get("entailment_accuracy", flat.get(m, 0)))
            else:
                vals.append(flat.get(m, 0))
        ax.bar(x + i * width, vals, width, label=method)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(available, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_heatmap(data: Dict[str, Dict[str, float]], output_path: Path, title: str = "Ablation Study") -> None:
    rows = sorted(data.keys())
    cols = sorted({k for r in data.values() for k in r.keys()})
    mat = np.array([[data[r].get(c, 0) for c in cols] for r in rows])
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.8), max(5, len(rows) * 0.5)))
    sns.heatmap(mat, annot=True, fmt=".3f", xticklabels=cols, yticklabels=rows, cmap="YlOrRd", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_radar(metrics: Dict[str, float], output_path: Path, title: str = "Performance Radar") -> None:
    keys = [k for k in metrics if isinstance(metrics[k], (int, float)) and 0 <= metrics[k] <= 1]
    if len(keys) < 3:
        return
    vals = [metrics[k] for k in keys]
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    vals += vals[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, "o-", linewidth=2)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys)
    ax.set_title(title)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot(distributions: Dict[str, List[float]], output_path: Path, title: str = "Per-sample Scores") -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels, data = [], []
    for k, v in distributions.items():
        if v:
            labels.append(k)
            data.append(v)
    if data:
        ax.boxplot(data, labels=labels)
        ax.set_title(title)
        ax.set_ylabel("Score")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
