#!/usr/bin/env python3
"""Generate publication tables, figures, paper sections, and error analysis."""
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.error_analysis import run_error_analysis
from src.analysis.human_eval import generate_annotation_sheet
from src.analysis.paper_generator import write_paper_support
from src.analysis.plotting import plot_ablation_heatmap, plot_metric_comparison, plot_radar
from src.analysis.statistical_analysis import save_analysis
from src.evaluation.metrics_suite import MetricsSuite
from src.utils import get_config


def main():
    config = get_config()
    cfg = config.config
    root = Path(cfg["dataset"]["root_dir"])
    results_root = Path(cfg["paths"]["results"])

    # Collect all zero-shot metrics
    zs_root = Path(cfg["paths"]["zero_shot"])
    all_metrics = {}
    for metrics_file in zs_root.glob("*/metrics.json"):
        name = metrics_file.parent.name
        with open(metrics_file) as f:
            all_metrics[name] = json.load(f)

    # Canonical run protection: prefer archived full-test metrics if present.
    canonical_name = "every_5th_structured_event_test"
    canonical_archive = zs_root / canonical_name / "metrics_full_226.json"
    if canonical_archive.exists():
        with open(canonical_archive) as f:
            all_metrics[canonical_name] = json.load(f)

    # experiment_tables.md (auto section — see manual header in file for full documentation)
    tables_dir = results_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    full_test_rows = {
        k: v
        for k, v in all_metrics.items()
        if v.get("num_samples") == 226 and k != "every_5th_structured_event_test_collage"
    }

    lines = [
        "# Experiment Tables (auto-generated ablation grid)\n",
        "> **Note:** This grid includes full-test runs only (**N=226**). "
        "Canonical baseline `every_5th_structured_event_test` is read from "
        "`metrics_full_226.json` when available.\n",
        "| Run | N | BLEU-1 | ROUGE-L | METEOR | BERTScore | CIDEr | NLI Ent. |",
        "|-----|---|--------|---------|--------|-----------|-------|----------|",
    ]
    suite = MetricsSuite(cfg)
    for name, m in sorted(full_test_rows.items()):
        flat = suite.flatten_for_table(m)
        n = m.get("num_samples", flat.get("num_samples", "?"))
        lines.append(
            f"| {name} | {n} | {flat.get('bleu_1', 0):.4f} | {flat.get('rouge_l', 0):.4f} | "
            f"{flat.get('meteor', 0):.4f} | {flat.get('bertscore', 0):.4f} | "
            f"{flat.get('cider', 0):.4f} | {flat.get('nli_entailment_acc', 0):.4f} |"
        )
    auto_path = tables_dir / "ablation_metrics_grid.md"
    auto_path.write_text("\n".join(lines), encoding="utf-8")
    # Keep root experiment_tables.md if it has manual sections; only refresh grid appendix
    root_doc = results_root / "experiment_tables.md"
    if root_doc.exists() and "Section 1" in root_doc.read_text(encoding="utf-8"):
        appendix = root_doc.read_text(encoding="utf-8").split("<!-- AUTO_GRID -->")[0].rstrip()
        root_doc.write_text(
            appendix + "\n\n<!-- AUTO_GRID -->\n\n## Auto-generated ablation grid\n\n"
            + auto_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    else:
        (results_root / "experiment_tables.md").write_text("\n".join(lines), encoding="utf-8")

    # comparison_report.json
    comparison = {
        "zero_shot_runs": all_metrics,
        "publication_full_test_runs": full_test_rows,
        "canonical_source": (
            str(canonical_archive)
            if canonical_archive.exists()
            else "results/zero_shot/every_5th_structured_event_test/metrics.json"
        ),
    }
    ablation_file = Path(cfg["paths"].get("ablation", "results/ablation")) / "ablation_results.json"
    if ablation_file.exists():
        with open(ablation_file) as f:
            comparison["ablations"] = json.load(f)
    with open(results_root / "comparison_report.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Figures
    fig_dir = Path(cfg["paths"].get("publication_figures", "results/publication_figures"))
    flat_by_method = {k: suite.flatten_for_table(v) for k, v in all_metrics.items()}
    if flat_by_method:
        plot_metric_comparison(flat_by_method, fig_dir / "metric_comparison.png")
        if flat_by_method:
            first = list(flat_by_method.values())[0]
            plot_radar(first, fig_dir / "performance_radar.png")

    ablation_csv = Path(cfg["paths"].get("ablation", "results/ablation")) / "ablation_tables.csv"
    if ablation_csv.exists():
        import pandas as pd
        df = pd.read_csv(ablation_csv)
        heat = {}
        for _, row in df.iterrows():
            heat[row["config"]] = {"rouge_l": row.get("rouge_l", 0)}
        plot_ablation_heatmap(heat, fig_dir / "ablation_heatmap.png")

    # Error analysis on latest detailed results
    qual_dir = Path(cfg["paths"].get("qualitative_examples", "results/qualitative_examples"))
    for det_file in sorted(zs_root.glob("*/detailed_results.json")):
        with open(det_file) as f:
            results = json.load(f)
        run_error_analysis(results, qual_dir / det_file.parent.name)
        generate_annotation_sheet(results, qual_dir / det_file.parent.name / "human_eval_sheet.csv", n_samples=50)
        break

    # LaTeX tables
    latex_dir = Path(cfg["paths"].get("latex_tables", "results/latex_tables"))
    latex_dir.mkdir(parents=True, exist_ok=True)
    latex = ["\\begin{table}[t]", "\\centering", "\\caption{Crash-1500 summarization results}", "\\begin{tabular}{lcccc}", "\\hline", "Method & BLEU-1 & ROUGE-L & METEOR & NLI \\\\", "\\hline"]
    for name, m in sorted(all_metrics.items()):
        flat = suite.flatten_for_table(m)
        latex.append(
            f"{name.replace('_', ' ')} & {flat.get('bleu_1', 0):.3f} & {flat.get('rouge_l', 0):.3f} & "
            f"{flat.get('meteor', 0):.3f} & {flat.get('nli_entailment_acc', 0):.3f} \\\\"
        )
    latex.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    (latex_dir / "main_results.tex").write_text("\n".join(latex), encoding="utf-8")

    # Paper support
    paper_dir = results_root / "paper_support"
    best_metrics = list(all_metrics.values())[0] if all_metrics else {}
    write_paper_support(paper_dir, cfg, best_metrics)

    save_analysis(Path(cfg["paths"].get("statistical_analysis", "results/statistical_analysis")), {"runs": list(all_metrics.keys())})

    print("Publication outputs generated:")
    print(f"  - {results_root / 'experiment_tables.md'}")
    print(f"  - {results_root / 'comparison_report.json'}")
    print(f"  - {fig_dir}")
    print(f"  - {latex_dir}")


if __name__ == "__main__":
    main()
