"""Unified evaluation metrics for crash video summarization."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.evaluation.bleu_evaluator import BLEUEvaluator
from src.evaluation.nli_evaluator import NLIEvaluator


class MetricsSuite:
    """Compute BLEU, ROUGE, METEOR, BERTScore, CIDEr, SPICE, and NLI."""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        eval_cfg = config.get("evaluation", {})
        self.bleu = BLEUEvaluator(
            max_order=eval_cfg.get("bleu", {}).get("max_order", 4),
            smooth=eval_cfg.get("bleu", {}).get("smooth", True),
        )
        nli_cfg = eval_cfg.get("nli", {})
        self.nli = NLIEvaluator(
            model_name=nli_cfg.get("model_name", "roberta-large-mnli"),
            device=nli_cfg.get("device", "cpu"),
            batch_size=nli_cfg.get("batch_size", 8),
        )
        self.bertscore_device = eval_cfg.get("bertscore_device", "cpu")

    def compute_all(self, predictions: List[str], references: List[str]) -> Dict:
        if not predictions:
            return {"num_samples": 0}

        metrics: Dict = {"num_samples": len(predictions)}

        metrics["bleu"] = self.bleu.compute_bleu_batch(predictions, references)

        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            r1, r2, rl = [], [], []
            for p, r in zip(predictions, references):
                s = scorer.score(r, p)
                r1.append(s["rouge1"].fmeasure)
                r2.append(s["rouge2"].fmeasure)
                rl.append(s["rougeL"].fmeasure)
            metrics["rouge_1"] = float(np.mean(r1))
            metrics["rouge_2"] = float(np.mean(r2))
            metrics["rouge_l"] = float(np.mean(rl))
        except Exception as e:
            metrics["rouge_error"] = str(e)

        try:
            from nltk.translate.meteor_score import meteor_score

            meteor_vals = [meteor_score([r.split()], p.split()) for p, r in zip(predictions, references)]
            metrics["meteor"] = float(np.mean(meteor_vals))
        except Exception as e:
            metrics["meteor_error"] = str(e)

        try:
            from bert_score import score as bert_score

            _, _, f1 = bert_score(
                predictions, references, lang="en", verbose=False, device=self.bertscore_device
            )
            metrics["bertscore"] = float(f1.mean())
        except Exception as e:
            metrics["bertscore"] = 0.0
            metrics["bertscore_error"] = str(e)

        try:
            from pycocoevalcap.cider.cider import Cider

            gts = {str(i): [references[i]] for i in range(len(references))}
            res = {str(i): [predictions[i]] for i in range(len(predictions))}
            cider = Cider()
            score, _ = cider.compute_score(gts, res)
            metrics["cider"] = float(score)
        except Exception as e:
            metrics["cider_error"] = str(e)

        try:
            from pycocoevalcap.spice.spice import Spice

            gts = {i: [references[i]] for i in range(len(references))}
            res = {i: [predictions[i]] for i in range(len(predictions))}
            spice = Spice()
            score, _ = spice.compute_score(gts, res)
            metrics["spice"] = float(score)
        except Exception as e:
            metrics["spice_error"] = str(e)

        try:
            metrics["nli"] = self.nli.evaluate(predictions, references)
        except Exception as e:
            metrics["nli_error"] = str(e)

        return metrics

    @staticmethod
    def flatten_for_table(metrics: Dict) -> Dict[str, float]:
        flat = {"num_samples": metrics.get("num_samples", 0)}
        bleu = metrics.get("bleu", {})
        for k, v in bleu.items():
            flat[k] = v
        for key in ("meteor", "rouge_1", "rouge_2", "rouge_l", "bertscore", "cider", "spice"):
            if key in metrics:
                flat[key] = metrics[key]
        nli = metrics.get("nli", {})
        if nli:
            flat["nli_entailment_acc"] = nli.get("entailment_accuracy", 0)
            flat["nli_avg_entailment_prob"] = nli.get("avg_entailment_prob", 0)
        return flat
