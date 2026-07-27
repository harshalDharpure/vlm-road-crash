# Experiment results tables (generated)

Project root: `/DATA/vaneet_2221cs15/vlm-road-crash`

## Training and validation loss

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 3.6252 |  |
| 2 | 3.4323 |  |
| 3 | 3.4132 |  |
| 4 | 3.3921 |  |
| 5 | 3.3692 |  |

## Zero-shot evaluation metrics

_No `results/zero_shot/metrics.json` yet._

## Fine-tuned evaluation metrics

Source: `/DATA/vaneet_2221cs15/vlm-road-crash/results/finetuned/best_checkpoint/metrics.json`

| Metric | Value |
|---|---|
| bertscore | 0.870730 |
| bleu_1 | 0.333811 |
| bleu_2 | 0.221847 |
| bleu_3 | 0.148082 |
| bleu_4 | 0.098007 |
| cider | 0.000000 |
| meteor | 0.293332 |
| nli_avg_entailment_prob | 0.327311 |
| nli_contradiction_rate | 0.216814 |
| nli_entailment_accuracy | 0.340708 |
| nli_neutral_rate | 0.442478 |
| nli_total_samples | 226.000000 |
| num_samples | 226.000000 |
| rouge_1 | 0.398435 |
| rouge_2 | 0.171914 |
| rouge_l | 0.260635 |

## Zero-shot vs fine-tuned (scalars)

_Need both zero-shot and fine-tuned metrics._

## Comparison report (JSON)

Key improvements from `comparison_report.json`:

| Metric | Zero-shot | Fine-tuned | Improvement % |
|---|---|---|---|
| bertscore | 0.842649 | 0.870730 | +3.33% |
| meteor | 0.219544 | 0.293332 | +33.61% |
| rouge_1 | 0.325149 | 0.398435 | +22.54% |
| rouge_l | 0.205691 | 0.260635 | +26.71% |
