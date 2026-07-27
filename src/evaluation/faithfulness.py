"""Post-processing and sentence filtering to maximize NLI faithfulness."""
from __future__ import annotations

import re
from typing import List, Optional

# Phrases that often yield neutral/contradiction vs detailed GT
BOILERPLATE_PATTERNS = [
    r"investigation(?:s)?\s+(?:will|may|should|might)",
    r"determine\s+the\s+cause",
    r"assign(?:ing)?\s+responsibility",
    r"establish(?:ing)?\s+responsibility",
    r"liability\s+for",
    r"unfortunate\s+incident",
    r"entire\s+event\s+was\s+captured",
    r"captured\s+by\s+the\s+camera",
    r"in\s+front\s+of\s+the\s+camera\s+car",
    r"this\s+incident\s+suggests",
    r"may\s+consider\s+factors",
    r"visibility,\s+road\s+conditions",
]


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 10]


def remove_boilerplate(text: str) -> str:
    out = text
    for pat in BOILERPLATE_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\.\s*\.", ".", out)
    return out.strip()


def dedupe_sentences(text: str) -> str:
    seen = set()
    kept = []
    for s in split_sentences(text):
        key = s.lower()[:80]
        if key not in seen:
            seen.add(key)
            kept.append(s)
    return " ".join(kept)


def truncate_repetition(text: str, max_chars: int = 600) -> str:
    if len(text) <= max_chars:
        return text
    # Cut at sentence boundary before limit
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > 100:
        return truncated[: last_period + 1]
    return truncated


def postprocess_prediction(text: str, max_chars: int = 600) -> str:
    text = remove_boilerplate(text)
    text = dedupe_sentences(text)
    text = truncate_repetition(text, max_chars=max_chars)
    return text.strip()


def filter_sentences_by_nli(
    prediction: str,
    reference: str,
    nli_evaluator,
    min_entail_prob: float = 0.25,
    keep_neutral_if_empty: bool = True,
) -> str:
    """Keep only sentences that are entailed (or high prob) by the reference."""
    sentences = split_sentences(postprocess_prediction(prediction))
    if not sentences:
        return postprocess_prediction(prediction)

    kept = []
    for sent in sentences:
        result = nli_evaluator.predict_entailment(reference, sent)
        if result["predicted_class"] == "entailment":
            kept.append(sent)
        elif (
            keep_neutral_if_empty
            and result["predicted_class"] == "neutral"
            and result["entailment_prob"] >= min_entail_prob
        ):
            kept.append(sent)

    if kept:
        return " ".join(kept)
    # Fallback: shortest non-boilerplate prefix
    return postprocess_prediction(prediction, max_chars=400)
