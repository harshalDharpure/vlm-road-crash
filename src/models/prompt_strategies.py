"""Prompt engineering strategies for crash video summarization."""
from __future__ import annotations

from typing import Dict, List, Optional


PROMPT_STRATEGIES = [
    "basic_caption",
    "temporal_sequence",
    "safety_critical",
    "chain_of_thought",
    "structured_event",
    "faithfulness",
]


def get_prompt(strategy: str, num_frames: int = 0, frame_indices: Optional[List[int]] = None) -> str:
    frame_indices = frame_indices or []
    temporal_hint = ""
    if frame_indices:
        temporal_hint = f" Frame indices (temporal order): {frame_indices[:12]}{'...' if len(frame_indices) > 12 else ''}."

    prompts: Dict[str, str] = {
        "basic_caption": (
            "Describe this road traffic crash video in a concise factual caption. "
            "Focus on vehicles, the collision, and the outcome."
        ),
        "temporal_sequence": (
            f"These {num_frames or 'sequential'} frames are temporally ordered samples from a road crash video.{temporal_hint} "
            "Describe the event chronologically: pre-crash context, collision moment, and post-crash state."
        ),
        "safety_critical": (
            "You are a traffic safety analyst. From these crash frames, report safety-critical facts: "
            "vehicles involved, risky maneuvers, point of impact, severity indicators, and road-user exposure. "
            "Avoid speculation beyond visible evidence."
        ),
        "chain_of_thought": (
            "Analyze step by step: (1) scene and road layout, (2) vehicles and their motion before impact, "
            "(3) collision mechanism, (4) impact location, (5) immediate outcome. "
            "Then provide a concise 4-6 sentence factual summary."
        ),
        "structured_event": (
            "These frames are sequential samples from a road crash video. Describe:\n"
            "1. Vehicles involved\n"
            "2. Crash sequence\n"
            "3. Point of collision\n"
            "4. Final outcome\n"
            "Provide a concise factual summary suitable for an incident report. "
            "Do not hallucinate unseen details."
        ),
        "faithfulness": (
            "You are writing a factual dashcam incident report from this single frame. "
            "Describe ONLY what is clearly visible: road type, weather/lighting if visible, "
            "vehicle colors and types, motions before impact, how the collision occurred, "
            "and impact location on vehicles. "
            "Use 3-5 short sentences. Be specific about colors and vehicle roles. "
            "Do NOT discuss investigations, liability, or generic safety advice. "
            "Do NOT repeat sentences. Do NOT invent vehicles or events not visible."
        ),
    }
    if strategy not in prompts:
        raise ValueError(f"Unknown prompt strategy: {strategy}. Choose from {PROMPT_STRATEGIES}")
    return prompts[strategy]


def list_strategies() -> List[str]:
    return list(PROMPT_STRATEGIES)
