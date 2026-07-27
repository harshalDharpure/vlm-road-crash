"""Unified VLM inference interface."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.models.llava_next_wrapper import LLaVANeXTWrapper
from src.models.prompt_strategies import get_prompt


class UnifiedVLM:
    """Config-driven wrapper for primary and baseline VLMs."""

    def __init__(self, model_name: str = "llava-next", config: Optional[Dict] = None):
        self.model_name = model_name
        self.config = config or {}
        self._backend = None
        self._load()

    def _load(self) -> None:
        device = self.config.get("model", {}).get("device", "cuda")
        if self.model_name in ("llava-next", "llava_next", "llava"):
            vision = self.config.get("model", {}).get(
                "vision_model", "llava-hf/llava-v1.6-mistral-7b-hf"
            )
            self._backend = LLaVANeXTWrapper(model_name=vision, device=device, config=self.config)
        else:
            raise NotImplementedError(
                f"Model {self.model_name} not yet integrated. Use llava-next or extend unified_vlm.py."
            )

    def generate_summary(
        self,
        frames: List[np.ndarray],
        prompt_strategy: str = "structured_event",
        frame_indices: Optional[List[int]] = None,
        use_collage: bool = True,
        num_key_frames: int = 4,
    ) -> Dict[str, Any]:
        prompt = get_prompt(prompt_strategy, num_frames=len(frames), frame_indices=frame_indices)
        text = self._backend.generate_caption(
            frames, prompt, use_collage=use_collage, num_key_frames=num_key_frames
        )
        return {"text_summary": text, "prompt_strategy": prompt_strategy}
