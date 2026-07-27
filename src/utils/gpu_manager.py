"""Automatic GPU selection and VRAM logging."""
from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Optional, Tuple


def get_gpu_memory() -> List[Dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        gpus = []
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                    "memory_free_mb": float(parts[4]),
                })
        return gpus
    except Exception:
        return []


def select_freest_gpu(min_free_gb: float = 8.0) -> Optional[int]:
    gpus = get_gpu_memory()
    if not gpus:
        return None
    min_free_mb = min_free_gb * 1024
    eligible = [g for g in gpus if g["memory_free_mb"] >= min_free_mb]
    if not eligible:
        eligible = gpus
    best = max(eligible, key=lambda g: g["memory_free_mb"])
    return best["index"]


def setup_gpu_from_config(config: Dict) -> str:
    gpu_cfg = config.get("gpu", {})
    if gpu_cfg.get("auto_select", True) and "CUDA_VISIBLE_DEVICES" not in os.environ:
        idx = select_freest_gpu(gpu_cfg.get("min_free_gb", 8))
        if idx is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            print(f"Auto-selected GPU {idx}")
    device = "cuda" if _cuda_available() else "cpu"
    if gpu_cfg.get("log_vram", True) and device == "cuda":
        for g in get_gpu_memory():
            print(
                f"  GPU {g['index']}: {g['memory_free_mb']:.0f}MB free / "
                f"{g['memory_total_mb']:.0f}MB total"
            )
    return device


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def log_vram(prefix: str = "") -> None:
    for g in get_gpu_memory():
        print(f"{prefix}GPU {g['index']}: used={g['memory_used_mb']:.0f}MB free={g['memory_free_mb']:.0f}MB")
