#!/usr/bin/env python3
"""
Run full-test (N=226) ablation grid across available GPUs.

Grid: frame_strategies x prompt_strategies from config.yaml ablation section.
Outputs: results/zero_shot/<strategy>_<prompt>_test/ (metrics.json + detailed_results.json)
Then regenerates publication tables.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_config
from src.utils.gpu_manager import get_gpu_memory, select_freest_gpu


@dataclass(frozen=True)
class Job:
    strategy: str
    prompt: str
    run_prefix: str = "fulltest"

    @property
    def name(self) -> str:
        return f"{self.strategy}_{self.prompt}_test"

    @property
    def output_name(self) -> str:
        return f"{self.run_prefix}_{self.name}"


def _eligible_gpus(min_free_gb: float) -> List[int]:
    min_free_mb = min_free_gb * 1024
    gpus = get_gpu_memory()
    eligible = [g["index"] for g in gpus if g["memory_free_mb"] >= min_free_mb]
    if eligible:
        return eligible
    # fallback: pick top-2 freest to avoid oversubscription
    gpus_sorted = sorted(gpus, key=lambda g: g["memory_free_mb"], reverse=True)
    return [g["index"] for g in gpus_sorted[:2]] if gpus_sorted else []


def _run_one(job: Job, gpu: int, cfg: Dict) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable,
        str(project_root / "scripts/02_evaluate_zero_shot.py"),
        "--split",
        "test",
        "--strategy",
        job.strategy,
        "--prompt",
        job.prompt,
        "--no-collage",
    ]
    print(f"\n=== RUN {job.name} on GPU {gpu} ===")
    print(">>", " ".join(cmd))
    return subprocess.call(cmd, env=env, cwd=str(project_root))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-gb", type=float, default=12.0)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="fulltest",
        help="Prefix for zero-shot output directories to avoid canonical overwrite",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow writing into existing output directories",
    )
    args = parser.parse_args()

    config = get_config()
    cfg = config.config
    ab = cfg.get("ablation", {})
    strategies = ab.get("frame_strategies", ["every_5th"])
    prompts = ab.get("prompt_strategies", ["structured_event"])

    jobs = [Job(s, p, args.run_prefix) for s in strategies for p in prompts]

    gpus = _eligible_gpus(args.min_free_gb)
    if not gpus:
        g = select_freest_gpu(args.min_free_gb)  # may be None
        gpus = [g] if g is not None else [0]

    max_parallel = args.max_parallel or len(gpus)
    gpus = gpus[:max_parallel]

    print("Eligible GPUs:", gpus)
    print(f"Total jobs: {len(jobs)}")

    # Simple queue with at most one job per GPU.
    running: Dict[int, Tuple[Job, subprocess.Popen]] = {}
    pending = jobs[:]
    completed: List[Tuple[Job, int]] = []

    while pending or running:
        # Launch up to free GPU slots
        for gpu in gpus:
            if gpu in running:
                continue
            if not pending:
                continue
            job = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = [
                sys.executable,
                str(project_root / "scripts/02_evaluate_zero_shot.py"),
                "--split",
                "test",
                "--strategy",
                job.strategy,
                "--prompt",
                job.prompt,
                "--no-collage",
                "--output-name",
                job.output_name,
            ]
            if not args.allow_overwrite:
                cmd.append("--fail-if-exists")
            p = subprocess.Popen(cmd, env=env, cwd=str(project_root))
            running[gpu] = (job, p)
            print(f"Started {job.name} -> {job.output_name} on GPU {gpu} (pid={p.pid})")

        # Poll
        time.sleep(5)
        done_gpus = []
        for gpu, (job, proc) in running.items():
            rc = proc.poll()
            if rc is None:
                continue
            completed.append((job, rc))
            done_gpus.append(gpu)
            print(f"Finished {job.name} on GPU {gpu} rc={rc}")
        for gpu in done_gpus:
            del running[gpu]

    # Summarize
    failed = [(j, rc) for (j, rc) in completed if rc != 0]
    print("\n=== FULL-TEST ABLATION SUMMARY ===")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    for j, rc in failed[:10]:
        print(f"  FAIL {j.name}: rc={rc}")

    # Regenerate tables
    print("\nRegenerating publication outputs...")
    subprocess.call([sys.executable, str(project_root / "scripts/07_generate_publication_outputs.py")], cwd=str(project_root))

    # Exit non-zero if any failures
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

