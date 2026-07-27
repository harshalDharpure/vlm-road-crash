#!/usr/bin/env python3
"""
Evaluate fine-tuned model on test set with full metric suite.
"""

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch

import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import LLaVANeXTWrapper
from src.evaluation.metrics_suite import MetricsSuite
from src.utils import get_config
from src.utils.video_ids import annotation_key_from_path
from src.models.frame_utils import frames_to_collage, collage_temporal_prompt_suffix
from src.models.prompt_strategies import get_prompt

from PIL import Image


# ------------------------------------------------------------------
def load_frames(video_path: str, max_frames: int = 30):
    """Load frames from video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample every 5th frame (as per config)
        if frame_count % 5 == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    return frames


# ------------------------------------------------------------------
def load_finetuned_model(
    checkpoint_path: str,
    base_model_name: str,
    device: str,
    use_collage: bool = True,
    num_key_frames: int = 4,
):
    """Load fine-tuned model from checkpoint."""
    from transformers import (
        LlavaNextProcessor,
        LlavaNextForConditionalGeneration,
        LlavaProcessor,
        LlavaForConditionalGeneration
    )
    
    print(f"Loading base model: {base_model_name}")
    
    # Determine if it's LLaVA-NeXT or LLaVA-1.5
    is_next = "llava-v1.6" in base_model_name or "llava-next" in base_model_name.lower()
    
    # Don't use device_map - load to CPU first, then move to GPU manually
    # This avoids device mismatch issues
    if is_next:
        processor = LlavaNextProcessor.from_pretrained(base_model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            device_map=None  # Load to CPU first
        )
    else:
        processor = LlavaProcessor.from_pretrained(base_model_name)
        model = LlavaForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            device_map=None  # Load to CPU first
        )
    
    # Move model to device after loading (if not CPU)
    if device != "cpu":
        model = model.to(device)
    
    # Load checkpoint to CPU first to avoid OOM, then move to device
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Check if checkpoint contains LoRA weights (has "lora_A" or "lora_B" in keys)
    state_dict = checkpoint["model_state_dict"]
    has_lora = any("lora_A" in key or "lora_B" in key for key in state_dict.keys())
    
    if has_lora:
        print("Detected LoRA checkpoint, applying LoRA configuration...")
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # Configure LoRA (same as training)
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
            lora_config = LoraConfig(
                r=8,  # LoRA rank (same as training)
                lora_alpha=16,  # LoRA alpha (same as training)
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            # Apply LoRA to model
            model = get_peft_model(model, lora_config)
            print("✓ LoRA configuration applied")
            
            # Move model to device BEFORE loading weights (ensures consistent device placement)
            if device != "cpu":
                model = model.to(device)
                print("✓ Model moved to device")
            
            # Load LoRA weights (state_dict is already on CPU from checkpoint load)
            # Filter to only LoRA weights to avoid conflicts
            lora_state_dict = {k: v for k, v in state_dict.items() if "lora" in k.lower()}
            if lora_state_dict:
                # Move LoRA weights to device before loading
                lora_state_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in lora_state_dict.items()}
                model.load_state_dict(lora_state_dict, strict=False)
                print("✓ LoRA weights loaded")
            else:
                # Fallback: load all weights
                state_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
                print("✓ All weights loaded")
            
        except ImportError:
            print("WARNING: PEFT library not available. Trying to load without LoRA...")
            # Try loading without LoRA (will fail if checkpoint is LoRA-only)
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"WARNING: LoRA loading failed: {e}")
            print("Trying to load without LoRA...")
            model.load_state_dict(state_dict, strict=False)
    else:
        # Full model checkpoint
        print("Loading full model checkpoint...")
        # Move model to device if needed (if not using device_map)
        if device != "cpu" and not hasattr(model, "hf_device_map"):
            model = model.to(device)
        model.load_state_dict(state_dict, assign=True)
    
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Training loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    # Create wrapper-like interface
    class FineTunedWrapper:
        def __init__(self, model, processor, device, is_next, use_collage, num_key_frames):
            self.model = model
            self.processor = processor
            self.device = device
            self.is_next = is_next
            self.use_collage = use_collage
            self.num_key_frames = num_key_frames
            # Get the actual device from model
            try:
                self.actual_device = next(model.parameters()).device
            except Exception:
                self.actual_device = torch.device(device if device != "cuda" else "cuda:0")
        
        def generate_summary(self, frames, task_type="v2t"):
            """Generate summary using fine-tuned model."""
            if len(frames) == 0:
                return {"text_summary": ""}
            
            prompt = get_prompt("structured_event", num_frames=len(frames))
            if self.use_collage and len(frames) > 1:
                rep_img = frames_to_collage(frames, max_frames=self.num_key_frames)
                n_key = min(self.num_key_frames, len(frames))
                prompt = prompt + collage_temporal_prompt_suffix(n_key)
            else:
                rep = frames[len(frames) // 2]
                rep_img = Image.fromarray(cv2.cvtColor(rep, cv2.COLOR_BGR2RGB)).convert("RGB")
            
            formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            
            if self.is_next:
                inputs = self.processor(
                    text=formatted_prompt,
                    images=rep_img,
                    return_tensors="pt"
                )
            else:
                inputs = self.processor(
                    images=rep_img,
                    text=formatted_prompt,
                    return_tensors="pt"
                )
            
            # Move all inputs to the actual device where model is
            # Ensure correct dtypes: all integer tensors should be Long, pixel_values should match model dtype
            processed_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(self.actual_device)
                    # Fix dtype issues: all integer tensors (input_ids, attention_mask, etc.) should be Long (int64)
                    # This is critical - Char (int8) causes dtype mismatch errors
                    # Convert any integer type that's not int64 to long
                    if v.dtype != torch.int64 and v.dtype in [torch.int8, torch.int16, torch.int32, torch.uint8]:
                        v = v.long()
                    # Ensure pixel_values are float32 (model expects float32, not float16 for inputs)
                    elif k == "pixel_values":
                        v = v.float()
                    processed_inputs[k] = v
                else:
                    processed_inputs[k] = v
            inputs = processed_inputs
            
            if self.is_next and "image_sizes" not in inputs:
                w, h = rep_img.size
                inputs["image_sizes"] = torch.tensor([[h, w]], device=self.actual_device, dtype=torch.long)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                )
            
            text = self.processor.decode(outputs[0], skip_special_tokens=True)
            if "ASSISTANT:" in text:
                text = text.split("ASSISTANT:")[-1].strip()
            
            return {"text_summary": text}
    
    wrapper = FineTunedWrapper(model, processor, device, is_next, use_collage, num_key_frames)
    return wrapper


# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model on test set"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned checkpoint (.pt file)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name (default: from config)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "val"],
        default="test",
        help="Which split to evaluate on (default: test)"
    )
    parser.add_argument("--use-collage", action="store_true", default=False)
    parser.add_argument("--no-collage", action="store_false", dest="use_collage")
    parser.add_argument("--num-key-frames", type=int, default=4)
    args = parser.parse_args()
    
    # Load config
    config = get_config(args.config)
    
    # Setup paths
    root_dir = Path(config["dataset"]["root_dir"])
    processed_dir = root_dir / config["dataset"]["processed_dir"]
    split_info_file = processed_dir / "split_info.json"
    
    split_annotations_file = processed_dir / f"annotations_{args.split}.json"
    
    if not split_info_file.exists() or not split_annotations_file.exists():
        print("ERROR: Run scripts/01_process_data.py first.")
        sys.exit(1)
    
    with open(split_info_file) as f:
        split_info = json.load(f)
    
    with open(split_annotations_file) as f:
        annotations = json.load(f)
    
    split_videos = split_info["splits"][args.split]
    
    print("=" * 60)
    print(f"EVALUATING FINE-TUNED MODEL ({args.split.upper()} SET)")
    print("=" * 60)
    print(f"Videos: {len(split_videos)}")
    print(f"Multi-frame collage: {args.use_collage} (key frames={args.num_key_frames})")
    
    # Device setup
    device = config["model"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")
    
    # Load model
    base_model_name = args.base_model or config["model"]["vision_model"]
    model = load_finetuned_model(
        args.checkpoint,
        base_model_name,
        device,
        use_collage=args.use_collage,
        num_key_frames=args.num_key_frames,
    )
    
    metrics_suite = MetricsSuite(config.config)
    
    predictions, references, results = [], [], []
    
    # Inference
    print("\nRunning inference...")
    for video_path in tqdm(split_videos, desc="Evaluating"):
        video_path = Path(video_path)
        video_id = video_path.stem
        ann_key = annotation_key_from_path(str(video_path))

        if ann_key not in annotations:
            continue

        gt = annotations[ann_key]["text_summary"]
        if not gt.strip():
            continue
        
        frames = load_frames(str(video_path), max_frames=config["model"]["max_frames"])
        if not frames:
            continue
        
        try:
            out = model.generate_summary(frames, task_type="v2t")
            pred = out.get("text_summary", "").strip()
        except Exception as e:
            print(f"Error on {video_id}: {e}")
            continue
        
        if pred:
            predictions.append(pred)
            references.append(gt)
            results.append({
                "video_id": video_id,
                "prediction": pred,
                "reference": gt
            })
    
    if len(predictions) == 0:
        print("No valid predictions generated.")
        return
    
    # Metrics
    print("\nComputing metrics...")
    metrics = metrics_suite.compute_all(predictions, references)
    metrics["checkpoint"] = args.checkpoint
    metrics["split"] = args.split
    metrics["use_collage"] = args.use_collage
    metrics["num_key_frames"] = args.num_key_frames if args.use_collage else 1
    # Legacy keys for compare script
    metrics["bleu_scores"] = metrics.get("bleu", {})
    metrics["nli_scores"] = metrics.get("nli", {})
    
    # Save results
    checkpoint_name = Path(args.checkpoint).stem
    suffix = "_collage" if args.use_collage else ""
    results_dir = Path(config["paths"]["results"]) / "finetuned" / f"{checkpoint_name}{suffix}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    flat = metrics_suite.flatten_for_table(metrics)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples: {len(predictions)}")
    print(f"BLEU-1: {flat.get('bleu_1', 0):.4f}")
    print(f"BLEU-4: {flat.get('bleu_4', 0):.4f}")
    print(f"METEOR: {flat.get('meteor', 0):.4f}")
    print(f"ROUGE-L: {flat.get('rouge_l', 0):.4f}")
    print(f"BERTScore: {flat.get('bertscore', 0):.4f}")
    print(f"CIDEr: {flat.get('cider', 0):.4f}")
    print(f"NLI Entailment: {flat.get('nli_entailment_acc', 0):.4f}")
    print(f"\nResults saved to: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

