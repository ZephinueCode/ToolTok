# src/eval/evaluate_vlm.py

import torch
import os
import json
import shutil
import numpy as np
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from ..utils.parameters import HYPERPARAMS as HP
from ..utils.sft_screenspot import ScreenSpotDataManager
from ..tools.runner import Runner
from ..train.reward import batch_compute_rewards
from ..tools.visual_utils import visualize_trajectory

try:
    from .baseline_vlm import BaselineAPIRunner
except ImportError:
    BaselineAPIRunner = None

def evaluate_model(mode="trained", limit=None, model_path=None):
    """
    Main Evaluation Loop.
    Args:
        mode: "trained" (Local Model) or "api_baseline" (GPT-4o/Claude etc.)
        limit: Max samples to evaluate (None for all)
    """
    if model_path is None:
        model_path = HP.SFT_2_OUTPUT_PATH

    # 1. Setup Output Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = "./results/eval"
    eval_name = f"{mode}_{timestamp}"
    result_dir = os.path.join(base_save_dir, eval_name)
    img_save_dir = os.path.join(result_dir, "images")
    
    os.makedirs(img_save_dir, exist_ok=True)
    print(f"\n[EVAL] Starting Evaluation: {mode.upper()}")
    print(f"[EVAL] Saving results to: {result_dir}")

    # 2. Load Data (Use Test Split)
    print(f"[EVAL] Loading ScreenSpot Test Set...")
    ss_manager = ScreenSpotDataManager()
    dataset = ss_manager.raw_test # Use the reserved 10% test split
    
    if limit:
        print(f"[EVAL] Limiting to first {limit} samples.")
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    print(f"[EVAL] Samples to process: {len(dataset)}")

    # 3. Initialize Model & Runner
    model = None
    processor = None
    runner = None

    if mode == "api_baseline":
        if BaselineAPIRunner is None:
            raise ImportError("BaselineAPIRunner not found. Check src/eval/baseline_vlm.py")
        print(f"[EVAL] Initializing API Runner...")
        runner = BaselineAPIRunner()
    else:
        print(f"[EVAL] Loading Local Model from: {model_path}")
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, 
                device_map="auto", 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            runner = Runner()
        except Exception as e:
            print(f"[Error] Failed to load model: {e}")
            return

    # 4. Main Loop
    results = []
    metrics = {
        "success_count": 0,
        "total_steps": 0,
        "total_reward": 0,
        "failed_reasons": {1:0, 2:0, 3:0, 4:0}
    }

    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        try:
            # Prepare Image (Keep Original Size)
            raw_img = sample['image'].convert("RGB")
            w, h = raw_img.size
            
            # Prepare GT BBox
            bbox = sample.get('bbox', None)
            if not bbox and 'point' in sample:
                p = sample['point']
                bbox = [p[0], p[1], p[0], p[1]]
            
            # [CRITICAL FIX] Robust BBox Scaling
            # Ensure bbox is in absolute pixels for Runner & Visualization
            final_bbox = [0, 0, 0, 0]
            if bbox:
                # Check if bbox is normalized (0.0-1.0)
                if all(0.0 <= x <= 1.0 for x in bbox):
                    final_bbox = [
                        bbox[0] * w, bbox[1] * h,
                        bbox[2] * w, bbox[3] * h
                    ]
                else:
                    # Assume absolute pixels
                    final_bbox = bbox

            # Construct GT Data Entry
            gt_data_entry = {
                "image": raw_img, 
                "instruction": sample['instruction'],
                "action_type": "click", 
                "bbox": final_bbox 
            }
            ground_truth_data = [gt_data_entry]

            # --- Run Inference ---
            if mode == "api_baseline":
                traj, _, _ = runner.run_trajectory(
                    input_text=sample['instruction'],
                    ground_truth_data=ground_truth_data,
                    max_steps=HP.EVAL_MAX_STEPS if hasattr(HP, 'EVAL_MAX_STEPS') else 10
                )
            else:
                traj, _, _ = runner.run_trajectory(
                    model=model,
                    processor=processor,
                    input_text=sample['instruction'],
                    ground_truth_data=ground_truth_data,
                    max_steps=10, 
                    temperature=0.0 
                )

            # --- Metrics ---
            is_success = (traj.failed == 0)
            reward = batch_compute_rewards([traj])[0] 
            
            metrics["total_steps"] += traj.step_count
            metrics["total_reward"] += reward
            if is_success: 
                metrics["success_count"] += 1
            else:
                if traj.failed in metrics["failed_reasons"]:
                    metrics["failed_reasons"][traj.failed] += 1

            # --- Visualization ---
            # Visualize on the raw image using correct absolute bbox
            vis_img = visualize_trajectory(
                base_image=raw_img,
                cursor_path=traj.cursor_path,
                actions=traj.tools,
                gt_bbox=final_bbox, 
                success=is_success,
                instruction=traj.global_question
            )
            
            status_str = "PASS" if is_success else "FAIL"
            img_filename = f"id_{i:04d}_{status_str}_rew{reward:.1f}.png"
            vis_img.save(os.path.join(img_save_dir, img_filename))

            # --- Log Result ---
            results.append({
                "id": i,
                "instruction": sample['instruction'],
                "success": is_success,
                "steps": traj.step_count,
                "reward": reward,
                "fail_code": traj.failed,
                "history": traj.tools,
                "vis_file": img_filename
            })
            
            # Clear Cache
            if i % 10 == 0: torch.cuda.empty_cache()

        except Exception as e:
            print(f"[EVAL] Error processing sample {i}: {e}")
            continue

    # 5. Final Report
    count = len(dataset)
    if count == 0: count = 1
    
    acc = metrics["success_count"] / count
    avg_steps = metrics["total_steps"] / count
    avg_rew = metrics["total_reward"] / count
    
    print("\n" + "="*40)
    print(f"EVAL REPORT: {mode.upper()}")
    print(f"Total Samples: {count}")
    print(f"Success Rate:  {acc:.2%}")
    print(f"Avg Steps:     {avg_steps:.2f}")
    print(f"Avg Reward:    {avg_rew:.2f}")
    print(f"Failure Stats: {metrics['failed_reasons']}")
    print("="*40 + "\n")

    # Save JSON
    report = {
        "meta": {
            "mode": mode,
            "model_path": model_path if mode != "api_baseline" else "API",
            "timestamp": timestamp,
            "num_samples": count
        },
        "metrics": {
            "accuracy": acc,
            "avg_steps": avg_steps,
            "avg_reward": avg_rew,
            "failed_breakdown": metrics["failed_reasons"]
        },
        "details": results
    }
    
    json_path = os.path.join(result_dir, "report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"[EVAL] Full report saved to: {json_path}")

if __name__ == "__main__":
    # Limit serves as a quick smoke test, can be set to None for full run
    evaluate_model(mode="trained", limit=HP.EVAL_DATASET_SIZE if hasattr(HP, 'EVAL_DATASET_SIZE') else 20)