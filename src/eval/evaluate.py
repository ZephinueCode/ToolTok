# src/eval/evaluate.py

import torch
import os
import json
from tqdm import tqdm
from datetime import datetime
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from ..utils.parameters import HYPERPARAMS as HP
from ..utils.dataset import prepare_grpo1_dataset
from ..tools.runner import Runner, ToolData, AgentTrajectory
from ..train.reward import batch_compute_rewards
from .baseline_vlm import BaselineAPIRunner

def evaluate_model(mode="trained", num_samples=None):
    """
    Args:
        mode: 
            "trained"     -> Local SFT/RL Model + Local Runner.
            "api_baseline"-> 235B API Model + Zero-Shot Prompt.
    """
    # 1. Config & Init
    dataset = prepare_grpo1_dataset(HP.EVAL_DATA_PATH, size=num_samples)
    
    runner = None
    model = None
    processor = None
    
    if mode == "api_baseline":
        print(f"[EVAL] Mode: API BASELINE (Qwen-235B)")
        runner = BaselineAPIRunner()
        # No model/processor needed for API runner (it handles client internally)
        
    else: # trained
        model_path = HP.GRPO1_OUTPUT_PATH # Default to GRPO, or change to SFT
        print(f"[EVAL] Mode: TRAINED MODEL ({model_path})")
        runner = Runner()
        
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading local model: {e}")
            return
    
    print(f"[EVAL] Dataset size: {len(dataset)}")
    
    # 2. Evaluation Loop
    results = []
    success_count = 0
    total_steps = 0
    total_reward = 0
    
    # Use tqdm for progress
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        
        # Run Trajectory
        # Note: API Runner ignores model/processor args
        if mode == "api_baseline":
            traj, _, _ = runner.run_trajectory(
                input_text=sample['question'],
                ground_truth_data=sample['ground_truth_traj'],
                max_steps=HP.EVAL_MAX_STEPS
            )
        else:
            tool_input = ToolData(image=sample['image'], text=sample['question'])
            traj, _, _ = runner.run_trajectory(
                model=model,
                processor=processor,
                input_data=tool_input,
                ground_truth_bbox=sample['ground_truth_traj'], # Local runner specific arg name
                max_steps=HP.EVAL_MAX_STEPS,
                temperature=0.0
            )
        
        # Compute Metric
        is_success = (traj.failed == 0)
        if is_success: success_count += 1
        total_steps += traj.step_count
        reward = batch_compute_rewards([traj])[0]
        total_reward += reward
        
        results.append({
            "id": i,
            "instruction": sample['question'],
            "success": is_success,
            "steps": traj.step_count,
            "reward": reward,
            "fail_reason": traj.failed,
            "history": traj.tools
        })
        
    # 3. Summary
    count = len(dataset)
    accuracy = success_count / count
    avg_steps = total_steps / count
    avg_reward = total_reward / count
    
    print("\n" + "="*40)
    print(f"EVAL REPORT: {mode.upper()}")
    print(f"Accuracy:   {accuracy:.2%}")
    print(f"Avg Steps:  {avg_steps:.2f}")
    print(f"Avg Reward: {avg_reward:.2f}")
    print("="*40 + "\n")
    
    # 4. Save
    os.makedirs(HP.EVAL_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(HP.EVAL_OUTPUT_DIR, f"{mode}_eval_{timestamp}.json")
    
    with open(save_path, "w") as f:
        json.dump({
            "meta": {"mode": mode, "samples": count},
            "metrics": {"accuracy": accuracy, "avg_steps": avg_steps, "avg_reward": avg_reward},
            "details": results
        }, f, indent=2)
    
    print(f"Saved detailed results to {save_path}")

if __name__ == "__main__":
    # Usage:
    # 1. To test API Baseline:
    evaluate_model(mode="api_baseline", num_samples=HP.EVAL_DATASET_SIZE)
    
    # 2. To test Trained Model:
    # evaluate_model(mode="trained", num_samples=HP.EVAL_DATASET_SIZE)