# src/eval/eval_holo2.py

import torch
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from pydantic import BaseModel, Field
from typing import Any, Literal, TypeAlias, Tuple

# Transformers & Holo2 dependencies
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

# Import your existing utilities
from ..utils.parameters import HYPERPARAMS as HP
from ..tools.visual_utils import visualize_trajectory
from ..tools.runner import AgentTrajectory, GTTrajectory, GTStep
from ..train.reward import batch_compute_rewards

# [NEW] Import the prompt from utils
from ..utils.prompts import BASELINE_GROUNDING_PROMPT

# =============================================================================
# 1. Holo2 Specific Logic (Copied & Adapted from Cookbook)
# =============================================================================

class ClickCoordinates(BaseModel):
    x: int = Field(ge=0, le=1000, description="The x coordinate, normalized between 0 and 1000.")
    y: int = Field(ge=0, le=1000, description="The y coordinate, normalized between 0 and 1000.")

class Holo2Predictor:
    def __init__(self, model_path="Hcompany/Holo2-4B", device="cuda"):
        print(f"[Holo2] Loading model: {model_path}...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.image_processor_config = self.processor.image_processor
        self.device = self.model.device

    def get_chat_messages(self, task: str, image: Image.Image) -> list[dict]:
        """Create the prompt structure for navigation task using Baseline Prompt"""
        
        # [MODIFIED] Use BASELINE_GROUNDING_PROMPT instead of hardcoded schema
        # Holo2 coordinates are 0-1000, so we inject that into the prompt template
        system_prompt = BASELINE_GROUNDING_PROMPT.replace("{WIDTH}", "1000").replace("{HEIGHT}", "1000")

        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"User Query: {task}"},
                ],
            },
        ]

    def parse_reasoning(self, generated_ids: torch.tensor) -> Tuple[str, str]:
        """Parse content from generated_ids, handling Think tokens"""
        all_ids = generated_ids[0].tolist()
        
        # Holo2 specific tokens for thinking blocks
        start_token = 151667
        end_token = 151668
        
        try:
            think_start_index = all_ids.index(start_token)
            try:
                think_end_index = all_ids.index(end_token)
            except ValueError:
                # If no end token, assume rest is thinking
                think_end_index = len(all_ids) 
            
            thinking_content = self.processor.decode(all_ids[think_start_index+1:think_end_index], skip_special_tokens=True).strip("\n")
            # Content comes after thinking
            content_ids = all_ids[think_end_index+1:]
        except ValueError:
            # No thinking block found
            thinking_content = ""
            content_ids = all_ids

        content = self.processor.decode(content_ids, skip_special_tokens=True).strip("\n")
        return content, thinking_content

    def predict(self, image: Image.Image, task: str) -> Tuple[dict, str]:
        """
        Returns: 
            result_dict: {'x': int, 'y': int} (normalized 0-1000) or None if failed
            thinking: str
        """
        # 1. Resize logic
        w, h = image.size
        resized_height, resized_width = smart_resize(
            h, w,
            factor=self.image_processor_config.patch_size * self.image_processor_config.merge_size,
            min_pixels=self.image_processor_config.size.get("shortest_edge", None),
            max_pixels=self.image_processor_config.size.get("longest_edge", None),
        )
        processed_image = image.resize(size=(resized_width, resized_height), resample=Image.Resampling.LANCZOS)

        # 2. Prepare inputs
        messages = self.get_chat_messages(task, processed_image)
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, thinking=False
        )
        
        inputs = self.processor(
            text=[text_prompt],
            images=[processed_image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # 3. Inference
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        # 4. Parse
        content, thinking = self.parse_reasoning(generated_ids)
        
        # 5. Extract JSON
        # Note: If the model adheres to the Baseline Prompt, it might output a <tool_call> or similar.
        # This parser currently attempts to find a JSON object.
        try:
            # Strip Markdown code blocks if present
            clean_content = content.replace("```json", "").replace("```", "").strip()
            
            # [ADAPTATION] If model outputs tool call style but we want coords:
            # Attempt generic JSON parse first.
            action = ClickCoordinates.model_validate_json(clean_content)
            return {"x": action.x, "y": action.y}, thinking
        except Exception as e:
            content = content.strip()
            Lbrace_index = content.rfind('{')

            if Lbrace_index != -1:
                content = content[Lbrace_index:]
                Rbrace_index = content.find('}')
                if Rbrace_index != -1:
                    content = content[:Rbrace_index + 1]
                    try:
                        # Attempt to map potential "coordinate": [x, y] to x, y if structure differs
                        data = json.loads(content)
                        if "x" in data and "y" in data:
                             return {"x": data["x"], "y": data["y"]}, thinking
                        elif "coordinate" in data and isinstance(data["coordinate"], list):
                             return {"x": data["coordinate"][0], "y": data["coordinate"][1]}, thinking
                        
                        # Fallback to Pydantic
                        action = ClickCoordinates.model_validate_json(content)
                        return {"x": action.x, "y": action.y}, thinking
                    except Exception as second_exception:
                        print(f"[Holo2 Error] Second JSON Parse failed: {content} | Error: {second_exception}")
                        return None, thinking
                else:
                    print("[Holo2 Error] No closing '}' found in content.")
                    return None, thinking
            else:
                print("[Holo2 Error] No '{' field found in content.")
                return None, thinking

# =============================================================================
# 2. Evaluation Main Loop
# =============================================================================

def eval_holo2(dataset_name, limit=None, model_path="Hcompany/Holo2-4B", bbox_expansion=0.05):
    
    # --- Setup Directories ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./results/eval/holo2_{dataset_name}_{timestamp}"
    img_save_dir = os.path.join(save_dir, "images")
    os.makedirs(img_save_dir, exist_ok=True)

    print(f"=== Holo2 Evaluation: {dataset_name} ===")
    print(f"Model: {model_path}")
    print(f"Save Dir: {save_dir}")

    # --- Load Data Manager based on args ---
    if dataset_name == "screenspot":
        from ..utils.sft_screenspot import ScreenSpotDataManager
        dm = ScreenSpotDataManager()
        dataset = dm.raw_test
    elif dataset_name == "screenspot_pro":
        try:
            from ..utils.sft_screenspot_pro import ScreenSpotDataManager
            dm = ScreenSpotDataManager()
            dataset = dm.raw_test
        except ImportError:
            print("ScreenSpot Pro not found, falling back to standard.")
            from ..utils.sft_screenspot import ScreenSpotDataManager
            dm = ScreenSpotDataManager()
            dataset = dm.raw_test
    elif dataset_name == "mind2web":
        from ..utils.sft_m2w import Mind2WebDataManager
        dm = Mind2WebDataManager()
        dataset = dm.raw_test
    elif dataset_name == "screenspot_v2":
        from ..utils.sft_screenspot_v2 import ScreenSpotDataManager
        dm = ScreenSpotDataManager()
        dataset = dm.raw_test
    else:
        raise ValueError("Unknown dataset")

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # --- Initialize Model ---
    predictor = Holo2Predictor(model_path=model_path)

    # --- Metrics ---
    metrics = {
        "success": 0,
        "total": 0,
        "failed_json": 0,
        "total_reward": 0
    }
    results = []

    # --- Loop ---
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        # 1. Image Loading (Robustness check)
        raw_img = None
        if "image_path" in sample and sample["image_path"]:
            try: raw_img = Image.open(sample["image_path"]).convert("RGB")
            except: pass
        if raw_img is None and "image" in sample:
            img_data = sample["image"]
            raw_img = img_data.convert("RGB") if isinstance(img_data, Image.Image) else Image.open(img_data).convert("RGB")
        
        if raw_img is None: continue

        w, h = raw_img.size

        # 2. Prepare GT BBox (Absolute Coordinates)
        bbox = sample.get('bbox', None)
        if not bbox and 'point' in sample:
            p = sample['point']
            bbox = [p[0], p[1], p[0], p[1]]
        
        final_bbox = [0,0,0,0]
        if bbox:
            if all(0.0 <= x <= 1.0 for x in bbox):
                final_bbox = [bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h]
            else:
                final_bbox = [
                    bbox[0], bbox[1],
                    bbox[0] + bbox[2], bbox[1] + bbox[3]
                ]
        
        # BBox Expansion (Relaxation)
        if bbox_expansion > 0:
            mx, my = w * bbox_expansion, h * bbox_expansion
            final_bbox = [
                max(0, final_bbox[0] - mx), max(0, final_bbox[1] - my),
                min(w, final_bbox[2] + mx), min(h, final_bbox[3] + my)
            ]

        # 3. Inference
        instruction = sample['instruction']
        pred_norm, thinking = predictor.predict(raw_img, instruction)
        
        # 4. Check Result
        is_success = False
        fail_reason = 0 # 0: Success, 1: JSON Fail, 3: Miss
        pred_abs_x, pred_abs_y = 0, 0
        
        if pred_norm is None:
            metrics["failed_json"] += 1
            fail_reason = 1
            action_text = "JSON_PARSE_ERROR"
        else:
            # Convert 0-1000 to Absolute
            pred_abs_x = (pred_norm['x'] / 1000) * w
            pred_abs_y = (pred_norm['y'] / 1000) * h
            
            # Hit Check
            if (final_bbox[0] <= pred_abs_x <= final_bbox[2]) and \
               (final_bbox[1] <= pred_abs_y <= final_bbox[3]):
                is_success = True
                fail_reason = 0
            else:
                is_success = False
                fail_reason = 3 # Miss
            
            action_text = f"click({int(pred_abs_x)}, {int(pred_abs_y)})"

        # 5. Construct Synthetic Trajectory for Compatibility
        start_step = GTStep(
            image=raw_img,
            bbox=final_bbox,
            instruction=instruction,
            action_type="click" # Assume click for grounding eval
        )

        traj = AgentTrajectory(instruction, start_step, 0)
        
        # Manually overwrite trajectory path (since it's one-shot)
        traj.cursor_path = [(pred_abs_x, pred_abs_y)] 
        traj.tools = [action_text]
        traj.failed = fail_reason
        traj.step_count = 1
        
        # Compute Reward
        reward = 1.0 if is_success else 0.0

        metrics["total"] += 1
        metrics["total_reward"] += reward
        if is_success: metrics["success"] += 1

        # 6. Visualize
        vis_img = visualize_trajectory(
            base_image=raw_img,
            cursor_path=traj.cursor_path,
            actions=traj.tools,
            gt_bbox=final_bbox,
            success=is_success,
            instruction=instruction + f"\n[Think] {thinking[:50]}..."
        )
        status = "PASS" if is_success else "FAIL"
        img_filename = f"holo2_{i}_{status}.png"
        vis_img.save(os.path.join(img_save_dir, img_filename))

        # 7. Log
        results.append({
            "id": i,
            "instruction": instruction,
            "success": is_success,
            "pred_norm": pred_norm,
            "pred_abs": [pred_abs_x, pred_abs_y],
            "gt_bbox": final_bbox,
            "thinking": thinking,
            "reward": reward
        })
        
        if i % 10 == 0: torch.cuda.empty_cache()

    # --- Final Report ---
    acc = metrics["success"] / metrics["total"] if metrics["total"] > 0 else 0
    print("\n" + "="*40)
    print(f"HOLO2 EVAL REPORT: {dataset_name}")
    print(f"Accuracy: {acc:.2%}")
    print(f"JSON Failures: {metrics['failed_json']}")
    print("="*40)

    with open(os.path.join(save_dir, "report.json"), "w") as f:
        json.dump({"metrics": metrics, "details": results}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="screenspot", choices=["screenspot", "screenspot_pro", "mind2web", "screenspot_v2"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model_path", type=str, default="./checkpoints/Holo2-4B")
    parser.add_argument("--expansion", type=float, default=0.05)
    
    args = parser.parse_args()
    
    eval_holo2(
        dataset_name=args.dataset, 
        limit=args.limit, 
        model_path=args.model_path,
        bbox_expansion=args.expansion
    )