import torch
import random
import os
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

from torch.utils.data import Dataset
from transformers import (
    Qwen3VLForConditionalGeneration, 
    AutoProcessor, 
    TrainingArguments
)

# Project imports
from ..utils.parameters import HYPERPARAMS as HP
from ..utils.prompts import AGENT_SYSTEM_PROMPT
from ..tools.visual_utils import draw_cursor
from ..utils.sft_m2w import Mind2WebDataManager, get_m2w_trajectory

# Reuse Trainer/Collator from SFT3
from .sft3 import WeightedActionTrainer, SFTDataCollator
from .sft2 import ScreenSpotSFTDataset

# =============================================================================
# 1. CoT Logic
# =============================================================================

COT_TEMPLATES = {
    # Templates for general intent
    "intent": [
        "Instruction received. I need to breakdown the task.",
        "Analyzing user's intentions. I need to find the immediate next action.",
        "Mission start. Scanning the screenshot for what I should do the next step.",
    ],
    "localization": [
        "The cursor is currently in the **{c_region}** region, while the target lies in the **{t_region}** region.",
        "Scanning screen... Cursor found at **{c_region}**. Target identified at **{t_region}**.",
        "Position check: Cursor is at **{c_region}**; Target is at **{t_region}**.",
    ],
    "direction": [
        "The target is located to the **{rel_pos}** of the current cursor position.",
        "To reach the target, I need to move towards the **{rel_pos}**.",
        "There is a spatial offset. The target is **{rel_pos}** relative to the cursor.",
    ],
    "arrival": [
        "The cursor is positioned **over** the target.",
        "Target acquired. The cursor is aligned with the element.",
        "Zero distance. The cursor is exactly where it needs to be.",
    ],
    "plan_move": [
        "Action: Adjust position to close the gap.",
        "Plan: Move cursor towards the target.",
    ],
    "plan_click": [
        "Target reached. Executing click.",
        "Interaction required. Clicking now.",
    ],
    "plan_long_click": [
        "Target reached. Holding click for context menu.",
        "Interaction required. Executing long press.",
    ],
    "plan_text": [
        "Target is an input field. Typing content: \"{text}\".",
        "Text entry required. Inputting: \"{text}\".",
    ]
}

def get_grid_region(x: int, y: int, width: int, height: int) -> str:
    if y < height / 3: v_tag = "Top"
    elif y < 2 * height / 3: v_tag = "Mid"
    else: v_tag = "Bottom"
    if x < width / 3: h_tag = "Left"
    elif x < 2 * width / 3: h_tag = "Center"
    else: h_tag = "Right"
    if v_tag == "Mid" and h_tag == "Center": return "Center"
    elif v_tag == "Mid": return f"Mid-{h_tag}"
    else: return f"{v_tag}-{h_tag}"

def generate_sft2_style_cot(
    cursor_pos: Tuple[int, int], 
    target_bbox: List[float], 
    next_action_token: str, 
    instruction: str, 
    img_size: Tuple[int, int],
    text_content: Optional[str] = None,
    element_id: str = "unknown"  # [NEW] Accept element ID
) -> str:
    """
    Generates Chain-of-Thought reasoning string.
    [NEW] Injects the 'element_id' into the reasoning chain.
    """
    cx, cy = cursor_pos
    w, h = img_size
    x1, y1, x2, y2 = target_bbox
    tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
    
    cot = ""
    
    # 1. Intent + [NEW] ID Injection
    t_intent = random.choice(COT_TEMPLATES["intent"])
    base_intent = t_intent.format(instruction=instruction)
    # Injecting the requested phrase:
    cot += f"Reasoning: {base_intent} I need to get to {element_id}. "

    # 2. Localization
    t_loc = random.choice(COT_TEMPLATES["localization"])
    c_region = get_grid_region(cx, cy, w, h)
    t_region = get_grid_region(tx, ty, w, h)
    cot += t_loc.format(c_region=c_region, t_region=t_region) + " "
    
    if c_region == t_region:
        rel_cx, rel_cy = round(cx / w, 2), round(cy / h, 2)
        rel_tx, rel_ty = round(tx / w, 2), round(ty / h, 2)
        cot += f"Refining position... Cursor: [{rel_cx:.1f}, {rel_cy:.1f}], Target: [{rel_tx:.1f}, {rel_ty:.1f}] (relative). "
    
    # 3. Direction / Arrival
    dx, dy = tx - cx, ty - cy
    dist = math.hypot(dx, dy)
    is_near = dist <= 60 
    
    if not is_near:
        margin = 15
        v_rel = "Top" if dy < -margin else ("Bottom" if dy > margin else "")
        h_rel = "Left" if dx < -margin else ("Right" if dx > margin else "")
        if v_rel and h_rel: rel_pos_str = f"{v_rel}-{h_rel}"
        elif v_rel: rel_pos_str = v_rel
        elif h_rel: rel_pos_str = h_rel
        else: rel_pos_str = "Overlapping"
        cot += random.choice(COT_TEMPLATES["direction"]).format(rel_pos=rel_pos_str) + " "
    else:
        cot += random.choice(COT_TEMPLATES["arrival"]) + " "

    # 4. Action Plan
    if "<TEXT" in next_action_token or (text_content is not None):
        t_str = text_content if text_content else "..."
        cot += random.choice(COT_TEMPLATES["plan_text"]).format(text=t_str)
    elif "<CLICK_LONG>" in next_action_token:
        cot += random.choice(COT_TEMPLATES["plan_long_click"])
    elif "<CLICK" in next_action_token: 
        cot += random.choice(COT_TEMPLATES["plan_click"])
    else: 
        cot += random.choice(COT_TEMPLATES["plan_move"])
        if abs(dx) > abs(dy): direction = "RIGHT" if dx > 0 else "LEFT"
        else: direction = "DOWN" if dy > 0 else "UP"
        cot += f" Direction: {direction}."
    
    return f"{cot} I'm ready for action.\nAction: {next_action_token}"

# =============================================================================
# 2. Dataset Implementation (M2W Only)
# =============================================================================

class Mind2WebSFTDataset(Dataset):
    def __init__(self, split="train"):
        self.manager = Mind2WebDataManager()
        self.data_source = self.manager.raw_train if split == "train" else self.manager.raw_eval
        print(f"[M2W Dataset] Split '{split}': {len(self.data_source)} samples.")

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        return self._process_sample(self.data_source[idx])

    def _process_sample(self, sample):
        # 1. Load Image
        try:
            raw_img = Image.open(sample['image_path']).convert("RGB")
        except:
            raw_img = Image.new("RGB", (HP.IMAGE_SIZE, HP.IMAGE_SIZE), (0, 0, 0))
        w, h = raw_img.size
        
        # 2. Extract Labels
        bbox = sample['bbox']
        instruction = sample['instruction']
        final_action_type = sample['action_type'] 
        text_value = sample.get('action_value', None)
        target_id = sample.get('target_id', 'unknown_element') # [NEW] Get ID

        # 3. Generate Trajectory
        start_pos = (w // 2, h // 2)
        full_path = get_m2w_trajectory(
            start_pos=start_pos,
            target_bbox=bbox,
            action_type=final_action_type,
            text_value=text_value,
            img_size=(w, h)
        )

        # 4. Sampling
        token_steps = [s for s in full_path if not s[0].startswith("STR:")]
        if not token_steps:
            curr_pos = start_pos
            action_token = "<CLICK_SHORT>"
            history_str = "None"
        else:
            is_interaction = (len(token_steps) > 1 and random.random() < 0.3)
            if is_interaction: step_idx = len(token_steps) - 1
            else: step_idx = random.randint(0, max(0, len(token_steps) - 2))
            
            target_step = token_steps[step_idx]
            action_token = target_step[0]
            
            if step_idx == 0:
                curr_pos = start_pos
                history_tokens = []
            else:
                prev_step = token_steps[step_idx - 1]
                curr_pos = prev_step[1] if isinstance(prev_step[1], tuple) else start_pos
                history_tokens = [s[0] for s in token_steps[:step_idx]]
            history_str = " -> ".join(history_tokens[-5:]) if history_tokens else "None"

        # 5. Visual
        image = draw_cursor(raw_img, curr_pos[0], curr_pos[1])
        
        # 6. Prompt Generation
        user_content = f"Task: {instruction}\nHistory: {history_str}"
        
        # [NEW] Pass target_id to CoT generator
        cot_response = generate_sft2_style_cot(
            cursor_pos=curr_pos,
            target_bbox=bbox,
            next_action_token=action_token,
            instruction=instruction,
            img_size=(w, h),
            text_content=text_value,
            element_id=target_id 
        )
        
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": cot_response}
        ]
        return {"messages": messages, "image": image}

# =============================================================================
# 3. Mixed Dataset (M2W + SFT3)
# =============================================================================

class MixedSFTDataset(Dataset):
    """
    Wrapper Dataset to mix Mind2Web (SFT4) with ScreenSpot (SFT3).
    Target Ratio: 3:1
    """
    def __init__(self, split="train"):
        print(f"\n[Mixed Dataset] Initializing mixing strategy for split '{split}'...")
        
        self.m2w_ds = Mind2WebSFTDataset(split)
        self.sft2_ds = ScreenSpotSFTDataset(split)
        
        self.mixed_indices = []
        
        m2w_len = len(self.m2w_ds)
        target_sft3_len = int(m2w_len / 3)
        
        print(f"[Mixed Dataset] M2W Count: {m2w_len} | Target SFT3 Count: {target_sft3_len}")
        
        # Add M2W indices
        for i in range(m2w_len):
            self.mixed_indices.append(('m2w', i))
            
        # Add SFT3 indices (Cycle if needed)
        sft3_real_len = len(self.sft2_ds)
        if sft3_real_len > 0:
            for i in range(target_sft3_len):
                real_idx = i % sft3_real_len
                self.mixed_indices.append(('sft3', real_idx))
        
        random.shuffle(self.mixed_indices)

    def __len__(self):
        return len(self.mixed_indices)

    def __getitem__(self, idx):
        ds_type, real_idx = self.mixed_indices[idx]
        if ds_type == 'm2w':
            return self.m2w_ds[real_idx]
        else:
            return self.sft2_ds[real_idx]

# =============================================================================
# 4. Main Execution
# =============================================================================

def run_sft_mind2web():
    input_path = HP.SFT_4_INPUT_PATH
    output_path = HP.SFT_4_OUTPUT_PATH
    
    print(f"[SFT-4] Loading Model from {input_path}...")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        input_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(input_path, trust_remote_code=True)
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # Freeze logic (Adjust if you have VRAM for vision training)
    for name, param in model.named_parameters():
        if "visual" in name: 
            param.requires_grad = False
        else: 
            param.requires_grad = True
            
    train_ds = MixedSFTDataset("train")
    eval_ds = MixedSFTDataset("eval")
    
    collator = SFTDataCollator(processor) 
    
    args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=HP.SFT_4_EPOCHS,
        per_device_train_batch_size=HP.SFT_4_BATCH_SIZE, 
        gradient_accumulation_steps=HP.SFT_4_GRAD_ACCUM_STEPS,
        learning_rate=HP.SFT_4_LEARN_RATE,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=800,
        eval_steps=800,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        warmup_ratio=0.1
    )
    
    trainer = WeightedActionTrainer(
        model=model, 
        args=args, 
        train_dataset=train_ds, 
        eval_dataset=eval_ds, 
        data_collator=collator, 
        processing_class=processor
    )
    
    print("[SFT-4] Starting Mixed Training...")
    trainer.train()
    
    trainer.save_model(output_path)
    processor.save_pretrained(output_path)
    print(f"[SFT-4] Training Complete. Saved to {output_path}")

if __name__ == "__main__":
    run_sft_mind2web()