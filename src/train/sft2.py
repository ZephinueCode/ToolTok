# src/train/sft2.py

import torch
import random
import math
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from PIL import Image

from torch.utils.data import Dataset
from transformers import (
    Qwen3VLForConditionalGeneration, 
    AutoProcessor, 
    Trainer, 
    TrainingArguments
)

# Project imports
from ..utils.parameters import HYPERPARAMS as HP
from ..utils.action import ACTION_TOKENS
from ..utils.action_logic import MOVE_DELTAS
from ..utils.prompts import AGENT_SYSTEM_PROMPT
from ..tools.visual_utils import draw_cursor
from ..utils.sft_screenspot import ScreenSpotDataManager

# =============================================================================
# 1. Logic Helpers (CoT & Pathing)
# =============================================================================

COT_TEMPLATES = {
    "intent": [
        "The user wants to '{instruction}'.",
        "Goal: Execute the command '{instruction}'.",
        "Instruction received: '{instruction}'.",
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
    "plan_move_far": [
        "There is a significant gap. I need a large jump.",
        "The distance is large. A long-range movement is required.",
    ],
    "plan_move_mid": [
        "The target is moderately away. I need a standard step.",
        "Medium distance detected. A normal move command is appropriate.",
    ],
    "plan_move_close": [
        "The target is very close. I need a micro-adjustment.",
        "Almost there. A fine-tuning step is needed.",
    ],
    "plan_click": [
        "I will perform a click.",
        "Interaction required. Clicking now.",
    ]
}

def get_grid_region(x: int, y: int, width: int, height: int) -> str:
    """Determine 3x3 grid region."""
    if y < height / 3: v_tag = "Top"
    elif y < 2 * height / 3: v_tag = "Mid"
    else: v_tag = "Bottom"
    
    if x < width / 3: h_tag = "Left"
    elif x < 2 * width / 3: h_tag = "Center"
    else: h_tag = "Right"
    
    if v_tag == "Mid" and h_tag == "Center": return "Center"
    elif v_tag == "Mid": return f"Mid-{h_tag}"
    else: return f"{v_tag}-{h_tag}"

def generate_cot_for_step(cursor_pos, target_pos, instruction, next_action_token, img_size):
    """
    Predict reasoning + next action based on INPUT state.
    """
    cx, cy = cursor_pos
    tx, ty = target_pos
    w, h = img_size
    
    cot = ""

    # --- 1. Intent & Localization ---
    t_intent = random.choice(COT_TEMPLATES["intent"])
    t_loc = random.choice(COT_TEMPLATES["localization"])
    
    c_region = get_grid_region(cx, cy, w, h)
    t_region = get_grid_region(tx, ty, w, h)
    
    cot += f"Reasoning: {t_intent.format(instruction=instruction)} "
    cot += t_loc.format(c_region=c_region, t_region=t_region) + " "
    
    # --- 2. Exact Relative Coordinates ---
    rel_cx, rel_cy = round(cx / w, 2), round(cy / h, 2)
    rel_tx, rel_ty = round(tx / w, 2), round(ty / h, 2)
    
    if c_region == t_region:
        cot += f"Refining position... Cursor: [{rel_cx:.1f}, {rel_cy:.1f}], Target: [{rel_tx:.1f}, {rel_ty:.1f}] (relative). "

    # --- 3. Spatial Direction ---
    dx, dy = tx - cx, ty - cy
    margin = 15
    v_rel = "Top" if dy < -margin else ("Bottom" if dy > margin else "")
    h_rel = "Left" if dx < -margin else ("Right" if dx > margin else "")
    
    if v_rel and h_rel: rel_pos_str = f"{v_rel}-{h_rel}"
    elif v_rel: rel_pos_str = v_rel
    elif h_rel: rel_pos_str = h_rel
    else: rel_pos_str = "Overlapping"

    if rel_pos_str != "Overlapping":
        cot += random.choice(COT_TEMPLATES["direction"]).format(rel_pos=rel_pos_str) + " "
    else:
        cot += random.choice(COT_TEMPLATES["arrival"]) + " "

    # --- 4. Direction Hints ---
    if "MOVE" in next_action_token:
        if abs(dx) > abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
        cot += f"Direction **{direction}** requires coverage. "

    # --- 5. Plan ---
    if "CLICK" in next_action_token:
        cot += random.choice(COT_TEMPLATES["plan_click"])
    else:
        if "FAR" in next_action_token:
            cot += random.choice(COT_TEMPLATES["plan_move_far"])
        elif "MID" in next_action_token:
            cot += random.choice(COT_TEMPLATES["plan_move_mid"])
        else:
            cot += random.choice(COT_TEMPLATES["plan_move_close"])

    return f"{cot}\nAction: {next_action_token}"

def get_shortest_path_actions_dynamic(start_pos, target_pos, img_size):
    """
    Greedy pathfinding respecting dynamic image boundaries.
    """
    cx, cy = start_pos
    tx, ty = target_pos
    w, h = img_size
    path = []
    
    valid_moves = [(k, v) for k, v in MOVE_DELTAS.items() if "MOVE" in k]
    valid_moves.sort(key=lambda x: math.hypot(x[1][0], x[1][1]), reverse=True)

    steps = 0
    max_steps = 10 
    
    while steps < max_steps:
        dist = math.hypot(tx - cx, ty - cy)
        if dist <= 20: break 
        
        best_token = None
        best_pos = (cx, cy)
        min_dist = dist
        found = False
        
        for token, (mx, my) in valid_moves:
            nx, ny = cx + mx, cy + my
            if not (0 <= nx < w and 0 <= ny < h): continue
            
            rem = math.hypot(tx - nx, ty - ny)
            if rem < min_dist:
                min_dist = rem
                best_token = token
                best_pos = (nx, ny)
                found = True
        
        if found:
            path.append((best_token, best_pos))
            cx, cy = best_pos
            steps += 1
        else:
            break
            
    path.append(("<CLICK_SHORT>", (cx, cy)))
    return path

# =============================================================================
# 2. Weighted Trainer with Separate Logging
# =============================================================================

class WeightedActionTrainer(Trainer):
    """
    Applies custom weighting and Logs 'Action Loss' vs 'CoT Loss'.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.my_tokenizer = self.tokenizer if self.tokenizer else getattr(self, "processing_class", None)
        if hasattr(self.my_tokenizer, "tokenizer"):
            self.my_tokenizer = self.my_tokenizer.tokenizer
            
        # Cache action token IDs
        self.valid_action_ids = set()
        for t in ACTION_TOKENS:
            ids = self.my_tokenizer.encode(t, add_special_tokens=False)
            if ids: self.valid_action_ids.add(ids[-1])
            ids_sp = self.my_tokenizer.encode(" " + t, add_special_tokens=False)
            if ids_sp: self.valid_action_ids.add(ids_sp[-1])
        
        # Blacklist IDs for heuristic search
        self.im_end = self.my_tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eos = self.my_tokenizer.eos_token_id
        self.nl = 198

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Shift for Causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Clean up graph
        del logits

        # 1. Calculate Per-Token Loss (Raw)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        # Shape: [Batch_Size, Seq_Len]
        raw_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        raw_loss = raw_loss.view(labels.size(0), -1)
        
        del shift_logits

        # 2. Build Weight Map
        weights = torch.ones_like(raw_loss)
        # Using 20.0 as discussed
        ACTION_WEIGHT = 80.0 
        
        # Masks for logging
        action_mask_log = torch.zeros_like(raw_loss, dtype=torch.bool)

        for i in range(labels.size(0)):
            valid_mask = (shift_labels[i] != -100)
            valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(1)
            
            if len(valid_idx) > 0:
                target_idx = None
                # Heuristic: Search last 15 tokens for Action
                search = torch.flip(valid_idx, [0])[:15]
                
                # Pass 1: Exact Match
                for idx in search:
                    if shift_labels[i, idx].item() in self.valid_action_ids:
                        target_idx = idx
                        break
                
                # Pass 2: Fallback (non-special tokens)
                if target_idx is None:
                    for idx in search:
                        tid = shift_labels[i, idx].item()
                        if tid not in [self.im_end, self.eos, self.nl]:
                            target_idx = idx
                            break
                
                if target_idx is not None:
                    weights[i, target_idx] = ACTION_WEIGHT
                    action_mask_log[i, target_idx] = True

        # 3. Calculate Final Weighted Loss
        valid_mask = (shift_labels != -100).float()
        final_weights = weights * valid_mask
        
        total_loss = (raw_loss * final_weights).sum() / (final_weights.sum() + 1e-8)

        # 4. [NEW] Logging Logic: Split Action vs CoT
        # We check if we are in a logging step
        if self.state.global_step % self.args.logging_steps == 0:
            with torch.no_grad():
                # Action Loss: Avg loss of action tokens only
                act_mask = action_mask_log & (shift_labels != -100)
                if act_mask.sum() > 0:
                    act_loss_val = (raw_loss * act_mask.float()).sum() / act_mask.sum()
                else:
                    act_loss_val = 0.0
                
                # CoT Loss: Avg loss of non-action tokens (but valid)
                cot_mask = (~action_mask_log) & (shift_labels != -100)
                if cot_mask.sum() > 0:
                    cot_loss_val = (raw_loss * cot_mask.float()).sum() / cot_mask.sum()
                else:
                    cot_loss_val = 0.0
                
                # Print to console (Simple & Effective)
                if self.is_world_process_zero():
                    print(
                        f"\n[Step {self.state.global_step}] "
                        f"Total Weighted: {total_loss.item():.4f} | "
                        f"Action Loss: {act_loss_val:.4f} | "
                        f"CoT Loss: {cot_loss_val:.4f}"
                    )

        return (total_loss, outputs) if return_outputs else total_loss

# =============================================================================
# 3. Dataset & Collator
# =============================================================================

@dataclass
class SFTDataCollator:
    processor: AutoProcessor
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts, images = [], []
        for f in features:
            msgs = f["messages"]
            fmt = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": msgs[1]["content"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": msgs[2]["content"]}]}
            ]
            txt = self.processor.apply_chat_template(fmt, tokenize=False, add_generation_prompt=False)
            texts.append(f"{AGENT_SYSTEM_PROMPT}\n{txt}")
            images.append(f["image"])
            
        batch = self.processor(
            text=texts, images=images, padding=True, truncation=True, max_length=8192, return_tensors="pt"
        )
        
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        im_start = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        for i in range(len(labels)):
            starts = (batch["input_ids"][i] == im_start).nonzero(as_tuple=True)[0]
            if len(starts) >= 2:
                labels[i, :starts[-1] + 1] = -100 
        
        batch["labels"] = labels
        return batch

class ScreenSpotSFTDataset(Dataset):
    def __init__(self, split="train"):
        self.ss_manager = ScreenSpotDataManager()
        self.split = split
        self.data = self.ss_manager.raw_train if split == "train" else self.ss_manager.raw_eval
        print(f"[SFT-2 Dataset] Split '{split}': {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._process_sample(self.data[idx])

    def _process_sample(self, sample):
        raw_img = sample['image'].convert("RGB")
        
        # --- Random Resizing Logic ---
        orig_w, orig_h = raw_img.size
        curr_max_dim = max(orig_w, orig_h)
        
        # Decide target size based on probability
        target_max_side = 1600

        # Calculate scale and resize if needed
        scale = 1.0
        if curr_max_dim > target_max_side:
            scale = target_max_side / curr_max_dim
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            raw_img = raw_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Update width/height for coordinate calculation
        w, h = raw_img.size
        
        # --- BBox & Coordinates ---
        bbox = sample.get('bbox', sample.get('point', [0,0]*2))
        if len(bbox) == 2: bbox = [bbox[0], bbox[1], bbox[0], bbox[1]]
        
        if all(0.0 <= c <= 1.0 for c in bbox):
            # If normalized, use new width/height directly
            tx = int((bbox[0]*w + bbox[2]*w)/2)
            ty = int((bbox[1]*h + bbox[3]*h)/2)
        else:
            # If absolute, apply scale factor
            if scale != 1.0:
                tx = int(((bbox[0] + bbox[2])/2) * scale)
                ty = int(((bbox[1] + bbox[3])/2) * scale)
            else:
                tx = int((bbox[0] + bbox[2])/2)
                ty = int((bbox[1] + bbox[3])/2)
                
        tx = max(0, min(w-1, tx))
        ty = max(0, min(h-1, ty))

        # --- [NEW] Randomized Start Position Logic ---
        # 1. 50% Random Position: Simulates mid-task state
        # 2. 40% Opposite Corner: Forces LONG distance (FAR tokens)
        # 3. 10% Center: Simulates clean start
        
        rand_val = random.random()
        
        if rand_val < 0.1:
            start_x = random.randint(0, w - 1)
            start_y = random.randint(0, h - 1)
        elif rand_val < 0.2:
            # Opposite corner logic to force FAR tokens
            # If target is Left (0..w/2), start at Right edge
            # If target is Top (0..h/2), start at Bottom edge
            margin_w = max(10, int(w * 0.1))
            margin_h = max(10, int(h * 0.1))
            
            if tx < w // 2: 
                start_x = random.randint(w - margin_w, w - 1)
            else:           
                start_x = random.randint(0, margin_w)
            
            if ty < h // 2: 
                start_y = random.randint(h - margin_h, h - 1)
            else:           
                start_y = random.randint(0, margin_h)
        else:
            # Traditional Center Start
            start_x, start_y = w // 2, h // 2
            
        # Clamp to bounds just in case
        start_x = max(0, min(w-1, start_x))
        start_y = max(0, min(h-1, start_y))

        # --- Path Generation ---
        # Use the randomized start position
        full_path = get_shortest_path_actions_dynamic((start_x, start_y), (tx, ty), (w, h))
        
        # --- Step Selection ---
        if not full_path:
            action_token = "<CLICK_SHORT>"
            history_tokens = []
            curr_pos = (start_x, start_y)
        else:
            if random.random() < 0.2:
                step_idx = len(full_path) - 1
            else:
                step_idx = random.randint(0, max(0, len(full_path) - 2))
            
            target_step = full_path[step_idx]
            action_token = target_step[0]
            
            history_tokens = []
            # Initialize loop from randomized start
            curr_cx, curr_cy = start_x, start_y
            
            for i in range(step_idx):
                h_token, (nx, ny) = full_path[i]
                history_tokens.append(h_token)
                curr_cx, curr_cy = nx, ny
            
            curr_pos = (curr_cx, curr_cy)

        # --- Prompt Formatting ---
        if not history_tokens:
            history_str = "None (Start)"
        else:
            history_str = " -> ".join(history_tokens)
        
        user_content = f"[Action] Task: {sample['instruction']}\nPrevious Actions: {history_str}"

        # --- Vis & CoT ---
        image = draw_cursor(raw_img, curr_pos[0], curr_pos[1])
        
        cot = generate_cot_for_step(
            cursor_pos=curr_pos,
            target_pos=(tx, ty),
            instruction=sample['instruction'],
            next_action_token=action_token,
            img_size=(w, h)
        )
        
        msgs = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}, 
            {"role": "assistant", "content": cot}
        ]
        
        return {"messages": msgs, "image": image}

# =============================================================================
# 4. Execution
# =============================================================================

def run_sft_screenspot():
    if not os.path.exists(HP.SFT_2_INPUT_PATH):
        print(f"[Error] Model not found: {HP.SFT_2_INPUT_PATH}")
        return

    print(f"[SFT-2] Loading model from {HP.SFT_2_INPUT_PATH}...")
    # NOTE: device_map="auto" REMOVED for DDP/Accelerate compatibility
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        HP.SFT_2_INPUT_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(HP.SFT_2_INPUT_PATH, trust_remote_code=True)
    
    # Gradient Checkpointing (Save VRAM)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # Freeze Visual, Train LLM + Embeddings
    model.get_input_embeddings().weight.requires_grad = True
    for name, param in model.named_parameters():
        if "visual" in name: param.requires_grad = False
        else: param.requires_grad = True
        
    train_ds = ScreenSpotSFTDataset("train")
    eval_ds = ScreenSpotSFTDataset("eval")
    collator = SFTDataCollator(processor)
    
    args = TrainingArguments(
        output_dir=HP.SFT_2_OUTPUT_PATH,
        num_train_epochs=HP.SFT_2_EPOCHS,
        per_device_train_batch_size=HP.SFT_2_BATCH_SIZE,
        gradient_accumulation_steps=HP.SFT_2_GRAD_ACCUM_STEPS,
        learning_rate=HP.SFT_2_LEARN_RATE,
        bf16=True,
        logging_steps=5, # Logs will appear every 5 steps
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=240,
        eval_steps=240,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        prediction_loss_only=True,
        warmup_ratio=0.1
    )
    
    trainer = WeightedActionTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, 
        data_collator=collator, processing_class=processor
    )
    
    print("[SFT-2] Starting Training with Detailed Loss Logging...")
    trainer.train()
    
    trainer.save_model(HP.SFT_2_OUTPUT_PATH)
    processor.save_pretrained(HP.SFT_2_OUTPUT_PATH)
    train_ds.ss_manager.save_test_set()

if __name__ == "__main__":
    run_sft_screenspot()