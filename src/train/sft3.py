# src/train/sft3.py

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
from ..utils.sft_screenspot_pro import ScreenSpotDataManager

# Reuse logic from Phase 2 (for CoT generation)
from .sft2 import generate_cot_for_step

# =============================================================================
# 1. Local Helper: Dynamic Path Finder
# =============================================================================

def get_shortest_path_actions_dynamic(start_pos: Tuple[int, int], target_pos: Tuple[int, int], img_size: Tuple[int, int]) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Calculates greedy shortest path respecting DYNAMIC image boundaries.
    """
    cx, cy = start_pos
    tx, ty = target_pos
    w, h = img_size
    path = []
    
    valid_moves = [(k, v) for k, v in MOVE_DELTAS.items() if "MOVE" in k]
    valid_moves.sort(key=lambda x: math.hypot(x[1][0], x[1][1]), reverse=True)

    max_steps = 10 
    steps = 0
    
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
# 2. Weighted Trainer (Copied & Adapted from SFT2)
# =============================================================================

class WeightedActionTrainer(Trainer):
    """
    Applies custom weighting (60x) to Action Tokens and logs split losses.
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
        
        self.im_end = self.my_tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eos = self.my_tokenizer.eos_token_id
        self.nl = 198

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            del logits

            # 1. Raw Per-Token Loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            raw_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            raw_loss = raw_loss.view(labels.size(0), -1)
            
            del shift_logits 

            # 2. Apply Weights
            weights = torch.ones_like(raw_loss)
            ACTION_WEIGHT = 60.0  
            
            action_mask_log = torch.zeros_like(raw_loss, dtype=torch.bool)

            for i in range(labels.size(0)):
                valid_mask = (shift_labels[i] != -100)
                valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(1)
                
                if len(valid_idx) > 0:
                    target_idx = None
                    search = torch.flip(valid_idx, [0])[:15] 
                    
                    # Priority 1: Exact Action Token Match
                    for idx in search:
                        if shift_labels[i, idx].item() in self.valid_action_ids:
                            target_idx = idx
                            break
                    
                    # Priority 2: Fallback
                    if target_idx is None:
                        for idx in search:
                            tid = shift_labels[i, idx].item()
                            if tid not in [self.im_end, self.eos, self.nl]:
                                target_idx = idx
                                break
                    
                    if target_idx is not None:
                        weights[i, target_idx] = ACTION_WEIGHT
                        action_mask_log[i, target_idx] = True

            # 3. Final Weighted Loss
            valid_mask = (shift_labels != -100).float()
            final_weights = weights * valid_mask
            total_loss = (raw_loss * final_weights).sum() / (final_weights.sum() + 1e-8)

            # 4. Logging (Action vs CoT)
            if self.state.global_step % self.args.logging_steps == 0:
                with torch.no_grad():
                    act_mask = action_mask_log & (shift_labels != -100)
                    if act_mask.sum() > 0:
                        act_loss_val = (raw_loss * act_mask.float()).sum() / act_mask.sum()
                    else:
                        act_loss_val = 0.0
                    
                    cot_mask = (~action_mask_log) & (shift_labels != -100)
                    if cot_mask.sum() > 0:
                        cot_loss_val = (raw_loss * cot_mask.float()).sum() / cot_mask.sum()
                    else:
                        cot_loss_val = 0.0
                    
                    if self.is_world_process_zero():
                        print(
                            f"\n[Step {self.state.global_step}] "
                            f"Total: {total_loss.item():.4f} | "
                            f"Act Loss: {act_loss_val:.4f} | "
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

class ScreenSpotProSFTDataset(Dataset):
    def __init__(self, split="train"):
        self.ss_manager = ScreenSpotDataManager()
        self.split = split
        self.data_source = self.ss_manager.raw_train if split == "train" else self.ss_manager.raw_eval
        print(f"[SFT-3 Dataset] Split '{split}': {len(self.data_source)} base samples.")

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        return self._process_sample(self.data_source[idx])

    def _process_sample(self, sample):
        # 1. Load Image
        try:
            image_path = sample['image_path']
            raw_img = Image.open(image_path).convert("RGB")
        except Exception:
            raw_img = Image.new("RGB", (HP.IMAGE_SIZE, HP.IMAGE_SIZE), (0, 0, 0))
        
        # --- Resizing Logic (Updated) ---
        orig_w, orig_h = raw_img.size
        curr_max_dim = max(orig_w, orig_h)
        
        target_max_side = 1600

        scale = 1.0
        if curr_max_dim > target_max_side:
            scale = target_max_side / curr_max_dim
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            raw_img = raw_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        w, h = raw_img.size
        
        # 2. Get Target BBox & Scale Coordinates
        bbox = sample.get('bbox', None)
        if bbox:
            abs_x1, abs_y1, abs_x2, abs_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # Apply scaling factor if image was resized
            if scale != 1.0:
                abs_x1 *= scale
                abs_y1 *= scale
                abs_x2 *= scale
                abs_y2 *= scale

            tx = int((abs_x1 + abs_x2) / 2)
            ty = int((abs_y1 + abs_y2) / 2)
            tx = max(0, min(w - 1, tx))
            ty = max(0, min(h - 1, ty))
        else:
            tx, ty = w // 2, h // 2

        # 3. Path Planning with Randomized Start (Updated)
        # 50% Random, 40% Opposite Corner (Adversarial), 10% Center
        rand_val = random.random()
        
        if rand_val < 0.1:
            # Random Position
            start_x = random.randint(0, w - 1)
            start_y = random.randint(0, h - 1)
        elif rand_val < 0.2:
            # Opposite corner logic to force FAR tokens
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
            
        # Clamp to bounds
        start_x = max(0, min(w-1, start_x))
        start_y = max(0, min(h-1, start_y))

        full_path = get_shortest_path_actions_dynamic((start_x, start_y), (tx, ty), (w, h))
        
        # 4. Step Sampling
        if not full_path:
            curr_pos = (start_x, start_y)
            action_token = "<CLICK_SHORT>"
            history_tokens = []
        else:
            if random.random() < 0.33:
                step_idx = len(full_path) - 1
            else:
                step_idx = random.randint(0, max(0, len(full_path) - 2))
            
            target_step = full_path[step_idx]
            action_token = target_step[0]
            
            history_tokens = []
            curr_cx, curr_cy = start_x, start_y
            
            for i in range(step_idx):
                h_token, (nx, ny) = full_path[i]
                history_tokens.append(h_token)
                curr_cx, curr_cy = nx, ny
            
            curr_pos = (curr_cx, curr_cy)

        # 5. Format Input
        if not history_tokens:
            history_str = "None (Start)"
        else:
            history_str = " -> ".join(history_tokens)
        
        user_content = f"[Action] Task: {sample['instruction']}\nPrevious Actions: {history_str}"

        # 6. CoT & Visuals
        image = draw_cursor(raw_img, curr_pos[0], curr_pos[1])
        
        cot_response = generate_cot_for_step(
            cursor_pos=curr_pos, 
            target_pos=(tx, ty), 
            instruction=sample['instruction'], 
            next_action_token=action_token,
            img_size=(w, h)
        )
        
        msgs = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": cot_response}
        ]
        
        return {"messages": msgs, "image": image}

# =============================================================================
# 4. Run Logic
# =============================================================================

def run_sft_screenspot_pro():
    input_path = HP.SFT_2_OUTPUT_PATH 
    output_path = getattr(HP, "SFT_3_OUTPUT_PATH", "./checkpoints/sft_phase3")
    
    if not os.path.exists(input_path):
        print(f"[Error] Stage 2 model not found at {input_path}")
        return

    print(f"[SFT-3] Loading Stage 2 Model from {input_path}...")
    # NOTE: device_map="auto" REMOVED for DDP compatibility
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        input_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(input_path, trust_remote_code=True)
    
    # Enable Gradient Checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Trainable Params: Embeddings + LLM Layers
    model.get_input_embeddings().weight.requires_grad = True
    for name, param in model.named_parameters():
        if "visual" in name: param.requires_grad = False
        else: param.requires_grad = True
        
    train_ds = ScreenSpotProSFTDataset("train")
    eval_ds = ScreenSpotProSFTDataset("eval")
    collator = SFTDataCollator(processor)
    
    epochs = getattr(HP, "SFT_3_EPOCHS", 3)
    batch_size = getattr(HP, "SFT_3_BATCH_SIZE", 2) # Low batch size for high-res images
    grad_accum = getattr(HP, "SFT_3_GRAD_ACCUM_STEPS", 8)
    lr = getattr(HP, "SFT_3_LEARN_RATE", 1e-5)

    args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=150, # Frequent saving for complex task
        eval_steps=150,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=0.1
    )
    
    trainer = WeightedActionTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, 
        data_collator=collator, processing_class=processor
    )
    
    print("[SFT-3] Starting ScreenSpot Pro Training (Weighted Action + Logging)...")
    trainer.train()
    
    trainer.save_model(output_path)
    processor.save_pretrained(output_path)
    train_ds.ss_manager.save_test_set()

if __name__ == "__main__":
    run_sft_screenspot_pro()