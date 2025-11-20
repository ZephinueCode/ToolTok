# src/train_baseline/sft.py

import json
import torch
import os
import random
import math
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    Qwen3VLForConditionalGeneration, 
    AutoProcessor, 
    Trainer, 
    TrainingArguments
)
from PIL import Image, ImageDraw
from dataclasses import dataclass
from typing import List, Dict, Any

from ..utils.parameters import HYPERPARAMS as HP
from ..utils.prompts import AGENT_SYSTEM_PROMPT as BASELINE_NL_SYSTEM_PROMPT

# =============================================================================
# 1. Spatial Logic (Natural Language Output)
# =============================================================================

def get_spatial_label_nl_with_cot(cursor_pos, target_center, intent):
    """
    Generates reasoning and Natural Language action.
    """
    cx, cy = cursor_pos
    tx, ty = target_center
    
    dx = tx - cx
    dy = ty - cy
    dist = math.hypot(dx, dy)
    
    reasoning = f"Reasoning: Cursor at ({cx}, {cy}). Target '{intent.get('name', 'target')}' at ({tx}, {ty}). "
    reasoning += f"Vector: dx={dx}, dy={dy}, dist={int(dist)}. "
    
    HIT_THRESHOLD = 30
    action = ""
    
    if dist < HIT_THRESHOLD:
        reasoning += "Cursor is ON target. "
        if intent["type"] == "click":
            action = "click"
        elif intent["type"] == "long_click":
            action = "long click"
        elif intent["type"] == "text":
            content = intent.get("content", "text")
            action = f"type '{content}'"
        else:
            action = "end action"
    else:
        reasoning += "Need to move. "
        if abs(dx) > abs(dy):
            direction = "right" if dx > 0 else "left"
            d_val = abs(dx)
            suffix = "far" if d_val > 200 else ("mid" if d_val > 40 else "close")
            action = f"move {direction} {suffix}"
        else:
            direction = "down" if dy > 0 else "up"
            d_val = abs(dy)
            suffix = "far" if d_val > 200 else ("mid" if d_val > 40 else "close")
            action = f"move {direction} {suffix}"
            
    return f"{reasoning}\nAction: {action}"

# =============================================================================
# 2. Dataset: Hybrid (Adapting JSONL tokens to Text)
# =============================================================================
class BaselineSFTDataset(Dataset):
    def __init__(self, data_path):
        # 1. Load JSONL Data (Semantic)
        if not os.path.exists(data_path):
            print(f"[Error] {data_path} not found.")
            self.jsonl_data = []
        else:
            print(f"[Dataset] Loading Semantic data from {data_path}...")
            with open(data_path, 'r', encoding='utf-8') as f:
                self.jsonl_data = [json.loads(line) for line in f]
        
        # 2. Load Real UI Images
        self.bg_images = self._load_real_images(limit=200)
        
        self.num_spatial = len(self.jsonl_data)
        self.total_len = len(self.jsonl_data) + self.num_spatial
        
        print(f"[Dataset] Baseline Total Samples: {self.total_len}")
        
        self.ui_names = ["Submit", "Cancel", "Search", "Menu", "Settings", "Back"]
        self.text_contents = ["hello", "test", "1234"]

    def _load_real_images(self, limit=200):
        print("[Dataset] Fetching real UI images...")
        images = []
        try:
            ds = load_dataset("rootsautomation/ScreenSpot", split="test", streaming=True)
            iterator = iter(ds)
            for _ in range(limit):
                try:
                    sample = next(iterator)
                    img = sample['image'].convert("RGB").resize((HP.IMAGE_SIZE, HP.IMAGE_SIZE))
                    images.append(img)
                except StopIteration:
                    break
        except Exception as e:
            print(f"[Warning] Failed to load ScreenSpot: {e}")
        return images

    def get_random_background(self):
        if self.bg_images:
            return random.choice(self.bg_images).copy()
        return Image.new("RGB", (HP.IMAGE_SIZE, HP.IMAGE_SIZE), (0, 0, 0))

    def token_to_nl(self, text):
        """
        Converts special tokens <MOVE_RIGHT_FAR> to 'move right far'.
        Used to adapt the existing JSONL data on the fly.
        """
        text = text.replace("<", "").replace(">", "")
        text = text.replace("_", " ").lower()
        # Fix formatting for consistency
        text = text.replace("text start", "type '").replace(" text end", "'")
        return text

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # === TYPE 1: SEMANTIC (JSONL) - ADAPTED TO NL ===
        if idx < len(self.jsonl_data):
            item = self.jsonl_data[idx]
            data_type = item.get("data_type", "action")
            
            if data_type == "explanation":
                image = Image.new('RGB', (HP.IMAGE_SIZE, HP.IMAGE_SIZE), color=(0, 0, 0))
            else:
                random_pixels = np.random.randint(0, 255, (HP.IMAGE_SIZE, HP.IMAGE_SIZE, 3), dtype=np.uint8)
                image = Image.fromarray(random_pixels)
                
            # Convert the Assistant response from Token format to NL format
            orig_response = item["messages"][1]["content"] # Assumes User/Assistant pair
            nl_response = self.token_to_nl(orig_response)

            msgs = [
                {"role": "system", "content": BASELINE_NL_SYSTEM_PROMPT},
                {"role": "user", "content": item["messages"][0]["content"]},
                {"role": "assistant", "content": nl_response}
            ]
            
            return {"messages": msgs, "image": image}

        # === TYPE 2: SPATIAL (Synthetic) - NATIVE NL ===
        else:
            image = self.get_random_background()
            draw = ImageDraw.Draw(image)
            margin = 50
            
            tx = random.randint(margin, HP.IMAGE_SIZE - margin)
            ty = random.randint(margin, HP.IMAGE_SIZE - margin)
            target_name = random.choice(self.ui_names)
            cx = random.randint(margin, HP.IMAGE_SIZE - margin)
            cy = random.randint(margin, HP.IMAGE_SIZE - margin)
            
            if random.random() < 0.3: cx, cy = tx, ty

            # Draw (simplified for brevity, logic same as main SFT)
            draw.rectangle([tx-30, ty-20, tx+30, ty+20], outline="green", width=3)
            try: draw.text((tx-20, ty-35), target_name, fill="green")
            except: pass
            draw.line([(cx-15, cy), (cx+15, cy)], fill="red", width=4)
            draw.line([(cx, cy-15), (cx, cy+15)], fill="red", width=4)

            intent_type = random.choice(["click", "move", "text"])
            intent = {"type": intent_type, "name": target_name}
            if intent_type == "text": intent["content"] = random.choice(self.text_contents)
            
            instr = f"Interact with {target_name}"
            
            # Use NL Generator
            cot_response = get_spatial_label_nl_with_cot((cx, cy), (tx, ty), intent)
            
            msgs = [
                {"role": "system", "content": BASELINE_NL_SYSTEM_PROMPT},
                {"role": "user", "content": f"[Action] {instr}"},
                {"role": "assistant", "content": cot_response}
            ]
            
            return {"messages": msgs, "image": image}

# =============================================================================
# 3. Collator (Identical to Main SFT, with Fix)
# =============================================================================
@dataclass
class SFTDataCollator:
    processor: AutoProcessor
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        images = []
        
        for feature in features:
            original_messages = feature["messages"]
            image = feature["image"]
            
            formatted_messages = []
            for m in original_messages:
                if m["role"] == "user":
                    content = [{"type": "image"}, {"type": "text", "text": m["content"]}]
                    formatted_messages.append({"role": "user", "content": content})
                else:
                    formatted_messages.append(m)
            
            text = self.processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            images.append(image)
            
        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=2048, 
            return_tensors="pt"
        )
        
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        
        for i in range(len(input_ids)):
            start_indices = (input_ids[i] == im_start_id).nonzero(as_tuple=True)[0]
            if len(start_indices) > 0:
                last_turn_idx = start_indices[-1]
                labels[i, :last_turn_idx + 2] = -100
            
        batch["labels"] = labels
        return batch

# =============================================================================
# 4. Setup (NO Surgical Tuning, just LLM Unfreeze)
# =============================================================================

def setup_model_for_baseline(model):
    print("[Baseline Setup] Freezing Vision, Unfreezing LLM. (No special tokens)")
    
    trainable = 0
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False
        else:
            # Train full LLM logic, but no special embedding hooks
            param.requires_grad = True
            trainable += param.numel()
            
    print(f"Trainable Params: {trainable:,}")

def run_baseline_sft():
    print(f"[Baseline SFT] Loading from {HP.BASE_MODEL_PATH}")
    # Note: Loading base model, NOT resizing embeddings for tokens
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        HP.BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(HP.BASE_MODEL_PATH, trust_remote_code=True)
    
    setup_model_for_baseline(model)
    
    train_dataset = BaselineSFTDataset(HP.SFT_DATA_PATH)
    collator = SFTDataCollator(processor=processor)
    
    # Use a separate output dir for baseline
    output_dir = HP.SFT_OUTPUT_PATH + "_baseline"
    
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=HP.SFT_EPOCHS,
        per_device_train_batch_size=HP.SFT_BATCH_SIZE,
        gradient_accumulation_steps=HP.SFT_GRAD_ACCUM_STEPS,
        learning_rate=HP.SFT_LEARN_RATE,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False
    )
    
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, data_collator=collator)
    
    print("[Baseline SFT] Starting Training (Natural Language Goal)...")
    trainer.train()
    
    print(f"[Baseline SFT] Saving Model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    run_baseline_sft()