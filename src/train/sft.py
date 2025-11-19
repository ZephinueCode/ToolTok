import json
import torch
import os
import random
import math
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    Qwen3VLForConditionalGeneration, 
    AutoProcessor, 
    Trainer, 
    TrainingArguments
)
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from typing import List, Dict, Any

from ..utils.parameters import HYPERPARAMS as HP
from ..utils.action import ACTION_TOKENS

# =============================================================================
# Helper Logic for Spatial Calculation (For Synthetic Data)
# =============================================================================

def get_spatial_label(cursor_pos, target_center, intent):
    """
    Determines the correct token based on distance and intent.
    """
    cx, cy = cursor_pos
    tx, ty = target_center
    
    dx = tx - cx
    dy = ty - cy
    dist = math.hypot(dx, dy)
    
    # Threshold for "arrived" (Hit logic)
    HIT_THRESHOLD = 25
    
    if dist < HIT_THRESHOLD:
        # === ARRIVED STATE ===
        if intent["type"] == "click":
            return "<CLICK_SHORT>"
        elif intent["type"] == "long_click":
            return "<CLICK_LONG>"
        elif intent["type"] == "text":
            return f"<TEXT_START> {intent['content']} <TEXT_END>"
        else: # type == "move" (Just "Move to X")
            return "<END_ACTION>" 
    else:
        # === APPROACH STATE ===
        if abs(dx) > abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
            d_val = abs(dx)
            suffix = "_FAR" if d_val > 150 else ("_MID" if d_val > 25 else "_CLO")
            return f"<MOVE_{direction}{suffix}>"
        else:
            direction = "DOWN" if dy > 0 else "UP"
            d_val = abs(dy)
            suffix = "_FAR" if d_val > 150 else ("_MID" if d_val > 25 else "_CLO")
            return f"<MOVE_{direction}{suffix}>"

# =============================================================================
# 1. Dataset: Tri-Modal Mixing
# =============================================================================
class SFTDataset(Dataset):
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            print(f"[Warning] {data_path} not found.")
            self.jsonl_data = []
            self.num_actions = 0
        else:
            print(f"[Dataset] Loading SFT data from {data_path}...")
            with open(data_path, 'r', encoding='utf-8') as f:
                self.jsonl_data = [json.loads(line) for line in f]
        
        # Count action samples to balance synthetic data
        self.num_actions = sum(1 for item in self.jsonl_data if item.get("data_type") == "action")
        
        # Resources for synthetic generation
        self.ui_names = ["submit", "cancel", "ok", "edit", "search", "home", "menu", "profile", "settings", "back"]
        self.text_contents = ["hello", "1234", "user", "search query", "password"]
        self.scroll_dirs = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        # Dataset Structure:
        # Part 1 (0 ~ len-1): Original JSONL Data (Explanation + Semantic Action)
        # Part 2 (len ~ end): Synthetic Spatial Data (Visual Grounding) - 1:1 ratio with Semantic Actions
        self.jsonl_len = len(self.jsonl_data)
        self.total_len = self.jsonl_len + self.num_actions
        
        print(f"[Dataset] JSONL Samples: {self.jsonl_len}")
        print(f"[Dataset] Synthetic Spatial Samples to Generate: {self.num_actions}")
        print(f"[Dataset] Total Training Samples: {self.total_len}")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # ===============================================================
        # PART 1: JSONL DATA (Semantic Understanding)
        # ===============================================================
        if idx < self.jsonl_len:
            item = self.jsonl_data[idx]
            data_type = item.get("data_type", "action") # Default to action if missing
            
            if data_type == "explanation":
                # [STRATEGY] Text-Only Mode -> Use Pure Black Image
                # This signals the model: "Ignore vision, focus on language definition"
                image = Image.new('RGB', (HP.IMAGE_SIZE, HP.IMAGE_SIZE), color=(0, 0, 0))
                
            else: # data_type == "action"
                # [STRATEGY] Semantic Action Mode -> Use Random Noise
                # This signals: "Visuals are irrelevant noise, follow the text instruction pattern"
                # Using noise prevents the model from overfitting to "Black = Explanation" logic 
                # when it sees "Perform action" text.
                random_pixels = np.random.randint(0, 255, (HP.IMAGE_SIZE, HP.IMAGE_SIZE, 3), dtype=np.uint8)
                image = Image.fromarray(random_pixels)
            
            return {
                "messages": item["messages"],
                "image": image
            }
            
        # ===============================================================
        # PART 2: SYNTHETIC DATA (Visual Grounding Logic)
        # ===============================================================
        else:
            # Create a geometric scene
            image = Image.new("RGB", (HP.IMAGE_SIZE, HP.IMAGE_SIZE), (0, 0, 0))
            draw = ImageDraw.Draw(image)
            margin = 60
            
            # Random Cursor Position
            cx = random.randint(margin, HP.IMAGE_SIZE - margin)
            cy = random.randint(margin, HP.IMAGE_SIZE - margin)
            
            # Draw Cursor (Red Cross)
            draw.line([(cx-12, cy), (cx+12, cy)], fill="red", width=3)
            draw.line([(cx, cy-12), (cx, cy+12)], fill="red", width=3)

            # Decide Task: Scroll vs Interact
            intent_cat = random.choice(["interact", "interact", "scroll"])
            
            # --- Type A: SCROLL (No Target) ---
            if intent_cat == "scroll":
                direction = random.choice(self.scroll_dirs)
                label_token = f"<SCROLL_{direction}>"
                
                verbs = ["Scroll", "Swipe", "Pan"]
                v = random.choice(verbs)
                d_map = {"UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right"}
                instr = f"{v} {d_map[direction]}"
                
                user_content = f"[Action] Perform a step for the following action: {instr}"

            # --- Type B: INTERACT (With Target) ---
            else:
                intent_type = random.choice(["click", "click", "long_click", "text", "move"])
                target_name = random.choice(self.ui_names)
                text_payload = random.choice(self.text_contents) if intent_type == "text" else ""
                
                intent = {"type": intent_type, "content": text_payload}
                
                # Target Pos
                tx = random.randint(margin, HP.IMAGE_SIZE - margin)
                ty = random.randint(margin, HP.IMAGE_SIZE - margin)
                
                # Draw Target (Green Box)
                box = [tx-30, ty-20, tx+30, ty+20]
                draw.rectangle(box, outline="green", width=3)
                try:
                    font = ImageFont.truetype("arial.ttf", 14)
                except:
                    font = ImageFont.load_default()
                draw.text((tx-20, ty-35), target_name, fill="green", font=font)
                
                # Logic: Start ON target or AWAY?
                is_overlap = random.random() < 0.5
                if is_overlap:
                    # Move cursor to target
                    offset_x = random.randint(-10, 10)
                    offset_y = random.randint(-10, 10)
                    cx, cy = tx + offset_x, ty + offset_y
                    
                    # Redraw scene to reflect new cursor pos
                    image.paste((0,0,0), (0,0, HP.IMAGE_SIZE, HP.IMAGE_SIZE))
                    draw.rectangle(box, outline="green", width=3)
                    draw.text((tx-20, ty-35), target_name, fill="green", font=font)
                    draw.line([(cx-12, cy), (cx+12, cy)], fill="red", width=3)
                    draw.line([(cx, cy-12), (cx, cy+12)], fill="red", width=3)
                else:
                    # Ensure distance
                    while math.hypot(tx-cx, ty-cy) < 100:
                        cx = random.randint(margin, HP.IMAGE_SIZE - margin)
                        cy = random.randint(margin, HP.IMAGE_SIZE - margin)

                # Generate Prompt
                if intent_type == "click":
                    instr = f"Click '{target_name}'"
                elif intent_type == "long_click":
                    instr = f"Long press {target_name}"
                elif intent_type == "text":
                    instr = f"Type '{text_payload}' into {target_name}"
                else:
                    instr = f"Move to {target_name}"
                
                user_content = f"[Action] Perform a step for the following action: {instr}"
                label_token = get_spatial_label((cx, cy), (tx, ty), intent)

            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": label_token}
            ]
            
            return {
                "messages": messages,
                "image": image
            }

# =============================================================================
# 2. Collator
# =============================================================================
@dataclass
class SFTDataCollator:
    processor: AutoProcessor
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        images = []
        
        for feature in features:
            messages = feature["messages"]
            image = feature["image"]
            
            formatted_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": messages[0]["content"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": messages[1]["content"]}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            images.append(image)
            
        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt"
        )
        
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        
        for i in range(len(input_ids)):
            start_indices = (input_ids[i] == im_start_id).nonzero(as_tuple=True)[0]
            if len(start_indices) > 0:
                assistant_start_idx = start_indices[-1]
                mask_end = assistant_start_idx + 2
                labels[i, :mask_end] = -100
            else:
                labels[i, :] = -100
        
        batch["labels"] = labels
        return batch

# =============================================================================
# 3. Model Setup
# =============================================================================
def setup_model_for_sft(model, processor):
    print("[Model Setup] Configuring Surgical Partial Unfreeze for Embeddings...")
    action_token_ids = [processor.tokenizer.convert_tokens_to_ids(t) for t in ACTION_TOKENS]
    action_token_ids_tensor = torch.tensor(action_token_ids, device=model.device)
    
    model.enable_input_require_grads()
    input_embeddings = model.get_input_embeddings()
    input_embeddings.weight.requires_grad = True
    
    def zero_out_non_action_grads(grad):
        mask = torch.zeros_like(grad)
        valid_indices = action_token_ids_tensor.to(grad.device)
        mask[valid_indices] = 1.0
        return grad * mask

    input_embeddings.weight.register_hook(zero_out_non_action_grads)
    
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if "embed_tokens" in name or "wte" in name:
            pass 
        elif any(k in name for k in ["layers", "blocks", "transformer", "visual", "lm_head", "merger"]):
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    print(f"[Model Setup] Trainable Params: {trainable_params:,} / {all_params:,}")

# =============================================================================
# 4. Main Run
# =============================================================================
def run_sft():
    print(f"[SFT] Loading initialized model from {HP.INIT_MODEL_PATH}")
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            HP.INIT_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(HP.INIT_MODEL_PATH, trust_remote_code=True)
    except OSError:
        print(f"[Error] Model not found at {HP.INIT_MODEL_PATH}")
        return

    setup_model_for_sft(model, processor)
    
    train_dataset = SFTDataset(HP.SFT_DATA_PATH)
    collator = SFTDataCollator(processor=processor)
    
    args = TrainingArguments(
        output_dir=HP.SFT_OUTPUT_PATH,
        num_train_epochs=HP.SFT_EPOCHS,
        per_device_train_batch_size=HP.SFT_BATCH_SIZE,
        gradient_accumulation_steps=HP.SFT_GRAD_ACCUM_STEPS,
        learning_rate=HP.SFT_LEARN_RATE,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False, 
        dataloader_pin_memory=False,
        warmup_ratio=0.1
    )
    
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, data_collator=collator)
    
    print("[SFT] Starting Semantic & Spatial Injection Training...")
    trainer.train()
    
    print(f"[SFT] Saving Model to {HP.SFT_OUTPUT_PATH}")
    trainer.save_model(HP.SFT_OUTPUT_PATH)
    processor.save_pretrained(HP.SFT_OUTPUT_PATH)