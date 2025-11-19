# src/train/sft_1.py

import json
import torch
from torch.utils.data import Dataset
from transformers import (
    Qwen3VLForConditionalGeneration, 
    AutoProcessor, 
    Trainer, 
    TrainingArguments
)
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Any

from ..utils.parameters import HYPERPARAMS as HP
from ..utils.action import ACTION_TOKENS

# =============================================================================
# 1. Dataset: JSONL Loader with Blank Images
# =============================================================================
class SFT1Dataset(Dataset):
    """
    Loads GUI instructions and pairs them with blank (black) images.
    Goal: Teach semantic meaning of action tokens without visual dependency first.
    """
    def __init__(self, data_path):
        print(f"[Dataset] Loading SFT data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        print(f"[Dataset] Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # item structure: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
        
        # Create a blank black image (640x640)
        # Qwen3-VL handles images dynamically, but we provide a standard canvas here.
        image = Image.new('RGB', (HP.IMAGE_SIZE, HP.IMAGE_SIZE), color=(0, 0, 0))
        
        return {
            "messages": item["messages"],
            "image": image
        }

# =============================================================================
# 2. Data Collator: Formatting for Qwen3-VL
# =============================================================================
@dataclass
class SFTDataCollator:
    processor: AutoProcessor
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Prepare batch lists
        texts = []
        images = []
        
        for feature in features:
            messages = feature["messages"]
            image = feature["image"]
            
            # 1. Format conversation for Qwen3-VL
            # The processor expects a list of messages. 
            # We need to ensure the image placeholder is in the user message.
            # Assuming generation script produced plain text, we inject image here.
            
            # Check if image is already in content (it shouldn't be based on generation script)
            user_content = messages[0]["content"]
            
            # Construct formatted messages with Image
            formatted_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_content}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": messages[1]["content"]} # The Action Token
                    ]
                }
            ]
            
            # Apply chat template to get the raw text prompt
            text = self.processor.apply_chat_template(
                formatted_messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            texts.append(text)
            images.append(image)
            
        # 2. Tokenize and Process
        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=HP.SFT_MAX_LENGTH,
            return_tensors="pt"
        )
        
        # 3. Create Labels (Masking User Prompts)
        # We strictly want to train on the ASSISTANT's output (The Action Token)
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        
        # Mask pad tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        # Simplified masking strategy: Mask everything until the last header token or 
        # rely on the fact that action tokens are unique.
        # Better approach: Tokenize just the user prompt part to get its length.
        
        for i, msgs in enumerate(features):
            # Reconstruct user-only prompt to measure length
            user_only_msgs = [
                {
                    "role": "user", 
                    "content": [{"type": "image", "image": images[i]}, {"type": "text", "text": msgs["messages"][0]["content"]}]
                },
                # We add a dummy assistant start to get the boundary
                 {"role": "assistant", "content": ""} 
            ]
            user_text = self.processor.apply_chat_template(user_only_msgs, tokenize=False, add_generation_prompt=True)
            
            # Tokenize user part (without adding special tokens again if template added them)
            # We rely on the length of ids to mask labels
            user_tokens = self.processor(text=[user_text], images=[images[i]], return_tensors="pt")["input_ids"][0]
            mask_len = len(user_tokens)
            
            # Apply mask (safeguard index)
            safe_len = min(mask_len, labels.shape[1])
            labels[i, :safe_len] = -100
            
        batch["labels"] = labels
        return batch

# =============================================================================
# 3. Model Setup: Surgical Partial Unfreeze
# =============================================================================
def setup_model_for_sft_1(model, processor):
    """
    Freezes embedding layer for OLD tokens.
    Unfreezes embedding layer for NEW (GUI) tokens.
    Keeps all other layers (Layers, Heads) trainable.
    """
    print("[Model Setup] Configuring Surgical Partial Unfreeze for Embeddings...")
    
    # 1. Get Action Token IDs
    action_token_ids = [processor.tokenizer.convert_tokens_to_ids(t) for t in ACTION_TOKENS]
    action_token_ids_tensor = torch.tensor(action_token_ids, device=model.device)
    
    # 2. Enable Gradient for Input Embeddings (Prerequisite for hook)
    model.enable_input_require_grads()
    input_embeddings = model.get_input_embeddings()
    input_embeddings.weight.requires_grad = True
    
    # 3. Register Hook to Zero-out Gradients for Non-Action Tokens
    def zero_out_non_action_grads(grad):
        mask = torch.zeros_like(grad)
        # Only allow updates for action token indices
        valid_indices = action_token_ids_tensor.to(grad.device)
        mask[valid_indices] = 1.0
        return grad * mask

    input_embeddings.weight.register_hook(zero_out_non_action_grads)
    
    # 4. Configure other layers
    # We want to train the transformer blocks to adapt to the new semantic space
    trainable_params = 0
    all_params = 0
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        
        if "embed_tokens" in name:
            # Handled by hook, counted as trainable technically, but effectively restricted
            pass 
        elif any(k in name for k in ["layers", "blocks", "transformer", "visual", "lm_head", "merger"]):
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            # Freeze absolute position embeddings etc if necessary, usually fine to train
            param.requires_grad = False

    print(f"[Model Setup] Trainable Parameters (Approx): {trainable_params:,} / {all_params:,}")
    print(f"[Model Setup] Embedding Layer: Gradients active ONLY for {len(action_token_ids)} GUI tokens.")

# =============================================================================
# 4. Main Run Function
# =============================================================================
def run_sft_1():
    # 1. Load Initialized Model (From Phase 0)
    print(f"[SFT-1] Loading initialized model from {HP.INIT_MODEL_PATH}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        HP.INIT_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        HP.INIT_MODEL_PATH,
        trust_remote_code=True
    )
    
    # 2. Setup Model (Freezing Logic)
    setup_model_for_sft_1(model, processor)
    
    # 3. Load Data
    train_dataset = SFT1Dataset(HP.SFT_1_DATA_PATH)
    collator = SFTDataCollator(processor=processor)
    
    # 4. Training Arguments
    args = TrainingArguments(
        output_dir=HP.SFT_1_OUTPUT_PATH,
        num_train_epochs=HP.SFT_EPOCHS,
        per_device_train_batch_size=HP.SFT_BATCH_SIZE,
        gradient_accumulation_steps=HP.SFT_GRAD_ACCUM_STEPS,
        learning_rate=HP.SFT_LEARN_RATE,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False, # CRITICAL for custom collators
        dataloader_pin_memory=False  # Sometimes helps with custom collators
    )
    
    # 5. Train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator
    )
    
    print("[SFT-1] Starting Semantic Injection Training...")
    trainer.train()
    
    # 6. Save
    print(f"[SFT-1] Saving Semantic Tuned Model to {HP.SFT_1_OUTPUT_PATH}")
    trainer.save_model(HP.SFT_1_OUTPUT_PATH)
    processor.save_pretrained(HP.SFT_1_OUTPUT_PATH)