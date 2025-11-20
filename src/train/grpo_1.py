# src/train/grpo_1.py

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from trl import GRPOConfig

from ..utils.parameters import HYPERPARAMS as HP
from ..utils.dataset import prepare_grpo1_dataset, train_eval_split
from ..tools.runner import Runner
from .trainer import ToolGRPOTrainer
from ..utils.action import ACTION_TOKENS # Import Action Tokens for freezing logic

def run_grpo_1():
    print(f"[GRPO-1] Initializing Pipeline...")
    
    # 1. Load Post-SFT Model
    model_path = HP.SFT_OUTPUT_PATH 
    print(f"[GRPO-1] Loading model from {model_path}")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 2. Prepare Dataset
    full_dataset = prepare_grpo1_dataset(HP.GRPO1_DATA_PATH, size=HP.GRPO_DATASET_SIZE)
    train_data, eval_data = train_eval_split(full_dataset, eval_ratio=0.1)
    
    print(f"[GRPO-1] Data: {len(train_data)} Train, {len(eval_data)} Eval")
    
    # 3. Initialize Runner
    runner = Runner()
    
    # 4. Config
    training_args = GRPOConfig(
        output_dir=HP.GRPO1_OUTPUT_PATH,
        learning_rate=HP.GRPO1_LEARN_RATE,
        num_train_epochs=HP.GRPO1_NUM_EPOCHS,
        per_device_train_batch_size=HP.GRPO1_BATCH_SIZE,
        gradient_accumulation_steps=HP.GRPO1_GRAD_ACCUM_STEPS,
        num_generations=HP.GRPO1_NUM_GENERATIONS,
        max_prompt_length=1024,
        max_completion_length=512, 
        logging_steps=HP.GRPO1_LOGGING_STEPS,
        save_steps=HP.GRPO1_SAVE_STEPS,
        max_grad_norm=1.0,
        temperature=HP.GRPO1_TEMPERATURE,
        report_to="none",
        remove_unused_columns=False,
        beta=0.02, # KL Penalty
        
        # ============================================================
        # [NEW] Evaluation Configuration
        # ============================================================
        do_eval=True,
        eval_strategy="steps",
        eval_steps=HP.GRPO1_EVAL_STEPS, 
        per_device_eval_batch_size=4,  
        eval_on_start=False,
        save_total_limit=HP.GRPO1_MAX_CHECKPOINTS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_reward",
        greater_is_better=True,
        # ============================================================
    )
    
    # 5. Trainer
    trainer = ToolGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=processor,
        runner=runner,
        max_tool_steps=10,
        beta=0.1
    )
    
    # =========================================================================
    # CRITICAL: Apply Surgical Freeze Hook to GRPO Model
    # =========================================================================
    print("[GRPO-1] Re-applying Surgical Embedding Hooks for RL...")
    
    # 1. Identify Action IDs
    action_token_ids = [
        processor.tokenizer.convert_tokens_to_ids(t) 
        for t in ACTION_TOKENS
    ]
    action_token_ids = [idx for idx in action_token_ids if idx is not None]
    target_indices_tensor = torch.tensor(action_token_ids)
    
    # 2. Hook Function
    def gradient_mask_hook(grad):
        mask = torch.zeros_like(grad)
        indices = target_indices_tensor.to(grad.device)
        mask[indices] = 1.0
        return grad * mask

    # 3. Enable Grads & Register
    model.enable_input_require_grads()
    input_embeddings = model.get_input_embeddings()
    input_embeddings.weight.requires_grad = True
    input_embeddings.weight.register_hook(gradient_mask_hook)
    
    print("[GRPO-1] Hook Registered. Only Action Token Embeddings will update.")
    # =========================================================================
    
    # 6. Train
    print("[GRPO-1] Starting Reinforcement Learning...")
    trainer.train()
    
    # 7. Save
    print(f"[GRPO-1] Saving model to {HP.GRPO1_OUTPUT_PATH}")
    trainer.save_model(HP.GRPO1_OUTPUT_PATH)
    processor.save_pretrained(HP.GRPO1_OUTPUT_PATH)