# src/train/grpo_1.py

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from trl import GRPOConfig

from ..utils.parameters import HYPERPARAMS as HP
from ..utils.dataset import prepare_grpo1_dataset, train_eval_split
from ..tools.runner import Runner
from .trainer import ToolGRPOTrainer

def run_grpo_1():
    print(f"[GRPO-1] Initializing Pipeline...")
    
    # 1. Load Post-SFT Model
    model_path = HP.SFT_OUTPUT_PATH # Start from SFT 1 checkpoint
    print(f"[GRPO-1] Loading model from {model_path}")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 2. Prepare Dataset
    full_dataset = prepare_grpo1_dataset(HP.GRPO1_DATA_PATH, size=None)
    train_data, eval_data = train_eval_split(full_dataset, eval_ratio=0.1)
    
    print(f"[GRPO-1] Data: {len(train_data)} Train, {len(eval_data)} Eval")
    
    # 3. Initialize Runner (Environment)
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
        max_completion_length=256,
        logging_steps=HP.GRPO1_LOGGING_STEPS,
        save_steps=HP.GRPO1_SAVE_STEPS,
        max_grad_norm=1.0,
        temperature=HP.GRPO1_TEMPERATURE,
        report_to="none",
        remove_unused_columns=False
    )
    
    # 5. Trainer
    trainer = ToolGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=processor,
        runner=runner,
        max_tool_steps=10, # Give it enough steps to move cursor
        beta=0.1
    )
    
    # 6. Train
    print("[GRPO-1] Starting Reinforcement Learning...")
    trainer.train()
    
    # 7. Save
    print(f"[GRPO-1] Saving model to {HP.GRPO1_OUTPUT_PATH}")
    trainer.save_model(HP.GRPO1_OUTPUT_PATH)
    processor.save_pretrained(HP.GRPO1_OUTPUT_PATH)