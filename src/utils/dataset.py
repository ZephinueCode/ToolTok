# src/utils/datasets.py

import torch
from torch.utils.data import Dataset, random_split
from datasets import load_dataset
from PIL import Image
from .parameters import HYPERPARAMS as HP
import numpy as np # Keep numpy if needed for other logic, though not for image conversion here

class GRPO1Dataset(Dataset):
    """
    Wrapper for ScreenSpot dataset compatible with ToolGRPOTrainer.
    Passes raw images without resizing to preserve UI details.
    """
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def prepare_grpo1_dataset(
    dataset_path: str = "rootsautomation/ScreenSpot",
    seed: int = HP.GRPO1_SEED,
    size: int = None
):
    """
    Load and format ScreenSpot dataset for GRPO1.
    """
    print(f"[Dataset] Loading GRPO1 data from {dataset_path}...")
    try:
        ds = load_dataset(dataset_path)
    except Exception as e:
        print(f"[Warning] Failed to load from path, trying default/test split: {e}")
        ds = load_dataset("rootsautomation/ScreenSpot", split="test")

    split = "train" if "train" in ds else "test"
    dataset = ds[split]
    
    if size:
        print(f"[Dataset] Subsetting to {size} samples.")
        dataset = dataset.select(range(min(size, len(dataset))))

    def format_sample(sample):
        # 1. Keep Original Image (Clean PIL Copy)
        # [CRITICAL FIX]: Use .copy() or create new Image to detach from underlying buffer
        # This solves the serialization crash without converting to numpy.
        raw_img = sample['image'].convert("RGB")
        orig_img = raw_img.copy()
        
        # 2. Handle BBox / Point
        bbox = sample.get('bbox', None)
        if not bbox:
            point = sample.get('point', [0,0])
            px, py = point[0], point[1]
            bbox = [px-10, py-10, px+10, py+10]

        instruction = sample['instruction']
        
        # 3. Construct GT Trajectory Structure
        # Now passing pure PIL Image
        gt_steps = [
            {
                "step_idx": 0,
                "image": orig_img, 
                "bbox": bbox, 
                "instruction": instruction,
                "action_type": "click" 
            }
        ]

        # The 'question' is the high-level intent presented to the model
        question = f"{instruction}"
        
        return {
            "question": question,
            "image": orig_img, 
            "ground_truth_traj": gt_steps
        }

    print("[Dataset] Formatting samples (Using PIL Copy)...")
    # Keep num_proc=1 to be safe with PIL serialization
    dataset = dataset.map(format_sample, num_proc=1)
    
    return GRPO1Dataset(dataset)

def train_eval_split(dataset, eval_ratio: float):
    total_size = len(dataset)
    eval_size = int(total_size * eval_ratio)
    train_size = total_size - eval_size
    
    print(f"[Dataset] Splitting: {train_size} Train, {eval_size} Eval")
    
    return random_split(
        dataset, 
        [train_size, eval_size], 
        generator=torch.Generator().manual_seed(42)
    )