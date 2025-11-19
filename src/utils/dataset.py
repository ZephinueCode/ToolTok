import torch
from torch.utils.data import Dataset, random_split
from datasets import load_dataset
from PIL import Image
from .parameters import HYPERPARAMS as HP

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
    
    Crucial Changes:
    1. NO Image Resizing. Keeps original resolution.
    2. Constructs a multi-step compatible GT structure (List of steps).
    3. Explicitly sets 'action_type' to 'click'.
    """
    print(f"[Dataset] Loading GRPO1 data from {dataset_path}...")
    try:
        # Try loading standard HF dataset
        ds = load_dataset(dataset_path)
    except Exception as e:
        print(f"[Warning] Failed to load from path, trying default/test split: {e}")
        ds = load_dataset("rootsautomation/ScreenSpot", split="test")

    # ScreenSpot usually puts data in 'test' split (it's a benchmark)
    split = "train" if "train" in ds else "test"
    dataset = ds[split]
    
    if size:
        print(f"[Dataset] Subsetting to {size} samples.")
        dataset = dataset.select(range(min(size, len(dataset))))

    def format_sample(sample):
        # 1. Keep Original Image (No Resize)
        # Converting to RGB to ensure consistency (remove Alpha channel if png)
        orig_img = sample['image'].convert("RGB")
        
        # 2. Handle BBox / Point
        # ScreenSpot provides bbox [x1, y1, x2, y2] in absolute coordinates
        bbox = sample.get('bbox', None)
        
        if not bbox:
            # Fallback if point is provided (create a 20x20 box around it)
            point = sample.get('point', [0,0])
            px, py = point[0], point[1]
            bbox = [px-10, py-10, px+10, py+10]

        instruction = sample['instruction']
        
        # 3. Construct GT Trajectory Structure
        # This prepares us for future multi-step datasets (AndroidControl etc.)
        # For ScreenSpot, it's a list with 1 element.
        gt_steps = [
            {
                "step_idx": 0,
                "image": orig_img,       # The raw image state
                "bbox": bbox,            # The target absolute bbox
                "instruction": instruction,
                "action_type": "click"   # Hard constraint for ScreenSpot
            }
        ]

        # The 'question' is the high-level intent presented to the model
        question = f"[Action] Perform a step for the following action: {instruction}"
        
        return {
            "question": question,
            "image": orig_img,           # Initial image for the Trainer
            "ground_truth_traj": gt_steps # Complete path definition
        }

    print("[Dataset] Formatting samples (Structure setup only, no resizing)...")
    # We use num_proc=1 or 4 depending on memory. 
    # Since we store full-res images in RAM, be careful with large datasets.
    dataset = dataset.map(format_sample, num_proc=4)
    
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