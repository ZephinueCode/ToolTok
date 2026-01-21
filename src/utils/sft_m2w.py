import math
import os
import json
import torch
import random
from PIL import Image
from datasets import load_dataset, Dataset as HFDataset
from typing import List, Tuple, Dict, Any, Union
from tqdm import tqdm

# Import project parameters and logic
from .parameters import HYPERPARAMS as HP
from .action_logic import MOVE_DELTAS

# =============================================================================
# 1. Data Manager (Image Processing & Caching)
# =============================================================================

class Mind2WebDataManager:
    """
    Manages loading, parsing, and cropping of the Mind2Web dataset.
    Features:
    - Random Jitter Cropping (to fix centering bias).
    - Robust Target ID Extraction (to handle messy HTML).
    """
    def __init__(self):
        self.dataset_name = "./data/mind2web"  # Local or HF path
        
        # Path configuration
        self.data_path = getattr(HP, "M2W_CACHE_PATH", "./data/mind2web_processed")
        self.max_image_dim = 1920
        self.context_padding = 1000  # Base padding around target
        
        self.processed_dir = os.path.join(self.data_path, f"images_cropped_{self.max_image_dim}")
        self.metadata_cache_path = os.path.join(self.data_path, "mind2web_full_metadata.json")
        
        os.makedirs(self.processed_dir, exist_ok=True)

        # 1. Limit number of samples
        self.total_limit = getattr(HP, "M2W_TOTAL_SIZE", 4000)
        if self.total_limit is None: 
            self.total_limit = 999999

        self.samples = []
        
        # 2. Try loading from Cache
        if os.path.exists(self.metadata_cache_path):
            print(f"[Mind2Web] Checking metadata cache: {self.metadata_cache_path}")
            try:
                with open(self.metadata_cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                print(f"[Mind2Web] Cache contains {len(cached_data)} samples.")
                
                if len(cached_data) > 0:
                    self.samples = cached_data
                    if len(self.samples) > self.total_limit:
                        self.samples = self.samples[:self.total_limit]
            except Exception as e:
                print(f"[Error] Failed to read cache: {e}. Will re-process.")
                self.samples = []

        # 3. Process from Raw Dataset if cache is empty
        if not self.samples:
            print(f"[Mind2Web] Processing from raw HF dataset (Target: {self.total_limit})...")
            try:
                self.hf_dataset = load_dataset(self.dataset_name, split='train')
                self.samples = self._process_dataset(self.hf_dataset, limit=self.total_limit)
                
                if self.samples:
                    print(f"[Mind2Web] Saving {len(self.samples)} processed samples to cache...")
                    with open(self.metadata_cache_path, 'w', encoding='utf-8') as f:
                        json.dump(self.samples, f)
                        
            except Exception as e:
                print(f"[Error] Raw processing failed: {e}")
                self.raw_train, self.raw_eval, self.raw_test = [], [], []
                return

        if not self.samples:
            print("[Error] No valid samples available.")
            self.raw_train, self.raw_eval, self.raw_test = [], [], []
            return

        # 4. Convert to HF Dataset for splitting
        ds = HFDataset.from_list(self.samples)
        ds = ds.shuffle(seed=HP.SFT_SEED)
        
        final_count = min(len(ds), self.total_limit)
        ds = ds.select(range(final_count))
        
        print(f"[Mind2Web] Final Pool Size: {final_count}")

        # 5. Split Logic
        train_ratio = getattr(HP, "M2W_TRAIN_RATIO", 0.8)
        eval_ratio = getattr(HP, "M2W_EVAL_RATIO", 0.1)
        
        train_end = int(final_count * train_ratio)
        eval_end = int(final_count * (train_ratio + eval_ratio))

        self.raw_train = ds.select(range(0, train_end))
        self.raw_eval = ds.select(range(train_end, eval_end))
        self.raw_test = ds.select(range(eval_end, final_count))
        
        print(f"[Mind2Web] Splits: Train={len(self.raw_train)}, Eval={len(self.raw_eval)}, Test={len(self.raw_test)}")

    def _extract_raw_rect_and_id(self, candidate_str: str) -> Tuple[List[float], str]:
        """
        Parses JSON to get [x, y, w, h] and the best available element ID.
        Uses a fallback chain to deal with missing IDs.
        """
        try:
            cand_dict = json.loads(candidate_str)
            attributes_str = cand_dict.get("attributes", "{}")
            
            if isinstance(attributes_str, str):
                attributes = json.loads(attributes_str)
            else:
                attributes = attributes_str
            
            # 1. Get BBox
            rect = attributes.get("bounding_box_rect", None)
            if isinstance(rect, str):
                rect = [float(x) for x in rect.split(',')]
            
            if not rect or len(rect) < 4:
                return None, None
            
            # 2. Get Descriptor (Fallback Chain)
            # Try: id -> name -> aria_label -> type -> value -> class
            el_id = (attributes.get("id") or 
                     attributes.get("name") or 
                     attributes.get("aria_label") or 
                     attributes.get("type") or 
                     attributes.get("value"))
            
            # 3. Last Resort
            if not el_id:
                fallback_term = random.choice(["button", "bar", "link", "target", "element", "clickable"])
                bid = attributes.get("backend_node_id", "")
                el_id = f"{fallback_term} {bid}" if bid else fallback_term
            
            # Clean up text
            el_id = str(el_id).replace("\n", " ").strip()[:50]
            
            return rect, el_id

        except Exception:
            return None, None

    def _crop_resize_save(self, image: Image.Image, raw_rect: List[float], image_id: str) -> Tuple[str, List[float], List[int]]:
        """
        Crops image with RANDOM JITTER. 
        This prevents the target from always being in the center, forcing the model to learn MOVE actions.
        """
        orig_w, orig_h = image.size
        bx, by, bw, bh = raw_rect
        
        # 1. Random Jitter Calculation
        # Shift the crop window randomly by +/- 500px
        jitter_x = random.randint(-500, 500)
        jitter_y = random.randint(-500, 500)
        
        crop_x1 = int(bx - self.context_padding + jitter_x)
        crop_y1 = int(by - self.context_padding + jitter_y)
        crop_x2 = int(bx + bw + self.context_padding + jitter_x)
        crop_y2 = int(by + bh + self.context_padding + jitter_y)
        
        # 2. Clamp to image boundaries
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(orig_w, crop_x2)
        crop_y2 = min(orig_h, crop_y2)
        
        # Safety check for min size
        if crop_x2 - crop_x1 < 100: crop_x2 = min(orig_w, crop_x1 + 100)
        if crop_y2 - crop_y1 < 100: crop_y2 = min(orig_h, crop_y1 + 100)

        cropped_img = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        crop_w, crop_h = cropped_img.size
        
        # 3. Relative bbox calculation
        rel_x = bx - crop_x1
        rel_y = by - crop_y1
        
        # 4. Resize logic
        if crop_w <= self.max_image_dim and crop_h <= self.max_image_dim:
            scale = 1.0
            final_img = cropped_img
            final_w, final_h = crop_w, crop_h
        else:
            scale = min(self.max_image_dim / crop_w, self.max_image_dim / crop_h)
            final_w = int(crop_w * scale)
            final_h = int(crop_h * scale)
            final_img = cropped_img.resize((final_w, final_h), Image.Resampling.LANCZOS)

        # 5. Save
        save_name = f"{image_id}_crop.png"
        save_path = os.path.join(self.processed_dir, save_name)
        
        if not os.path.exists(save_path):
            final_img.save(save_path)
            
        # 6. Final BBox scaling
        final_x1 = rel_x * scale
        final_y1 = rel_y * scale
        final_x2 = (rel_x + bw) * scale
        final_y2 = (rel_y + bh) * scale
        
        return save_path, [final_x1, final_y1, final_x2, final_y2], [final_w, final_h]

    def _process_dataset(self, hf_split, limit: int) -> List[Dict[str, Any]]:
        """
        Iterates over HF dataset and extracts standard fields + element ID.
        """
        samples = []
        print(f"[Mind2Web] Processing raw samples until {limit} valid items...")
        
        pbar = tqdm(total=limit, desc="Collecting Samples")
        
        for idx, row in enumerate(hf_split):
            if len(samples) >= limit:
                break

            try:
                action_uid = row.get("action_uid", f"unknown_{idx}")
                image = row.get("screenshot")
                pos_candidates = row.get("pos_candidates", [])
                operation_str = row.get("operation", "{}")
                instruction = row.get("confirmed_task", "")
                
                if not image or not pos_candidates: continue
                
                # Extract BBox and ID
                raw_rect, target_element_id = self._extract_raw_rect_and_id(pos_candidates[0])
                if not raw_rect: continue

                # Image Processing (Crop + Resize)
                final_path, final_bbox, final_size = self._crop_resize_save(image, raw_rect, action_uid)
                
                # Action Parsing
                if isinstance(operation_str, str):
                    op_data = json.loads(operation_str)
                else:
                    op_data = operation_str
                
                op_type = op_data.get("op", "").upper()
                op_value = op_data.get("value", "")

                target_action_token = "<CLICK_SHORT>"
                action_value = None

                if op_type == "CLICK":
                    target_action_token = "<CLICK_SHORT>"
                elif op_type == "SELECT":
                    target_action_token = "<CLICK_LONG>"
                elif op_type == "TYPE":
                    target_action_token = "<TEXT_START>"
                    action_value = op_value
                
                samples.append({
                    "image_path": final_path,
                    "bbox": final_bbox, 
                    "instruction": instruction,
                    "id": action_uid,             
                    "target_id": target_element_id, 
                    "img_size": final_size,
                    "action_type": target_action_token,
                    "action_value": action_value
                })
                
                pbar.update(1)

            except Exception as e:
                continue
        
        pbar.close()
        return samples

# =============================================================================
# 2. Trajectory Generation Logic
# =============================================================================

def is_point_in_bbox(pt: Tuple[int, int], bbox: List[float]) -> bool:
    x, y = pt
    # Standard check. Using small tolerance is fine.
    return (bbox[0]) <= x <= (bbox[2]) and (bbox[1]) <= y <= (bbox[3])

def get_m2w_trajectory(
    start_pos: Tuple[int, int], 
    target_bbox: List[float], 
    action_type: str, 
    img_size: Tuple[int, int], 
    text_value: str = None
) -> List[Tuple[str, Union[Tuple[int, int], str]]]:
    """
    Generates a CANONICAL (L-Shape) path.
    Prioritizes the larger delta (X or Y) first to reduce zigzag ambiguity.
    Returns [] if target is unreachable.
    """
    x1, y1, x2, y2 = target_bbox
    tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
    img_w, img_h = img_size
    
    path = []
    cx, cy = start_pos
    
    valid_moves = [(k, v) for k, v in MOVE_DELTAS.items() if "MOVE" in k]
    # Sort moves by distance magnitude (Largest moves first)
    valid_moves.sort(key=lambda x: math.hypot(x[1][0], x[1][1]), reverse=True)
    
    max_steps = 20
    
    for _ in range(max_steps):
        # Check if arrived (distance or bbox inclusion)
        dist_to_center = math.hypot(tx - cx, ty - cy)
        if is_point_in_bbox((cx, cy), target_bbox) or dist_to_center < 15:
            break
            
        dx = tx - cx
        dy = ty - cy
        
        # Strategy: Prioritize axis with larger error (> 20px difference)
        prioritize_x = abs(dx) > 20
        
        best_token = None
        best_pos = (cx, cy)
        min_rem = math.hypot(dx, dy)
        found = False
        
        # 1. Filter moves based on priority axis
        candidate_moves = []
        for token, (mx, my) in valid_moves:
            if prioritize_x:
                if abs(my) > abs(mx): continue # Skip Y heavy moves
            else:
                if abs(mx) > abs(my): continue # Skip X heavy moves
            candidate_moves.append((token, mx, my))
            
        # 2. Greedy search in priority moves
        for token, mx, my in candidate_moves:
            nx, ny = cx + mx, cy + my
            if not (0 <= nx < img_w and 0 <= ny < img_h): continue
            
            rem = math.hypot(tx - nx, ty - ny)
            if rem < min_rem:
                min_rem = rem
                best_token = token
                best_pos = (nx, ny)
                found = True
                break 

        # 3. Fallback: If priority axis fails, try all moves
        if not found:
             for token, (mx, my) in valid_moves:
                nx, ny = cx + mx, cy + my
                if not (0 <= nx < img_w and 0 <= ny < img_h): continue
                rem = math.hypot(tx - nx, ty - ny)
                if rem < min_rem:
                    min_rem = rem
                    best_token = token
                    best_pos = (nx, ny)
                    found = True
                    break
        
        if found:
            path.append((best_token, best_pos))
            cx, cy = best_pos
        else:
            break

    # Final Validation
    final_dist = math.hypot(tx - cx, ty - cy)
    if is_point_in_bbox((cx, cy), target_bbox) or final_dist < 30:
        path.append((action_type, (cx, cy)))
        return path
    else:
        # Failed to reach target -> Return empty list
        return []