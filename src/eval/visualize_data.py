import json
import os
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from typing import List, Dict, Any

# =============================================================================
# 0. 外部依赖 (需要 datasets 库来加载真实背景)
# =============================================================================
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    print("[Warning] 'datasets' library not found. Will use black background.")
    HAS_DATASETS = False

# =============================================================================
# 1. MOCK UTILS & PARAMETERS
# =============================================================================

class HP:
    IMAGE_SIZE = 1000  # Qwen-VL default
    SFT_DATA_PATH = "dummy_path.jsonl" # 强制进入合成逻辑

AGENT_SYSTEM_PROMPT = "You are a helpful GUI Agent."

def draw_cursor(image, x, y):
    """画模拟鼠标"""
    draw = ImageDraw.Draw(image)
    r = 10
    # 红色圆圈
    draw.ellipse((x-r, y-r, x+r, y+r), outline="red", width=3)
    # 十字准星
    draw.line((x-r*2, y, x+r*2, y), fill="red", width=2)
    draw.line((x, y-r*2, x, y+r*2), fill="red", width=2)
    return image

# =============================================================================
# 2. CORE LOGIC (CoT & Spatial)
# =============================================================================

def get_grid_region(x: int, y: int, width: int, height: int) -> str:
    if y < height / 3: v_tag = "Top"
    elif y < 2 * height / 3: v_tag = "Mid"
    else: v_tag = "Bottom"
    if x < width / 3: h_tag = "Left"
    elif x < 2 * width / 3: h_tag = "Center"
    else: h_tag = "Right"
    
    if v_tag == "Mid" and h_tag == "Center": return "Center"
    elif v_tag == "Mid": return f"Mid-{h_tag}"
    else: return f"{v_tag}-{h_tag}"

def get_spatial_label_with_cot(cursor_pos, target_center, intent):
    # Navigation
    if intent["type"] == "nav":
        mode = intent["mode"]
        reasoning = "Reasoning: "
        if mode == "back":
            reasoning += "I need to go back. Action: <GO_BACK>"
        elif mode == "home":
            reasoning += "Return to main interface. Action: <GO_HOME>"
        return reasoning
    
    # Interaction
    cx, cy = cursor_pos
    tx, ty = target_center
    dx = tx - cx
    dy = ty - cy
    dist = math.hypot(dx, dy)
    
    c_region = get_grid_region(cx, cy, HP.IMAGE_SIZE, HP.IMAGE_SIZE)
    t_region = get_grid_region(tx, ty, HP.IMAGE_SIZE, HP.IMAGE_SIZE)
    target_name = intent.get('name', 'target')
    
    reasoning = "Reasoning: "
    reasoning += f"Cursor is in **{c_region}**. Target '{target_name}' is in **{t_region}**. "

    rel_cx, rel_cy = round(cx / HP.IMAGE_SIZE, 1), round(cy / HP.IMAGE_SIZE, 1)
    rel_tx, rel_ty = round(tx / HP.IMAGE_SIZE, 1), round(ty / HP.IMAGE_SIZE, 1)
    
    if c_region == t_region:
        reasoning += f"Examining closely. Cursor ~[{rel_cx}, {rel_cy}], Target ~[{rel_tx}, {rel_ty}]. "
    
    HIT_THRESHOLD = 15
    action = ""
    
    if dist < HIT_THRESHOLD:
        reasoning += f"Cursor is **over** target. "
        if intent["type"] == "click": action = "<CLICK_SHORT>"
        elif intent["type"] == "long_click": action = "<CLICK_LONG>"
        elif intent["type"] == "text": 
            content = intent.get("content", "text")
            action = f"<TEXT_START> {content} <TEXT_END>"
        else: action = "<END_ACTION>"
    else:
        # Direction Logic
        if abs(dx) > abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
            d_val = abs(dx)
        else:
            direction = "DOWN" if dy > 0 else "UP"
            d_val = abs(dy)
            
        reasoning += f"**{direction}** is farthest. "
        suffix = "FAR" if d_val > 300 else ("MID" if d_val > 100 else "CLO")
        action = f"<MOVE_{direction}_{suffix}>"
            
    return f"{reasoning}\nAction: {action}"

# =============================================================================
# 3. DATASET (Restored Real Background Logic)
# =============================================================================

class SFTDataset:
    def __init__(self, data_path):
        self.jsonl_data = [] # Empty to force synthetic
        
        # === 恢复真实背景加载逻辑 ===
        self.bg_images = self._load_real_images(limit=50)
        
        self.ui_names = ["Submit", "Cancel", "Search", "Menu", "Settings", "Back", "Profile", "Login", "Options", "Button"]
        self.text_contents = ["hello", "test", "123456", "user", "admin"]
        
        self.prompts_click = ["Click {name}", "Select {name}", "Tap on {name}"]
        self.prompts_move = ["Move to {name}", "Hover over {name}", "Point at {name}"]
        self.prompts_text = ["Type '{content}'", "Enter '{content}' "]
        self.prompts_back = ["Go back", "Return"]
        self.prompts_home = ["Go home", "Main screen"]

    def _load_real_images(self, limit=50):
        """尝试从 ScreenSpot 加载真实 GUI 截图"""
        images = []
        if not HAS_DATASETS:
            return images
            
        print(f"[Dataset] Loading {limit} real backgrounds from ScreenSpot (Streaming)...")
        try:
            # 使用 streaming=True 避免下载整个数据集
            ds = load_dataset("rootsautomation/ScreenSpot", split="test", streaming=True)
            iterator = iter(ds)
            for _ in range(limit):
                try:
                    sample = next(iterator)
                    img = sample['image'].convert("RGB")
                    images.append(img)
                except StopIteration:
                    break
            print(f"[Dataset] Successfully loaded {len(images)} backgrounds.")
        except Exception as e:
            print(f"[Dataset] Failed to load real images: {e}")
            print("[Dataset] Falling back to black backgrounds.")
        return images

    def get_random_background(self):
        """如果有真实背景就用真实的，没有就用黑的"""
        if self.bg_images:
            img = random.choice(self.bg_images).copy()
            # 强制 Resize 到 1000x1000 以匹配合成逻辑的坐标系
            return img.resize((HP.IMAGE_SIZE, HP.IMAGE_SIZE), Image.Resampling.LANCZOS)
        
        # Fallback
        return Image.new("RGB", (HP.IMAGE_SIZE, HP.IMAGE_SIZE), (20, 20, 20))

    def __getitem__(self, idx):
        # 1. 获取背景 (ScreenSpot 或 黑色)
        image = self.get_random_background()
        draw = ImageDraw.Draw(image)
        margin = 50
        
        sample_rng = random.random()
        
        tx = random.randint(margin, HP.IMAGE_SIZE - margin)
        ty = random.randint(margin, HP.IMAGE_SIZE - margin)
        target_name = random.choice(self.ui_names)
        
        # CASE A: Moving (70%)
        if sample_rng < 0.7:
            while True:
                cx = random.randint(margin, HP.IMAGE_SIZE - margin)
                cy = random.randint(margin, HP.IMAGE_SIZE - margin)
                if math.hypot(cx-tx, cy-ty) > 50: break
            
            sub_type = random.choice(["click", "move", "text"])
            if sub_type == "click":
                intent = {"type": "click", "name": target_name}
                instr = random.choice(self.prompts_click).format(name=target_name)
            elif sub_type == "move":
                intent = {"type": "move", "name": target_name}
                instr = random.choice(self.prompts_move).format(name=target_name)
            else:
                content = random.choice(self.text_contents)
                intent = {"type": "text", "name": target_name, "content": content}
                instr = random.choice(self.prompts_text).format(name=target_name, content=content)

        # CASE B: Direct Clicking (15%)
        elif sample_rng < 0.85:
            cx = tx + random.randint(-20, 20)
            cy = ty + random.randint(-20, 20)
            intent = {"type": "click", "name": target_name}
            instr = random.choice(self.prompts_click).format(name=target_name)

        # CASE C: Other Tasks (15%)
        else:
            if random.random() < 0.5:
                # Text Ready
                cx = tx + random.randint(-10, 10)
                cy = ty + random.randint(-10, 10)
                content = random.choice(self.text_contents)
                intent = {"type": "text", "name": target_name, "content": content}
                instr = random.choice(self.prompts_text).format(name=target_name, content=content)
            else:
                # Navigation
                intent_mode = random.choice(["back", "home"])
                intent = {"type": "nav", "mode": intent_mode, "name": "System"}
                cx = random.randint(margin, HP.IMAGE_SIZE - margin)
                cy = random.randint(margin, HP.IMAGE_SIZE - margin)
                tx, ty = 0, 0
                if intent_mode == "back": instr = random.choice(self.prompts_back)
                else: instr = random.choice(self.prompts_home)

        # Draw Target UI Element (Green Box)
        if intent.get("type") != "nav":
            # 颜色鲜艳一点，防止在复杂背景上看不清
            color = random.choice(("green", "lime", "cyan", "magenta"))
            draw.rectangle([tx-40, ty-25, tx+40, ty+25], outline=color, width=4)
            # 在背景上写字
            try: draw.text((tx-30, ty-10), target_name, fill=color)
            except: pass

        # Draw Distractors (干扰项)
        for _ in range(random.randint(2, 5)):
            dx_rand = random.randint(margin, HP.IMAGE_SIZE - margin)
            dy_rand = random.randint(margin, HP.IMAGE_SIZE - margin)
            if abs(dx_rand - tx) > 60 or abs(dy_rand - ty) > 60:
                # 干扰项用灰色或黑色，稍微细一点
                draw.rectangle([dx_rand-30, dy_rand-20, dx_rand+30, dy_rand+20], outline="blue", width=2)

        # Draw Cursor
        image = draw_cursor(image, cx, cy)

        # Generate CoT
        cot_response = get_spatial_label_with_cot((cx, cy), (tx, ty), intent)
        
        msgs = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"[Action] {instr}"},
            {"role": "assistant", "content": cot_response}
        ]
        
        return {"messages": msgs, "image": image}

# =============================================================================
# 4. RUN VISUALIZATION
# =============================================================================

def run_visualization():
    print("Initializing Dataset with Real Backgrounds...")
    dataset = SFTDataset(HP.SFT_DATA_PATH)
    
    output_dir = "vis_results_real"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating 4 samples in '{output_dir}/'...")
    
    for i in range(4):
        # 随机取索引，增加背景多样性
        sample = dataset[random.randint(0, 100)]
        image = sample["image"]
        msgs = sample["messages"]
        
        # Save Image
        img_filename = os.path.join(output_dir, f"sample_real_{i}.png")
        image.save(img_filename)
        
        # Save Text Log
        log_filename = os.path.join(output_dir, f"sample_real_{i}.txt")
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write(f"=== SAMPLE {i} ===\n")
            for m in msgs:
                role = m["role"].upper()
                content = m["content"]
                f.write(f"\n[{role}]\n{content}\n")
                f.write("-" * 40)
        
        print(f"Saved {img_filename}")
        print(f"Instruction: {msgs[1]['content']}")
        print(f"Action: {msgs[2]['content'].split('Action:')[-1].strip()}")
        print("-" * 20)

if __name__ == "__main__":
    run_visualization()