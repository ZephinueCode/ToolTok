import os
import json
import torch
import base64
from io import BytesIO
from tqdm import tqdm
from datetime import datetime
from PIL import Image

# Hugging Face libraries
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset

# --- Configuration ---
MODEL_PATH = "./checkpoints/Qwen3-VL-4B-Instruct" 
OUTPUT_DIR = "./results/simplevqa_qwen3_hf"
MAX_NEW_TOKENS = 128

def decode_base64_image(data):
    """Helper: Decode string/bytes to PIL Image"""
    try:
        # If it's already an image, return it
        if isinstance(data, Image.Image):
            return data
        
        # If it's a string (Base64), decode it
        if isinstance(data, str):
            if "," in data:
                data = data.split(",")[1]
            image_data = base64.b64decode(data)
            return Image.open(BytesIO(image_data)).convert("RGB")
        
        # If it's bytes, open it
        if isinstance(data, bytes):
            return Image.open(BytesIO(data)).convert("RGB")
            
        return None
    except Exception as e:
        # print(f"[Debug] Image decode error: {e}") # Uncomment to debug
        return None

def main():
    # 1. Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[Init] Model: {MODEL_PATH}")

    # 2. Load Model
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"[Fatal] Failed to load model: {e}")
        return

    # 3. Load Dataset
    print("[Init] Loading dataset from Hugging Face...")
    try:
        # split="test" ensures we get the dataset object, not a dict of splits
        ds = load_dataset("m-a-p/SimpleVQA", split="test")
    except Exception as e:
        print(f"[Fatal] Failed to load dataset: {e}")
        return

    print(f"[Init] Total Samples: {len(ds)}")

    # 4. Inference Loop
    results = []
    correct_count = 0
    total_count = 0
    
    model.eval()

    # Iterate
    for i, sample in tqdm(enumerate(ds), total=len(ds)):
        try:
            # --- Extract Data ---
            raw_image = sample.get("image") 
            question = sample.get("question", "")
            gt_answer = str(sample.get("answer", "")).strip()
            data_id = sample.get("data_id", i)

            # --- Fix: Handle Data Type (String vs Image) ---
            image = decode_base64_image(raw_image)

            # If image is still invalid, skip
            if image is None:
                # print(f"[Warn] Sample {i} has invalid image data.")
                continue

            # --- Prepare Prompt ---
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]

            # Text Prompt
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Image Inputs
            image_inputs = [image]
            
            # Process
            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=None, 
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            # --- Generate ---
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False
                )

            # --- Decode ---
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            # --- Metric ---
            is_correct = False
            if gt_answer:
                # Case insensitive check
                if (gt_answer.lower() in output_text.lower()) or \
                   (output_text.lower() in gt_answer.lower()):
                    is_correct = True
            
            if is_correct:
                correct_count += 1
            total_count += 1

            results.append({
                "id": str(data_id),
                "question": question,
                "prediction": output_text,
                "gt": gt_answer,
                "correct": is_correct
            })

        except Exception as e:
            print(f"[Error] Sample {i}: {e}")
            continue

    # 5. Report
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print("\n" + "="*40)
    print(f"EVAL REPORT: SimpleVQA")
    print(f"Valid Samples: {total_count}")
    print(f"Accuracy:      {accuracy:.2%}")
    print("="*40 + "\n")

    report_path = os.path.join(OUTPUT_DIR, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {"model": MODEL_PATH, "accuracy": accuracy},
            "details": results
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()