# src/utils/parameters.py

class HYPERPARAMS:
    # ================= PATHS =================
    BASE_MODEL_PATH = "./checkpoints/Qwen3-VL-4B-Instruct"
    INIT_MODEL_PATH = "./checkpoints/Qwen3-VL-GUI-Initialized"
    
    # SFT 1 (Semantic Injection)
    SFT_1_DATA_PATH = "./data/sft_1.jsonl"
    SFT_1_OUTPUT_PATH = "./checkpoints/Qwen3-VL-GUI-SFT-1"
    
    # ================= TRAINING PARAMS FOR SFT1 =================
    # Image
    IMAGE_SIZE = 640
    
    # Optimization
    SFT_LEARN_RATE = 1e-5
    SFT_BATCH_SIZE = 4
    SFT_GRAD_ACCUM_STEPS = 4
    SFT_EPOCHS = 1
    SFT_MAX_LENGTH = 1024
    
    # Seeds
    SEED = 42