# src/utils/parameters.py

class HYPERPARAMS:
    # ================= PATHS =================
    BASE_MODEL_PATH = "./checkpoints/Qwen3-VL-4B-Instruct"
    INIT_MODEL_PATH = "./checkpoints/Qwen3-VL-GUI-Init"

    # ================= TRAINING PARAMS FOR SFT =================
    # SFT (Semantic injection)
    SFT_SAMPLES_PER_ACTION = 90
    SFT_DATA_PATH = "./data/sft.jsonl"
    SFT_OUTPUT_PATH = "./checkpoints/Qwen3-VL-GUI-SFT"
    
    # Image
    IMAGE_SIZE = 1920
    
    # Optimization
    SFT_LEARN_RATE = 1.5e-5
    SFT_BATCH_SIZE = 1
    SFT_GRAD_ACCUM_STEPS = 8
    SFT_EPOCHS = 1
    SFT_MAX_LENGTH = 4096
    
    # Seeds
    SFT_SEED = 42
    
    # ================= PHASE 2: SFT (ScreenSpot Trajectory) =================
    # Output of Stage 1 is Input of Stage 2
    SFT_2_INPUT_PATH = "./checkpoints/Qwen3-VL-GUI-SFT" 
    SFT_2_OUTPUT_PATH = "./checkpoints/Qwen3-VL-GUI-SFT-ScreenSpot"
        
    # ScreenSpot Config
    SCREENSPOT_DATA_PATH = "rootsautomation/ScreenSpot"
    SCREENSPOT_TOTAL_SIZE = 1200  # Load 1200 samples total
    SCREENSPOT_TRAIN_RATIO = 0.8  # 80% for Trajectory Training
    SCREENSPOT_EVAL_RATIO = 0.1   # 10% for Eval (Loss calculation)
    # Remaining 10% is reserved as 'Test' (Raw data)
        
    SFT_2_LEARN_RATE = 1e-5 # Slightly lower LR for Stage 2
    SFT_2_BATCH_SIZE = 1    # Images are heavy, smaller batch
    SFT_2_GRAD_ACCUM_STEPS = 8
    SFT_2_EPOCHS = 3        # More epochs to learn visual features

    SFT_3_OUTPUT_PATH = "./checkpoints/Qwen3-VL-GUI-SFT-ScreenSpot-Pro"
    SCREENSPOT_PRO_TOTAL_SIZE = 1800

    SFT_3_LEARN_RATE = 5e-6 # Slightly lower LR for Stage 2
    SFT_3_BATCH_SIZE = 1    # Images are heavy, smaller batch
    SFT_3_GRAD_ACCUM_STEPS = 8
    SFT_3_EPOCHS = 2        # More epochs to learn visual features
    
    SFT_4_INPUT_PATH = "./checkpoints/Qwen3-VL-GUI-SFT-ScreenSpot"
    SFT_4_OUTPUT_PATH = "./checkpoints/Qwen3-VL-GUI-SFT-Mind2Web"

    # Mind2Web Data Config
    M2W_TOTAL_SIZE = 4000       # Reasonable size for SFT generalization
    M2W_TRAIN_RATIO = 0.8       # 80% Train
    M2W_EVAL_RATIO = 0.1        # 10% Eval
    M2W_TEST_RATIO = 0.1
    M2W_CACHE_PATH = "./data/mind2web_resized/"

    # Training Config (Fine-tuning requires lower LR)
    SFT_4_LEARN_RATE = 5e-6    
    SFT_4_BATCH_SIZE = 1        
    SFT_4_GRAD_ACCUM_STEPS = 8 # High accumulation for stability
    SFT_4_EPOCHS = 2

    # ================= EVALUATION PARAMS =================
    VLM_API_KEY = "sk-e9d4a0b8248b4802b0458ca26612f25c"
    VLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    BASELINE_MODEL_NAME = "qwen3-vl-235b-a22b-instruct" 
    
    # Evaluation Config
    EVAL_DATA_PATH = "rootsautomation/ScreenSpot"
    EVAL_DATASET_SIZE = 10 # Number of samples to test
    EVAL_MAX_STEPS = 30
    EVAL_OUTPUT_DIR = "./eval_results"