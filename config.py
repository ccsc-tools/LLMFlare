"""Configuration module for LLMFlare package.

Centralizes path management, model names, and hyperparameters across all scripts.
Uses relative paths for portability across environments.
"""

import os
import argparse

# ============================================================================
# PATH CONFIGURATION (Relative paths for portability)
# ============================================================================
DATA_DIR = "data"
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = "model"

# Data files
X_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "X_train.npy")
X_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "y_test.npy")
FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "selected_features.json")
SHAP_VALUES_PATH = os.path.join(PROCESSED_DATA_DIR, "shap_values_full.npy")
MTST_MODEL_PATH = os.path.join(MODEL_DIR, "mtst_flare_model.pth")

# LLM datasets
LLM_TRAINING_DATASET_PATH = os.path.join(DATA_DIR, "llm_training_dataset_train_split.csv")
LLM_EVAL_PATH = os.path.join(DATA_DIR, "llm_eval_200.csv")

# LLM model paths
LLM_ADAPTER_PATH = os.path.join(MODEL_DIR, "llm_analyst_invincible_model_V2")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# MTST Architecture
MTST_CONFIG = {
    "num_features": 16,  # Will be overridden by actual feature count
    "time_steps": 24,
    "patch_length": 2,
    "d_model": 128,
    "num_heads": 8,
    "num_encoder_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.1,
}

# LLM Models
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B"
JUDGE_MODEL_NAME = "gemini-2.5-pro"

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
# Stage 1 (Explanation-only)
STAGE1_BATCH_SIZE = 1
STAGE1_GRAD_ACCUMULATION = 4
STAGE1_LR = 2.98e-4
STAGE1_EPOCHS = 1

# Stage 2 (All skills)
STAGE2_BATCH_SIZE = 1
STAGE2_GRAD_ACCUMULATION = 4
STAGE2_LR = 2.98e-4 / 2
STAGE2_EPOCHS = 1

# LoRA Configuration
LORA_R = 32
LORA_ALPHA = 128
LORA_DROPOUT = 0.066

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
NUM_EVAL_SAMPLES = 200
MAX_RETRIES = 3
EVAL_MAX_NEW_TOKENS = 256
DELAY_BETWEEN_REQUESTS = 1

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_google_api_key():
    """Load Google API key from environment variable."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
    return api_key


def create_argparse_for_script(script_name):
    """Factory function to create common argparse configs."""
    parser = argparse.ArgumentParser(
        description=f"LLMFlare package script: {script_name}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser
