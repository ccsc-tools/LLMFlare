"""Generate SHAP values for solar flare prediction using the MTST model.

This script computes gradient-based SHAP explanations for the entire test set,
which are later used to generate natural language explanations for fine-tuning.
"""

# NOTE: This script requires the MTST solar flare predictor model.
# Download from: https://zenodo.org/records/16780646
# If using pre-computed data (X_test.npy, shap_values_full.npy),
# skip this script and start from 2_create_narratives_invincible_V2.py

import argparse
import logging
import os

import json
import numpy as np
import pandas as pd
import shap
import torch

from config import (
    X_TRAIN_PATH,
    X_TEST_PATH,
    FEATURES_PATH,
    MTST_MODEL_PATH,
    SHAP_VALUES_PATH,
    MTST_CONFIG,
)
from utils.model_arch import MTST

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(args=None):
    """Generate SHAP values for the full test set.

    Args:
        args: Argparse namespace (optional). If None, uses defaults from config.
    """
    if args is None:
        args = argparse.Namespace()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info("Loading data and model...")
    X_test = np.load(X_TEST_PATH)
    X_train = np.load(X_TRAIN_PATH)
    with open(FEATURES_PATH, 'r') as f:
        feature_names = json.load(f)

    model = MTST(
        num_features=len(feature_names),
        **{k: v for k, v in MTST_CONFIG.items() if k != 'num_features'}
    )
    model.load_state_dict(torch.load(MTST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    logging.info("Model and data loaded successfully.")

    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)

    logging.info("Creating GradientExplainer...")
    background = X_train_tensor[np.random.choice(X_train_tensor.shape[0], 100, replace=False)]
    explainer = shap.GradientExplainer(model, background)

    logging.info(f"Calculating SHAP values for the FULL test set ({len(X_test_tensor)} samples). This will take several hours...")
    shap_values_full = explainer.shap_values(X_test_tensor)
    logging.info(f"SHAP values calculated. Shape: {shap_values_full.shape}")

    np.save(SHAP_VALUES_PATH, shap_values_full)
    logging.info(f"Full SHAP values saved to {SHAP_VALUES_PATH}")
    logging.info("Script finished successfully.")

if __name__ == "__main__":
    main()
