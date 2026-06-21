# 5_generate_paper_examples_solar.py
"""Generate high-risk and low-risk examples for paper LaTeX tables.

Demonstrates the model's explanation, recommendation, and counterfactual
generation capabilities on real solar flare data.
"""

# NOTE: This script requires the MTST model.
# Download from: https://zenodo.org/records/16780646

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    X_TEST_PATH,
    FEATURES_PATH,
    SHAP_VALUES_PATH,
    MTST_MODEL_PATH,
    LLM_ADAPTER_PATH,
    BASE_MODEL_NAME,
    MTST_CONFIG,
)
from utils.model_arch import MTST

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def format_inference_prompt(instruction, prompt_data):
    """Format a prompt for inference."""
    return f"### Instruction:\n{instruction}\n\n### Input:\n{prompt_data}\n\n### Response:\n"


def generate_full_response(model, tokenizer, prompt_data):
    """Generate explanations, recommendations, and counterfactuals for given data.

    Args:
        model: Fine-tuned model instance.
        tokenizer: Tokenizer instance.
        prompt_data: Input data as string.

    Returns:
        dict: Responses keyed by skill (Explanation, Recommendation, Counterfactual).
    """
    responses = {}
    instructions = {
        "Explanation": "Analyze the provided solar flare prediction data and generate a summary of the key drivers.",
        "Recommendation": "Based on the input, what is the recommended next step?",
        "Counterfactual": "Based on the input, what would need to change to flip the prediction?"
    }

    for skill, instruction in instructions.items():
        prompt_text = format_inference_prompt(instruction, prompt_data)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id, do_sample=False, repetition_penalty=1.2
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        responses[skill] = response.strip()

    return responses

def print_latex_table(role, utterance):
    utterance = utterance.replace('&', '\\&').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#').replace('_', '\\_').replace('{', '\\{').replace('}', '\\}')
    print(f"\\textbf{{{role}}} & {utterance} \\\\")
    print("\\hline")


def main(args=None):
    """Generate high-risk and low-risk solar flare examples for paper LaTeX tables.

    Args:
        args: Argparse namespace (optional). If None, uses defaults from config.
    """
    if args is None:
        args = argparse.Namespace()

    logging.info("Loading V2 AI Analyst model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, LLM_ADAPTER_PATH).merge_and_unload()
    model.eval()
    logging.info("Model loaded.")

    logging.info("Loading solar flare data and MTST model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(FEATURES_PATH, 'r') as f:
        feature_names = json.load(f)
    X_test = np.load(X_TEST_PATH)
    mtst_model = MTST(
        num_features=len(feature_names),
        **{k: v for k, v in MTST_CONFIG.items() if k != 'num_features'}
    )
    mtst_model.load_state_dict(torch.load(MTST_MODEL_PATH, map_location=device))
    mtst_model.to(device).eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(
            mtst_model(torch.from_numpy(X_test).float().to(device))
        ).squeeze().cpu().numpy()
    base_df = pd.DataFrame(X_test[:, -1, :], columns=feature_names)
    base_df['prediction_probability'] = probabilities
    shap_values_full = np.load(SHAP_VALUES_PATH)
    shap_df = pd.DataFrame(
        np.mean(shap_values_full, axis=1).squeeze(),
        columns=[f"SHAP_{f}" for f in feature_names]
    )
    full_df = pd.concat([base_df.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)

    high_risk_example = full_df.loc[full_df['prediction_probability'].idxmax()]
    high_risk_prompt_data = ", ".join([f"{col}: {val:.4f}" for col, val in high_risk_example.items()])
    high_risk_responses = generate_full_response(model, tokenizer, high_risk_prompt_data)

    print("\n\n" + "="*80)
    print("--- HIGH-RISK SOLAR FLARE LATEX TABLE (COPY THIS) ---")
    print("="*80)
    print_latex_table("User Input", f"\\texttt{{{high_risk_prompt_data[:100]}...}}")
    print_latex_table("AI Analyst (Explanation)", high_risk_responses["Explanation"])
    print_latex_table("AI Analyst (Recommendation)", high_risk_responses["Recommendation"])
    print_latex_table("AI Analyst (Counterfactual)", high_risk_responses["Counterfactual"])

    low_risk_example = full_df[
        (full_df['prediction_probability'] > 0.001) & (full_df['prediction_probability'] < 0.01)
    ].iloc[0]
    low_risk_prompt_data = ", ".join([f"{col}: {val:.4f}" for col, val in low_risk_example.items()])
    low_risk_responses = generate_full_response(model, tokenizer, low_risk_prompt_data)

    print("\n" + "="*80)
    print("--- LOW-RISK SOLAR FLARE LATEX TABLE (COPY THIS) ---")
    print("="*80)
    print_latex_table("User Input", f"\\texttt{{{low_risk_prompt_data[:100]}...}}")
    print_latex_table("AI Analyst (Explanation)", low_risk_responses["Explanation"])
    print_latex_table("AI Analyst (Recommendation)", low_risk_responses["Recommendation"])
    print_latex_table("AI Analyst (Counterfactual)", low_risk_responses["Counterfactual"])
    print("\n")

if __name__ == "__main__":
    main()
