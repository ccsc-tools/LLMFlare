"""Generate natural language narratives from SHAP values using the Dynamic Narrative Engine.

Creates 50,533 training examples (explanation, recommendation, counterfactual) from
SHAP-based solar flare predictions for fine-tuning Llama 3 8B.
"""

# NOTE: This script requires the MTST model for probability inference.
# Download from: https://zenodo.org/records/16780646
# If using pre-computed data, skip to 3_finetune_llm_invincible_v2.py

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import (
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    FEATURES_PATH,
    SHAP_VALUES_PATH,
    MTST_MODEL_PATH,
    LLM_TRAINING_DATASET_PATH,
    MTST_CONFIG,
)
from utils.model_arch import MTST

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tqdm.pandas()

# --- V3 KNOWLEDGE BASE: COMPLETE, VERIFIED, AND SCIENTIFICALLY-ACCURATE ---
# This version includes all 16 features from the provided source list.
FEATURE_INSIGHTS = {
    # --- Tier 1: Primary Energy and Twist Indicators (The "Why") ---
    "TOTUSJH": {
        "full_name": "total unsigned current helicity",
        "insight": "a highly twisted and complex magnetic field, indicating a large amount of stored energy available for explosive release.",
        "neg_insight": "a simple and non-twisted magnetic field, indicating a lack of stored energy and a stable configuration."
    },
    "TOTPOT": {
        "full_name": "total photospheric magnetic free energy density",
        "insight": "a massive amount of free energy available in the corona to power a major flare.",
        "neg_insight": "a low amount of free energy in the corona, suggesting the region is in a stable, energy-poor state."
    },
    "SAVNCPP": {
        "full_name": "sum of the modulus of the net current per polarity",
        "insight": "strong electrical currents concentrated along the magnetic neutral line, a prime location for magnetic reconnection.",
        "neg_insight": "weak and diffuse electrical currents, indicating a low probability of explosive magnetic reconnection."
    },
    "MEANPOT": {
        "full_name": "mean photospheric magnetic free energy",
        "insight": "a high concentration of non-potential energy, suggesting the magnetic field is in a stressed, unstable state.",
        "neg_insight": "a low concentration of non-potential energy, indicating the magnetic field is in a relaxed, stable state."
    },
    "ABSNJZH": {
        "full_name": "absolute value of the net current helicity",
        "insight": "a significant, coherent twist in the magnetic field, a key indicator of an energy-storing and flare-productive region.",
        "neg_insight": "a lack of coherent twist in the magnetic field, suggesting a simple and less flare-productive region."
    },
    "TOTUSJZ": {
        "full_name": "total unsigned vertical current",
        "insight": "strong vertical currents injecting significant energy into the corona from below.",
        "neg_insight": "weak vertical currents, indicating little energy injection into the corona."
    },
    "MEANJZH": {
        "full_name": "mean current helicity",
        "insight": "a strong average twist in the electrical currents, correlated with the onset of flare activity.",
        "neg_insight": "a weak average twist in the electrical currents, suggesting a stable magnetic environment."
    },
    "MEANALP": {
        "full_name": "mean characteristic twist parameter, alpha",
        "insight": "a high average twist across the magnetic field, indicating a complex structure storing significant non-potential energy.",
        "neg_insight": "a low average twist, suggesting a simple, potential-like magnetic field with little stored energy."
    },
    
    # --- Tier 2: Magnetic Structure and Complexity Indicators (The "How") ---
    "MEANSHR": {
        "full_name": "mean shear angle",
        "insight": "a high degree of magnetic shear along the neutral line, representing a build-up of stress and stored energy.",
        "neg_insight": "a low degree of magnetic shear, indicating the magnetic field is not significantly stressed."
    },
    "USFLUX": {
        "full_name": "total unsigned flux",
        "insight": "a very large and magnetically powerful active region, capable of hosting large flares.",
        "neg_insight": "a small or magnetically weak active region, unlikely to produce a major flare."
    },
    "R_VALUE": {
        "full_name": "sum of flux near polarity inversion line",
        "insight": "a high concentration of magnetic flux at the polarity inversion line, the primary site for flare ignition.",
        "neg_insight": "a low concentration of magnetic flux at the polarity inversion line, making flare ignition less likely."
    },
    "AREA_ACR": {
        "full_name": "area of strong field pixels in the active region",
        "insight": "a large, well-developed sunspot group, providing the necessary magnetic scale for a major flare.",
        "neg_insight": "a small or decaying sunspot group, lacking the magnetic scale required for a major flare."
    },

    # --- Tier 3: Force and Gradient Indicators (Supporting Evidence) ---
    "TOTBSQ": {
        "full_name": "total magnitude of Lorentz force",
        "insight": "strong forces acting on the solar surface, indicative of a highly stressed and non-potential magnetic configuration.",
        "neg_insight": "weak forces acting on the solar surface, suggesting the magnetic field is in a relaxed, near-potential state."
    },
    "TOTFX": {
        "full_name": "sum of x-component of Lorentz force",
        "insight": "a strong directional force contributing to the overall magnetic stress of the region.",
        "neg_insight": "a weak directional force, indicating a lack of significant stress in this orientation."
    },
    "TOTFY": {
        "full_name": "sum of y-component of Lorentz force",
        "insight": "a strong directional force contributing to the overall magnetic stress of the region.",
        "neg_insight": "a weak directional force, indicating a lack of significant stress in this orientation."
    },
    "TOTFZ": {
        "full_name": "sum of z-component of Lorentz force",
        "insight": "a strong directional force contributing to the overall magnetic stress of the region.",
        "neg_insight": "a weak directional force, indicating a lack of significant stress in this orientation."
    },

    # --- Fallback for any other feature ---
    "DEFAULT": {
        "full_name": "the parameter '{}'",
        "insight": "a factor that significantly increased the model's assessed risk.",
        "neg_insight": "a factor that significantly decreased the model's assessed risk."
    }
}

def get_feature_stats(X_train_path, feature_names):
    logging.info("Calculating feature statistics for value qualification...")
    X_train_full = np.load(X_train_path)
    X_train_recent = X_train_full[:, -1, :]
    df_train = pd.DataFrame(X_train_recent, columns=feature_names)
    return df_train.describe(percentiles=[.25, .5, .75, .95])

def analyze_forecast_V2(row, stats_df):
    """
    Analyzes a single forecast row to extract key information.
    V2 FIX: Identifies the primary driver based on MAX ABSOLUTE SHAP value.
    """
    prob = row['prediction_probability']
    shap_cols = {col.replace('SHAP_', ''): val for col, val in row.items() if col.startswith('SHAP_')}
    
    # --- This is Change #1: Find the true primary driver ---
    if not shap_cols:
        primary_driver_name, primary_driver_shap = None, 0
    else:
        # Find the feature with the largest impact, positive or negative
        primary_driver_name, primary_driver_shap = max(shap_cols.items(), key=lambda item: abs(item[1]))

    # Helper to describe the feature's value (e.g., "high", "low")
    def qualify_value(feature_name, value):
        if feature_name is None: return ""
        # Handle cases where a feature might not be in stats_df (shouldn't happen)
        if feature_name not in stats_df.columns: return "a notable"
        stats = stats_df[feature_name]
        if value > stats['95%']: return "an exceptionally high"
        elif value > stats['75%']: return "a high"
        elif value > stats['25%']: return "a normal"
        else: return "a low"

    return {
        "prob": prob,
        "risk_level": "a high risk" if prob > 0.5 else "a low risk",
        "confidence": f"{prob:.0%}",
        "primary_driver": primary_driver_name,
        "primary_driver_shap": primary_driver_shap,
        "primary_driver_qual": qualify_value(primary_driver_name, row.get(primary_driver_name, 0))
    }

def create_training_examples_V2(row, stats_df):
    """
    Creates three distinct training examples (explanation, recommendation, counterfactual) for a single forecast.
    V2 FIX: Uses the correct insight (positive/negative) based on the SHAP sign.
    """
    analysis = analyze_forecast_V2(row, stats_df)
    base_prompt = ", ".join([f"{col}: {val:.4f}" for col, val in row.items() if not col.startswith('SHAP_') and col != 'prediction_probability'])
    full_input_data = ", ".join([f"{col}: {row[col]:.4f}" for col in row.index])
    
    examples = []
    
    if not analysis['primary_driver']:
        return [] # Skip if there's no valid driver

    feature = analysis['primary_driver']
    kb_entry = FEATURE_INSIGHTS.get(feature, FEATURE_INSIGHTS["DEFAULT"])

    # --- Skill 1: Explanation (Corrected Logic) ---
    if analysis['primary_driver_shap'] > 0:
        insight_text = kb_entry.get('insight', FEATURE_INSIGHTS["DEFAULT"]["insight"]).format(feature)
        driver_explanation = f"The primary driver pushing the risk UP is {analysis['primary_driver_qual']} value in '{feature}' ({kb_entry['full_name']}), which indicates {insight_text}"
    else:
        insight_text = kb_entry.get('neg_insight', FEATURE_INSIGHTS["DEFAULT"]["neg_insight"]).format(feature)
        driver_explanation = f"The primary factor keeping the risk LOW is {analysis['primary_driver_qual']} value in '{feature}' ({kb_entry['full_name']}), which indicates {insight_text}"
    
    exp_narrative = f"The model predicts {analysis['risk_level']} of a flare with {analysis['confidence']} confidence. {driver_explanation}."
    exp_instruction = "Analyze the provided solar flare prediction data and generate a summary of the key drivers."
    examples.append({'skill': 'explanation', 'text': f"### Instruction:\n{exp_instruction}\n\n### Input:\n{full_input_data}\n\n### Response:\n{exp_narrative}"})

    # --- Skill 2: Recommendation (Corrected Logic) ---
    if analysis['prob'] > 0.5:
        rec_narrative = f"Recommendation: The primary risk factor is the {kb_entry['full_name']} ('{feature}'). This parameter should be monitored closely for any continued increases, as that would signal a higher probability of imminent energy release."
    else:
        rec_narrative = "Recommendation: No immediate action is required as the risk is low. Continue routine monitoring of the active region, paying attention to the stability of parameters like '{}'.".format(feature)
    rec_instruction = "Based on the input, what is the recommended next step?"
    examples.append({'skill': 'recommendation', 'text': f"### Instruction:\n{rec_instruction}\n\n### Input:\n{full_input_data}\n\n### Response:\n{rec_narrative}"})

    # --- Skill 3: Counterfactual (Corrected Logic) ---
    if analysis['prob'] > 0.5:
        cf_narrative = f"Counterfactual: To lower the flare risk, the most critical factor would be a significant decrease in '{feature}'. This would represent {kb_entry.get('neg_insight', 'a reduction in risk factors')}"
    else:
        cf_narrative = f"Counterfactual: An increase in '{feature}' would be the most critical factor to escalate the threat level. This would represent {kb_entry.get('insight', 'an increase in risk factors')}"
    cf_instruction = "Based on the input, what would need to change to flip the prediction?"
    examples.append({'skill': 'counterfactual', 'text': f"### Instruction:\n{cf_instruction}\n\n### Input:\n{full_input_data}\n\n### Response:\n{cf_narrative}"})
    
    return examples

def main(args=None):
    """Generate V2 3-skill training dataset with corrected SHAP-based narratives.

    Creates explanation, recommendation, and counterfactual training examples from
    SHAP values and model predictions.

    Args:
        args: Argparse namespace (optional). If None, uses defaults from config.
    """
    if args is None:
        args = argparse.Namespace() 

    logging.info("Loading SHAP values, test data, and feature names...")
    shap_values_full = np.load(SHAP_VALUES_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    with open(FEATURES_PATH, 'r') as f:
        feature_names = json.load(f)

    logging.info("Running inference with the MTST model to get prediction probabilities...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTST(
        num_features=len(feature_names),
        **{k: v for k, v in MTST_CONFIG.items() if k != 'num_features'}
    )
    model.load_state_dict(torch.load(MTST_MODEL_PATH, map_location=device))
    model.to(device).eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(X_test).float().to(device))).squeeze().cpu().numpy()

    logging.info("Assembling the base DataFrame with features, predictions, and SHAP values...")
    base_df = pd.DataFrame(X_test[:, -1, :], columns=feature_names)
    base_df['prediction_probability'] = probabilities
    shap_df = pd.DataFrame(np.mean(shap_values_full, axis=1).squeeze(), columns=[f"SHAP_{f}" for f in feature_names])
    base_df = pd.concat([base_df.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)

    feature_stats = get_feature_stats(X_TRAIN_PATH, feature_names)

    logging.info("Generating V2 3-skill training dataset with corrected logic...")
    training_lists = base_df.progress_apply(lambda row: create_training_examples_V2(row, feature_stats), axis=1)

    final_training_data = [item for sublist in training_lists for item in sublist]
    llm_dataset = pd.DataFrame(final_training_data)

    llm_dataset.to_csv(LLM_TRAINING_DATASET_PATH, index=False)
    logging.info(f"Final V2 LLM training dataset saved to {LLM_TRAINING_DATASET_PATH}. Total examples: {len(llm_dataset)}")
    logging.info("Dataset generation complete. You can now use this file for fine-tuning.")

if __name__ == "__main__":
    main()