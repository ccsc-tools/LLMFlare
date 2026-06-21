# LLMFlare: Explainable Solar Flare Prediction Using Large Language Models

This repository contains the code and data for the LLMFlare package.

## Overview

This package implements a "Predictor-Analyst" framework that:
1. Uses a pre-trained deep learning model to predict solar flares
2. Computes SHAP values for each prediction
3. Fine-tunes Llama 3 8B to generate natural language explanations (Explanation, Recommendation, Counterfactual)
4. Evaluates the AI Analyst using Google Gemini 2.5 Pro as an LLM-as-a-Judge

## Repository Structure

```
LLMFlare/
├── data/
│   ├── processed/
│   │   ├── X_test.npy                              # Test set features (16,911 x 24 x 16)
│   │   ├── shap_values_full.npy                    # SHAP values for test set
│   │   └── scaler.pkl                              # Feature scaler
│   ├── llm_training_dataset_invincible_V2.csv      # Full synthetic dataset (50,733 examples)
│   ├── llm_training_dataset_train_split.csv        # Training split (50,533 examples)
│   └── llm_eval_200.csv                            # Evaluation split (200 examples)
├── model/
│   └── llm_analyst_invincible_model_V2/            # Fine-tuned LoRA adapter weights
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       └── tokenizer files
├── 1_generate_shap_data.py                         # Generate SHAP values (requires MIST model)
├── 2_create_narratives_invincible_V2.py            # Dynamic Narrative Engine
├── 3_finetune_llm_invincible_v2.py                 # Fine-tune Llama 3 8B
├── 4_evaluate_model_invincible_v2.py               # Evaluate fine-tuned model
├── 4b_evaluate_zeroshot_baseline.py                # Evaluate zero-shot baseline
├── 5_generate_paper_examples_solar.py              # Generate paper examples (requires MIST model)
├── config.py                                       # Centralized configuration and paths
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11
- PyTorch 2.x
- CUDA-capable GPU (A100 recommended for fine-tuning)

Install dependencies:
```bash
pip install torch transformers peft trl google-generativeai optuna shap pandas numpy scikit-learn accelerate bitsandbytes
```

## Setup

### 1. HuggingFace Access
Request access to Llama 3 8B at:
https://huggingface.co/meta-llama/Llama-3.1-8B

Then set your token:
```bash
export HUGGING_FACE_HUB_TOKEN="your_token_here"
```

### 2. Google Gemini API Key
Get a key at: https://aistudio.google.com

```bash
export GOOGLE_API_KEY="your_key_here"
```

## Running the Pipeline

### Option A — Run inference only (fastest, recommended)

The fine-tuned LoRA adapter and pre-computed data are included. No GPU needed for inference:

```bash
export HUGGING_FACE_HUB_TOKEN="your_token"
export GOOGLE_API_KEY="your_key"
python 4_evaluate_model_invincible_v2.py
```

### Option B — Re-generate training data and fine-tune (requires MIST model)

**⚠️ Important**: Scripts 1 and 2 require the MIST solar flare predictor model, which is **not included** in this repository.
The pre-computed outputs (`X_test.npy` and `shap_values_full.npy`) are already provided.

To get the MIST model:
- See: https://zenodo.org/records/16780646
- Or contact the FLAIRS paper authors

If you have the MIST model, the full pipeline is:

**Step 1: Generate SHAP values** (requires MIST model)
```bash
python 1_generate_shap_data.py
```

**Step 2: Create training narratives**
```bash
python 2_create_narratives_invincible_V2.py
```

**Step 3: Fine-tune Llama 3 8B** (requires A100 GPU, ~2 hours)
```bash
python 3_finetune_llm_invincible_v2.py
```
Uses two-stage curriculum learning:
- Stage 1: Train on Explanation skill (LR = 2.98e-4)
- Stage 2: Train on all three skills (LR = 1.49e-4)

**Step 4: Evaluate fine-tuned model**
```bash
export GOOGLE_API_KEY="your_key"
python 4_evaluate_model_invincible_v2.py
```

**Step 5: Evaluate zero-shot baseline**
```bash
export GOOGLE_API_KEY="your_key"
python 4b_evaluate_zeroshot_baseline.py
```

**Step 6: Generate paper examples** (requires MIST model)
```bash
python 5_generate_paper_examples_solar.py
```

## Expected Results

| Model | Faithfulness | Insightfulness | Fluency |
|---|---|---|---|
| Zero-shot Llama 3 8B | 1.85 ± 1.99 | 1.29 ± 0.65 | 9.21 ± 1.72 |
| Fine-tuned AI Analyst | 8.04 ± 2.96 | 6.07 ± 3.88 | 9.93 ± 0.32 |

Scores are mean ± std across 200 evaluated examples, judged by Google Gemini 2.5 Pro.

## Data Sources

- SHARP solar magnetogram data: http://jsoc.stanford.edu/HMI/HARPS.html
- Pre-trained solar flare predictor (MIST model): https://zenodo.org/records/16780646

## Notes for SLURM Users

SLURM scripts are provided for HPC cluster users. Before running:
1. Update `ENV_PATH` to your conda environment path
2. Update `#SBATCH --account` to your cluster account
3. Set `HUGGING_FACE_HUB_TOKEN` and `GOOGLE_API_KEY` environment variables

## License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
