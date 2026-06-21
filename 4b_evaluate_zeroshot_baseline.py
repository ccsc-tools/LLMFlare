"""Evaluate zero-shot baseline (untuned) Llama 3 8B using Gemini as judge.

Provides comparison baseline for the fine-tuned model evaluation.
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    LLM_EVAL_PATH,
    BASE_MODEL_NAME,
    JUDGE_MODEL_NAME,
    NUM_EVAL_SAMPLES,
    MAX_RETRIES,
    EVAL_MAX_NEW_TOKENS,
    DELAY_BETWEEN_REQUESTS,
    get_google_api_key,
)

import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(args=None):
    """Evaluate zero-shot Llama 3 8B baseline using Gemini as judge.

    Args:
        args: Argparse namespace (optional). If None, uses defaults from config.
    """
    if args is None:
        args = argparse.Namespace()

    genai.configure(api_key=get_google_api_key())

    logging.info(f"Loading dataset and preparing {NUM_EVAL_SAMPLES} samples...")
    df = pd.read_csv(LLM_EVAL_PATH)
    eval_df = df

    logging.info(f"Loading BASE model only (zero-shot): {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base_model.eval()
    logging.info("Base model loaded successfully. No adapter applied.")

    judge_model = genai.GenerativeModel(JUDGE_MODEL_NAME)

    all_scores = []
    failed_samples = 0

    for index, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating Zero-Shot Baseline"):
        full_text = row['text']
        student_prompt_text = full_text.split("### Response:")[0] + "### Response:\n"
        reference_answer = full_text.split("### Response:")[1].strip()
        query = student_prompt_text.split("### Input:")[1].split("### Response:")[0].strip()

        inputs = tokenizer(student_prompt_text, return_tensors="pt").to(base_model.device)
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs, max_new_tokens=EVAL_MAX_NEW_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False, repetition_penalty=1.2
            )
        student_answer = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
        )

        judge_prompt_text = f"""You are an expert AI evaluator. Your task is to evaluate a student model's response based on a provided context and a reference answer.
Score the student's response on three criteria: Faithfulness, Insightfulness, and Fluency, each on a scale of 1 to 10.

**Context (The Data the Student Saw):**
{query}

**Reference Answer (The Ideal Ground Truth):**
{reference_answer}

**Student's Answer:**
{student_answer}

**Evaluation Criteria:**
1.  **Faithfulness (1-10):** How accurately does the student's answer reflect the provided context data?
2.  **Insightfulness (1-10):** How much valuable insight does the student's answer provide? 
3.  **Fluency (1-10):** Is the student's answer well-written, clear, and grammatically correct?

Your response MUST be a single JSON object and nothing else. Do not include any text before or after the JSON.
Example format: {{"faithfulness": 8, "insightfulness": 7, "fluency": 9}}
"""

        scores = None
        for attempt in range(MAX_RETRIES):
            try:
                response = judge_model.generate_content(judge_prompt_text)
                if not response.parts:
                    raise ValueError("API returned an empty/blocked response.")
                json_str = response.text[response.text.find('{'):response.text.rfind('}')+1]
                if not json_str:
                    raise ValueError("Extracted JSON string is empty.")
                scores = json.loads(json_str)
                break
            except Exception as e:
                error_msg = str(e)
                logging.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed. Error: {error_msg}")
                # SMART RETRY for Quota Limits
                if "429" in error_msg or "Quota" in error_msg or "limit: 0" in error_msg:
                    logging.info("Rate limit hit. Sleeping for 45 seconds to reset quota...")
                    time.sleep(45)
                else:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(2 ** attempt)
                    else:
                        logging.error(f"Failed to evaluate sample {index} after all retries.")
                        failed_samples += 1

        if scores:
            all_scores.append(scores)
        
        # Slowing down to respect the Free Tier RPM limit
        time.sleep(DELAY_BETWEEN_REQUESTS)

    if all_scores:
        avg_scores = {}
        for key in ["faithfulness", "insightfulness", "fluency"]:
            valid_scores =[s.get(key) for s in all_scores if isinstance(s.get(key), (int, float))]
            avg_scores[key] = np.mean(valid_scores) if valid_scores else 0.0

        print("\n" + "="*50)
        print("--- ZERO-SHOT BASELINE EVALUATION RESULTS (Table II) ---")
        print(json.dumps(avg_scores, indent=4))
        print("="*50)

if __name__ == "__main__":
    main()