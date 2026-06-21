# 4_evaluate_model_invincible_v2.py
"""Evaluate fine-tuned Llama 3 8B model using Gemini 2.5 Pro as LLM-as-a-Judge.

Scores student model responses on three criteria:
  - Faithfulness: accuracy relative to context data
  - Insightfulness: depth and meaningfulness of insights
  - Fluency: language quality and clarity
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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    LLM_EVAL_PATH,
    BASE_MODEL_NAME,
    LLM_ADAPTER_PATH,
    JUDGE_MODEL_NAME,
    NUM_EVAL_SAMPLES,
    MAX_RETRIES,
    EVAL_MAX_NEW_TOKENS,
    get_google_api_key,
)

import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(args=None):
    """Evaluate fine-tuned model using Gemini as judge on held-out examples.

    Args:
        args: Argparse namespace (optional). If None, uses defaults from config.
    """
    if args is None:
        args = argparse.Namespace()

    genai.configure(api_key=get_google_api_key())

    logging.info(f"Loading dataset and preparing {NUM_EVAL_SAMPLES} samples...")
    df = pd.read_csv(LLM_EVAL_PATH)
    eval_df = df

    logging.info(f"Loading student model: {BASE_MODEL_NAME} with adapter {LLM_ADAPTER_PATH}...")
    student_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    student_tokenizer.pad_token = student_tokenizer.eos_token
    student_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    student_model = PeftModel.from_pretrained(student_model, LLM_ADAPTER_PATH)
    student_model = student_model.merge_and_unload()
    student_model.eval()
    logging.info("Student model loaded successfully.")
    
    # --- Initialize the Judge Model ---
    judge_model = genai.GenerativeModel(JUDGE_MODEL_NAME)
    safety_config = {
        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
    }

    all_scores = []
    failed_samples = 0
    for index, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating with Gemini Judge"):
        full_text = row['text']
        student_prompt_text = full_text.split("### Response:")[0] + "### Response:\n"
        reference_answer = full_text.split("### Response:")[1].strip()
        query = student_prompt_text.split("### Input:")[1].split("### Response:")[0].strip()

        student_inputs = student_tokenizer(student_prompt_text, return_tensors="pt").to(student_model.device)
        with torch.no_grad():
            outputs = student_model.generate(
                **student_inputs, max_new_tokens=EVAL_MAX_NEW_TOKENS,
                eos_token_id=student_tokenizer.eos_token_id,
                pad_token_id=student_tokenizer.eos_token_id
            )
        student_answer = student_tokenizer.decode(
            outputs[0][student_inputs['input_ids'].shape[1]:], skip_special_tokens=True
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
1.  **Faithfulness (1-10):** How accurately does the student's answer reflect the provided context data? Does it hallucinate or misrepresent facts from the data? A high score means the student's answer is strictly grounded in the context.
2.  **Insightfulness (1-10):** How much valuable insight does the student's answer provide? Does it simply state the obvious, or does it connect the data to deeper, meaningful conclusions? A high score indicates a deep, non-obvious understanding.
3.  **Fluency (1-10):** Is the student's answer well-written, clear, and grammatically correct? A high score means the language is natural and easy to understand.

Your response MUST be a single JSON object and nothing else. Do not include any text before or after the JSON.
Example format: {{"faithfulness": 8, "insightfulness": 7, "fluency": 9}}
"""
        
        scores = None
        for attempt in range(MAX_RETRIES):
            try:
                response = judge_model.generate_content(judge_prompt_text, safety_settings=safety_config)
                
                if not response.parts:
                    raise ValueError("API returned an empty/blocked response.")

                json_str = response.text[response.text.find('{'):response.text.rfind('}')+1]
                if not json_str:
                    raise ValueError("Extracted JSON string is empty.")
                
                scores = json.loads(json_str)
                break # Success, exit the retry loop

            except Exception as e:
                logging.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} for sample {index} failed. Error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt) # Exponential backoff: 1s, 2s, 4s
                else:
                    logging.error(f"Failed to evaluate sample {index} after all retries.")
                    failed_samples += 1

        if scores:
            all_scores.append(scores)

    # --- Final Results Calculation ---
    if all_scores:
        avg_scores = {}
        for key in ["faithfulness", "insightfulness", "fluency"]:
            # Filter out any potential malformed score entries
            valid_scores = [s.get(key) for s in all_scores if isinstance(s.get(key), (int, float))]
            if valid_scores:
                avg_scores[key] = np.mean(valid_scores)
            else:
                avg_scores[key] = 0.0

        print("\n" + "="*50)
        print(f"--- FINAL EVALUATION RESULTS (Gemini 2.5 Pro) ---")
        print(f"Total Samples Attempted: {NUM_EVAL_SAMPLES}")
        print(f"Successfully Evaluated: {len(all_scores)}")
        print(f"Failed Samples: {failed_samples}")
        print("-" * 50)
        print("Average Scores (out of 10):")
        print(json.dumps(avg_scores, indent=4))
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
