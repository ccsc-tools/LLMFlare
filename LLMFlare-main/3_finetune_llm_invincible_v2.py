# 3_finetune_llm_invincible_v2.py
"""Fine-tune Llama 3 8B on solar flare explanation narratives using curriculum learning.

Uses two-stage curriculum learning with LoRA/PEFT:
  Stage 1: Train on explanation task only
  Stage 2: Train on all three tasks (explanation, recommendation, counterfactual)
"""

import argparse
import gc
import logging
import os

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from config import (
    LLM_TRAINING_DATASET_PATH,
    LLM_ADAPTER_PATH,
    BASE_MODEL_NAME,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    STAGE1_BATCH_SIZE,
    STAGE1_GRAD_ACCUMULATION,
    STAGE1_LR,
    STAGE1_EPOCHS,
    STAGE2_BATCH_SIZE,
    STAGE2_GRAD_ACCUMULATION,
    STAGE2_LR,
    STAGE2_EPOCHS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args=None):
    """Fine-tune Llama 3 8B with two-stage curriculum learning.

    Stage 1: Train on explanation examples only.
    Stage 2: Train on all skills (explanation, recommendation, counterfactual).

    Args:
        args: Argparse namespace (optional). If None, uses defaults from config.
    """
    if args is None:
        args = argparse.Namespace()

    logging.info(f"Loading V2 dataset from {LLM_TRAINING_DATASET_PATH}...")
    df = pd.read_csv(LLM_TRAINING_DATASET_PATH)
    explanation_df = df[df['skill'] == 'explanation'].copy()
    full_dataset = Dataset.from_pandas(df)
    explanation_dataset = Dataset.from_pandas(explanation_df)

    logging.info(f"Stage 1 Dataset Size (Explanation only): {len(explanation_dataset)}")
    logging.info(f"Stage 2 Dataset Size (All skills): {len(full_dataset)}")

    logging.info(f"Loading base model: {BASE_MODEL_NAME} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    logging.info("Model and tokenizer loaded.")

    logging.info("Configuring LoRA with optimal parameters...")
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logging.info("Tokenizing datasets...")
    tokenized_explanation_dataset = explanation_dataset.map(
        lambda samples: tokenizer(samples["text"], truncation=True, max_length=1024),
        batched=True
    )
    tokenized_full_dataset = full_dataset.map(
        lambda samples: tokenizer(samples["text"], truncation=True, max_length=1024),
        batched=True
    )

    data_collator = lambda data: {
        'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
        'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]),
        'labels': torch.stack([torch.tensor(f['input_ids']) for f in data])
    }

    logging.info("="*50)
    logging.info("STARTING CURRICULUM: STAGE 1 (Explanation Only)")
    logging.info("="*50)

    stage1_args = TrainingArguments(
        output_dir=os.path.join(LLM_ADAPTER_PATH, "stage1_chkpt"),
        per_device_train_batch_size=STAGE1_BATCH_SIZE,
        gradient_accumulation_steps=STAGE1_GRAD_ACCUMULATION,
        learning_rate=STAGE1_LR,
        num_train_epochs=STAGE1_EPOCHS,
        logging_steps=100,
        bf16=True,
        save_strategy="no",
    )

    trainer_stage1 = Trainer(
        model=model, args=stage1_args, train_dataset=tokenized_explanation_dataset,
        tokenizer=tokenizer, data_collator=data_collator
    )
    trainer_stage1.train()
    logging.info("STAGE 1 COMPLETE.")

    logging.info("="*50)
    logging.info("STARTING CURRICULUM: STAGE 2 (All Skills)")
    logging.info("="*50)

    stage2_args = TrainingArguments(
        output_dir=os.path.join(LLM_ADAPTER_PATH, "stage2_chkpt"),
        per_device_train_batch_size=STAGE2_BATCH_SIZE,
        gradient_accumulation_steps=STAGE2_GRAD_ACCUMULATION,
        learning_rate=STAGE2_LR,
        num_train_epochs=STAGE2_EPOCHS,
        logging_steps=100,
        bf16=True,
        save_strategy="epoch",
        save_total_limit=1,
    )

    trainer_stage2 = Trainer(
        model=model, args=stage2_args, train_dataset=tokenized_full_dataset,
        tokenizer=tokenizer, data_collator=data_collator
    )
    trainer_stage2.train()
    logging.info("STAGE 2 COMPLETE.")

    logging.info(f"Saving final 'invincible' V2 model to {LLM_ADAPTER_PATH}...")
    trainer_stage2.save_model(LLM_ADAPTER_PATH)
    logging.info("Model saved successfully. Project training is complete.")

if __name__ == "__main__":
    main()
